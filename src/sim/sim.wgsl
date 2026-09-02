// FLIP/PIC fluid on a MAC grid. All buffers are packed:
//   particles : vec4f per particle (xy = position in grid cells, zw = velocity cells/s). x < 0 => dead
//   acc       : atomic<i32> fixed-point accumulators [U | V | wU | wV | count]
//   grid      : f32 [u | v | uPrev | vPrev | sdf | trail]
//   cells     : u32 [cellType(0 air,1 fluid,2 solid) | flags(bit0 solid, bit1 solution)]
//   stats     : [alive, drained]

struct Params {
  W: u32, H: u32, N: u32, nU: u32,
  nV: u32, nC: u32, spawnCursor: u32, spawnTotal: u32,
  inletCount: u32, srcCount: u32, frame: u32, _p0: u32,
  dt: f32, gravity: f32, flip: f32, radius: f32,
  restDensity: f32, driftK: f32, overrelax: f32, drainY: f32,
  inletX: f32, inletY: f32, inletR: f32, inletV: f32,
  srcX: f32, srcY: f32, srcR: f32, trailDecay: f32,
  trailScale: f32, maxSpeed: f32, _p1: f32, _p2: f32,
};

@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, read_write> particles: array<vec4f>;
@group(0) @binding(2) var<storage, read_write> acc: array<atomic<i32>>;
@group(0) @binding(3) var<storage, read_write> grid: array<f32>;
@group(0) @binding(4) var<storage, read_write> cells: array<u32>;
@group(0) @binding(5) var<storage, read_write> stats: array<atomic<u32>>;
@group(1) @binding(0) var<uniform> parity: u32;

const SCALE_V: f32 = 2048.0;
const SCALE_W: f32 = 16384.0;

fn offV() -> u32 { return P.nU; }
fn offWU() -> u32 { return P.nU + P.nV; }
fn offWV() -> u32 { return 2u * P.nU + P.nV; }
fn offCount() -> u32 { return 2u * P.nU + 2u * P.nV; }
fn offUp() -> u32 { return P.nU + P.nV; }
fn offVp() -> u32 { return 2u * P.nU + P.nV; }
fn offSdf() -> u32 { return 2u * P.nU + 2u * P.nV; }
fn offTrail() -> u32 { return 2u * P.nU + 2u * P.nV + P.nC; }

fn cellIdx(i: i32, j: i32) -> u32 { return u32(j) * P.W + u32(i); }
fn inBounds(i: i32, j: i32) -> bool { return i >= 0 && j >= 0 && i < i32(P.W) && j < i32(P.H); }
fn isSolid(i: i32, j: i32) -> bool {
  if (!inBounds(i, j)) { return true; }
  return (cells[P.nC + cellIdx(i, j)] & 1u) == 1u;
}
fn uIdx(i: i32, j: i32) -> u32 { return u32(j) * (P.W + 1u) + u32(i); }
fn vIdx(i: i32, j: i32) -> u32 { return u32(j) * P.W + u32(i); }

fn pcg(v: u32) -> u32 {
  let s = v * 747796405u + 2891336453u;
  let w = ((s >> ((s >> 28u) + 4u)) ^ s) * 277803737u;
  return (w >> 22u) ^ w;
}
fn rnd(seed: u32) -> f32 { return f32(pcg(seed)) / 4294967295.0; }

// ---- bilinear helpers over a gw x gh lattice whose node (i,j) sits at lattice coords (i,j)
struct Stencil { i0: i32, j0: i32, i1: i32, j1: i32, w00: f32, w10: f32, w01: f32, w11: f32 };
fn stencil(q: vec2f, gw: u32, gh: u32) -> Stencil {
  let mx = f32(gw - 1u) - 0.0001;
  let my = f32(gh - 1u) - 0.0001;
  let qc = clamp(q, vec2f(0.0), vec2f(mx, my));
  let i0 = i32(floor(qc.x));
  let j0 = i32(floor(qc.y));
  let fx = qc.x - f32(i0);
  let fy = qc.y - f32(j0);
  var s: Stencil;
  s.i0 = i0; s.j0 = j0;
  s.i1 = min(i0 + 1, i32(gw) - 1);
  s.j1 = min(j0 + 1, i32(gh) - 1);
  s.w00 = (1.0 - fx) * (1.0 - fy);
  s.w10 = fx * (1.0 - fy);
  s.w01 = (1.0 - fx) * fy;
  s.w11 = fx * fy;
  return s;
}
fn sampleG(q: vec2f, off: u32, gw: u32, gh: u32) -> f32 {
  let s = stencil(q, gw, gh);
  return grid[off + u32(s.j0) * gw + u32(s.i0)] * s.w00
       + grid[off + u32(s.j0) * gw + u32(s.i1)] * s.w10
       + grid[off + u32(s.j1) * gw + u32(s.i0)] * s.w01
       + grid[off + u32(s.j1) * gw + u32(s.i1)] * s.w11;
}
fn scatter(q: vec2f, val: f32, offVal: u32, offW: u32, gw: u32, gh: u32) {
  let s = stencil(q, gw, gh);
  let a = u32(s.j0) * gw + u32(s.i0);
  let b = u32(s.j0) * gw + u32(s.i1);
  let c = u32(s.j1) * gw + u32(s.i0);
  let d = u32(s.j1) * gw + u32(s.i1);
  atomicAdd(&acc[offVal + a], i32(val * s.w00 * SCALE_V));
  atomicAdd(&acc[offVal + b], i32(val * s.w10 * SCALE_V));
  atomicAdd(&acc[offVal + c], i32(val * s.w01 * SCALE_V));
  atomicAdd(&acc[offVal + d], i32(val * s.w11 * SCALE_V));
  atomicAdd(&acc[offW + a], i32(s.w00 * SCALE_W));
  atomicAdd(&acc[offW + b], i32(s.w10 * SCALE_W));
  atomicAdd(&acc[offW + c], i32(s.w01 * SCALE_W));
  atomicAdd(&acc[offW + d], i32(s.w11 * SCALE_W));
}
fn sampleSdf(p: vec2f) -> f32 { return sampleG(p - vec2f(0.5), offSdf(), P.W, P.H); }
fn sdfGrad(p: vec2f) -> vec2f {
  let e = 0.5;
  let gx = sampleSdf(p + vec2f(e, 0.0)) - sampleSdf(p - vec2f(e, 0.0));
  let gy = sampleSdf(p + vec2f(0.0, e)) - sampleSdf(p - vec2f(0.0, e));
  let g = vec2f(gx, gy);
  let l = length(g);
  if (l < 1e-5) { return vec2f(0.0, -1.0); }
  return g / l;
}

// ---- kernels -------------------------------------------------------------

@compute @workgroup_size(256)
fn resetParticles(@builtin(global_invocation_id) gid: vec3u) {
  let k = gid.x;
  if (k == 0u) { atomicStore(&stats[1], 0u); atomicStore(&stats[0], 0u); }
  if (k >= P.N) { return; }
  particles[k] = vec4f(-1.0, -1.0, 0.0, 0.0);
}

@compute @workgroup_size(256)
fn clearGrid(@builtin(global_invocation_id) gid: vec3u) {
  let k = gid.x;
  if (k == 0u) { atomicStore(&stats[0], 0u); }
  if (k >= offCount() + P.nC) { return; }
  atomicStore(&acc[k], 0);
}

@compute @workgroup_size(256)
fn spawn(@builtin(global_invocation_id) gid: vec3u) {
  let t = gid.x;
  if (t >= P.spawnTotal) { return; }
  let k = (P.spawnCursor + t) % P.N;
  if (particles[k].x >= 0.0) { return; }
  var c: vec2f;
  var r: f32;
  var v0: vec2f;
  if (t < P.inletCount) {
    c = vec2f(P.inletX, P.inletY); r = P.inletR; v0 = vec2f(0.0, P.inletV);
  } else {
    c = vec2f(P.srcX, P.srcY); r = P.srcR; v0 = vec2f(0.0, 0.0);
  }
  let seed = k * 3u + P.frame * 7919u;
  let ang = rnd(seed) * 6.2831853;
  let rad = r * sqrt(rnd(seed + 1u));
  let pos = c + vec2f(cos(ang), sin(ang)) * rad;
  let i = i32(floor(pos.x));
  let j = i32(floor(pos.y));
  if (isSolid(i, j)) { return; }
  let cnt = f32(atomicLoad(&acc[offCount() + cellIdx(i, j)]));
  if (cnt > P.restDensity * 0.8) { return; }
  // don't pour into a backed-up column: the cell a few rows below must also have room
  let jb = min(j + 4, i32(P.H) - 1);
  if (!isSolid(i, jb)) {
    let below = f32(atomicLoad(&acc[offCount() + cellIdx(i, jb)]));
    if (below > P.restDensity * 0.8) { return; }
  }
  let jit = (vec2f(rnd(seed + 2u), rnd(seed + 7u)) - 0.5) * 2.0;
  particles[k] = vec4f(pos, v0 + jit);
}

@compute @workgroup_size(256)
fn p2g(@builtin(global_invocation_id) gid: vec3u) {
  let k = gid.x;
  if (k >= P.N) { return; }
  let p = particles[k];
  if (p.x < 0.0) { return; }
  scatter(p.xy - vec2f(0.0, 0.5), p.z, 0u, offWU(), P.W + 1u, P.H);
  scatter(p.xy - vec2f(0.5, 0.0), p.w, offV(), offWV(), P.W, P.H + 1u);
  let i = clamp(i32(floor(p.x)), 0, i32(P.W) - 1);
  let j = clamp(i32(floor(p.y)), 0, i32(P.H) - 1);
  atomicAdd(&acc[offCount() + cellIdx(i, j)], 1);
}

@compute @workgroup_size(256)
fn gridBuild(@builtin(global_invocation_id) gid: vec3u) {
  let k = gid.x;
  if (k < P.nU) {
    let i = i32(k % (P.W + 1u));
    let j = i32(k / (P.W + 1u));
    var u = 0.0;
    let w = atomicLoad(&acc[offWU() + k]);
    if (w > 0) { u = (f32(atomicLoad(&acc[k])) / SCALE_V) / (f32(w) / SCALE_W); }
    if (isSolid(i - 1, j) || isSolid(i, j)) { u = 0.0; }
    grid[k] = u;
    grid[offUp() + k] = u;
  }
  if (k < P.nV) {
    let i = i32(k % P.W);
    let j = i32(k / P.W);
    var v = 0.0;
    let w = atomicLoad(&acc[offWV() + k]);
    if (w > 0) { v = (f32(atomicLoad(&acc[offV() + k])) / SCALE_V) / (f32(w) / SCALE_W); }
    if (isSolid(i, j - 1) || isSolid(i, j)) { v = 0.0; }
    grid[offV() + k] = v;
    grid[offVp() + k] = v;
  }
  if (k < P.nC) {
    let solid = (cells[P.nC + k] & 1u) == 1u;
    let cnt = atomicLoad(&acc[offCount() + k]);
    var t = 0u;
    if (solid) { t = 2u; } else if (cnt > 0) { t = 1u; }
    cells[k] = t;
  }
}

// Red-black Gauss-Seidel projection applied directly to face velocities
// (with particle-density drift compensation so pools keep their volume).
@compute @workgroup_size(256)
fn solve(@builtin(global_invocation_id) gid: vec3u) {
  let c = gid.x;
  if (c >= P.nC) { return; }
  let i = i32(c % P.W);
  let j = i32(c / P.W);
  if (u32((i + j) & 1) != parity) { return; }
  if (cells[c] != 1u) { return; }
  let sx0 = select(1.0, 0.0, isSolid(i - 1, j));
  let sx1 = select(1.0, 0.0, isSolid(i + 1, j));
  let sy0 = select(1.0, 0.0, isSolid(i, j - 1));
  let sy1 = select(1.0, 0.0, isSolid(i, j + 1));
  let s = sx0 + sx1 + sy0 + sy1;
  if (s == 0.0) { return; }
  let iu0 = uIdx(i, j);
  let iu1 = uIdx(i + 1, j);
  let iv0 = offV() + vIdx(i, j);
  let iv1 = offV() + vIdx(i, j + 1);
  var div = grid[iu1] - grid[iu0] + grid[iv1] - grid[iv0];
  let dens = f32(atomicLoad(&acc[offCount() + c]));
  let comp = dens - P.restDensity;
  if (comp > 0.0) { div -= P.driftK * comp; }
  let p = -div / s * P.overrelax;
  grid[iu0] -= sx0 * p;
  grid[iu1] += sx1 * p;
  grid[iv0] -= sy0 * p;
  grid[iv1] += sy1 * p;
}

var<workgroup> wgAlive: atomic<u32>;

@compute @workgroup_size(256)
fn g2p(@builtin(global_invocation_id) gid: vec3u, @builtin(local_invocation_id) lid: vec3u) {
  if (lid.x == 0u) { atomicStore(&wgAlive, 0u); }
  workgroupBarrier();
  let k = gid.x;
  var alive = 0u;
  if (k < P.N) {
    var pr = particles[k];
    if (pr.x >= 0.0) {
      let pos = pr.xy;
      let qu = pos - vec2f(0.0, 0.5);
      let qv = pos - vec2f(0.5, 0.0);
      let un = sampleG(qu, 0u, P.W + 1u, P.H);
      let uo = sampleG(qu, offUp(), P.W + 1u, P.H);
      let vn = sampleG(qv, offV(), P.W, P.H + 1u);
      let vo = sampleG(qv, offVp(), P.W, P.H + 1u);
      let vNew = vec2f(un, vn);
      let vOld = vec2f(uo, vo);
      var vel = P.flip * (pr.zw + vNew - vOld) + (1.0 - P.flip) * vNew;
      vel.y += P.gravity * P.dt;
      let sp = length(vel);
      if (sp > P.maxSpeed) { vel *= P.maxSpeed / sp; }
      var np = pos + vel * P.dt;
      if (np.y > P.drainY) {
        pr = vec4f(-1.0, -1.0, 0.0, 0.0);
        atomicAdd(&stats[1], 1u);
      } else {
        for (var it = 0; it < 2; it++) {
          let d = sampleSdf(np);
          if (d < P.radius) {
            let n = sdfGrad(np);
            np += n * (P.radius - d);
            let vnrm = dot(vel, n);
            if (vnrm < 0.0) { vel -= vnrm * n; }
          }
        }
        let lo = vec2f(0.5 + P.radius);
        let hi = vec2f(f32(P.W) - 0.5 - P.radius, f32(P.H) - 0.5 - P.radius);
        np = clamp(np, lo, hi);
        pr = vec4f(np, vel);
        alive = 1u;
      }
      particles[k] = pr;
    }
  }
  atomicAdd(&wgAlive, alive);
  workgroupBarrier();
  if (lid.x == 0u) { atomicAdd(&stats[0], atomicLoad(&wgAlive)); }
}

@compute @workgroup_size(256)
fn trail(@builtin(global_invocation_id) gid: vec3u) {
  let c = gid.x;
  if (c >= P.nC) { return; }
  let i = i32(c % P.W);
  let j = i32(c / P.W);
  let t = grid[offTrail() + c];
  var s = 0.0;
  if (cells[c] == 1u) {
    let u = 0.5 * (grid[uIdx(i, j)] + grid[uIdx(i + 1, j)]);
    let v = 0.5 * (grid[offV() + vIdx(i, j)] + grid[offV() + vIdx(i, j + 1)]);
    s = min(length(vec2f(u, v)) * P.trailScale, 1.0);
  }
  grid[offTrail() + c] = max(t * P.trailDecay, s);
}
