// Rendering: particles -> HDR additive splat, two-pass half-res bloom, composite with maze + flow trail.

struct RP {
  originX: f32, originY: f32, scale: f32, vpW: f32,
  vpH: f32, W: f32, H: f32, radiusPx: f32,
  speedRef: f32, exposure: f32, showSolution: f32, time: f32,
  nC: u32, nU: u32, nV: u32, bloom: f32,
};

@group(0) @binding(0) var<uniform> R: RP;
@group(0) @binding(1) var<storage, read> particles: array<vec4f>;
@group(0) @binding(2) var<storage, read> grid: array<f32>;
@group(0) @binding(3) var<storage, read> cells: array<u32>;
@group(0) @binding(4) var samp: sampler;
@group(0) @binding(5) var hdrTex: texture_2d<f32>;
@group(0) @binding(6) var bloomTex: texture_2d<f32>;
@group(0) @binding(7) var blurSrc: texture_2d<f32>;
@group(0) @binding(8) var<uniform> blurU: vec4f; // dir.xy, target size zw

// ---------------- particles ----------------
struct VOut {
  @builtin(position) pos: vec4f,
  @location(0) uv: vec2f,
  @location(1) col: vec3f,
};

@vertex
fn vsParticle(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> VOut {
  var o: VOut;
  let p = particles[ii];
  if (p.x < 0.0) {
    o.pos = vec4f(2.0, 2.0, 0.0, 1.0);
    o.uv = vec2f(0.0);
    o.col = vec3f(0.0);
    return o;
  }
  var corners = array<vec2f, 6>(
    vec2f(-1.0, -1.0), vec2f(1.0, -1.0), vec2f(1.0, 1.0),
    vec2f(-1.0, -1.0), vec2f(1.0, 1.0), vec2f(-1.0, 1.0));
  let c = corners[vi];
  let px = vec2f(R.originX, R.originY) + p.xy * R.scale + c * R.radiusPx;
  let clip = vec2f(px.x / R.vpW * 2.0 - 1.0, 1.0 - px.y / R.vpH * 2.0);
  let sp = length(p.zw);
  let t = clamp(sp / R.speedRef, 0.0, 1.0);
  let deep = vec3f(0.02, 0.16, 0.80);
  let mid = vec3f(0.10, 0.62, 1.00);
  let hot = vec3f(0.85, 0.97, 1.00);
  let col = mix(mix(deep, mid, smoothstep(0.0, 0.45, t)), hot, smoothstep(0.45, 1.0, t));
  o.pos = vec4f(clip, 0.0, 1.0);
  o.uv = c;
  o.col = col;
  return o;
}

@fragment
fn fsParticle(in: VOut) -> @location(0) vec4f {
  let d = length(in.uv);
  let a = smoothstep(1.0, 0.15, d);
  return vec4f(in.col * a * R.exposure, a);
}

// ---------------- fullscreen ----------------
@vertex
fn vsFull(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4f {
  let x = f32(i32(vi & 1u) * 4 - 1);
  let y = f32(i32(vi >> 1u) * 4 - 1);
  return vec4f(x, y, 0.0, 1.0);
}

@fragment
fn fsBlur(@builtin(position) fp: vec4f) -> @location(0) vec4f {
  let srcDim = vec2f(textureDimensions(blurSrc));
  let uv = fp.xy / blurU.zw;
  let step = blurU.xy / srcDim;
  var w = array<f32, 5>(0.2270270270, 0.1945945946, 0.1216216216, 0.0540540541, 0.0162162162);
  var o = array<f32, 5>(0.0, 1.0, 2.0, 3.0, 4.0);
  var sum = textureSample(blurSrc, samp, uv).rgb * w[0];
  for (var i = 1; i < 5; i++) {
    sum += textureSample(blurSrc, samp, uv + step * o[i] * 1.5).rgb * w[i];
    sum += textureSample(blurSrc, samp, uv - step * o[i] * 1.5).rgb * w[i];
  }
  return vec4f(sum, 1.0);
}

fn sampleCell(q: vec2f, off: u32) -> f32 {
  let gw = u32(R.W);
  let gh = u32(R.H);
  let qc = clamp(q, vec2f(0.0), vec2f(R.W - 1.0001, R.H - 1.0001));
  let i0 = u32(floor(qc.x));
  let j0 = u32(floor(qc.y));
  let fx = qc.x - f32(i0);
  let fy = qc.y - f32(j0);
  let i1 = min(i0 + 1u, gw - 1u);
  let j1 = min(j0 + 1u, gh - 1u);
  return grid[off + j0 * gw + i0] * (1.0 - fx) * (1.0 - fy)
       + grid[off + j0 * gw + i1] * fx * (1.0 - fy)
       + grid[off + j1 * gw + i0] * (1.0 - fx) * fy
       + grid[off + j1 * gw + i1] * fx * fy;
}

@fragment
fn fsComposite(@builtin(position) fp: vec4f) -> @location(0) vec4f {
  let uv = fp.xy / vec2f(R.vpW, R.vpH);
  let water = textureSample(hdrTex, samp, uv).rgb;
  let bloom = textureSample(bloomTex, samp, uv).rgb;
  let g = (fp.xy - vec2f(R.originX, R.originY)) / R.scale;
  var col = vec3f(0.022, 0.024, 0.040);
  let offSdf = 2u * R.nU + 2u * R.nV;
  let offTrail = offSdf + R.nC;
  if (g.x >= 0.0 && g.y >= 0.0 && g.x < R.W && g.y < R.H) {
    let i = u32(g.x);
    let j = u32(g.y);
    let c = j * u32(R.W) + i;
    let flags = cells[R.nC + c];
    let sdf = sampleCell(g - vec2f(0.5), offSdf);
    if ((flags & 1u) == 1u) {
      col = vec3f(0.115, 0.125, 0.175);
      let edge = 1.0 - smoothstep(0.0, 1.4, -sdf);
      col += edge * vec3f(0.09, 0.10, 0.15);
    } else {
      col = vec3f(0.040, 0.044, 0.070);
      if (R.showSolution > 0.5 && (flags & 2u) == 2u) {
        col += vec3f(0.11, 0.045, 0.0);
      }
      let t = sampleCell(g - vec2f(0.5), offTrail);
      col += t * vec3f(1.0, 0.42, 0.10) * 0.45;
    }
  }
  var trailGlow = 0.0;
  if (g.x >= 0.0 && g.y >= 0.0 && g.x < R.W && g.y < R.H) {
    trailGlow = sampleCell(g - vec2f(0.5), offTrail);
  }
  let w = vec3f(1.0) - exp(-water);
  let cover = min(1.0, max(w.r, max(w.g, w.b)));
  col = col * (1.0 - cover * 0.85) + w + bloom * R.bloom;
  col += trailGlow * cover * vec3f(0.9, 0.5, 0.15) * 0.6;
  return vec4f(col, 1.0);
}
