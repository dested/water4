/**
 * Maze generation + rasterization into the fluid grid.
 *
 * Layout (sim cells): a `wall`-thick border, then `cols`x`rows` maze cells,
 * each `res` cells wide with a `wall`-thick partition between neighbours.
 * A `gutter`-tall spout region sits above the maze (water is poured there and
 * falls into the entrance at top-left) and a `gutter`-tall drain region sits
 * below it (particles that reach it are removed and counted).
 */

export interface MazeConfig {
  cols: number;
  rows: number;
  /** interior width of one maze cell, in sim cells */
  res: number;
  /** wall thickness, in sim cells */
  wall: number;
  seed: number;
}

export interface MazeLayout {
  cfg: MazeConfig;
  /** total sim grid size */
  W: number;
  H: number;
  gutter: number;
  /** W*H, bit0 = solid, bit1 = on the true solution path */
  cells: Uint32Array<ArrayBuffer>;
  /** W*H signed distance (cells) to the nearest wall face; negative inside walls */
  sdf: Float32Array<ArrayBuffer>;
  inlet: { x: number; y: number; r: number };
  /** particles whose y exceeds this are drained */
  drainY: number;
  openCells: number;
}

const N_BIT = 1;
const E_BIT = 2;
const S_BIT = 4;
const W_BIT = 8;

function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** Iterative recursive-backtracker: perfect maze, exactly one path between any two cells. */
function carveMaze(cols: number, rows: number, rnd: () => number): Uint8Array {
  const walls = new Uint8Array(cols * rows).fill(N_BIT | E_BIT | S_BIT | W_BIT);
  const visited = new Uint8Array(cols * rows);
  const stack: number[] = [0];
  visited[0] = 1;
  const dirs: Array<{ dx: number; dy: number; bit: number; opp: number }> = [
    { dx: 0, dy: -1, bit: N_BIT, opp: S_BIT },
    { dx: 1, dy: 0, bit: E_BIT, opp: W_BIT },
    { dx: 0, dy: 1, bit: S_BIT, opp: N_BIT },
    { dx: -1, dy: 0, bit: W_BIT, opp: E_BIT },
  ];
  while (stack.length > 0) {
    const cur = stack[stack.length - 1];
    if (cur === undefined) break;
    const cx = cur % cols;
    const cy = (cur - cx) / cols;
    const options: Array<{ n: number; bit: number; opp: number }> = [];
    for (const d of dirs) {
      const nx = cx + d.dx;
      const ny = cy + d.dy;
      if (nx < 0 || ny < 0 || nx >= cols || ny >= rows) continue;
      const n = ny * cols + nx;
      if (visited[n]) continue;
      options.push({ n, bit: d.bit, opp: d.opp });
    }
    if (options.length === 0) {
      stack.pop();
      continue;
    }
    const pick = options[Math.floor(rnd() * options.length)];
    if (!pick) break;
    walls[cur] = (walls[cur] ?? 0) & ~pick.bit;
    walls[pick.n] = (walls[pick.n] ?? 0) & ~pick.opp;
    visited[pick.n] = 1;
    stack.push(pick.n);
  }
  return walls;
}

/** BFS from (0,0) to (cols-1,rows-1); returns the cell indices along the path. */
function solve(walls: Uint8Array, cols: number, rows: number): number[] {
  const target = rows * cols - 1;
  const prev = new Int32Array(cols * rows).fill(-1);
  const queue: number[] = [0];
  prev[0] = 0;
  let head = 0;
  while (head < queue.length) {
    const cur = queue[head++];
    if (cur === undefined) break;
    if (cur === target) break;
    const cx = cur % cols;
    const cy = (cur - cx) / cols;
    const w = walls[cur] ?? 0;
    const tryPush = (open: boolean, n: number): void => {
      if (open && prev[n] === -1) {
        prev[n] = cur;
        queue.push(n);
      }
    };
    if (cy > 0) tryPush((w & N_BIT) === 0, cur - cols);
    if (cx < cols - 1) tryPush((w & E_BIT) === 0, cur + 1);
    if (cy < rows - 1) tryPush((w & S_BIT) === 0, cur + cols);
    if (cx > 0) tryPush((w & W_BIT) === 0, cur - 1);
  }
  const path: number[] = [];
  let c = target;
  while (c !== 0) {
    path.push(c);
    const p = prev[c];
    if (p === undefined || p < 0) break;
    c = p;
  }
  path.push(0);
  path.reverse();
  return path;
}

/** 1-D squared Euclidean distance transform (Felzenszwalb and Huttenlocher). */
function edt1d(f: Float32Array, n: number, out: Float32Array, v: Int32Array, z: Float32Array): void {
  let k = 0;
  v[0] = 0;
  z[0] = -Infinity;
  z[1] = Infinity;
  for (let q = 1; q < n; q++) {
    const fq = f[q] ?? Infinity;
    let s: number;
    for (;;) {
      const vk = v[k] ?? 0;
      const fvk = f[vk] ?? Infinity;
      s = (fq + q * q - (fvk + vk * vk)) / (2 * q - 2 * vk);
      if (s > (z[k] ?? -Infinity)) break;
      k--;
    }
    k++;
    v[k] = q;
    z[k] = s;
    z[k + 1] = Infinity;
  }
  k = 0;
  for (let q = 0; q < n; q++) {
    while ((z[k + 1] ?? Infinity) < q) k++;
    const vk = v[k] ?? 0;
    out[q] = (q - vk) * (q - vk) + (f[vk] ?? Infinity);
  }
}

/** Squared distance from every cell to the nearest cell where `isSource` is true. */
function edt2d(W: number, H: number, isSource: (i: number) => boolean): Float32Array {
  const INF = 1e12;
  const g = new Float32Array(W * H);
  const n = Math.max(W, H);
  const f = new Float32Array(n);
  const out = new Float32Array(n);
  const v = new Int32Array(n);
  const z = new Float32Array(n + 1);
  for (let x = 0; x < W; x++) {
    for (let y = 0; y < H; y++) f[y] = isSource(y * W + x) ? 0 : INF;
    edt1d(f, H, out, v, z);
    for (let y = 0; y < H; y++) g[y * W + x] = out[y] ?? INF;
  }
  const d = new Float32Array(W * H);
  for (let y = 0; y < H; y++) {
    for (let x = 0; x < W; x++) f[x] = g[y * W + x] ?? INF;
    edt1d(f, W, out, v, z);
    for (let x = 0; x < W; x++) d[y * W + x] = out[x] ?? INF;
  }
  return d;
}

export function generateMaze(cfg: MazeConfig): MazeLayout {
  const { cols, rows, res, wall } = cfg;
  const rnd = mulberry32(cfg.seed);
  const walls = carveMaze(cols, rows, rnd);
  const path = solve(walls, cols, rows);

  const pitch = res + wall;
  const gutter = res;
  const W = cols * pitch + wall;
  const mazeH = rows * pitch + wall;
  const H = gutter + mazeH + gutter;
  const cells = new Uint32Array(W * H).fill(1);

  const paint = (x0: number, y0: number, x1: number, y1: number, bit: number, set: boolean): void => {
    for (let y = Math.max(0, y0); y < Math.min(H, y1); y++) {
      for (let x = Math.max(0, x0); x < Math.min(W, x1); x++) {
        const i = y * W + x;
        const cur = cells[i] ?? 0;
        cells[i] = set ? cur | bit : cur & ~bit;
      }
    }
  };
  const interiorX = (cx: number): [number, number] => [wall + cx * pitch, wall + cx * pitch + res];
  const interiorY = (cy: number): [number, number] => [
    gutter + wall + cy * pitch,
    gutter + wall + cy * pitch + res,
  ];

  for (let cy = 0; cy < rows; cy++) {
    for (let cx = 0; cx < cols; cx++) {
      const w = walls[cy * cols + cx] ?? 0;
      const [x0, x1] = interiorX(cx);
      const [y0, y1] = interiorY(cy);
      paint(x0, y0, x1, y1, 1, false);
      if ((w & E_BIT) === 0) paint(x1, y0, x1 + wall, y1, 1, false);
      if ((w & S_BIT) === 0) paint(x0, y1, x1, y1 + wall, 1, false);
    }
  }
  // entrance spout above (0,0) and drain below (cols-1, rows-1)
  {
    const [x0, x1] = interiorX(0);
    paint(x0, 0, x1, gutter + wall, 1, false);
  }
  {
    const [x0, x1] = interiorX(cols - 1);
    const [, y1] = interiorY(rows - 1);
    paint(x0, y1, x1, H, 1, false);
  }
  // solution overlay
  for (let k = 0; k < path.length; k++) {
    const c = path[k];
    if (c === undefined) continue;
    const cx = c % cols;
    const cy = (c - cx) / cols;
    const [x0, x1] = interiorX(cx);
    const [y0, y1] = interiorY(cy);
    paint(x0, y0, x1, y1, 2, true);
    const next = path[k + 1];
    if (next === undefined) continue;
    const nx = next % cols;
    const ny = (next - nx) / cols;
    if (nx === cx + 1) paint(x1, y0, x1 + wall, y1, 2, true);
    if (nx === cx - 1) paint(x0 - wall, y0, x0, y1, 2, true);
    if (ny === cy + 1) paint(x0, y1, x1, y1 + wall, 2, true);
    if (ny === cy - 1) paint(x0, y0 - wall, x1, y0, 2, true);
  }

  const isSolid = (i: number): boolean => ((cells[i] ?? 1) & 1) === 1;
  const dSolid = edt2d(W, H, isSolid);
  const dOpen = edt2d(W, H, (i) => !isSolid(i));
  const sdf = new Float32Array(W * H);
  let openCells = 0;
  for (let i = 0; i < W * H; i++) {
    if (isSolid(i)) {
      sdf[i] = -(Math.sqrt(dOpen[i] ?? 0) - 0.5);
    } else {
      sdf[i] = Math.sqrt(dSolid[i] ?? 0) - 0.5;
      openCells++;
    }
  }

  return {
    cfg,
    W,
    H,
    gutter,
    cells,
    sdf,
    inlet: { x: wall + res / 2, y: gutter * 0.45, r: Math.max(1, res * 0.32) },
    drainY: gutter + mazeH + 1.0,
    openCells,
  };
}
