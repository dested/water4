import simWgsl from './sim.wgsl?raw';
import renderWgsl from './render.wgsl?raw';
import type { MazeLayout } from '../maze/generate';

export interface SimSettings {
  /** total particle buffer size */
  particleBudget: number;
  /** particles per second poured at the inlet while `pouring` */
  pourRate: number;
  pouring: boolean;
  /** cells / s^2 */
  gravity: number;
  /** 0 = pure PIC (viscous), 1 = pure FLIP (splashy) */
  flip: number;
  substeps: number;
  /** red-black Gauss-Seidel sweeps per substep */
  iterations: number;
  /** particles per cell at rest */
  restDensity: number;
  driftK: number;
  overrelax: number;
  /** collision radius in cells */
  particleRadius: number;
  /** point sprite radius in device pixels */
  pointRadiusPx: number;
  exposure: number;
  /** speed (cells/s) that maps to white */
  speedRef: number;
  bloom: number;
  showSolution: boolean;
  paused: boolean;
  /** simulated 60 Hz frames per rendered frame */
  timeScale: number;
  trailDecay: number;
  trailScale: number;
}

export const DEFAULT_SETTINGS: SimSettings = {
  particleBudget: 2_000_000,
  pourRate: 250_000,
  pouring: true,
  gravity: 400,
  flip: 0.8,
  substeps: 3,
  iterations: 50,
  restDensity: 24,
  driftK: 1.0,
  overrelax: 1.9,
  particleRadius: 0.25,
  pointRadiusPx: 2.0,
  exposure: 0.035,
  speedRef: 120,
  bloom: 0.35,
  showSolution: false,
  paused: false,
  timeScale: 3,
  trailDecay: 0.98,
  trailScale: 0.012,
};

export interface SimStats {
  fps: number;
  alive: number;
  drained: number;
  /** compute pass GPU time in ms (NaN when timestamp queries are unavailable) */
  gpuMs: number;
  /** JS time spent encoding a frame */
  cpuMs: number;
  frame: number;
  gridW: number;
  gridH: number;
}

type KernelName = 'resetParticles' | 'clearGrid' | 'spawn' | 'p2g' | 'gridBuild' | 'solve' | 'g2p' | 'trail';
const KERNELS: readonly KernelName[] = ['resetParticles', 'clearGrid', 'spawn', 'p2g', 'gridBuild', 'solve', 'g2p', 'trail'];

const WG = 256;
const groups = (n: number): number => Math.max(1, Math.ceil(n / WG));

interface GridBuffers {
  particles: GPUBuffer;
  acc: GPUBuffer;
  grid: GPUBuffer;
  cells: GPUBuffer;
  nU: number;
  nV: number;
  nC: number;
  N: number;
  bg0: GPUBindGroup;
}

interface Targets {
  hdr: GPUTexture;
  bloomA: GPUTexture;
  bloomB: GPUTexture;
  particleBG: GPUBindGroup;
  blurABG: GPUBindGroup;
  blurBBG: GPUBindGroup;
  compositeBG: GPUBindGroup;
}

export class FluidSim {
  readonly device: GPUDevice;
  settings: SimSettings;
  readonly stats: SimStats = { fps: 0, alive: 0, drained: 0, gpuMs: NaN, cpuMs: 0, frame: 0, gridW: 0, gridH: 0 };
  readonly maxParticles: number;

  private readonly canvas: HTMLCanvasElement;
  private readonly ctx: GPUCanvasContext;

  private readonly paramsBuf: GPUBuffer;
  private readonly paramsData = new ArrayBuffer(128);
  private readonly pf = new Float32Array(this.paramsData);
  private readonly pu = new Uint32Array(this.paramsData);
  private readonly parityBufs: [GPUBuffer, GPUBuffer];
  private readonly parityBGs: [GPUBindGroup, GPUBindGroup];
  private readonly statsBuf: GPUBuffer;
  private readonly readback: GPUBuffer;
  private readonly querySet: GPUQuerySet | null;
  private readonly tsResolve: GPUBuffer | null;
  private readonly bgl0: GPUBindGroupLayout;
  private readonly pipes: Record<KernelName, GPUComputePipeline>;

  private readonly renderParamsBuf: GPUBuffer;
  private readonly rpData = new ArrayBuffer(64);
  private readonly rpf = new Float32Array(this.rpData);
  private readonly rpu = new Uint32Array(this.rpData);
  private readonly blurBufs: [GPUBuffer, GPUBuffer];
  private readonly sampler: GPUSampler;
  private readonly particlePipe: GPURenderPipeline;
  private readonly blurPipe: GPURenderPipeline;
  private readonly compositePipe: GPURenderPipeline;

  private layout: MazeLayout | null = null;
  private gb: GridBuffers | null = null;
  private targets: Targets | null = null;
  private vpW = 0;
  private vpH = 0;
  private originX = 0;
  private originY = 0;
  private scale = 1;

  private frameIdx = 0;
  private spawnCursor = 0;
  private spawnAccum = 0;
  private pointer: { x: number; y: number } | null = null;
  private mapPending = false;
  private raf = 0;
  private lastT = 0;
  private fpsSmooth = 0;
  private destroyed = false;
  private pendingReset = false;

  private constructor(
    device: GPUDevice,
    canvas: HTMLCanvasElement,
    ctx: GPUCanvasContext,
    format: GPUTextureFormat,
    hasTimestamps: boolean,
    settings: SimSettings,
  ) {
    this.device = device;
    this.canvas = canvas;
    this.ctx = ctx;
    this.settings = { ...settings };
    this.maxParticles = Math.floor(Math.min(device.limits.maxStorageBufferBindingSize, device.limits.maxBufferSize) / 16);

    const simModule = device.createShaderModule({ code: simWgsl });
    const renderModule = device.createShaderModule({ code: renderWgsl });

    this.bgl0 = device.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
        { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
        { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
        { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
        { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
      ],
    });
    const bgl1 = device.createBindGroupLayout({
      entries: [{ binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } }],
    });
    const computeLayout = device.createPipelineLayout({ bindGroupLayouts: [this.bgl0, bgl1] });
    const pipes: Partial<Record<KernelName, GPUComputePipeline>> = {};
    for (const k of KERNELS) {
      pipes[k] = device.createComputePipeline({ layout: computeLayout, compute: { module: simModule, entryPoint: k } });
    }
    this.pipes = pipes as Record<KernelName, GPUComputePipeline>;

    this.paramsBuf = device.createBuffer({ size: 128, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    const mkParity = (p: number): GPUBuffer => {
      const b = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
      device.queue.writeBuffer(b, 0, new Uint32Array([p, 0, 0, 0]));
      return b;
    };
    this.parityBufs = [mkParity(0), mkParity(1)];
    this.parityBGs = [
      device.createBindGroup({ layout: bgl1, entries: [{ binding: 0, resource: { buffer: this.parityBufs[0] } }] }),
      device.createBindGroup({ layout: bgl1, entries: [{ binding: 0, resource: { buffer: this.parityBufs[1] } }] }),
    ];
    this.statsBuf = device.createBuffer({ size: 16, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
    this.readback = device.createBuffer({ size: 32, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    if (hasTimestamps) {
      this.querySet = device.createQuerySet({ type: 'timestamp', count: 2 });
      this.tsResolve = device.createBuffer({ size: 16, usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC });
    } else {
      this.querySet = null;
      this.tsResolve = null;
    }

    this.renderParamsBuf = device.createBuffer({ size: 64, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.blurBufs = [
      device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST }),
      device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST }),
    ];
    this.sampler = device.createSampler({ magFilter: 'linear', minFilter: 'linear', addressModeU: 'clamp-to-edge', addressModeV: 'clamp-to-edge' });

    this.particlePipe = device.createRenderPipeline({
      layout: 'auto',
      vertex: { module: renderModule, entryPoint: 'vsParticle' },
      fragment: {
        module: renderModule,
        entryPoint: 'fsParticle',
        targets: [
          {
            format: 'rgba16float',
            blend: {
              color: { srcFactor: 'one', dstFactor: 'one', operation: 'add' },
              alpha: { srcFactor: 'one', dstFactor: 'one', operation: 'add' },
            },
          },
        ],
      },
      primitive: { topology: 'triangle-list' },
    });
    this.blurPipe = device.createRenderPipeline({
      layout: 'auto',
      vertex: { module: renderModule, entryPoint: 'vsFull' },
      fragment: { module: renderModule, entryPoint: 'fsBlur', targets: [{ format: 'rgba16float' }] },
      primitive: { topology: 'triangle-list' },
    });
    this.compositePipe = device.createRenderPipeline({
      layout: 'auto',
      vertex: { module: renderModule, entryPoint: 'vsFull' },
      fragment: { module: renderModule, entryPoint: 'fsComposite', targets: [{ format }] },
      primitive: { topology: 'triangle-list' },
    });
  }

  static async create(canvas: HTMLCanvasElement, settings: SimSettings = DEFAULT_SETTINGS): Promise<FluidSim> {
    if (!('gpu' in navigator) || navigator.gpu === undefined) {
      throw new Error('WebGPU is not available in this browser. Use Chrome or Edge 113+.');
    }
    const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
    if (!adapter) throw new Error('No WebGPU adapter found.');
    const hasTimestamps = adapter.features.has('timestamp-query');
    const requiredFeatures: GPUFeatureName[] = hasTimestamps ? ['timestamp-query'] : [];
    const wantBinding = Math.min(adapter.limits.maxStorageBufferBindingSize, 8_000_000 * 16);
    const wantBuffer = Math.min(adapter.limits.maxBufferSize, 8_000_000 * 16);
    const device = await adapter.requestDevice({
      requiredFeatures,
      requiredLimits: {
        maxStorageBufferBindingSize: wantBinding,
        maxBufferSize: wantBuffer,
      },
    });
    const ctx = canvas.getContext('webgpu');
    if (!ctx) throw new Error('Could not create a WebGPU canvas context.');
    const format = navigator.gpu.getPreferredCanvasFormat();
    ctx.configure({ device, format, alphaMode: 'opaque' });
    return new FluidSim(device, canvas, ctx, format, hasTimestamps, settings);
  }

  /** Install a new maze. Rebuilds every grid-sized buffer and clears the water. */
  setLayout(layout: MazeLayout): void {
    this.layout = layout;
    this.stats.gridW = layout.W;
    this.stats.gridH = layout.H;
    this.rebuildGridBuffers();
    this.fitViewport();
  }

  /** Apply new settings; grows/shrinks the particle buffer if the budget changed. */
  updateSettings(next: SimSettings): void {
    const budgetChanged = next.particleBudget !== this.settings.particleBudget;
    this.settings = { ...next, particleBudget: Math.min(next.particleBudget, this.maxParticles) };
    if (budgetChanged && this.layout) this.rebuildGridBuffers();
  }

  /** Pointer in canvas CSS pixels, or null when released. */
  setPointer(css: { x: number; y: number } | null): void {
    if (!css) {
      this.pointer = null;
      return;
    }
    const dpr = this.canvas.width / Math.max(1, this.canvas.clientWidth);
    const px = css.x * dpr;
    const py = css.y * dpr;
    this.pointer = { x: (px - this.originX) / this.scale, y: (py - this.originY) / this.scale };
  }

  resetWater(): void {
    this.pendingReset = true;
  }

  start(): void {
    if (this.raf) return;
    this.lastT = performance.now();
    const loop = (t: number): void => {
      if (this.destroyed) return;
      this.frame(t);
      this.raf = requestAnimationFrame(loop);
    };
    this.raf = requestAnimationFrame(loop);
  }

  destroy(): void {
    this.destroyed = true;
    if (this.raf) cancelAnimationFrame(this.raf);
    this.raf = 0;
    this.destroyTargets();
    this.destroyGridBuffers();
    this.device.destroy();
  }

  // ------------------------------------------------------------------ buffers

  private destroyGridBuffers(): void {
    if (!this.gb) return;
    this.gb.particles.destroy();
    this.gb.acc.destroy();
    this.gb.grid.destroy();
    this.gb.cells.destroy();
    this.gb = null;
  }

  private rebuildGridBuffers(): void {
    const layout = this.layout;
    if (!layout) return;
    this.destroyGridBuffers();
    const { W, H } = layout;
    const nU = (W + 1) * H;
    const nV = W * (H + 1);
    const nC = W * H;
    const N = Math.min(this.settings.particleBudget, this.maxParticles);
    const d = this.device;

    const particles = d.createBuffer({ size: N * 16, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    const acc = d.createBuffer({ size: (2 * nU + 2 * nV + nC) * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    const grid = d.createBuffer({ size: (2 * nU + 2 * nV + 2 * nC) * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    const cells = d.createBuffer({ size: 2 * nC * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    d.queue.writeBuffer(grid, (2 * nU + 2 * nV) * 4, layout.sdf);
    d.queue.writeBuffer(cells, nC * 4, layout.cells);
    d.queue.writeBuffer(this.statsBuf, 0, new Uint32Array([0, 0, 0, 0]));

    const bg0 = d.createBindGroup({
      layout: this.bgl0,
      entries: [
        { binding: 0, resource: { buffer: this.paramsBuf } },
        { binding: 1, resource: { buffer: particles } },
        { binding: 2, resource: { buffer: acc } },
        { binding: 3, resource: { buffer: grid } },
        { binding: 4, resource: { buffer: cells } },
        { binding: 5, resource: { buffer: this.statsBuf } },
      ],
    });
    this.gb = { particles, acc, grid, cells, nU, nV, nC, N, bg0 };
    this.spawnCursor = 0;
    this.pendingReset = true;
    // render bind groups reference the particle/grid buffers
    if (this.targets) this.rebuildTargets();
  }

  private destroyTargets(): void {
    if (!this.targets) return;
    this.targets.hdr.destroy();
    this.targets.bloomA.destroy();
    this.targets.bloomB.destroy();
    this.targets = null;
  }

  private rebuildTargets(): void {
    this.destroyTargets();
    const gb = this.gb;
    if (!gb || this.vpW === 0 || this.vpH === 0) return;
    const d = this.device;
    const usage = GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING;
    const hdr = d.createTexture({ size: [this.vpW, this.vpH], format: 'rgba16float', usage });
    const hw = Math.max(1, Math.floor(this.vpW / 2));
    const hh = Math.max(1, Math.floor(this.vpH / 2));
    const bloomA = d.createTexture({ size: [hw, hh], format: 'rgba16float', usage });
    const bloomB = d.createTexture({ size: [hw, hh], format: 'rgba16float', usage });
    d.queue.writeBuffer(this.blurBufs[0], 0, new Float32Array([1, 0, hw, hh]));
    d.queue.writeBuffer(this.blurBufs[1], 0, new Float32Array([0, 1, hw, hh]));

    const particleBG = d.createBindGroup({
      layout: this.particlePipe.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.renderParamsBuf } },
        { binding: 1, resource: { buffer: gb.particles } },
      ],
    });
    const blurABG = d.createBindGroup({
      layout: this.blurPipe.getBindGroupLayout(0),
      entries: [
        { binding: 4, resource: this.sampler },
        { binding: 7, resource: hdr.createView() },
        { binding: 8, resource: { buffer: this.blurBufs[0] } },
      ],
    });
    const blurBBG = d.createBindGroup({
      layout: this.blurPipe.getBindGroupLayout(0),
      entries: [
        { binding: 4, resource: this.sampler },
        { binding: 7, resource: bloomA.createView() },
        { binding: 8, resource: { buffer: this.blurBufs[1] } },
      ],
    });
    const compositeBG = d.createBindGroup({
      layout: this.compositePipe.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.renderParamsBuf } },
        { binding: 2, resource: { buffer: gb.grid } },
        { binding: 3, resource: { buffer: gb.cells } },
        { binding: 4, resource: this.sampler },
        { binding: 5, resource: hdr.createView() },
        { binding: 6, resource: bloomB.createView() },
      ],
    });
    this.targets = { hdr, bloomA, bloomB, particleBG, blurABG, blurBBG, compositeBG };
  }

  private fitViewport(): void {
    const dpr = Math.min(2, window.devicePixelRatio || 1);
    const w = Math.max(1, Math.floor(this.canvas.clientWidth * dpr));
    const h = Math.max(1, Math.floor(this.canvas.clientHeight * dpr));
    const sizeChanged = w !== this.vpW || h !== this.vpH;
    if (sizeChanged) {
      this.canvas.width = w;
      this.canvas.height = h;
      this.vpW = w;
      this.vpH = h;
    }
    const layout = this.layout;
    if (layout) {
      const pad = 24 * dpr;
      this.scale = Math.min((w - pad * 2) / layout.W, (h - pad * 2) / layout.H);
      this.originX = (w - layout.W * this.scale) / 2;
      this.originY = (h - layout.H * this.scale) / 2;
    }
    if (sizeChanged || !this.targets) this.rebuildTargets();
  }

  // ------------------------------------------------------------------ frame

  private writeParams(spawnTotal: number, inletCount: number, srcCount: number, dt: number): void {
    const layout = this.layout;
    const gb = this.gb;
    if (!layout || !gb) return;
    const s = this.settings;
    const pu = this.pu;
    const pf = this.pf;
    pu[0] = layout.W;
    pu[1] = layout.H;
    pu[2] = gb.N;
    pu[3] = gb.nU;
    pu[4] = gb.nV;
    pu[5] = gb.nC;
    pu[6] = this.spawnCursor;
    pu[7] = spawnTotal;
    pu[8] = inletCount;
    pu[9] = srcCount;
    pu[10] = this.frameIdx;
    pu[11] = 0;
    pf[12] = dt;
    pf[13] = s.gravity;
    pf[14] = s.flip;
    pf[15] = s.particleRadius;
    pf[16] = s.restDensity;
    pf[17] = s.driftK;
    pf[18] = s.overrelax;
    pf[19] = layout.drainY;
    pf[20] = layout.inlet.x;
    pf[21] = layout.inlet.y;
    pf[22] = layout.inlet.r;
    pf[23] = 20;
    const ptr = this.pointer;
    pf[24] = ptr ? ptr.x : -100;
    pf[25] = ptr ? ptr.y : -100;
    pf[26] = Math.max(1.5, layout.cfg.res * 0.4);
    pf[27] = s.trailDecay;
    pf[28] = s.trailScale;
    pf[29] = 0.9 / dt; // CFL: at most ~one cell per substep
    pf[30] = 0;
    pf[31] = 0;
    this.device.queue.writeBuffer(this.paramsBuf, 0, this.paramsData);
  }

  private writeRenderParams(t: number): void {
    const layout = this.layout;
    const gb = this.gb;
    if (!layout || !gb) return;
    const s = this.settings;
    const f = this.rpf;
    f[0] = this.originX;
    f[1] = this.originY;
    f[2] = this.scale;
    f[3] = this.vpW;
    f[4] = this.vpH;
    f[5] = layout.W;
    f[6] = layout.H;
    f[7] = s.pointRadiusPx * Math.min(2, window.devicePixelRatio || 1);
    f[8] = s.speedRef;
    f[9] = s.exposure;
    f[10] = s.showSolution ? 1 : 0;
    f[11] = t;
    this.rpu[12] = gb.nC;
    this.rpu[13] = gb.nU;
    this.rpu[14] = gb.nV;
    f[15] = s.bloom;
    this.device.queue.writeBuffer(this.renderParamsBuf, 0, this.rpData);
  }

  /** One simulated 60 Hz frame: spawn, then `substeps` × (P2G → project → G2P). Own submission so the uniform write lands per step. */
  private simulateStep(first: boolean, last: boolean): void {
    const gb = this.gb;
    if (!gb) return;
    const s = this.settings;
    const encoder = this.device.createCommandEncoder();
    const frameDt = 1 / 60;
    const dt = frameDt / s.substeps;
    let inletCount = 0;
    if (s.pouring) {
      this.spawnAccum += s.pourRate * frameDt;
      inletCount = Math.floor(this.spawnAccum);
      this.spawnAccum -= inletCount;
    }
    const srcCount = this.pointer ? Math.floor(s.pourRate * frameDt * 0.6) : 0;
    const spawnTotal = inletCount + srcCount;
    this.writeParams(spawnTotal, inletCount, srcCount, dt);

    let tsWrites: GPUComputePassTimestampWrites | undefined;
    if (this.querySet && !this.mapPending && (first || last)) {
      tsWrites = { querySet: this.querySet };
      if (first) tsWrites.beginningOfPassWriteIndex = 0;
      if (last) tsWrites.endOfPassWriteIndex = 1;
    }
    const pass = encoder.beginComputePass(tsWrites ? { timestampWrites: tsWrites } : {});
    pass.setBindGroup(0, gb.bg0);
    pass.setBindGroup(1, this.parityBGs[0]);
    if (spawnTotal > 0) {
      pass.setPipeline(this.pipes.spawn);
      pass.dispatchWorkgroups(groups(spawnTotal));
    }
    const gridGroups = groups(Math.max(gb.nU, gb.nV, gb.nC));
    const particleGroups = groups(gb.N);
    for (let sub = 0; sub < s.substeps; sub++) {
      pass.setPipeline(this.pipes.clearGrid);
      pass.dispatchWorkgroups(groups(2 * gb.nU + 2 * gb.nV + gb.nC));
      pass.setPipeline(this.pipes.p2g);
      pass.dispatchWorkgroups(particleGroups);
      pass.setPipeline(this.pipes.gridBuild);
      pass.dispatchWorkgroups(gridGroups);
      pass.setPipeline(this.pipes.solve);
      for (let it = 0; it < s.iterations; it++) {
        pass.setBindGroup(1, this.parityBGs[0]);
        pass.dispatchWorkgroups(groups(gb.nC));
        pass.setBindGroup(1, this.parityBGs[1]);
        pass.dispatchWorkgroups(groups(gb.nC));
      }
      pass.setBindGroup(1, this.parityBGs[0]);
      pass.setPipeline(this.pipes.g2p);
      pass.dispatchWorkgroups(particleGroups);
      pass.setPipeline(this.pipes.trail);
      pass.dispatchWorkgroups(groups(gb.nC));
    }
    pass.end();
    this.device.queue.submit([encoder.finish()]);
    this.spawnCursor = (this.spawnCursor + spawnTotal) % gb.N;
    this.frameIdx++;
    this.stats.frame = this.frameIdx;
  }

  private frame(t: number): void {
    const cpuStart = performance.now();
    const realDt = Math.min(0.1, (t - this.lastT) / 1000);
    this.lastT = t;
    if (realDt > 0) {
      const fps = 1 / realDt;
      this.fpsSmooth = this.fpsSmooth === 0 ? fps : this.fpsSmooth * 0.9 + fps * 0.1;
      this.stats.fps = this.fpsSmooth;
    }
    this.fitViewport();
    const layout = this.layout;
    const gb = this.gb;
    const targets = this.targets;
    if (!layout || !gb || !targets) return;

    const s = this.settings;

    if (this.pendingReset) {
      this.pendingReset = false;
      this.writeParams(0, 0, 0, 1 / 60);
      const resetEnc = this.device.createCommandEncoder();
      const pass = resetEnc.beginComputePass();
      pass.setBindGroup(0, gb.bg0);
      pass.setBindGroup(1, this.parityBGs[0]);
      pass.setPipeline(this.pipes.resetParticles);
      pass.dispatchWorkgroups(groups(gb.N));
      pass.setPipeline(this.pipes.clearGrid);
      pass.dispatchWorkgroups(groups(2 * gb.nU + 2 * gb.nV + gb.nC));
      pass.end();
      resetEnc.clearBuffer(gb.grid, (2 * gb.nU + 2 * gb.nV + gb.nC) * 4, gb.nC * 4);
      this.device.queue.submit([resetEnc.finish()]);
      this.spawnAccum = 0;
    }

    const steps = s.paused ? 0 : Math.max(1, Math.round(s.timeScale));
    for (let f = 0; f < steps; f++) this.simulateStep(f === 0, f === steps - 1);

    const encoder = this.device.createCommandEncoder();
    // ---- render
    this.writeRenderParams(t / 1000);
    {
      const pass = encoder.beginRenderPass({
        colorAttachments: [{ view: targets.hdr.createView(), loadOp: 'clear', storeOp: 'store', clearValue: { r: 0, g: 0, b: 0, a: 0 } }],
      });
      pass.setPipeline(this.particlePipe);
      pass.setBindGroup(0, targets.particleBG);
      pass.draw(6, gb.N);
      pass.end();
    }
    {
      const pass = encoder.beginRenderPass({
        colorAttachments: [{ view: targets.bloomA.createView(), loadOp: 'clear', storeOp: 'store' }],
      });
      pass.setPipeline(this.blurPipe);
      pass.setBindGroup(0, targets.blurABG);
      pass.draw(3);
      pass.end();
    }
    {
      const pass = encoder.beginRenderPass({
        colorAttachments: [{ view: targets.bloomB.createView(), loadOp: 'clear', storeOp: 'store' }],
      });
      pass.setPipeline(this.blurPipe);
      pass.setBindGroup(0, targets.blurBBG);
      pass.draw(3);
      pass.end();
    }
    {
      const pass = encoder.beginRenderPass({
        colorAttachments: [{ view: this.ctx.getCurrentTexture().createView(), loadOp: 'clear', storeOp: 'store' }],
      });
      pass.setPipeline(this.compositePipe);
      pass.setBindGroup(0, targets.compositeBG);
      pass.draw(3);
      pass.end();
    }

    const doReadback = !this.mapPending;
    if (doReadback) {
      encoder.copyBufferToBuffer(this.statsBuf, 0, this.readback, 0, 16);
      if (this.querySet && this.tsResolve && !s.paused) {
        encoder.resolveQuerySet(this.querySet, 0, 2, this.tsResolve, 0);
        encoder.copyBufferToBuffer(this.tsResolve, 0, this.readback, 16, 16);
      }
    }
    this.device.queue.submit([encoder.finish()]);
    this.stats.cpuMs = performance.now() - cpuStart;

    if (doReadback) {
      this.mapPending = true;
      const paused = s.paused;
      this.readback
        .mapAsync(GPUMapMode.READ)
        .then(() => {
          if (this.destroyed) return;
          const range = this.readback.getMappedRange();
          const u = new Uint32Array(range, 0, 4);
          this.stats.alive = u[0] ?? 0;
          this.stats.drained = u[1] ?? 0;
          if (this.querySet && !paused) {
            const ts = new BigUint64Array(range, 16, 2);
            const a = ts[0] ?? 0n;
            const b = ts[1] ?? 0n;
            if (b > a) this.stats.gpuMs = Number(b - a) / 1e6;
          }
          this.readback.unmap();
        })
        .catch(() => undefined)
        .finally(() => {
          this.mapPending = false;
        });
    }
  }
}
