# water4 — CliffNotes

> Living map of the project. Read this before any coding session.
> Last updated: 2026-09-02.

## What this is

A browser toy that solves a maze with water. A perfect maze is generated, water is
poured into the top-left entrance, and a GPU FLIP fluid simulation with millions of
particles fills dead ends and streams through the corridor that leads to the drain at
the bottom-right exit. No server, no persistence — one WebGPU canvas plus a React
control panel.

## Quick Reference

- **Dev:** `bun run dev` → https://water4.localhost (portless; raw Vite port is whatever portless assigns)
- **Entry point:** `src/main.tsx` → `<App />` (`src/App.tsx`)
- **Type-check:** `bun run typecheck` (`tsc --noEmit`, strict + `noUncheckedIndexedAccess`)
- **Build:** `bun run build`
- **Test:** no test runner configured
- **Units:** the simulation world is measured in **grid cells**. Positions are cell
  coordinates (y down, gravity is +y), velocities are cells/s. One maze corridor is
  `res` cells wide (default 12), walls are `wall` cells (default 2).
- **Requires WebGPU** (Chrome/Edge 113+). No fallback renderer — the app shows an error card otherwise.

## Stack

| Layer | Choice | Notes |
| --- | --- | --- |
| Runtime / pkg mgr | Bun | `bun install`, `bun run …` |
| Framework | React 19 + Vite 7 | plain SPA, no router |
| Styling | Tailwind v4 (`@tailwindcss/vite`) | tokens in `ui.md` |
| Simulation | WebGPU compute (WGSL) | FLIP/PIC on a MAC grid, atomics P2G |
| Rendering | WebGPU render pipelines | additive HDR splat → bloom → composite |
| Dev URL | portless (`portless.json` name `water4`) | HMR is pinned to `wss://water4.localhost:443` in `vite.config.ts` |
| Deploy | none yet | static build works (`dist/`) |

## Directory structure

```
index.html                 Vite HTML shell
vite.config.ts             React + Tailwind plugins; PORT from portless; HMR over wss
portless.json              { "name": "water4" }
src/
  main.tsx                 React root
  index.css                Tailwind import + page background/font
  App.tsx                  Canvas + pointer handling + <Panel/>; owns maze/settings state
  maze/
    generate.ts            Maze gen (recursive backtracker), BFS solution, rasterize to grid, exact EDT signed distance field
  sim/
    FluidSim.ts            WebGPU orchestration: buffers, pipelines, per-frame compute + render, stats readback
    sim.wgsl               Compute kernels (spawn, p2g, gridBuild, solve, g2p, trail, clear, reset)
    render.wgsl            Particle splat, separable blur, composite fragment
  ui/
    useFluidSim.ts         React hook: creates the sim once, pushes maze/settings, polls stats, detects "solved"
    Panel.tsx              Control panel (sliders/toggles/stats)
cliffnotes.md ui.md decisions.md updates.md verify.md   Doc kit
plans/                     Dated working docs (create on first need)
```

## File map (concept → path)

| Concept / task | Location |
| --- | --- |
| Default sim tuning (gravity, FLIP ratio, iterations, exposure…) | `src/sim/FluidSim.ts` → `DEFAULT_SETTINGS` |
| Default maze size / corridor / wall | `src/App.tsx` → `DEFAULT_MAZE` |
| Maze geometry, entrance/exit placement, SDF | `src/maze/generate.ts` |
| Physics kernels | `src/sim/sim.wgsl` |
| Water colours, wall look, trail glow, bloom | `src/sim/render.wgsl` (`vsParticle` colours, `fsComposite`) |
| Uniform layout (Params struct ↔ `writeParams`) | `src/sim/sim.wgsl` `Params` + `FluidSim.ts` `writeParams` — keep in lockstep |
| Panel controls | `src/ui/Panel.tsx` |
| Pointer pouring | `src/App.tsx` (events) → `FluidSim.setPointer` → `spawn` kernel `srcX/srcY` |

## Routes / URLs

| Route / URL | Serves | File |
| --- | --- | --- |
| `/` | the whole app | `src/App.tsx` |

## Architecture

```
React (App/Panel) ── settings, MazeConfig ──▶ useFluidSim ──▶ FluidSim
                                                              │
   generateMaze(cfg) ─ cells flags + SDF ─────────────────────┤ setLayout → grid buffers
                                                              │
   each rAF: timeScale × simulateStep()  →  render passes  →  stats readback (mapAsync)
```

Per simulated 60 Hz frame (`simulateStep`): `spawn` → for each substep:
`clearGrid` → `p2g` (atomic fixed-point scatter of velocity + weights + per-cell count)
→ `gridBuild` (normalise, zero solid faces, classify cells) → `solve` × iterations
(red-black Gauss-Seidel pressure projection applied straight to face velocities, with
particle-density drift compensation) → `g2p` (FLIP/PIC blend, gravity, advect, SDF
push-out, drain kill, alive count) → `trail` (decayed per-cell speed used to light the
active path).

Render: particles drawn as instanced quads into an `rgba16float` HDR target with
additive blending, two half-res blur passes for bloom, then one composite fragment that
draws walls from the cell flags/SDF, the amber trail, and tonemapped water.

## Key types

| Type | Where | Purpose |
| --- | --- | --- |
| `MazeConfig` / `MazeLayout` | `src/maze/generate.ts` | input knobs / rasterised grid (W,H, `cells`, `sdf`, inlet, drainY) |
| `SimSettings` | `src/sim/FluidSim.ts` | every runtime knob; `DEFAULT_SETTINGS` is the tuned baseline |
| `SimStats` | `src/sim/FluidSim.ts` | fps, alive, drained, GPU ms (timestamp queries), grid size |
| `Params` (WGSL) | `src/sim/sim.wgsl` | 32-word uniform; word order must match `writeParams` |

## Systems

### GPU buffer packing
Limits on storage buffers per stage are tight, so everything is packed:
- `particles`: `vec4f` per particle — xy position (cells), zw velocity; **x < 0 means dead**.
- `acc` (`atomic<i32>`): `[U | V | wU | wV | count]` fixed-point (SCALE_V 2048, SCALE_W 16384).
- `grid` (`f32`): `[u | v | uPrev | vPrev | sdf | trail]`.
- `cells` (`u32`): `[cellType 0 air/1 fluid/2 solid | flags bit0 solid, bit1 solution]`.
Offsets are derived from `nU=(W+1)·H`, `nV=W·(H+1)`, `nC=W·H` in both WGSL and TS.

### Particle lifecycle
Fixed-size ring buffer of `particleBudget` slots. `spawn` walks a cursor through the
buffer and revives dead slots at the inlet (or the pointer). Spawning is gated on the
target cell holding < 0.8 × rest density, so pouring self-throttles when backed up.
`g2p` kills particles below `drainY` and counts them in `stats[1]`.

### Solve / "the maze solving"
Gravity does the search: dead ends fill and go still; the corridor to the drain keeps
flowing. The `trail` grid stores decayed speed per cell and is rendered warm, so the
live path glows and dead pools fade to blue. "Solved" in the UI = first drained particle.

## Common tasks (how to modify)

### Add a sim knob
1. Add the field to `SimSettings` + `DEFAULT_SETTINGS` (`FluidSim.ts`).
2. If the GPU needs it: add a word to `Params` in `sim.wgsl` **and** the same index in `writeParams`.
3. Add a `<Slider/>` in `Panel.tsx`.

### Change the look of the water
`render.wgsl`: `vsParticle` picks colour by speed (`deep/mid/hot`), `fsParticle` the
sprite falloff, `fsComposite` walls/trail/tonemap. Brightness defaults in `DEFAULT_SETTINGS`.

## Gotchas & hard rules

- **Params word order is a contract** between `sim.wgsl` and `writeParams`. Misalign one and everything silently explodes.
- **CFL:** `maxSpeed = 0.9 / dt` per substep. Raising gravity a lot without raising substeps makes water tunnel through walls.
- **Never `await` inside the frame loop.** Stats come back through a single `readback` buffer guarded by `mapPending`.
- `queue.writeBuffer` before `submit` all land in order, but a uniform written twice before one submit keeps only the last value — that's why `simulateStep` submits per step.
- Particle budget above the device limits is clamped (`maxParticles`, from `maxStorageBufferBindingSize`).
- HMR over portless needs the explicit `server.hmr` block; without it Vite's client dials `ws://localhost:24678` and fails.

## Status

- **[Done]** maze gen + SDF, WebGPU FLIP solver, additive/bloom renderer, control panel, pointer pouring, solved detection, GPU timing.
- **[Not built]** WebGL2 fallback, screen-space fluid surface, deployment.
- **Next:** tune defaults on a real full-screen window; consider APIC transfer to remove FLIP striations.
