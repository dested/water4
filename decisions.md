# water4 — Decisions

> Why things are the way they are. A recorded decision is settled unless the
> user reopens it. Newest first.

## 2026-09-02 — Plain Vite + React, no sal-starter / server
**Why:** pure client-side simulation; tRPC/Prisma would be dead weight. Confirmed by the user.
**Rejected:** sal-starter (unused server), Next.js (no SSR value for a WebGPU canvas).

## 2026-09-02 — WebGPU compute only, no WebGL2 fallback
**Why:** "millions of particles" needs compute shaders + atomics; a WebGL2 path would cap at ~500k and double the shader code. Confirmed by the user.
**Rejected:** WebGL2 transform feedback, CPU workers, dual pipeline.

## 2026-09-02 — FLIP/PIC on a MAC grid with red-black Gauss-Seidel projection on face velocities
**Why:** simplest incompressible solver that runs fully in parallel on a GPU; drift compensation (per-cell particle count vs rest density) keeps pools from collapsing. ~10 ms GPU for 2M particles on a 282×306 grid at time scale 3×.
**Rejected:** SPH (neighbour search cost, compressibility), separate pressure Poisson solve + gradient pass (more passes for the same result at this scale), multigrid (not needed yet).

## 2026-09-02 — Gravity pour + drain as the "solver"; flow trail as the reveal
**Why:** physically honest: dead ends fill and go still, the exit corridor keeps flowing. A decayed per-cell speed field rendered amber makes the live path readable without cheating.
**Rejected:** pressure-driven inlet→outlet with no gravity (dye-stream look, less "pouring water"). Could be added later as a toggle — the solver doesn't care.

## 2026-09-02 — Packed storage buffers (5 bindings) instead of one buffer per field
**Why:** default `maxStorageBuffersPerShaderStage` is 8; the naive layout needed 17. Offsets derive from nU/nV/nC in both WGSL and TS.
**Rejected:** requesting higher limits (not universally available).

## 2026-09-02 — Fixed-point `atomic<i32>` P2G scatter
**Why:** WGSL has no float atomics. SCALE_V 2048 / SCALE_W 16384 keeps 1000+ particles per face inside i32.
**Rejected:** counting-sort gather (better quality, much more code — revisit if striations bother us).

## 2026-09-02 — Speed-tinted additive particles + bloom
**Why:** cheapest look that still reads as water at 2M+ points; per-particle speed colour doubles as flow visualisation. Confirmed by the user.
**Rejected:** screen-space fluid surface (metaball threshold + refraction) — nice, but hides the "millions of particles" the user asked to see.
