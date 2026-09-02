# water4 — Verify

> How to prove the app works. Scale to the change: [cheap] always,
> flow recipes when touched, [heavy] only after asking.

## Commands

| What | Command | Cost |
| --- | --- | --- |
| Type-check | `bun run typecheck` | [cheap] |
| Build | `bun run build` | [cheap] |
| Dev server | `bun run dev` → https://water4.localhost | [cheap] |

No test runner. WebGPU is only exercised in a browser — use the `bx` skill.

## Critical flows

### Boots and simulates [cheap]
Touchpoints: `src/sim/*`, `src/ui/useFluidSim.ts`
1. `bx open https://water4.localhost`
2. `bx wait 4000; bx text aside` → fps ≈ 60, `sim gpu` a few ms, `particles` climbing.
3. `bx console` → nothing but the Windows `powerPreference` warning.

### Solves the default maze [medium]
Touchpoints: `src/maze/generate.ts`, `sim.wgsl` (`spawn`, `g2p`, `solve`)
1. Open, wait ~45 s at default settings (time scale 3×; solved at ~40 s on a 20×20 maze).
2. `bx text aside` → `drained` > 0 and the status pill reads "Solved — …".
3. `bx snap` → blue pools in dead ends, an amber stream along the exit corridor.

### Controls round-trip [cheap]
Touchpoints: `src/ui/Panel.tsx`, `FluidSim.updateSettings`
1. Click "New maze" → grid changes, particles reset to 0 and refill.
2. Drag "Particle budget" → no console errors, panel "maze holds" unchanged, sim continues.
3. Click "Drain all" → particles → 0, drained → 0.
