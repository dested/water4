# water4 — UI / Visual Language

> The source of truth for how water4 **looks and feels**. Follow it for anything visual.

## North star

**Dark lab bench, glowing water.** The canvas is the product; the UI is a quiet
instrument panel floating over it. Reference feel: Shadertoy demos + a well-made
synth plugin. Failure modes: a busy dashboard that competes with the sim (too much
chrome), or a bare debug page (too raw).

1. **Canvas first** — panel is translucent, narrow, and never covers the maze centre.
2. **One accent, two meanings** — sky blue = controls/active; amber = "the answer / the flow".
3. **Numbers are data** — every value is `tabular-nums`, right-aligned, terse.
4. **No decoration that isn't light** — glow, bloom and translucency, never gradients-for-fun.

## Tokens

### Color
| Token | Value / class | Use |
| --- | --- | --- |
| Page background | `#07080d` (`index.css`) | behind the canvas |
| Letterbox / canvas bg | `vec3(0.022,0.024,0.040)` in `fsComposite` | outside the maze |
| Maze floor | `vec3(0.040,0.044,0.070)` | open cells |
| Wall | `vec3(0.115,0.125,0.175)` + edge lift `+vec3(0.09,0.10,0.15)` | solid cells, bevel via SDF |
| Water deep / mid / hot | `(0.02,0.16,0.80)` / `(0.10,0.62,1.00)` / `(0.85,0.97,1.00)` | slow → fast particles |
| Trail / solution | amber `(1.0,0.42,0.10)`, solution overlay `+(0.11,0.045,0)` | live flow path, "show answer" |
| Panel surface | `bg-[#0b0e16]/85 backdrop-blur-md border-white/10` | the control panel |
| Text primary / muted / eyebrow | `text-slate-100` / `text-slate-400` / `text-slate-500` | |
| Accent (controls) | `sky-400` family (`border-sky-400/60 bg-sky-400/15 text-sky-200`) | active toggles, primary button |
| Status: solved | `amber-300` family | the solved badge only |
| Danger | `red-400/30` border, `red-200` text | WebGPU-missing card only |

### Typography
| Role | Class | Use |
| --- | --- | --- |
| Title | `text-base font-semibold tracking-tight` | "water4" |
| Section eyebrow | `text-[10px] font-semibold uppercase tracking-[0.2em] text-slate-500` | STATS / MAZE / … |
| Slider label | `text-[11px] uppercase tracking-wider text-slate-400` | |
| Values | `text-xs tabular-nums text-slate-200` | stats, slider readouts |
| Help copy | `text-[11px] leading-snug text-slate-500` | one line max |

System UI font stack (`index.css`). No web fonts.

### Spacing, shape, elevation
| Token | Value | Use |
| --- | --- | --- |
| Panel | `w-72 p-4 gap-5 rounded-xl shadow-2xl` | fixed top-right, `inset-4` |
| Controls | `rounded-md px-2.5 py-1 text-xs` | toggles + buttons |
| Radii | `rounded-md` controls, `rounded-xl` panel, nothing else | |
| Elevation | translucency + hairline border, one `shadow-2xl` on the panel | no other shadows |

## Layout

Full-viewport canvas (`h-full w-full`), maze letterboxed with a 24px×dpr margin.
Panel is absolutely positioned top-right, scrolls internally (`max-h-[calc(100vh-2rem)]`),
`pointer-events-none` wrapper with `pointer-events-auto` on the panel so clicks fall
through to the canvas.

## Components

| Component | File | Purpose |
| --- | --- | --- |
| `Slider` | `src/ui/Panel.tsx` | label + readout + native range (accent via `accent-color`) |
| `Toggle` | `src/ui/Panel.tsx` | pill button; sky when on |
| `Section` | `src/ui/Panel.tsx` | eyebrow + stack |
| `Panel` | `src/ui/Panel.tsx` | the whole side panel |

Toggle (on state):
```tsx
<button className="rounded-md border border-sky-400/60 bg-sky-400/15 px-2.5 py-1 text-xs text-sky-200">Pouring</button>
```

## States

- Sim loading: stats show `—`.
- No WebGPU: centred red-bordered card over a `bg-black/70` scrim.
- Searching / solved: single status pill under the stats, muted → amber.

## Voice / copy

Lowercase eyebrows, sentence-case buttons, terse. "Drain all", "New maze", "Show answer".
Never exclamation marks. Never "Oops".

## Don'ts

- ❌ Any second accent colour — amber is reserved for flow/answer, sky for controls.
- ❌ Opaque panels or panels wider than `w-72` — the maze is the product.
- ❌ Web fonts, icon libraries, gradients on UI chrome.
- ❌ Rendering text/overlays on the canvas itself — everything textual lives in the panel.
