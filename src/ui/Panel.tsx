import type { ReactNode } from 'react';
import type { SimSettings, SimStats } from '../sim/FluidSim';
import type { MazeConfig } from '../maze/generate';

interface SliderProps {
  label: string;
  value: number;
  min: number;
  max: number;
  step: number;
  format?: (v: number) => string;
  onChange: (v: number) => void;
}

function Slider({ label, value, min, max, step, format, onChange }: SliderProps): ReactNode {
  return (
    <label className="block">
      <div className="flex justify-between text-[11px] uppercase tracking-wider text-slate-400">
        <span>{label}</span>
        <span className="tabular-nums text-slate-200">{format ? format(value) : value}</span>
      </div>
      <input
        type="range"
        className="mt-1 w-full"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => onChange(Number(e.target.value))}
      />
    </label>
  );
}

function Toggle({ label, value, onChange }: { label: string; value: boolean; onChange: (v: boolean) => void }): ReactNode {
  return (
    <button
      type="button"
      onClick={() => onChange(!value)}
      className={`rounded-md border px-2.5 py-1 text-xs transition ${
        value
          ? 'border-sky-400/60 bg-sky-400/15 text-sky-200'
          : 'border-white/10 bg-white/5 text-slate-300 hover:bg-white/10'
      }`}
    >
      {label}
    </button>
  );
}

function Section({ title, children }: { title: string; children: ReactNode }): ReactNode {
  return (
    <section className="space-y-2.5">
      <h2 className="text-[10px] font-semibold uppercase tracking-[0.2em] text-slate-500">{title}</h2>
      {children}
    </section>
  );
}

const fmtM = (v: number): string => (v >= 1_000_000 ? `${(v / 1_000_000).toFixed(1)}M` : `${Math.round(v / 1000)}k`);
const fmtInt = (v: number): string => v.toLocaleString('en-US');

export interface PanelProps {
  maze: MazeConfig;
  onMaze: (m: MazeConfig) => void;
  settings: SimSettings;
  onSettings: (s: SimSettings) => void;
  stats: SimStats | null;
  openCells: number;
  solvedAt: number | null;
  onReset: () => void;
  maxParticles: number;
}

export function Panel({ maze, onMaze, settings, onSettings, stats, openCells, solvedAt, onReset, maxParticles }: PanelProps): ReactNode {
  const set = <K extends keyof SimSettings>(k: K, v: SimSettings[K]): void => onSettings({ ...settings, [k]: v });
  const capacity = openCells * settings.restDensity;
  return (
    <aside className="pointer-events-auto flex max-h-[calc(100vh-2rem)] w-72 flex-col gap-5 overflow-y-auto rounded-xl border border-white/10 bg-[#0b0e16]/85 p-4 shadow-2xl backdrop-blur-md">
      <header>
        <h1 className="text-base font-semibold tracking-tight text-slate-100">water4</h1>
        <p className="text-xs text-slate-400">FLIP fluid maze solver · WebGPU</p>
      </header>

      <Section title="Stats">
        <dl className="grid grid-cols-2 gap-x-3 gap-y-1 text-xs tabular-nums">
          <dt className="text-slate-500">fps</dt>
          <dd className="text-right text-slate-200">{stats ? stats.fps.toFixed(0) : '—'}</dd>
          <dt className="text-slate-500">sim gpu</dt>
          <dd className="text-right text-slate-200">{stats && Number.isFinite(stats.gpuMs) ? `${stats.gpuMs.toFixed(2)} ms` : 'n/a'}</dd>
          <dt className="text-slate-500">particles</dt>
          <dd className="text-right text-slate-200">{stats ? fmtInt(stats.alive) : '—'}</dd>
          <dt className="text-slate-500">drained</dt>
          <dd className="text-right text-slate-200">{stats ? fmtInt(stats.drained) : '—'}</dd>
          <dt className="text-slate-500">grid</dt>
          <dd className="text-right text-slate-200">{stats ? `${stats.gridW}×${stats.gridH}` : '—'}</dd>
          <dt className="text-slate-500">maze holds</dt>
          <dd className="text-right text-slate-200">{fmtM(capacity)}</dd>
        </dl>
        <div
          className={`rounded-md border px-2.5 py-1.5 text-xs ${
            solvedAt !== null
              ? 'border-amber-300/50 bg-amber-300/10 text-amber-200'
              : 'border-white/10 bg-white/5 text-slate-400'
          }`}
        >
          {solvedAt !== null ? `Solved — first drop reached the exit at ${solvedAt.toFixed(1)}s` : 'Searching… water is filling dead ends'}
        </div>
      </Section>

      <Section title="Playback">
        <div className="flex flex-wrap gap-2">
          <Toggle label={settings.paused ? 'Paused' : 'Running'} value={!settings.paused} onChange={(v) => set('paused', !v)} />
          <Toggle label="Pouring" value={settings.pouring} onChange={(v) => set('pouring', v)} />
          <Toggle label="Show answer" value={settings.showSolution} onChange={(v) => set('showSolution', v)} />
          <button
            type="button"
            onClick={onReset}
            className="rounded-md border border-white/10 bg-white/5 px-2.5 py-1 text-xs text-slate-300 hover:bg-white/10"
          >
            Drain all
          </button>
        </div>
        <p className="text-[11px] leading-snug text-slate-500">Click and hold anywhere in the maze to pour there.</p>
      </Section>

      <Section title="Maze">
        <Slider label="Size" value={maze.cols} min={6} max={48} step={1} format={(v) => `${v}×${v}`} onChange={(v) => onMaze({ ...maze, cols: v, rows: v })} />
        <Slider label="Corridor (cells)" value={maze.res} min={6} max={28} step={1} onChange={(v) => onMaze({ ...maze, res: v })} />
        <Slider label="Wall (cells)" value={maze.wall} min={1} max={6} step={1} onChange={(v) => onMaze({ ...maze, wall: v })} />
        <button
          type="button"
          onClick={() => onMaze({ ...maze, seed: (maze.seed * 1664525 + 1013904223) >>> 0 })}
          className="w-full rounded-md border border-sky-400/40 bg-sky-400/10 px-2.5 py-1.5 text-xs font-medium text-sky-200 hover:bg-sky-400/20"
        >
          New maze
        </button>
      </Section>

      <Section title="Water">
        <Slider label="Particle budget" value={settings.particleBudget} min={250_000} max={maxParticles} step={250_000} format={fmtM} onChange={(v) => set('particleBudget', v)} />
        <Slider label="Pour rate /s" value={settings.pourRate} min={10_000} max={1_000_000} step={10_000} format={fmtM} onChange={(v) => set('pourRate', v)} />
        <Slider label="Gravity" value={settings.gravity} min={20} max={600} step={10} onChange={(v) => set('gravity', v)} />
        <Slider label="FLIP ↔ PIC" value={settings.flip} min={0} max={1} step={0.01} format={(v) => v.toFixed(2)} onChange={(v) => set('flip', v)} />
        <Slider label="Rest density" value={settings.restDensity} min={4} max={64} step={1} onChange={(v) => set('restDensity', v)} />
        <Slider label="Pressure iters" value={settings.iterations} min={5} max={120} step={5} onChange={(v) => set('iterations', v)} />
        <Slider label="Substeps" value={settings.substeps} min={1} max={4} step={1} onChange={(v) => set('substeps', v)} />
        <Slider label="Time scale" value={settings.timeScale} min={1} max={8} step={1} format={(v) => `${v}×`} onChange={(v) => set('timeScale', v)} />
      </Section>

      <Section title="Look">
        <Slider label="Brightness" value={settings.exposure} min={0.01} max={0.5} step={0.005} format={(v) => v.toFixed(3)} onChange={(v) => set('exposure', v)} />
        <Slider label="Point size (px)" value={settings.pointRadiusPx} min={0.5} max={5} step={0.1} format={(v) => v.toFixed(1)} onChange={(v) => set('pointRadiusPx', v)} />
        <Slider label="Bloom" value={settings.bloom} min={0} max={2} step={0.05} format={(v) => v.toFixed(2)} onChange={(v) => set('bloom', v)} />
        <Slider label="Speed → white" value={settings.speedRef} min={10} max={400} step={5} onChange={(v) => set('speedRef', v)} />
      </Section>
    </aside>
  );
}
