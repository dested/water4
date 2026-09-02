import { useCallback, useRef, useState, type PointerEvent as ReactPointerEvent } from 'react';
import { DEFAULT_SETTINGS, type SimSettings } from './sim/FluidSim';
import type { MazeConfig } from './maze/generate';
import { useFluidSim } from './ui/useFluidSim';
import { Panel } from './ui/Panel';

const DEFAULT_MAZE: MazeConfig = { cols: 20, rows: 20, res: 12, wall: 2, seed: 20260902 };

export function App() {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const [maze, setMaze] = useState<MazeConfig>(DEFAULT_MAZE);
  const [settings, setSettings] = useState<SimSettings>(DEFAULT_SETTINGS);
  const { stats, error, openCells, solvedAt, resetWater, setPointer } = useFluidSim(canvasRef, maze, settings);

  const pointerAt = useCallback(
    (e: ReactPointerEvent<HTMLCanvasElement>) => {
      const r = e.currentTarget.getBoundingClientRect();
      setPointer({ x: e.clientX - r.left, y: e.clientY - r.top });
    },
    [setPointer],
  );

  return (
    <div className="relative h-full w-full select-none">
      <canvas
        ref={canvasRef}
        className="block h-full w-full touch-none"
        onPointerDown={(e) => {
          e.currentTarget.setPointerCapture(e.pointerId);
          pointerAt(e);
        }}
        onPointerMove={(e) => {
          if (e.buttons & 1) pointerAt(e);
        }}
        onPointerUp={() => setPointer(null)}
        onPointerCancel={() => setPointer(null)}
        onPointerLeave={() => setPointer(null)}
      />
      <div className="pointer-events-none absolute inset-4 flex items-start justify-end">
        <Panel
          maze={maze}
          onMaze={setMaze}
          settings={settings}
          onSettings={setSettings}
          stats={stats}
          openCells={openCells}
          solvedAt={solvedAt}
          onReset={resetWater}
          maxParticles={8_000_000}
        />
      </div>
      {error && (
        <div className="absolute inset-0 flex items-center justify-center bg-black/70 p-8">
          <div className="max-w-md rounded-xl border border-red-400/30 bg-[#140b0e] p-6 text-sm text-red-200">
            <h2 className="mb-2 text-base font-semibold">WebGPU unavailable</h2>
            <p>{error}</p>
          </div>
        </div>
      )}
    </div>
  );
}
