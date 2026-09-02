import { useEffect, useRef, useState, type RefObject } from 'react';
import { FluidSim, type SimSettings, type SimStats } from '../sim/FluidSim';
import { generateMaze, type MazeConfig, type MazeLayout } from '../maze/generate';

export interface SimHandle {
  stats: SimStats | null;
  error: string | null;
  openCells: number;
  /** seconds from last reset until the first particle drained; null until solved */
  solvedAt: number | null;
  resetWater: () => void;
  setPointer: (p: { x: number; y: number } | null) => void;
}

export function useFluidSim(canvasRef: RefObject<HTMLCanvasElement | null>, maze: MazeConfig, settings: SimSettings): SimHandle {
  const simRef = useRef<FluidSim | null>(null);
  const layoutRef = useRef<MazeLayout | null>(null);
  const settingsRef = useRef(settings);
  settingsRef.current = settings;
  const resetTimeRef = useRef(performance.now());

  const [error, setError] = useState<string | null>(null);
  const [stats, setStats] = useState<SimStats | null>(null);
  const [openCells, setOpenCells] = useState(0);
  const [solvedAt, setSolvedAt] = useState<number | null>(null);

  // create once per canvas
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    let cancelled = false;
    let sim: FluidSim | null = null;
    FluidSim.create(canvas, settingsRef.current)
      .then((s) => {
        if (cancelled) {
          s.destroy();
          return;
        }
        sim = s;
        simRef.current = s;
        const layout = layoutRef.current ?? generateMaze(maze);
        layoutRef.current = layout;
        s.setLayout(layout);
        s.updateSettings(settingsRef.current);
        s.start();
        setError(null);
      })
      .catch((e: unknown) => {
        setError(e instanceof Error ? e.message : String(e));
      });
    const timer = window.setInterval(() => {
      const s = simRef.current;
      if (!s) return;
      setStats({ ...s.stats });
    }, 200);
    return () => {
      cancelled = true;
      window.clearInterval(timer);
      simRef.current = null;
      sim?.destroy();
    };
    // the initial maze is only used before the maze effect below has run
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [canvasRef]);

  // maze changes
  useEffect(() => {
    const layout = generateMaze(maze);
    layoutRef.current = layout;
    setOpenCells(layout.openCells);
    resetTimeRef.current = performance.now();
    setSolvedAt(null);
    simRef.current?.setLayout(layout);
  }, [maze]);

  // settings changes
  useEffect(() => {
    simRef.current?.updateSettings(settings);
  }, [settings]);

  // solved detection
  useEffect(() => {
    if (!stats) return;
    if (stats.drained > 0 && solvedAt === null) {
      setSolvedAt((performance.now() - resetTimeRef.current) / 1000);
    }
  }, [stats, solvedAt]);

  return {
    stats,
    error,
    openCells,
    solvedAt,
    resetWater: () => {
      resetTimeRef.current = performance.now();
      setSolvedAt(null);
      simRef.current?.resetWater();
    },
    setPointer: (p) => simRef.current?.setPointer(p),
  };
}
