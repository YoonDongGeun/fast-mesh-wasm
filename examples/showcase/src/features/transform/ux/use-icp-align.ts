// Rigid ICP registration from the align panel (#195): which meshes move and
// stay fixed (a pick while it is loaded, else the selection or the first
// mesh, and never the same mesh twice), the clamped iteration, overlap and
// distance options, and the run through `runIcpAlign` with its result.
import type { MeshHandle, RegisterResult } from "fast-mesh-wasm";
import type { SceneStore } from "fast-mesh-wasm/store";
import { useState } from "react";
import type { Item } from "../../../app/state/app-state";
import { runIcpAlign } from "../../../engine/workbench-ops";

/** `current` while it is loaded and not `exclude`, else the first mesh other than `exclude`. */
export function pickDistinctMesh(items: readonly Item[], exclude: MeshHandle | null, current: MeshHandle | null): MeshHandle | null {
  if (current !== null && items.some((i) => i.mesh === current) && current !== exclude) {
    return current;
  }
  return items.find((i) => i.mesh !== exclude)?.mesh ?? null;
}

/** The align panel's picks, options, run state and result. */
export function useIcpAlign(store: SceneStore, items: readonly Item[], selection: MeshHandle | null) {
  const [movingPick, setMovingPick] = useState<MeshHandle | null>(null);
  const [fixedPick, setFixedPick] = useState<MeshHandle | null>(null);
  const [maxIterations, setMaxIterations] = useState(50);
  const [overlap, setOverlap] = useState(1);
  const [maxDistance, setMaxDistance] = useState(0);

  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<RegisterResult | null>(null);

  const resolvedMoving =
    (movingPick !== null && items.some((i) => i.mesh === movingPick) ? movingPick : null) ??
    (selection !== null && items.some((i) => i.mesh === selection) ? selection : null) ??
    items[0]?.mesh ??
    null;
  const resolvedFixed = pickDistinctMesh(items, resolvedMoving, fixedPick);

  const canRun = !busy && resolvedMoving !== null && resolvedFixed !== null && resolvedMoving !== resolvedFixed;

  const onRun = async () => {
    if (!canRun || resolvedMoving === null || resolvedFixed === null) {
      return;
    }
    setBusy(true);
    setError(null);
    try {
      const r = await runIcpAlign({
        store,
        moving: resolvedMoving,
        fixed: resolvedFixed,
        maxIterations: Math.max(1, Math.round(maxIterations)),
        overlap: Math.max(0.05, Math.min(1, overlap)),
        maxDistance: maxDistance > 0 ? maxDistance : undefined,
      });
      setResult(r);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  };

  return {
    resolvedMoving,
    resolvedFixed,
    canRun,
    maxIterations,
    overlap,
    maxDistance,
    busy,
    error,
    result,
    /** Picks the moving mesh by its index in `items`. */
    chooseMoving: (index: number) => setMovingPick(items[index]?.mesh ?? null),
    /** Picks the fixed mesh by its index in `items`. */
    chooseFixed: (index: number) => setFixedPick(items[index]?.mesh ?? null),
    /** Sets the iteration cap from a typed number (a whole number, at least 1). */
    setMaxIterations: (n: number) => setMaxIterations(Math.max(1, Math.round(n || 1))),
    /** Sets the expected overlap from a typed number (0.05 to 1). */
    setOverlap: (n: number) => setOverlap(Math.max(0.05, Math.min(1, n || 1))),
    /** Sets the correspondence distance cap (mm; 0 means none). */
    setMaxDistance: (mm: number) => setMaxDistance(Math.max(0, mm)),
    onRun,
  };
}
