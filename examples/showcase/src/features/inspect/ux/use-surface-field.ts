// Surface field analysis in the inspect panel: the field modes and
// colormaps it offers, which mesh it colors (and measures against), and
// the compute and recolor calls behind its controls.
import type { FieldHandle, MeshHandle } from "fast-mesh-wasm";
import { useScene } from "fast-mesh-wasm/react";
import { useRef, useState } from "react";
import {
  type AppState,
  type Item,
  type SurfaceColormap,
  type SurfaceFieldMode,
  type SurfaceFieldState,
  useApp,
} from "../../../app/state/app-state";
import {
  applySurfaceField,
  clearAppliedSurfaceField,
  isWorkerScene,
} from "../../../engine/workbench-ops";

export const SURFACE_FIELD_MODES: readonly { id: SurfaceFieldMode; label: string; title: string }[] = [
  { id: "none", label: "Off", title: "Turn off surface field analysis" },
  { id: "distance", label: "Distance", title: "Contact distance field to another mesh" },
  { id: "thickness", label: "Thickness", title: "Wall thickness to opposite surface" },
  { id: "meanCurvature", label: "Mean κ", title: "Signed mean curvature (addCurvatureField)" },
  { id: "gaussianCurvature", label: "Gauss κ", title: "Discrete Gaussian curvature (angle defect / area)" },
];

export const COLORMAPS: readonly { id: SurfaceColormap; label: string; gradient: string }[] = [
  { id: "turbo", label: "Turbo", gradient: "linear-gradient(90deg,#30123b,#287ced,#2ed673,#f9b724,#7a0403)" },
  { id: "viridis", label: "Viridis", gradient: "linear-gradient(90deg,#440154,#3b528b,#21918c,#5ec962,#fde725)" },
  { id: "coolwarm", label: "Coolwarm", gradient: "linear-gradient(90deg,#3b4cc0,#dddddd,#b40426)" },
  { id: "grayscale", label: "Grayscale", gradient: "linear-gradient(90deg,#000000,#ffffff)" },
];

/** A computed field's value statistics, for the readout and the Auto range. */
export interface SurfaceFieldSummary {
  min: number;
  max: number;
  mean: number;
  count: number;
}

/**
 * The surface field controls' state and handlers. The inspected mesh is the
 * field's own pick while it is loaded, else the selection, else the first
 * mesh; the distance target is its own pick unless that is the inspected
 * mesh, else the first other mesh.
 */
export function useSurfaceField(app: AppState, items: readonly Item[], selection: MeshHandle | null) {
  const store = useScene();
  const sf = useApp(app, (s) => s.surfaceField);
  const overlayValues = useApp(app, (s) => s.surfaceFieldValues);

  const activeFieldRef = useRef<{ mesh: MeshHandle | null; field: FieldHandle | null }>({ mesh: null, field: null });
  const [summary, setSummary] = useState<SurfaceFieldSummary | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const inspectedMesh =
    (sf.mesh !== null && items.some((i) => i.mesh === sf.mesh) ? sf.mesh : null) ??
    (selection !== null && items.some((i) => i.mesh === selection) ? selection : null) ??
    items[0]?.mesh ??
    null;
  const targetMesh =
    (sf.target !== null && items.some((i) => i.mesh === sf.target) && sf.target !== inspectedMesh ? sf.target : null) ??
    items.find((i) => i.mesh !== inspectedMesh)?.mesh ??
    null;

  const computeAndApply = async (nextState: SurfaceFieldState, autoRange: boolean) => {
    clearAppliedSurfaceField(store, app, activeFieldRef.current);
    activeFieldRef.current = { mesh: null, field: null };
    setError(null);
    if (nextState.mode === "none" || nextState.mesh === null) {
      setSummary(null);
      return;
    }
    setBusy(true);
    try {
      const applied = await applySurfaceField({ store, app, state: nextState, autoRange });
      activeFieldRef.current = { mesh: applied.mesh, field: applied.field };
      setSummary(applied.summary);
    } catch (e) {
      setSummary(null);
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  };

  const switchMode = (mode: SurfaceFieldMode) => {
    const nextState: SurfaceFieldState = {
      ...sf,
      mode,
      mesh: inspectedMesh,
      target: targetMesh,
    };
    app.setSurfaceField(nextState);
    void computeAndApply(nextState, true);
  };

  /** Picks the field's mesh by its index in `items`; a target equal to it moves to another mesh. */
  const chooseMesh = (index: number) => {
    const nextMesh = items[index]?.mesh ?? null;
    const nextTarget = targetMesh === nextMesh ? (items.find((i) => i.mesh !== nextMesh)?.mesh ?? null) : targetMesh;
    const nextState = { ...sf, mesh: nextMesh, target: nextTarget };
    app.setSurfaceField(nextState);
    void computeAndApply(nextState, true);
  };

  /** Picks the distance target by its index in `items`. */
  const chooseTarget = (index: number) => {
    const nextTarget = items[index]?.mesh ?? null;
    const nextState = { ...sf, mesh: inspectedMesh, target: nextTarget };
    app.setSurfaceField(nextState);
    void computeAndApply(nextState, true);
  };

  /** Sets the distance or thickness limit (mm, at least 0.01) and recomputes without an auto range. */
  const setLimit = (mm: number) => {
    const nextMax = Math.max(0.01, mm);
    const nextState = { ...sf, mesh: inspectedMesh, target: targetMesh, maxDistance: nextMax };
    app.setSurfaceField({ maxDistance: nextMax });
    void computeAndApply(nextState, false);
  };

  const updateAppearance = (patch: { map?: SurfaceColormap; min?: number; max?: number }) => {
    const nextMap = patch.map ?? sf.map;
    const nextMin = patch.min ?? sf.min;
    const nextMax = patch.max ?? sf.max;
    app.setSurfaceField({ map: nextMap, min: nextMin, max: nextMax });
    if (sf.mode === "none" || inspectedMesh === null) return;
    const lo = nextMin;
    const hi = nextMax > nextMin ? nextMax : nextMin + 1;

    if (overlayValues !== null && overlayValues.mesh === inspectedMesh) {
      app.setSurfaceFieldValues({ ...overlayValues, map: nextMap, min: lo, max: hi });
      return;
    }
    if (sf.mode === "distance" && isWorkerScene(store.scene) && targetMesh !== null && targetMesh !== inspectedMesh) {
      store.setDisplay(inspectedMesh, {
        coloring: { distanceTo: targetMesh, maxDistance: Math.max(sf.maxDistance, 1e-4), map: nextMap, range: [lo, hi] },
      });
      return;
    }
    if (!isWorkerScene(store.scene) && activeFieldRef.current.field !== null) {
      store.scene.inspect?.setColoring(inspectedMesh, {
        field: activeFieldRef.current.field,
        map: nextMap,
        range: [lo, hi],
      });
    }
  };

  /** Sets one end of the color range from a typed number, ignoring non-finite input. */
  const setRangeEnd = (end: "min" | "max", v: number) => {
    if (Number.isFinite(v)) updateAppearance(end === "min" ? { min: v } : { max: v });
  };

  /** Fits the color range to the computed values (a flat field gets a unit span). */
  const autoRange = () => {
    if (summary === null) return;
    const lo = summary.min;
    const hi = summary.max > summary.min ? summary.max : summary.min + 1;
    updateAppearance({ min: lo, max: hi });
  };

  const mapMeta = COLORMAPS.find((c) => c.id === sf.map) ?? COLORMAPS[0];

  return {
    sf,
    summary,
    busy,
    error,
    inspectedMesh,
    targetMesh,
    mapMeta,
    switchMode,
    chooseMesh,
    chooseTarget,
    setLimit,
    updateAppearance,
    setRangeEnd,
    autoRange,
  };
}
