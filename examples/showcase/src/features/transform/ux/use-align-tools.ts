// The align panel's landmark, coarse + ICP and symmetry tools (#253). The
// landmark pairs belong to the panel's moving and fixed meshes (choosing
// others clears them) and picking stands down when another pointer tool
// arms. Every alignment is one undo step; so is a mirror across the plane.
import type { MeshHandle, Vec3 } from "fast-mesh-wasm";
import type { SceneStore } from "fast-mesh-wasm/store";
import { useEffect, useState } from "react";
import { type AppState, type Item, useApp } from "../../../app/state/app-state";
import {
  mirrorMesh,
  runDetectSymmetryPlane,
  runLandmarkAlign,
  runMultistageAlign,
} from "../../../engine/workbench-ops";
import {
  type AlignStage,
  completePairs,
  landmarkPairs,
  landmarkResiduals,
  MIN_LANDMARK_PAIRS,
  multistageStages,
  planeToLocal,
  planeToWorld,
  removeLastLandmark,
  startLandmarks,
} from "./align-tools";

const message = (e: unknown) => (e instanceof Error ? e.message : String(e));

/** `pick` while it is loaded, else `selection` while it is loaded, else the first mesh. */
export function resolveMesh(items: readonly Item[], pick: MeshHandle | null, selection: MeshHandle | null): MeshHandle | null {
  const loaded = (m: MeshHandle | null) => m !== null && items.some((i) => i.mesh === m);
  return loaded(pick) ? pick : loaded(selection) ? selection : (items[0]?.mesh ?? null);
}

/** Disarms the other pointer tools, so landmark picking owns the canvas. */
function disarmOtherTools(app: AppState): void {
  const s = app.getSnapshot();
  if (s.sculpt !== null) app.setSculpt(null);
  if (s.maskPaint) app.setMaskPaint(false);
  if (s.inspectTool !== null) app.setInspectTool(null);
  if (s.placementTool.pick !== null) app.setPlacementTool({ pick: null });
  app.disarmWave6Tools();
}

/** The latest landmark alignment: its RMSE (mm), scale and each pair's distance after it (mm). */
export interface LandmarkResult {
  rmse: number;
  scale: number;
  residuals: number[];
}

/** Landmark pairs between `moving` and `fixed`: picking, editing and the alignment. */
export function useLandmarkAlign(app: AppState, store: SceneStore, moving: MeshHandle | null, fixed: MeshHandle | null) {
  const set = useApp(app, (s) => s.alignTool.landmarks);
  const otherToolArmed = useApp(
    app,
    (s) =>
      s.sculpt !== null ||
      s.maskPaint ||
      s.inspectTool !== null ||
      s.placementTool.pick !== null ||
      s.wave6.select.activeTool !== null ||
      s.wave6.deform.activeTool !== null ||
      s.wave6.printPrep.activeTool !== null,
  );
  const [allowScaling, setAllowScaling] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<LandmarkResult | null>(null);

  // The pairs are on the panel's meshes: other meshes, or a removed one, clear them.
  useEffect(() => {
    if (set !== null && (set.moving !== moving || set.fixed !== fixed)) {
      app.setAlignTool({ landmarks: null });
      setResult(null);
    }
  }, [app, set, moving, fixed]);

  useEffect(() => {
    if (set !== null && set.picking && otherToolArmed) {
      app.setAlignTool({ landmarks: { ...set, picking: false } });
    }
  }, [app, set, otherToolArmed]);

  const pairs = set === null ? 0 : completePairs(set);
  const canAlign = !busy && set !== null && pairs >= MIN_LANDMARK_PAIRS;

  /** Starts (or resumes) picking pairs on the canvas. */
  const pick = () => {
    if (moving === null || fixed === null || moving === fixed) return;
    disarmOtherTools(app);
    const current = app.getSnapshot().alignTool.landmarks;
    const same = current !== null && current.moving === moving && current.fixed === fixed;
    app.setAlignTool({ landmarks: same ? { ...current, picking: true } : startLandmarks(moving, fixed) });
    setError(null);
  };

  /** Stops picking and keeps the points. */
  const stop = () => {
    const current = app.getSnapshot().alignTool.landmarks;
    if (current !== null && current.picking) {
      app.setAlignTool({ landmarks: { ...current, picking: false } });
    }
  };

  /** Drops the latest point. */
  const undoPoint = () => {
    const current = app.getSnapshot().alignTool.landmarks;
    if (current !== null) {
      app.setAlignTool({ landmarks: removeLastLandmark(current) });
    }
  };

  /** Drops every point and the report. */
  const clear = () => {
    app.setAlignTool({ landmarks: null });
    setResult(null);
    setError(null);
  };

  /** Moves the moving mesh onto the fixed one by the complete pairs (one undo step). */
  const align = async () => {
    const current = app.getSnapshot().alignTool.landmarks;
    if (!canAlign || current === null) return;
    stop();
    setBusy(true);
    setError(null);
    try {
      const scene = store.scene;
      const r = await runLandmarkAlign({
        store,
        moving: current.moving,
        fixed: current.fixed,
        pairs: landmarkPairs(current, scene.matrix4(current.moving), scene.matrix4(current.fixed)),
        allowScaling,
      });
      const residuals = landmarkResiduals(current, scene.matrix4(current.moving), scene.matrix4(current.fixed));
      setResult({ rmse: r.rmse, scale: r.scale, residuals });
    } catch (e) {
      setError(message(e));
    } finally {
      setBusy(false);
    }
  };

  return { set, pairs, canAlign, allowScaling, setAllowScaling, busy, error, result, pick, stop, undoPoint, clear, align };
}

/** Coarse + ICP between `moving` and `fixed`, with the ICP options of the panel. */
export function useMultistageAlign(
  store: SceneStore,
  moving: MeshHandle | null,
  fixed: MeshHandle | null,
  options: { maxIterations: number; overlap: number; maxDistance: number },
) {
  const [coarseMethod, setCoarseMethod] = useState<"orientedBoundingBox" | "pcaCentroid">("orientedBoundingBox");
  const [objective, setObjective] = useState<"pointToPlane" | "pointToPoint" | "symmetricNormal">("pointToPlane");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [stages, setStages] = useState<AlignStage[] | null>(null);

  const canRun = !busy && moving !== null && fixed !== null && moving !== fixed;

  /** Coarse pre-alignment, then ICP (one undo step). */
  const run = async () => {
    if (!canRun || moving === null || fixed === null) return;
    setBusy(true);
    setError(null);
    try {
      const { coarse, fine } = await runMultistageAlign({
        store,
        moving,
        fixed,
        coarseMethod,
        objective,
        maxIterations: Math.max(1, Math.round(options.maxIterations)),
        overlap: Math.max(0.05, Math.min(1, options.overlap)),
        maxDistance: options.maxDistance > 0 ? options.maxDistance : undefined,
      });
      setStages(multistageStages(coarseMethod, coarse, fine));
    } catch (e) {
      setError(message(e));
    } finally {
      setBusy(false);
    }
  };

  return { coarseMethod, setCoarseMethod, objective, setObjective, busy, error, stages, canRun, run };
}

/** The symmetry plane: which mesh to detect it on, the detection, and the mirror across it. */
export function useSymmetryPlane(app: AppState, store: SceneStore, items: readonly Item[], selection: MeshHandle | null) {
  const [meshPick, setMeshPick] = useState<MeshHandle | null>(null);
  const [running, setRunning] = useState<"detect" | "mirror" | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [mirrored, setMirrored] = useState(false);
  const busy = running !== null;
  const shown = useApp(app, (s) => s.alignTool.symmetry);
  const mesh = resolveMesh(items, meshPick, selection);

  // The plane goes with its mesh.
  useEffect(() => {
    if (shown !== null && !items.some((i) => i.mesh === shown.mesh)) {
      app.setAlignTool({ symmetry: null });
    }
  }, [app, items, shown]);

  /** Detects the plane on the chosen mesh and shows it. */
  const detect = async () => {
    if (mesh === null || busy) return;
    setRunning("detect");
    setError(null);
    setMirrored(false);
    try {
      const r = await runDetectSymmetryPlane(store, mesh);
      const local = planeToLocal(store.scene.matrix4(mesh), r.planePoint, r.planeNormal);
      app.setAlignTool({
        symmetry: { mesh, point: local.point, normal: local.normal, errorRms: r.symmetryErrorRms, score: r.symmetryScore },
      });
    } catch (e) {
      setError(message(e));
    } finally {
      setRunning(null);
    }
  };

  /** Mirrors the plane's mesh across the plane where it is now (one undo step). */
  const mirror = async () => {
    const plane = app.getSnapshot().alignTool.symmetry;
    if (plane === null || busy) return;
    setRunning("mirror");
    setError(null);
    try {
      await mirrorMesh(store, plane.mesh, planeToWorld(store.scene.matrix4(plane.mesh), plane.point, plane.normal));
      setMirrored(true);
    } catch (e) {
      setError(message(e));
    } finally {
      setRunning(null);
    }
  };

  /** Hides the plane. */
  const clear = () => {
    app.setAlignTool({ symmetry: null });
    setMirrored(false);
    setError(null);
  };

  // The plane's world normal where its mesh is now (as of this render).
  let worldNormal: Vec3 | null = null;
  if (shown !== null) {
    try {
      worldNormal = planeToWorld(store.scene.matrix4(shown.mesh), shown.point, shown.normal).normal;
    } catch {
      // Removed meanwhile: the effect above hides the plane.
    }
  }

  return {
    mesh,
    shown,
    worldNormal,
    busy,
    running,
    error,
    mirrored,
    /** Picks the mesh to detect on by its index in `items`. */
    chooseMesh: (index: number) => setMeshPick(items[index]?.mesh ?? null),
    detect,
    mirror,
    clear,
  };
}
