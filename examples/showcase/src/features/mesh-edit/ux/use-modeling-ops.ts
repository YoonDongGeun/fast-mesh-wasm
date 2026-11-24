// The Modeling & Resampling panel's behaviour (#195): the Boolean CSG pair
// and operation (with Keep originals), the destructive plane cut, offset and
// shell, simplify and subdivide parameters; the active mesh's live triangle
// count and cut-axis range, kept current as it changes; and each op run
// through `engine/workbench-ops.ts` with a one-line report.
import type {
  CutReport,
  MeshHandle,
  OffsetReport,
  ShellReport,
  SimplifyReport,
  SubdivideReport,
} from "fast-mesh-wasm";
import type { SceneStore } from "fast-mesh-wasm/store";
import { useEffect, useState } from "react";
import type { AppState, Item } from "../../../app/state/app-state";
import {
  type BooleanOperation,
  type CutAxis,
  type CutKeepSide,
  extractLiveMeshData,
  type OffsetModeAlgorithm,
  runBooleanCsg,
  runOffset,
  runPlaneCut,
  runShell,
  runSimplify,
  runSubdivide,
} from "../../../engine/workbench-ops";
import { describeReport } from "../../mesh-repair/ux/repair-summary";
import { fromDisplayLength, type LengthUnit } from "../../../shared/ux/units";
import { computeLocalExtents } from "../../transform/ux/transform-math";

/** One op's report line (or its error). */
export interface ModelingReport {
  label: string;
  text: string;
  error?: boolean;
}

/** The Boolean operations and their button labels. */
export const BOOLEAN_OPERATIONS: readonly (readonly [BooleanOperation, string])[] = [
  ["union", "Union"],
  ["intersect", "Intersection"],
  ["subtract", "Difference"],
];

/** `current` while it is a mesh other than `primary`, else the first other mesh (or the first mesh). */
export function pickOtherMesh(
  items: readonly Item[],
  primary: MeshHandle | null,
  current: MeshHandle | null,
): MeshHandle | null {
  if (current !== null && items.some((i) => i.mesh === current) && current !== primary) {
    return current;
  }
  return items.find((i) => i.mesh !== primary)?.mesh ?? items[0]?.mesh ?? null;
}

/** The simplify target for a percentage of `triangles` (at least 1). */
export function simplifyTargetFor(triangles: number, percent: number): number {
  return Math.max(1, Math.round((triangles * percent) / 100));
}

/** The simplify percentage field: 1 to 100; an empty or invalid entry reads as 50. */
export function simplifyPercentInput(text: string): number {
  return Math.max(1, Math.min(100, Number(text) || 50));
}

/** The simplify triangle-count field: a whole number of at least 1. */
export function triangleCountInput(text: string): number {
  return Math.max(1, Math.round(Number(text) || 1));
}

/** The cut offset slider's range along `axis`: the mesh's extent padded by a quarter (at least 1) each side. */
export function cutRangeAlong(ext: ReturnType<typeof computeLocalExtents>, axis: CutAxis): [number, number] {
  const idx = axis === "x" ? 0 : axis === "y" ? 1 : 2;
  const pad = Math.max(ext.extents[idx] * 0.25, 1);
  return [ext.min[idx] - pad, ext.max[idx] + pad];
}

/** Whether the offset runs on a voxel grid (BVH SDF, FH EDT): a zero distance then remeshes. */
export function isVoxelOffset(algorithm: OffsetModeAlgorithm): boolean {
  return algorithm === "bvhSdf" || algorithm === "fhEdt";
}

/**
 * Why Offset cannot run with these fields: a zero distance on the normal
 * offset (nothing to do), or a typed voxel size that is not positive.
 */
export function offsetBlocked(algorithm: OffsetModeAlgorithm, distance: number, voxelSize: string): boolean {
  return (
    (!isVoxelOffset(algorithm) && distance === 0) ||
    (isVoxelOffset(algorithm) && voxelSize !== "" && !(Number(voxelSize) > 0))
  );
}

export function useModelingOps(args: {
  app: AppState;
  store: SceneStore;
  items: readonly Item[];
  /** The selected mesh, else the first one. */
  activeItem: Item | null;
  units: LengthUnit;
}) {
  const { app, store, items, activeItem, units } = args;

  // Boolean CSG state
  const [boolTarget, setBoolTarget] = useState<MeshHandle | null>(null);
  const [boolSource, setBoolSource] = useState<MeshHandle | null>(null);
  const [boolOp, setBoolOp] = useState<BooleanOperation>("union");
  const [keepOriginals, setKeepOriginals] = useState(true);

  // Plane Cut state
  const [cutAxis, setCutAxis] = useState<CutAxis>("y");
  const [cutCustomNormal, setCutCustomNormal] = useState<[number, number, number]>([0, 1, 0]);
  const [cutOffset, setCutOffset] = useState(0);
  const [cutCap, setCutCap] = useState(true);
  const [cutKeepSide, setCutKeepSide] = useState<CutKeepSide>("+");

  // Offset & Shell state
  const [offsetAlgorithm, setOffsetAlgorithm] = useState<OffsetModeAlgorithm>("auto");
  const [offsetDistance, setOffsetDistance] = useState(0.5);
  const [offsetVoxelSize, setOffsetVoxelSize] = useState("");
  const [shellThickness, setShellThickness] = useState(1);

  // Resampling state
  const [lockBoundaries, setLockBoundaries] = useState(true);
  const [simplifyRatio, setSimplifyRatio] = useState(50);
  const [simplifyTarget, setSimplifyTarget] = useState(500);
  const [simplifyUseRatio, setSimplifyUseRatio] = useState(true);
  const [subdivideLevels, setSubdivideLevels] = useState(1);

  // Live triangle count & axis bounds of active mesh
  const [liveTris, setLiveTris] = useState(0);
  const [cutBounds, setCutBounds] = useState<[number, number]>([-50, 50]);

  const [busy, setBusy] = useState<string | null>(null);
  const [report, setReport] = useState<ModelingReport | null>(null);

  const resolvedTarget =
    (boolTarget !== null && items.some((i) => i.mesh === boolTarget) ? boolTarget : null) ??
    activeItem?.mesh ??
    null;
  const resolvedSource = pickOtherMesh(items, resolvedTarget, boolSource);

  useEffect(() => {
    if (activeItem === null) {
      setLiveTris(0);
      setCutBounds([-50, 50]);
      return;
    }
    const updateStats = () => {
      try {
        const snap = extractLiveMeshData(store.scene, activeItem.mesh);
        const tris = Math.floor(snap.indices.length / 3);
        setLiveTris(tris);
        if (simplifyUseRatio) {
          setSimplifyTarget(simplifyTargetFor(tris, simplifyRatio));
        }
        setCutBounds(cutRangeAlong(computeLocalExtents(snap.positions, snap.indices), cutAxis));
      } catch {
        setLiveTris(activeItem.triangles);
      }
    };
    updateStats();
    const unsubStore = store.subscribe(updateStats);
    const unsubScene = store.scene.onChanges((changes) => {
      if (changes.some((c) => c.mesh === activeItem.mesh)) {
        updateStats();
      }
    });
    return () => {
      unsubStore();
      unsubScene();
    };
  }, [activeItem, store, cutAxis, simplifyUseRatio, simplifyRatio]);

  const runAction = async (label: string, fn: () => Promise<string>) => {
    setBusy(label);
    setReport(null);
    try {
      const text = await fn();
      setReport({ label, text });
    } catch (e) {
      setReport({ label, text: e instanceof Error ? e.message : String(e), error: true });
    } finally {
      setBusy(null);
    }
  };

  const targetItem = resolvedTarget === null ? null : (items.find((i) => i.mesh === resolvedTarget) ?? null);
  const sourceItem = resolvedSource === null ? null : (items.find((i) => i.mesh === resolvedSource) ?? null);
  /** The triangle count Simplify aims for: the ratio of the live count, or the typed count. */
  const simplifyCount = simplifyUseRatio ? simplifyTargetFor(liveTris, simplifyRatio) : simplifyTarget;

  /** Sets one component of the custom cut normal from its field (an invalid entry reads as 0). */
  const setCutNormalComponent = (idx: number, text: string) => {
    const next: [number, number, number] = [...cutCustomNormal];
    next[idx] = Number(text) || 0;
    setCutCustomNormal(next);
  };

  /** The offset voxel field as typed in display units, kept in mm (empty for automatic). */
  const setOffsetVoxelInput = (typed: string) => {
    setOffsetVoxelSize(typed === "" ? "" : String(fromDisplayLength(Number(typed), units)));
  };

  /** A new simplify percentage, which also retargets the count from the live triangles. */
  const setSimplifyPercent = (text: string) => {
    const pct = simplifyPercentInput(text);
    setSimplifyRatio(pct);
    if (liveTris > 0) {
      setSimplifyTarget(simplifyTargetFor(liveTris, pct));
    }
  };

  const runBoolean = () => {
    if (!targetItem || !sourceItem) return;
    void runAction("Boolean", async () => {
      const res = await runBooleanCsg({
        store,
        app,
        target: targetItem,
        source: sourceItem,
        op: boolOp,
        keepOriginals,
      });
      return `${res.name}: ${res.triangles.toLocaleString()} triangles`;
    });
  };

  const runCut = () => {
    if (!activeItem) return;
    void runAction("Plane cut", async () => {
      const r: CutReport = await runPlaneCut({
        store,
        mesh: activeItem.mesh,
        axis: cutAxis,
        customNormal: cutCustomNormal,
        offset: cutOffset,
        cap: cutCap,
        keepSide: cutKeepSide,
      });
      return `Kept ${r.keptTriangles.toLocaleString()} tris (removed ${r.removedTriangles.toLocaleString()}, cap ${r.capTriangles.toLocaleString()})`;
    });
  };

  const runOffsetOp = () => {
    if (!activeItem) return;
    void runAction("Offset", async () => {
      const r: OffsetReport = await runOffset({
        store,
        mesh: activeItem.mesh,
        distance: offsetDistance,
        algorithm: offsetAlgorithm,
        voxelSize: offsetVoxelSize === "" ? undefined : Number(offsetVoxelSize),
      });
      return describeReport("offset", r, units);
    });
  };

  const runShellOp = () => {
    if (!activeItem) return;
    void runAction("Shell", async () => {
      const r: ShellReport = await runShell({
        store,
        mesh: activeItem.mesh,
        thickness: shellThickness,
      });
      return describeReport("shell", r, units);
    });
  };

  const runSimplifyOp = () => {
    if (!activeItem) return;
    const targetCount = simplifyCount;
    void runAction("Simplify", async () => {
      const r: SimplifyReport = await runSimplify({
        store,
        mesh: activeItem.mesh,
        targetTriangles: targetCount,
        preserveBoundary: lockBoundaries,
      });
      return describeReport("simplify", r, units);
    });
  };

  const runSubdivideOp = () => {
    if (!activeItem) return;
    void runAction("Subdivide", async () => {
      const r: SubdivideReport = await runSubdivide({
        store,
        mesh: activeItem.mesh,
        levels: subdivideLevels,
      });
      return describeReport("subdivide", r, units);
    });
  };

  return {
    resolvedTarget,
    setBoolTarget,
    resolvedSource,
    setBoolSource,
    boolOp,
    setBoolOp,
    keepOriginals,
    setKeepOriginals,
    targetItem,
    sourceItem,
    cutAxis,
    setCutAxis,
    cutCustomNormal,
    setCutNormalComponent,
    cutOffset,
    setCutOffset,
    cutCap,
    setCutCap,
    cutKeepSide,
    setCutKeepSide,
    cutBounds,
    offsetAlgorithm,
    setOffsetAlgorithm,
    offsetDistance,
    setOffsetDistance,
    offsetVoxelSize,
    setOffsetVoxelInput,
    shellThickness,
    setShellThickness,
    lockBoundaries,
    setLockBoundaries,
    simplifyRatio,
    setSimplifyPercent,
    simplifyTarget,
    setSimplifyTarget,
    simplifyUseRatio,
    setSimplifyUseRatio,
    simplifyCount,
    subdivideLevels,
    setSubdivideLevels,
    liveTris,
    busy,
    report,
    runBoolean,
    runCut,
    runOffset: runOffsetOp,
    runShell: runShellOp,
    runSimplify: runSimplifyOp,
    runSubdivide: runSubdivideOp,
  };
}
