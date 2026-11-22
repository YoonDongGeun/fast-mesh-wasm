// What the Mesh Healer section does (#235): the selective heal categories
// and the repair ops they build ([data-repair-op="selectiveHeal"]), the
// before/after delta readout ([data-healer-delta]), the defect hotspots and
// extra counts read from the mesh's buffers, and click-to-focus on one
// hotspot.
import type { Analysis, MeshHandle, RepairOp, SelfIntersectionMode } from "fast-mesh-wasm";
import type { WorkerScene } from "fast-mesh-wasm/worker";
import type { AppState } from "../../../app/state/app-state";
import {
  countHealerOverlays,
  type DefectHotspot,
  extractDefectHotspots,
  type HealerOverlayCounts,
  isHealerOverlayKind,
} from "./healer-defects";
import type { AnalysisSummary } from "./repair-summary";

export type SelectiveHealCategory =
  | "weld"
  | "removeDegenerate"
  | "orient"
  | "makeManifold"
  | "removeSmallComponents"
  | "fillHoles"
  | "cleanupSlivers"
  | "resolveSelfIntersections";

export const SELECTIVE_HEAL_CATEGORIES: readonly { key: SelectiveHealCategory; label: string }[] = [
  { key: "weld", label: "Weld coincident vertices" },
  { key: "removeDegenerate", label: "Degenerate & duplicate tris" },
  { key: "orient", label: "Orient normals & shells" },
  { key: "makeManifold", label: "Make manifold (edges/pinch)" },
  { key: "removeSmallComponents", label: "Prune small islands" },
  { key: "fillHoles", label: "Fill holes" },
  { key: "cleanupSlivers", label: "Clean sliver faces" },
  { key: "resolveSelfIntersections", label: "Resolve self-intersections" },
];

/** The categories checked to start with: every stage but self-intersection healing. */
export const DEFAULT_SELECTIVE_HEAL: Readonly<Record<SelectiveHealCategory, boolean>> = {
  weld: true,
  removeDegenerate: true,
  orient: true,
  makeManifold: true,
  removeSmallComponents: true,
  fillHoles: true,
  cleanupSlivers: true,
  resolveSelfIntersections: false,
};

export interface SelectiveHealSettings {
  weldTolerance: number;
  minTriangles: number;
  maxEdges: number;
  fair: boolean;
  anySize: number;
  selfIntersectionMode: SelfIntersectionMode;
  siVoxelSize: string;
}

export function buildSelectiveHealOps(
  enabled: Record<SelectiveHealCategory, boolean>,
  settings: SelectiveHealSettings,
): RepairOp[] {
  const ops: RepairOp[] = [];
  if (enabled.weld) {
    ops.push({ op: "weld", tolerance: settings.weldTolerance });
  }
  if (enabled.removeDegenerate) {
    ops.push({ op: "removeDegenerate" });
  }
  if (enabled.orient) {
    ops.push({ op: "orient" });
  }
  if (enabled.makeManifold) {
    ops.push({ op: "makeManifold" });
  }
  if (enabled.removeSmallComponents) {
    ops.push({ op: "removeSmallComponents", minTriangles: settings.minTriangles });
  }
  if (enabled.fillHoles) {
    ops.push({
      op: "fillHoles",
      maxEdges: settings.maxEdges > 0 ? settings.maxEdges : settings.anySize,
      fair: settings.fair,
    });
  }
  if (enabled.cleanupSlivers) {
    ops.push({ op: "cleanupSlivers", maxAspectRatio: 20 });
  }
  if (enabled.resolveSelfIntersections) {
    const mode = settings.selfIntersectionMode === "none" ? "deleteAndFill" : settings.selfIntersectionMode;
    ops.push({
      op: "resolveSelfIntersections",
      mode,
      fair: settings.fair,
      ...(settings.siVoxelSize !== "" && Number(settings.siVoxelSize) > 0
        ? { voxelSize: Number(settings.siVoxelSize) }
        : {}),
    });
  }
  return ops;
}

export function formatHealerDelta(
  before: AnalysisSummary,
  after: AnalysisSummary,
  beforeExtra?: Partial<HealerOverlayCounts> | null,
  afterExtra?: Partial<HealerOverlayCounts> | null,
): string {
  const parts: string[] = [
    `Holes: ${before.holes.length} → ${after.holes.length}`,
    `Non-manifold: ${before.nonManifoldEdges} → ${after.nonManifoldEdges}`,
    `Orientation: ${before.orientationConflicts} → ${after.orientationConflicts}`,
    `Degenerate: ${before.degenerate + before.duplicate} → ${after.degenerate + after.duplicate}`,
    `Components: ${before.components.length} → ${after.components.length}`,
  ];
  if (beforeExtra?.slivers !== undefined && afterExtra?.slivers !== undefined) {
    parts.push(`Slivers: ${beforeExtra.slivers} → ${afterExtra.slivers}`);
  }
  if (beforeExtra?.selfIntersections !== undefined && afterExtra?.selfIntersections !== undefined) {
    parts.push(`Self-intersect tris: ${beforeExtra.selfIntersections} → ${afterExtra.selfIntersections}`);
  }
  return parts.join(" · ");
}

/** The mesh buffers the healer reads: a sync `Scene`'s views, or the Worker scene's mirrored ones. */
export type HealerMeshViews =
  | { views: (mesh: MeshHandle) => { positions(): Float32Array; indices(): Uint32Array } }
  | WorkerScene;

/**
 * The defect hotspots and the healer's extra counts for `mesh` under
 * `rawAnalysis`: none without an analysis, or when the mesh's buffers are
 * gone meanwhile.
 */
export function readHealerDefects(
  rawAnalysis: Analysis | null,
  scene: HealerMeshViews,
  mesh: MeshHandle,
): { hotspots: DefectHotspot[]; extraCounts: HealerOverlayCounts } {
  if (rawAnalysis === null) {
    return {
      hotspots: [] as DefectHotspot[],
      extraCounts: { selfIntersections: 0, inverted: 0, slivers: 0 } satisfies HealerOverlayCounts,
    };
  }
  try {
    const views = scene.views(mesh);
    const pos = views.positions();
    const idx = views.indices();
    return {
      hotspots: extractDefectHotspots(rawAnalysis, pos, idx),
      extraCounts: countHealerOverlays(rawAnalysis, pos, idx),
    };
  } catch {
    return {
      hotspots: [] as DefectHotspot[],
      extraCounts: { selfIntersections: 0, inverted: 0, slivers: 0 } satisfies HealerOverlayCounts,
    };
  }
}

/** The healer's extra counts for one analysis of `mesh`, or `null` when its buffers are unreadable. */
export function healerCountsOf(scene: HealerMeshViews, mesh: MeshHandle, raw: Analysis): HealerOverlayCounts | null {
  try {
    const views = scene.views(mesh);
    return countHealerOverlays(raw, views.positions(), views.indices());
  } catch {
    return null;
  }
}

/** Focuses the camera on `spot`, turning on the overlay that draws its kind. */
export function focusDefectHotspot(app: AppState, mesh: MeshHandle, spot: DefectHotspot): void {
  if (isHealerOverlayKind(spot.kind)) {
    app.setHealerOverlay(spot.kind, true);
  } else {
    app.setRepairOverlay(spot.kind, true);
  }
  app.focusDefect(mesh, spot.center, spot.radius);
}
