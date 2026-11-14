// What the repair panel shows: an `Analysis` reduced to the counts it lists
// (holes largest first, components largest first), and one line per repair
// report.
import type {
  Analysis,
  AutoRepairPreset,
  AutoRepairReport,
  AutoRepairRequest,
  JoinReport,
  LoopId,
  ManifoldReport,
  OffsetReport,
  RepairOp,
  RepairReport,
  SelfIntersectReport,
  SelfIntersectionMode,
  SliverReport,
} from "fast-mesh-wasm";
import { type LengthUnit, toDisplayLength } from "./units";

export interface AnalysisSummary {
  /** Closed boundary loops, largest first, with the ids bridging takes. */
  holes: { id: LoopId; edges: number; perimeter: number }[];
  /** Boundary chains that do not close (never filled). */
  openChains: number;
  /** Interior edges used the same way by both their triangles. */
  orientationConflicts: number;
  /** Largest first. */
  components: { triangles: number; closed: boolean }[];
  nonManifoldEdges: number;
  nonManifoldVertices: number;
  pinchVertices: number;
  degenerate: number;
  duplicate: number;
  /** Nothing above to repair (a single closed, consistently oriented component). */
  clean: boolean;
}

export function summarize(a: Analysis): AnalysisSummary {
  const holes = a.loops
    .filter((l) => l.closed)
    .map((l) => ({ id: l.id, edges: l.id.len, perimeter: l.perimeter }))
    .sort((x, y) => y.edges - x.edges);
  const components = a.components
    .map((c) => ({ triangles: c.triangles.length, closed: c.closed }))
    .sort((x, y) => y.triangles - x.triangles);
  const s = {
    holes,
    openChains: a.loops.length - holes.length,
    orientationConflicts: a.orientationConflicts,
    components,
    nonManifoldEdges: a.nonManifoldEdges,
    nonManifoldVertices: a.nonManifoldVertices.length,
    pinchVertices: a.pinchVertices.length,
    degenerate: a.degenerate,
    duplicate: a.duplicate,
  };
  const clean =
    s.holes.length === 0 &&
    s.openChains === 0 &&
    s.orientationConflicts === 0 &&
    s.components.length <= 1 &&
    s.nonManifoldEdges === 0 &&
    s.nonManifoldVertices === 0 &&
    s.pinchVertices === 0 &&
    s.degenerate === 0 &&
    s.duplicate === 0;
  return { ...s, clean };
}

/** `1 triangle`, `2 triangles` (or the given plural). */
export function plural(n: number, word: string, words = `${word}s`): string {
  return `${n.toLocaleString("en")} ${n === 1 ? word : words}`;
}

/** Compact float for one-line summaries (`0.102`, `2`, `1.78`). */
function fmtNum(x: number): string {
  return `${Number(x.toPrecision(3))}`;
}

/**
 * One line for a repair report. Lengths come as bare numbers without a
 * `unit`, in it with one (the report holds mm).
 */
export function describeReport(op: RepairOp["op"], report: RepairReport, unit?: LengthUnit): string {
  const fmtLength = (x: number): string =>
    unit === undefined ? fmtNum(x) : `${fmtNum(toDisplayLength(x, unit))} ${unit}`;
  switch (op) {
    case "weld": {
      const r = report as Extract<RepairReport, { mergedVertices: number }>;
      return `Weld: merged ${plural(r.mergedVertices, "vertex", "vertices")}, removed ${plural(r.removedTriangles, "triangle")}`;
    }
    case "removeDegenerate": {
      const r = report as Extract<RepairReport, { removedVertices: number; removedTriangles: number }>;
      return `Remove degenerate: removed ${plural(r.removedTriangles, "triangle")}, ${plural(r.removedVertices, "vertex", "vertices")}`;
    }
    case "orient": {
      const r = report as Extract<RepairReport, { flippedTriangles: number }>;
      const outward =
        r.flippedOutwardComponents > 0 ? ` (${plural(r.flippedOutwardComponents, "component")} turned outward)` : "";
      const bad =
        r.nonOrientablePatches > 0 ? `; ${plural(r.nonOrientablePatches, "patch", "patches")} not orientable` : "";
      return `Orient: flipped ${plural(r.flippedTriangles, "triangle")} in ${plural(r.patches, "patch", "patches")}${outward}${bad}`;
    }
    case "removeSmallComponents": {
      const r = report as Extract<RepairReport, { removedComponents: number }>;
      return `Remove small components: removed ${plural(r.removedComponents, "component")} (${plural(r.removedTriangles, "triangle")})`;
    }
    case "fillHoles": {
      const r = report as Extract<RepairReport, { filled: unknown }>;
      const triangles = r.filled.reduce((n, f) => n + f.trianglesAdded, 0);
      const vertices = r.filled.reduce((n, f) => n + f.verticesAdded, 0);
      let line = `Fill holes: filled ${plural(r.filled.length, "hole")} (${plural(triangles, "triangle")}, ${plural(vertices, "vertex", "vertices")} added)`;
      if (r.skipped.length > 0) {
        const bounds = r.skipped.filter((s) => s.reason === "bounds").length;
        const unclean = r.skipped.length - bounds;
        const why = [
          bounds > 0 ? `${bounds} beyond the size limit` : "",
          unclean > 0 ? `${unclean} unclean or open` : "",
        ]
          .filter((s) => s !== "")
          .join(", ");
        line += `; skipped ${r.skipped.length} (${why})`;
      }
      if (r.failed.length > 0) {
        line += `; ${r.failed.length} could not be filled`;
      }
      return line;
    }
    case "inflate": {
      const r = report as Extract<RepairReport, { moved: number; maxDisplacement: number }>;
      return `Inflate: moved ${plural(r.moved, "vertex", "vertices")} (max ${fmtLength(r.maxDisplacement)})`;
    }
    case "smooth": {
      const r = report as Extract<RepairReport, { moved: number; maxDisplacement: number }>;
      return `Smooth: moved ${plural(r.moved, "vertex", "vertices")} (max ${fmtLength(r.maxDisplacement)})`;
    }
    case "remesh": {
      const r = report as Extract<RepairReport, { splits: number; collapses: number; flips: number; rounds: number }>;
      return `Remesh: ${plural(r.splits, "split")}, ${plural(r.collapses, "collapse")}, ${plural(r.flips, "flip")} in ${plural(r.rounds, "round")}`;
    }
    case "bridgeLoops": {
      const r = report as Extract<RepairReport, { trianglesAdded: number }>;
      return `Bridge loops: added ${plural(r.trianglesAdded, "triangle")}`;
    }
    case "grab": {
      const r = report as Extract<RepairReport, { moved: number; maxDisplacement: number }>;
      return `Grab: moved ${plural(r.moved, "vertex", "vertices")} (max ${fmtLength(r.maxDisplacement)})`;
    }
    case "flatten": {
      const r = report as Extract<RepairReport, { moved: number; maxDisplacement: number }>;
      return `Flatten: moved ${plural(r.moved, "vertex", "vertices")} (max ${fmtLength(r.maxDisplacement)})`;
    }
    case "crease": {
      const r = report as Extract<RepairReport, { moved: number; maxDisplacement: number }>;
      return `Crease: moved ${plural(r.moved, "vertex", "vertices")} (max ${fmtLength(r.maxDisplacement)})`;
    }
    case "clay": {
      const r = report as Extract<RepairReport, { moved: number; maxDisplacement: number }>;
      return `Clay: moved ${plural(r.moved, "vertex", "vertices")} (max ${fmtLength(r.maxDisplacement)})`;
    }
    case "twist": {
      const r = report as Extract<RepairReport, { moved: number; maxDisplacement: number }>;
      return `Twist: moved ${plural(r.moved, "vertex", "vertices")} (max ${fmtLength(r.maxDisplacement)})`;
    }
    case "captureStrokeState":
      return "Captured stroke state";
    case "relax": {
      const r = report as Extract<RepairReport, { moved: number; maxDisplacement: number }>;
      return `Relax: moved ${plural(r.moved, "vertex", "vertices")} (max ${fmtLength(r.maxDisplacement)})`;
    }
    case "offset": {
      const r = report as OffsetReport;
      if (r.trianglesAdded > 0 || r.trianglesRemoved > 0) {
        if (r.trianglesAdded === 0) {
          return "Offset: cleared mesh";
        }
        return `Offset: rebuilt ${plural(r.trianglesAdded, "triangle")}, ${plural(r.verticesAdded, "vertex", "vertices")} (voxel ${fmtLength(r.voxelSize)})`;
      }
      return `Offset: moved ${plural(r.moved, "vertex", "vertices")} (max ${fmtLength(r.maxDisplacement)})`;
    }
    case "shell": {
      const r = report as Extract<RepairReport, { verticesAdded: number; trianglesAdded: number }>;
      return `Shell: added ${plural(r.verticesAdded, "vertex", "vertices")} and ${plural(r.trianglesAdded, "triangle")}`;
    }
    case "simplify": {
      const r = report as Extract<RepairReport, { collapsed: number; trianglesRemoved: number }>;
      return `Simplify: ${plural(r.collapsed, "collapse")} (removed ${plural(r.trianglesRemoved, "triangle")})`;
    }
    case "subdivide": {
      const r = report as Extract<RepairReport, { verticesAdded: number; trianglesAdded: number }>;
      return `Subdivide: added ${plural(r.verticesAdded, "vertex", "vertices")} and ${plural(r.trianglesAdded, "triangle")}`;
    }
    case "splitStretched": {
      const r = report as Extract<RepairReport, { splits: number; remaining: number }>;
      return `Split stretched: ${plural(r.splits, "split")}`;
    }
    case "captureRestSizes":
      return "Captured rest sizes";
    case "makeManifold": {
      const r = report as ManifoldReport;
      return `Make manifold: removed ${plural(r.removedDuplicateTriangles, "duplicate")}, pruned ${plural(r.prunedFinTriangles, "fin")}, split ${plural(r.splitNonManifoldEdges, "non-manifold edge")} and ${plural(r.splitPinchVertices, "pinch vertex", "pinch vertices")}`;
    }
    case "resolveSelfIntersections": {
      const r = report as SelfIntersectReport;
      const dropped = r.uvsDropped ? " (UVs dropped)" : "";
      return `Self-intersections (${r.modeUsed}): ${r.initialPairs} → ${r.remainingPairs} pairs (-${plural(r.removedTriangles, "triangle")}, +${plural(r.addedTriangles, "triangle")})${dropped}`;
    }
    case "cleanupSlivers": {
      const r = report as SliverReport;
      return `Cleanup slivers: collapsed ${plural(r.collapsedNeedles, "needle")}, flipped ${plural(r.flippedCaps, "cap")}, removed ${plural(r.removedDegenerate, "degenerate triangle")}`;
    }
    default: {
      const _exhaustive: never = op;
      throw new Error(`unreachable repair op: ${String(_exhaustive)}`);
    }
  }
}

export const AUTO_REPAIR_PRESET_LABELS: Record<AutoRepairPreset, string> = {
  general: "General",
  scan3d: "3D Scan",
  print3d: "3D Print",
  cadImport: "CAD Import",
};

export const SELF_INTERSECTION_MODE_LABELS: Record<SelfIntersectionMode, string> = {
  none: "None",
  deleteAndFill: "Delete & Fill",
  cutAndStitch: "Cut & Stitch",
  voxelHeal: "Voxel Heal",
};

export interface PresetRepairSettings {
  preset: AutoRepairPreset;
  selfIntersectionMode?: SelfIntersectionMode;
  weldTolerance: number;
  minTriangles: number;
  /** Maximum hole edges (`0` means unbounded for fill presets, or keep boundary openings for `cadImport`). */
  maxEdges: number;
  fair: boolean;
  voxelSize?: number;
}

export function buildPresetAutoRepairRequest(settings: PresetRepairSettings): AutoRepairRequest {
  const req: AutoRepairRequest = {
    kind: "auto",
    preset: settings.preset,
    weldTolerance: settings.weldTolerance,
    minTriangles: settings.minTriangles,
    fair: settings.fair,
  };
  if (settings.preset === "cadImport") {
    if (settings.maxEdges > 0) {
      req.maxEdges = settings.maxEdges;
    }
  } else {
    req.maxEdges = settings.maxEdges > 0 ? settings.maxEdges : Number.POSITIVE_INFINITY;
  }
  if (settings.selfIntersectionMode !== undefined) {
    req.selfIntersectionMode = settings.selfIntersectionMode;
  }
  if (settings.voxelSize !== undefined) {
    req.voxelSize = settings.voxelSize;
  }
  return req;
}

function beforeAfterCount(n: readonly [number, number], word: string, words = `${word}s`): string {
  return `${plural(n[0], word, words)} → ${n[1] === 0 ? "none" : plural(n[1], word, words)}`;
}

export function formatPresetAutoRepairReport(
  preset: AutoRepairPreset,
  report: AutoRepairReport,
  before: AnalysisSummary,
  after: AnalysisSummary,
  beforeSi?: boolean | null,
  afterSi?: boolean | null,
  unit?: LengthUnit,
): { stageLines: string[]; summaryLine: string; line: string } {
  const stageLines: string[] = [];
  for (const [op, r] of Object.entries(report)) {
    if (r !== undefined) {
      stageLines.push(describeReport(op as RepairOp["op"], r as RepairReport, unit));
    }
  }
  const parts = [
    beforeAfterCount([before.holes.length, after.holes.length], "hole"),
    beforeAfterCount([before.nonManifoldEdges, after.nonManifoldEdges], "non-manifold edge"),
    beforeAfterCount([before.pinchVertices, after.pinchVertices], "pinch vertex", "pinch vertices"),
    beforeAfterCount([before.components.length, after.components.length], "component"),
  ];
  if (beforeSi !== null && beforeSi !== undefined && afterSi !== null && afterSi !== undefined) {
    parts.push(`self-intersections ${beforeSi ? "yes" : "none"} → ${afterSi ? "yes" : "none"}`);
  }
  const summaryLine = parts.join(", ");
  const label = AUTO_REPAIR_PRESET_LABELS[preset];
  const line = `Auto repair (${label}, one undo step): ${summaryLine}. ${stageLines.join("; ")}`;
  return { stageLines, summaryLine, line };
}

/** One line for a join report. */
export function describeJoin(report: JoinReport): string {
  return `Join: copied ${plural(report.verticesAdded, "vertex", "vertices")} and ${plural(report.trianglesAdded, "triangle")}, bridged ${plural(report.bridgeTrianglesAdded, "triangle")}`;
}
