// What the repair panel shows: an `Analysis` reduced to the counts it lists
// (holes largest first, components largest first), and one line per repair
// report.
import type { Analysis, JoinReport, LoopId, RepairOp, RepairReport } from "fast-mesh-wasm";

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

/** One line for a repair report. */
export function describeReport(op: RepairOp["op"], report: RepairReport): string {
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
      return `Inflate: moved ${plural(r.moved, "vertex", "vertices")} (max ${fmtNum(r.maxDisplacement)})`;
    }
    case "smooth": {
      const r = report as Extract<RepairReport, { moved: number; maxDisplacement: number }>;
      return `Smooth: moved ${plural(r.moved, "vertex", "vertices")} (max ${fmtNum(r.maxDisplacement)})`;
    }
    case "remesh": {
      const r = report as Extract<RepairReport, { splits: number; collapses: number; flips: number; rounds: number }>;
      return `Remesh: ${plural(r.splits, "split")}, ${plural(r.collapses, "collapse")}, ${plural(r.flips, "flip")} in ${plural(r.rounds, "round")}`;
    }
    case "bridgeLoops": {
      const r = report as Extract<RepairReport, { trianglesAdded: number }>;
      return `Bridge loops: added ${plural(r.trianglesAdded, "triangle")}`;
    }
  }
}

/** One line for a join report. */
export function describeJoin(report: JoinReport): string {
  return `Join: copied ${plural(report.verticesAdded, "vertex", "vertices")} and ${plural(report.trianglesAdded, "triangle")}, bridged ${plural(report.bridgeTrianglesAdded, "triangle")}`;
}
