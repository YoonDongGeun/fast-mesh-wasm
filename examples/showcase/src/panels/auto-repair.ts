// The one-click auto-repair button: the library's `repair(mesh, { kind: "auto" })`
// with the panel's live settings, a per-stage report and a before/after summary
// (#125, second step of the lane after #124's overlays; #140 moved the pipeline
// into the library, which owns its order).
//
// Undo granularity: GROUPED single step. The auto request is a single
// transaction (one undo step, all-or-nothing), so Ctrl+Z restores the mesh as
// it was before the button, and a mid-pipeline failure leaves nothing
// half-repaired. The panel's per-op buttons stay for stage-by-stage work.
//
// The order and defaults are documented next to the button (no magic): the
// request carries the panel's live settings (weld tolerance, the component
// cutoff, the fill size limit and fairing) so the pipeline and the
// individual ops never disagree. Analysis and the #124 3D overlays refresh
// through the panel's existing triangle-change effect.
import type { AutoRepairReport, AutoRepairRequest, RepairOp, RepairReport } from "fast-mesh-wasm";
import { describeReport, plural, type AnalysisSummary } from "./repair-summary";

/** "Any size" for Fill holes (an empty field): a bound past any real loop. */
export const AUTO_REPAIR_ANY_SIZE = 1e9;

/** The pipeline's live settings: the panel's per-op fields, shared. */
export interface AutoRepairSettings {
  /** Vertices at most this far apart are welded (mesh units). */
  weldTolerance: number;
  /** Components with fewer triangles are removed. */
  minTriangles: number;
  /** Only holes with at most this many edges; 0 means any size. */
  maxEdges: number;
  /** Refine and fair each fill so it follows the surrounding curvature. */
  fair: boolean;
}

/** The request for `repair(mesh, request)`: one transaction (one undo step), all-or-nothing. */
export function buildAutoRepairRequest(settings: AutoRepairSettings): AutoRepairRequest {
  return {
    kind: "auto",
    weldTolerance: settings.weldTolerance,
    minTriangles: settings.minTriangles,
    maxEdges: settings.maxEdges > 0 ? settings.maxEdges : Number.POSITIVE_INFINITY,
    fair: settings.fair,
  };
}

/** One line for a pipeline stage's report (the per-op wording, reused). */
export function describeAutoRepairStage(op: RepairOp["op"], report: RepairReport): string {
  return describeReport(op, report);
}

/** `2 holes → none`: a count pair for the before/after summary. */
function beforeAfter(n: readonly [before: number, after: number], word: string, words = `${word}s`): string {
  return `${plural(n[0], word, words)} → ${n[1] === 0 ? "none" : plural(n[1], word, words)}`;
}

/**
 * The before/after summary from two analyses (holes, components,
 * degenerate triangles, orientation conflicts).
 */
export function describeAutoRepairSummary(before: AnalysisSummary, after: AnalysisSummary): string {
  const degenerate: [number, number] = [before.degenerate + before.duplicate, after.degenerate + after.duplicate];
  return [
    beforeAfter([before.holes.length, after.holes.length], "hole"),
    beforeAfter([before.components.length, after.components.length], "component"),
    beforeAfter(degenerate, "degenerate triangle"),
    beforeAfter([before.orientationConflicts, after.orientationConflicts], "orientation conflict"),
  ].join(", ");
}

/** The whole button report: the summary first, then one line per stage. */
export function formatAutoRepairReport(stageLines: string[], summaryLine: string): string {
  return `Auto repair (one undo step): ${summaryLine}. ${stageLines.join("; ")}`;
}

export interface AutoRepairDeps {
  /**
   * One repair call with the auto request (one transaction, one undo
   * step); the panel passes `store.repair`. The runner calls it exactly
   * once per pipeline run: that single call IS the grouped undo.
   */
  repair: (request: AutoRepairRequest) => Promise<AutoRepairReport>;
  /** A fresh analysis summary; before the run and after (the overlays' refresh reads the same scene). */
  analyze: () => Promise<AnalysisSummary>;
}

export interface AutoRepairResult {
  request: AutoRepairRequest;
  report: AutoRepairReport;
  stageLines: string[];
  before: AnalysisSummary;
  after: AnalysisSummary;
  summaryLine: string;
  line: string;
}

/** Runs the pipeline: analyse, one repair call, analyse again, report. */
export async function runAutoRepairPipeline(
  deps: AutoRepairDeps,
  settings: AutoRepairSettings,
): Promise<AutoRepairResult> {
  const before = await deps.analyze();
  const request = buildAutoRepairRequest(settings);
  const report = await deps.repair(request);
  const after = await deps.analyze();
  // The report's keys come in the order the library ran the stages.
  const stageLines = Object.entries(report).map(([op, r]) => describeAutoRepairStage(op as RepairOp["op"], r));
  const summaryLine = describeAutoRepairSummary(before, after);
  return { request, report, stageLines, before, after, summaryLine, line: formatAutoRepairReport(stageLines, summaryLine) };
}
