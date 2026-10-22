// What the repair and collision panels share outside React, for the test hook
// (scripts/showcase-check.mjs reads it through `window.__showcase`): the
// latest readouts and analysis, and how often they were refreshed.
import type { AnalysisSummary } from "./repair-summary";

/** The collision readouts of the pair as last queried. */
export interface Readouts {
  /**
   * The deepest penetration (`-min` of both fields' values), `null` before a
   * field has values; `atLeast` when a bounded field (fast mode) may hide a
   * deeper one (see `deepestPenetration`).
   */
  penetration: { depth: number; atLeast: boolean } | null;
  /** The fields' bound (`maxDistance`) the depth was measured within; `Infinity` when exact. */
  bound: number;
  /** The surfaces' distance (`meshDistance`); 0 when they touch or intersect. */
  gap: number | null;
  intersects: boolean;
  /** Triangles within the tolerance of the other mesh (`findContacts({ margin: tolerance })`). */
  contacts: { a: number; b: number };
}

export const inspect: {
  collision: {
    readouts: Readouts | null;
    /** Readout refreshes so far, and those that landed while the gizmo dragged. */
    updates: number;
    updatesWhileDragging: number;
    /** No change waits for a refresh and none runs: the readouts are those of the scene as it is. */
    settled: boolean;
    /**
     * Targets that are open (boundary loops, open chains or open components)
     * or inconsistently oriented (orientation conflicts), from `analyze` of
     * each mesh of the pair.
     */
    warnings: string[];
  };
  repair: {
    /** The mesh analysed (its name) and the summary, `null` before an analysis. */
    analysis: { mesh: string; summary: AnalysisSummary } | null;
    analyses: number;
    /** The latest repair report line. */
    report: string | null;
    /** The op running, if any. */
    busy: string | null;
  };
} = {
  collision: { readouts: null, updates: 0, updatesWhileDragging: 0, settled: true, warnings: [] },
  repair: { analysis: null, analyses: 0, report: null, busy: null },
};
