// What the repair and collision panels share outside React, for the test hook
// (scripts/showcase-check.mjs reads it through `window.__showcase`): the
// latest readouts and analysis, and how often they were refreshed.
import type { MeshHandle } from "fast-mesh-wasm";
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

/** One pair of a contact group, as last queried. */
export interface PairReadout {
  a: MeshHandle;
  b: MeshHandle;
  /** The surfaces' distance (`meshDistance`); 0 when they touch or intersect. */
  gap: number | null;
  intersects: boolean;
  /**
   * The deepest penetration between the two: `-min` over either member's
   * values at the vertices the other is nearest to (see `pairDepths`);
   * `null` before the fields have values.
   */
  depth: number | null;
  /** A bounded field (fast mode) may hide a deeper penetration (see `mayBeDeeper`). */
  atLeast: boolean;
}

/** The contact group's readouts as last queried. */
export interface GroupReadouts {
  /** The fields' bound (`maxDistance`) the depths were measured within; `Infinity` when exact. */
  bound: number;
  pairs: PairReadout[];
}

/**
 * A live readout: the latest value, how often it was refreshed, how its
 * queries met gizmo drags, and whether it is current.
 */
export interface LiveProbe<T> {
  readouts: T | null;
  updates: number;
  /** Readout queries started so far, and those started while the gizmo dragged (paused: none). */
  starts: number;
  startsWhileDragging: number;
  /** Results that landed while the gizmo dragged, dropped (the drag's end refreshes). */
  dropped: number;
  /** Refreshes since the last drag ended. */
  updatesAfterDrag: number;
  /** No change waits for a refresh and none runs: the readouts are those of the scene as it is. */
  settled: boolean;
  /** Set by the test hook: a query's result waits for it before it lands (to land one mid-drag). */
  hold: Promise<void> | null;
}

export const inspect: {
  collision: LiveProbe<Readouts> & {
    /**
     * Targets that are open (boundary loops, open chains or open components)
     * or inconsistently oriented (orientation conflicts), from `analyze` of
     * each mesh of the pair.
     */
    warnings: string[];
  };
  /** The collision panel's Group mode: its readouts and the members' topology warnings. */
  group: LiveProbe<GroupReadouts> & { warnings: string[] };
  /** The hover tip: the mesh under the pointer, the partner nearest to the vertex there, and its distance. */
  hover: { mesh: MeshHandle; partner: MeshHandle | null; distance: number | null } | null;
  repair: {
    /** The mesh analysed (its name) and the summary, `null` before an analysis. */
    analysis: { mesh: string; summary: AnalysisSummary } | null;
    analyses: number;
    /** The latest repair report line. */
    report: string | null;
    /** The op running, if any. */
    busy: string | null;
  };
  /** The sculpt brush: strokes painted, dabs applied, stroke-end remeshes, grab drag applied, and whether the ball shows. */
  sculpt: { strokes: number; dabs: number; remeshes: number; grabTotal: number; hovering: boolean };
} = {
  collision: { ...liveProbe(), warnings: [] },
  group: { ...liveProbe(), warnings: [] },
  hover: null,
  repair: { analysis: null, analyses: 0, report: null, busy: null },
  sculpt: { strokes: 0, dabs: 0, remeshes: 0, grabTotal: 0, hovering: false },
};

function liveProbe(): LiveProbe<never> {
  return { readouts: null, updates: 0, starts: 0, startsWhileDragging: 0, dropped: 0, updatesAfterDrag: 0, settled: true, hold: null };
}
