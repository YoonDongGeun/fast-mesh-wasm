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
    /** The 3D overlay toggles, and how many primitives of each kind it draws. */
    overlays: {
      kinds: { holes: boolean; nonManifold: boolean; degenerate: boolean; components: boolean };
      counts: { holes: number; nonManifold: number; degenerate: number; components: number };
    };
  };
  /** The volume view: the series shown, its preset, MIP, slice planes, and whether it rendered. */
  volume: { id: string | null; preset: string; mip: boolean; planes: number; ready: boolean };
  /** The slice views: the series shown, its window preset, the slice indices, and whether all rendered. */
  slices: { id: string | null; window: string; axial: number; coronal: number; sagittal: number; ready: boolean };
  /** The scene overlay: the series whose slices float in the main scene, and how many show. */
  sceneSlices: { id: string | null; shown: number };
  /** The scene's volume rendering: the series raymarched in the main scene, and its 3D preset. */
  sceneVolume: { id: string | null; preset: string };
  /** The inspect tools (#110): the armed tool, the ruler, the query, the section and the clip box. */
  tools: {
    tool: "ruler" | "query" | "section" | "clip" | null;
    /** The ruler's world endpoints and their distance (`null` until both clicked). */
    ruler: { a: [number, number, number]; b: [number, number, number] | null; distance: number | null };
    /** The query hit: mesh name, vertex slot, world point and coloring value. */
    query: { mesh: string; slot: number; point: [number, number, number]; value: number | null } | null;
    /** The section plane and how many meshes it cuts. */
    section: { plane: { origin: [number, number, number]; normal: [number, number, number] }; meshes: number } | null;
    /** The clip box corners. */
    clip: { min: [number, number, number]; max: [number, number, number] } | null;
  };
  /** The sculpt brush: strokes painted, dabs applied, stroke-end and live remeshes, grab drag applied, whether the ball shows, and the last grab dab's center. */
  sculpt: {
    strokes: number;
    dabs: number;
    remeshes: number;
    liveRemeshes: number;
    grabTotal: number;
    hovering: boolean;
    /** How many times a brush cursor was laid on the surface: it has settled once this stops. */
    cursorShows: number;
    lastDab: [number, number, number] | null;
    /** Dabs the mask dropped (every vertex in the ball was masked). */
    maskedDabs: number;
    /** Masks cleared against their will (a topology edit took a masked vertex). */
    maskStale: number;
    /** Stroke dabs: per send (bounded), the main-thread ms the mask took and the ms from sending to landing. */
    dabCost: { maskMs: number[]; landMs: number[] };
    /** The mask overlay: triangles drawn (0 when hidden), rebuilds, and the build time in ms, last and worst. */
    maskOverlay: { triangles: number; builds: number; lastMs: number; maxMs: number };
  };
} = {
  collision: { ...liveProbe(), warnings: [] },
  group: { ...liveProbe(), warnings: [] },
  hover: null,
  repair: {
    analysis: null,
    analyses: 0,
    report: null,
    busy: null,
    overlays: {
      kinds: { holes: true, nonManifold: true, degenerate: true, components: true },
      counts: { holes: 0, nonManifold: 0, degenerate: 0, components: 0 },
    },
  },
  volume: { id: null, preset: "", mip: false, planes: 0, ready: false },
  slices: { id: null, window: "", axial: 0, coronal: 0, sagittal: 0, ready: false },
  sceneSlices: { id: null, shown: 0 },
  sceneVolume: { id: null, preset: "" },
  sculpt: { strokes: 0, dabs: 0, remeshes: 0, liveRemeshes: 0, grabTotal: 0, hovering: false, cursorShows: 0, lastDab: null, maskedDabs: 0, maskStale: 0, dabCost: { maskMs: [], landMs: [] }, maskOverlay: { triangles: 0, builds: 0, lastMs: 0, maxMs: 0 } },
  tools: {
    tool: null,
    ruler: { a: [0, 0, 0], b: null, distance: null },
    query: null,
    section: null,
    clip: null,
  },
};

function liveProbe(): LiveProbe<never> {
  return { readouts: null, updates: 0, starts: 0, startsWhileDragging: 0, dropped: 0, updatesAfterDrag: 0, settled: true, hold: null };
}
