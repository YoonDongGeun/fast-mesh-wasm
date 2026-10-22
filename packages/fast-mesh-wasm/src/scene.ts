// The public `Scene` class: editable meshes with placements, zero-copy render
// views, per-frame change records and undo history. Wraps the raw `WasmScene`
// binding; every call into WebAssembly goes through `call` (error codes and
// panic poisoning).
import type { Indices, Mat3, Placement, PlacementInput, Positions64, Quat, UpdateMode, Vec3 } from "./api.js";
import { decodeChanges, type Range } from "./changes.js";
import { call, ensureInitialized, FastMeshError, finalizer, release, wasmMemory } from "./core.js";
import { FieldHandle, MeshHandle, rotationMatrix, SectionHandle } from "./handles.js";
import type { SceneLike } from "./scene-like.js";
import { type MeshViews, meshViews, ViewGuard } from "./views.js";
import { WasmScene } from "./wasm/fast_mesh_wasm.js";

export type { Range } from "./changes.js";
export type { MeshViews } from "./views.js";
// Wasm-free, so the Worker client can use them without the wasm glue.
export { FieldHandle, MeshHandle, rotationMatrix, SectionHandle } from "./handles.js";

/** The id behind a handle and the scene it belongs to. Kept here, not on the handle. */
interface HandleId {
  scene: Scene;
  index: number;
  generation: number;
}

const meshIds = new WeakMap<MeshHandle, HandleId>();
const rawIds = new WeakMap<FieldHandle | SectionHandle, HandleId>();

/**
 * What changed in one mesh since the previous {@link Scene.takeChanges}.
 * Range lists are ascending; `"all"` means re-upload the whole buffer.
 */
export interface MeshChange {
  mesh: MeshHandle;
  /** The mesh was added; every buffer is `"all"`. */
  added: boolean;
  /** The mesh was removed; drop its GPU buffers. Nothing else is set. */
  removed: boolean;
  /** Its placement changed: read {@link Scene.matrix4} again. */
  placementChanged: boolean;
  /**
   * Its vertex or triangle slot count changed: recreate its GPU buffers. Every
   * buffer is then `"all"`, so upload each one whole.
   */
  capacityChanged: boolean;
  /** Vertex slots. */
  positions: Range[] | "all";
  /** Vertex slots. */
  normals: Range[] | "all";
  /** Vertex slots. */
  colors: Range[] | "all";
  /** Triangle slots. */
  indices: Range[] | "all";
  fieldsChanged: FieldHandle[];
  sectionsChanged: SectionHandle[];
}

/**
 * A drag of one mesh: every {@link DragSession.move} until
 * {@link DragSession.end} is one undo step.
 *
 * The drag is over once it is ended or cancelled, and also when its step goes
 * away without it: an `undo()`, `redo()`, `clearHistory()` or `removeMesh()` of
 * the dragged mesh ends it first, and a `cancelStep()` (or an `endStep()` that closes the drag's step) abandons it.
 */
export interface DragSession {
  /**
   * Sets the mesh's placement. Throws `INVALID_ARGUMENT` once the drag is over.
   *
   * Each move recomputes the eager fields that depend on the mesh, so
   * coalesce pointer events to one move per animation frame (keep the latest
   * pose and apply it before {@link Scene.commitFrame}).
   */
  move(placement: PlacementInput): void;
  /**
   * False once the drag is over: ended, cancelled, or ended or abandoned by
   * the scene (`undo()`, `redo()`, `clearHistory()`, `endStep()`,
   * `cancelStep()`, `removeMesh()` of the dragged mesh).
   */
  readonly active: boolean;
  /**
   * Finishes the drag, recording it as one undo step (closing any step levels
   * opened inside it too). A no-op once it is over.
   */
  end(): void;
  /**
   * Puts the placement back to where the drag started and records nothing. A
   * no-op once it is over.
   *
   * When no step was open at `drag()`, this reverts the drag's whole step, so
   * edits made during the drag (including in steps opened inside it) are
   * reverted too. Inside a caller's step, it restores the placement and closes
   * the drag's levels, leaving the caller's step open.
   */
  cancel(): void;
}

/** Options of {@link Scene.addMesh}. */
export interface AddMeshOptions {
  /**
   * The point subtracted from every position before it is stored as `f32`.
   * Defaults to the bounding-box centre, which keeps local coordinates small.
   */
  origin?: Readonly<Vec3>;
  /** Defaults to `"eager"`. */
  mode?: UpdateMode;
  /** Keep vertex normals up to date. Defaults to `true`. */
  normals?: boolean;
}

/** Options of the {@link Scene} constructor. */
export interface SceneOptions {
  /**
   * Development checks, off by default. Render views ({@link Scene.views},
   * {@link Scene.fieldValues}) are then `Proxy` objects that warn once per
   * scene (`console.warn`, no throw) when read after a scene call that may
   * have moved or freed their buffer. Read-only calls (slot counts,
   * `placement`, `matrix4`, `canUndo`, ...) keep views valid. Proxies are
   * slower and are not genuine typed arrays: pass `view.subarray()` (a plain
   * typed array) to APIs that need one, such as `queue.writeBuffer`. Without
   * `debug`, views are plain typed arrays with no overhead.
   */
  debug?: boolean;
}

/** A ray hit from {@link Scene.raycast}, in world coordinates. */
export interface RayHit {
  mesh: MeshHandle;
  /** The triangle slot hit. */
  triangle: number;
  point: Vec3;
  /** The world distance from the ray origin. */
  distance: number;
  /** Barycentric coordinates of `point` in the triangle. */
  barycentric: Vec3;
}

/** Options of {@link Scene.raycast}, in world units. */
export interface RaycastOptions {
  /** Smallest distance that counts as a hit (inclusive). Defaults to 0. */
  minDistance?: number;
  /** Largest distance that counts as a hit (inclusive). Defaults to `Infinity`. */
  maxDistance?: number;
  /** Faces to ignore, by the triangle winding. Defaults to `"none"`. */
  cull?: "none" | "back" | "front";
}

/** The nearest surface point from {@link Scene.closestPoint}, in world coordinates. */
export interface ClosestHit {
  mesh: MeshHandle;
  /** The triangle slot the point lies on. */
  triangle: number;
  point: Vec3;
  distance: number;
}

/** Options of {@link Scene.closestPoint}, in world units. */
export interface ClosestPointOptions {
  /** Largest distance that counts as found (inclusive). Defaults to `Infinity`. */
  maxDistance?: number;
  /** Search this mesh only. Defaults to every mesh. */
  mesh?: MeshHandle;
}

/** The closest pair between two meshes from {@link Scene.meshDistance}, in world coordinates. */
export interface MeshDistance {
  /** 0 when the surfaces touch or intersect. */
  distance: number;
  /** The point on the first mesh. */
  a: Vec3;
  /** The point on the second mesh. */
  b: Vec3;
  /** The first mesh's triangle slot that `a` lies on. */
  triangleA: number;
  /** The second mesh's triangle slot that `b` lies on. */
  triangleB: number;
}

/** Two meshes within the margin of {@link Scene.findContacts}. */
export interface Contact {
  /** The mesh with the lower id. */
  a: MeshHandle;
  b: MeshHandle;
  /** Whether the meshes intersect or touch, decided exactly. */
  intersecting: boolean;
  /** The distance between the surfaces; 0 when `intersecting`. */
  distance: number;
  /** Triangle slots of `a` within the margin of `b`, ascending. */
  trianglesA: Uint32Array;
  /** Triangle slots of `b` within the margin of `a`, ascending. */
  trianglesB: Uint32Array;
}

/** Options of {@link Scene.findContacts}. */
export interface FindContactsOptions {
  /** Report pairs whose surfaces are at most this far apart (world units). Defaults to 0. */
  margin?: number;
  /** Only these pairs, either way round. Defaults to every pair of meshes. */
  pairs?: readonly (readonly [MeshHandle, MeshHandle])[];
}

/** How a distance field decides which side of the target surface is inside. */
export type SignMode = "auto" | "pseudonormal" | "winding";

/** A signed distance field to add with {@link Scene.addDistanceField}. */
export interface DistanceFieldDesc {
  /** Unique among the fields on `on`. */
  name: string;
  /** The mesh whose vertices carry the values. */
  on: MeshHandle;
  /** The mesh the distances are measured to. */
  to: MeshHandle;
  /** Largest distance computed (inclusive); farther vertices hold `Infinity`. May be `Infinity`. */
  maxDistance: number;
  /** Defaults to `"auto"`. */
  sign?: SignMode;
}

/** The state of a field, from {@link Scene.fieldStatus}. */
export interface FieldStatus {
  /** True when the values match the meshes and placements as they are now. */
  current: boolean;
  /** What the latest sign data build found out about the target mesh. */
  sign: {
    /** The mode in use (never `"auto"`). */
    mode: Exclude<SignMode, "auto">;
    /** No boundary or non-manifold edge. */
    closed: boolean;
    consistentOrientation: boolean;
    nonManifoldVertices: number;
    degenerateTriangles: number;
  };
}

/** An RGBA8 colour, each channel an integer in 0..255. */
export type Rgba = readonly [r: number, g: number, b: number, a: number];

/**
 * A colormap: a built-in one, or colour stops `[t, rgba]` (at least two, `t`
 * in [0, 1] strictly increasing) interpolated linearly.
 */
export type Colormap =
  | "turbo"
  | "viridis"
  | "coolwarm"
  | "grayscale"
  | { stops: readonly (readonly [t: number, color: Rgba])[] };

/** What to show for a value outside the colour range. */
export type OutOfRange = "clamp" | "base";

/** How {@link Scene.setColoring} turns a field into vertex colours. */
export interface ColorRule {
  /** A field on the coloured mesh. */
  field: FieldHandle;
  map: Colormap;
  /** `[lo, hi]` mapped onto the colormap; finite, `lo < hi`. */
  range: readonly [lo: number, hi: number];
  /** Values below `lo`: the colormap's end colour or `base`. Defaults to `"clamp"`. */
  below?: OutOfRange;
  /** Values above `hi`, including far (`Infinity`): defaults to `"base"`. */
  above?: OutOfRange;
  /** A value within `range` pinned to the middle of the colormap. */
  center?: number;
  /** The `"base"` colour. Defaults to light grey `[200, 200, 200, 255]`. */
  base?: Rgba;
  /** The colour of NaN (dead or unused vertices). Defaults to `[128, 128, 128, 255]`. */
  nan?: Rgba;
}

/** A section plane in world coordinates. */
export interface SectionPlane {
  origin: Readonly<Vec3>;
  /** Points to the plane's `+` side; normalised. */
  normal: Readonly<Vec3>;
}

/** A plane section to add with {@link Scene.addSection}. */
export interface SectionDesc {
  plane: SectionPlane;
  /** The meshes cut, each once. */
  meshes: readonly MeshHandle[];
  /** Whether results carry the filled cross-section. Defaults to `false`. */
  fill?: boolean;
}

/** One chain of a section: a closed loop or an open polyline. */
export interface SectionContour {
  /** World xyz, 3 floats per point. A copy. */
  points: Float32Array;
  closed: boolean;
}

/** A section's result for one mesh, from {@link Scene.section}. Copies, in world coordinates. */
export interface SectionResult {
  /** `"stale"` when not computed yet or something changed since: call {@link Scene.updateSections}. */
  status: "current" | "stale";
  contours: SectionContour[];
  /** Summed length of every contour. */
  length: number;
  /** Summed signed area of the closed, oriented contours (positive counter-clockwise about the normal). */
  area: number;
  /** The filled cross-section, when the section has fill on and it was computed. */
  fill?: { positions: Float32Array; indices: Uint32Array };
}

/** The closest pair between two meshes' contours, from {@link Scene.sectionGap}. */
export interface SectionGap {
  /** Measured in the plane; 0 when the contours cross or touch. */
  distance: number;
  /** World point on the first mesh's contours. */
  a: Vec3;
  /** World point on the second mesh's contours. */
  b: Vec3;
}

/**
 * An id produced by an earlier op of the same {@link Scene.edit} list: id `i`
 * of the result of op `opIndex` (a vertex of `addVertices`, a triangle of
 * `addTriangles`).
 */
export interface EditRef {
  ref: readonly [opIndex: number, i: number];
}

/**
 * One op of {@link Scene.edit}. Positions are in world space (3 per vertex);
 * ids are vertex or triangle slots, or {@link EditRef}s to ids added earlier
 * in the same list.
 */
export type EditOp =
  | { op: "setPositions"; vertices: Uint32Array | readonly number[]; positions: Positions64 }
  | { op: "addVertices"; positions: Positions64 }
  | { op: "addTriangles"; triangles: readonly (number | EditRef)[] | Uint32Array }
  | { op: "removeTriangles"; triangles: Uint32Array | readonly number[] }
  | { op: "removeVertices"; vertices: Uint32Array | readonly number[] | readonly EditRef[] };

/** What {@link Scene.edit} added. */
export interface EditResult {
  /** Per op, in order: the new vertex or triangle ids (empty for ops that add nothing). */
  added: Uint32Array[];
}

/**
 * A boundary loop's content fingerprint, valid while its vertex sequence is
 * unchanged: pass it back to `fillHoles` to pick that loop.
 */
export interface LoopId {
  /** The smallest vertex id on the loop. */
  minVertex: number;
  /** The number of vertices on the loop. */
  len: number;
  /** FNV-1a 64 of the loop's vertex ids. */
  hash: bigint;
}

/** A chain of boundary edges, from {@link Scene.analyze}. */
export interface BoundaryLoop {
  id: LoopId;
  /** In loop order; a closed loop starts at `id.minVertex`. */
  vertices: Uint32Array;
  /** Length of the closed polygon. */
  perimeter: number;
  /** Vector area of the closed polygon. */
  area: number;
  /** `false` for an open chain (never filled). */
  closed: boolean;
  /** `false` when the loop touches a non-manifold edge, an unordered pinch or an orientation seam. */
  clean: boolean;
}

/** Triangles connected through shared edges, from {@link Scene.analyze}. */
export interface MeshComponent {
  /** Ascending. */
  triangles: Uint32Array;
  area: number;
  /** No boundary and no non-manifold edge. */
  closed: boolean;
}

/**
 * The defects of a mesh, from {@link Scene.analyze}. Lists with a count beside
 * them keep their first `maxListed` entries; counts are exact.
 */
export interface Analysis {
  loops: BoundaryLoop[];
  /** Groups of two or more coincident vertices. */
  coincidentGroups: number;
  /** The groups, by smallest id, each ascending. */
  coincident: Uint32Array[];
  /** Degenerate triangles (repeated vertex or near-zero area). */
  degenerate: number;
  degenerateList: Uint32Array;
  /** Triangles repeating the vertex set of a triangle with a smaller id. */
  duplicate: number;
  duplicateList: Uint32Array;
  /** Edges with three or more triangles. */
  nonManifoldEdges: number;
  /** Those edges as `[min, max]`, ascending. */
  nonManifoldEdgeList: [number, number][];
  nonManifoldVertices: Uint32Array;
  /** Vertices whose triangle fan splits into several pieces. */
  pinchVertices: Uint32Array;
  components: MeshComponent[];
  /** Interior edges used in the same direction by both their triangles. */
  orientationConflicts: number;
  /** Whether two triangles sharing no vertex intersect; `null` when not checked. */
  selfIntersects: boolean | null;
}

/**
 * Options of {@link Scene.analyze}. Distances are in mesh units: placements are
 * rigid, so local and world distances agree.
 */
export interface AnalyzeOptions {
  /**
   * Vertices at most this far apart count as coincident. Defaults to 0 (exact
   * duplicates). Converted to `f32`: a value below the `f32` range rounds to 0.
   */
  weldTolerance?: number;
  /**
   * Check for self-intersection with the mesh's BVH. A stale `"manual"` mesh
   * throws `STALE_BVH`; others are brought up to date first. Defaults to `false`.
   */
  selfIntersection?: boolean;
  /** The most entries kept in each counted list. Defaults to 1000. */
  maxListed?: number;
}

/**
 * One repair operation for {@link Scene.repair}. Distances and areas are in
 * mesh units (equal to world units). They are converted to `f32`, so a
 * tolerance or area below the `f32` range rounds to 0.
 */
export type RepairOp =
  | { op: "weld"; tolerance: number }
  | { op: "removeDegenerate"; areaEpsilon?: number; removeUnusedVertices?: boolean }
  | { op: "removeSmallComponents"; minTriangles?: number; minArea?: number; keepLargest?: boolean }
  | { op: "orient" }
  | {
      op: "fillHoles";
      maxEdges?: number;
      maxPerimeter?: number;
      maxArea?: number;
      loops?: readonly LoopId[];
      allowUnclean?: boolean;
      fair?: boolean;
      dpLimit?: number;
    };

/** What `weld` did. */
export interface WeldReport {
  mergedVertices: number;
  removedTriangles: number;
}

/** What `removeDegenerate` did. */
export interface DegenerateReport {
  removedTriangles: number;
  removedVertices: number;
}

/** What `removeSmallComponents` did. */
export interface RemoveReport {
  removedComponents: number;
  removedTriangles: number;
  removedVertices: number;
}

/** What `orient` did. */
export interface OrientReport {
  flippedTriangles: number;
  patches: number;
  nonOrientablePatches: number;
  flippedOutwardComponents: number;
}

/** A loop `fillHoles` filled. */
export interface FilledLoop {
  /** The loop as it was before the fill. */
  id: LoopId;
  trianglesAdded: number;
  /** Vertices added by refinement. */
  verticesAdded: number;
  /** Whether fairing converged; `true` when it did not run. */
  converged: boolean;
  /** The Laplacian weights fairing used; `null` when it did not run. */
  weights: "cotangent" | "uniform" | null;
}

/** What `fillHoles` did, each list in loop order. */
export interface FillReport {
  filled: FilledLoop[];
  /** Loops left out: beyond a bound, or unclean (or open). */
  skipped: { id: LoopId; reason: "bounds" | "unclean" }[];
  /** Selected loops that could not be filled; they stay open. */
  failed: { id: LoopId; reason: "projectionFolds" | "noValidTriangulation" | "solver" }[];
}

/** The report of a {@link RepairOp}. */
export type RepairReport = WeldReport | DegenerateReport | RemoveReport | OrientReport | FillReport;

/** The report type of a repair op. */
export type RepairReportOf<O extends RepairOp> = O extends { op: "weld" }
  ? WeldReport
  : O extends { op: "removeDegenerate" }
    ? DegenerateReport
    : O extends { op: "removeSmallComponents" }
      ? RemoveReport
      : O extends { op: "orient" }
        ? OrientReport
        : O extends { op: "fillHoles" }
          ? FillReport
          : RepairReport;

/** The scene's view of its active drag. */
interface ActiveDrag {
  /** The step depth right after the drag's `beginStep`. */
  depth: number;
  /** The mesh dragged. */
  mesh: MeshHandle;
  end(): void;
  /** Marks the drag over without touching the history (its step is gone). */
  abandon(): void;
}

const MODES: Record<UpdateMode, number> = { eager: 0, lazy: 1, manual: 2 };

function modeCode(mode: UpdateMode): number {
  if (!Object.hasOwn(MODES, mode)) {
    throw invalid('mode must be "eager", "lazy" or "manual"');
  }
  return MODES[mode];
}

const CULLS = { none: 0, back: 1, front: 2 } as const;
const SIGNS: Record<SignMode, number> = { auto: 0, pseudonormal: 1, winding: 2 };
const MAPS = { turbo: 0, viridis: 1, coolwarm: 2, grayscale: 3 } as const;
const OUT_OF_RANGE: Record<OutOfRange, number> = { clamp: 0, base: 1 };
const DEFAULT_BASE: Rgba = [200, 200, 200, 255];
const DEFAULT_NAN: Rgba = [128, 128, 128, 255];

/** The code of a string option in `table`, or `INVALID_ARGUMENT` naming `what`. */
function codeIn(table: Record<string, number>, value: string, what: string): number {
  if (!Object.hasOwn(table, value)) {
    throw invalid(`${what} must be one of ${Object.keys(table).join(", ")}`);
  }
  return table[value];
}

/** Three numbers of a flat result, from `at`. */
function vec3At(a: Float64Array, at: number): Vec3 {
  return [a[at], a[at + 1], a[at + 2]];
}

/** A colour rule as the binding's `(map, rule)`: see `parse_rule` in the Rust crate. */
function encodeRule(rule: ColorRule): { map: number; values: Float32Array } {
  if (rule.range?.length !== 2) {
    throw invalid("range must be [lo, hi]");
  }
  // NaN is the binding's "no centre": reject it here rather than drop it.
  if (rule.center !== undefined && Number.isNaN(rule.center)) {
    throw invalid("color center must lie within the range");
  }
  const base = rule.base ?? DEFAULT_BASE;
  const nan = rule.nan ?? DEFAULT_NAN;
  if (base.length !== 4) {
    throw invalid("base must have 4 channels");
  }
  if (nan.length !== 4) {
    throw invalid("nan must have 4 channels");
  }
  const values = [
    rule.range[0],
    rule.range[1],
    rule.center ?? Number.NaN,
    codeIn(OUT_OF_RANGE, rule.below ?? "clamp", "below"),
    codeIn(OUT_OF_RANGE, rule.above ?? "base", "above"),
    ...base,
    ...nan,
  ];
  let map: number;
  if (typeof rule.map === "string") {
    map = codeIn(MAPS, rule.map, "map");
  } else if (typeof rule.map === "object" && rule.map !== null && Array.isArray(rule.map.stops)) {
    map = 4;
    for (const stop of rule.map.stops as unknown[]) {
      if (!Array.isArray(stop) || stop.length !== 2) {
        throw invalid("each stop must be [t, rgba]");
      }
      const [t, color] = stop as [number, ArrayLike<number> | null];
      if (typeof color !== "object" || color === null || color.length !== 4) {
        throw invalid("stop colours must have 4 channels");
      }
      values.push(t, ...Array.from(color));
    }
  } else {
    throw invalid('map must be "turbo", "viridis", "coolwarm", "grayscale" or { stops }');
  }
  return { map, values: Float32Array.from(values) };
}

function invalid(message: string): FastMeshError {
  return new FastMeshError("INVALID_ARGUMENT", message);
}

const U32_LIMIT = 2 ** 32;
const EDIT_CODES = {
  setPositions: 0,
  addVertices: 1,
  addTriangles: 2,
  removeTriangles: 3,
  removeVertices: 4,
} as const;

/** An id list as the binding's `n, r, ids..., (slot, op, i)...`; see `crate::edit`. */
function encodeIds(list: unknown, what: string, words: ArrayLike<number>[]): void {
  if (list instanceof Uint32Array) {
    words.push([list.length, 0], list);
    return;
  }
  if (!Array.isArray(list)) {
    throw invalid(`${what} must be a Uint32Array or an array`);
  }
  const ids: number[] = new Array(list.length);
  const refs: number[] = [];
  // An index loop, not forEach: a hole in a sparse array reads as `undefined`
  // here and is rejected, where forEach would skip it and leave the id unset.
  for (let slot = 0; slot < list.length; slot++) {
    const x: unknown = list[slot];
    if (typeof x === "number") {
      if (!Number.isInteger(x) || x < 0 || x >= U32_LIMIT) {
        throw new FastMeshError("INVALID_INDEX", `${what}: ${x} is not an id`);
      }
      ids[slot] = x;
      continue;
    }
    const ref = (x as EditRef | null)?.ref;
    if (!Array.isArray(ref) || ref.length !== 2) {
      throw invalid(`${what} must hold ids or { ref: [opIndex, i] }`);
    }
    for (const r of ref) {
      if (!Number.isInteger(r) || r < 0 || r >= U32_LIMIT) {
        throw new FastMeshError("INVALID_INDEX", `${what}: ref [${ref.join(", ")}] is out of range`);
      }
    }
    ids[slot] = 0;
    refs.push(slot, ref[0], ref[1]);
  }
  words.push([list.length, refs.length / 3], ids, refs);
}

/** World positions of an op, checked to be `3 · count` long (any count when `count` is undefined). */
function opPositions(positions: Positions64, count: number | undefined, what: string): Float64Array {
  const p = toFloat64(positions ?? []);
  if (p.length % 3 !== 0 || (count !== undefined && p.length !== 3 * count)) {
    throw invalid(`${what}: positions need 3 values per vertex`);
  }
  return p;
}

/** An edit op list as the binding's `(ops, values)`; see `crate::edit` in the Rust crate. */
function encodeEdit(ops: readonly EditOp[]): { words: Uint32Array; values: Float64Array } {
  if (!Array.isArray(ops)) {
    throw invalid("ops must be an array");
  }
  const words: ArrayLike<number>[] = [];
  const values: Float64Array[] = [];
  ops.forEach((o, k) => {
    const what = `op ${k}`;
    const code = codeIn(EDIT_CODES, o?.op, `${what}: op`);
    words.push([code]);
    switch (o.op) {
      case "setPositions": {
        encodeIds(o.vertices, `${what}: vertices`, words);
        values.push(opPositions(o.positions, o.vertices?.length, what));
        break;
      }
      case "addVertices": {
        const p = opPositions(o.positions, undefined, what);
        words.push([p.length / 3]);
        values.push(p);
        break;
      }
      case "addTriangles":
        encodeIds(o.triangles, `${what}: triangles`, words);
        break;
      case "removeTriangles":
        encodeIds(o.triangles, `${what}: triangles`, words);
        break;
      case "removeVertices":
        encodeIds(o.vertices, `${what}: vertices`, words);
        break;
    }
  });
  const out = new Uint32Array(words.reduce((n, w) => n + w.length, 0));
  let at = 0;
  for (const w of words) {
    out.set(w, at);
    at += w.length;
  }
  const flat = new Float64Array(values.reduce((n, v) => n + v.length, 0));
  at = 0;
  for (const v of values) {
    flat.set(v, at);
    at += v.length;
  }
  return { words: out, values: flat };
}

const REPAIR_CODES = { weld: 0, removeDegenerate: 1, removeSmallComponents: 2, orient: 3, fillHoles: 4 } as const;

/** An optional number for the binding: NaN when unset; an explicit NaN is rejected. */
function optional(v: number | undefined, what: string): number {
  if (v === undefined) {
    return Number.NaN;
  }
  if (typeof v !== "number" || Number.isNaN(v)) {
    throw invalid(`${what} must be a number`);
  }
  return v;
}

function flagOf(v: boolean | undefined, fallback: boolean): number {
  return (v ?? fallback) ? 1 : 0;
}

/** A loop id as the binding's `minVertex, len, hash low, hash high`. */
function encodeLoopId(id: LoopId, out: number[]): void {
  const { minVertex, len, hash } = id ?? ({} as LoopId);
  if (typeof hash !== "bigint" || hash < 0n || hash >= 1n << 64n) {
    throw invalid("a loop id's hash must be a bigint in 0..2^64");
  }
  for (const v of [minVertex, len]) {
    if (!Number.isInteger(v) || v < 0 || v >= U32_LIMIT) {
      throw invalid("a loop id's minVertex and len must be u32 integers");
    }
  }
  out.push(minVertex, len, Number(hash & 0xffffffffn), Number(hash >> 32n));
}

/** A repair op list as the binding's `(ops, loops)`; see `crate::edit` in the Rust crate. */
function encodeRepair(ops: readonly RepairOp[]): { values: Float64Array; loops: Uint32Array } {
  const values: number[] = [];
  const loops: number[] = [];
  for (const o of ops) {
    values.push(codeIn(REPAIR_CODES, o?.op, "repair op"));
    switch (o.op) {
      case "weld":
        values.push(optional(o.tolerance, "tolerance"));
        break;
      case "removeDegenerate":
        values.push(optional(o.areaEpsilon, "areaEpsilon"), flagOf(o.removeUnusedVertices, true));
        break;
      case "removeSmallComponents":
        values.push(
          optional(o.minTriangles, "minTriangles"),
          optional(o.minArea, "minArea"),
          flagOf(o.keepLargest, true),
        );
        break;
      case "orient":
        break;
      case "fillHoles": {
        let count = Number.NaN;
        if (o.loops !== undefined) {
          if (!Array.isArray(o.loops)) {
            throw invalid("loops must be an array of loop ids");
          }
          count = o.loops.length;
          for (const id of o.loops) {
            encodeLoopId(id, loops);
          }
        }
        values.push(
          optional(o.maxEdges, "maxEdges"),
          optional(o.maxPerimeter, "maxPerimeter"),
          optional(o.maxArea, "maxArea"),
          count,
          flagOf(o.allowUnclean, false),
          flagOf(o.fair, true),
          optional(o.dpLimit, "dpLimit"),
        );
        break;
      }
    }
  }
  return { values: Float64Array.from(values), loops: Uint32Array.from(loops) };
}

/** Reads a flat `f64` result front to back. */
class Reader {
  at = 0;
  constructor(readonly data: Float64Array) {}

  next(): number {
    return this.data[this.at++];
  }

  flag(): boolean {
    return this.next() === 1;
  }

  /** `n` ids as a copy. */
  ids(n: number): Uint32Array {
    const out = Uint32Array.from(this.data.subarray(this.at, this.at + n));
    this.at += n;
    return out;
  }

  /** A count, then that many ids. */
  list(): Uint32Array {
    return this.ids(this.next());
  }

  loopId(): LoopId {
    const minVertex = this.next();
    const len = this.next();
    const lo = BigInt(this.next());
    const hi = BigInt(this.next());
    return { minVertex, len, hash: lo | (hi << 32n) };
  }

  /** A count, then that many items. */
  many<T>(item: () => T): T[] {
    const n = this.next();
    const out: T[] = [];
    for (let k = 0; k < n; k++) {
      out.push(item());
    }
    return out;
  }
}

const SKIP_REASONS = ["bounds", "unclean"] as const;
const FAIL_REASONS = ["projectionFolds", "noValidTriangulation", "solver"] as const;
const WEIGHTS = [null, "cotangent", "uniform"] as const;

/** One op's report from a repair result; see `crate::edit` in the Rust crate. */
function decodeReport(op: RepairOp["op"], r: Reader): RepairReport {
  switch (op) {
    case "weld":
      return { mergedVertices: r.next(), removedTriangles: r.next() };
    case "removeDegenerate":
      return { removedTriangles: r.next(), removedVertices: r.next() };
    case "removeSmallComponents":
      return { removedComponents: r.next(), removedTriangles: r.next(), removedVertices: r.next() };
    case "orient":
      return {
        flippedTriangles: r.next(),
        patches: r.next(),
        nonOrientablePatches: r.next(),
        flippedOutwardComponents: r.next(),
      };
    case "fillHoles":
      return {
        filled: r.many(() => ({
          id: r.loopId(),
          trianglesAdded: r.next(),
          verticesAdded: r.next(),
          converged: r.flag(),
          weights: WEIGHTS[r.next()],
        })),
        skipped: r.many(() => ({ id: r.loopId(), reason: SKIP_REASONS[r.next()] })),
        failed: r.many(() => ({ id: r.loopId(), reason: FAIL_REASONS[r.next()] })),
      };
  }
}

/** An analysis result; see `encode_analysis` in the Rust crate. */
function decodeAnalysis(r: Reader): Analysis {
  const loops = r.many(
    (): BoundaryLoop => ({
      id: r.loopId(),
      vertices: r.list(),
      perimeter: r.next(),
      area: r.next(),
      closed: r.flag(),
      clean: r.flag(),
    }),
  );
  const coincidentGroups = r.next();
  const coincident = r.many(() => r.list());
  const degenerate = r.next();
  const degenerateList = r.list();
  const duplicate = r.next();
  const duplicateList = r.list();
  const nonManifoldEdges = r.next();
  const nonManifoldEdgeList = r.many((): [number, number] => [r.next(), r.next()]);
  const nonManifoldVertices = r.list();
  const pinchVertices = r.list();
  const components = r.many(
    (): MeshComponent => ({ triangles: r.list(), area: r.next(), closed: r.flag() }),
  );
  const orientationConflicts = r.next();
  const si = r.next();
  return {
    loops,
    coincidentGroups,
    coincident,
    degenerate,
    degenerateList,
    duplicate,
    duplicateList,
    nonManifoldEdges,
    nonManifoldEdgeList,
    nonManifoldVertices,
    pinchVertices,
    components,
    orientationConflicts,
    selfIntersects: si < 0 ? null : si === 1,
  };
}

function toFloat64(values: Positions64): Float64Array {
  return values instanceof Float64Array ? values : Float64Array.from(values);
}

function toUint32(values: Indices): Uint32Array {
  return values instanceof Uint32Array ? values : Uint32Array.from(values);
}

/**
 * Stepped field updates of a scene (the scene crate's `field_step` API),
 * for the Worker engine only: not part of the public API, and not exported
 * from the package entries.
 *
 * - `setDefer(on)`: with `on`, the changes that would update the fields of
 *   `"eager"` meshes leave them due instead (adding a field still computes
 *   it at once).
 * - `step(points)`: computes up to `points` vertices of the due fields,
 *   opening a unit of every due field first (an undivided part). The call
 *   that computes the last applies the unit and returns `"done"`, exactly as
 *   a one-shot update at the state it opened at; `"idle"`: nothing was due.
 *   Any other scene call that may change the scene drops an open unit. A
 *   field error throws and ends the unit (the failing fields stay due).
 * - `abandon()`: drops the open unit; its fields stay due.
 * - `meshes()`: the meshes of the open unit's fields, each once.
 */
interface FieldSteps {
  setDefer(on: boolean): void;
  step(points: number): "idle" | "pending" | "done";
  abandon(): void;
  meshes(): MeshHandle[];
}

/** Set by `Scene`'s static block. */
let fieldStepsOf: (scene: Scene) => FieldSteps;

/** @internal The stepped field updates of `scene` (see {@link FieldSteps}); for the Worker engine. */
export function fieldSteps(scene: Scene): FieldSteps {
  return fieldStepsOf(scene);
}

/**
 * A scene of editable meshes. Create it after `await init()`; release it with
 * {@link Scene.free} or a `using` declaration (otherwise it is freed when
 * garbage collected).
 *
 * Render views ({@link Scene.views}) point into WebAssembly memory and are
 * only valid until the next scene call that may allocate or move buffers: a
 * kept view may read other data or be empty. Re-acquire them after every
 * scene call; see {@link MeshViews}.
 */
export class Scene implements SceneLike {
  #raw: WasmScene | undefined;
  readonly #guard: ViewGuard | undefined;
  /** Live meshes' handles by `"index:generation"`. */
  readonly #meshes = new Map<string, MeshHandle>();
  /** Handles of meshes removed since the last `takeChanges`, for their removal records. */
  readonly #removed = new Map<string, MeshHandle>();
  /**
   * Field and section handles by raw id, kept for the scene's lifetime (raw
   * ids are never reused), so change records reporting a removed field or
   * section give back the handle the caller holds.
   */
  readonly #fields = new Map<number, FieldHandle>();
  readonly #sections = new Map<number, SectionHandle>();
  readonly #listeners = new Set<(changes: MeshChange[]) => void>();
  /** History step nesting depth, mirrored from the calls made through this class. */
  #stepDepth = 0;
  #drag: ActiveDrag | undefined;

  constructor(options: SceneOptions = {}) {
    ensureInitialized("new Scene()");
    this.#raw = call(() => new WasmScene());
    this.#guard = options.debug ? new ViewGuard() : undefined;
    finalizer.register(this, this.#raw, this);
  }

  /** Releases the scene and everything in it. Later calls throw `INVALID_ARGUMENT`; calling it again does nothing. */
  free(): void {
    const raw = this.#raw;
    if (raw === undefined) {
      return;
    }
    this.#raw = undefined;
    this.#guard?.bump();
    this.#drag = undefined;
    finalizer.unregister(this);
    release(raw);
  }

  /** Same as {@link Scene.free}, for `using scene = new Scene()`. */
  [Symbol.dispose](): void {
    this.free();
  }

  /** The live raw scene, for a call that may allocate or move buffers. */
  #scene(): WasmScene {
    const s = this.#peek();
    this.#guard?.bump();
    return s;
  }

  /** The live raw scene, for a read-only call that leaves every view valid. */
  #peek(): WasmScene {
    if (this.#raw === undefined) {
      throw invalid("scene is freed");
    }
    return this.#raw;
  }

  /** The live mesh id of a handle of this scene. */
  #id(mesh: MeshHandle): HandleId {
    this.#peek();
    const id = meshIds.get(mesh);
    if (id === undefined) {
      throw invalid("not a mesh handle");
    }
    if (id.scene !== this) {
      throw invalid("mesh handle belongs to another scene");
    }
    return id;
  }

  #meshHandle(index: number, generation: number): MeshHandle {
    const key = `${index}:${generation}`;
    let h = this.#meshes.get(key) ?? this.#removed.get(key);
    if (h === undefined) {
      h = new MeshHandle();
      meshIds.set(h, { scene: this, index, generation });
      this.#meshes.set(key, h);
    }
    return h;
  }

  #rawHandle<H extends FieldHandle | SectionHandle>(
    map: Map<number, H>,
    make: () => H,
    raw: number,
  ): H {
    let h = map.get(raw);
    if (h === undefined) {
      h = make();
      rawIds.set(h, { scene: this, index: raw, generation: 0 });
      map.set(raw, h);
    }
    return h;
  }

  /** The raw id of a field or section handle of this scene. */
  #rawId(handle: FieldHandle | SectionHandle, kind: typeof FieldHandle | typeof SectionHandle): number {
    this.#peek();
    const what = kind === FieldHandle ? "field" : "section";
    const id = handle instanceof kind ? rawIds.get(handle) : undefined;
    if (id === undefined) {
      throw invalid(`not a ${what} handle`);
    }
    if (id.scene !== this) {
      throw invalid(`${what} handle belongs to another scene`);
    }
    return id.index;
  }

  #field(field: FieldHandle): number {
    return this.#rawId(field, FieldHandle);
  }

  #section(section: SectionHandle): number {
    return this.#rawId(section, SectionHandle);
  }

  /** Meshes as the binding's flat `[index, generation, ...]` list. */
  #flatIds(meshes: readonly MeshHandle[]): Uint32Array {
    if (!Array.isArray(meshes)) {
      throw invalid("meshes must be an array of mesh handles");
    }
    const out = new Uint32Array(meshes.length * 2);
    meshes.forEach((m, k) => {
      const { index, generation } = this.#id(m);
      out[2 * k] = index;
      out[2 * k + 1] = generation;
    });
    return out;
  }

  /** `[origin, normal]` as the binding's 6 numbers. */
  #plane(plane: SectionPlane): Float64Array {
    if (plane?.origin?.length !== 3 || plane.normal?.length !== 3) {
      throw invalid("plane needs a 3-number origin and normal");
    }
    return Float64Array.from([...plane.origin, ...plane.normal]);
  }

  // ---- Meshes ----

  /**
   * Adds a mesh from world positions. They are stored as `p − origin` in
   * `f32` (see {@link AddMeshOptions.origin}); the placement starts as the
   * identity, so {@link Scene.matrix4} maps the local coordinates back to
   * where they were given.
   *
   * @throws `INVALID_ARGUMENT` for bad lengths or options, `INVALID_INDEX` for an index out of range.
   */
  addMesh(positions: Positions64, indices: Indices, options: AddMeshOptions = {}): MeshHandle {
    const s = this.#scene();
    const mode = modeCode(options.mode ?? "eager");
    const origin = options.origin === undefined ? undefined : Float64Array.from(options.origin);
    const p = toFloat64(positions);
    const i = toUint32(indices);
    const id = call(() => s.add_mesh(p, i, origin, mode, options.normals ?? true));
    return this.#meshHandle(id[0], id[1]);
  }

  /**
   * Removes a mesh. Its handle then gives `UNKNOWN_MESH`. An active drag of
   * the mesh is ended first, closing its step; the mesh's history entries
   * (the drag's included) are then dropped with it.
   */
  removeMesh(mesh: MeshHandle): void {
    const { index, generation } = this.#id(mesh);
    // A drag of this mesh ends first, recording its moves as one step.
    if (this.#drag?.mesh === mesh) {
      this.#drag.end();
    }
    call(() => this.#scene().remove_mesh(index, generation));
    const key = `${index}:${generation}`;
    this.#meshes.delete(key);
    this.#removed.set(key, mesh);
  }

  /**
   * Sets a mesh's placement: world = rotation · local + translation, where
   * local is relative to the mesh origin. A quaternion is normalised and
   * converted in double precision.
   *
   * @throws `INVALID_ARGUMENT` when the rotation is not a proper rotation or a quaternion is zero.
   */
  setPlacement(mesh: MeshHandle, placement: PlacementInput): void {
    const { index, generation } = this.#id(mesh);
    const r = Float64Array.from(rotationMatrix(placement.rotation));
    const t = Float64Array.from(placement.translation);
    call(() => this.#scene().set_placement(index, generation, r, t));
  }

  /** A mesh's placement, with the rotation as a row-major matrix. */
  placement(mesh: MeshHandle): Placement {
    const { index, generation } = this.#id(mesh);
    const p = call(() => this.#peek().placement(index, generation));
    return { rotation: Array.from(p.subarray(0, 9)), translation: [p[9], p[10], p[11]] };
  }

  /**
   * The column-major world matrix of the local render coordinates,
   * `placement · translate(origin)`, ready for renderers (convert to `f32`
   * yourself).
   */
  matrix4(mesh: MeshHandle): Float64Array {
    const { index, generation } = this.#id(mesh);
    return call(() => this.#peek().matrix4(index, generation));
  }

  /** Sets when the mesh's derived data follows its edits; see {@link UpdateMode}. */
  setMode(mesh: MeshHandle, mode: UpdateMode): void {
    const { index, generation } = this.#id(mesh);
    const m = modeCode(mode);
    call(() => this.#scene().set_mode(index, generation, m));
  }

  /** Brings the mesh's derived data up to date now, whatever its mode. */
  update(mesh: MeshHandle): void {
    const { index, generation } = this.#id(mesh);
    call(() => this.#scene().update(index, generation));
  }

  /** Brings every mesh that is not `"manual"` up to date. */
  flush(): void {
    const s = this.#scene();
    call(() => s.flush());
  }

  /** Vertex slots, alive or dead: the vertex length of the render buffers. */
  vertexSlots(mesh: MeshHandle): number {
    const { index, generation } = this.#id(mesh);
    return call(() => this.#peek().vertex_slots(index, generation));
  }

  /** Triangle slots, alive or dead: the triangle length of the index buffer. */
  triangleSlots(mesh: MeshHandle): number {
    const { index, generation } = this.#id(mesh);
    return call(() => this.#peek().triangle_slots(index, generation));
  }

  /** Alive vertices. */
  vertexCount(mesh: MeshHandle): number {
    const { index, generation } = this.#id(mesh);
    return call(() => this.#peek().vertex_count(index, generation));
  }

  /** Alive triangles. */
  triangleCount(mesh: MeshHandle): number {
    const { index, generation } = this.#id(mesh);
    return call(() => this.#peek().triangle_count(index, generation));
  }

  /**
   * Zero-copy views of the mesh's render buffers. Each getter builds a fresh
   * view; re-acquire views after every scene call that may allocate, since a
   * kept one may read other data or be empty (see {@link MeshViews}).
   */
  views(mesh: MeshHandle): MeshViews {
    const { index, generation } = this.#id(mesh);
    // Fail now, not on the first getter, for a removed mesh.
    call(() => this.#peek().vertex_slots(index, generation));
    return meshViews(() => this.#peek(), index, generation, this.#guard);
  }

  // ---- Changes ----

  /**
   * Everything that changed since the previous call, one record per mesh.
   * There is a single consumer: apps that use {@link Scene.onChanges} call
   * {@link Scene.commitFrame} instead.
   */
  takeChanges(): MeshChange[] {
    const s = this.#scene();
    const raw = decodeChanges(call(() => s.take_changes()));
    const out = raw.map(
      (c): MeshChange => ({
        mesh: this.#meshHandle(c.index, c.generation),
        added: c.added,
        removed: c.removed,
        placementChanged: c.placementChanged,
        capacityChanged: c.capacityChanged,
        positions: c.positions,
        normals: c.normals,
        colors: c.colors,
        indices: c.indices,
        fieldsChanged: c.fields.map((f) =>
          this.#rawHandle(this.#fields, () => new FieldHandle(), f),
        ),
        sectionsChanged: c.sections.map((x) =>
          this.#rawHandle(this.#sections, () => new SectionHandle(), x),
        ),
      }),
    );
    // A removed mesh's id never matches again: drop its handle, whatever
    // removed it. Every removal is in this message (or was never reported,
    // for a mesh added and removed in between), so removed handles can go.
    for (const c of raw) {
      if (c.removed) {
        this.#meshes.delete(`${c.index}:${c.generation}`);
      }
    }
    this.#removed.clear();
    return out;
  }

  /**
   * Registers a listener for {@link Scene.commitFrame}. Returns a function
   * that unregisters it.
   */
  onChanges(listener: (changes: MeshChange[]) => void): () => void {
    this.#scene();
    this.#listeners.add(listener);
    return () => {
      this.#listeners.delete(listener);
    };
  }

  /**
   * Takes the changes once and delivers the same array to every listener,
   * then returns it. Call it once per animation frame. Every listener runs
   * even if one throws; the first error is rethrown afterwards.
   */
  commitFrame(): MeshChange[] {
    const changes = this.takeChanges();
    let failure: { error: unknown } | undefined;
    for (const listener of [...this.#listeners]) {
      try {
        listener(changes);
      } catch (error) {
        failure ??= { error };
      }
    }
    if (failure !== undefined) {
      throw failure.error;
    }
    return changes;
  }

  // ---- History ----

  /**
   * Undoes the latest step; an active drag is ended first. Without an open
   * step around the drag, the drag is what gets undone; inside a caller's
   * open step, that whole step (drag included) is closed and undone. Returns
   * `false` when there is nothing to undo.
   */
  undo(): boolean {
    const s = this.#scene();
    this.#drag?.end();
    const done = call(() => s.undo());
    this.#stepDepth = 0;
    return done;
  }

  /** Redoes the latest undone step. Returns `false` when there is nothing to redo. */
  redo(): boolean {
    const s = this.#scene();
    this.#drag?.end();
    const done = call(() => s.redo());
    this.#stepDepth = 0;
    return done;
  }

  /** True if {@link Scene.undo} would undo something. */
  canUndo(): boolean {
    const s = this.#peek();
    return call(() => s.can_undo());
  }

  /** True if {@link Scene.redo} would redo something. */
  canRedo(): boolean {
    const s = this.#peek();
    return call(() => s.can_redo());
  }

  /** Closes an open step, then drops every undo and redo step. */
  clearHistory(): void {
    const s = this.#scene();
    this.#drag?.end();
    call(() => s.clear_history());
    this.#stepDepth = 0;
  }

  /**
   * Opens a history step, or one more nesting level of the open one. Edits
   * and placement changes until the matching {@link Scene.endStep} are
   * undone together.
   */
  beginStep(): void {
    const s = this.#scene();
    call(() => s.begin_step());
    this.#stepDepth++;
  }

  /** Ends one nesting level of the open step. Throws `STEP_NOT_OPEN` when none is open. */
  endStep(): void {
    const s = this.#scene();
    call(() => s.end_step());
    this.#stepDepth--;
    // The caller closed the drag's step themselves.
    if (this.#drag !== undefined && this.#stepDepth < this.#drag.depth) {
      this.#drag.abandon();
    }
  }

  /** Reverts and discards the open step, however deeply nested. Throws `STEP_NOT_OPEN` when none is open. */
  cancelStep(): void {
    const s = this.#scene();
    call(() => s.cancel_step());
    this.#stepDepth = 0;
    // The drag's step is gone with it: later moves must not record steps.
    this.#drag?.abandon();
  }

  /**
   * Starts dragging a mesh: the moves until {@link DragSession.end} form one
   * undo step (nested in the open step, if there is one). One drag at a time.
   * Every {@link DragSession.move} recomputes the eager fields, so make at most
   * one move per animation frame, with the latest pointer pose.
   */
  drag(mesh: MeshHandle): DragSession {
    this.#id(mesh);
    if (this.#drag !== undefined) {
      throw invalid("a drag is already active; end or cancel it first");
    }
    const start = this.placement(mesh);
    const outermost = this.#stepDepth === 0;
    this.beginStep();
    const depth = this.#stepDepth;
    let active = true;
    const finish = (): boolean => {
      if (!active) {
        return false;
      }
      active = false;
      this.#drag = undefined;
      return true;
    };
    // Closes the drag's level and any opened inside it.
    const close = () => {
      while (this.#stepDepth >= depth) {
        this.endStep();
      }
    };
    const session: DragSession & ActiveDrag = {
      depth,
      mesh,
      get active() {
        return active;
      },
      move: (placement) => {
        if (!active) {
          throw invalid("the drag is over");
        }
        this.setPlacement(mesh, placement);
      },
      end: () => {
        if (finish()) {
          close();
        }
      },
      cancel: () => {
        if (!finish()) {
          return;
        }
        if (outermost) {
          // The drag's step is the whole step: reverting it puts the
          // placement back and leaves no entry (a restore-then-end would
          // leave a no-op one).
          this.cancelStep();
          return;
        }
        // Inside the caller's step: restore, keeping their step open.
        try {
          this.setPlacement(mesh, start);
        } finally {
          close();
        }
      },
      abandon: () => {
        finish();
      },
    };
    this.#drag = session;
    return session;
  }

  // ---- Geometry edits and repair ----

  /**
   * Edits a mesh's geometry with a list of ops, run in Rust as one
   * transaction: one undo step (or part of the open step), and when any op
   * fails nothing changes and nothing is recorded. Positions are world
   * coordinates, mapped to local with the mesh's placement and origin. Ids
   * may be {@link EditRef}s to ids added by earlier ops of the list; every
   * ref is checked before anything is written. Normals, colours, fields,
   * sections and the BVH follow by the mesh's mode.
   *
   * @throws `INVALID_INDEX` for a bad ref or an id that is not alive;
   * `INVALID_ARGUMENT` for a malformed op, positions not 3 per vertex or not
   * finite in local `f32`, or removing a vertex still in use.
   */
  edit(mesh: MeshHandle, ops: readonly EditOp[]): EditResult {
    const { index, generation } = this.#id(mesh);
    const { words, values } = encodeEdit(ops);
    const out = call(() => this.#scene().edit(index, generation, words, values));
    const added: Uint32Array[] = [];
    let at = 0;
    while (at < out.length) {
      const n = out[at];
      added.push(out.slice(at + 1, at + 1 + n));
      at += 1 + n;
    }
    return { added };
  }

  /**
   * Removes the mesh's dead slots and renumbers the rest. Its next change
   * record has every buffer `"all"` and `capacityChanged` when the slot counts
   * shrank. The mesh's earlier edits leave the history (its placement changes
   * stay undoable).
   */
  compact(mesh: MeshHandle): void {
    const { index, generation } = this.#id(mesh);
    call(() => this.#scene().compact(index, generation));
  }

  /**
   * A read-only report of the mesh's defects: boundary loops (whose ids
   * `fillHoles` takes), coincident vertices, degenerate and duplicate
   * triangles, non-manifold edges and vertices, components, orientation
   * conflicts and, on request, self-intersection.
   *
   * @throws `INVALID_ARGUMENT` for a negative or non-finite `weldTolerance`; `STALE_BVH`, see {@link AnalyzeOptions.selfIntersection}.
   */
  analyze(mesh: MeshHandle, options: AnalyzeOptions = {}): Analysis {
    const { index, generation } = this.#id(mesh);
    const maxListed = options.maxListed ?? 1000;
    if (!Number.isInteger(maxListed) || maxListed < 0 || maxListed >= U32_LIMIT) {
      throw invalid("maxListed must be a non-negative integer");
    }
    const tolerance = options.weldTolerance ?? 0;
    const si = options.selfIntersection ?? false;
    const r = call(() => this.#scene().analyze(index, generation, tolerance, si, maxListed));
    return decodeAnalysis(new Reader(r));
  }

  /**
   * Runs one repair operation and returns its report, or a list of them in
   * order and returns their reports. A list is one transaction: one undo step,
   * and when any operation fails nothing changes and nothing is recorded.
   *
   * @throws `INVALID_ARGUMENT` for invalid options (`fillHoles` needs a bound or `loops`); `STALE_LOOPS` when a loop id of `fillHoles` no longer matches a loop of the mesh.
   */
  repair<const O extends RepairOp>(mesh: MeshHandle, op: O): RepairReportOf<O>;
  repair<const O extends readonly RepairOp[]>(
    mesh: MeshHandle,
    ops: O,
  ): { -readonly [K in keyof O]: RepairReportOf<O[K]> };
  repair(mesh: MeshHandle, op: RepairOp | readonly RepairOp[]): RepairReport | RepairReport[] {
    const { index, generation } = this.#id(mesh);
    const ops: readonly RepairOp[] = Array.isArray(op) ? op : [op as RepairOp];
    const { values, loops } = encodeRepair(ops);
    const r = new Reader(call(() => this.#scene().repair(index, generation, values, loops)));
    const reports = ops.map((o) => decodeReport(o.op, r));
    return Array.isArray(op) ? reports : reports[0];
  }

  // ---- Queries (world space) ----

  /**
   * The nearest hit of a world ray over every mesh, or `null`. `direction`
   * need not be unit length; `distance` is in world units. Lazy meshes are
   * brought up to date first.
   *
   * @throws `INVALID_ARGUMENT` for a non-finite origin or a zero direction; `STALE_BVH` for a stale `"manual"` mesh.
   */
  raycast(origin: Readonly<Vec3>, direction: Readonly<Vec3>, options: RaycastOptions = {}): RayHit | null {
    const s = this.#scene();
    const o = Float64Array.from(origin);
    const d = Float64Array.from(direction);
    const cull = codeIn(CULLS, options.cull ?? "none", "cull");
    const min = options.minDistance ?? 0;
    const max = options.maxDistance ?? Number.POSITIVE_INFINITY;
    const h = call(() => s.raycast(o, d, min, max, cull));
    if (h.length === 0) {
      return null;
    }
    return {
      mesh: this.#meshHandle(h[0], h[1]),
      triangle: h[2],
      point: vec3At(h, 3),
      distance: h[6],
      barycentric: vec3At(h, 7),
    };
  }

  /**
   * The surface point nearest to a world point, over every mesh or one, or
   * `null` when none is within `maxDistance`.
   */
  closestPoint(point: Readonly<Vec3>, options: ClosestPointOptions = {}): ClosestHit | null {
    const s = this.#scene();
    const p = Float64Array.from(point);
    const max = options.maxDistance ?? Number.POSITIVE_INFINITY;
    let h: Float64Array;
    if (options.mesh === undefined) {
      h = call(() => s.closest_point(p, max));
    } else {
      const { index, generation } = this.#id(options.mesh);
      h = call(() => s.closest_point_on(index, generation, p, max));
    }
    if (h.length === 0) {
      return null;
    }
    return {
      mesh: this.#meshHandle(h[0], h[1]),
      triangle: h[2],
      point: vec3At(h, 3),
      distance: h[6],
    };
  }

  /**
   * The distance between two meshes' surfaces as placed, with a closest pair
   * of points, or `null` when it exceeds `maxDistance` (default `Infinity`)
   * or a mesh has no triangle.
   */
  meshDistance(a: MeshHandle, b: MeshHandle, options: { maxDistance?: number } = {}): MeshDistance | null {
    const ia = this.#id(a);
    const ib = this.#id(b);
    const max = options.maxDistance ?? Number.POSITIVE_INFINITY;
    const d = call(() =>
      this.#scene().mesh_distance(ia.index, ia.generation, ib.index, ib.generation, max),
    );
    if (d.length === 0) {
      return null;
    }
    return { distance: d[0], a: vec3At(d, 1), b: vec3At(d, 4), triangleA: d[7], triangleB: d[8] };
  }

  /** Whether two meshes intersect or touch as placed. */
  intersects(a: MeshHandle, b: MeshHandle): boolean {
    const ia = this.#id(a);
    const ib = this.#id(b);
    return call(() => this.#scene().intersects(ia.index, ia.generation, ib.index, ib.generation));
  }

  /** Every mesh pair (or every pair of `pairs`) within `margin`, sorted by mesh id. */
  findContacts(options: FindContactsOptions = {}): Contact[] {
    const s = this.#scene();
    const pairs = options.pairs === undefined ? undefined : this.#flatIds(options.pairs.flat());
    const msg = call(() => s.find_contacts(options.margin ?? 0, pairs));
    const out: Contact[] = [];
    let at = 1;
    for (let k = 0; k < msg[0]; k++) {
      const na = msg[at + 6];
      const nb = msg[at + 7];
      const start = at + 8;
      out.push({
        a: this.#meshHandle(msg[at], msg[at + 1]),
        b: this.#meshHandle(msg[at + 2], msg[at + 3]),
        intersecting: msg[at + 4] === 1,
        distance: msg[at + 5],
        trianglesA: Uint32Array.from(msg.subarray(start, start + na)),
        trianglesB: Uint32Array.from(msg.subarray(start + na, start + na + nb)),
      });
      at = start + na + nb;
    }
    return out;
  }

  // ---- Fields and colours ----

  /**
   * Adds a signed distance field: one value per vertex slot of `on`, the
   * distance to `to` as placed (negative inside), `Infinity` beyond
   * `maxDistance`, NaN for dead or unused vertices. It follows edits and
   * placements by the mesh modes, like normals.
   *
   * @throws `INVALID_ARGUMENT` when `on` is `to`, `maxDistance` is negative or NaN, or the name is taken.
   */
  addDistanceField(desc: DistanceFieldDesc): FieldHandle {
    const s = this.#scene();
    const ids = this.#flatIds([desc.on, desc.to]);
    const sign = codeIn(SIGNS, desc.sign ?? "auto", "sign");
    if (typeof desc.name !== "string") {
      throw invalid("name must be a string");
    }
    const raw = call(() => s.add_distance_field(desc.name, ids, desc.maxDistance, sign));
    return this.#rawHandle(this.#fields, () => new FieldHandle(), raw);
  }

  /** Removes a field, and a colouring by it. Its handle then gives `UNKNOWN_FIELD`. */
  removeField(field: FieldHandle): void {
    const raw = this.#field(field);
    call(() => this.#scene().remove_field(raw));
  }

  /**
   * A zero-copy view of the field's values as of its last update, one per
   * vertex slot. Like render views, it is only valid until the next call
   * that may allocate (a kept one may read other data or be empty): get it
   * right before use.
   */
  fieldValues(field: FieldHandle): Float32Array {
    const raw = this.#field(field);
    const s = this.#peek();
    const ptr = call(() => s.field_values_ptr(raw));
    const view = new Float32Array(wasmMemory().buffer, ptr, call(() => s.field_values_len(raw)));
    return this.#guard === undefined ? view : this.#guard.wrap(view);
  }

  /** Whether the field is current, and what its sign data found out about the target mesh. */
  fieldStatus(field: FieldHandle): FieldStatus {
    const raw = this.#field(field);
    const st = call(() => this.#peek().field_status(raw));
    return {
      current: st[0] === 1,
      sign: {
        mode: st[1] === 2 ? "winding" : "pseudonormal",
        closed: st[2] === 1,
        consistentOrientation: st[3] === 1,
        nonManifoldVertices: st[4],
        degenerateTriangles: st[5],
      },
    };
  }

  /**
   * Colours a mesh from one of its fields (`views(mesh).colors()`), or turns
   * colouring off with `null`, after which `colors()` returns `null`. Every
   * vertex is recoloured and reported as `"all"`; later field updates (edits,
   * drags) report only the vertices whose colour may change. A display
   * setting: never recorded in history.
   *
   * @throws `INVALID_ARGUMENT` when the field is not on `mesh` or the rule is invalid; `UNKNOWN_FIELD`.
   */
  setColoring(mesh: MeshHandle, rule: ColorRule | null): void {
    const { index, generation } = this.#id(mesh);
    if (rule === null) {
      call(() => this.#scene().clear_coloring(index, generation));
      return;
    }
    const field = this.#field(rule.field);
    const { map, values } = encodeRule(rule);
    call(() => this.#scene().set_coloring(index, generation, field, map, values));
  }

  // ---- Stepped field updates (internal: see fieldSteps) ----

  static {
    fieldStepsOf = (scene) => ({
      setDefer: (on) => call(() => scene.#peek().set_defer_fields(on)),
      step: (points) => {
        const s = scene.#scene();
        const state = call(() => s.field_step(points));
        return state === 0 ? "idle" : state === 1 ? "pending" : "done";
      },
      abandon: () => call(() => scene.#peek().abandon_field_step()),
      meshes: () => {
        const flat = call(() => scene.#peek().field_step_meshes());
        const out: MeshHandle[] = [];
        for (let k = 0; k < flat.length; k += 2) {
          out.push(scene.#meshHandle(flat[k], flat[k + 1]));
        }
        return out;
      },
    });
  }

  // ---- Sections ----

  /**
   * Adds a plane section of `meshes`. Results of `"eager"` meshes are computed
   * right away; others on {@link Scene.updateSections} or `flush()`.
   *
   * @throws `INVALID_ARGUMENT` for a zero normal or a mesh listed twice.
   */
  addSection(desc: SectionDesc): SectionHandle {
    const s = this.#scene();
    const plane = this.#plane(desc.plane);
    const meshes = this.#flatIds(desc.meshes);
    const raw = call(() => s.add_section(plane, meshes, desc.fill ?? false));
    return this.#rawHandle(this.#sections, () => new SectionHandle(), raw);
  }

  /** Moves a section's plane. */
  setSectionPlane(section: SectionHandle, plane: SectionPlane): void {
    const raw = this.#section(section);
    const p = this.#plane(plane);
    call(() => this.#scene().set_section_plane(raw, p));
  }

  /**
   * A section's plane as last set by {@link Scene.addSection} or
   * {@link Scene.setSectionPlane}, the normal normalised. Results not yet
   * recomputed (`"stale"`) may still be cut with an earlier plane.
   */
  sectionPlane(section: SectionHandle): SectionPlane {
    const raw = this.#section(section);
    const p = call(() => this.#peek().section_desc_plane(raw));
    return { origin: vec3At(p, 0), normal: vec3At(p, 3) };
  }

  /** Changes the meshes a section cuts. */
  setSectionMeshes(section: SectionHandle, meshes: readonly MeshHandle[]): void {
    const raw = this.#section(section);
    const ids = this.#flatIds(meshes);
    call(() => this.#scene().set_section_meshes(raw, ids));
  }

  /** Turns a section's fill on or off. */
  setSectionFill(section: SectionHandle, fill: boolean): void {
    const raw = this.#section(section);
    call(() => this.#scene().set_section_fill(raw, fill));
  }

  /** Removes a section. Its handle then gives `UNKNOWN_SECTION`. */
  removeSection(section: SectionHandle): void {
    const raw = this.#section(section);
    call(() => this.#scene().remove_section(raw));
  }

  /** Computes every stale section result, whatever the mesh modes. */
  updateSections(): void {
    const s = this.#scene();
    call(() => s.update_sections());
  }

  /**
   * A section's result for one of its meshes as of its last computation
   * (empty before the first), copied, in world coordinates.
   *
   * @throws `INVALID_ARGUMENT` when the section does not cut `mesh`.
   */
  section(section: SectionHandle, mesh: MeshHandle): SectionResult {
    const raw = this.#section(section);
    const { index, generation } = this.#id(mesh);
    const r = call(() => this.#peek().section(raw, index, generation));
    const contours: SectionContour[] = [];
    let at = 4;
    for (let k = 0; k < r[3]; k++) {
      const n = 3 * r[at + 1];
      contours.push({ closed: r[at] === 1, points: Float32Array.from(r.subarray(at + 2, at + 2 + n)) });
      at += 2 + n;
    }
    const result: SectionResult = {
      status: r[0] === 0 ? "current" : "stale",
      contours,
      length: r[1],
      area: r[2],
    };
    if (r[at] === 1) {
      const np = 3 * r[at + 1];
      const positions = Float32Array.from(r.subarray(at + 2, at + 2 + np));
      at += 2 + np;
      const indices = Uint32Array.from(r.subarray(at + 1, at + 1 + 3 * r[at]));
      result.fill = { positions, indices };
    }
    return result;
  }

  /**
   * The closest pair between two meshes' contours in a section, measured in
   * the plane, or `null` when either has none or their results were cut with
   * different planes (call {@link Scene.updateSections} first).
   */
  sectionGap(section: SectionHandle, a: MeshHandle, b: MeshHandle): SectionGap | null {
    const raw = this.#section(section);
    const ia = this.#id(a);
    const ib = this.#id(b);
    const g = call(() =>
      this.#scene().section_gap(raw, ia.index, ia.generation, ib.index, ib.generation),
    );
    return g.length === 0 ? null : { distance: g[0], a: vec3At(g, 1), b: vec3At(g, 4) };
  }
}
