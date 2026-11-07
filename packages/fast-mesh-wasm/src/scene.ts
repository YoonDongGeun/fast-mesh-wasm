// The public `Scene` class: editable meshes with placements, zero-copy render
// views, per-frame change records and undo history. Wraps the raw `WasmScene`
// binding; every call into WebAssembly goes through `call` (error codes and
// panic poisoning).
import { checkRegisterOptions } from "./register.js";
import type { Bounds, Indices, Mat3, Placement, PlacementInput, Positions64, Quat, UpdateMode, Vec3 } from "./api.js";
import { decodeChanges, type Range } from "./changes.js";
import { call, ensureInitialized, FastMeshError, finalizer, release, wasmMemory } from "./core.js";
import { FieldHandle, MeshHandle, rotationMatrix, SectionHandle, targetsOf } from "./handles.js";
import { copyRule, encodeRule } from "./rule.js";
import type { SceneLike } from "./scene-like.js";
import {
  brushPick,
  buildTopology,
  checkSelection,
  checkSpec,
  componentOf,
  growIds,
  invertMatrix4,
  lassoPick,
  shrinkIds,
  transformPoint,
  type IsolateReport,
  type SelectSpec,
  type VertexSelection,
} from "./selection.js";
import { meshViews, type SceneMeshViews, ViewGuard } from "./views.js";
import { WasmScene } from "./wasm/fast_mesh_wasm.js";

export type { Range } from "./changes.js";
export type { MeshViews, SceneMeshViews } from "./views.js";
export type {
  BrushSelect,
  ComponentSelect,
  IsolateReport,
  LassoSelect,
  SelectSpec,
  VertexSelection,
} from "./selection.js";
// Wasm-free, so the Worker client can use them without the wasm glue.
export { FieldHandle, MeshHandle, rotationMatrix, SectionHandle } from "./handles.js";

/** The id behind a handle and the scene it belongs to. Kept here, not on the handle. */
interface HandleId {
  scene: Scene;
  index: number;
  generation: number;
}

/**
 * The `WasmScene` class new scenes wrap: the single-threaded glue's, unless
 * the Worker entry started the threaded glue ({@link useWasmScene}).
 */
let RawScene: typeof WasmScene = WasmScene;

/**
 * Internal (the Worker entry): makes later scenes from `ctor`, the
 * `WasmScene` of the glue it initialised (the threaded one on cross-origin
 * isolated pages). Pair it with `setMemory` for that glue's memory.
 */
export function useWasmScene(ctor: typeof WasmScene): void {
  RawScene = ctor;
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
  /**
   * Vertex slots. Only on a record of a mesh that has UVs or just lost them:
   * `"all"` with `views(mesh).uvs()` reading `null` means an operation dropped
   * them, and `"all"` after `undo` that they came back.
   */
  uvs?: Range[] | "all";
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
 * An `endStep()` lands a pending coalesced pose in the closing step first; a
 * `cancelStep()` drops it with the step.
 */
export interface DragSession {
  /**
   * Sets the mesh's placement. Throws `INVALID_ARGUMENT` once the drag is over.
   *
   * Each move recomputes the eager fields that depend on the mesh, so
   * coalesce pointer events to one move per animation frame (keep the latest
   * pose and apply it before {@link Scene.commitFrame}), or start the drag
   * with {@link DragOptions.coalesce}.
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

/** Options of {@link Scene.drag}. */
export interface DragOptions {
  /**
   * Coalesce moves: each {@link DragSession.move} only remembers its pose,
   * and the latest one is applied once at the next {@link Scene.commitFrame}
   * (or {@link DragSession.end}). For pointer events at 120–250 Hz: without
   * it, every move recomputes the eager fields. Until the frame,
   * {@link Scene.placement} and {@link Scene.matrix4} show the last applied
   * pose. `WorkerScene` drags always coalesce. Defaults to `false`.
   */
  coalesce?: boolean;
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
  /**
   * Texture coordinates: 2 finite values per vertex, stored as `f32` and read
   * back with `views(mesh).uvs()`. Without them the mesh has no UV channel and
   * pays nothing. Every operation keeps them aligned or drops them, undoably;
   * see `docs/uv-channel.md`. Weld keeps seams apart.
   */
  uvs?: Float32Array | readonly number[];
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
  /**
   * The mesh the distances are measured to, or several (at least one, all
   * different, not `on`): a vertex then holds the smallest signed distance
   * over them (the nearest surface, or the deepest penetration), and
   * {@link Scene.fieldNearest} tells which one it came from.
   */
  to: MeshHandle | readonly MeshHandle[];
  /** Largest distance computed (inclusive); farther vertices hold `Infinity`. May be `Infinity`. */
  maxDistance: number;
  /** Defaults to `"auto"`. */
  sign?: SignMode;
}

/**
 * A per-vertex wall-thickness field to add with
 * {@link Scene.addThicknessField}: the distance to the opposite surface
 * along each vertex's smooth normal, in world units (the placement's scale
 * applied, so scaled meshes read scaled).
 *
 * Vertices whose rays escape in both directions (open sheets, the rims of
 * open meshes), dead slots and unused vertices hold NaN; hits beyond
 * `maxDistance` hold `Infinity`. Colour it like any field: NaN takes the
 * rule's `nan` colour, `Infinity` follows `above`.
 */
export interface ThicknessFieldDesc {
  /** Unique among the fields on `on`. */
  name: string;
  /** The mesh whose vertices carry the values. */
  on: MeshHandle;
  /** Largest thickness kept (inclusive); thicker vertices hold `Infinity`. May be `Infinity`. */
  maxDistance: number;
}

/**
 * A per-vertex curvature field to add with
 * {@link Scene.addCurvatureField}: the signed mean curvature at each vertex,
 * in 1/world-units (positive on domes, about `1 / r` on a sphere of radius
 * `r` with outward normals; negative in bowls and dents; zero on flat
 * spots; a saddle takes the sign of its dominant bend).
 *
 * Dead slots and vertices no triangle uses hold NaN; colour it like any
 * field: NaN takes the rule's `nan` colour.
 */
export interface CurvatureFieldDesc {
  /** Unique among the fields on `on`. */
  name: string;
  /** The mesh whose vertices carry the values. */
  on: MeshHandle;
}

/** The state of a field, from {@link Scene.fieldStatus}. */
export interface FieldStatus {
  /** True when the values match the meshes and placements as they are now. */
  current: boolean;
  /**
   * What the latest sign data builds found out about the target meshes,
   * together: `closed` and `consistentOrientation` when every target is, the
   * counts summed, and the first target's `mode`. See
   * {@link Scene.fieldTargetStatus} for each target.
   */
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
  /**
   * World xyz, 3 floats per point. A copy. Where the plane passes exactly
   * through a vertex, the contour repeats that point (a zero-length segment).
   */
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
  | { op: "removeVertices"; vertices: Uint32Array | readonly number[] | readonly EditRef[] }
  /**
   * Sets the UVs of vertices (2 values each; undoable like the rest of the
   * edit), `addVertices` ones included. `addVertices` alone gives a mesh with
   * UVs the UV `(0, 0)`, and `setPositions` leaves them as they were.
   *
   * @throws `INVALID_ARGUMENT` on a mesh without UVs.
   */
  | {
      op: "setUvs";
      vertices: Uint32Array | readonly (number | EditRef)[];
      uvs: Float32Array | readonly number[];
    };

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
 * Scalar measurements of a mesh, from {@link Scene.measure}. The report is
 * in world coordinates (the placement applied), so scaled meshes measure
 * scaled. Thickness (#116) and curvature (#117) build on this.
 */
export interface MeasureResult {
  /** The total triangle area; always defined. */
  area: number;
  /**
   * The enclosed volume, or `null` when the mesh is not watertight (open
   * or non-manifold), with `volumeReason` saying why. The volume is the
   * absolute signed sum, so inconsistent winding does not fail the call:
   * it only shows up in `orientationConflicts`, while partly cancelling
   * the number. Require `orientationConflicts === 0` for a strict reading.
   */
  volume: number | null;
  /** No boundary loop or open chain and no non-manifold edge. */
  closed: boolean;
  /** Why `volume` is `null`; `null` exactly when `closed`. */
  volumeReason: "open" | "non-manifold" | null;
  /** The boundary loops and open chains, as `analyze` reports them. */
  boundaryLoops: number;
  /** How many edges have three or more triangles. */
  nonManifoldEdges: number;
  /** Interior edges used in the same direction by both their triangles. */
  orientationConflicts: number;
}

/**
 * Options of {@link Scene.analyze}. Distances are in mesh units; with a
 * rigid placement local and world distances agree, with a scaled one local
 * distances are in unscaled units.
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
      /** Reject a minimum-weight triangulation past this dihedral, in radians. */
      maxDihedral?: number;
      /** Stop refining a patch once it has this many new vertices. */
      maxRefineVertices?: number;
    }
  | {
      op: "inflate";
      /** Brush center in local mesh coordinates, as in `views(mesh).positions()`. */
      center: Readonly<Vec3>;
      radius: number;
      falloff: "linear" | "smooth";
      /** Displacement scale; negative pulls instead of pushing. */
      strength: number;
    }
  | {
      op: "smooth";
      /** Brush center in local mesh coordinates, as in `views(mesh).positions()`. */
      center: Readonly<Vec3>;
      radius: number;
      falloff: "linear" | "smooth";
      /** Displacement scale; the per-vertex weight is `strength * falloff`. */
      strength: number;
      /** How many Taubin λ/μ cycles to run; 0 moves nothing. */
      cycles: number;
    }
  | {
      op: "remesh";
      /** Region center in local mesh coordinates, as in `views(mesh).positions()`. */
      center: Readonly<Vec3>;
      radius: number;
      /** Target edge length; unset means the mesh-wide mean edge length. */
      targetLength?: number;
      /** Maximum split/collapse/flip rounds; unset means the default (10). */
      maxRounds?: number;
      /** Split long edges only (no collapses, no flips); unset means a full pass. */
      splitOnly?: boolean;
    }
  | {
      op: "bridgeLoops";
      /** The two loops to connect, from `analyze(mesh).loops`. Order is insignificant. */
      loops: readonly [LoopId, LoopId];
      /** Maximum vertices per loop; unset means the default (128). */
      maxEdges?: number;
    }
  | {
      op: "grab";
      /** Brush center in local mesh coordinates, as in `views(mesh).positions()`. */
      center: Readonly<Vec3>;
      radius: number;
      falloff: "linear" | "smooth";
      /** Displacement scale; the per-vertex offset is `strength * falloff * direction`. */
      strength: number;
      /** The drag vector in local coordinates; zero moves nothing. */
      direction: Readonly<Vec3>;
    }
  | {
      op: "relax";
      /** Brush center in local mesh coordinates, as in `views(mesh).positions()`. */
      center: Readonly<Vec3>;
      radius: number;
      falloff: "linear" | "smooth";
      /** Displacement scale; the per-vertex weight is `strength * falloff`. */
      strength: number;
      /** How many plain Laplacian passes to run; 0 moves nothing. */
      cycles: number;
    }
  | {
      op: "offset";
      /**
       * How far to move the surface along its smooth normals, in mesh units
       * (equal to world units). Positive follows the normals (outward on an
       * outward-wound closed mesh); negative sinks inward. Must be finite;
       * 0 moves nothing.
       */
      distance: number;
    }
  | {
      op: "shell";
      /**
       * Wall thickness in mesh units (equal to world units): the input
       * surface stays and gains a reversed copy one thickness against its
       * smooth normals, with every boundary rim stitched shut. Must be
       * positive and finite.
       */
      thickness: number;
    }
  | {
      op: "simplify";
      /**
       * Stop once this many triangles are alive; must be at least 1. A target
       * at or above the live count leaves the mesh alone.
       */
      targetTriangles: number;
      /** Keep boundary vertices in place; defaults to `true`. */
      preserveBoundary?: boolean;
    }
  | {
      op: "subdivide";
      /**
       * Subdivision scheme; only `"loop"` exists. Defaults to `"loop"`.
       */
      method?: "loop";
      /**
       * How many times to split every triangle; must be an integer of at
       * least 1. Each level quadruples the triangle count. Defaults to `1`.
       */
      levels?: number;
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
  /** Whether refinement stopped early on its vertex budget or rounds. */
  refineCapped: boolean;
}

/** What `fillHoles` did, each list in loop order. */
export interface FillReport {
  filled: FilledLoop[];
  /** Loops left out: beyond a bound, or unclean (or open). */
  skipped: { id: LoopId; reason: "bounds" | "unclean" | "uvSeam" }[];
  /** Selected loops that could not be filled; they stay open. */
  failed: { id: LoopId; reason: "projectionFolds" | "noValidTriangulation" | "solver" }[];
}

/** What `inflate` did. */
export interface InflateReport {
  moved: number;
  maxDisplacement: number;
}

/** What `smooth` did. */
export interface SmoothReport {
  moved: number;
  maxDisplacement: number;
}

/** What `remesh` did. */
export interface RemeshReport {
  splits: number;
  collapses: number;
  flips: number;
  rounds: number;
}

/** What `bridgeLoops` did. */
export interface BridgeReport {
  trianglesAdded: number;
}

/** What `grab` did. */
export interface GrabReport {
  moved: number;
  maxDisplacement: number;
}

/** What `relax` did. */
export interface RelaxReport {
  moved: number;
  maxDisplacement: number;
}

/** What `offset` did. */
export interface OffsetReport {
  moved: number;
  maxDisplacement: number;
  /** Degenerate triangles the cleanup pass removed. */
  degenerateRemoved: number;
  /** Vertices the cleanup pass removed. */
  verticesRemoved: number;
}

/** What `shell` did. */
export interface ShellReport {
  /** Copy vertices added. */
  verticesAdded: number;
  /** Copy and wall triangles added. */
  trianglesAdded: number;
  /** Boundary rims stitched to their copy. A closed mesh has none. */
  loopsClosed: number;
  /** Degenerate triangles the cleanup pass removed. */
  degenerateRemoved: number;
  /** Vertices the cleanup pass removed. */
  verticesRemoved: number;
}

/** What `simplify` did. */
export interface SimplifyReport {
  /** Edge collapses performed. */
  collapsed: number;
  /** Triangles removed (one per boundary collapse, two per interior one). */
  trianglesRemoved: number;
  /** Vertices removed (one per collapse). */
  verticesRemoved: number;
}

/** What `subdivide` did. */
export interface SubdivideReport {
  /** Vertices added, one per subdivided edge per level. */
  verticesAdded: number;
  /** Triangles added: three per subdivided triangle per level. */
  trianglesAdded: number;
}

/** Options of {@link Scene.join}. */
export interface JoinOptions {
  /** The loop on the target, from `analyze(target).loops`. */
  loopA: LoopId;
  /** The loop on the source, from `analyze(source).loops`. */
  loopB: LoopId;
  /** Maximum vertices per loop; unset means the default (128). */
  maxEdges?: number;
}

/** What {@link Scene.join} did. */
export interface JoinReport {
  /** Source vertices copied into the target. */
  verticesAdded: number;
  /** Source triangles copied into the target. */
  trianglesAdded: number;
  /** Bridge triangles between the two loops. */
  bridgeTrianglesAdded: number;
}

/** Options of {@link Scene.register}. */
export interface RegisterOptions {
  /** The most solver steps. 0 only measures the current placement. Default 50. */
  maxIterations?: number;
  /**
   * Stops when one step moves every point by at most this fraction of the
   * moving mesh's extent (the largest distance of its points from their
   * centroid). Default `1e-6`.
   */
  tolerance?: number;
  /**
   * The fraction of the moving mesh expected to lie over the fixed one, in
   * `(0, 1]`: each iteration uses that many of its points, closest first, so
   * the rest (the part beyond the overlap, and outliers) do not pull the
   * result. Set it a little below the true overlap for partial scans.
   * Default 1: every point counts, and nothing is rejected.
   */
  overlap?: number;
  /** Points farther than this from the fixed mesh (world units) are not used. Default: no limit. */
  maxDistance?: number;
  /**
   * The most points of the moving mesh to use, evenly spaced over its
   * triangles' centroids and its vertices. `rmse` is measured on these.
   * Default 100,000.
   */
  maxSamples?: number;
  /**
   * Whether a good result becomes the moving mesh's placement, as one undo
   * step: it must converge, with `rmse` within `maxRmse` and at least half of
   * the points `overlap` expects in range. Anything else is not applied,
   * whatever this says; {@link RegisterResult.notApplied} says why. Default
   * `true`.
   */
  apply?: boolean;
  /**
   * The largest `rmse` (world units) that may be applied, to keep a wrong local
   * minimum from moving the mesh. Default: 1% of the moving points' extent,
   * reported as {@link RegisterResult.maxRmse}. Scans noisier than that need
   * their own limit; `Infinity` turns the guard off.
   */
  maxRmse?: number;
}

/** Why {@link Scene.register} did not apply its result. */
export type RegisterNotApplied = "notRequested" | "notConverged" | "rmseAboveLimit" | "tooFewInliers";

/**
 * A rigid motion the data do not determine, in world coordinates: a point `p`
 * moves along `translation + rotation x (p - about)`, scaled so that the moving
 * mesh's points move by 1 (root mean square).
 */
export interface RegisterMotion {
  translation: Vec3;
  /** The rotation part, as an axis scaled by its rate. */
  rotation: Vec3;
  /** The point the rotation turns about: the moved points' centroid. */
  about: Vec3;
  /** The rmse of the other pose found along this motion, which fits as well. */
  alternativeRmse: number;
}

/**
 * Why {@link Scene.register} stopped:
 * - `"converged"`: a step fell below the tolerance, and restarting from displaced
 *   poses came back to the result, or only found worse fits: the data determine
 *   every rigid motion locally.
 * - `"maxIterations"`: the steps ran out first, with no motion found undetermined.
 * - `"underconstrained"`: the data do not determine the pose. A restart stayed
 *   displaced with a fit as good, as for a plane's slide, or a sphere's or a
 *   cylinder's spin, even when facets or noise let the iteration settle with a
 *   tiny `rmse`. {@link RegisterResult.undetermined} names the motions, and the
 *   pose along them is arbitrary.
 * - `"tooFewCorrespondences"`: fewer than six points have a usable closest
 *   point.
 * - `"diverged"`: a step took points out of the range the query can represent.
 */
export type RegisterStatus = "converged" | "maxIterations" | "underconstrained" | "tooFewCorrespondences" | "diverged";

/** What {@link Scene.register} found. */
export interface RegisterResult {
  /**
   * The moving mesh's aligned placement, with its scale unchanged: the
   * converged result, or the best estimate when it did not converge.
   */
  placement: Placement;
  /**
   * The root mean square distance (world units) from the used points to the
   * fixed mesh at `placement`; `Infinity` when no point has a closest point.
   */
  rmse: number;
  /** Solver steps taken. */
  iterations: number;
  /**
   * `status === "converged"`. ICP is local, so this says the iteration settled
   * and the data determine the result locally, not that it found the right
   * alignment: a far guess can settle in a wrong minimum. `maxRmse` guards
   * applying against that.
   */
  converged: boolean;
  /** Why iteration stopped. */
  status: RegisterStatus;
  /** Whether `placement` was set on the moving mesh. */
  applied: boolean;
  /** Why it was not; `null` exactly when `applied`. */
  notApplied: RegisterNotApplied | null;
  /** The `rmse` limit that applied (world units). */
  maxRmse: number;
  /** Points of the moving mesh sampled. */
  samples: number;
  /** Sampled points used at `placement`: the rest were trimmed or out of range. */
  inliers: number;
  /**
   * The smallest eigenvalue of the solver's normal equations over the largest
   * at the last step, in `[0, 1]`; 0 when no step ran. Geometry only, with no
   * reference to noise: a low value warns of a weak motion, but a high one does
   * not show that it is determined. That is `undetermined`.
   */
  conditioning: number;
  /** The motions the data do not determine; non-empty only when `"underconstrained"`. */
  undetermined: RegisterMotion[];
}

/** Options of {@link Scene.cutByPlane}. */
export interface CutOptions {
  /**
   * Close both openings with flat, unfaired patches. Defaults to `true`;
   * without it both parts keep the open rim the cut leaves.
   */
  cap?: boolean;
}

/** What {@link Scene.cutByPlane} did. */
export interface CutReport {
  /** The original handle, now holding the `+` side. */
  kept: MeshHandle;
  /** The new handle holding the `-` side. */
  removed: MeshHandle;
  /** Alive triangles of {@link CutReport.kept}, cap included. */
  keptTriangles: number;
  /** Alive triangles of {@link CutReport.removed}, cap included. */
  removedTriangles: number;
  /** Cap triangles added on both sides together. */
  capTriangles: number;
}

/** Which boolean for {@link Scene.boolean} to compute. */
export type BooleanOp = "union" | "subtract" | "intersect";

/**
 * The standard cleanup pipeline for {@link Scene.repair}: weld, remove
 * degenerate, orient, remove small components, fill holes, as one transaction.
 * The library owns the order; an option left out takes the library's default.
 */
export interface AutoRepairRequest {
  kind: "auto";
  /** Vertices at most this far apart are welded, in mesh units. Default `1e-5`. */
  weldTolerance?: number;
  /** Components with fewer triangles are removed; the largest always stays. Default `500`. */
  minTriangles?: number;
  /** Fill only holes with at most this many edges; `Infinity` fills any size. Default `64`. */
  maxEdges?: number;
  /** Refine and fair each fill so it follows the surrounding curvature. Default `true`. */
  fair?: boolean;
}

/**
 * What an {@link AutoRepairRequest} did, one report per stage, keyed by the
 * stage's {@link RepairOp} name. The keys iterate in the order the stages ran.
 */
export interface AutoRepairReport {
  weld: WeldReport;
  removeDegenerate: DegenerateReport;
  orient: OrientReport;
  removeSmallComponents: RemoveReport;
  fillHoles: FillReport;
}

/** The report of a {@link RepairOp}. */
export type RepairReport =
  | WeldReport
  | DegenerateReport
  | RemoveReport
  | OrientReport
  | FillReport
  | InflateReport
  | SmoothReport
  | RemeshReport
  | BridgeReport
  | GrabReport
  | RelaxReport
  | OffsetReport
  | ShellReport
  | SimplifyReport
  | SubdivideReport;

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
          : O extends { op: "inflate" }
            ? InflateReport
            : O extends { op: "smooth" }
              ? SmoothReport
              : O extends { op: "remesh" }
                ? RemeshReport
                : O extends { op: "bridgeLoops" }
                  ? BridgeReport
                  : O extends { op: "grab" }
                    ? GrabReport
                    : O extends { op: "relax" }
                      ? RelaxReport
                      : O extends { op: "offset" }
                        ? OffsetReport
                        : O extends { op: "shell" }
                          ? ShellReport
                          : O extends { op: "simplify" }
                            ? SimplifyReport
                            : O extends { op: "subdivide" }
                              ? SubdivideReport
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
  /** Applies a coalesced drag's pending pose, if it has one. */
  applyPending(): void;
}

const MODES: Record<UpdateMode, number> = { eager: 0, lazy: 1, manual: 2 };
const MODE_NAMES: readonly UpdateMode[] = ["eager", "lazy", "manual"];

function modeCode(mode: UpdateMode): number {
  if (!Object.hasOwn(MODES, mode)) {
    throw invalid('mode must be "eager", "lazy" or "manual"');
  }
  return MODES[mode];
}

const CULLS = { none: 0, back: 1, front: 2 } as const;
const SIGNS: Record<SignMode, number> = { auto: 0, pseudonormal: 1, winding: 2 };

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

function invalid(message: string): FastMeshError {
  return new FastMeshError("INVALID_ARGUMENT", message);
}

/**
 * The mean stretch of a column-major 4x4's linear part: the cube root of
 * its determinant. World-space brush radii divide by it, exact for rigid
 * placements and a documented approximation under non-uniform scale.
 */
function meanStretch(m: Float64Array): number {
  const det =
    m[0] * (m[5] * m[10] - m[6] * m[9]) - m[4] * (m[1] * m[10] - m[2] * m[9]) + m[8] * (m[1] * m[6] - m[2] * m[5]);
  const s = Math.cbrt(Math.abs(det));
  if (s === 0 || !Number.isFinite(s)) {
    throw invalid("the mesh placement is not invertible");
  }
  return s;
}

const U32_LIMIT = 2 ** 32;
const EDIT_CODES = {
  setPositions: 0,
  addVertices: 1,
  addTriangles: 2,
  removeTriangles: 3,
  removeVertices: 4,
  setUvs: 5,
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
      case "setUvs": {
        encodeIds(o.vertices, `${what}: vertices`, words);
        const uvs = Float64Array.from(o.uvs ?? []);
        if (uvs.length !== 2 * o.vertices.length) {
          throw invalid(`${what}: uvs need 2 values per vertex`);
        }
        values.push(uvs);
        break;
      }
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

const REPAIR_CODES = {
  weld: 0,
  removeDegenerate: 1,
  removeSmallComponents: 2,
  orient: 3,
  fillHoles: 4,
  inflate: 5,
  smooth: 6,
  remesh: 7,
  bridgeLoops: 8,
  grab: 9,
  relax: 10,
  offset: 11,
  shell: 12,
  simplify: 13,
  subdivide: 14,
} as const;

/** The op code of an {@link AutoRepairRequest}; see `crate::edit` in the Rust crate. */
const AUTO_REPAIR_CODE = 15;

const FALLOFF_CODES = { linear: 0, smooth: 1 } as const;
const SUBDIVIDE_METHOD_CODES = { loop: 0 } as const;

/** An optional number for the binding: NaN when unset; an explicit NaN is rejected. */
/** The `status` codes of `WasmScene.register`, in order. */
const REGISTER_STATUSES: readonly RegisterStatus[] = [
  "converged",
  "maxIterations",
  "underconstrained",
  "tooFewCorrespondences",
  "diverged",
];

/** The `not applied` codes of `WasmScene.register`, in order. */
const REGISTER_NOT_APPLIED: readonly (RegisterNotApplied | null)[] = [
  null,
  "notRequested",
  "notConverged",
  "rmseAboveLimit",
  "tooFewInliers",
];

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

/** A brush center as 3 values; anything else is `INVALID_ARGUMENT`. */
function centerOf(center: Readonly<Vec3>, values: number[]): void {
  const c = center as readonly number[] | undefined;
  if (c?.length !== 3 || ![c[0], c[1], c[2]].every(Number.isFinite)) {
    throw invalid("center must be 3 finite numbers");
  }
  values.push(c[0], c[1], c[2]);
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

function isAutoRepair(op: RepairOp | readonly RepairOp[] | AutoRepairRequest): op is AutoRepairRequest {
  return (op as AutoRepairRequest | undefined)?.kind === "auto";
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
          optional(o.maxDihedral, "maxDihedral"),
          optional(o.maxRefineVertices, "maxRefineVertices"),
        );
        break;
      }
      case "inflate":
        centerOf(o.center, values);
        values.push(
          optional(o.radius, "radius"),
          codeIn(FALLOFF_CODES, o.falloff, "falloff"),
          optional(o.strength, "strength"),
        );
        break;
      case "smooth":
      case "relax":
        centerOf(o.center, values);
        values.push(
          optional(o.radius, "radius"),
          codeIn(FALLOFF_CODES, o.falloff, "falloff"),
          optional(o.strength, "strength"),
          optional(o.cycles, "cycles"),
        );
        break;
      case "remesh":
        centerOf(o.center, values);
        values.push(
          optional(o.radius, "radius"),
          optional(o.targetLength, "targetLength"),
          optional(o.maxRounds, "maxRounds"),
          flagOf(o.splitOnly, false),
        );
        break;
      case "bridgeLoops": {
        if (!Array.isArray(o.loops) || o.loops.length !== 2) {
          throw invalid("loops must be an array of two loop ids");
        }
        values.push(optional(o.maxEdges, "maxEdges"));
        for (const id of o.loops) {
          encodeLoopId(id, loops);
        }
        break;
      }
      case "grab": {
        centerOf(o.center, values);
        values.push(
          optional(o.radius, "radius"),
          codeIn(FALLOFF_CODES, o.falloff, "falloff"),
          optional(o.strength, "strength"),
        );
        const d = o.direction as readonly number[] | undefined;
        if (d?.length !== 3 || ![d[0], d[1], d[2]].every(Number.isFinite)) {
          throw invalid("direction must be 3 finite numbers");
        }
        values.push(d[0], d[1], d[2]);
        break;
      }
      case "offset":
        values.push(optional(o.distance, "distance"));
        break;
      case "shell":
        values.push(optional(o.thickness, "thickness"));
        break;
      case "simplify":
        values.push(
          optional(o.targetTriangles, "targetTriangles"),
          flagOf(o.preserveBoundary, true),
        );
        break;
      case "subdivide":
        values.push(
          o.method === undefined ? Number.NaN : codeIn(SUBDIVIDE_METHOD_CODES, o.method, "method"),
          optional(o.levels, "levels"),
        );
        break;
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

const SKIP_REASONS = ["bounds", "unclean", "uvSeam"] as const;
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
          refineCapped: r.flag(),
        })),
        skipped: r.many(() => ({ id: r.loopId(), reason: SKIP_REASONS[r.next()] })),
        failed: r.many(() => ({ id: r.loopId(), reason: FAIL_REASONS[r.next()] })),
      };
    case "inflate":
      return { moved: r.next(), maxDisplacement: r.next() };
    case "smooth":
    case "relax":
      return { moved: r.next(), maxDisplacement: r.next() };
    case "remesh":
      return { splits: r.next(), collapses: r.next(), flips: r.next(), rounds: r.next() };
    case "bridgeLoops":
      return { trianglesAdded: r.next() };
    case "grab":
      return { moved: r.next(), maxDisplacement: r.next() };
    case "offset":
      return {
        moved: r.next(),
        maxDisplacement: r.next(),
        degenerateRemoved: r.next(),
        verticesRemoved: r.next(),
      };
    case "shell":
      return {
        verticesAdded: r.next(),
        trianglesAdded: r.next(),
        loopsClosed: r.next(),
        degenerateRemoved: r.next(),
        verticesRemoved: r.next(),
      };
    case "simplify":
      return { collapsed: r.next(), trianglesRemoved: r.next(), verticesRemoved: r.next() };
    case "subdivide":
      return { verticesAdded: r.next(), trianglesAdded: r.next() };
  }
}

/** An auto repair's result, stage by stage in the order they ran; see `crate::edit` in the Rust crate. */
function decodeAutoReport(r: Reader): AutoRepairReport {
  const ops = Object.keys(REPAIR_CODES) as RepairOp["op"][];
  const stages: Partial<Record<RepairOp["op"], RepairReport>> = {};
  while (r.at < r.data.length) {
    const op = ops[r.next()];
    stages[op] = decodeReport(op, r);
  }
  return stages as unknown as AutoRepairReport;
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
 * - `step(points)`: advances the due fields by a budget of `points`
 *   (values, or about 32 times as many sign data elements, or 16 times as
 *   many triangle pairs of a near search), opening a unit of every due
 *   field first, or continuing the open unit. The call that finishes the
 *   last applies the unit and returns `"done"`, exactly as a one-shot update
 *   at the state it opened at; `"idle"`: nothing was due. Any other scene
 *   call that may change the scene drops an open unit. A field error throws
 *   and ends the unit (the failing fields stay due).
 * - `addDeferred(desc)`: adds a field like `addDistanceField`, computing
 *   nothing yet (no sign data either): it is due until stepped.
 * - `stepField(field, points)`: `step` for one field alone, opening a unit
 *   over it (dropping another open unit) or continuing its own; `"idle"`:
 *   the field is current.
 * - `withdraw(field)`: removes a field that was never computed, reporting
 *   nothing; `false` (nothing done) when it was computed or colours its mesh.
 * - `abandon()`: drops the open unit; its fields stay due.
 * - `meshes()`: the meshes of the open unit's fields, each once.
 */
interface FieldSteps {
  setDefer(on: boolean): void;
  step(points: number): "idle" | "pending" | "done";
  addDeferred(desc: DistanceFieldDesc): FieldHandle;
  stepField(field: FieldHandle, points: number): "idle" | "pending" | "done";
  withdraw(field: FieldHandle): boolean;
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
 * Meshes added in steps (the store crate's `MeshBuilder`, then the scene
 * crate's `MeshBuild`), for the Worker engine only: not part of the public
 * API.
 *
 * - `beginUpload(vertices, triangles, options)`: reserves a mesh outside
 *   the scene, as `addMesh` would add it; returns the build's token. The
 *   scene is unchanged until `finish`.
 * - `pushPositions(token, chunk)` / `pushIndices(token, chunk)` /
 *   `pushUvs(token, chunk)`: append a chunk of the inputs; the chunks must
 *   add up to the declared counts (UVs are optional: none, or 2 per vertex).
 * - `step(token, budget)`: builds about `budget` more units: first the
 *   store (vertices or triangles), then the BVH (a triangle's box, or a
 *   triangle visited per tree level), then the vertex normals when they
 *   are on (4 triangles or 16 vertices a unit); `true` once built. The
 *   upload must be complete first. A failed build throws and is dropped.
 * - `buffers(token)`: views of the build's render buffers (normals `null`
 *   until built or when off): what `views(mesh)` will show once it is
 *   added, until the mesh changes. Valid until the next call that may
 *   allocate, like any view. Throws until the store is built.
 * - `finish(token)`: adds the mesh (finishing its store, BVH and normals
 *   first), exactly as `addMesh` with the same inputs would, and returns
 *   its handle.
 * - `stamp(mesh)`: the mesh's content stamp, which goes up whenever its
 *   positions, triangles or normals change (undo and redo included).
 * - `drop(token)`: forgets a build; nothing is added.
 */
interface MeshAdds {
  beginUpload(vertices: number, triangles: number, options?: AddMeshOptions): number;
  pushPositions(token: number, chunk: Float64Array): void;
  pushIndices(token: number, chunk: Uint32Array): void;
  pushUvs(token: number, chunk: Float32Array): void;
  step(token: number, budget: number): boolean;
  buffers(token: number): { positions: Float32Array; normals: Float32Array | null; indices: Uint32Array };
  finish(token: number): MeshHandle;
  stamp(mesh: MeshHandle): number;
  drop(token: number): void;
}

/** Set by `Scene`'s static block. */
let meshAddsOf: (scene: Scene) => MeshAdds;

/** @internal The stepped mesh adds of `scene` (see {@link MeshAdds}); for the Worker engine. */
export function meshAdds(scene: Scene): MeshAdds {
  return meshAddsOf(scene);
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
  /** The rule each coloured mesh was last given by `setColoring`. */
  readonly #rules = new WeakMap<MeshHandle, ColorRule>();
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
    this.#raw = call(() => new RawScene());
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
   * @throws `INVALID_ARGUMENT` for bad lengths or options (`uvs` not 2 finite
   * values per vertex included), `INVALID_INDEX` for an index out of range.
   */
  addMesh(positions: Positions64, indices: Indices, options: AddMeshOptions = {}): MeshHandle {
    const s = this.#scene();
    const mode = modeCode(options.mode ?? "eager");
    const origin = options.origin === undefined ? undefined : Float64Array.from(options.origin);
    const p = toFloat64(positions);
    const i = toUint32(indices);
    const normals = options.normals ?? true;
    if (options.uvs === undefined) {
      const id = call(() => s.add_mesh(p, i, origin, mode, normals));
      return this.#meshHandle(id[0], id[1]);
    }
    const uvs = options.uvs instanceof Float32Array ? options.uvs : Float32Array.from(options.uvs);
    const id = call(() => s.add_mesh_with_uvs(p, i, uvs, origin, mode, normals));
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
   * Joins `source` into `target`: copies the source's geometry into the
   * target (keeping its world shape), bridges `options.loopA` to the copied
   * `options.loopB`, and removes the source. The copy and the bridge undo as
   * one step; the removal is permanent. A drag of the source ends first,
   * recording its moves as one step, so they survive a failed join.
   *
   * @throws `UNKNOWN_MESH` for a removed handle; `INVALID_ARGUMENT` for one
   * mesh twice or loops that cannot bridge; `STALE_LOOPS` for a loop that is
   * not there. A failed join changes nothing.
   */
  join(target: MeshHandle, source: MeshHandle, options: JoinOptions): JoinReport {
    const ta = this.#id(target);
    const sb = this.#id(source);
    const loops: number[] = [];
    encodeLoopId(options.loopA, loops);
    encodeLoopId(options.loopB, loops);
    const maxEdges = optional(options.maxEdges, "maxEdges");
    // A drag of the source ends first, recording its moves as one step.
    if (this.#drag?.mesh === source) {
      this.#drag.end();
    }
    const r = call(() =>
      this.#scene().join(ta.index, ta.generation, sb.index, sb.generation, Uint32Array.from(loops), maxEdges),
    );
    const key = `${sb.index}:${sb.generation}`;
    this.#meshes.delete(key);
    this.#removed.set(key, source);
    return { verticesAdded: r[0], trianglesAdded: r[1], bridgeTrianglesAdded: r[2] };
  }

  /**
   * Cuts `mesh` with a world `plane`: the `+` side
   * (`signedDistance >= 0`) stays on the mesh and the `-` side is added as
   * a new mesh with the same placement, so the world shape is kept.
   * Cutting a closed mesh with `cap` (the default) yields two closed
   * meshes, capped with flat, unfaired patches.
   *
   * The cut undoes in one step; the added mesh stays, like the removed
   * source of {@link Scene.join}.
   *
   * @throws `UNKNOWN_MESH` for a removed handle; `INVALID_ARGUMENT` when
   * the plane does not cut the mesh or a rim cannot be capped. A failed cut
   * changes nothing.
   */
  cutByPlane(mesh: MeshHandle, plane: SectionPlane, options?: CutOptions): CutReport {
    const { index, generation } = this.#id(mesh);
    const p = this.#plane(plane);
    const r = call(() => this.#scene().cut_by_plane(index, generation, p, options?.cap ?? true));
    return {
      kept: this.#meshHandle(r[0], r[1]),
      removed: this.#meshHandle(r[2], r[3]),
      keptTriangles: r[4],
      removedTriangles: r[5],
      capTriangles: r[6],
    };
  }

  /**
   * Combines `source` into `target` in the world: the union, subtraction
   * (`target` minus `source`) or intersection. Returns `target`, now
   * holding the result with its placement; `source` is unchanged.
   *
   * Both meshes must be closed and manifold. The boolean undoes in one
   * step. Combining a mesh with itself is a no-op for `"union"` and
   * `"intersect"`, and clears it for `"subtract"`.
   *
   * @throws `UNKNOWN_MESH` for a removed handle; `INVALID_ARGUMENT` for an
   * input that is not closed, manifold and consistently wound. A failed
   * boolean changes nothing.
   */
  boolean(target: MeshHandle, source: MeshHandle, op: BooleanOp): MeshHandle {
    const ta = this.#id(target);
    const sb = this.#id(source);
    const code = codeIn({ union: 0, subtract: 1, intersect: 2 }, op, "op");
    const r = call(() =>
      this.#scene().boolean(ta.index, ta.generation, sb.index, sb.generation, code),
    );
    return this.#meshHandle(r[0], r[1]);
  }

  /**
   * Rigidly aligns the mesh `moving` to the mesh `fixed` by point-to-plane ICP,
   * with the closest points found through `fixed`'s BVH and trimmed by
   * `options.overlap`. A drag of either mesh ends first, recording its moves
   * as one step, once the arguments have been checked.
   *
   * **The initial guess is the current placements.** ICP finds the nearest
   * fit, so put the meshes roughly in place first with `setPlacement`.
   *
   * **Applying.** By default a good result becomes `moving`'s placement as one
   * undo step (`applied` in the result): converged, with `rmse` within `maxRmse`
   * and enough points in range. Otherwise nothing moves, and `notApplied` says
   * why; `placement` still holds the estimate, for `setPlacement` if it will do.
   * `apply: false` only returns it.
   *
   * **Undetermined poses.** A plane's slide and a sphere's or a cylinder's spin
   * are not fixed by the data, though facets or noise can let the iteration
   * settle with a tiny `rmse`. These report `"underconstrained"` and list the
   * motions in `undetermined`. So do coarse regular meshes with exact fits close
   * by: a 16-sided sphere fits every 22.5 degrees. The check is local, so a
   * cube's 90-degree symmetry does not count. It adds a bounded cost: up to 12
   * restarts of up to 12 steps on at most 10,000 points.
   *
   * **Scales.** Only rotation and translation are solved. `moving`'s scale is
   * applied to its points and kept. `fixed`'s scale, uniform or not, is
   * followed exactly on a copy baked through it, so each such call costs a BVH
   * build. All distances are in world units.
   *
   * @throws `UNKNOWN_MESH` for a removed handle; `INVALID_ARGUMENT` for one
   * mesh twice, a mesh without triangles or an invalid option; `STALE_BVH` for
   * a stale Manual `fixed`. Bad options and one mesh twice change nothing,
   * and a drag keeps running. The other errors come after a drag has ended, and
   * change nothing else.
   */
  register(moving: MeshHandle, fixed: MeshHandle, options: RegisterOptions = {}): RegisterResult {
    const m = this.#id(moving);
    const f = this.#id(fixed);
    checkRegisterOptions(moving === fixed, options);
    const maxIterations = optional(options.maxIterations, "maxIterations");
    const tolerance = optional(options.tolerance, "tolerance");
    const overlap = optional(options.overlap, "overlap");
    const maxDistance = optional(options.maxDistance, "maxDistance");
    const maxSamples = optional(options.maxSamples, "maxSamples");
    const maxRmse = optional(options.maxRmse, "maxRmse");
    const apply = options.apply ?? true;
    if (this.#drag?.mesh === moving || this.#drag?.mesh === fixed) {
      this.#drag.end();
    }
    const r = call(() =>
      this.#scene().register(
        m.index,
        m.generation,
        f.index,
        f.generation,
        maxIterations,
        tolerance,
        overlap,
        maxDistance,
        maxSamples,
        apply,
        maxRmse,
      ),
    );
    const status = REGISTER_STATUSES[r[17]];
    const undetermined: RegisterMotion[] = [];
    for (let at = 24; at < r.length; at += 10) {
      undetermined.push({
        translation: vec3At(r, at),
        rotation: vec3At(r, at + 3),
        about: vec3At(r, at + 6),
        alternativeRmse: r[at + 9],
      });
    }
    return {
      placement: {
        rotation: Array.from(r.subarray(0, 9)),
        translation: [r[9], r[10], r[11]],
        scale: [r[12], r[13], r[14]],
      },
      rmse: r[15],
      iterations: r[16],
      converged: status === "converged",
      status,
      applied: r[18] === 1,
      samples: r[19],
      inliers: r[20],
      conditioning: r[21],
      notApplied: REGISTER_NOT_APPLIED[r[22]],
      maxRmse: r[23],
      undetermined,
    };
  }

  /**
   * Sets a mesh's placement: world = rotation · (scale · local) + translation,
   * where local is relative to the mesh origin. A quaternion is normalised
   * and converted in double precision. The scale defaults to `[1, 1, 1]`
   * (rigid) and must be 3 finite positive numbers.
   *
   * @throws `INVALID_ARGUMENT` when the rotation is not a proper rotation, the
   * scale is not 3 finite positive numbers, or a quaternion is zero.
   */
  setPlacement(mesh: MeshHandle, placement: PlacementInput): void {
    const { index, generation } = this.#id(mesh);
    const r = Float64Array.from(rotationMatrix(placement.rotation));
    const t = Float64Array.from(placement.translation);
    const s = placement.scale === undefined ? undefined : Float64Array.from(placement.scale);
    call(() => this.#scene().set_placement(index, generation, r, t, s));
  }

  /** A mesh's placement, with the rotation as a row-major matrix. */
  placement(mesh: MeshHandle): Placement {
    const { index, generation } = this.#id(mesh);
    const p = call(() => this.#peek().placement(index, generation));
    return {
      rotation: Array.from(p.subarray(0, 9)),
      translation: [p[9], p[10], p[11]],
      scale: [p[12], p[13], p[14]],
    };
  }

  /**
   * The column-major world matrix of the local render coordinates,
   * `placement · translate(origin)`, ready for renderers (convert to `f32`
   * yourself). The rotation columns carry the scale; normals need the
   * inverse-transpose (three.js' normal matrix does this from the matrix).
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

  /** When the mesh's derived data follows its edits; see {@link UpdateMode}. */
  mode(mesh: MeshHandle): UpdateMode {
    const { index, generation } = this.#id(mesh);
    const code = call(() => this.#peek().mode(index, generation));
    return MODE_NAMES[code] ?? "eager";
  }

  /**
   * The box around the vertices the live triangles use, in the mesh's local
   * render coordinates ({@link Scene.matrix4} maps them to the world). Dead
   * slots keep stale positions, so they are excluded. `null` without live
   * triangles.
   */
  bounds(mesh: MeshHandle): Bounds | null {
    const { index, generation } = this.#id(mesh);
    const b = call(() => this.#peek().bounds(index, generation));
    if (b.length === 0) {
      return null;
    }
    const min: Vec3 = [b[0], b[1], b[2]];
    const max: Vec3 = [b[3], b[4], b[5]];
    return {
      min,
      max,
      center: [(min[0] + max[0]) / 2, (min[1] + max[1]) / 2, (min[2] + max[2]) / 2],
      size: [max[0] - min[0], max[1] - min[1], max[2] - min[2]],
    };
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
  views(mesh: MeshHandle): SceneMeshViews {
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
    // A coalesced drag's latest pose lands first, so the frame reports it.
    this.#drag?.applyPending();
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
        ...(c.uvs === undefined ? {} : { uvs: c.uvs }),
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
    const drag = this.#drag;
    if (drag !== undefined && this.#stepDepth > 0 && this.#stepDepth <= drag.depth) {
      // The caller is closing the drag's step with a coalesced pose still
      // pending: land it in the closing step, as a non-coalesced move would.
      drag.applyPending();
    }
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
   * one move per animation frame, with the latest pointer pose, or pass
   * `{ coalesce: true }` to have the drag do that for you.
   */
  drag(mesh: MeshHandle, options: DragOptions = {}): DragSession {
    this.#id(mesh);
    if (this.#drag !== undefined) {
      throw invalid("a drag is already active; end or cancel it first");
    }
    const coalesce = options.coalesce ?? false;
    const start = this.placement(mesh);
    const outermost = this.#stepDepth === 0;
    this.beginStep();
    const depth = this.#stepDepth;
    let active = true;
    /** The latest coalesced pose, not yet applied. */
    let pending: PlacementInput | undefined;
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
    const applyPending = () => {
      if (pending !== undefined) {
        const pose = pending;
        pending = undefined;
        this.setPlacement(mesh, pose);
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
        if (!coalesce) {
          this.setPlacement(mesh, placement);
          return;
        }
        // Validate the shape and the values now, so a bad pose throws here
        // and not at the frame. Whether a matrix is a proper rotation (and
        // the scale positive) is still checked when the pose is applied.
        const rotation = rotationMatrix(placement.rotation);
        if (!rotation.every(Number.isFinite)) {
          throw invalid("rotation must be finite numbers");
        }
        const t = placement.translation;
        if (t?.length !== 3 || ![t[0], t[1], t[2]].every(Number.isFinite)) {
          throw invalid("translation must be 3 finite numbers");
        }
        const s = placement.scale;
        if (s !== undefined && (s.length !== 3 || ![s[0], s[1], s[2]].every(Number.isFinite))) {
          throw invalid("scale must be 3 finite numbers");
        }
        pending = {
          rotation,
          translation: [t[0], t[1], t[2]],
          ...(s === undefined ? {} : { scale: [s[0], s[1], s[2]] as Vec3 }),
        };
      },
      end: () => {
        if (finish()) {
          try {
            applyPending();
          } finally {
            close();
          }
        }
      },
      cancel: () => {
        if (!finish()) {
          return;
        }
        pending = undefined;
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
      applyPending,
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
   * Scalar measurements of a mesh in world coordinates: its surface area
   * (always) and, when watertight, its enclosed volume. An open or
   * non-manifold mesh still gets its area; only `volume` is `null`, with
   * `volumeReason` saying why. A read-only call: it leaves every view
   * valid.
   */
  measure(mesh: MeshHandle): MeasureResult {
    const { index, generation } = this.#id(mesh);
    const r = call(() => this.#peek().measure(index, generation));
    const closed = r[2] === 1;
    return {
      area: r[0],
      volume: Number.isNaN(r[1]) ? null : r[1],
      closed,
      volumeReason: closed ? null : r[3] === 2 ? "non-manifold" : "open",
      boundaryLoops: r[4],
      nonManifoldEdges: r[5],
      orientationConflicts: r[6],
    };
  }

  /**
   * Runs one repair operation and returns its report, or a list of them in
   * order and returns their reports. A list is one transaction: one undo step,
   * and when any operation fails nothing changes and nothing is recorded.
   * `{ kind: "auto" }` runs the standard cleanup pipeline the same way and
   * returns a report per stage; see {@link AutoRepairRequest}.
   *
   * A single `inflate`, `relax` or `grab` costs by the vertices under the brush, not the
   * mesh (it uses the BVH, and reaches only vertices a triangle uses); in a list
   * they scan the mesh, so a sculpt stroke should send its dabs one at a time.
   *
   * @throws `INVALID_ARGUMENT` for invalid options (`fillHoles` needs a bound or `loops`); `STALE_LOOPS` when a loop id of `fillHoles` no longer matches a loop of the mesh.
   */
  repair(mesh: MeshHandle, request: AutoRepairRequest): AutoRepairReport;
  repair<const O extends RepairOp>(mesh: MeshHandle, op: O): RepairReportOf<O>;
  repair<const O extends readonly RepairOp[]>(
    mesh: MeshHandle,
    ops: O,
  ): { -readonly [K in keyof O]: RepairReportOf<O[K]> };
  repair(
    mesh: MeshHandle,
    op: RepairOp | readonly RepairOp[] | AutoRepairRequest,
  ): RepairReport | RepairReport[] | AutoRepairReport {
    const { index, generation } = this.#id(mesh);
    if (isAutoRepair(op)) {
      const values = Float64Array.of(
        AUTO_REPAIR_CODE,
        optional(op.weldTolerance, "weldTolerance"),
        optional(op.minTriangles, "minTriangles"),
        optional(op.maxEdges, "maxEdges"),
        op.fair === undefined ? Number.NaN : flagOf(op.fair, true),
      );
      const noLoops = new Uint32Array();
      return decodeAutoReport(new Reader(call(() => this.#scene().repair(index, generation, values, noLoops))));
    }
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

  // ---- Selections ----

  /**
   * Selects vertices of a mesh: a sphere brush, a lasso polygon, or the
   * connected component holding a seed vertex. The result is a snapshot
   * (ascending ids plus the mesh's content stamp); any edit, repair, compact,
   * undo or redo of the mesh invalidates it, and the selection methods then
   * throw `STALE_SELECTION` instead of silently remapping ids. Placement
   * changes keep selections valid. Only vertices a live triangle uses can
   * be selected (dead or never-used slots never are).
   *
   * @throws `UNKNOWN_MESH` for a removed mesh; `INVALID_ARGUMENT` for a bad
   * spec; `INVALID_INDEX` for an out-of-range component seed.
   */
  select(mesh: MeshHandle, spec: SelectSpec): VertexSelection {
    checkSpec(spec);
    this.#id(mesh);
    const views = this.views(mesh);
    const slots = this.vertexSlots(mesh);
    const positions = views.positions().slice();
    const indices = views.indices().slice();
    const topology = buildTopology(indices, slots);
    let ids: number[];
    switch (spec.kind) {
      case "brush": {
        let center = spec.center;
        let radius = spec.radius;
        if (spec.space === "world") {
          const inv = invertMatrix4(this.matrix4(mesh));
          center = transformPoint(inv, spec.center);
          radius = spec.radius / meanStretch(this.matrix4(mesh));
        }
        ids = brushPick(positions, topology.used, center, radius);
        break;
      }
      case "lasso":
        ids = lassoPick(positions, topology.used, spec.polygon, spec.axis ?? "xy");
        break;
      case "component":
        ids = componentOf(topology, spec.seed);
        break;
    }
    return { mesh, vertices: Uint32Array.from(ids), stamp: meshAdds(this).stamp(mesh) };
  }

  /**
   * Whether `sel` is still usable: its mesh is alive and no topology edit
   * landed since it was taken. Never throws.
   */
  selectionAlive(sel: VertexSelection): boolean {
    try {
      if (sel === null || typeof sel !== "object" || !(sel.vertices instanceof Uint32Array)) {
        return false;
      }
      this.#id(sel.mesh);
      return Number.isInteger(sel.stamp) && sel.stamp === meshAdds(this).stamp(sel.mesh);
    } catch {
      return false;
    }
  }

  /** Reads the mesh's topology and checks `sel` against its stamp. */
  #selectionTopology(sel: VertexSelection): ReturnType<typeof buildTopology> {
    this.#id(sel.mesh);
    checkSelection(sel, sel.mesh, meshAdds(this).stamp(sel.mesh));
    const views = this.views(sel.mesh);
    return buildTopology(views.indices().slice(), this.vertexSlots(sel.mesh));
  }

  /**
   * Grows a selection by `steps` rings of edge neighbours (staying on live
   * vertices). The result carries the current stamp, so a fresh selection
   * can be grown right after an edit. `steps` 0 returns an equal copy.
   *
   * @throws `STALE_SELECTION` when the mesh changed since the selection;
   * `UNKNOWN_MESH` for a removed mesh.
   */
  selectGrow(sel: VertexSelection, steps = 1): VertexSelection {
    const topology = this.#selectionTopology(sel);
    return { mesh: sel.mesh, vertices: Uint32Array.from(growIds(topology, sel.vertices, steps)), stamp: meshAdds(this).stamp(sel.mesh) };
  }

  /**
   * Shrinks a selection by `steps` rings, dropping the members with a live
   * neighbour outside the set. Interior selections (no boundary) survive
   * unchanged; shrinking past the core gives an empty (still valid)
   * selection.
   *
   * @throws `STALE_SELECTION` when the mesh changed since the selection;
   * `UNKNOWN_MESH` for a removed mesh.
   */
  selectShrink(sel: VertexSelection, steps = 1): VertexSelection {
    const topology = this.#selectionTopology(sel);
    return { mesh: sel.mesh, vertices: Uint32Array.from(shrinkIds(topology, sel.vertices, steps)), stamp: meshAdds(this).stamp(sel.mesh) };
  }

  /**
   * Inverts a selection: every live vertex outside it. The result carries
   * the current stamp.
   *
   * @throws `STALE_SELECTION` when the mesh changed since the selection;
   * `UNKNOWN_MESH` for a removed mesh.
   */
  selectInvert(sel: VertexSelection): VertexSelection {
    const topology = this.#selectionTopology(sel);
    const inSet = new Set(sel.vertices);
    const out: number[] = [];
    for (let v = 0; v < topology.used.length; v++) {
      if (topology.used[v] === 1 && !inSet.has(v)) {
        out.push(v);
      }
    }
    return { mesh: sel.mesh, vertices: Uint32Array.from(out), stamp: meshAdds(this).stamp(sel.mesh) };
  }

  /**
   * Isolates a selection: removes the triangles fully outside it (no
   * selected vertex), one undo step. Boundary triangles with at least one
   * selected vertex stay. An empty selection selects nothing to keep and
   * throws `INVALID_ARGUMENT` rather than clearing the mesh.
   *
   * @throws `STALE_SELECTION` when the mesh changed since the selection;
   * `UNKNOWN_MESH` for a removed mesh.
   */
  isolate(mesh: MeshHandle, sel: VertexSelection): IsolateReport {
    this.#id(mesh);
    checkSelection(sel, mesh, meshAdds(this).stamp(mesh));
    if (sel.vertices.length === 0) {
      throw invalid("isolate needs a non-empty selection");
    }
    const inSet = new Set(sel.vertices);
    const views = this.views(mesh);
    const indices = views.indices().slice();
    const drop: number[] = [];
    for (let t = 0; t * 3 + 2 < indices.length; t++) {
      const a = indices[t * 3];
      const b = indices[t * 3 + 1];
      const c = indices[t * 3 + 2];
      if (a === 0 && b === 0 && c === 0) {
        continue;
      }
      if (!inSet.has(a) && !inSet.has(b) && !inSet.has(c)) {
        drop.push(t);
      }
    }
    if (drop.length > 0) {
      this.edit(mesh, [{ op: "removeTriangles", triangles: Uint32Array.from(drop) }]);
    }
    return {
      removedTriangles: drop.length,
      selection: { mesh, vertices: sel.vertices.slice(), stamp: meshAdds(this).stamp(mesh) },
    };
  }

  // ---- Fields and colours ----

  /**
   * Adds a signed distance field: one value per vertex slot of `on`, the
   * distance to `to` as placed (negative inside), `Infinity` beyond
   * `maxDistance`, NaN for dead or unused vertices. With several meshes in
   * `to`, the smallest over them, ties going to the earliest. It follows
   * edits and placements by the mesh modes, like normals. Removing a mesh of
   * `to` removes it from the field, or the field with its last target.
   *
   * @throws `INVALID_ARGUMENT` when `to` is empty, holds `on` or a mesh twice, `maxDistance` is negative or NaN, or the name is taken.
   */
  addDistanceField(desc: DistanceFieldDesc): FieldHandle {
    return this.#addField(desc, false);
  }

  /**
   * Adds a per-vertex wall-thickness field: one value per vertex slot of
   * `on`, the distance to the opposite surface along the vertex's smooth
   * normal, in world units. Open sheets and rims read NaN, hits beyond
   * `maxDistance` read `Infinity`. It follows edits and placements by the
   * mesh modes, like normals, and colours like any field.
   *
   * @throws `INVALID_ARGUMENT` when `maxDistance` is negative or NaN, or the name is taken.
   */
  addThicknessField(desc: ThicknessFieldDesc): FieldHandle {
    const s = this.#scene();
    const ids = this.#flatIds([desc.on]);
    if (typeof desc.name !== "string") {
      throw invalid("name must be a string");
    }
    const raw = call(() => s.add_thickness_field(desc.name, ids, desc.maxDistance));
    return this.#rawHandle(this.#fields, () => new FieldHandle(), raw);
  }

  /**
   * Adds a per-vertex curvature field: one value per vertex slot of `on`,
   * the signed mean curvature there, in 1/world-units (positive on domes,
   * negative in bowls, zero on flats). It follows edits and placements by
   * the mesh modes, like normals, and colours like any field.
   *
   * @throws `INVALID_ARGUMENT` when the name is taken.
   */
  addCurvatureField(desc: CurvatureFieldDesc): FieldHandle {
    const s = this.#scene();
    const ids = this.#flatIds([desc.on]);
    if (typeof desc.name !== "string") {
      throw invalid("name must be a string");
    }
    const raw = call(() => s.add_curvature_field(desc.name, ids));
    return this.#rawHandle(this.#fields, () => new FieldHandle(), raw);
  }

  /** `addDistanceField`, or with `deferred` its internal variant that computes nothing yet. */
  #addField(desc: DistanceFieldDesc, deferred: boolean): FieldHandle {
    const s = this.#scene();
    const ids = this.#flatIds([desc.on, ...targetsOf(desc.to)]);
    const sign = codeIn(SIGNS, desc.sign ?? "auto", "sign");
    if (typeof desc.name !== "string") {
      throw invalid("name must be a string");
    }
    const raw = call(() =>
      deferred
        ? s.add_distance_field_deferred(desc.name, ids, desc.maxDistance, sign)
        : s.add_distance_field(desc.name, ids, desc.maxDistance, sign),
    );
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

  /**
   * Per vertex slot, the index in {@link Scene.fieldTargets} of the target
   * the field's value came from, or `0xffffffff` for none (`Infinity` or
   * NaN). A copy, as of the field's last update.
   */
  fieldNearest(field: FieldHandle): Uint32Array {
    const raw = this.#field(field);
    return call(() => this.#peek().field_nearest(raw));
  }

  /** The field's target meshes, in index order (a removed target leaves the list). */
  fieldTargets(field: FieldHandle): MeshHandle[] {
    const raw = this.#field(field);
    const flat = call(() => this.#peek().field_targets(raw));
    const out: MeshHandle[] = [];
    for (let i = 0; i < flat.length; i += 2) {
      out.push(this.#meshHandle(flat[i], flat[i + 1]));
    }
    return out;
  }

  /**
   * What the sign data found out about each of the field's targets, in the
   * order of {@link Scene.fieldTargets}.
   */
  fieldTargetStatus(field: FieldHandle): FieldStatus["sign"][] {
    const raw = this.#field(field);
    const st = call(() => this.#peek().field_target_status(raw));
    const out: FieldStatus["sign"][] = [];
    for (let i = 0; i < st.length; i += 5) {
      out.push({
        mode: st[i] === 2 ? "winding" : "pseudonormal",
        closed: st[i + 1] === 1,
        consistentOrientation: st[i + 2] === 1,
        nonManifoldVertices: st[i + 3],
        degenerateTriangles: st[i + 4],
      });
    }
    return out;
  }

  /**
   * Whether the field is current, and what its sign data found out about the
   * target meshes, together (see {@link FieldStatus.sign}).
   */
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
      this.#rules.delete(mesh);
      return;
    }
    const field = this.#field(rule.field);
    const { map, values } = encodeRule(rule);
    call(() => this.#scene().set_coloring(index, generation, field, map, values));
    // A copy: later changes to the caller's object do not leak in.
    this.#rules.set(mesh, copyRule(rule));
  }

  /**
   * The rule the mesh is coloured by (a frozen copy of the one given to
   * {@link Scene.setColoring}, the same object on every read), or `null`
   * without colouring (never set, turned off, or its field removed).
   *
   * @throws `UNKNOWN_MESH`.
   */
  coloring(mesh: MeshHandle): ColorRule | null {
    const { index, generation } = this.#id(mesh);
    // Removing the field ends the colouring in the engine.
    const on = call(() => this.#peek().values_ptr(index, generation)) !== 0;
    return on ? (this.#rules.get(mesh) ?? null) : null;
  }

  // ---- Stepped mesh adds (internal: see meshAdds) ----

  static {
    meshAddsOf = (scene) => ({
      beginUpload: (vertices, triangles, options = {}) => {
        const s = scene.#scene();
        const mode = modeCode(options.mode ?? "eager");
        const origin = options.origin === undefined ? undefined : Float64Array.from(options.origin);
        return call(() => s.begin_add_mesh_upload(vertices, triangles, origin, mode, options.normals ?? true));
      },
      pushPositions: (token, chunk) => call(() => scene.#scene().add_mesh_push_positions(token, chunk)),
      pushIndices: (token, chunk) => call(() => scene.#scene().add_mesh_push_indices(token, chunk)),
      pushUvs: (token, chunk) => call(() => scene.#scene().add_mesh_push_uvs(token, chunk)),
      step: (token, budget) => call(() => scene.#scene().add_mesh_step(token, budget)),
      buffers: (token) => {
        const [pp, pl, np, nl, ip, il] = call(() => scene.#peek().add_mesh_buffers(token));
        const memory = wasmMemory().buffer;
        return {
          positions: new Float32Array(memory, pp, pl),
          normals: np === 0 ? null : new Float32Array(memory, np, nl),
          indices: new Uint32Array(memory, ip, il),
        };
      },
      finish: (token) => {
        const id = call(() => scene.#scene().finish_add_mesh(token));
        return scene.#meshHandle(id[0], id[1]);
      },
      stamp: (mesh) => {
        const { index, generation } = scene.#id(mesh);
        return call(() => scene.#peek().content_stamp(index, generation));
      },
      drop: (token) => call(() => scene.#peek().drop_add_mesh(token)),
    });
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
      addDeferred: (desc) => scene.#addField(desc, true),
      stepField: (field, points) => {
        const raw = scene.#field(field);
        const s = scene.#scene();
        const state = call(() => s.field_step_field(raw, points));
        return state === 0 ? "idle" : state === 1 ? "pending" : "done";
      },
      withdraw: (field) => {
        const raw = scene.#field(field);
        return call(() => scene.#scene().withdraw_field(raw));
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
