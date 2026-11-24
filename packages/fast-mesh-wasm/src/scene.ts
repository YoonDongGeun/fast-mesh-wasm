// The public `Scene` class: editable meshes with placements, zero-copy render
// views, per-frame change records and undo history. Wraps the raw `WasmScene`
// binding; every call into WebAssembly goes through `call` (error codes and
// panic poisoning).
import {
  checkDelaunayFlipOptions,
  checkDenoiseOptions,
  decodeDelaunayFlipReport,
  decodeDenoiseReport,
  type DelaunayFlipOptions,
  type DelaunayFlipReport,
  type DenoiseOptions,
  type DenoiseReport,
} from "./denoise.js";
import {
  decodeExtrudeBase,
  decodeExtrudeFaces,
  type ExtrudeBaseOptions,
  type ExtrudeBaseResult,
  type ExtrudeFacesOptions,
  type ExtrudeFacesResult,
  extrudeBaseArgs,
  extrudeFacesArgs,
} from "./extrude.js";
import {
  type AddHeightfieldOptions,
  checkHeightfield,
  checkHeightfieldPlacement,
  checkHeightMapRows,
  decodeHeightMapFrame,
  encodeHeightfield,
  encodeHeightMapFrame,
  encodeHeightMapOptions,
  type Heightfield,
  type HeightfieldResult,
  type HeightMap,
  type HeightMapFrame,
  type HeightMapOptions,
  heightMapMeshList,
  isHeightMapFrame,
} from "./height-map.js";
import {
  type ConvexHullOptions,
  type ConvexHullPoints,
  type ConvexHullResult,
  decodeHullCounts,
  hullMeshList,
  hullNormals,
  hullPointsBuffer,
} from "./hull.js";
import {
  checkLatticeOptions,
  decodeLatticePlan,
  decodeLatticePreview,
  decodeLatticeReport,
  DEFAULT_LATTICE_MAX_VOXELS,
  DEFAULT_LATTICE_PREVIEW_VOXELS,
  encodeLatticeOptions,
  type LatticeOptions,
  type LatticePlan,
  type LatticePreview,
  type LatticeReport,
} from "./lattice.js";
import { checkRegisterOptions } from "./register.js";
import {
  checkWrapOptions,
  decodeWrapCounts,
  decodeWrapPlan,
  encodeWrapOptions,
  type WrapGridOptions,
  wrapMeshList,
  type WrapOptions,
  type WrapPlan,
  type WrapResult,
} from "./wrap.js";
import {
  BOOLEAN_METHOD_CODES,
  BOOLEAN_OP_CODES,
  checkSurfacePathOptions,
  checkU32Int,
  COARSE_ALIGN_CODES,
  DYNTOPO_MODE_CODES,
  encodeCsgTree,
  FILL_CONTINUITY_CODES,
  FILL_CONTINUITY_FROM_CODE,
  GEODESIC_METHOD_CODES,
  ICP_OBJECTIVE_CODES,
  isBrushSculptOp,
  normalizeBooleanArgs,
  normalizeCsgBatchArgs,
  normalizeDeviationOptions,
  normalizeDyntopoCenterRadius,
  normalizeDyntopoOptions,
  normalizeFillHolesContinuityOptions,
  normalizeGeodesicOptions,
  normalizeLandmarkPairs,
  normalizeOverhangOptions,
  normalizePartialBridgeOptions,
  normalizeQualityArgs,
  QUALITY_METRIC_CODES,
  normalizeRegistrationOptions,
  normalizeSourceVertices,
  normalizeSurfacePathTarget,
  normalizeTrimCurveArgs,
  normalizeU32List,
  throwIfAborted,
} from "./options.js";
import type { Bounds, Indices, Mat3, Placement, PlacementInput, Positions64, Quat, UpdateMode, Vec3 } from "./api.js";
import { decodeChanges, type Range } from "./changes.js";
import { call, ensureInitialized, FastMeshError, finalizer, release, wasmMemory } from "./core.js";
import {
  createAlignDomain,
  createBooleanDomain,
  createDeformDomain,
  createInspectDomain,
  createRemeshDomain,
  createRepairDomain,
  createSculptDomain,
  type DeformBackend,
  type RemeshBackend,
  type RepairBackend,
  type SculptBackend,
  type SyncAlignDomain,
  type SyncBooleanDomain,
  type SyncDeformDomain,
  type SyncInspectDomain,
  type SyncRemeshDomain,
  type SyncRepairDomain,
  type SyncSculptDomain,
} from "./domains/index.js";
import { FieldHandle, MeshHandle, rotationMatrix, SectionHandle, targetsOf } from "./handles.js";
import { checkInteractionPoints, copyRule, encodeRule } from "./rule.js";
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
import {
  checkBooleanArg,
  checkNonNegativeArg,
  checkU32Arg,
  cloudBuffer,
  decodeCollisionRule,
  decodeCollisionStepReport,
  decodeContourSlices,
  decodeDrainHoleReport,
  decodeFaceEditReport,
  decodeFfdLattice,
  decodeFitPrimitiveReport,
  decodeFreeformReport,
  decodeHollowReport,
  decodePlaceMeshReport,
  decodePlacement15,
  encodeCollisionRule,
  normalizeAlignToPrimitiveArgs,
  normalizeArapArgs,
  normalizeBuildFfdLatticeArgs,
  normalizeCollisionRule,
  normalizeDrainHoleArgs,
  normalizeFfdArgs,
  normalizeFitPrimitiveArgs,
  normalizeGrowShrinkArgs,
  normalizeHollowArgs,
  normalizeKelvinletArgs,
  normalizePlaceMeshArgs,
  normalizePlacementInputArgs,
  normalizeReconstructArgs,
  normalizeScreenSelectArgs,
  normalizeSelectInSphereArgs,
  normalizeSelectSmallIslandsArgs,
  normalizeSliceContoursArgs,
  normalizeSliceContourStackArgs,
  normalizeU32Selection,
  normalizeVertexMaskSelectArgs,
  normalizeXpbdArgs,
  type AdjacencyKind,
  type ArapOptions,
  type CollisionRule,
  type CollisionRuleInput,
  type CollisionStepReport,
  type ContourSlice,
  type DrainHoleOptions,
  type DrainHoleReport,
  type FaceEditReport,
  type FaceSelectionNormalCone,
  type FfdBasisInput,
  type FfdLatticeData,
  type FfdOptions,
  type FitPrimitiveOptions,
  type FittedPrimitive,
  type FittedPrimitiveReport,
  type FreeformReport,
  type HollowOptions,
  type HollowReport,
  type KelvinletOptions,
  type PackedPointCloud,
  type PlaceMeshOptions,
  type PlaceMeshReport,
  type ScreenSelectionShape,
  type ScreenSelectOptions,
  type SplitFacesReport,
  type SurfaceReconstructOptions,
  type XpbdOptions,
} from "./wave6.js";

export type { Range } from "./changes.js";
export type { MeshViews, SceneMeshViews } from "./views.js";
export {
  DEFAULT_COLLISION_RULE,
  packPointCloud,
  unpackPointCloud,
  type AdjacencyKind,
  type ArapAnchorInput,
  type ArapHandleInput,
  type ArapOptions,
  type ArapSolver,
  type ArapWeights,
  type CollisionDistanceBackend,
  type CollisionMode,
  type CollisionPairOutcome,
  type CollisionRole,
  type CollisionRule,
  type CollisionRuleInput,
  type CollisionStepReport,
  type ContourExportFormat,
  type ContourLoop,
  type ContourSlice,
  type CutTrigger,
  type DrainHoleOptions,
  type DrainHoleReport,
  type FaceEditReport,
  type FaceSelectionNormalCone,
  type FfdBasis,
  type FfdBasisInput,
  type FfdFrameData,
  type FfdLattice,
  type FfdLatticeData,
  type FfdOptions,
  type FfdOutsideMode,
  type FitPrimitiveOptions,
  type FittedPrimitive,
  type FittedPrimitiveReport,
  type FreeformProtection,
  type FreeformReport,
  type FreeformScope,
  type HollowBackend,
  type HollowOptions,
  type HollowReport,
  type KelvinletLoad,
  type KelvinletOptions,
  type KelvinletScale,
  type MaterialMemory,
  type OutlierReport,
  type PackedPointCloud,
  type PlaceMeshOptions,
  type PlaceMeshReport,
  type PlaceMode,
  type PlaceTransform,
  type PlaceUpAxis,
  type PointCloudFilterResult,
  type PointCloudFormat,
  type PrimitiveKind,
  type PrimitiveKindInput,
  type ScreenSelectionShape,
  type ScreenSelectOptions,
  type SoftDeformSolver,
  type SplitFacesReport,
  type SurfaceReconstructMethod,
  type SurfaceReconstructOptions,
  type XpbdHandleInput,
  type XpbdOptions,
  type XpbdPinInput,
} from "./wave6.js";
export {
  parsePointCloudBytes,
  parsePointCloudText,
  pointCloudEstimateNormals,
  pointCloudRor,
  pointCloudSor,
  pointCloudVoxelDownsample,
  sliceContoursExport,
} from "./wave6-wasm.js";
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

/** Internal: the `WasmScene` class new scenes wrap ({@link useWasmScene}), for its statics. */
export function activeWasmScene(): typeof WasmScene {
  return RawScene;
}

const meshIds = new WeakMap<MeshHandle, HandleId>();
const rawIds = new WeakMap<FieldHandle | SectionHandle, HandleId>();

/**
 * What changed in one mesh since the previous {@link Scene.takeChanges}.
 * Range lists are ascending; `"all"` means re-upload the whole buffer.
 */
export interface MeshChange {
  mesh: MeshHandle;
  /**
   * The mesh was added, or is back (an undone removal, a redone add); every
   * buffer is `"all"`. A mesh that is back has the handle it had.
   */
  added: boolean;
  /**
   * The mesh was removed (by a call, by the undo of its add or by the redo
   * of its removal); drop its GPU buffers. Nothing else is set but
   * {@link MeshChange.restorable}.
   */
  removed: boolean;
  /**
   * Only on a removal: undo or redo can bring the mesh back, as `added` with
   * the same handle (see {@link Scene.isRestorable}). Keep what shows it.
   */
  restorable?: true;
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
  /**
   * Record the add in the history, in the open step or as a step of its own:
   * undo removes the mesh, redo brings it back with the same handle. Defaults
   * to `false` (the add is permanent).
   */
  undoable?: boolean;
}

/** Options of {@link Scene.removeMesh}. */
export interface RemoveMeshOptions {
  /**
   * Remove the mesh as an undo step, or as part of the open one: undo brings
   * it back with the same handle. Defaults to `false` (removed for good, and
   * its history entries with it).
   */
  undoable?: boolean;
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

/** Which normal direction `inflate` pushes vertices along. */
export type InflateMethod = "vertexNormal" | "averageNormal" | "strokeNormal";

/** Surface relaxation kernel for `smooth`. */
export type SmoothMethod = "taubin" | "laplacian" | "hcLaplacian" | "cotangent";

/** Spatial or elastic kernel for `grab`. */
export type GrabMethod = "sphere" | "geodesic" | "kelvinlet";

/** Which side(s) of the brush plane `flatten` projects toward the plane. */
export type FlattenMode = "both" | "peaks" | "valleys";

/** Footprint profile for `clay`. */
export type ClayProfile = "round" | "strips";

/** Rotational kernel for `twist`. */
export type TwistMethod = "vortex" | "kelvinlet";

/** Surface offsetting / reconstruction algorithm for `repair(mesh, { op: "offset", ... })`. */
export type OffsetAlgorithm = "normal" | "fhEdt" | "bvhSdf";

/**
 * Composable per-vertex surface feature mask multiplied into a brush's
 * spatial falloff.
 */
export type SurfaceFilter =
  | { kind: "none" }
  | { kind: "peaksOnly"; threshold?: number; contrast?: number }
  | { kind: "valleysOnly"; threshold?: number; contrast?: number }
  | { kind: "normalAngle"; normal?: Readonly<Vec3>; maxAngleDeg?: number };

/** Mirror symmetry axes across local coordinate planes `x = 0`, `y = 0`, `z = 0`. */
export interface Symmetry {
  x?: boolean;
  y?: boolean;
  z?: boolean;
}

/**
 * One sculpt brush, stroke capture, or dynamic-topology operation for
 * {@link Scene.sculpt} (and accepted identically by {@link Scene.repair}).
 * Distances are in local mesh units.
 */
export type SculptOp =
  | {
      op: "inflate";
      /** Brush center in local mesh coordinates, as in `views(mesh).positions()`. */
      center: Readonly<Vec3>;
      radius: number;
      falloff: "linear" | "smooth";
      /** Displacement scale; negative pulls instead of pushing. */
      strength: number;
      /** Normal evaluation mode; unset means `"vertexNormal"`. */
      method?: InflateMethod;
      /** Optional per-vertex surface feature mask. */
      filter?: SurfaceFilter;
      /** Optional mirror symmetry across local coordinate planes. */
      symmetry?: Symmetry;
      /** Vertices this op must hold in place, or move less. */
      protect?: VertexProtection;
    }
  | {
      op: "smooth";
      /** Brush center in local mesh coordinates, as in `views(mesh).positions()`. */
      center: Readonly<Vec3>;
      radius: number;
      falloff: "linear" | "smooth";
      /** Displacement scale; the per-vertex weight is `strength * falloff`. */
      strength: number;
      /** How many relaxation cycles to run; 0 moves nothing. */
      cycles: number;
      /** Relaxation kernel; unset means `"taubin"`. */
      method?: SmoothMethod;
      /** Optional per-vertex surface feature mask. */
      filter?: SurfaceFilter;
      /** Optional mirror symmetry across local coordinate planes. */
      symmetry?: Symmetry;
      /** Vertices this op must hold in place, or move less. */
      protect?: VertexProtection;
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
      /** Optional per-vertex surface feature mask. */
      filter?: SurfaceFilter;
      /** Optional mirror symmetry across local coordinate planes. */
      symmetry?: Symmetry;
      /** Vertices this op must hold in place, or move less. */
      protect?: VertexProtection;
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
      /** Spatial or elastic kernel; unset means `"sphere"`. */
      method?: GrabMethod;
      /** Elastic Poisson ratio in `[0, 0.5]` for `"kelvinlet"`; unset means `0.4`. */
      poisson?: number;
      /** Evaluate offsets from stroke-start rest positions (`captureStrokeState`). */
      fromStroke?: boolean;
      /** Automatically widen brush reach on tangential surface drags to prevent fold-over. */
      adaptiveReach?: boolean;
      /** Optional per-vertex surface feature mask. */
      filter?: SurfaceFilter;
      /** Optional mirror symmetry across local coordinate planes. */
      symmetry?: Symmetry;
      /** Vertices this op must hold in place, or move less. */
      protect?: VertexProtection;
    }
  | {
      op: "flatten";
      /** Brush center in local mesh coordinates, as in `views(mesh).positions()`. */
      center: Readonly<Vec3>;
      radius: number;
      falloff: "linear" | "smooth";
      /** Projection blend scale toward the brush plane. */
      strength: number;
      /** Which side(s) of the plane to project; unset means `"both"`. */
      mode?: FlattenMode;
      /** Explicit plane origin in local coordinates; unset fits the falloff-weighted centroid. */
      planeOrigin?: Readonly<Vec3>;
      /** Explicit non-zero plane normal in local coordinates; unset fits the falloff-weighted average normal. */
      planeNormal?: Readonly<Vec3>;
      /** Optional per-vertex surface feature mask. */
      filter?: SurfaceFilter;
      /** Optional mirror symmetry across local coordinate planes. */
      symmetry?: Symmetry;
      /** Vertices this op must hold in place, or move less. */
      protect?: VertexProtection;
    }
  | {
      op: "crease";
      /** Brush center in local mesh coordinates, as in `views(mesh).positions()`. */
      center: Readonly<Vec3>;
      radius: number;
      falloff: "linear" | "smooth";
      /** Normal groove/ridge scale (`< 0` carves a crease, `> 0` raises a ridge, `0` pinches only). */
      strength: number;
      /** Tangent-plane radial pinch factor in `[0, 1]`; unset means `0.5`. */
      pinch?: number;
      /** Optional per-vertex surface feature mask. */
      filter?: SurfaceFilter;
      /** Optional mirror symmetry across local coordinate planes. */
      symmetry?: Symmetry;
      /** Vertices this op must hold in place, or move less. */
      protect?: VertexProtection;
    }
  | {
      op: "clay";
      /** Brush center in local mesh coordinates, as in `views(mesh).positions()`. */
      center: Readonly<Vec3>;
      radius: number;
      falloff: "linear" | "smooth";
      /** Buildup/carve scale (`> 0` builds up, `< 0` carves down). */
      strength: number;
      /** Footprint profile (`"round"` or `"strips"`); unset means `"round"`. */
      profile?: ClayProfile;
      /** Optional stroke tangent direction in local coordinates for `"strips"`. */
      tangent?: Readonly<Vec3>;
      /** Optional per-vertex surface feature mask. */
      filter?: SurfaceFilter;
      /** Optional mirror symmetry across local coordinate planes. */
      symmetry?: Symmetry;
      /** Vertices this op must hold in place, or move less. */
      protect?: VertexProtection;
    }
  | {
      op: "twist";
      /** Brush center in local mesh coordinates, as in `views(mesh).positions()`. */
      center: Readonly<Vec3>;
      radius: number;
      falloff: "linear" | "smooth";
      /** Twist angle scale in radians. */
      strength: number;
      /** Rotational kernel; unset means `"vortex"`. */
      method?: TwistMethod;
      /** Explicit non-zero twist axis in local coordinates; unset uses the average brush normal. */
      axis?: Readonly<Vec3>;
      /** Optional per-vertex surface feature mask. */
      filter?: SurfaceFilter;
      /** Optional mirror symmetry across local coordinate planes. */
      symmetry?: Symmetry;
      /** Vertices this op must hold in place, or move less. */
      protect?: VertexProtection;
    }
  | {
      /**
       * Snapshots lazy stroke-start rest positions and smooth normals for
       * `inflate` (`method: "strokeNormal"`) and `grab` (`fromStroke: true`).
       * Not an edit: no undo step, and it must be the only op of its list.
       */
      op: "captureStrokeState";
      /** Keep an already active stroke snapshot on this mesh. Unset means `false`. */
      onlyIfMissing?: boolean;
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
      /**
       * Records each vertex's longest edge as its rest size, for
       * `splitStretched`. Run it before the deformation starts. Not an edit:
       * no undo step, and it must be the only op of its list. Running it again
       * accepts the mesh as it is then as the new rest shape; `onlyIfMissing`
       * avoids that. The sizes last until a topology edit other than
       * `splitStretched`, or an undo or redo of an edit from before the capture;
       * undoing a topology edit from after it brings them back.
       */
      op: "captureRestSizes";
      /** Keep rest sizes the mesh still holds. Unset means `false`. */
      onlyIfMissing?: boolean;
    }
  | {
      /**
       * Splits the edges a deformation stretched, and nothing else: an edge
       * splits once it is past `limit` times the smaller rest size of its
       * endpoints, measured by `captureRestSizes`. Existing vertices keep their
       * ids and positions, nothing collapses or flips, and a triangle with no
       * split edge keeps its slot. Fails while the mesh holds no valid rest
       * sizes: see `captureRestSizes`.
       */
      op: "splitStretched";
      /** Region center in local mesh coordinates; edges with an endpoint in the region are checked. */
      center: Readonly<Vec3>;
      radius: number;
      /** Multiple of the rest size an edge may reach, above 1; unset means 4/3. */
      limit?: number;
      /** Most interior splits per call, the longest first; boundary edges do not count. Unset means 2^20. */
      maxSplits?: number;
      /** New interior vertices follow the surface's curvature; `false` uses midpoints. Unset means `true`. */
      smooth?: boolean;
      /**
       * With `smooth`, the angle in degrees past which two faces count as
       * separated by a crease, in `(0, 180]`; unset means 45. A new vertex on
       * such an edge sits at its midpoint, and faces beyond it do not tilt a
       * curved point. A gentler crease rounds off, and a flat face beside it
       * bulges: at 45, 0.20 beside a 30 degree crease on a 6.0 edge. Set it below
       * a model's gentlest crease to keep that crease's flat faces flat; a coarse
       * curved surface then rounds less.
       */
      sharpAngle?: number;
    };

/** Domain preset for {@link AutoRepairRequest}. */
export type AutoRepairPreset = "general" | "scan3d" | "print3d" | "cadImport";

/** Strategy for {@link resolveSelfIntersections} and {@link AutoRepairRequest}. */
export type SelfIntersectionMode = "none" | "deleteAndFill" | "cutAndStitch" | "voxelHeal";

/** Options for the `"makeManifold"` repair op. */
export interface ManifoldOptions {
  /** Remove exact duplicate triangles sharing the same unordered vertex triple. Defaults to `true`. */
  removeDuplicates?: boolean;
  /** Prune small dangling boundary fins attached to non-manifold edges. Defaults to `true`. */
  pruneFins?: boolean;
  /** Separate non-manifold edges (`> 2` incident faces) across sheets. Defaults to `true`. */
  splitNonManifoldEdges?: boolean;
  /** Separate pinch (bow-tie) vertices into distinct fan components. Defaults to `true`. */
  splitPinchVertices?: boolean;
}

/** Options for the `"resolveSelfIntersections"` repair op. */
export interface SelfIntersectOptions {
  /** Resolution mode. Defaults to `"deleteAndFill"`. */
  mode?: SelfIntersectionMode;
  /** Number of 1-ring neighbor layers deleted around colliding faces (`"deleteAndFill"` and local fallback). Defaults to `1`. */
  ringExpansion?: number;
  /** Explicit voxel pitch in mesh units for `"voxelHeal"`. Omit for automatic sizing. */
  voxelSize?: number;
  /** Maximum voxel count for `"voxelHeal"` (`>= 64`). Defaults to `16_777_216` (`256^3`). */
  maxVoxels?: number;
  /** Optional hole boundary edge cap passed to `fillHoles`. Defaults to unbounded. */
  maxHoleEdges?: number;
  /** Whether hole patches in `"deleteAndFill"` are faired. Defaults to `true`. */
  fair?: boolean;
  /** Spatial weld tolerance for shared cut seam vertices in `"cutAndStitch"`. Defaults to `1e-5`. */
  stitchTolerance?: number;
}

/** Options for the `"cleanupSlivers"` repair op. */
export interface SliverOptions {
  /** Maximum allowed circumradius-to-2-inradius aspect ratio (`>= 1`). Defaults to `20`. */
  maxAspectRatio?: number;
  /** Minimum interior angle threshold in degrees (`[0, 60]`). Defaults to `2`. */
  minAngleDeg?: number;
  /** Maximum interior angle threshold in degrees (`[60, 180]`). Defaults to `176`. */
  maxAngleDeg?: number;
  /** Maximum cleanup passes over the mesh. Defaults to `3`. */
  maxIterations?: number;
}

/**
 * One repair operation for {@link Scene.repair}. Distances and areas are in
 * mesh units (equal to world units). They are converted to `f32`, so a
 * tolerance or area below the `f32` range rounds to 0.
 */
export type RepairOp =
  | SculptOp
  | { op: "weld"; tolerance: number }
  | { op: "removeDegenerate"; areaEpsilon?: number; removeUnusedVertices?: boolean }
  | { op: "removeSmallComponents"; minTriangles?: number; minArea?: number; keepLargest?: boolean }
  | { op: "orient" }
  | ({ op: "makeManifold" } & ManifoldOptions)
  | ({ op: "resolveSelfIntersections" } & SelfIntersectOptions)
  | ({ op: "cleanupSlivers" } & SliverOptions)
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
      op: "bridgeLoops";
      /** The two loops to connect, from `analyze(mesh).loops`. Order is insignificant. */
      loops: readonly [LoopId, LoopId];
      /** Maximum vertices per loop; unset means the default (128). */
      maxEdges?: number;
    }
  | {
      op: "offset";
      /**
       * Offset distance in local mesh units.
       * - `"normal"` (default): moves vertices along smooth normals (`0` moves nothing).
       * - `"fhEdt"` / `"bvhSdf"`: on closed outward-wound solids, positive expands,
       *   negative shrinks, and `0` voxel-remeshes at the current surface; on open
       *   sheets, `distance > 0` builds a closed thickened shell and `distance <= 0`
       *   throws `INVALID_ARGUMENT`. Drops UVs (`uvsDropped: true`, undoable).
       */
      distance: number;
      /** Offsetting algorithm; defaults to `"normal"`. */
      algorithm?: OffsetAlgorithm;
      /**
       * Grid spacing in local mesh units for `"fhEdt"` and `"bvhSdf"`; must be
       * positive and finite when provided. Omit for automatic (`bbox_diag / 64`,
       * clamped to `maxVoxels`). Ignored when `algorithm` is `"normal"`.
       */
      voxelSize?: number;
      /**
       * Maximum 3D grid voxel count (`nx * ny * nz`) for `"fhEdt"` and `"bvhSdf"`;
       * must be an integer `>= 64`. Defaults to `16_777_216` (`256^3`).
       */
      maxVoxels?: number;
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
       * Stop once this many triangles are alive; must be at least 1 when set.
       * Optional when `targetRatio` or `maxDeviation` is provided; a target
       * at or above the live count leaves the mesh alone.
       */
      targetTriangles?: number;
      /**
       * Target fraction of live input triangles in `(0, 1]`. When provided,
       * resolves the triangle budget to `(live * targetRatio).round().max(1)`.
       */
      targetRatio?: number;
      /**
       * Maximum allowed root-quadric geometric deviation; collapses exceeding
       * `maxDeviation^2` stop or skip cleanly.
       */
      maxDeviation?: number;
      /** Keep boundary vertices in place; defaults to `true`. */
      preserveBoundary?: boolean;
      /**
       * Use Hoppe appearance-aware UV quadrics, protect UV chart borders, and
       * block collapses that fold 2D UV triangles; defaults to `false`.
       */
      preserveUvs?: boolean;
      /**
       * Penalize perpendicular displacement off sharp dihedral creases above
       * `creaseAngleDeg` and lock crease corners; defaults to `false`.
       */
      lockCreases?: boolean;
      /**
       * Dihedral angle threshold in degrees (`[0, 180]`) for feature creases
       * when `lockCreases` is enabled; defaults to `30`.
       */
      creaseAngleDeg?: number;
      /**
       * Reject collapses that invert any 1-ring incident triangle normal by
       * `90` degrees or more; defaults to `true`.
       */
      preventNormalInversion?: boolean;
    }
  | {
      op: "remeshSurface";
      /** Uniform `"isotropic"` (default) or curvature-`"adaptive"` edge sizing. */
      mode?: SurfaceRemeshMode;
      /** Nominal target edge length in local mesh units (positive finite). */
      targetEdgeLength: number;
      /**
       * Lower edge-length bound in `"adaptive"` mode; defaults to
       * `0.35 * targetEdgeLength`.
       */
      minEdgeLength?: number;
      /**
       * Upper edge-length bound in `"adaptive"` mode; defaults to
       * `2 * targetEdgeLength`.
       */
      maxEdgeLength?: number;
      /**
       * Number of split/collapse/flip/relax/reproject passes (capped at `64`);
       * defaults to `5`.
       */
      iterations?: number;
      /** Tangential relaxation strength in `[0, 1]`; defaults to `0.5`. */
      smoothStrength?: number;
      /**
       * Reproject moved interior vertices onto the input reference surface via
       * BVH; defaults to `true`.
       */
      projectToSurface?: boolean;
      /** Keep open boundary vertices fixed; defaults to `true`. */
      lockBoundaries?: boolean;
      /**
       * Detect sharp dihedral edges above `creaseAngleDeg` and preserve sharp
       * creases and corners; defaults to `true`.
       */
      lockCreases?: boolean;
      /**
       * Dihedral angle threshold in degrees (`[0, 180]`) for feature creases;
       * defaults to `30`.
       */
      creaseAngleDeg?: number;
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

/**
 * Vertices a brush op holds back, with optional geodesic soft-band feathering.
 * After the op runs, each listed vertex's move is scaled by its weight: `0`
 * leaves it at exactly the bytes it had, `1` keeps the whole move. When
 * `band > 0`, vertices within geodesic surface distance `band` of `0`-weight
 * vertices feather smoothly across `[0, band]`. The op's report describes its
 * own move before scaling, and it stays one undo step; an op alone whose brush
 * holds only vertices at `0` reports 0 and leaves no step.
 *
 * @throws `INVALID_ARGUMENT` when a vertex is not alive, a weight is outside
 * `[0, 1]`, `band` is negative or non-finite, or the lists differ in length.
 */
export interface VertexProtection {
  /** The vertex slots to protect. */
  vertices: Indices;
  /** One weight per vertex, each in `[0, 1]`; unset holds every listed vertex fully (all `0`). */
  weights?: ArrayLike<number>;
  /** Geodesic soft-falloff transition band width in local units around `0`-weight vertices; unset means `0`. */
  band?: number;
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

/** What `flatten` did. */
export interface FlattenReport {
  moved: number;
  maxDisplacement: number;
}

/** What `crease` did. */
export interface CreaseReport {
  moved: number;
  maxDisplacement: number;
}

/** What `clay` did. */
export interface ClayReport {
  moved: number;
  maxDisplacement: number;
}

/** What `twist` did. */
export interface TwistReport {
  moved: number;
  maxDisplacement: number;
}

/** What `captureStrokeState` did: nothing to report. */
export type CaptureStrokeReport = Record<string, never>;

/** Edge sizing strategy for `remeshSurface`. */
export type SurfaceRemeshMode = "isotropic" | "adaptive";

/** What `remeshSurface` did. */
export interface SurfaceRemeshReport {
  /** Alive triangles before remeshing. */
  initialTriangles: number;
  /** Alive triangles after remeshing. */
  finalTriangles: number;
  /** Edge splits performed across all passes. */
  splits: number;
  /** Edge collapses performed across all passes. */
  collapses: number;
  /** Valence-reducing interior edge flips performed across all passes. */
  flips: number;
  /** Iterations executed. */
  iterationsRun: number;
  /** Whether a UV channel existed and was dropped undoably. */
  uvsDropped: boolean;
}

/** What `remesh` did. */
export interface RemeshReport {
  splits: number;
  collapses: number;
  flips: number;
  rounds: number;
}

/** What `splitStretched` did. */
export interface StretchReport {
  splits: number;
  /** Edges still past the limit because `maxSplits` ran out, for a later call. */
  remaining: number;
  /**
   * `new, a, b` per split, in order: the new vertex and its edge's endpoints,
   * which may be new vertices from earlier in the list. For a caller that holds
   * per-vertex state of its own and has to carry it over.
   */
  vertices: Uint32Array;
}

/** What `captureRestSizes` did: nothing to report. */
export type CaptureRestReport = Record<string, never>;

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
  /** Vertices removed (cleanup pass for `"normal"`, replaced + cleanup for volumetric). */
  verticesRemoved: number;
  /** New vertices added by Marching Cubes (`0` for `"normal"`). */
  verticesAdded: number;
  /** New triangles added by Marching Cubes (`0` for `"normal"`). */
  trianglesAdded: number;
  /** Original triangles replaced (`0` for `"normal"`). */
  trianglesRemoved: number;
  /** Voxel spacing used in local mesh units (`0` for `"normal"`). */
  voxelSize: number;
  /** Whether the `"uv"` channel was dropped (`false` for `"normal"` or meshes without UVs). */
  uvsDropped: boolean;
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
  /** Maximum root-quadric geometric error observed across accepted collapses. */
  maxDeviationAchieved: number;
}

/** What `subdivide` did. */
export interface SubdivideReport {
  /** Vertices added, one per subdivided edge per level. */
  verticesAdded: number;
  /** Triangles added: three per subdivided triangle per level. */
  trianglesAdded: number;
}

/** What `makeManifold` did. */
export interface ManifoldReport {
  removedDuplicateTriangles: number;
  prunedFinTriangles: number;
  splitNonManifoldEdges: number;
  splitPinchVertices: number;
  verticesAdded: number;
  removedVertices: number;
}

/** What `resolveSelfIntersections` did. */
export interface SelfIntersectReport {
  initialPairs: number;
  remainingPairs: number;
  removedTriangles: number;
  addedTriangles: number;
  modeUsed: SelfIntersectionMode;
  uvsDropped: boolean;
}

/** What `cleanupSlivers` did. */
export interface SliverReport {
  collapsedNeedles: number;
  flippedCaps: number;
  removedDegenerate: number;
  removedVertices: number;
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
   * Close both openings with flat, unfaired patches, across UV seams on a
   * mesh with UVs. Defaults to `true`; without it both parts keep the open
   * rim the cut leaves.
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

/** Execution backend for boolean CSG operations. */
export type BooleanMethod = "manifold" | "bvhSdf" | "fhEdt";

/** Options for {@link Scene.boolean}, {@link Scene.booleanBatch}, and {@link BooleanDomain} operations. */
export interface BooleanOptions {
  /** Boolean operation when invoking a generic boolean call. */
  op?: BooleanOp;
  /** Exact topological arrangement (`"manifold"`, default) or volumetric Marching Cubes fallback (`"bvhSdf"`, `"fhEdt"`). */
  method?: BooleanMethod;
  /** Coplanar facet snap/classification tolerance in mesh units. */
  coplanarTolerance?: number;
  /** Collapse needle/sliver triangles created along cut seams (default `true`). */
  snapSlivers?: boolean;
  /** Voxel pitch for `"bvhSdf"` / `"fhEdt"` volumetric fallback (defaults to automatic pitch). */
  voxelPitch?: number;
  /** Optional target triangle budget for post-boolean simplification. */
  maxTriangles?: number;
}

/** A node of an N-ary CSG boolean tree over meshes or mesh indices. */
export type CsgNode =
  | { kind: "leaf"; meshIndex: number }
  | { kind: "leaf"; mesh: MeshHandle }
  | { kind: "op"; op: BooleanOp; left: CsgNode; right: CsgNode };

/** Options for conforming polyline imprinting and closed-curve trimming via {@link Scene.trimByCurve}. */
export interface TrimCurveOptions {
  /** World-space polyline vertices to imprint onto the mesh surface, all on one component of it. */
  points: readonly Readonly<Vec3>[];
  /** Treat the curve as a closed loop and delete one side of it from its component (default `false`). */
  closed?: boolean;
  /**
   * When `closed` is `true`, delete the inside of the loop (`true`, default) or
   * its outside (`false`), as {@link Scene.trimByCurve} tells them apart.
   */
  deleteInside?: boolean;
  /** Snap tolerance in world units (`0` / omitted chooses automatically). */
  snapTolerance?: number;
}

/**
 * What {@link Scene.trimByCurve} changed.
 *
 * Imprinting the curve replaces the triangles it crosses with smaller ones on
 * the same surface; deleting a side of a loop takes surface away. Either way
 * the mesh has `trianglesAdded - trianglesRemoved` more triangles than before,
 * so those two do not tell a cut from an imprint. `removedArea` and
 * `cutTriangles` do: both are `0` when nothing was cut away (an open curve, or
 * a loop that separates nothing, such as one around a torus's tube).
 */
export interface TrimReport {
  /** Handle of the trimmed mesh (`kept`). */
  kept: MeshHandle;
  /** Triangles the mesh did not have before: the pieces left of the triangles the curve split. */
  trianglesAdded: number;
  /**
   * Triangles the mesh no longer has, replaced by the imprint or cut away:
   * `imprintTriangles + cutTriangles`.
   */
  trianglesRemoved: number;
  /** Of those, the ones replaced with smaller triangles that all stayed: none of their surface is gone. */
  imprintTriangles: number;
  /** Of those, the ones that lost surface, in whole or in part. */
  cutTriangles: number;
  /** The area of the surface taken away, in world units squared. */
  removedArea: number;
  /** New vertices along the curve. */
  verticesAdded: number;
  /** Distinct mesh edges along the curve. */
  seamEdges: number;
}

/** Options for printability overhang inspection via {@link Scene.overhangs}. */
export interface OverhangOptions {
  /** Build direction vector in world coordinates (defaults to `[0, 0, 1]`). */
  buildDirection?: Readonly<Vec3>;
  /** Critical overhang angle from horizontal in degrees (defaults to `45`). */
  criticalAngleDeg?: number;
}

/** Overhang report and per-vertex overhang angles returned by {@link Scene.overhangs}. */
export interface OverhangReport {
  totalSupportArea: number;
  overhangFaceCount: number;
  maxOverhangAngleDeg: number;
  /** Per-vertex overhang angles in degrees, aligned with `views(mesh).positions()` slots. */
  vertexAnglesDeg: Float32Array;
}

/** Element quality metric evaluated by {@link Scene.triangleQuality}. */
export type QualityMetric = "radiusRatio" | "minAngleDeg" | "aspectRatio";

/** Options for {@link Scene.triangleQuality}. */
export interface TriangleQualityOptions {
  metric?: QualityMetric;
  /** Threshold below (or above for `"aspectRatio"`) which a live triangle counts as a sliver. */
  sliverThreshold?: number;
}

/** Triangle quality report and per-vertex worst-incident quality returned by {@link Scene.triangleQuality}. */
export interface QualityReport {
  min: number;
  max: number;
  mean: number;
  sliverTriangleCount: number;
  liveTriangleCount: number;
  /** Worst incident triangle quality per vertex slot, aligned with `views(mesh).positions()`. */
  vertexQuality: Float32Array;
}

/** Options for Hausdorff / signed mesh deviation inspection via {@link Scene.meshDeviation}. */
export interface DeviationOptions {
  /** Return signed deviations using target normals (`false` = unsigned Euclidean distance, default). */
  signed?: boolean;
  /** Optional maximum search distance cutoff in world units. */
  maxDistance?: number;
}

/** Mesh-to-mesh deviation statistics and per-vertex deviations returned by {@link Scene.meshDeviation}. */
export interface DeviationStats {
  min: number;
  max: number;
  mean: number;
  rms: number;
  p95: number;
  sampleCount: number;
  /** Per-vertex deviations in world units, aligned with `views(source).positions()` slots. */
  vertexDeviations: Float32Array;
}

/** Algorithm used by {@link Scene.geodesicDistances}. */
export type GeodesicMethod = "dijkstra" | "heatMethod";

/** Options for surface geodesic distance evaluation via {@link Scene.geodesicDistances}. */
export interface GeodesicOptions {
  /** `"dijkstra"` (fast graph distance, default) or `"heatMethod"` (Crane PDE diffusion). */
  method?: GeodesicMethod;
  /** Optional maximum distance cutoff in world units. */
  maxDistance?: number;
  /** Dimensionless time-step scale `m` (`t = m * h^2`) for `"heatMethod"`. */
  timeStepScale?: number;
  /**
   * Cancels the query: its promise rejects with the signal's reason at once,
   * and a `WorkerScene`'s Worker stops computing at the query's next step. A
   * `Scene` computes in one call, so there the signal only counts when it is
   * aborted already: the call then throws its reason.
   */
  signal?: AbortSignal;
}

/** Sample point along an intrinsic surface shortest path. */
export interface SurfacePathPoint {
  /** World-space position on the surface. */
  position: Vec3;
  /** Triangle slot index containing this point. */
  triangle: number;
  /** Barycentric coordinates `[u, v, w]` inside `triangle`. */
  barycentric: Vec3;
}

/** Options for {@link Scene.shortestSurfacePath}. */
export interface SurfacePathOptions extends GeodesicOptions {
  /** Precomputed geodesic distances from {@link Scene.geodesicDistances}. */
  distances?: Float32Array;
}

/** Corresponding world-space landmark pair `(source, target)` for {@link Scene.alignByLandmarks}. */
export interface LandmarkPair {
  source: Readonly<Vec3>;
  target: Readonly<Vec3>;
}

/** Result of N-point Umeyama landmark SVD alignment via {@link Scene.alignByLandmarks}. */
export interface LandmarkAlignment {
  placement: Placement;
  scale: number;
  rmse: number;
}

/** Coarse pre-alignment strategy for {@link Scene.registerMultistage}. */
export type CoarseAlignMethod = "none" | "pcaCentroid" | "orientedBoundingBox";

/** Fine ICP error metric for {@link Scene.registerMultistage}. */
export type IcpObjective = "pointToPoint" | "pointToPlane" | "symmetricNormal";

/** Options for multi-stage coarse + ICP registration via {@link Scene.registerMultistage}. */
export interface RegistrationOptions {
  coarseMethod?: CoarseAlignMethod;
  objective?: IcpObjective;
  maxIterations?: number;
  tolerance?: number;
  overlap?: number;
  maxDistance?: number;
  targetRmse?: number;
  sampleCount?: number;
  andersonWindow?: number;
}

/** Result of {@link Scene.registerMultistage}. */
export interface RegistrationReport {
  placement: Placement;
  /** Column-major 4x4 transform matrix applied to `moving`. */
  transform: number[];
  finalRmse: number;
  iterations: number;
  status: RegisterStatus;
  converged: boolean;
}

/** Symmetry plane result of {@link Scene.detectSymmetryPlane}. */
export interface SymmetryPlaneReport {
  planeNormal: Vec3;
  planePoint: Vec3;
  symmetryErrorRms: number;
  symmetryScore: number;
}

/** Continuity mode across boundary seam edges for {@link Scene.fillHoleContinuity} (#210). */
export type FillContinuity = "flat" | "tangent" | "curvature";

/** Options for {@link Scene.fillHoleContinuity} and {@link Scene.fillAllHolesContinuity} (#210). */
export interface FillHolesContinuityOptions {
  /** Continuity mode (`"curvature"` by default). */
  continuity?: FillContinuity;
  /** Maximum boundary loop perimeter to fill (unbounded when unset). */
  maxHolePerimeter?: number;
  /** Maximum boundary vertex count to fill (unbounded when unset). */
  maxHoleVertices?: number;
  /** Refine and fair the interior patch (`true` by default). */
  refinePatch?: boolean;
  /** Iterative relaxation rounds during refinement (`1` by default). */
  smoothIterations?: number;
  /** Detect and merge nested island loops enclosed inside the hole (`true` by default). */
  detectIslands?: boolean;
}

/** Result statistics returned by continuity hole filling (#210). */
export interface FillContinuityStats {
  loopIndex: number;
  filledVertices: number;
  filledTriangles: number;
  perimeter: number;
  maxDihedralAngleDeg: number;
  continuity: FillContinuity;
  islandLoopsMerged: number;
  converged: boolean;
}

/** Options for {@link Scene.bridgePartialArcs} (#210). */
export interface PartialBridgeOptions {
  /** Index of the first boundary loop (`0` by default; overridden when `loopA` vertices are provided). */
  loopAIndex?: number;
  /** Inclusive-start vertex index pair `[start, end]` along loop A (`[0, 1]` by default). */
  arcARange?: readonly [number, number];
  /** Index of the second boundary loop (`0` by default; overridden when `loopB` vertices are provided). */
  loopBIndex?: number;
  /** Inclusive-start vertex index pair `[start, end]` along loop B (`[2, 3]` by default). */
  arcBRange?: readonly [number, number];
  /** Cyclic offset applied when pairing `arcB` vertices (`0` by default). */
  twistOffset?: number;
  /** Explicit grid subdivisions across the bridge (auto-matched when omitted). */
  subdivisions?: number;
}

/** Result statistics returned by {@link Scene.bridgePartialArcs} (#210). */
export interface BridgeStats {
  trianglesAdded: number;
  verticesAdded: number;
  resultingLoops: number;
  arcAVertices: number;
  arcBVertices: number;
}

/** Dynamic topology detail policy (#212). */
export type DyntopoMode = "split-collapse" | "split-only" | "brush-relative" | "constant-detail";

/** Configuration for localized dynamic topology remeshing (#212). */
export interface DyntopoOptions {
  /** Detail sizing and split/collapse mode (`"split-collapse"` by default). */
  mode?: DyntopoMode;
  /** Target edge length in local units (or brush radius multiplier for `"brush-relative"`). */
  detailSize?: number;
  /** Collapse threshold multiplier relative to `detailSize` (`0.4` by default). */
  minDetailRatio?: number;
  /** Split threshold multiplier relative to `detailSize` (`1.33` by default). */
  maxDetailRatio?: number;
  /** Equalize vertex valence via Delaunay/valence interior edge flips (`true` by default). */
  equalizeValence?: boolean;
  /** Apply tangential relaxation inside the ROI (`false` by default). */
  smoothTangent?: boolean;
}

/** Topological delta returned by {@link Scene.applyDyntopo} (#212). */
export interface DyntopoReport {
  splits: number;
  collapses: number;
  flips: number;
  smoothed: number;
  createdVertices: Uint32Array;
  removedVertices: Uint32Array;
}

/** Combined brush and dynamic topology report returned by {@link Scene.dyntopoDab} (#212). */
export interface DyntopoDabReport extends DyntopoReport {
  moved: number;
  maxDisplacement: number;
}

/** Sculpt brush ops that operate over a localized brush sphere and support Dyntopo. */
export type SculptBrushOp = Extract<
  SculptOp,
  { op: "inflate" | "smooth" | "relax" | "grab" | "flatten" | "crease" | "clay" | "twist" }
>;

/**
 * The standard cleanup pipeline for {@link Scene.repair}: weld, remove
 * degenerate, orient, remove small components, fill holes, as one transaction.
 * The library owns the order; an option left out takes the library's default.
 */
export interface AutoRepairRequest {
  kind: "auto";
  /** Domain preset configuring stages and defaults. Defaults to `"general"`. */
  preset?: AutoRepairPreset;
  /** Optional self-intersection resolution override for this run. */
  selfIntersectionMode?: SelfIntersectionMode;
  /** Voxel spacing in mesh units for volumetric healing (`"voxelHeal"` or `"print3d"` fallback). */
  voxelSize?: number;
  /** Maximum voxel count (`>= 64`) for volumetric healing. Defaults to `16_777_216`. */
  maxVoxels?: number;
  /** Vertices at most this far apart are welded, in mesh units. Default depends on `preset` (`1e-5` for `"general"`). */
  weldTolerance?: number;
  /** Components with fewer triangles are removed; the largest always stays. Default depends on `preset` (`500` for `"general"`). */
  minTriangles?: number;
  /** Fill only holes with at most this many edges; `Infinity` fills any size (`"cadImport"` preserves open boundaries unless finite). Default depends on `preset` (`64` for `"general"`). */
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
  makeManifold?: ManifoldReport;
  cleanupSlivers?: SliverReport;
  resolveSelfIntersections?: SelfIntersectReport;
}

/** The report of a {@link SculptOp}. */
export type SculptReport =
  | InflateReport
  | SmoothReport
  | RelaxReport
  | GrabReport
  | FlattenReport
  | CreaseReport
  | ClayReport
  | TwistReport
  | CaptureStrokeReport
  | CaptureRestReport
  | StretchReport
  | RemeshReport;

/** The report type of a sculpt op. */
export type SculptReportOf<O extends SculptOp> = O extends { op: "inflate" }
  ? InflateReport
  : O extends { op: "smooth" }
    ? SmoothReport
    : O extends { op: "relax" }
      ? RelaxReport
      : O extends { op: "grab" }
        ? GrabReport
        : O extends { op: "flatten" }
          ? FlattenReport
          : O extends { op: "crease" }
            ? CreaseReport
            : O extends { op: "clay" }
              ? ClayReport
              : O extends { op: "twist" }
                ? TwistReport
                : O extends { op: "captureStrokeState" }
                  ? CaptureStrokeReport
                  : O extends { op: "captureRestSizes" }
                    ? CaptureRestReport
                    : O extends { op: "splitStretched" }
                      ? StretchReport
                      : O extends { op: "remesh" }
                        ? RemeshReport
                        : SculptReport;

/** The report of a {@link RepairOp}. */
export type RepairReport =
  | SculptReport
  | WeldReport
  | DegenerateReport
  | RemoveReport
  | OrientReport
  | FillReport
  | BridgeReport
  | OffsetReport
  | ShellReport
  | SimplifyReport
  | SurfaceRemeshReport
  | SubdivideReport
  | ManifoldReport
  | SelfIntersectReport
  | SliverReport;

/** The report type of a repair op. */
export type RepairReportOf<O extends RepairOp> = O extends { op: "weld" }
  ? WeldReport
  : O extends { op: "removeDegenerate" }
    ? DegenerateReport
    : O extends { op: "removeSmallComponents" }
      ? RemoveReport
      : O extends { op: "orient" }
        ? OrientReport
        : O extends { op: "makeManifold" }
          ? ManifoldReport
          : O extends { op: "resolveSelfIntersections" }
            ? SelfIntersectReport
            : O extends { op: "cleanupSlivers" }
              ? SliverReport
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
                      : O extends { op: "flatten" }
                        ? FlattenReport
                        : O extends { op: "crease" }
                          ? CreaseReport
                          : O extends { op: "clay" }
                            ? ClayReport
                            : O extends { op: "twist" }
                              ? TwistReport
                              : O extends { op: "captureStrokeState" }
                                ? CaptureStrokeReport
                                : O extends { op: "offset" }
                                  ? OffsetReport
                                  : O extends { op: "shell" }
                                    ? ShellReport
                                    : O extends { op: "simplify" }
                                      ? SimplifyReport
                                      : O extends { op: "remeshSurface" }
                                        ? SurfaceRemeshReport
                                        : O extends { op: "subdivide" }
                                          ? SubdivideReport
                                          : O extends { op: "splitStretched" }
                                            ? StretchReport
                                            : O extends { op: "captureRestSizes" }
                                              ? CaptureRestReport
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

/** The repair list's prefix code for a {@link VertexProtection}. */
const PROTECT_CODE = 32;
/** The repair list's prefix code for {@link Symmetry} and {@link SurfaceFilter}. */
const BRUSH_MODIFIERS_CODE = 33;

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
  simplify: 28,
  subdivide: 14,
  splitStretched: 16,
  captureRestSizes: 17,
  captureStrokeState: 18,
  flatten: 19,
  crease: 20,
  clay: 21,
  twist: 22,
  makeManifold: 23,
  resolveSelfIntersections: 24,
  cleanupSlivers: 25,
  remeshSurface: 27,
} as const;

/** The op code of a legacy {@link AutoRepairRequest}; see `crate::edit` in the Rust crate. */
const AUTO_REPAIR_CODE = 15;
/** The op code of an extended preset {@link AutoRepairRequest}; see `crate::edit` in the Rust crate. */
const AUTO_REPAIR_PRESET_CODE = 26;

const AUTO_STAGE_OPS: Record<number, RepairOp["op"]> = {
  0: "weld",
  1: "removeDegenerate",
  2: "removeSmallComponents",
  3: "orient",
  4: "fillHoles",
  23: "makeManifold",
  24: "resolveSelfIntersections",
  25: "cleanupSlivers",
};

const PRESET_CODES: Record<AutoRepairPreset, number> = {
  general: 0,
  scan3d: 1,
  print3d: 2,
  cadImport: 3,
};

const SELF_INTERSECTION_MODES: readonly SelfIntersectionMode[] = [
  "none",
  "deleteAndFill",
  "cutAndStitch",
  "voxelHeal",
];

const SELF_INTERSECTION_MODE_CODES: Record<SelfIntersectionMode, number> = {
  none: 0,
  deleteAndFill: 1,
  cutAndStitch: 2,
  voxelHeal: 3,
};

const FALLOFF_CODES = { linear: 0, smooth: 1 } as const;
const INFLATE_METHOD_CODES: Record<InflateMethod, number> = {
  vertexNormal: 0,
  averageNormal: 1,
  strokeNormal: 2,
};
const SMOOTH_METHOD_CODES: Record<SmoothMethod, number> = {
  taubin: 0,
  laplacian: 1,
  hcLaplacian: 2,
  cotangent: 3,
};
const GRAB_METHOD_CODES: Record<GrabMethod, number> = {
  sphere: 0,
  geodesic: 1,
  kelvinlet: 2,
};
const FLATTEN_MODE_CODES: Record<FlattenMode, number> = {
  both: 0,
  peaks: 1,
  valleys: 2,
};
const CLAY_PROFILE_CODES: Record<ClayProfile, number> = {
  round: 0,
  strips: 1,
};
const TWIST_METHOD_CODES: Record<TwistMethod, number> = {
  vortex: 0,
  kelvinlet: 1,
};
const OFFSET_ALGORITHM_CODES = { normal: 0, fhEdt: 1, bvhSdf: 2 } as const;
const SUBDIVIDE_METHOD_CODES = { loop: 0 } as const;
const SURFACE_REMESH_MODE_CODES: Record<SurfaceRemeshMode, number> = {
  isotropic: 0,
  adaptive: 1,
};

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

function optVoxelSize(v: number | undefined): number {
  if (v === undefined) {
    return Number.NaN;
  }
  if (typeof v !== "number" || !Number.isFinite(v) || v <= 0) {
    throw invalid("voxelSize must be positive and finite");
  }
  return v;
}

function optMaxVoxels(v: number | undefined): number {
  if (v === undefined) {
    return Number.NaN;
  }
  if (typeof v !== "number" || !Number.isInteger(v) || v < 64 || v > Number.MAX_SAFE_INTEGER) {
    throw invalid("maxVoxels must be an integer of at least 64");
  }
  return v;
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

function optionalVec3(v: Readonly<Vec3> | undefined, what: string, values: number[]): void {
  if (v === undefined) {
    values.push(0, 0, 0, 0);
    return;
  }
  const c = v as readonly number[];
  if (c?.length !== 3 || ![c[0], c[1], c[2]].every((x) => typeof x === "number" && Number.isFinite(x))) {
    throw invalid(`${what} must be 3 finite numbers`);
  }
  values.push(1, c[0], c[1], c[2]);
}

/** The prefix that wraps the op after it in a {@link VertexProtection}; see `crate::edit` in the Rust crate. */
function encodeProtect(protect: VertexProtection, values: number[]): void {
  const { vertices, weights, band } = protect;
  if (vertices?.length === undefined) {
    throw invalid("protect.vertices must be an array of vertex ids");
  }
  const n = vertices.length;
  if (weights !== undefined && weights.length !== n) {
    throw invalid("protect.weights must have one weight per vertex");
  }
  values.push(PROTECT_CODE, n, optional(band, "protect.band"));
  for (let k = 0; k < n; k++) {
    values.push(vertices[k]);
  }
  for (let k = 0; k < n; k++) {
    values.push(weights === undefined ? 0 : weights[k]);
  }
}

/** Encodes prefix `33` (`brushModifiers`: symmetry + surface filter) before a brush op. */
function encodeBrushModifiers(
  symmetry: Symmetry | undefined,
  filter: SurfaceFilter | undefined,
  values: number[],
): void {
  if (symmetry === undefined && filter === undefined) {
    return;
  }
  let symMask = 0;
  if (symmetry !== undefined) {
    if (typeof symmetry !== "object" || symmetry === null) {
      throw invalid("symmetry must be an object");
    }
    if (symmetry.x) symMask |= 1;
    if (symmetry.y) symMask |= 2;
    if (symmetry.z) symMask |= 4;
  }
  let filterKind = 0;
  let p0 = Number.NaN;
  let p1 = Number.NaN;
  let normal: Readonly<Vec3> | undefined;
  if (filter !== undefined) {
    if (typeof filter !== "object" || filter === null) {
      throw invalid("filter must be an object");
    }
    switch (filter.kind) {
      case "none":
        break;
      case "peaksOnly":
      case "valleysOnly":
        filterKind = filter.kind === "peaksOnly" ? 1 : 2;
        p0 = optional(filter.threshold, "filter.threshold");
        p1 = optional(filter.contrast, "filter.contrast");
        break;
      case "normalAngle":
        filterKind = 3;
        p0 = optional(filter.maxAngleDeg, "filter.maxAngleDeg");
        normal = filter.normal;
        break;
      default:
        throw invalid('filter.kind must be "none", "peaksOnly", "valleysOnly" or "normalAngle"');
    }
  }
  values.push(BRUSH_MODIFIERS_CODE, symMask, filterKind, p0, p1);
  optionalVec3(normal, "filter.normal", values);
}

/** A repair op list as the binding's `(ops, loops)`; see `crate::edit` in the Rust crate. */
function encodeRepair(ops: readonly RepairOp[]): { values: Float64Array; loops: Uint32Array } {
  const values: number[] = [];
  const loops: number[] = [];
  for (const o of ops) {
    const code = codeIn(REPAIR_CODES, o?.op, "repair op");
    const protect = (o as { protect?: VertexProtection } | undefined)?.protect;
    if (protect !== undefined) {
      encodeProtect(protect, values);
    }
    const symmetry = (o as { symmetry?: Symmetry } | undefined)?.symmetry;
    const filter = (o as { filter?: SurfaceFilter } | undefined)?.filter;
    encodeBrushModifiers(symmetry, filter, values);
    values.push(code);
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
          o.method === undefined ? Number.NaN : codeIn(INFLATE_METHOD_CODES, o.method, "inflate method"),
        );
        break;
      case "smooth":
        centerOf(o.center, values);
        values.push(
          optional(o.radius, "radius"),
          codeIn(FALLOFF_CODES, o.falloff, "falloff"),
          optional(o.strength, "strength"),
          optional(o.cycles, "cycles"),
          o.method === undefined ? Number.NaN : codeIn(SMOOTH_METHOD_CODES, o.method, "smooth method"),
        );
        break;
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
        values.push(
          d[0],
          d[1],
          d[2],
          o.method === undefined ? Number.NaN : codeIn(GRAB_METHOD_CODES, o.method, "grab method"),
          optional(o.poisson, "poisson"),
          flagOf(o.fromStroke, false),
          flagOf(o.adaptiveReach, false),
        );
        break;
      }
      case "flatten":
        centerOf(o.center, values);
        values.push(
          optional(o.radius, "radius"),
          codeIn(FALLOFF_CODES, o.falloff, "falloff"),
          optional(o.strength, "strength"),
          o.mode === undefined ? Number.NaN : codeIn(FLATTEN_MODE_CODES, o.mode, "flatten mode"),
        );
        optionalVec3(o.planeOrigin, "planeOrigin", values);
        optionalVec3(o.planeNormal, "planeNormal", values);
        break;
      case "crease":
        centerOf(o.center, values);
        values.push(
          optional(o.radius, "radius"),
          codeIn(FALLOFF_CODES, o.falloff, "falloff"),
          optional(o.strength, "strength"),
          optional(o.pinch, "pinch"),
        );
        break;
      case "clay":
        centerOf(o.center, values);
        values.push(
          optional(o.radius, "radius"),
          codeIn(FALLOFF_CODES, o.falloff, "falloff"),
          optional(o.strength, "strength"),
          o.profile === undefined ? Number.NaN : codeIn(CLAY_PROFILE_CODES, o.profile, "clay profile"),
        );
        optionalVec3(o.tangent, "tangent", values);
        break;
      case "twist":
        centerOf(o.center, values);
        values.push(
          optional(o.radius, "radius"),
          codeIn(FALLOFF_CODES, o.falloff, "falloff"),
          optional(o.strength, "strength"),
          o.method === undefined ? Number.NaN : codeIn(TWIST_METHOD_CODES, o.method, "twist method"),
        );
        optionalVec3(o.axis, "axis", values);
        break;
      case "offset": {
        const distance = optional(o.distance, "distance");
        const alg =
          o.algorithm === undefined
            ? Number.NaN
            : codeIn(OFFSET_ALGORITHM_CODES, o.algorithm, "algorithm");
        values.push(distance, alg, optVoxelSize(o.voxelSize), optMaxVoxels(o.maxVoxels));
        break;
      }
      case "shell":
        values.push(optional(o.thickness, "thickness"));
        break;
      case "simplify":
        if (
          o.targetTriangles === undefined &&
          o.targetRatio === undefined &&
          o.maxDeviation === undefined
        ) {
          throw invalid("simplify requires targetTriangles, targetRatio, or maxDeviation");
        }
        values.push(
          optional(o.targetTriangles, "targetTriangles"),
          optional(o.targetRatio, "targetRatio"),
          optional(o.maxDeviation, "maxDeviation"),
          flagOf(o.preserveBoundary, true),
          flagOf(o.preserveUvs, false),
          flagOf(o.lockCreases, false),
          optional(o.creaseAngleDeg, "creaseAngleDeg"),
          flagOf(o.preventNormalInversion, true),
        );
        break;
      case "remeshSurface":
        if (
          typeof o.targetEdgeLength !== "number" ||
          !Number.isFinite(o.targetEdgeLength) ||
          o.targetEdgeLength <= 0
        ) {
          throw invalid("targetEdgeLength must be positive and finite");
        }
        values.push(
          o.mode === undefined ? Number.NaN : codeIn(SURFACE_REMESH_MODE_CODES, o.mode, "mode"),
          o.targetEdgeLength,
          optional(o.minEdgeLength, "minEdgeLength"),
          optional(o.maxEdgeLength, "maxEdgeLength"),
          optional(o.iterations, "iterations"),
          optional(o.smoothStrength, "smoothStrength"),
          flagOf(o.projectToSurface, true),
          flagOf(o.lockBoundaries, true),
          flagOf(o.lockCreases, true),
          optional(o.creaseAngleDeg, "creaseAngleDeg"),
        );
        break;
      case "subdivide":
        values.push(
          o.method === undefined ? Number.NaN : codeIn(SUBDIVIDE_METHOD_CODES, o.method, "method"),
          optional(o.levels, "levels"),
        );
        break;
      case "splitStretched":
        centerOf(o.center, values);
        values.push(
          optional(o.radius, "radius"),
          optional(o.limit, "limit"),
          optional(o.maxSplits, "maxSplits"),
          flagOf(o.smooth, true),
          optional(o.sharpAngle, "sharpAngle"),
        );
        break;
      case "captureRestSizes":
      case "captureStrokeState":
        values.push(flagOf(o.onlyIfMissing, false));
        break;
      case "makeManifold":
        values.push(
          flagOf(o.removeDuplicates, true),
          flagOf(o.pruneFins, true),
          flagOf(o.splitNonManifoldEdges, true),
          flagOf(o.splitPinchVertices, true),
        );
        break;
      case "resolveSelfIntersections":
        values.push(
          o.mode === undefined ? Number.NaN : codeIn(SELF_INTERSECTION_MODE_CODES, o.mode, "mode"),
          optional(o.ringExpansion, "ringExpansion"),
          optVoxelSize(o.voxelSize),
          optMaxVoxels(o.maxVoxels),
          optional(o.maxHoleEdges, "maxHoleEdges"),
          flagOf(o.fair, true),
          optional(o.stitchTolerance, "stitchTolerance"),
        );
        break;
      case "cleanupSlivers":
        values.push(
          optional(o.maxAspectRatio, "maxAspectRatio"),
          optional(o.minAngleDeg, "minAngleDeg"),
          optional(o.maxAngleDeg, "maxAngleDeg"),
          optional(o.maxIterations, "maxIterations"),
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
    case "smooth":
    case "relax":
    case "grab":
    case "flatten":
    case "crease":
    case "clay":
    case "twist":
      return { moved: r.next(), maxDisplacement: r.next() };
    case "remesh":
      return { splits: r.next(), collapses: r.next(), flips: r.next(), rounds: r.next() };
    case "bridgeLoops":
      return { trianglesAdded: r.next() };
    case "offset":
      return {
        moved: r.next(),
        maxDisplacement: r.next(),
        degenerateRemoved: r.next(),
        verticesRemoved: r.next(),
        verticesAdded: r.next(),
        trianglesAdded: r.next(),
        trianglesRemoved: r.next(),
        voxelSize: r.next(),
        uvsDropped: r.flag(),
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
      return {
        collapsed: r.next(),
        trianglesRemoved: r.next(),
        verticesRemoved: r.next(),
        maxDeviationAchieved: r.next(),
      };
    case "remeshSurface":
      return {
        initialTriangles: r.next(),
        finalTriangles: r.next(),
        splits: r.next(),
        collapses: r.next(),
        flips: r.next(),
        iterationsRun: r.next(),
        uvsDropped: r.flag(),
      };
    case "subdivide":
      return { verticesAdded: r.next(), trianglesAdded: r.next() };
    case "splitStretched":
    {
      const splits = r.next();
      const remaining = r.next();
      const vertices = new Uint32Array(3 * splits);
      for (let k = 0; k < vertices.length; k++) {
        vertices[k] = r.next();
      }
      return { splits, remaining, vertices };
    }
    case "captureRestSizes":
    case "captureStrokeState":
      return {};
    case "makeManifold":
      return {
        removedDuplicateTriangles: r.next(),
        prunedFinTriangles: r.next(),
        splitNonManifoldEdges: r.next(),
        splitPinchVertices: r.next(),
        verticesAdded: r.next(),
        removedVertices: r.next(),
      };
    case "resolveSelfIntersections":
      return {
        initialPairs: r.next(),
        remainingPairs: r.next(),
        removedTriangles: r.next(),
        addedTriangles: r.next(),
        modeUsed: SELF_INTERSECTION_MODES[r.next()],
        uvsDropped: r.flag(),
      };
    case "cleanupSlivers":
      return {
        collapsedNeedles: r.next(),
        flippedCaps: r.next(),
        removedDegenerate: r.next(),
        removedVertices: r.next(),
      };
  }
}

/** An auto repair's result, stage by stage in the order they ran; see `crate::edit` in the Rust crate. */
function decodeAutoReport(r: Reader): AutoRepairReport {
  const stages: Partial<Record<RepairOp["op"], RepairReport>> = {};
  while (r.at < r.data.length) {
    const code = r.next();
    const op = AUTO_STAGE_OPS[code];
    if (op === undefined) {
      throw invalid(`unknown auto repair stage code ${code}`);
    }
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

function decodeFillContinuityStats(raw: Float64Array): FillContinuityStats {
  return {
    loopIndex: raw[0],
    filledVertices: raw[1],
    filledTriangles: raw[2],
    perimeter: raw[3],
    maxDihedralAngleDeg: raw[4],
    continuity: FILL_CONTINUITY_FROM_CODE[raw[5]] ?? "curvature",
    islandLoopsMerged: raw[6],
    converged: raw[7] === 1,
  };
}

function decodeDyntopoPatch(raw: Float64Array, offset: number): DyntopoReport {
  const newCount = raw[offset + 4];
  const removedCount = raw[offset + 5];
  const start = offset + 6;
  return {
    splits: raw[offset],
    collapses: raw[offset + 1],
    flips: raw[offset + 2],
    smoothed: raw[offset + 3],
    createdVertices: Uint32Array.from(raw.subarray(start, start + newCount)),
    removedVertices: Uint32Array.from(
      raw.subarray(start + newCount, start + newCount + removedCount),
    ),
  };
}

/** @internal The stepped mesh adds of `scene` (see {@link MeshAdds}); for the Worker engine. */
export function meshAdds(scene: Scene): MeshAdds {
  return meshAddsOf(scene);
}

/**
 * Geodesic queries in steps (the scene crate's `GeodesicJob`), for the
 * Worker engine only: not part of the public API.
 *
 * - `begin(mesh, sources, options)`: starts `geodesicDistances` with the
 *   same arguments (the signal aside), computing nothing yet; returns the
 *   query's token.
 * - `step(token, work)`: computes over about `work` more elements of the
 *   mesh, at least one pass over it or one iteration of a solve; `true` once
 *   the distances are ready. A query that fails throws and is dropped, as
 *   when its mesh changed meanwhile.
 * - `finish(token)`: the distances, as `geodesicDistances` gives them.
 * - `drop(token)`: forgets a query.
 */
interface GeodesicSteps {
  begin(mesh: MeshHandle, sources: number | readonly number[], options?: GeodesicOptions): number;
  step(token: number, work: number): boolean;
  finish(token: number): Float32Array;
  drop(token: number): void;
}

/** Set by `Scene`'s static block. */
let geodesicStepsOf: (scene: Scene) => GeodesicSteps;

/** @internal The stepped geodesic queries of `scene` (see {@link GeodesicSteps}); for the Worker engine. */
export function geodesicSteps(scene: Scene): GeodesicSteps {
  return geodesicStepsOf(scene);
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
  /** Handles by `"index:generation"`, of live meshes and of those undo or redo can bring back. */
  readonly #meshes = new Map<string, MeshHandle>();
  /** A handle in `#meshes` may be of a mesh the history has dropped since. */
  #prune = false;
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

  /**
   * Callable repair domain (`scene.repair(mesh, op)` and `scene.repair.<method>(...)`).
   */
  readonly repair: SyncRepairDomain = createRepairDomain({
    repair: ((mesh: MeshHandle, op: RepairOp | readonly RepairOp[] | AutoRepairRequest) =>
      this.#runRepair(mesh, op)) as RepairBackend<false>["repair"],
    analyze: (mesh, options) => this.analyze(mesh, options),
    join: (target, source, options) => this.join(target, source, options),
    fillHoleContinuity: (mesh, loopIndex, options) => this.fillHoleContinuity(mesh, loopIndex, options),
    fillAllHolesContinuity: (mesh, options) => this.fillAllHolesContinuity(mesh, options),
    bridgePartialArcs: (mesh, loopA, loopB, options) => this.bridgePartialArcs(mesh, loopA, loopB, options),
    selectFacesScreen: (mesh, shape, options) => this.selectFacesScreen(mesh, shape, options),
    selectFacesInSphere: (mesh, center, radius, normalCone) =>
      this.selectFacesInSphere(mesh, center, radius, normalCone),
    selectByCreaseAngle: (mesh, seedTri, maxDihedralRad) =>
      this.selectByCreaseAngle(mesh, seedTri, maxDihedralRad),
    selectConnectedShell: (mesh, seedTri) => this.selectConnectedShell(mesh, seedTri),
    selectSmallFaceIslands: (mesh, maxFaceCount, maxArea) =>
      this.selectSmallFaceIslands(mesh, maxFaceCount, maxArea),
    growFaceSelection: (mesh, selectedTris, rings, adjacency) =>
      this.growFaceSelection(mesh, selectedTris, rings, adjacency),
    shrinkFaceSelection: (mesh, selectedTris, rings, adjacency) =>
      this.shrinkFaceSelection(mesh, selectedTris, rings, adjacency),
    fillSelectionHoles: (mesh, selectedTris, maxHoleTris) =>
      this.fillSelectionHoles(mesh, selectedTris, maxHoleTris),
    smoothSelectionBoundary: (mesh, selectedTris, iterations) =>
      this.smoothSelectionBoundary(mesh, selectedTris, iterations),
    faceSelectionToVertexMask: (mesh, selectedTris) => this.faceSelectionToVertexMask(mesh, selectedTris),
    vertexMaskToFaceSelection: (mesh, mask, threshold, requireAllVertices) =>
      this.vertexMaskToFaceSelection(mesh, mask, threshold, requireAllVertices),
    deleteSelectedFaces: (mesh, selectedTris) => this.deleteSelectedFaces(mesh, selectedTris),
    extractSelectedFaces: (mesh, selectedTris, options) => this.extractSelectedFaces(mesh, selectedTris, options),
    splitSelectedFaces: (mesh, selectedTris, options) => this.splitSelectedFaces(mesh, selectedTris, options),
  });

  /**
   * Callable sculpt domain (`scene.sculpt(mesh, op)` and `scene.sculpt.<method>(...)`).
   */
  readonly sculpt: SyncSculptDomain = createSculptDomain({
    sculpt: ((mesh: MeshHandle, op: SculptOp | readonly SculptOp[]) =>
      this.#runRepair(mesh, op as RepairOp)) as SculptBackend<false>["sculpt"],
    verticesInSphere: (mesh, center, radius) => this.verticesInSphere(mesh, center, radius),
    beginStep: () => this.beginStep(),
    endStep: () => this.endStep(),
    cancelStep: () => this.cancelStep(),
    applyDyntopo: (mesh, center, radius, options, pinned) =>
      this.applyDyntopo(mesh, center, radius, options, pinned),
    dyntopoDab: (mesh, dab, options, protection) => this.dyntopoDab(mesh, dab, options, protection),
  });

  /** Remesh, simplification, subdivision, and volumetric reconstruction domain. */
  readonly remesh: SyncRemeshDomain = createRemeshDomain({
    repair: ((mesh: MeshHandle, op: RepairOp) => this.#runRepair(mesh, op)) as RemeshBackend<false>["repair"],
  });

  /** Callable boolean, cutting, trimming, offsetting, and shell domain. */
  readonly boolean: SyncBooleanDomain = createBooleanDomain({
    isAsync: false,
    repair: ((mesh: MeshHandle, op: RepairOp) => this.#runRepair(mesh, op)) as RemeshBackend<false>["repair"],
    boolean: (target, source, op, options) => this.#runBoolean(target, source, op, options),
    booleanBatch: (meshesOrTree, treeOrOptions, maybeOptions) =>
      Array.isArray(meshesOrTree)
        ? this.booleanBatch(meshesOrTree, treeOrOptions as CsgNode, maybeOptions)
        : this.booleanBatch(meshesOrTree as CsgNode, treeOrOptions as BooleanOptions | undefined),
    cutByPlane: (mesh, plane, options) => this.cutByPlane(mesh, plane, options),
    trimByCurve: (mesh, optionsOrCurve, maybeOptions) =>
      Array.isArray(optionsOrCurve)
        ? this.trimByCurve(mesh, optionsOrCurve, maybeOptions)
        : this.trimByCurve(mesh, optionsOrCurve as TrimCurveOptions),
    hollowMesh: (mesh, options) => this.hollowMesh(mesh, options),
    punchDrainHole: (mesh, options) => this.punchDrainHole(mesh, options),
    placeMesh: (target, stamp, options) => this.placeMesh(target, stamp, options),
  });

  /** Surface deformation and vertex-protection domain. */
  readonly deform: SyncDeformDomain = createDeformDomain({
    sculpt: ((mesh: MeshHandle, op: SculptOp) =>
      this.#runRepair(mesh, op as RepairOp)) as DeformBackend<false>["sculpt"],
    buildFfdLattice: (mesh, dims, basis, padding) => this.buildFfdLattice(mesh, dims, basis, padding),
    applyFfd: (mesh, options) => this.applyFfd(mesh, options),
    applyArap: (mesh, options) => this.applyArap(mesh, options),
    applyKelvinlet: (mesh, options) => this.applyKelvinlet(mesh, options),
    applyXpbd: (mesh, options) => this.applyXpbd(mesh, options),
  });

  /** Defect analysis, metrology, queries, printability/quality/geodesic fields, paths, and section inspection domain. */
  readonly inspect: SyncInspectDomain = createInspectDomain({
    analyze: (mesh, options) => this.analyze(mesh, options),
    measure: (mesh) => this.measure(mesh),
    bounds: (mesh) => this.bounds(mesh),
    raycast: (origin, direction, options) => this.raycast(origin, direction, options),
    closestPoint: (point, options) => this.closestPoint(point, options),
    meshDistance: (a, b, options) => this.meshDistance(a, b, options),
    intersects: (a, b) => this.intersects(a, b),
    selfIntersectingTriangles: (mesh) => this.selfIntersectingTriangles(mesh),
    findContacts: (options) => this.findContacts(options),
    overhangs: (mesh, options) => this.overhangs(mesh, options),
    triangleQuality: (mesh, metricOrOptions, sliverThreshold) =>
      this.triangleQuality(mesh, metricOrOptions, sliverThreshold),
    meshDeviation: (source, target, options) => this.meshDeviation(source, target, options),
    geodesicDistances: (mesh, sources, options) => this.geodesicDistances(mesh, sources, options),
    releaseGeodesics: () => this.releaseGeodesics(),
    shortestSurfacePath: (mesh, sources, targetVertex, options) =>
      this.shortestSurfacePath(mesh, sources, targetVertex, options),
    addDistanceField: (desc) => this.addDistanceField(desc),
    addThicknessField: (desc) => this.addThicknessField(desc),
    addCurvatureField: (desc) => this.addCurvatureField(desc),
    removeField: (field) => this.removeField(field),
    fieldValues: (field) => this.fieldValues(field),
    fieldNearest: (field) => this.fieldNearest(field),
    fieldStatus: (field) => this.fieldStatus(field),
    fieldTargetStatus: (field) => this.fieldTargetStatus(field),
    setColoring: (mesh, rule) => this.setColoring(mesh, rule),
    coloring: (mesh) => this.coloring(mesh),
    addSection: (desc) => this.addSection(desc),
    setSectionPlane: (section, plane) => this.setSectionPlane(section, plane),
    sectionPlane: (section) => this.sectionPlane(section),
    setSectionMeshes: (section, meshes) => this.setSectionMeshes(section, meshes),
    setSectionFill: (section, fill) => this.setSectionFill(section, fill),
    removeSection: (section) => this.removeSection(section),
    section: (section, mesh) => this.section(section, mesh),
    sectionGap: (section, a, b) => this.sectionGap(section, a, b),
    updateSections: () => this.updateSections(),
    fitPrimitive: (mesh, options) => this.fitPrimitive(mesh, options),
    alignToFittedPrimitive: (mesh, primitive, targetAxis, targetOrigin) =>
      this.alignToFittedPrimitive(mesh, primitive, targetAxis, targetOrigin),
    sliceContours: (mesh, origin, normal) => this.sliceContours(mesh, origin, normal),
    sliceContourStack: (mesh, origin, normal, spacing, count) =>
      this.sliceContourStack(mesh, origin, normal, spacing, count),
    reconstructPointCloud: (cloud, options) => this.reconstructPointCloud(cloud, options),
  });

  /** Rigid registration, landmark SVD, multi-stage alignment, and symmetry detection domain. */
  readonly align: SyncAlignDomain = createAlignDomain({
    register: (moving, fixed, options) => this.register(moving, fixed, options),
    alignByLandmarks: (moving, fixed, pairs, allowScaling) =>
      this.alignByLandmarks(moving, fixed, pairs, allowScaling),
    registerMultistage: (moving, fixed, options) => this.registerMultistage(moving, fixed, options),
    detectSymmetryPlane: (mesh) => this.detectSymmetryPlane(mesh),
  });

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
    const uvs =
      options.uvs === undefined || options.uvs instanceof Float32Array ? options.uvs : Float32Array.from(options.uvs);
    const id = call(() => {
      if (options.undoable === true) {
        return s.add_mesh_undoable(p, i, uvs, origin, mode, normals);
      }
      return uvs === undefined
        ? s.add_mesh(p, i, origin, mode, normals)
        : s.add_mesh_with_uvs(p, i, uvs, origin, mode, normals);
    });
    return this.#meshHandle(id[0], id[1]);
  }

  /**
   * Removes a mesh. Its handle then gives `UNKNOWN_MESH`. An active drag of
   * the mesh is ended first, closing its step; the mesh's history entries
   * (the drag's included) are then dropped with it.
   *
   * With `undoable` the removal is an undo step (or part of the open one)
   * and the history stays: {@link Scene.undo} brings the mesh back with the
   * same handle, its data, placement, display (normals, colouring), the
   * fields on it and its collision rules (a rule once both its meshes are
   * back). Nothing is copied, whatever the mesh's size. What belongs to other
   * meshes is cut as by any removal and does not come back: fields on other
   * meshes measured to it, and its section entries. The kept mesh counts
   * against the history's memory budget, and is freed once the history drops
   * the step (see {@link Scene.isRestorable}).
   */
  removeMesh(mesh: MeshHandle, options: RemoveMeshOptions = {}): void {
    const { index, generation } = this.#id(mesh);
    // A drag of this mesh ends first, recording its moves as one step.
    if (this.#drag?.mesh === mesh) {
      this.#drag.end();
    }
    if (options.undoable === true) {
      // Its handle stays known, for the way back.
      call(() => this.#scene().remove_mesh_undoable(index, generation));
      this.#prune = true;
      return;
    }
    call(() => this.#scene().remove_mesh(index, generation));
    const key = `${index}:${generation}`;
    this.#meshes.delete(key);
    this.#removed.set(key, mesh);
  }

  /** Whether the scene has `mesh`: `false` once it is removed, or its add is undone. */
  has(mesh: MeshHandle): boolean {
    const { index, generation } = this.#id(mesh);
    return this.#peek().has_mesh(index, generation);
  }

  /**
   * Whether undo or redo can still bring back `mesh`: it was removed with
   * `undoable`, or its undoable add was undone, and the history has not
   * dropped the step (its memory budget, {@link Scene.clearHistory}, or a
   * new step after an undone add). `false` for a mesh in the scene.
   */
  isRestorable(mesh: MeshHandle): boolean {
    const { index, generation } = this.#id(mesh);
    return this.#peek().is_restorable(index, generation);
  }

  /**
   * Joins `source` into `target`: copies the source's geometry into the
   * target (keeping its world shape), bridges `options.loopA` to the copied
   * `options.loopB`, and removes the source. All of it is one undo step:
   * {@link Scene.undo} restores the target and brings the source back with
   * the same handle, as for {@link Scene.removeMesh} with `undoable` (what
   * comes back with it, and the memory it takes, are described there). A drag
   * of the source ends first, recording its moves as one step, so they
   * survive a failed join.
   *
   * @deprecated Use `scene.repair.join()` instead.
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
    // Its handle stays known, for the way back.
    this.#prune = true;
    return { verticesAdded: r[0], trianglesAdded: r[1], bridgeTrianglesAdded: r[2] };
  }

  /**
   * Cuts `mesh` with a world `plane`: the `+` side
   * (`signedDistance >= 0`) stays on the mesh and the `-` side is added as
   * a new mesh with the same placement, so the world shape is kept.
   * Cutting a closed mesh with `cap` (the default) yields two closed
   * meshes, capped with flat, unfaired patches. On a mesh with UVs, closed
   * means closed by exact position, so seams count as closed: the cap's UVs
   * come from the rim and blend across a seam. A rim that runs into a real
   * boundary of an open mesh stays open. A mesh without UVs is read by vertex
   * index, so its split vertices are such a boundary: weld it first.
   *
   * The cut and the added mesh are one undo step: undo restores the mesh
   * and removes the added one, redo brings it back with the same handle.
   *
   * @deprecated Use `scene.boolean.cut()` instead.
   * @throws `UNKNOWN_MESH` for a removed handle; `INVALID_ARGUMENT` when
   * the plane does not cut the mesh or a rim cannot be capped, as listed in
   * `docs/uv-channel.md`, for example a rim that closes by position with its
   * walls wound against each other. A failed cut changes nothing.
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
  #runBoolean(
    target: MeshHandle,
    source: MeshHandle,
    op: BooleanOp,
    options?: Omit<BooleanOptions, "op">,
  ): MeshHandle {
    const ta = this.#id(target);
    const sb = this.#id(source);
    const norm = normalizeBooleanArgs(op, options);
    const r =
      options === undefined
        ? call(() => this.#scene().boolean(ta.index, ta.generation, sb.index, sb.generation, BOOLEAN_OP_CODES[norm.op]))
        : call(() =>
            this.#scene().boolean_with_options(
              ta.index,
              ta.generation,
              sb.index,
              sb.generation,
              BOOLEAN_OP_CODES[norm.op],
              BOOLEAN_METHOD_CODES[norm.method],
              optional(norm.coplanarTolerance, "coplanarTolerance"),
              norm.snapSlivers,
              optional(norm.voxelPitch, "voxelPitch"),
              optional(norm.maxTriangles, "maxTriangles"),
            ),
          );
    return this.#meshHandle(r[0], r[1]);
  }

  /**
   * Evaluates an N-ary CSG tree over `meshes` (or handle-leaf nodes) and inserts
   * the resulting mesh into the scene at identity placement (one undo step). Its
   * {@link Scene.origin} is the centre of the bounds of the meshes used, so that it
   * is as precise far from the world origin as near it.
   */
  booleanBatch(meshes: readonly MeshHandle[], tree: CsgNode, options?: BooleanOptions): MeshHandle;
  booleanBatch(tree: CsgNode, options?: BooleanOptions): MeshHandle;
  booleanBatch(
    meshesOrTree: readonly MeshHandle[] | CsgNode,
    treeOrOptions?: CsgNode | BooleanOptions,
    maybeOptions?: BooleanOptions,
  ): MeshHandle {
    const { meshes, tree, options } = normalizeCsgBatchArgs(meshesOrTree, treeOrOptions, maybeOptions);
    const ids = this.#flatIds(meshes);
    const encodedTree = Float64Array.from(encodeCsgTree(tree));
    const r = call(() =>
      this.#scene().boolean_batch(
        ids,
        encodedTree,
        BOOLEAN_OP_CODES[options.op],
        BOOLEAN_METHOD_CODES[options.method],
        optional(options.coplanarTolerance, "coplanarTolerance"),
        options.snapSlivers,
        optional(options.voxelPitch, "voxelPitch"),
        optional(options.maxTriangles, "maxTriangles"),
      ),
    );
    return this.#meshHandle(r[0], r[1]);
  }

  /**
   * Adds the convex hull of `meshes` (one mesh or several) as a new mesh and
   * returns it with its counts. The hull is built from the vertices the
   * meshes' live triangles use; the meshes are not changed.
   *
   * The hull is exact (no tolerance): every vertex is inside it or on its
   * boundary, and it is convex, closed and wound outward, with no
   * degenerate triangle. Its vertices are input vertices, including every
   * corner. Duplicate, coplanar and collinear vertices are fine.
   *
   * When every mesh has the same placement (one mesh, say), the hull is built
   * in their own space and the new mesh gets that placement, so it moves
   * with them; with one origin too, its vertices are exactly theirs.
   * Otherwise it is built in world space at the identity placement.
   *
   * The new mesh is added as one undo step: {@link Scene.undo} removes it and
   * {@link Scene.redo} brings it back with the same handle.
   *
   * @throws {FastMeshError} `INVALID_ARGUMENT` for an empty list or when the
   *   vertices have no volume (all on one point, line or plane; the message
   *   says which), `UNKNOWN_MESH` for a removed mesh.
   */
  convexHull(meshes: MeshHandle | readonly MeshHandle[], options?: ConvexHullOptions): ConvexHullResult {
    const ids = this.#flatIds(hullMeshList(meshes));
    const normals = hullNormals(options);
    const r = call(() => this.#scene().convex_hull(ids, normals));
    return { mesh: this.#meshHandle(r[0], r[1]), ...decodeHullCounts(r) };
  }

  /**
   * Adds the convex hull of world `points` (flat positions or a packed point
   * cloud) as a new mesh at the identity placement, and returns it with its
   * counts. Exact and one undo step, as for {@link Scene.convexHull}.
   *
   * @throws {FastMeshError} `INVALID_ARGUMENT` for fewer than four points, a
   *   value that is not a finite number (or beyond `1e100`), or points without
   *   volume (all on one point, line or plane; the message says which).
   */
  convexHullOfPoints(points: ConvexHullPoints, options?: ConvexHullOptions): ConvexHullResult {
    const buf = hullPointsBuffer(points);
    const normals = hullNormals(options);
    const r = call(() => this.#scene().convex_hull_points(buf, normals));
    return { mesh: this.#meshHandle(r[0], r[1]), ...decodeHullCounts(r) };
  }

  /**
   * Fits a height map's frame to `meshes` (one mesh or several): an
   * orthographic image of square pixels looking along `options.direction`,
   * centred on the meshes, with their height range. `heightMap` samples it.
   *
   * @throws {FastMeshError} `INVALID_ARGUMENT` for an empty list, meshes
   *   without a triangle or without an extent across the direction, an option
   *   out of range, or an image over {@link MAX_HEIGHT_MAP_PIXELS};
   *   `UNKNOWN_MESH` for a removed mesh.
   */
  heightMapFrame(meshes: MeshHandle | readonly MeshHandle[], options?: HeightMapOptions): HeightMapFrame {
    const ids = this.#flatIds(heightMapMeshList(meshes));
    const values = Float64Array.from(encodeHeightMapOptions(options));
    return decodeHeightMapFrame(call(() => this.#scene().height_map_frame(ids, values)));
  }

  /**
   * An orthographic depth image of `meshes` (one mesh or several, in their
   * placements): the height of each pixel above the frame's zero plane, NaN
   * where no surface is seen. A pixel's ray runs through its centre and takes
   * the first surface within the frame's range (either side of a triangle);
   * a ray exactly through an edge or vertex takes the nearest of the triangles
   * meeting there, so a watertight mesh has no cracks.
   *
   * `frame` is a frame (from {@link Scene.heightMapFrame}, say), or the options
   * to fit one with. `rows` samples a band of the image's rows, `[first, end)`:
   * the bands of an image hold exactly what sampling it whole would, so a large
   * image can be taken in steps.
   *
   * `addHeightfield` with the heights, the frame's pixel size and
   * `heightMapPlacement(frame)` makes a relief that lies on the surface seen.
   *
   * @example
   * ```ts
   * const { frame, heights } = scene.heightMap(mesh, { direction: [0, 0, -1], resolution: 1024 });
   * ```
   * @throws {FastMeshError} `INVALID_ARGUMENT` for an empty list, an invalid
   *   frame, rows outside it, or as for {@link Scene.heightMapFrame};
   *   `UNKNOWN_MESH` for a removed mesh, `STALE_BVH` for a stale manual mesh.
   */
  heightMap(
    meshes: MeshHandle | readonly MeshHandle[],
    frame?: HeightMapFrame | HeightMapOptions,
    rows?: readonly [number, number],
  ): HeightMap {
    const ids = this.#flatIds(heightMapMeshList(meshes));
    let raw: Float64Array;
    if (isHeightMapFrame(frame)) {
      raw = Float64Array.from(encodeHeightMapFrame(frame));
    } else {
      const options = Float64Array.from(encodeHeightMapOptions(frame));
      raw = call(() => this.#scene().height_map_frame(ids, options));
    }
    const fitted = decodeHeightMapFrame(raw);
    const [first, end] = checkHeightMapRows(rows, fitted.height);
    const heights = call(() => this.#scene().height_map(ids, raw, first, end));
    return { frame: fitted, rows: [first, end], heights };
  }

  /**
   * Adds the relief of a grid of heights as a new mesh and returns it with its
   * counts: a closed, outward-wound solid from a flat base up to the heights,
   * with a vertex per sample at exactly its height (see `generateHeightfield`
   * for its frame). The buffers never leave the engine, so large reliefs cost
   * no copy on this side.
   *
   * The new mesh is added as one undo step: {@link Scene.undo} removes it and
   * {@link Scene.redo} brings it back with the same handle.
   *
   * @throws {FastMeshError} `INVALID_ARGUMENT` for fewer than 2 samples on a
   *   side, a wrong number of heights, a height that is infinite or not above
   *   `-base`, or an invalid placement; `OUT_OF_MEMORY` when the relief does
   *   not fit.
   */
  addHeightfield(field: Heightfield, options?: AddHeightfieldOptions): HeightfieldResult {
    const f = checkHeightfield(field);
    const p = checkHeightfieldPlacement(options);
    const values = encodeHeightfield(f);
    const rotation = Float64Array.from(p.rotation);
    const translation = Float64Array.from(p.translation);
    const scale = p.scale === undefined ? undefined : Float64Array.from(p.scale);
    const r = call(() =>
      this.#scene().add_heightfield(f.heights, f.width, f.height, values, rotation, translation, scale, p.normals),
    );
    return { mesh: this.#meshHandle(r[0], r[1]), vertices: r[2], triangles: r[3] };
  }

  /**
   * Smooths noise (from a scan, say) out of `mesh` while keeping its sharp
   * features, as one undo step: the face normals are filtered over their
   * neighbours, except across edges sharper than `featureAngle`, and the
   * vertices move to fit them. Flat faces become flat again and the edges
   * between them stay sharp. The triangles and UVs are kept.
   *
   * With `region` (a face selection), only those triangles are smoothed and
   * no vertex that another triangle uses moves. The report has the
   * surface's readouts before and after. Nothing is recorded when nothing
   * moves.
   *
   * @throws {FastMeshError} `INVALID_ARGUMENT` for an option out of range or
   *   a dead or out-of-range triangle in `region`, `UNKNOWN_MESH` for a
   *   removed mesh. Nothing changes then.
   */
  denoise(mesh: MeshHandle, options?: DenoiseOptions): DenoiseReport {
    const { index, generation } = this.#id(mesh);
    const o = checkDenoiseOptions(options);
    const raw = call(() =>
      this.#scene().denoise(
        index,
        generation,
        o.iterations ?? Number.NaN,
        o.strength ?? Number.NaN,
        o.featureAngle ?? Number.NaN,
        o.region,
      ),
    );
    return decodeDenoiseReport(raw);
  }

  /**
   * Improves `mesh`'s triangles without moving a vertex, as one undo step:
   * flips edges towards a Delaunay triangulation wherever that takes the
   * surface at most `dihedralLimit` degrees from what it was. Edges sharper
   * than the limit never flip, and the smallest corner angle never drops.
   * UVs are kept.
   *
   * With `region` (a face selection), only edges between two of its
   * triangles flip. The report has the surface's readouts before and after.
   * Nothing is recorded when nothing flips.
   *
   * @throws {FastMeshError} as {@link Scene.denoise}.
   */
  delaunayFlips(mesh: MeshHandle, options?: DelaunayFlipOptions): DelaunayFlipReport {
    const { index, generation } = this.#id(mesh);
    const o = checkDelaunayFlipOptions(options);
    const raw = call(() =>
      this.#scene().delaunay_flips(index, generation, o.dihedralLimit ?? Number.NaN, o.region),
    );
    return decodeDelaunayFlipReport(raw);
  }

  /**
   * Plans the voxel grid of {@link Scene.wrap} for the same meshes and
   * options without running it: its dimensions, voxel size, voxel count and
   * estimated memory, and whether it is within `maxVoxels`. Cheap (one pass
   * over the vertices), so it can follow a settings form.
   *
   * @throws {FastMeshError} `INVALID_ARGUMENT` for an empty list, bad options,
   *   meshes without triangles, or a voxel size that `f32` positions cannot
   *   resolve; `UNKNOWN_MESH` for a removed mesh. A grid over the limit is not
   *   an error here: it comes back with `fits: false`.
   */
  wrapPlan(meshes: MeshHandle | readonly MeshHandle[], options?: WrapGridOptions): WrapPlan {
    const ids = this.#flatIds(wrapMeshList(meshes));
    const values = encodeWrapOptions(checkWrapOptions(options));
    return decodeWrapPlan(call(() => this.#scene().wrap_plan(ids, values)));
  }

  /**
   * Wraps `meshes` (one mesh or several) in one closed envelope, added as a
   * new mesh, and returns it with its counts. A robust last resort for broken
   * scans and merged parts: the meshes may be open, holed, overlapping,
   * self-intersecting or wound either way, and are not changed.
   *
   * The envelope is closed, manifold, wound outward and free of
   * self-intersections. Gaps, slots and holes narrower than `gap` are closed,
   * and interior cavities are filled. Every vertex is within about half the
   * gap (and a voxel) of the meshes. It is built on a voxel grid: features
   * thinner than a voxel may be lost, and sharp edges are cut by up to a voxel,
   * less with `project`. A shell with a hole wider than the gap is open to the
   * outside, so its inside is not filled and its walls, with no thickness,
   * are lost: raise the gap over the hole. {@link Scene.wrapPlan} reports the
   * grid and its memory beforehand, and a grid over `maxVoxels` is refused.
   *
   * When every mesh has the same placement and it has no scale, the wrap is
   * built in their own space and the new mesh gets that placement; otherwise
   * it is built in world space at the identity placement. Lengths are world
   * units either way.
   *
   * The new mesh is added as one undo step: {@link Scene.undo} removes it and
   * {@link Scene.redo} brings it back with the same handle.
   *
   * At a million triangles this takes from about half a second to many
   * seconds, by detail and threads: use `WorkerScene.wrap` to keep the page
   * live.
   *
   * @throws {FastMeshError} `INVALID_ARGUMENT` for an empty list, bad options,
   *   a grid over `maxVoxels` (the message has its size and memory), or meshes
   *   that enclose no volume at this gap and voxel size (a flat sheet, or a
   *   shell with a hole wider than the gap); `UNKNOWN_MESH` for a removed mesh;
   *   `OUT_OF_MEMORY`.
   */
  wrap(meshes: MeshHandle | readonly MeshHandle[], options?: WrapOptions): WrapResult {
    const ids = this.#flatIds(wrapMeshList(meshes));
    const checked = checkWrapOptions(options);
    const r = call(() => this.#scene().wrap_meshes(ids, encodeWrapOptions(checked), checked.normals));
    return { mesh: this.#meshHandle(r[0], r[1]), ...decodeWrapCounts(r) };
  }

  /**
   * Conformatively imprints a world-space polyline onto `mesh`, and with
   * `closed: true` deletes the inside of the loop (or its outside, with
   * `deleteInside: false`), as one undo step.
   *
   * Only the component the curve lies on is imprinted and cut; every other one
   * stays as it is. A component is a set of triangles joined through shared
   * edges. On a mesh with UVs an edge also joins across a seam (two edges at
   * exactly the same positions), so the UV islands of one surface are one
   * component, and the curve and the cut cross from one to the next. Split
   * vertices on a mesh without UVs are a real border between components: weld
   * them first.
   *
   * All points must lie on one component: a curve with points on several
   * throws. Where components touch or coincide, a point lies on each of them.
   * When all points lie on several, the curve is on the one the first point
   * projects to, or else on the one with the smallest triangle id there.
   *
   * The inside of a loop is the side away from the component's open edges. On
   * a component with none (a closed one), or when every side reaches one, it
   * is the side with the smallest world area. A loop that separates nothing,
   * such as one around a torus's tube, deletes nothing: it is imprinted, and
   * {@link TrimReport.removedArea} is `0`.
   *
   * @throws {FastMeshError} `INVALID_ARGUMENT` for too few or non-finite
   *   points, or points on several components (nothing changes);
   *   `UNKNOWN_MESH` for a removed mesh.
   */
  trimByCurve(mesh: MeshHandle, options: TrimCurveOptions): TrimReport;
  trimByCurve(
    mesh: MeshHandle,
    curve: readonly (Readonly<Vec3> | SurfacePathPoint)[],
    options?: Omit<TrimCurveOptions, "points">,
  ): TrimReport;
  trimByCurve(
    mesh: MeshHandle,
    optionsOrCurve: TrimCurveOptions | readonly (Readonly<Vec3> | SurfacePathPoint)[],
    maybeOptions?: Omit<TrimCurveOptions, "points">,
  ): TrimReport {
    const { index, generation } = this.#id(mesh);
    const norm = normalizeTrimCurveArgs(optionsOrCurve, maybeOptions);
    const flatPoints = Float64Array.from(norm.points.flat());
    const r = call(() =>
      this.#scene().trim_by_curve(
        index,
        generation,
        flatPoints,
        norm.closed,
        norm.deleteInside,
        optional(norm.snapTolerance, "snapTolerance"),
      ),
    );
    return {
      kept: this.#meshHandle(r[0], r[1]),
      trianglesAdded: r[2],
      trianglesRemoved: r[3],
      imprintTriangles: r[4],
      cutTriangles: r[5],
      removedArea: r[6],
      verticesAdded: r[7],
      seamEdges: r[8],
    };
  }

  /**
   * Fills a single boundary loop with selectable continuity (`"flat"`, `"tangent"`, or `"curvature"`)
   * and nested island detection (one undo step).
   */
  fillHoleContinuity(
    mesh: MeshHandle,
    loopIndex: number,
    options: FillHolesContinuityOptions = {},
  ): FillContinuityStats {
    const { index, generation } = this.#id(mesh);
    const loopIdx = checkU32Int(loopIndex, "loopIndex");
    const norm = normalizeFillHolesContinuityOptions(options);
    const raw = call(() =>
      this.#scene().fill_hole_continuity(
        index,
        generation,
        loopIdx,
        FILL_CONTINUITY_CODES[norm.continuity],
        optional(norm.maxHolePerimeter, "maxHolePerimeter"),
        optional(norm.maxHoleVertices, "maxHoleVertices"),
        norm.refinePatch,
        optional(norm.smoothIterations, "smoothIterations"),
        norm.detectIslands,
      ),
    );
    return decodeFillContinuityStats(raw);
  }

  /**
   * Fills all eligible boundary loops with selectable continuity (one undo step).
   */
  fillAllHolesContinuity(
    mesh: MeshHandle,
    options: FillHolesContinuityOptions = {},
  ): FillContinuityStats {
    const { index, generation } = this.#id(mesh);
    const norm = normalizeFillHolesContinuityOptions(options);
    const raw = call(() =>
      this.#scene().fill_all_holes_continuity(
        index,
        generation,
        FILL_CONTINUITY_CODES[norm.continuity],
        optional(norm.maxHolePerimeter, "maxHolePerimeter"),
        optional(norm.maxHoleVertices, "maxHoleVertices"),
        norm.refinePatch,
        optional(norm.smoothIterations, "smoothIterations"),
        norm.detectIslands,
      ),
    );
    return decodeFillContinuityStats(raw);
  }

  /**
   * Bridges two boundary arcs (resolved from `loopA`/`loopB` vertex arrays when non-empty,
   * or via `options.loopAIndex`/`arcARange` and `loopBIndex`/`arcBRange`) in one undo step.
   */
  bridgePartialArcs(
    mesh: MeshHandle,
    loopA: Indices = [],
    loopB: Indices = [],
    options: PartialBridgeOptions = {},
  ): BridgeStats {
    const { index, generation } = this.#id(mesh);
    const vertsA = normalizeU32List(loopA, "loopA");
    const vertsB = normalizeU32List(loopB, "loopB");
    const norm = normalizePartialBridgeOptions(options);
    const raw = call(() =>
      this.#scene().bridge_partial_arcs(
        index,
        generation,
        Uint32Array.from(vertsA),
        Uint32Array.from(vertsB),
        norm.loopAIndex,
        norm.arcARange[0],
        norm.arcARange[1],
        norm.loopBIndex,
        norm.arcBRange[0],
        norm.arcBRange[1],
        norm.twistOffset,
        optional(norm.subdivisions, "subdivisions"),
      ),
    );
    return {
      trianglesAdded: raw[0],
      verticesAdded: raw[1],
      resultingLoops: raw[2],
      arcAVertices: raw[3],
      arcBVertices: raw[4],
    };
  }

  /**
   * Applies localized dynamic topology remeshing inside sphere `(center, radius)`,
   * locking any `pinned` vertex IDs (one undo step).
   */
  applyDyntopo(
    mesh: MeshHandle,
    center: Readonly<Vec3>,
    radius: number,
    options: DyntopoOptions = {},
    pinned: Indices = [],
  ): DyntopoReport {
    const { index, generation } = this.#id(mesh);
    const cr = normalizeDyntopoCenterRadius(center, radius);
    const norm = normalizeDyntopoOptions(options);
    const pinnedList = normalizeU32List(pinned, "pinned");
    const raw = call(() =>
      this.#scene().apply_dyntopo(
        index,
        generation,
        Float64Array.from(cr.center),
        cr.radius,
        DYNTOPO_MODE_CODES[norm.mode],
        optional(norm.detailSize, "detailSize"),
        optional(norm.minDetailRatio, "minDetailRatio"),
        optional(norm.maxDetailRatio, "maxDetailRatio"),
        norm.equalizeValence,
        norm.smoothTangent,
        Uint32Array.from(pinnedList),
      ),
    );
    return decodeDyntopoPatch(raw, 0);
  }

  /**
   * Applies localized dynamic topology remeshing followed by a brush dab (with optional
   * `protection`), coalescing cleanly inside an active `beginStep`/`endStep`.
   */
  dyntopoDab(
    mesh: MeshHandle,
    dab: SculptBrushOp,
    options: DyntopoOptions = {},
    protection?: VertexProtection,
  ): DyntopoDabReport {
    const { index, generation } = this.#id(mesh);
    if (!isBrushSculptOp(dab)) {
      throw invalid("dyntopoDab requires a brush sculpt operation");
    }
    const norm = normalizeDyntopoOptions(options);
    const effective = protection !== undefined ? ({ ...dab, protect: protection } as SculptBrushOp) : dab;
    const encoded = encodeRepair([effective]);
    const raw = call(() =>
      this.#scene().dyntopo_dab(
        index,
        generation,
        encoded.values,
        DYNTOPO_MODE_CODES[norm.mode],
        optional(norm.detailSize, "detailSize"),
        optional(norm.minDetailRatio, "minDetailRatio"),
        optional(norm.maxDetailRatio, "maxDetailRatio"),
        norm.equalizeValence,
        norm.smoothTangent,
      ),
    );
    return {
      moved: raw[0],
      maxDisplacement: raw[1],
      ...decodeDyntopoPatch(raw, 2),
    };
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
   * @deprecated Use `scene.align.register()` instead.
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
   * Aligns `moving` onto `fixed` from N corresponding world-space landmark pairs
   * using Umeyama SVD (one undo step).
   */
  alignByLandmarks(
    moving: MeshHandle,
    fixed: MeshHandle,
    pairs: readonly LandmarkPair[],
    allowScaling = false,
  ): LandmarkAlignment {
    const m = this.#id(moving);
    const f = this.#id(fixed);
    const norm = normalizeLandmarkPairs(moving === fixed, pairs, allowScaling);
    if (this.#drag?.mesh === moving || this.#drag?.mesh === fixed) {
      this.#drag.end();
    }
    const r = call(() =>
      this.#scene().align_by_landmarks(
        m.index,
        m.generation,
        f.index,
        f.generation,
        norm.sourcePoints,
        norm.targetPoints,
        norm.allowScaling,
      ),
    );
    return {
      placement: {
        rotation: Array.from(r.subarray(0, 9)),
        translation: vec3At(r, 9),
        scale: vec3At(r, 12),
      },
      scale: r[15],
      rmse: r[16],
    };
  }

  /**
   * Runs coarse pre-alignment (PCA centroid or OBB) followed by iterative ICP
   * with selectable error objective (`pointToPoint`, `pointToPlane`, or
   * `symmetricNormal`) and applies the placement to `moving` (one undo step).
   */
  registerMultistage(
    moving: MeshHandle,
    fixed: MeshHandle,
    options: RegistrationOptions = {},
  ): RegistrationReport {
    const m = this.#id(moving);
    const f = this.#id(fixed);
    const norm = normalizeRegistrationOptions(moving === fixed, options);
    if (this.#drag?.mesh === moving || this.#drag?.mesh === fixed) {
      this.#drag.end();
    }
    const r = call(() =>
      this.#scene().register_multistage(
        m.index,
        m.generation,
        f.index,
        f.generation,
        COARSE_ALIGN_CODES[norm.coarseMethod],
        ICP_OBJECTIVE_CODES[norm.objective],
        optional(norm.maxIterations, "maxIterations"),
        optional(norm.tolerance, "tolerance"),
        optional(norm.overlap, "overlap"),
        optional(norm.maxDistance, "maxDistance"),
        optional(norm.targetRmse, "targetRmse"),
        optional(norm.sampleCount, "sampleCount"),
        optional(norm.andersonWindow, "andersonWindow"),
      ),
    );
    return {
      placement: {
        rotation: Array.from(r.subarray(0, 9)),
        translation: vec3At(r, 9),
        scale: vec3At(r, 12),
      },
      transform: Array.from(r.subarray(15, 31)),
      finalRmse: r[31],
      iterations: r[32],
      status: REGISTER_STATUSES[r[33]],
      converged: r[34] === 1,
    };
  }

  /** Detects the dominant reflection symmetry plane of `mesh` in world space. */
  detectSymmetryPlane(mesh: MeshHandle): SymmetryPlaneReport {
    const { index, generation } = this.#id(mesh);
    const r = call(() => this.#scene().detect_symmetry_plane(index, generation));
    return {
      planeNormal: vec3At(r, 0),
      planePoint: vec3At(r, 3),
      symmetryErrorRms: r[6],
      symmetryScore: r[7],
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
   * The mesh's origin. `views(mesh).positions()` are relative to it, while
   * `edit`'s `setPositions` and `addVertices` take positions with it added.
   */
  origin(mesh: MeshHandle): Vec3 {
    const { index, generation } = this.#id(mesh);
    const o = call(() => this.#peek().origin(index, generation));
    return [o[0], o[1], o[2]];
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

  /**
   * Starts (`points`) or ends (`null`) an interaction: a gesture that changes
   * a placement or a mesh many times a second (a drag, a brush stroke), during
   * which a distance field too large to recompute per update stays live near
   * the contact, and is made exact when the gesture ends.
   *
   * While it lasts, a field with more finite values than `points` updates
   * only the vertices near its moved targets (within a margin that follows
   * the work, about `points` vertices an update), the vertices an edit moved
   * and a sweep of a quarter of `points` others. The rest keep their values
   * and colours, out of date by at most how far the meshes moved since, so
   * nothing turns far meanwhile. A smaller field updates as usual. Ending it
   * recomputes the fields it left out of date: their values and colours are
   * then those of an exact update. A gesture that changed nothing costs
   * nothing. The results do not depend on the thread count.
   *
   * @throws `INVALID_ARGUMENT` unless `points` is `null` or an integer in
   * `[1, 2^32)`.
   */
  setFieldInteraction(points: number | null): void {
    const budget = checkInteractionPoints(points);
    call(() => this.#scene().set_field_interaction(budget ?? undefined));
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
        ...(c.restorable ? { restorable: true as const } : {}),
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
    // The id of a mesh removed for good never matches again: drop its
    // handle, whatever removed it. Every removal is in this message (or was
    // never reported, for a mesh added and removed in between), so removed
    // handles can go. A restorable one is kept for the way back, until the
    // history drops it.
    for (const c of raw) {
      if (c.removed && !c.restorable) {
        this.#meshes.delete(`${c.index}:${c.generation}`);
      }
    }
    this.#removed.clear();
    if (this.#prune) {
      this.#prune = false;
      for (const [key, handle] of this.#meshes) {
        const { index, generation } = meshIds.get(handle) as HandleId;
        if (!s.has_mesh(index, generation) && !s.is_restorable(index, generation)) {
          this.#meshes.delete(key);
        }
      }
    }
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
   * open step, that whole step (drag included) is closed and undone. A mesh
   * the step added undoably leaves the scene, and one it removed undoably
   * comes back (see {@link MeshChange.added}). Returns `false` when there is
   * nothing to undo.
   */
  undo(): boolean {
    const s = this.#scene();
    this.#drag?.end();
    this.#prune = true;
    const done = call(() => s.undo());
    this.#stepDepth = 0;
    return done;
  }

  /** Redoes the latest undone step. Returns `false` when there is nothing to redo. */
  redo(): boolean {
    const s = this.#scene();
    this.#drag?.end();
    this.#prune = true;
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
    this.#prune = true;
    call(() => s.clear_history());
    this.#stepDepth = 0;
  }

  /**
   * Opens a history step, or one more nesting level of the open one. Edits,
   * placement changes and undoable adds and removals until the matching
   * {@link Scene.endStep} are undone together.
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
    this.#prune = true;
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
    const collisionActive = this.hasActiveCollisionResponse(mesh);
    const outermost = this.#stepDepth === 0;
    this.beginStep();
    const depth = this.#stepDepth;
    let active = true;
    let moved = false;
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
    const applyPose = (pose: PlacementInput) => {
      if (!collisionActive) {
        this.setPlacement(mesh, pose);
        return;
      }
      moved = true;
      this.moveWithCollision(mesh, pose, false);
    };
    const applyPending = () => {
      if (pending !== undefined) {
        const pose = pending;
        pending = undefined;
        applyPose(pose);
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
          applyPose(placement);
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
            if (collisionActive && moved) {
              this.commitCollisionInteraction(mesh);
            }
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
   * @deprecated Use `scene.inspect.analyze()` or `scene.repair.analyze()` instead.
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
   *
   * @deprecated Use `scene.inspect.measure()` instead.
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
   */
  #runRepair(
    mesh: MeshHandle,
    op: RepairOp | readonly RepairOp[] | AutoRepairRequest,
  ): RepairReport | RepairReport[] | AutoRepairReport {
    const { index, generation } = this.#id(mesh);
    if (isAutoRepair(op)) {
      const noLoops = new Uint32Array();
      if (
        op.preset === undefined &&
        op.selfIntersectionMode === undefined &&
        op.voxelSize === undefined &&
        op.maxVoxels === undefined
      ) {
        const values = Float64Array.of(
          AUTO_REPAIR_CODE,
          optional(op.weldTolerance, "weldTolerance"),
          optional(op.minTriangles, "minTriangles"),
          optional(op.maxEdges, "maxEdges"),
          op.fair === undefined ? Number.NaN : flagOf(op.fair, true),
        );
        return decodeAutoReport(new Reader(call(() => this.#scene().repair(index, generation, values, noLoops))));
      }
      const preset = op.preset === undefined ? Number.NaN : codeIn(PRESET_CODES, op.preset, "preset");
      const siMode =
        op.selfIntersectionMode === undefined
          ? Number.NaN
          : codeIn(SELF_INTERSECTION_MODE_CODES, op.selfIntersectionMode, "selfIntersectionMode");
      const values = Float64Array.of(
        AUTO_REPAIR_PRESET_CODE,
        preset,
        siMode,
        optional(op.weldTolerance, "weldTolerance"),
        optional(op.minTriangles, "minTriangles"),
        optional(op.maxEdges, "maxEdges"),
        op.fair === undefined ? Number.NaN : flagOf(op.fair, true),
        optVoxelSize(op.voxelSize),
        optMaxVoxels(op.maxVoxels),
      );
      return decodeAutoReport(new Reader(call(() => this.#scene().repair(index, generation, values, noLoops))));
    }
    const ops: readonly RepairOp[] = Array.isArray(op) ? op : [op as RepairOp];
    const { values, loops } = encodeRepair(ops);
    const r = new Reader(call(() => this.#scene().repair(index, generation, values, loops)));
    const reports = ops.map((o) => decodeReport(o.op, r));
    return Array.isArray(op) ? reports : reports[0];
  }

  /**
   * Live triangle-used vertex ids of `mesh` inside the closed local-space
   * sphere `(center, radius)`, in ascending order. Brings a `"lazy"` BVH up
   * to date first; throws `STALE_BVH` on a stale `"manual"` mesh.
   *
   * @deprecated Use `scene.sculpt.verticesInSphere()` instead.
   * @throws `UNKNOWN_MESH` for a removed handle; `INVALID_ARGUMENT` when
   * `center` is not 3 finite numbers or `radius` is negative or non-finite;
   * `STALE_BVH` for a stale `"manual"` mesh.
   */
  verticesInSphere(mesh: MeshHandle, center: Readonly<Vec3>, radius: number): Uint32Array {
    const { index, generation } = this.#id(mesh);
    const c = center as readonly number[] | undefined;
    if (c?.length !== 3 || ![c[0], c[1], c[2]].every((x) => typeof x === "number" && Number.isFinite(x))) {
      throw invalid("center must be 3 finite numbers");
    }
    if (typeof radius !== "number" || !Number.isFinite(radius) || radius < 0) {
      throw invalid("radius must be a finite non-negative number");
    }
    return call(() => this.#scene().vertices_in_sphere(index, generation, c[0], c[1], c[2], radius));
  }

  // ---- Queries (world space) ----

  /**
   * The nearest hit of a world ray over every mesh, or `null`. `direction`
   * need not be unit length; `distance` is in world units. Lazy meshes are
   * brought up to date first.
   *
   * @deprecated Use `scene.inspect.raycast()` instead.
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
   *
   * @deprecated Use `scene.inspect.closestPoint()` instead.
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
   *
   * @deprecated Use `scene.inspect.meshDistance()` instead.
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

  /**
   * Whether two meshes intersect or touch as placed.
   *
   * @deprecated Use `scene.inspect.intersects()` instead.
   */
  intersects(a: MeshHandle, b: MeshHandle): boolean {
    const ia = this.#id(a);
    const ib = this.#id(b);
    return call(() => this.#scene().intersects(ia.index, ia.generation, ib.index, ib.generation));
  }

  /**
   * The live triangle slots of `mesh` that intersect or touch another of its
   * own triangles, ascending, each once. Triangles that share a vertex are
   * never tested against each other (neighbours always touch), and
   * degenerate triangles are skipped, as for
   * {@link AnalyzeOptions.selfIntersection}. The placement does not change
   * the answer. The mesh's BVH is brought up to date first.
   *
   * @throws `STALE_BVH` for a `"manual"` mesh whose BVH is stale.
   */
  selfIntersectingTriangles(mesh: MeshHandle): Uint32Array {
    const { index, generation } = this.#id(mesh);
    return call(() => this.#scene().self_intersecting_triangles(index, generation));
  }

  /**
   * Every mesh pair (or every pair of `pairs`) within `margin`, sorted by mesh id.
   *
   * @deprecated Use `scene.inspect.findContacts()` instead.
   */
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

  /** Evaluates printability overhang angles and support surface area in world coordinates. */
  overhangs(mesh: MeshHandle, options: OverhangOptions = {}): OverhangReport {
    const { index, generation } = this.#id(mesh);
    const norm = normalizeOverhangOptions(options);
    const r = call(() =>
      this.#scene().overhangs(
        index,
        generation,
        Float64Array.from(norm.buildDirection),
        optional(norm.criticalAngleDeg, "criticalAngleDeg"),
      ),
    );
    return {
      totalSupportArea: r[0],
      overhangFaceCount: r[1],
      maxOverhangAngleDeg: r[2],
      vertexAnglesDeg: Float32Array.from(r.subarray(3)),
    };
  }

  /** Evaluates triangle element quality metrics across `mesh` in world coordinates. */
  triangleQuality(
    mesh: MeshHandle,
    metricOrOptions: QualityMetric | TriangleQualityOptions = "radiusRatio",
    sliverThreshold?: number,
  ): QualityReport {
    const { index, generation } = this.#id(mesh);
    const norm = normalizeQualityArgs(metricOrOptions, sliverThreshold);
    const r = call(() =>
      this.#scene().triangle_quality(index, generation, QUALITY_METRIC_CODES[norm.metric], norm.sliverThreshold),
    );
    return {
      min: r[0],
      max: r[1],
      mean: r[2],
      sliverTriangleCount: r[3],
      liveTriangleCount: r[4],
      vertexQuality: Float32Array.from(r.subarray(5)),
    };
  }

  /** Computes vertex-to-surface Hausdorff / signed deviation from `source` to `target` in world units. */
  meshDeviation(source: MeshHandle, target: MeshHandle, options: DeviationOptions = {}): DeviationStats {
    const src = this.#id(source);
    const tgt = this.#id(target);
    const norm = normalizeDeviationOptions(options);
    const r = call(() =>
      this.#scene().mesh_deviation(
        src.index,
        src.generation,
        tgt.index,
        tgt.generation,
        norm.signed,
        optional(norm.maxDistance, "maxDistance"),
      ),
    );
    return {
      min: r[0],
      max: r[1],
      mean: r[2],
      rms: r[3],
      p95: r[4],
      sampleCount: r[5],
      vertexDeviations: Float32Array.from(r.subarray(6)),
    };
  }

  /**
   * Computes per-vertex surface geodesic distances from `sources` in world
   * units. See {@link InspectDomain.geodesicDistances}.
   */
  geodesicDistances(
    mesh: MeshHandle,
    sources: number | readonly number[],
    options: GeodesicOptions = {},
  ): Float32Array {
    const { index, generation } = this.#id(mesh);
    const src = normalizeSourceVertices(sources);
    const norm = normalizeGeodesicOptions(options);
    throwIfAborted(options.signal);
    return call(() =>
      this.#scene().geodesic_distances(
        index,
        generation,
        src,
        GEODESIC_METHOD_CODES[norm.method],
        optional(norm.maxDistance, "maxDistance"),
        optional(norm.timeStepScale, "timeStepScale"),
      ),
    );
  }

  /** Frees what `"heatMethod"` queries keep for the next one of the same mesh. */
  releaseGeodesics(): void {
    call(() => this.#peek().release_geodesics());
  }

  /** Traces the intrinsic shortest surface path from `sources` to `targetVertex` in world coordinates. */
  shortestSurfacePath(
    mesh: MeshHandle,
    sources: number | readonly number[],
    targetVertex: number,
    options: SurfacePathOptions = {},
  ): SurfacePathPoint[] {
    const { index, generation } = this.#id(mesh);
    const src = normalizeSourceVertices(sources);
    const target = normalizeSurfacePathTarget(targetVertex);
    const { distances, geodesic } = checkSurfacePathOptions(options);
    throwIfAborted(options.signal);
    const dists =
      distances ??
      call(() =>
        this.#scene().geodesic_distances(
          index,
          generation,
          src,
          GEODESIC_METHOD_CODES[geodesic.method],
          optional(geodesic.maxDistance, "maxDistance"),
          optional(geodesic.timeStepScale, "timeStepScale"),
        ),
      );
    const r = call(() => this.#scene().shortest_surface_path(index, generation, dists, src, target));
    const points: SurfacePathPoint[] = [];
    for (let at = 0; at < r.length; at += 7) {
      points.push({
        position: vec3At(r, at),
        triangle: r[at + 3],
        barycentric: vec3At(r, at + 4),
      });
    }
    return points;
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
   * @deprecated Use `scene.inspect.addDistanceField()` instead.
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
   * @deprecated Use `scene.inspect.addThicknessField()` instead.
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
   * @deprecated Use `scene.inspect.addCurvatureField()` instead.
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

  // ---- Stepped geodesic queries (internal: see geodesicSteps) ----

  static {
    geodesicStepsOf = (scene) => ({
      begin: (mesh, sources, options = {}) => {
        const { index, generation } = scene.#id(mesh);
        const src = normalizeSourceVertices(sources);
        const norm = normalizeGeodesicOptions(options);
        return call(() =>
          scene
            .#peek()
            .begin_geodesic_distances(
              index,
              generation,
              src,
              GEODESIC_METHOD_CODES[norm.method],
              optional(norm.maxDistance, "maxDistance"),
              optional(norm.timeStepScale, "timeStepScale"),
            ),
        );
      },
      step: (token, work) => call(() => scene.#peek().step_geodesic_distances(token, work)),
      finish: (token) => call(() => scene.#peek().finish_geodesic_distances(token)),
      drop: (token) => call(() => scene.#peek().drop_geodesic_distances(token)),
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

  #hasNormals(index: number, generation: number): boolean {
    const s = this.#peek();
    return s.normals_len(index, generation) > 0 || s.vertex_count(index, generation) === 0;
  }

  // ---- #233 Face Selection ----

  /** Selects triangles enclosed by or intersecting a screen-space rectangle or lasso in NDC. */
  selectFacesScreen(mesh: MeshHandle, shape: ScreenSelectionShape, options: ScreenSelectOptions): Uint32Array {
    const { index, generation } = this.#id(mesh);
    const n = normalizeScreenSelectArgs(shape, options);
    return call(() =>
      this.#scene().select_faces_screen(
        index,
        generation,
        n.shapeKind,
        n.shapeData,
        n.viewProj,
        Float64Array.from(n.cameraOrigin),
        Float64Array.from(n.cameraForward),
        n.isOrthographic,
        n.visibleOnly,
        n.rejectBackfaces,
        n.includeTouching,
      ),
    );
  }

  /** Selects triangles intersecting a world-space sphere (optionally restricted by a normal cone). */
  selectFacesInSphere(
    mesh: MeshHandle,
    center: Readonly<Vec3>,
    radius: number,
    normalCone?: FaceSelectionNormalCone,
  ): Uint32Array {
    const { index, generation } = this.#id(mesh);
    const n = normalizeSelectInSphereArgs(center, radius, normalCone);
    return call(() =>
      this.#scene().select_faces_in_sphere(index, generation, n.center, n.radius, n.normal, n.coneAngleRad),
    );
  }

  /** Flood-fills triangles across edges whose dihedral angle does not exceed `maxDihedralRad`. */
  selectByCreaseAngle(mesh: MeshHandle, seedTri: number, maxDihedralRad: number): Uint32Array {
    const { index, generation } = this.#id(mesh);
    return call(() =>
      this.#peek().select_by_crease_angle(
        index,
        generation,
        checkU32Arg(seedTri, "seedTri"),
        checkNonNegativeArg(maxDihedralRad, "maxDihedralRad"),
      ),
    );
  }

  /** Selects the edge-connected shell containing `seedTri`. */
  selectConnectedShell(mesh: MeshHandle, seedTri: number): Uint32Array {
    const { index, generation } = this.#id(mesh);
    return call(() => this.#peek().select_connected_shell(index, generation, checkU32Arg(seedTri, "seedTri")));
  }

  /** Selects small edge-connected face islands bounded by `maxFaceCount` and optional `maxArea`. */
  selectSmallFaceIslands(mesh: MeshHandle, maxFaceCount: number, maxArea?: number): Uint32Array {
    const { index, generation } = this.#id(mesh);
    const n = normalizeSelectSmallIslandsArgs(maxFaceCount, maxArea);
    return call(() => this.#peek().select_small_face_islands(index, generation, n.maxFaceCount, n.maxArea));
  }

  /** Expands `selectedTris` outward by `rings` topological rings. */
  growFaceSelection(
    mesh: MeshHandle,
    selectedTris: Indices,
    rings = 1,
    adjacency: AdjacencyKind = "edge",
  ): Uint32Array {
    const { index, generation } = this.#id(mesh);
    const n = normalizeGrowShrinkArgs(selectedTris, rings, adjacency);
    return call(() => this.#peek().grow_face_selection(index, generation, n.selectedTris, n.rings, n.adjacency));
  }

  /** Erodes `selectedTris` inward by `rings` topological rings. */
  shrinkFaceSelection(
    mesh: MeshHandle,
    selectedTris: Indices,
    rings = 1,
    adjacency: AdjacencyKind = "edge",
  ): Uint32Array {
    const { index, generation } = this.#id(mesh);
    const n = normalizeGrowShrinkArgs(selectedTris, rings, adjacency);
    return call(() => this.#peek().shrink_face_selection(index, generation, n.selectedTris, n.rings, n.adjacency));
  }

  /** Fills interior unselected triangle holes up to `maxHoleTris`. */
  fillSelectionHoles(mesh: MeshHandle, selectedTris: Indices, maxHoleTris: number): Uint32Array {
    const { index, generation } = this.#id(mesh);
    return call(() =>
      this.#peek().fill_selection_holes(
        index,
        generation,
        normalizeU32Selection(selectedTris),
        checkU32Arg(maxHoleTris, "maxHoleTris"),
      ),
    );
  }

  /** Smooths jagged boundaries of a face selection via majority-neighbor voting. */
  smoothSelectionBoundary(mesh: MeshHandle, selectedTris: Indices, iterations = 1): Uint32Array {
    const { index, generation } = this.#id(mesh);
    return call(() =>
      this.#peek().smooth_selection_boundary(
        index,
        generation,
        normalizeU32Selection(selectedTris),
        checkU32Arg(iterations, "iterations"),
      ),
    );
  }

  /** Converts a face selection into per-vertex `[0, 1]` coverage weights aligned with `vertexSlots()`. */
  faceSelectionToVertexMask(mesh: MeshHandle, selectedTris: Indices): Float32Array {
    const { index, generation } = this.#id(mesh);
    return call(() => this.#peek().face_selection_to_vertex_mask(index, generation, normalizeU32Selection(selectedTris)));
  }

  /** Converts a per-vertex scalar mask into selected triangle indices. */
  vertexMaskToFaceSelection(
    mesh: MeshHandle,
    mask: Float32Array | ArrayLike<number>,
    threshold = 0.5,
    requireAllVertices = false,
  ): Uint32Array {
    const { index, generation } = this.#id(mesh);
    const n = normalizeVertexMaskSelectArgs(mask, threshold, requireAllVertices);
    return call(() =>
      this.#peek().vertex_mask_to_face_selection(index, generation, n.mask, n.threshold, n.requireAllVertices),
    );
  }

  /** Deletes `selectedTris` from `mesh` in one undo step. */
  deleteSelectedFaces(mesh: MeshHandle, selectedTris: Indices): FaceEditReport {
    const { index, generation } = this.#id(mesh);
    const raw = call(() => this.#scene().delete_selected_faces(index, generation, normalizeU32Selection(selectedTris)));
    return decodeFaceEditReport(raw);
  }

  /**
   * Extrudes `triangles` of `mesh`: moves them by `options.distance` (world
   * units) and builds side walls between where the selection's boundary was
   * and where it went, as one undo step. A closed, consistently wound mesh
   * stays closed and consistently wound.
   *
   * Each region of the selection moves along its average normal by default;
   * see {@link ExtrudeFacesOptions.direction}. Selections with holes get walls
   * around them, and regions that touch at a vertex separate there. Selected
   * edges on the mesh's open boundary get walls too. The triangles keep their
   * slots and are the moved faces, so the same selection extrudes them again.
   *
   * Self-intersections of the result are reported, not refused: pushing a
   * face through the other side of a thin part, say, gives a closed mesh that
   * crosses itself.
   *
   * @throws {FastMeshError} `INVALID_ARGUMENT` for an empty selection, a slot
   *   without a live triangle, a distance or direction that is zero or not
   *   finite, a selected edge shared by three or more triangles, selected
   *   triangles wound against each other, or a region whose normals cancel out
   *   (give a direction then); `UNKNOWN_MESH` for a removed mesh. The mesh is
   *   unchanged on error.
   */
  extrudeFaces(mesh: MeshHandle, triangles: Indices, options: ExtrudeFacesOptions): ExtrudeFacesResult {
    const { index, generation } = this.#id(mesh);
    const tris = normalizeU32Selection(triangles, "triangles");
    const { distance, mode, direction, check } = extrudeFacesArgs(options);
    return decodeExtrudeFaces(
      call(() => this.#scene().extrude_faces(index, generation, tris, distance, mode, direction, check)),
    );
  }

  /**
   * Turns an open surface into a solid, as one undo step: runs boundary loops
   * of `mesh` along a world direction (down by default) to a flat plane
   * perpendicular to it, joins each with a wall and closes it with a flat cap.
   * The plane is at `options.height`, or `options.offset` past the mesh's
   * furthest vertex.
   *
   * By default only the longest loop is extruded, so fill smaller holes first
   * (or pass `loops: "all"`, which makes them holes through the solid). A
   * surface that faces the base is turned around, so that the solid faces
   * outward ({@link ExtrudeBaseResult.flipped}).
   *
   * Self-intersections of the result are reported, not refused: a plane that
   * cuts through the surface, say.
   *
   * @throws {FastMeshError} `INVALID_ARGUMENT` when the mesh has no boundary
   *   loop, a loop is open or not clean (orient or repair the mesh first), a
   *   boundary vertex does not lie before the plane, or a loop's outline on
   *   the plane crosses itself; `STALE_LOOPS` for a loop id that matches no
   *   loop; `UNKNOWN_MESH` for a removed mesh. The mesh is unchanged on error.
   */
  extrudeBase(mesh: MeshHandle, options: ExtrudeBaseOptions): ExtrudeBaseResult {
    const { index, generation } = this.#id(mesh);
    const a = extrudeBaseArgs(options);
    return decodeExtrudeBase(
      call(() =>
        this.#scene().extrude_base(index, generation, a.direction, a.plane, a.value, a.which, a.loops, a.refine, a.check),
      ),
    );
  }

  /**
   * Extracts `selectedTris` into a new standalone mesh preserving exact world
   * coordinates, as one undo step: {@link Scene.undo} removes the new mesh and
   * {@link Scene.redo} brings it back with the same handle.
   */
  extractSelectedFaces(mesh: MeshHandle, selectedTris: Indices, options: { normals?: boolean } = {}): MeshHandle {
    const { index, generation } = this.#id(mesh);
    const normals = checkBooleanArg(options.normals, this.#hasNormals(index, generation), "normals");
    const id = call(() =>
      this.#scene().extract_selected_faces(index, generation, normalizeU32Selection(selectedTris), normals),
    );
    return this.#meshHandle(id[0], id[1]);
  }

  /**
   * Splits `selectedTris` off into a new detached mesh and removes them from
   * `mesh`, as one undo step: {@link Scene.undo} makes `mesh` whole again and
   * removes the new mesh, which {@link Scene.redo} brings back with the same
   * handle.
   */
  splitSelectedFaces(mesh: MeshHandle, selectedTris: Indices, options: { normals?: boolean } = {}): SplitFacesReport {
    const { index, generation } = this.#id(mesh);
    const normals = checkBooleanArg(options.normals, this.#hasNormals(index, generation), "normals");
    const raw = call(() =>
      this.#scene().split_selected_faces(index, generation, normalizeU32Selection(selectedTris), normals),
    );
    return {
      detached: this.#meshHandle(raw[0], raw[1]),
      removedTriangles: raw[2],
      removedVertices: raw[3],
    };
  }

  // ---- #193 Free-Form Deformation ----

  /** Builds a zero-offset world-aligned FFD lattice around `mesh`. */
  buildFfdLattice(
    mesh: MeshHandle,
    dims: readonly [number, number, number],
    basis: FfdBasisInput = "bernstein",
    padding = 0,
  ): FfdLatticeData {
    const { index, generation } = this.#id(mesh);
    const n = normalizeBuildFfdLatticeArgs(dims, basis, padding);
    return decodeFfdLattice(call(() => this.#peek().build_ffd_lattice(index, generation, n.dims, n.basis, n.padding)));
  }

  /** Applies Free-Form Deformation (FFD) in one undo step. */
  applyFfd(mesh: MeshHandle, options: FfdOptions): FreeformReport {
    const { index, generation } = this.#id(mesh);
    const n = normalizeFfdArgs(options);
    return decodeFreeformReport(
      call(() =>
        this.#scene().apply_ffd(
          index,
          generation,
          n.dims,
          n.basis,
          n.frame15,
          n.offsets,
          n.outsideMode,
          n.regionCenter,
          n.regionRadius,
          n.vertices,
          n.protectedVerts,
          n.protectionWeights,
        ),
      ),
    );
  }

  /** Applies As-Rigid-As-Possible (ARAP) surface deformation in one undo step. */
  applyArap(mesh: MeshHandle, options: ArapOptions): FreeformReport {
    const { index, generation } = this.#id(mesh);
    const n = normalizeArapArgs(options);
    return decodeFreeformReport(
      call(() =>
        this.#scene().apply_arap(
          index,
          generation,
          n.anchorVerts,
          n.anchorPos,
          n.anchorHasPos,
          n.handleVerts,
          n.handleTargets,
          n.weights,
          n.solver,
          n.iterations,
          n.tolerance,
          n.regionCenter,
          n.regionRadius,
          n.vertices,
          n.protectedVerts,
          n.protectionWeights,
        ),
      ),
    );
  }

  /** Applies a regularized elastic Kelvinlet deformation in one undo step. */
  applyKelvinlet(mesh: MeshHandle, options: KelvinletOptions): FreeformReport {
    const { index, generation } = this.#id(mesh);
    const n = normalizeKelvinletArgs(options);
    return decodeFreeformReport(
      call(() =>
        this.#scene().apply_kelvinlet(
          index,
          generation,
          n.center,
          n.loadKind,
          n.loadData,
          n.epsilon,
          n.nu,
          n.mu,
          n.scaleMode,
          n.scaleAlpha,
          n.symX,
          n.symY,
          n.symZ,
          n.regionCenter,
          n.regionRadius,
          n.vertices,
          n.protectedVerts,
          n.protectionWeights,
        ),
      ),
    );
  }

  /** Runs an XPBD soft-body solve with optional self-collision handling in one undo step. */
  applyXpbd(mesh: MeshHandle, options: XpbdOptions): FreeformReport {
    const { index, generation } = this.#id(mesh);
    const n = normalizeXpbdArgs(options);
    return decodeFreeformReport(
      call(() =>
        this.#scene().apply_xpbd(
          index,
          generation,
          n.substeps,
          n.iterations,
          n.dt,
          n.stretchCompliance,
          n.bendCompliance,
          n.volumeCompliance,
          n.damping,
          n.gravity,
          n.pinVerts,
          n.pinTargets,
          n.handleVerts,
          n.handleTargets,
          n.handleCompliances,
          n.selfCollision,
          n.collisionThickness,
          n.protectedVerts,
          n.protectionWeights,
        ),
      ),
    );
  }

  // ---- #194 Inter-Object Collision ----

  /** Sets the scene's fallback inter-object collision rule. */
  setDefaultCollisionRule(rule: CollisionRuleInput): void {
    const flat = encodeCollisionRule(normalizeCollisionRule(rule));
    call(() => this.#scene().set_default_collision_rule(flat));
  }

  /** Returns the scene's fallback inter-object collision rule. */
  defaultCollisionRule(): CollisionRule {
    return decodeCollisionRule(call(() => this.#peek().default_collision_rule()));
  }

  /** Configures the collision response rule for unordered pair `(a, b)`. */
  setCollisionRule(a: MeshHandle, b: MeshHandle, rule: CollisionRuleInput): void {
    const ia = this.#id(a);
    const ib = this.#id(b);
    const flat = encodeCollisionRule(normalizeCollisionRule(rule));
    call(() => this.#scene().set_collision_rule(ia.index, ia.generation, ib.index, ib.generation, flat));
  }

  /** Clears any pair-specific collision override for `(a, b)`, returning `true` if one existed. */
  clearCollisionRule(a: MeshHandle, b: MeshHandle): boolean {
    const ia = this.#id(a);
    const ib = this.#id(b);
    return call(() => this.#scene().clear_collision_rule(ia.index, ia.generation, ib.index, ib.generation));
  }

  /** Returns the effective collision rule for `(a, b)`. */
  collisionRule(a: MeshHandle, b: MeshHandle): CollisionRule {
    const ia = this.#id(a);
    const ib = this.#id(b);
    return decodeCollisionRule(call(() => this.#peek().collision_rule(ia.index, ia.generation, ib.index, ib.generation)));
  }

  /** True when `mesh` participates in at least one active non-passThrough collision rule. */
  hasActiveCollisionResponse(mesh: MeshHandle): boolean {
    const { index, generation } = this.#id(mesh);
    return call(() => this.#peek().has_active_collision_response(index, generation));
  }

  /** Moves `mesh` toward `placement` with inter-object collision detection and response. */
  moveWithCollision(mesh: MeshHandle, placement: PlacementInput, isRelease = false): CollisionStepReport {
    const { index, generation } = this.#id(mesh);
    const n = normalizePlacementInputArgs(placement);
    const scale: Vec3 = n.scale ? [n.scale[0], n.scale[1], n.scale[2]] : (this.placement(mesh).scale ?? [1, 1, 1]);
    const raw = call(() =>
      this.#scene().move_with_collision(
        index,
        generation,
        n.rotation,
        n.translation,
        n.scale,
        checkBooleanArg(isRelease, false, "isRelease"),
      ),
    );
    return decodeCollisionStepReport(raw, (i, g) => this.#meshHandle(i, g), scale);
  }

  /** Finalizes queued release actions (deferred cuts, elastic recovery) for `mesh`. */
  commitCollisionInteraction(mesh: MeshHandle): CollisionStepReport {
    const { index, generation } = this.#id(mesh);
    const scale = this.placement(mesh).scale;
    const raw = call(() => this.#scene().commit_collision_interaction(index, generation));
    return decodeCollisionStepReport(raw, (i, g) => this.#meshHandle(i, g), scale);
  }

  // ---- #234 Primitive Fit, Contours & Point Cloud ----

  /** Fits a geometric primitive (`plane`, `sphere`, `cylinder`, `cone`, or `auto`) in world space. */
  fitPrimitive(mesh: MeshHandle, options: FitPrimitiveOptions = {}): FittedPrimitiveReport {
    const { index, generation } = this.#id(mesh);
    const n = normalizeFitPrimitiveArgs(options);
    return decodeFitPrimitiveReport(
      call(() =>
        this.#peek().fit_primitive(
          index,
          generation,
          n.subsetTris,
          n.kind,
          n.inlierThreshold,
          n.ransacIterations,
          n.lmIterations,
          n.seed,
          n.minInlierRatio,
          n.normalWeight,
        ),
      ),
    );
  }

  /** Rigidly aligns `mesh`'s fitted datum onto `targetAxis` and `targetOrigin` in one undo step. */
  alignToFittedPrimitive(
    mesh: MeshHandle,
    primitive: FittedPrimitive,
    targetAxis: Readonly<Vec3>,
    targetOrigin: Readonly<Vec3>,
  ): Placement {
    const { index, generation } = this.#id(mesh);
    const n = normalizeAlignToPrimitiveArgs(primitive, targetAxis, targetOrigin);
    return decodePlacement15(
      call(() => this.#scene().align_to_fitted_primitive(index, generation, n.prim9, n.targetAxis, n.targetOrigin)),
    );
  }

  /** Slices world-space planar contours through `mesh`. */
  sliceContours(mesh: MeshHandle, origin: Readonly<Vec3>, normal: Readonly<Vec3>): ContourSlice {
    const { index, generation } = this.#id(mesh);
    const n = normalizeSliceContoursArgs(origin, normal);
    return decodeContourSlices(call(() => this.#peek().slice_contours(index, generation, n.origin, n.normal)))[0];
  }

  /** Slices an evenly spaced stack of `count` parallel contours through `mesh`. */
  sliceContourStack(
    mesh: MeshHandle,
    origin: Readonly<Vec3>,
    normal: Readonly<Vec3>,
    spacing: number,
    count: number,
  ): ContourSlice[] {
    const { index, generation } = this.#id(mesh);
    const n = normalizeSliceContourStackArgs(origin, normal, spacing, count);
    return decodeContourSlices(
      call(() => this.#peek().slice_contour_stack(index, generation, n.origin, n.normal, n.spacing, n.count)),
    );
  }

  /**
   * Reconstructs a triangle mesh from a packed point cloud and adds it to the
   * scene, as one undo step: {@link Scene.undo} removes it and
   * {@link Scene.redo} brings it back with the same handle.
   */
  reconstructPointCloud(
    cloud: PackedPointCloud | Float32Array,
    options: SurfaceReconstructOptions = {},
  ): MeshHandle {
    const buf = cloudBuffer(cloud);
    const n = normalizeReconstructArgs(options);
    const id = call(() =>
      this.#scene().reconstruct_point_cloud(buf, n.method, n.voxelSize, n.ballRadius, n.kNeighbors, n.normals),
    );
    return this.#meshHandle(id[0], id[1]);
  }

  // ---- #236 Hollow, Drain Holes & Place Mesh ----

  /** Hollows `mesh` with optional per-vertex wall thickness modulation in one undo step. */
  hollowMesh(mesh: MeshHandle, options: HollowOptions): HollowReport {
    const { index, generation } = this.#id(mesh);
    const n = normalizeHollowArgs(options);
    return decodeHollowReport(
      call(() =>
        this.#scene().hollow_mesh(
          index,
          generation,
          n.wallThickness,
          n.backend,
          n.voxelSize,
          n.maxVoxels,
          n.maxCavityTriangles,
          n.minCavityVolume,
          n.modulationVerts,
          n.modulationWeights,
        ),
      ),
    );
  }

  /**
   * Plans the voxel grid of {@link Scene.latticeFill} for the same mesh and
   * options without running it: its dimensions, voxel size, voxel count,
   * estimated triangles and memory, the share of space the walls fill, and
   * whether the grid is within `maxVoxels`. Costs a few passes over the mesh
   * (tens of milliseconds at a million triangles), so it can follow a
   * settings form.
   *
   * @throws {FastMeshError} as {@link Scene.latticeFill}, except that the mesh
   *   is not checked for being closed, and that a grid over the limit comes
   *   back with `fits: false`.
   */
  latticePlan(mesh: MeshHandle, options: LatticeOptions): LatticePlan {
    const { index, generation } = this.#id(mesh);
    const values = encodeLatticeOptions(checkLatticeOptions(options, DEFAULT_LATTICE_MAX_VOXELS));
    return decodeLatticePlan(call(() => this.#scene().lattice_plan(index, generation, values)));
  }

  /**
   * Fills the inside of the closed, outward-wound `mesh`, or the cavity of a
   * hollowed one, with a lattice: walls of uniform thickness that follow a
   * gyroid, Schwarz P or diamond surface. One undo step.
   *
   * - Without a shell, filling the inside turns the mesh into an open
   *   lattice, cut off at its surface.
   * - With `shellThickness`, the mesh's surface is kept as it is, and the
   *   space between the walls is carved out from that depth on, so shell and
   *   walls are one solid.
   * - In a cavity, the walls grow out of the shell that is there, whose outer
   *   surface is kept as it is.
   *
   * The result is closed, manifold and free of self-intersections, as far as
   * the surface that is kept is. It is built on a voxel grid that follows
   * from the lengths, or from `voxelSize`: edges where walls meet the mesh's
   * surface are cut by a fraction of a voxel. {@link Scene.latticePlan}
   * reports the grid, the triangles and the memory beforehand, and
   * {@link Scene.latticePreview} shows the walls on a small grid.
   *
   * Lengths and volumes are measured in the world. The cells are aligned with
   * the mesh's own axes, around the middle of its bounds. Its UVs are
   * dropped. A lattice has many triangles (about 2,000 per gyroid cell at ten
   * voxels a cell, 12,000 at 25) and takes seconds at a few million: use
   * `WorkerScene.latticeFill` to keep the page live.
   *
   * @throws {FastMeshError} `INVALID_ARGUMENT` for bad options, a mesh that is
   *   not closed and consistently wound outward, `region: "cavity"` without
   *   a cavity, a grid over `maxVoxels` (the message has its size and
   *   memory) or too coarse for the walls, the shell, the cells or the wall
   *   around a cavity, or a lattice that leaves nothing, or nothing open;
   *   `UNKNOWN_MESH` for a removed mesh; `OUT_OF_MEMORY`. The mesh is
   *   unchanged on error.
   */
  latticeFill(mesh: MeshHandle, options: LatticeOptions): LatticeReport {
    const { index, generation } = this.#id(mesh);
    const values = encodeLatticeOptions(checkLatticeOptions(options, DEFAULT_LATTICE_MAX_VOXELS));
    return decodeLatticeReport(call(() => this.#scene().lattice_fill(index, generation, values)));
  }

  /**
   * The walls {@link Scene.latticeFill} would put into `mesh`, on a small
   * grid, without changing the mesh: buffers in its local frame, to draw at
   * its `matrix4`. The grid has at most `maxVoxels` voxels (default
   * {@link DEFAULT_LATTICE_PREVIEW_VOXELS}); walls it is too coarse for are
   * drawn two voxels thick (`plan.wallThickness`), and a part with too many
   * cells for it shows its middle only (`plan.cropped`). With a shell the
   * walls start at its inner face.
   *
   * @throws {FastMeshError} as {@link Scene.latticeFill}, except that the grid
   *   always fits and that a lattice without walls is an empty preview.
   */
  latticePreview(mesh: MeshHandle, options: LatticeOptions): LatticePreview {
    const { index, generation } = this.#id(mesh);
    const checked = checkLatticeOptions(options, DEFAULT_LATTICE_PREVIEW_VOXELS);
    const values = encodeLatticeOptions(checked);
    return decodeLatticePreview(
      call(() => this.#scene().lattice_preview(index, generation, values, checked.maxVoxels)),
    );
  }

  /**
   * Punches a tapered/chamfered drain hole in `mesh` (with optional plug
   * generation) in one undo step: {@link Scene.undo} fills the hole again and
   * removes the plug, which {@link Scene.redo} brings back with the same
   * handle.
   */
  punchDrainHole(mesh: MeshHandle, options: DrainHoleOptions): DrainHoleReport {
    const { index, generation } = this.#id(mesh);
    const n = normalizeDrainHoleArgs(options);
    const plugNormals = n.plugNormals ?? this.#hasNormals(index, generation);
    const raw = call(() =>
      this.#scene().punch_drain_hole(
        index,
        generation,
        n.surfacePoint,
        n.surfaceNormal,
        n.diameter,
        n.depth,
        n.outerChamfer,
        n.taperAngleDeg,
        n.segments,
        n.booleanMethod,
        n.fallbackBvhSdf,
        n.generatePlug,
        n.plugClearance,
        plugNormals,
      ),
    );
    return decodeDrainHoleReport(raw, (i, g) => this.#meshHandle(i, g));
  }

  /**
   * Places `stamp` onto `target` at a hit point/normal, optionally unioning/engraving/conformance-wrapping,
   * in one undo step. Kept separate, the placed stamp is a new mesh with the hit point as its
   * {@link Scene.origin}, which {@link Scene.undo} removes and {@link Scene.redo} brings back with
   * the same handle.
   */
  placeMesh(target: MeshHandle, stamp: MeshHandle, options: PlaceMeshOptions): PlaceMeshReport {
    const t = this.#id(target);
    const s = this.#id(stamp);
    const n = normalizePlaceMeshArgs(options);
    const normals = n.normals ?? this.#hasNormals(s.index, s.generation);
    const raw = call(() =>
      this.#scene().place_mesh(
        t.index,
        t.generation,
        s.index,
        s.generation,
        n.hitPoint,
        n.hitNormal,
        n.upAxis,
        n.twistAngleRad,
        n.uniformScale,
        n.embedDepth,
        n.mode,
        n.conformToSurface,
        n.booleanMethod,
        normals,
      ),
    );
    return decodePlaceMeshReport(raw, (i, g) => this.#meshHandle(i, g));
  }
}
