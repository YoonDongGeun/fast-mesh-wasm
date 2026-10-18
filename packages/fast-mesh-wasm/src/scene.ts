// The public `Scene` class: editable meshes with placements, zero-copy render
// views, per-frame change records and undo history. Wraps the raw `WasmScene`
// binding; every call into WebAssembly goes through `call` (error codes and
// panic poisoning).
import type { Indices, Mat3, Placement, PlacementInput, Positions64, Quat, UpdateMode, Vec3 } from "./api.js";
import { decodeChanges, type Range } from "./changes.js";
import { call, ensureInitialized, FastMeshError, finalizer, release, wasmMemory } from "./core.js";
import { type MeshViews, meshViews, ViewGuard } from "./views.js";
import { WasmScene } from "./wasm/fast_mesh_wasm.js";

export type { Range } from "./changes.js";
export type { MeshViews } from "./views.js";

/** The id behind a handle and the scene it belongs to. Kept here, not on the handle. */
interface HandleId {
  scene: Scene;
  index: number;
  generation: number;
}

const meshIds = new WeakMap<MeshHandle, HandleId>();
const rawIds = new WeakMap<FieldHandle | SectionHandle, HandleId>();

/**
 * A mesh in a {@link Scene}. Opaque: pass it to the scene's methods. The same
 * mesh always gives the same handle object, so handles compare with `===`.
 */
export class MeshHandle {
  declare private readonly brand: "MeshHandle";
}

/** A field in a {@link Scene}. Opaque and deduplicated like {@link MeshHandle}. */
export class FieldHandle {
  declare private readonly brand: "FieldHandle";
}

/** A section in a {@link Scene}. Opaque and deduplicated like {@link MeshHandle}. */
export class SectionHandle {
  declare private readonly brand: "SectionHandle";
}

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
  /** Its vertex or triangle slot count changed: recreate its GPU buffers. */
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
 * away without it: an `undo()`, `redo()` or `clearHistory()` ends it first, and
 * a `cancelStep()` (or an `endStep()` that closes the drag's step) abandons it.
 */
export interface DragSession {
  /** Sets the mesh's placement. Throws `INVALID_ARGUMENT` once the drag is over. */
  move(placement: PlacementInput): void;
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
   * Development checks: warn once when a render view was kept across a call
   * that grew the WebAssembly memory (it is then detached). Off by default.
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

/** The scene's view of its active drag. */
interface ActiveDrag {
  /** The step depth right after the drag's `beginStep`. */
  depth: number;
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
  const values = [
    rule.range[0],
    rule.range[1],
    rule.center ?? Number.NaN,
    codeIn(OUT_OF_RANGE, rule.below ?? "clamp", "below"),
    codeIn(OUT_OF_RANGE, rule.above ?? "base", "above"),
    ...(rule.base ?? DEFAULT_BASE),
    ...(rule.nan ?? DEFAULT_NAN),
  ];
  if (values.length !== 13) {
    throw invalid("base and nan must have 4 channels");
  }
  let map: number;
  if (typeof rule.map === "string") {
    map = codeIn(MAPS, rule.map, "map");
  } else if (typeof rule.map === "object" && rule.map !== null && Array.isArray(rule.map.stops)) {
    map = 4;
    for (const [t, color] of rule.map.stops) {
      if (color.length !== 4) {
        throw invalid("stop colours must have 4 channels");
      }
      values.push(t, ...color);
    }
  } else {
    throw invalid('map must be "turbo", "viridis", "coolwarm", "grayscale" or { stops }');
  }
  return { map, values: Float32Array.from(values) };
}

function invalid(message: string): FastMeshError {
  return new FastMeshError("INVALID_ARGUMENT", message);
}

function toFloat64(values: Positions64): Float64Array {
  return values instanceof Float64Array ? values : Float64Array.from(values);
}

function toUint32(values: Indices): Uint32Array {
  return values instanceof Uint32Array ? values : Uint32Array.from(values);
}

/** A quaternion as a row-major rotation matrix, normalised first, in f64. */
function quatToMat3(q: Quat): number[] {
  const n = Math.hypot(q[0], q[1], q[2], q[3]);
  if (!(n > 0) || !Number.isFinite(n)) {
    throw invalid("quaternion must be non-zero and finite");
  }
  const x = q[0] / n;
  const y = q[1] / n;
  const z = q[2] / n;
  const w = q[3] / n;
  return [
    1 - 2 * (y * y + z * z),
    2 * (x * y - z * w),
    2 * (x * z + y * w),
    2 * (x * y + z * w),
    1 - 2 * (x * x + z * z),
    2 * (y * z - x * w),
    2 * (x * z - y * w),
    2 * (y * z + x * w),
    1 - 2 * (x * x + y * y),
  ];
}

function rotationMatrix(rotation: Mat3 | Quat): number[] {
  if (rotation.length === 4) {
    return quatToMat3(rotation as Quat);
  }
  if (rotation.length === 9) {
    return Array.from(rotation);
  }
  throw invalid("rotation must be a 3×3 matrix (9 numbers) or a quaternion (4 numbers)");
}

/**
 * A scene of editable meshes. Create it after `await init()`; release it with
 * {@link Scene.free} or a `using` declaration (otherwise it is freed when
 * garbage collected).
 *
 * Render views ({@link Scene.views}) point into WebAssembly memory and are
 * only valid until the next call that may allocate; see {@link MeshViews}.
 */
export class Scene {
  #raw: WasmScene | undefined;
  readonly #guard: ViewGuard | undefined;
  /** Live meshes' handles by `"index:generation"`. */
  readonly #meshes = new Map<string, MeshHandle>();
  /** Handles of meshes removed since the last `takeChanges`, for their removal records. */
  readonly #removed = new Map<string, MeshHandle>();
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
    this.#drag = undefined;
    finalizer.unregister(this);
    release(raw);
  }

  /** Same as {@link Scene.free}, for `using scene = new Scene()`. */
  [Symbol.dispose](): void {
    this.free();
  }

  #scene(): WasmScene {
    if (this.#raw === undefined) {
      throw invalid("scene is freed");
    }
    return this.#raw;
  }

  /** The live mesh id of a handle of this scene. */
  #id(mesh: MeshHandle): HandleId {
    this.#scene();
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
    this.#scene();
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

  /** Removes a mesh. Its handle then gives `UNKNOWN_MESH`. */
  removeMesh(mesh: MeshHandle): void {
    const { index, generation } = this.#id(mesh);
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
    const p = call(() => this.#scene().placement(index, generation));
    return { rotation: Array.from(p.subarray(0, 9)), translation: [p[9], p[10], p[11]] };
  }

  /**
   * The column-major world matrix of the local render coordinates,
   * `placement · translate(origin)`, ready for renderers (convert to `f32`
   * yourself).
   */
  matrix4(mesh: MeshHandle): Float64Array {
    const { index, generation } = this.#id(mesh);
    return call(() => this.#scene().matrix4(index, generation));
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
    return call(() => this.#scene().vertex_slots(index, generation));
  }

  /** Triangle slots, alive or dead: the triangle length of the index buffer. */
  triangleSlots(mesh: MeshHandle): number {
    const { index, generation } = this.#id(mesh);
    return call(() => this.#scene().triangle_slots(index, generation));
  }

  /** Alive vertices. */
  vertexCount(mesh: MeshHandle): number {
    const { index, generation } = this.#id(mesh);
    return call(() => this.#scene().vertex_count(index, generation));
  }

  /** Alive triangles. */
  triangleCount(mesh: MeshHandle): number {
    const { index, generation } = this.#id(mesh);
    return call(() => this.#scene().triangle_count(index, generation));
  }

  /**
   * Zero-copy views of the mesh's render buffers. Each getter builds a fresh
   * view; do not keep one across a call that may allocate (see {@link MeshViews}).
   */
  views(mesh: MeshHandle): MeshViews {
    const { index, generation } = this.#id(mesh);
    // Fail now, not on the first getter, for a removed mesh.
    call(() => this.#scene().vertex_slots(index, generation));
    return meshViews(() => this.#scene(), index, generation, this.#guard);
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
    const s = this.#scene();
    return call(() => s.can_undo());
  }

  /** True if {@link Scene.redo} would redo something. */
  canRedo(): boolean {
    const s = this.#scene();
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
   * that may allocate: get it right before use.
   */
  fieldValues(field: FieldHandle): Float32Array {
    const raw = this.#field(field);
    const s = this.#scene();
    const ptr = call(() => s.field_values_ptr(raw));
    const view = new Float32Array(wasmMemory().buffer, ptr, call(() => s.field_values_len(raw)));
    return this.#guard === undefined ? view : this.#guard.track(view);
  }

  /** Whether the field is current, and what its sign data found out about the target mesh. */
  fieldStatus(field: FieldHandle): FieldStatus {
    const raw = this.#field(field);
    const st = call(() => this.#scene().field_status(raw));
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
    const r = call(() => this.#scene().section(raw, index, generation));
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
