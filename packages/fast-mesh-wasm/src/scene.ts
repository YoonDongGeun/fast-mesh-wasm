// The public `Scene` class: editable meshes with placements, zero-copy render
// views, per-frame change records and undo history. Wraps the raw `WasmScene`
// binding; every call into WebAssembly goes through `call` (error codes and
// panic poisoning).
import type { Indices, Mat3, Placement, PlacementInput, Positions64, Quat, UpdateMode, Vec3 } from "./api.js";
import { decodeChanges, type Range } from "./changes.js";
import { call, ensureInitialized, FastMeshError, finalizer, release } from "./core.js";
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
 */
export interface DragSession {
  /** Sets the mesh's placement. Throws `INVALID_ARGUMENT` once the drag is over. */
  move(placement: PlacementInput): void;
  /** Finishes the drag, recording it as one undo step. A no-op once it is over. */
  end(): void;
  /** Puts the placement back to where the drag started and records nothing. A no-op once it is over. */
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

const MODES: Record<UpdateMode, number> = { eager: 0, lazy: 1, manual: 2 };

function modeCode(mode: UpdateMode): number {
  if (!Object.hasOwn(MODES, mode)) {
    throw invalid('mode must be "eager", "lazy" or "manual"');
  }
  return MODES[mode];
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
  #drag: { end(): void } | undefined;

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
    // Every removal is in this message (or was never reported, for a mesh
    // added and removed in between), so removed handles can go.
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
   * Undoes the latest step; an active drag is ended first, so it is what gets
   * undone. Returns `false` when there is nothing to undo.
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
  }

  /** Reverts and discards the open step, however deeply nested. Throws `STEP_NOT_OPEN` when none is open. */
  cancelStep(): void {
    const s = this.#scene();
    call(() => s.cancel_step());
    this.#stepDepth = 0;
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
    let active = true;
    const finish = (): boolean => {
      if (!active) {
        return false;
      }
      active = false;
      this.#drag = undefined;
      return true;
    };
    const session: DragSession = {
      move: (placement) => {
        if (!active) {
          throw invalid("the drag is over");
        }
        this.setPlacement(mesh, placement);
      },
      end: () => {
        if (finish()) {
          this.endStep();
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
          this.endStep();
        }
      },
    };
    this.#drag = session;
    return session;
  }
}
