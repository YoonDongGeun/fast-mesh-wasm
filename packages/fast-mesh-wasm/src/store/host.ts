// Internal: one shape over the synchronous `Scene` and the `WorkerScene`, so
// the store is written once. Types only from the scene modules, so the store
// entry never loads the WebAssembly glue; a Worker scene is told apart
// structurally (`"status" in scene`), not with `instanceof`.
import { FastMeshError } from "../core.js";
import type {
  AddMeshOptions,
  ColorRule,
  DistanceFieldDesc,
  EditOp,
  EditResult,
  FieldHandle,
  MeshHandle,
  RepairOp,
  RepairReport,
  Scene,
  SectionDesc,
  SectionHandle,
  SectionPlane,
} from "../scene.js";
import type { SceneLike } from "../scene-like.js";
import type { Indices, PlacementInput, Positions64 } from "../api.js";
import { addFieldKeyed, type KeyStatus, type WorkerScene } from "../worker/client.js";

/** Options of `SceneStore.addMesh`: those both scenes take (a store mesh is always eager). */
export type StoreAddMeshOptions = Pick<AddMeshOptions, "origin" | "normals">;

/** What the store needs of a scene, the same on both hosts. */
export interface Host {
  readonly mode: "local" | "worker";
  readonly scene: SceneLike;
  addMesh(positions: Positions64, indices: Indices, options?: StoreAddMeshOptions): MeshHandle;
  removeMesh(mesh: MeshHandle): void;
  setPlacement(mesh: MeshHandle, placement: PlacementInput): void;
  edit(mesh: MeshHandle, ops: readonly EditOp[]): Promise<EditResult>;
  repair(mesh: MeshHandle, op: RepairOp | readonly RepairOp[]): Promise<RepairReport | RepairReport[]>;
  undo(): void;
  redo(): void;
  canUndo(): boolean;
  canRedo(): boolean;
  /** The state of a request key (`mesh:<id>`, `display:<id>`, ...). */
  status(key: string): KeyStatus;
  /** Registers a listener for status (and undo/redo state) changes; returns its remover. */
  onStatus(listener: () => void): () => void;
  /** The id of a handle as used in status keys. */
  idOf(handle: MeshHandle | FieldHandle | SectionHandle): number;
  /**
   * Adds a distance field. On the Worker host, `key` (the `display:<id>` key
   * of the colouring it is built for) lets the engine abandon the build when
   * that colouring is superseded first; the add's status is then under `key`.
   */
  addField(desc: DistanceFieldDesc, key?: string): FieldHandle;
  removeField(field: FieldHandle): void;
  setColoring(mesh: MeshHandle, rule: ColorRule | null): void;
  addSection(desc: SectionDesc): SectionHandle;
  setSectionPlane(section: SectionHandle, plane: SectionPlane): void;
  setSectionMeshes(section: SectionHandle, meshes: readonly MeshHandle[]): void;
  setSectionFill(section: SectionHandle, fill: boolean): void;
  removeSection(section: SectionHandle): void;
  restart(): Promise<void>;
  /**
   * Runs `fn` (a call the store makes for a key, such as a colouring's)
   * under `key`: an exception sets the key's status to `"error"`, and is
   * rethrown, until a later call under the key succeeds.
   */
  track<T>(key: string, fn: () => T): T;
}

/** Whether `scene` is a Worker scene (structurally: it reports key statuses). */
export function isWorkerScene(scene: Scene | WorkerScene): scene is WorkerScene {
  return "status" in scene;
}

/** The host of `scene`. */
export function hostOf(scene: Scene | WorkerScene): Host {
  return isWorkerScene(scene) ? workerHost(scene) : localHost(scene);
}

/** The errors of calls made under a key that threw at once (see `Host.track`), by key. */
function tracker() {
  const errors = new Map<string, FastMeshError>();
  const track = <T>(key: string, fn: () => T): T => {
    try {
      const out = fn();
      errors.delete(key);
      return out;
    } catch (e) {
      errors.set(
        key,
        e instanceof FastMeshError ? e : new FastMeshError("INVALID_ARGUMENT", e instanceof Error ? e.message : String(e)),
      );
      throw e;
    }
  };
  return { errors, track };
}

/**
 * The synchronous scene. Calls apply at once, so every key is `"ready"`,
 * except one whose latest call made through `track` threw: that key is
 * `"error"` until a call under it succeeds. Nothing is ever pending, so
 * `onStatus` never fires.
 */
export function localHost(scene: Scene): Host {
  const ids = new WeakMap<object, number>();
  let next = 0;
  const { errors, track } = tracker();
  let rev = 0;
  const promised = <T>(fn: () => T): Promise<T> => new Promise<T>((resolve) => resolve(fn()));
  return {
    mode: "local",
    scene,
    addMesh: (positions, indices, options) => scene.addMesh(positions, indices, options),
    removeMesh: (mesh) => scene.removeMesh(mesh),
    setPlacement: (mesh, placement) => scene.setPlacement(mesh, placement),
    edit: (mesh, ops) => promised(() => scene.edit(mesh, ops)),
    // One op or a list: the cast only picks an overload (both take either).
    repair: (mesh, op) => promised(() => scene.repair(mesh, op as RepairOp)),
    undo: () => {
      scene.undo();
    },
    redo: () => {
      scene.redo();
    },
    canUndo: () => scene.canUndo(),
    canRedo: () => scene.canRedo(),
    status: (key) => {
      const error = errors.get(key);
      return error === undefined ? { status: "ready", revision: rev } : { status: "error", revision: rev, error };
    },
    onStatus: () => () => {},
    idOf: (handle) => {
      let id = ids.get(handle);
      if (id === undefined) {
        id = ++next;
        ids.set(handle, id);
      }
      return id;
    },
    track: (key, fn) => {
      rev++;
      return track(key, fn);
    },
    addField: (desc) => scene.addDistanceField(desc),
    removeField: (field) => scene.removeField(field),
    setColoring: (mesh, rule) => scene.setColoring(mesh, rule),
    addSection: (desc) => scene.addSection(desc),
    setSectionPlane: (section, plane) => scene.setSectionPlane(section, plane),
    setSectionMeshes: (section, meshes) => scene.setSectionMeshes(section, meshes),
    setSectionFill: (section, fill) => scene.setSectionFill(section, fill),
    removeSection: (section) => scene.removeSection(section),
    restart: () => Promise.reject(new FastMeshError("INVALID_ARGUMENT", "restart needs a Worker scene")),
  };
}

/**
 * The Worker scene: statuses, ids and history come from it as they are,
 * except a key whose latest call made through `track` threw here, before any
 * request went out: that key is `"error"` until a call under it succeeds.
 */
export function workerHost(scene: WorkerScene): Host {
  const { errors, track } = tracker();
  return {
    mode: "worker",
    scene,
    addMesh: (positions, indices, options) => scene.addMesh(positions, indices, options),
    removeMesh: (mesh) => scene.removeMesh(mesh),
    setPlacement: (mesh, placement) => scene.setPlacement(mesh, placement),
    edit: (mesh, ops) => scene.edit(mesh, ops),
    // One op or a list: the cast only picks an overload (both take either).
    repair: (mesh, op) => scene.repair(mesh, op as RepairOp),
    undo: () => scene.undo(),
    redo: () => scene.redo(),
    canUndo: () => scene.canUndo,
    canRedo: () => scene.canRedo,
    status: (key) => {
      const error = errors.get(key);
      const s = scene.status(key);
      return error === undefined ? s : { status: "error", revision: s.revision, error };
    },
    onStatus: (listener) => scene.onStatus(listener),
    idOf: (handle) => scene.idOf(handle),
    addField: (desc, key) => (key === undefined ? scene.addDistanceField(desc) : addFieldKeyed(scene, desc, key)),
    removeField: (field) => scene.removeField(field),
    setColoring: (mesh, rule) => scene.setColoring(mesh, rule),
    addSection: (desc) => scene.addSection(desc),
    setSectionPlane: (section, plane) => scene.setSectionPlane(section, plane),
    setSectionMeshes: (section, meshes) => scene.setSectionMeshes(section, meshes),
    setSectionFill: (section, fill) => scene.setSectionFill(section, fill),
    removeSection: (section) => scene.removeSection(section),
    restart: () => scene.restart(),
    track,
  };
}
