// `SceneStore`: an observable, immutable snapshot of a scene (sync `Scene` or
// `WorkerScene`) plus the actions that change it, for UI frameworks
// (`useSyncExternalStore`) and plain apps alike. Display state (opacity,
// visibility, colouring) and sections are declarative: a change updates
// `desired` at once and reaches the scene at the next `commitFrame()` (or on
// a microtask while no frame loop runs), the last value per frame winning.
import type { Indices, PlacementInput, Positions64 } from "../api.js";
import { FastMeshError, reportListenerError } from "../core.js";
import type {
  DragSession,
  EditOp,
  EditResult,
  MeshChange,
  MeshHandle,
  RepairOp,
  RepairReport,
  Scene,
  SectionDesc,
  SectionHandle,
  SectionPlane,
} from "../scene.js";
import type { SceneLike } from "../scene-like.js";
import type { WorkerScene } from "../worker/client.js";
import { sectionAspects } from "../worker/protocol.js";
import { type Host, hostOf, type StoreAddMeshOptions } from "./host.js";
import {
  type Combined,
  combine,
  DEFAULT_DISPLAY,
  type DisplayState,
  EMPTY_SNAPSHOT,
  frozenCopy,
  type MeshState,
  type SceneSnapshot,
  type SectionParts,
  type SectionState,
  sameData,
} from "./snapshot.js";

/** A change to a section's declarative parts. */
export type SectionPatch = Partial<SectionParts>;

/** A mesh the store tracks. */
interface MeshEntry {
  handle: MeshHandle;
  /** Its id in status keys, kept: a removed mesh's id cannot be asked for. */
  id: number;
  /** Applied (frozen). */
  display: DisplayState;
  /** Last requested (frozen). */
  desired: DisplayState;
}

/** A section the store tracks. */
interface SectionEntry {
  handle: SectionHandle;
  id: number;
  /** Applied (frozen). */
  applied: SectionParts;
  /** Last requested (frozen). */
  desired: SectionParts;
}

/**
 * The aspect suffixes of a section's `set` keys (`section:<id>:<aspects>`),
 * made by the client's own `sectionAspects` over every combination of the
 * three aspects (the empty one gives `none`).
 */
const ASPECT_KEYS: readonly string[] = [0, 1, 2, 3, 4, 5, 6, 7].map(
  (bits) =>
    sectionAspects({
      plane: bits & 1 ? { origin: [0, 0, 0], normal: [0, 0, 1] } : undefined,
      meshes: bits & 2 ? [] : undefined,
      fill: bits & 4 ? false : undefined,
    }).join("+") || "none",
);

function invalid(message: string): FastMeshError {
  return new FastMeshError("INVALID_ARGUMENT", message);
}

/** Set by `SceneStore`'s static block: the test-only {@link listenerCount}. */
let listenerCountOf: (store: SceneStore) => number;

/**
 * An observable store over a scene. Read with {@link getSnapshot} and
 * {@link subscribe}; change through its actions. Call {@link commitFrame}
 * once per animation frame instead of the scene's.
 */
export class SceneStore {
  /** The scene, for the renderer adapters (`bindMesh`, ...). */
  readonly scene: SceneLike;
  /** `"local"` over a `Scene`, `"worker"` over a `WorkerScene`. */
  readonly mode: "local" | "worker";
  readonly #host: Host;
  readonly #meshes = new Map<MeshHandle, MeshEntry>();
  readonly #sections = new Map<SectionHandle, SectionEntry>();
  readonly #listeners = new Set<() => void>();
  readonly #unsubscribe: (() => void)[];
  /** Meshes whose display changed since the last flush. */
  readonly #pendingDisplay = new Set<MeshEntry>();
  #selection: MeshHandle | null = null;
  #snapshot: SceneSnapshot = EMPTY_SNAPSHOT;
  #frameLoop = false;
  #flushQueued = false;
  #disposed = false;

  constructor(scene: Scene | WorkerScene) {
    this.#host = hostOf(scene);
    this.scene = this.#host.scene;
    this.mode = this.#host.mode;
    const refresh = (): void => {
      if (!this.#disposed) {
        this.#refresh();
      }
    };
    this.#unsubscribe = [scene.onChanges(refresh), this.#host.onStatus(refresh)];
    this.#refresh();
  }

  // ---- Reading ----

  /** The current snapshot: frozen, replaced (never mutated) on change. */
  getSnapshot = (): SceneSnapshot => this.#snapshot;

  /**
   * Registers a listener called after each new snapshot; returns the function
   * that removes it. A listener that throws does not stop the others nor the
   * action that changed the state (which has taken effect): its error is
   * reported with `reportError` (an async rethrow where there is none), as
   * the Worker scene does with its listeners.
   */
  subscribe = (listener: () => void): (() => void) => {
    this.#listeners.add(listener);
    return () => {
      this.#listeners.delete(listener);
    };
  };

  // ---- Frames ----

  /**
   * Call once per animation frame (instead of the scene's `commitFrame()`):
   * applies the frame's declarative changes, then commits the scene's frame
   * and returns its change records. From the first call on, declarative
   * changes wait for it instead of applying on a microtask.
   */
  commitFrame(): MeshChange[] {
    this.#alive();
    this.#frameLoop = true;
    this.#flush();
    const changes = this.#host.scene.commitFrame();
    this.#refresh();
    return changes;
  }

  // ---- Meshes ----

  /** Adds a mesh (default display) and returns its handle at once. */
  addMesh(positions: Positions64, indices: Indices, options?: StoreAddMeshOptions): MeshHandle {
    this.#alive();
    const handle = this.#host.addMesh(positions, indices, options);
    this.#meshes.set(handle, { handle, id: this.#host.idOf(handle), display: DEFAULT_DISPLAY, desired: DEFAULT_DISPLAY });
    this.#refresh();
    return handle;
  }

  /** Removes a mesh; it leaves the snapshot (and the selection) at once. */
  removeMesh(mesh: MeshHandle): void {
    this.#alive();
    this.#host.removeMesh(mesh);
    const entry = this.#meshes.get(mesh);
    if (entry !== undefined) {
      this.#meshes.delete(mesh);
      this.#pendingDisplay.delete(entry);
    }
    if (this.#selection === mesh) {
      this.#selection = null;
    }
    this.#refresh();
  }

  /** Sets a mesh's placement (one undo step). */
  setPlacement(mesh: MeshHandle, placement: PlacementInput): void {
    this.#alive();
    this.#host.setPlacement(mesh, placement);
    this.#refresh();
  }

  /** Starts a drag of a mesh; see the scene's `drag()`. Ending or cancelling it updates the snapshot. */
  drag(mesh: MeshHandle): DragSession {
    this.#alive();
    const session = this.#host.scene.drag(mesh);
    const after = (): void => {
      if (!this.#disposed) {
        this.#refresh();
      }
    };
    return {
      move: (placement) => session.move(placement),
      get active() {
        return session.active;
      },
      end: () => {
        session.end();
        after();
      },
      cancel: () => {
        session.cancel();
        after();
      },
    };
  }

  /** Edits a mesh's triangles (one undo step). */
  edit(mesh: MeshHandle, ops: readonly EditOp[]): Promise<EditResult> {
    this.#alive();
    return this.#settled(this.#host.edit(mesh, ops));
  }

  /** Repairs a mesh (one undo step). */
  repair(mesh: MeshHandle, op: RepairOp | readonly RepairOp[]): Promise<RepairReport | RepairReport[]> {
    this.#alive();
    return this.#settled(this.#host.repair(mesh, op));
  }

  /** Refreshes now (a synchronous host has applied the call) and once the promise settles. */
  #settled<T>(p: Promise<T>): Promise<T> {
    this.#refresh();
    const after = (): void => {
      if (!this.#disposed) {
        this.#refresh();
      }
    };
    return p.then(
      (v) => {
        after();
        return v;
      },
      (e: unknown) => {
        after();
        throw e;
      },
    );
  }

  // ---- History ----

  /** Undoes the latest step. */
  undo(): void {
    this.#alive();
    this.#host.undo();
    this.#refresh();
  }

  /** Redoes the latest undone step. */
  redo(): void {
    this.#alive();
    this.#host.redo();
    this.#refresh();
  }

  // ---- Display ----

  /**
   * Changes how a mesh is shown. `desired` changes at once; `display` follows
   * at the next `commitFrame()` (or a microtask without a frame loop). A
   * patch equal to the current request changes nothing.
   *
   * @throws `UNKNOWN_MESH` for a mesh not in the store, `INVALID_ARGUMENT` for a bad value.
   */
  setDisplay(mesh: MeshHandle, patch: Partial<DisplayState>): void {
    this.#alive();
    const entry = this.#mesh(mesh);
    const next: DisplayState = { ...entry.desired };
    if (patch.opacity !== undefined) {
      const o = patch.opacity;
      if (typeof o !== "number" || !(o >= 0 && o <= 1)) {
        throw invalid("opacity must be a number in [0, 1]");
      }
      next.opacity = o;
    }
    if (patch.visible !== undefined) {
      if (typeof patch.visible !== "boolean") {
        throw invalid("visible must be a boolean");
      }
      next.visible = patch.visible;
    }
    if (patch.coloring !== undefined) {
      if (patch.coloring !== null && typeof patch.coloring !== "object") {
        throw invalid("coloring must be an object or null");
      }
      if (!sameData(patch.coloring, entry.desired.coloring)) {
        next.coloring = frozenCopy(patch.coloring);
      }
    }
    if (
      next.opacity === entry.desired.opacity &&
      next.visible === entry.desired.visible &&
      next.coloring === entry.desired.coloring
    ) {
      return;
    }
    entry.desired = Object.freeze(next);
    this.#pendingDisplay.add(entry);
    this.#schedule();
    this.#refresh();
  }

  // ---- Sections ----

  /** Adds a plane section and returns its handle at once. */
  addSection(desc: SectionDesc): SectionHandle {
    this.#alive();
    const handle = this.#host.addSection(desc);
    const parts: SectionParts = frozenCopy({ plane: desc.plane, meshes: desc.meshes, fill: desc.fill ?? false });
    this.#sections.set(handle, { handle, id: this.#host.idOf(handle), applied: parts, desired: parts });
    this.#refresh();
    return handle;
  }

  /**
   * Changes a section's plane, meshes or fill. `desired` changes at once.
   *
   * @throws `UNKNOWN_SECTION` for a section not in the store.
   */
  setSection(section: SectionHandle, patch: SectionPatch): void {
    this.#alive();
    const entry = this.#section(section);
    const next: SectionParts = { ...entry.desired };
    if (patch.plane !== undefined && !sameData(patch.plane, next.plane)) {
      next.plane = frozenCopy(patch.plane as SectionPlane);
    }
    if (patch.meshes !== undefined && !sameData(patch.meshes, next.meshes)) {
      next.meshes = frozenCopy(patch.meshes);
    }
    if (patch.fill !== undefined) {
      if (typeof patch.fill !== "boolean") {
        throw invalid("fill must be a boolean");
      }
      next.fill = patch.fill;
    }
    if (next.plane === entry.desired.plane && next.meshes === entry.desired.meshes && next.fill === entry.desired.fill) {
      return;
    }
    entry.desired = Object.freeze(next);
    this.#refresh();
  }

  /** Removes a section; it leaves the snapshot at once. */
  removeSection(section: SectionHandle): void {
    this.#alive();
    this.#host.removeSection(section);
    this.#sections.delete(section);
    this.#refresh();
  }

  // ---- Selection ----

  /**
   * Selects a mesh, or nothing with `null`. UI state only: the scene never sees it.
   *
   * @throws `UNKNOWN_MESH` for a mesh not in the store.
   */
  select(mesh: MeshHandle | null): void {
    this.#alive();
    if (mesh !== null) {
      this.#mesh(mesh);
    }
    this.#selection = mesh;
    this.#refresh();
  }

  // ---- Lifecycle ----

  /**
   * Restarts the Worker scene (see `WorkerScene.restart()`). Rejects with
   * `INVALID_ARGUMENT` on a local (synchronous) scene.
   */
  restart(): Promise<void> {
    if (this.#disposed) {
      return Promise.reject(invalid("the store is disposed"));
    }
    return this.#settled(this.#host.restart());
  }

  /**
   * Stops listening to the scene. The scene itself is left as it is (it is
   * the app's). Later actions throw `INVALID_ARGUMENT`; the last snapshot stays readable.
   */
  dispose(): void {
    if (this.#disposed) {
      return;
    }
    this.#disposed = true;
    for (const off of this.#unsubscribe.splice(0)) {
      off();
    }
    this.#pendingDisplay.clear();
  }

  // ---- Internals ----

  #alive(): void {
    if (this.#disposed) {
      throw invalid("the store is disposed");
    }
  }

  #mesh(mesh: MeshHandle): MeshEntry {
    const entry = this.#meshes.get(mesh);
    if (entry === undefined) {
      throw new FastMeshError("UNKNOWN_MESH", "the mesh is not in the store (removed, or not added through it)");
    }
    return entry;
  }

  #section(section: SectionHandle): SectionEntry {
    const entry = this.#sections.get(section);
    if (entry === undefined) {
      throw new FastMeshError("UNKNOWN_SECTION", "the section is not in the store (removed, or not added through it)");
    }
    return entry;
  }

  /** Without a frame loop, flushes on a microtask; with one, the next `commitFrame()` does. */
  #schedule(): void {
    if (this.#frameLoop || this.#flushQueued) {
      return;
    }
    this.#flushQueued = true;
    queueMicrotask(() => {
      this.#flushQueued = false;
      if (this.#disposed || this.#frameLoop) {
        return;
      }
      this.#flush();
      this.#refresh();
    });
  }

  /** Applies the declarative changes made since the last flush, the latest value of each. */
  #flush(): void {
    for (const entry of this.#pendingDisplay) {
      const d = entry.display;
      const want = entry.desired;
      if (d.opacity !== want.opacity || d.visible !== want.visible) {
        // Colouring is applied with its field (declarative colouring).
        entry.display = Object.freeze({ ...d, opacity: want.opacity, visible: want.visible });
      }
    }
    this.#pendingDisplay.clear();
  }

  #key(key: string): Combined {
    const s = this.#host.status(key);
    return s.status === "error" ? { status: "error", error: s.error } : { status: s.status };
  }

  /**
   * The status of a state from its own keys (all of their status, errors
   * included) and the keys it depends on, which only add `"computing"`:
   * their errors belong to the requests that made them (a failed edit
   * rejects its promise) and are ignored here.
   */
  #status(own: readonly string[], dependent: readonly string[]): Combined {
    const parts = own.map((k) => this.#key(k));
    if (dependent.some((k) => this.#host.status(k).status === "computing")) {
      parts.push({ status: "computing" });
    }
    return combine(parts);
  }

  /**
   * A mesh's status: its own `mesh:` and `display:` keys, made `"computing"`
   * also by its `placement:` and `edit:` keys and the history.
   */
  #meshStatus(entry: MeshEntry): Combined {
    const id = entry.id;
    return this.#status([`mesh:${id}`, `display:${id}`], [`placement:${id}`, `edit:${id}`, "history"]);
  }

  /**
   * A section's status: its own `section:` keys, made `"computing"` also by
   * the `placement:` and `edit:` keys of its meshes and the history.
   */
  #sectionStatus(entry: SectionEntry): Combined {
    const own = [`section:${entry.id}`, ...ASPECT_KEYS.map((a) => `section:${entry.id}:${a}`)];
    const dependent = ["history"];
    for (const m of entry.applied.meshes) {
      const mesh = this.#meshes.get(m);
      if (mesh !== undefined) {
        dependent.push(`placement:${mesh.id}`, `edit:${mesh.id}`);
      }
    }
    return this.#status(own, dependent);
  }

  /**
   * Rebuilds the snapshot from the store and the scene, reusing every
   * unchanged part of the previous one; publishes (and notifies) only when
   * something changed.
   */
  #refresh(): void {
    const prev = this.#snapshot;
    const revision = prev.revision + 1;
    let changed = false;

    const meshes = new Map<MeshHandle, MeshState>();
    for (const entry of this.#meshes.values()) {
      const old = prev.meshes.get(entry.handle);
      const s = this.#meshStatus(entry);
      if (
        old !== undefined &&
        old.display === entry.display &&
        old.desired === entry.desired &&
        old.status === s.status &&
        old.error === s.error
      ) {
        meshes.set(entry.handle, old);
        continue;
      }
      changed = true;
      const state: MeshState = { display: entry.display, desired: entry.desired, status: s.status, revision };
      if (s.error !== undefined) {
        state.error = s.error;
      }
      meshes.set(entry.handle, Object.freeze(state));
    }
    if (meshes.size !== prev.meshes.size) {
      changed = true;
    }

    const sections = new Map<SectionHandle, SectionState>();
    for (const entry of this.#sections.values()) {
      const old = prev.sections.get(entry.handle);
      const s = this.#sectionStatus(entry);
      const a = entry.applied;
      if (
        old !== undefined &&
        old.plane === a.plane &&
        old.meshes === a.meshes &&
        old.fill === a.fill &&
        old.desired === entry.desired &&
        old.status === s.status &&
        old.error === s.error
      ) {
        sections.set(entry.handle, old);
        continue;
      }
      changed = true;
      const state: SectionState = { ...a, desired: entry.desired, status: s.status, revision };
      if (s.error !== undefined) {
        state.error = s.error;
      }
      sections.set(entry.handle, Object.freeze(state));
    }
    if (sections.size !== prev.sections.size) {
      changed = true;
    }

    const canUndo = this.#host.canUndo();
    const canRedo = this.#host.canRedo();
    let history = prev.history;
    if (history.canUndo !== canUndo || history.canRedo !== canRedo) {
      history = Object.freeze({ canUndo, canRedo });
      changed = true;
    }
    if (prev.selection !== this.#selection) {
      changed = true;
    }
    const overall = combine([...meshes.values(), ...sections.values(), this.#key("history")]);
    const status = overall.status === "ready" ? "idle" : overall.status;
    if (status !== prev.status || overall.error !== prev.error) {
      changed = true;
    }
    if (!changed) {
      return;
    }
    const snap: SceneSnapshot = {
      revision,
      meshes: Object.freeze(meshes),
      history,
      sections: Object.freeze(sections),
      selection: this.#selection,
      status,
    };
    if (overall.error !== undefined) {
      snap.error = overall.error;
    }
    this.#snapshot = Object.freeze(snap);
    // Every listener runs even if one throws; errors are reported, not thrown.
    for (const listener of [...this.#listeners]) {
      try {
        listener();
      } catch (error) {
        reportListenerError(error);
      }
    }
  }

  static {
    listenerCountOf = (store) => store.#listeners.size;
  }
}

/** A store over `scene` (a `Scene` or a `WorkerScene`). */
export function createSceneStore(scene: Scene | WorkerScene): SceneStore {
  return new SceneStore(scene);
}

/** Test-only, not exported from `fast-mesh-wasm/store`: the number of subscribed listeners. */
export function listenerCount(store: SceneStore): number {
  return listenerCountOf(store);
}
