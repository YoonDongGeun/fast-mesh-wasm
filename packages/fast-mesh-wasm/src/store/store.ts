// `SceneStore`: an observable, immutable snapshot of a scene (sync `Scene` or
// `WorkerScene`) plus the actions that change it, for UI frameworks
// (`useSyncExternalStore`) and plain apps alike. Display state (opacity,
// visibility, colouring) and sections are declarative: a change updates
// `desired` at once and reaches the scene at the next `commitFrame()` (or on
// a microtask while no frame loop runs), the last value per frame winning.
import type { Indices, PlacementInput, Positions64 } from "../api.js";
import { FastMeshError, reportListenerError } from "../core.js";
import type {
  ColorRule,
  DragSession,
  EditOp,
  EditResult,
  FieldHandle,
  MeshChange,
  MeshHandle,
  RepairOp,
  RepairReport,
  Scene,
  SectionDesc,
  SectionHandle,
  SectionPlane,
  SignMode,
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
  type DistanceColoring,
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

/**
 * The distance field the store owns for a mesh's colouring, keyed by
 * `(to, maxDistance, sign)`: kept while the colouring is off, reused when it
 * comes back with the same key, replaced when the key changes.
 */
interface OwnedField {
  handle: FieldHandle;
  /** Its id in status keys (`field:<id>`). */
  id: number;
  to: MeshHandle;
  /** The id of `to` in status keys, when the store tracks it. */
  toId: number | undefined;
  maxDistance: number;
  sign: SignMode;
}

/** A request sent to the scene and not yet applied: applied once its key's status reaches `rev`. */
interface InFlight<T> {
  value: T;
  rev: number;
}

/** A mesh the store tracks. */
interface MeshEntry {
  handle: MeshHandle;
  /** Its id in status keys, kept: a removed mesh's id cannot be asked for. */
  id: number;
  /** Applied (frozen). */
  display: DisplayState;
  /** Last requested by the app (frozen). */
  desired: DisplayState;
  /** The colouring last sent to the scene (frozen), or `null`. */
  sent: DistanceColoring | null;
  /** The colouring sent and not yet applied. */
  inflight?: InFlight<DistanceColoring | null>;
  field?: OwnedField;
}

/** The aspects of a section the store sets one by one. */
const ASPECTS = ["plane", "meshes", "fill"] as const;
type Aspect = (typeof ASPECTS)[number];

/** A section the store tracks. */
interface SectionEntry {
  handle: SectionHandle;
  id: number;
  /** Applied (frozen). */
  applied: SectionParts;
  /** Last requested by the app (frozen). */
  desired: SectionParts;
  /** Last sent to the scene (frozen). */
  sent: SectionParts;
  /** Aspects sent and not yet applied. */
  inflight: { [A in Aspect]?: InFlight<SectionParts[A]> };
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

/** The scene's colour rule for a declarative colouring by `field` (only the fields set). */
function colorRule(field: FieldHandle, c: DistanceColoring): ColorRule {
  const rule: ColorRule = { field, map: c.map, range: c.range };
  if (c.below !== undefined) {
    rule.below = c.below;
  }
  if (c.above !== undefined) {
    rule.above = c.above;
  }
  if (c.center !== undefined) {
    rule.center = c.center;
  }
  if (c.base !== undefined) {
    rule.base = c.base;
  }
  if (c.nan !== undefined) {
    rule.nan = c.nan;
  }
  return rule;
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
  /** Sections whose parts changed since the last flush. */
  readonly #pendingSections = new Set<SectionEntry>();
  /** Fields added so far, for their names (`fast-mesh-store:<n>`). */
  #fieldCount = 0;
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
    this.#meshes.set(handle, {
      handle,
      id: this.#host.idOf(handle),
      display: DEFAULT_DISPLAY,
      desired: DEFAULT_DISPLAY,
      sent: null,
    });
    this.#refresh();
    return handle;
  }

  /**
   * Removes a mesh; it leaves the snapshot (and the selection) at once. The
   * scene drops the fields measured to it, and the colourings by them: a
   * mesh coloured by distance to it shows no colouring, keeps its rule in
   * `desired`, and is in `"error"` (`UNKNOWN_MESH`) from the next flush
   * until its colouring changes. Sections stop cutting it.
   */
  removeMesh(mesh: MeshHandle): void {
    this.#alive();
    this.#host.removeMesh(mesh);
    const entry = this.#meshes.get(mesh);
    if (entry !== undefined) {
      this.#meshes.delete(mesh);
      this.#pendingDisplay.delete(entry);
      const display = `display:${entry.id}`;
      const field = entry.field === undefined ? undefined : `field:${entry.field.id}`;
      this.#host.forgetErrors((k) => k === display || k === field);
    }
    if (this.#selection === mesh) {
      this.#selection = null;
    }
    for (const other of this.#meshes.values()) {
      if (other.field?.to !== mesh) {
        continue;
      }
      // Gone from the scene with the mesh, and so is a colouring by it.
      const fieldKey = `field:${other.field.id}`;
      this.#host.forgetErrors((k) => k === fieldKey);
      other.field = undefined;
      other.sent = null;
      other.inflight = undefined;
      if (other.display.coloring !== null) {
        other.display = Object.freeze({ ...other.display, coloring: null });
      }
      if (other.desired.coloring !== null) {
        // The flush tries the rule again and reports why it cannot apply.
        this.#pendingDisplay.add(other);
        this.#schedule();
      }
    }
    const without = (meshes: readonly MeshHandle[]): readonly MeshHandle[] =>
      meshes.includes(mesh) ? Object.freeze(meshes.filter((m) => m !== mesh)) : meshes;
    for (const s of this.#sections.values()) {
      s.applied = this.#withMeshes(s.applied, without(s.applied.meshes));
      s.sent = this.#withMeshes(s.sent, without(s.sent.meshes));
      s.desired = this.#withMeshes(s.desired, without(s.desired.meshes));
      const inflight = s.inflight.meshes;
      if (inflight !== undefined) {
        inflight.value = without(inflight.value);
      }
    }
    this.#refresh();
  }

  /** `parts` with `meshes`, the same object when they are. */
  #withMeshes(parts: SectionParts, meshes: readonly MeshHandle[]): SectionParts {
    return meshes === parts.meshes ? parts : Object.freeze({ ...parts, meshes });
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
   * A `coloring` colours the mesh by distance to `distanceTo` through a
   * field the store owns (one per mesh, named `fast-mesh-store:<n>`, kept
   * while colouring is off and replaced only when `distanceTo`,
   * `maxDistance` or `sign` changes). `display.coloring` is the colouring
   * last applied: on a Worker scene, once its `display:` request is ready.
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
    this.#sections.set(handle, {
      handle,
      id: this.#host.idOf(handle),
      applied: parts,
      desired: parts,
      sent: parts,
      inflight: {},
    });
    this.#refresh();
    return handle;
  }

  /**
   * Changes a section's plane, meshes or fill. `desired` changes at once;
   * the section follows at the next `commitFrame()` (or a microtask without
   * a frame loop), each aspect set only when it changed, the last value
   * per frame winning.
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
    this.#pendingSections.add(entry);
    this.#schedule();
    this.#refresh();
  }

  /** Removes a section; it leaves the snapshot at once. */
  removeSection(section: SectionHandle): void {
    this.#alive();
    this.#host.removeSection(section);
    const entry = this.#sections.get(section);
    if (entry !== undefined) {
      this.#sections.delete(section);
      this.#pendingSections.delete(entry);
      const own = `section:${entry.id}`;
      this.#host.forgetErrors((k) => k === own || k.startsWith(`${own}:`));
    }
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
    const restarted = this.#host.restart();
    // The scene replays what it was sent (its fields, colourings and
    // sections), so nothing is sent again here. What was sent and not yet
    // applied (pending, or failed by the poisoning) settles with the replay,
    // whose requests now hold those keys: wait for them.
    for (const entry of this.#meshes.values()) {
      if (!sameData(entry.display.coloring, entry.sent)) {
        entry.inflight = { value: entry.sent, rev: this.#host.status(`display:${entry.id}`).revision };
      }
    }
    for (const entry of this.#sections.values()) {
      for (const a of ASPECTS) {
        if (!sameData(entry.applied[a], entry.sent[a])) {
          this.#arm(entry, a, entry.sent[a]);
        }
      }
    }
    return this.#settled(restarted);
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
    this.#pendingSections.clear();
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

  /**
   * Applies the declarative changes made since the last flush, the latest
   * value of each. Never throws: a scene call that fails puts its key (and
   * so the mesh or section) in `"error"`.
   */
  #flush(): void {
    for (const entry of this.#pendingDisplay) {
      const d = entry.display;
      const want = entry.desired;
      if (d.opacity !== want.opacity || d.visible !== want.visible) {
        entry.display = Object.freeze({ ...d, opacity: want.opacity, visible: want.visible });
      }
      this.#flushColoring(entry);
    }
    this.#pendingDisplay.clear();
    for (const entry of this.#pendingSections) {
      this.#flushSection(entry);
    }
    this.#pendingSections.clear();
  }

  /**
   * Sends a mesh's desired colouring when it differs from the one sent:
   * resolves its owned field (reused, or replaced when its key changed),
   * then sets the colour rule. Under the `display:` key, so a call that
   * throws sets that key's error (and one that succeeds clears it).
   */
  #flushColoring(entry: MeshEntry): void {
    const key = `display:${entry.id}`;
    const want = entry.desired.coloring;
    try {
      this.#host.track(key, () => {
        if (sameData(want, entry.sent)) {
          return;
        }
        if (want === null) {
          // The field is kept: colouring back on with the same key reuses it.
          this.#host.setColoring(entry.handle, null);
        } else {
          const field = this.#ownedField(entry, want, key);
          this.#host.setColoring(entry.handle, colorRule(field.handle, want));
        }
        entry.sent = want;
        entry.inflight = { value: want, rev: this.#host.status(key).revision };
      });
    } catch {
      // Reported through the key's status. A colouring the scene no longer
      // has (its field replaced before the failing call) is not shown.
      if (entry.sent === null && entry.display.coloring !== null) {
        entry.inflight = undefined;
        entry.display = Object.freeze({ ...entry.display, coloring: null });
      }
    }
  }

  /** The field for colouring `entry` by `want`: the owned one when its key matches, else a new one (replacing it). */
  #ownedField(entry: MeshEntry, want: DistanceColoring, key: string): OwnedField {
    const sign: SignMode = want.sign ?? "auto";
    const old = entry.field;
    if (
      old !== undefined &&
      old.to === want.distanceTo &&
      Object.is(old.maxDistance, want.maxDistance) &&
      old.sign === sign
    ) {
      return old;
    }
    if (old !== undefined) {
      entry.field = undefined;
      // Removing the field also removes the colouring by it.
      entry.sent = null;
      this.#host.removeField(old.handle);
    }
    const handle = this.#host.addField(
      {
        name: `fast-mesh-store:${++this.#fieldCount}`,
        on: entry.handle,
        to: want.distanceTo,
        maxDistance: want.maxDistance,
        sign,
      },
      key,
    );
    const field: OwnedField = {
      handle,
      id: this.#host.idOf(handle),
      to: want.distanceTo,
      toId: this.#idOf(want.distanceTo),
      maxDistance: want.maxDistance,
      sign,
    };
    entry.field = field;
    return field;
  }

  /**
   * A mesh's id in status keys, whether or not the store tracks it (a mesh
   * added straight to the scene); `undefined` for one the scene no longer
   * knows (removed).
   */
  #idOf(mesh: MeshHandle): number | undefined {
    try {
      return this.#host.idOf(mesh);
    } catch {
      return undefined;
    }
  }

  /** Sends each aspect of a section whose desired value differs from the one sent, under its own key. */
  #flushSection(entry: SectionEntry): void {
    for (const a of ASPECTS) {
      const want = entry.desired[a];
      if (sameData(want, entry.sent[a])) {
        continue;
      }
      try {
        this.#host.track(`section:${entry.id}:${a}`, () => {
          if (a === "plane") {
            this.#host.setSectionPlane(entry.handle, want as SectionPlane);
          } else if (a === "meshes") {
            this.#host.setSectionMeshes(entry.handle, want as readonly MeshHandle[]);
          } else {
            this.#host.setSectionFill(entry.handle, want as boolean);
          }
        });
      } catch {
        // Reported through the key's status.
        continue;
      }
      entry.sent = Object.freeze({ ...entry.sent, [a]: want });
      this.#arm(entry, a, want);
    }
  }

  /** Waits for the request now holding the key of a section's aspect, to apply `value`. */
  #arm<A extends Aspect>(entry: SectionEntry, a: A, value: SectionParts[A]): void {
    (entry.inflight as Record<Aspect, InFlight<unknown>>)[a] = {
      value,
      rev: this.#host.status(`section:${entry.id}:${a}`).revision,
    };
  }

  /**
   * Whether the request awaited under `key` is done: `"ready"` (apply it),
   * `"error"` (drop it), or `undefined` while it is pending.
   */
  #outcome(key: string, wait: InFlight<unknown>): "ready" | "error" | undefined {
    const s = this.#host.status(key);
    if (s.revision < wait.rev) {
      return undefined;
    }
    return s.status === "ready" ? "ready" : s.status === "error" ? "error" : undefined;
  }

  /** Applies the colouring and section aspects whose requests are done. */
  #applyDone(): void {
    for (const entry of this.#meshes.values()) {
      const wait = entry.inflight;
      if (wait === undefined) {
        continue;
      }
      const done = this.#outcome(`display:${entry.id}`, wait);
      if (done !== undefined) {
        entry.inflight = undefined;
        if (done === "ready" && entry.display.coloring !== wait.value) {
          entry.display = Object.freeze({ ...entry.display, coloring: wait.value });
        }
      }
    }
    for (const entry of this.#sections.values()) {
      for (const a of ASPECTS) {
        const wait = entry.inflight[a];
        if (wait === undefined) {
          continue;
        }
        const done = this.#outcome(`section:${entry.id}:${a}`, wait);
        if (done !== undefined) {
          delete entry.inflight[a];
          if (done === "ready" && entry.applied[a] !== wait.value) {
            entry.applied = Object.freeze({ ...entry.applied, [a]: wait.value });
          }
        }
      }
    }
  }

  #key(key: string): Combined {
    const s = this.#host.status(key);
    return s.status === "error" ? { status: "error", error: s.error } : { status: s.status };
  }

  /**
   * The status of a state from its own keys (all of their status, errors
   * included) and the keys it depends on, which only add `"computing"`:
   * their errors belong to the requests that made them (a failed edit
   * rejects its promise) and are ignored here. `pending` (a request
   * waiting for the next flush that differs from what was sent) also makes
   * it `"computing"`.
   */
  #status(own: readonly string[], dependent: readonly string[], pending = false): Combined {
    const parts = own.map((k) => this.#key(k));
    if (pending || dependent.some((k) => this.#host.status(k).status === "computing")) {
      parts.push({ status: "computing" });
    }
    return combine(parts);
  }

  /**
   * A mesh's status: its own `mesh:` and `display:` keys and the `field:`
   * key of its store field, made `"computing"` also by its `placement:` and
   * `edit:` keys, those of the mesh it is coloured by distance to, and the
   * history, and while a requested colouring waits for the next flush.
   * (Opacity and visibility are store-only: they never make it pending.)
   */
  #meshStatus(entry: MeshEntry): Combined {
    const id = entry.id;
    const own = [`mesh:${id}`, `display:${id}`];
    const dependent = [`placement:${id}`, `edit:${id}`, "history"];
    const field = entry.field;
    if (field !== undefined) {
      own.push(`field:${field.id}`);
      if (entry.sent !== null && field.toId !== undefined && field.toId !== id) {
        dependent.push(`placement:${field.toId}`, `edit:${field.toId}`);
      }
    }
    return this.#status(own, dependent, this.#pendingDisplay.has(entry) && !sameData(entry.desired.coloring, entry.sent));
  }

  /**
   * A section's status: its own `section:` keys, made `"computing"` also by
   * the `placement:` and `edit:` keys of its meshes and the history, and
   * while a requested aspect waits for the next flush.
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
    const pending = this.#pendingSections.has(entry) && ASPECTS.some((a) => !sameData(entry.desired[a], entry.sent[a]));
    return this.#status(own, dependent, pending);
  }

  /**
   * Rebuilds the snapshot from the store and the scene, reusing every
   * unchanged part of the previous one; publishes (and notifies) only when
   * something changed.
   */
  #refresh(): void {
    this.#applyDone();
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
