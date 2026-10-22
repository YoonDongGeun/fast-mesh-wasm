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
import { targetsOf } from "../handles.js";
import { sectionAspects } from "../worker/protocol.js";
import { type Host, hostOf, type StoreAddMeshOptions } from "./host.js";
import {
  type Combined,
  type ContactGroup,
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
 * `(to, maxDistance, sign)`, `to` taken as a set: kept while the colouring is
 * off, reused when it comes back with the same key, replaced when the key
 * changes.
 */
interface OwnedField {
  handle: FieldHandle;
  /** Its id in status keys (`field:<id>`). */
  id: number;
  /** Its targets, by ascending id (the order they were given to the scene). */
  to: readonly MeshHandle[];
  /** The ids of `to` in status keys (`undefined` for a mesh the scene no longer knows). */
  toIds: readonly (number | undefined)[];
  maxDistance: number;
  sign: SignMode;
}

/** `c` without `mesh` among its targets (the same object when it has none to drop). */
function withoutTarget(c: DistanceColoring | null, mesh: MeshHandle): DistanceColoring | null {
  if (c === null || !Array.isArray(c.distanceTo) || !c.distanceTo.includes(mesh)) {
    return c;
  }
  const to = (c.distanceTo as readonly MeshHandle[]).filter((m) => m !== mesh);
  return Object.freeze({ ...c, distanceTo: Object.freeze(to) });
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
  /** The status keys its state was last derived from (`history` aside), as indexed. */
  keys?: readonly string[];
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
  /** The status keys its state was last derived from (`history` aside), as indexed. */
  keys?: readonly string[];
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
/** Set by `SceneStore`'s static block: the internal {@link setFrameLoop}. */
let setFrameLoopOf: (store: SceneStore, on: boolean) => void;
/** Set by `SceneStore`'s static block: the test-only {@link setFullRefresh}. */
let setFullRefreshOf: (store: SceneStore, on: boolean) => void;

/** Adds `entry` under each of `keys` in `deps`. */
function addDeps<E>(deps: Map<string, Set<E>>, entry: E, keys: readonly string[]): void {
  for (const k of keys) {
    let set = deps.get(k);
    if (set === undefined) {
      set = new Set();
      deps.set(k, set);
    }
    set.add(entry);
  }
}

/** Removes `entry` from under each of `keys` in `deps`. */
function dropDeps<E>(deps: Map<string, Set<E>>, entry: E, keys: readonly string[]): void {
  for (const k of keys) {
    const set = deps.get(k);
    if (set !== undefined && set.delete(entry) && set.size === 0) {
      deps.delete(k);
    }
  }
}

/** The first state in `"error"`, as a status, or `undefined`. */
function firstError(states: Iterable<MeshState | SectionState>): Combined | undefined {
  for (const state of states) {
    if (state.status === "error") {
      return { status: "error", error: state.error };
    }
  }
  return undefined;
}

/** Whether two key lists are equal, in order. */
function sameKeys(a: readonly string[], b: readonly string[]): boolean {
  return a.length === b.length && a.every((k, i) => k === b[i]);
}

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
  // The refresh is incremental: it re-derives only the states marked dirty,
  // by the actions and flushes that touch them and by the status keys they
  // read (`#meshDeps`/`#sectionDeps`, from the host's `changed` keys).
  // Every state reads whether `history` is "computing": when that flips,
  // all of them are re-derived.
  /** Meshes (tracked or just removed) whose state must be re-derived. */
  readonly #dirtyMeshes = new Set<MeshHandle>();
  /** Sections (tracked or just removed) whose state must be re-derived. */
  readonly #dirtySections = new Set<SectionHandle>();
  /** Every state must be re-derived (at the start, and when `history` starts or stops computing). */
  #allDirty = true;
  /** Whether `history` was "computing" when the states were last derived. */
  #historyBusy = false;
  /** Test-only: every refresh re-derives every state (see {@link setFullRefresh}). */
  #fullRefresh = false;
  /** The meshes whose state reads a status key, by key (`history` aside). */
  readonly #meshDeps = new Map<string, Set<MeshEntry>>();
  /** The sections whose state reads a status key, by key (`history` aside). */
  readonly #sectionDeps = new Map<string, Set<SectionEntry>>();
  /** Meshes that may have a colouring in flight (a superset; settled ones leave at the next refresh). */
  readonly #inflightMeshes = new Set<MeshEntry>();
  /** Sections that may have an aspect in flight (a superset, likewise). */
  readonly #inflightSections = new Set<SectionEntry>();
  /** The states in the snapshot whose status is `"computing"`. */
  #busy = 0;
  /** The states in the snapshot whose status is `"error"`. */
  #failed = 0;
  /** The first state in "error" (meshes, then sections), kept until a state enters or leaves "error". */
  #firstFailed: Combined | undefined;
  /** A state entered or left "error" (or changed its error) since `#firstFailed` was found. */
  #failedMoved = false;
  /** A scene event came: the snapshot may be stale until the next refresh. */
  #stale = false;
  /** A refresh on a microtask is queued (for scene events, and snapshots not yet announced). */
  #updateQueued = false;
  /** A snapshot was published and its listeners not yet called. */
  #owed = false;
  /** Contact groups by id (frozen). */
  readonly #groups = new Map<string, ContactGroup>();
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
    // Scene events in one task make one snapshot, on a microtask; reading
    // the snapshot before then refreshes it (see `getSnapshot`).
    const event = (): void => {
      if (!this.#disposed) {
        this.#stale = true;
        this.#queueUpdate();
      }
    };
    this.#unsubscribe = [scene.onChanges(event), this.#host.onStatus(event), () => this.#host.stop()];
    this.#update();
  }

  // ---- Reading ----

  /**
   * The current snapshot: frozen, replaced (never mutated) on change. It is
   * up to date with the scene's statuses when read: a status change not yet
   * announced refreshes it here, and its listeners are called on a microtask.
   */
  getSnapshot = (): SceneSnapshot => {
    if (!this.#disposed && (this.#stale || this.#host.changed.size > 0)) {
      this.#refresh();
      if (this.#owed) {
        this.#queueUpdate();
      }
    }
    return this.#snapshot;
  };

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
    this.#update();
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
    this.#dirtyMeshes.add(handle);
    this.#update();
    return handle;
  }

  /**
   * Removes a mesh; it leaves the snapshot (and the selection) at once. The
   * scene drops the fields measured to it alone, and the colourings by them:
   * a mesh coloured by distance to it alone shows no colouring, keeps its
   * rule in `desired`, and is in `"error"` (`UNKNOWN_MESH`) from the next
   * flush until its colouring changes. A colouring measured to it and other
   * meshes keeps the others (its `distanceTo` loses the mesh). It leaves its
   * contact groups, whose members' partners follow. Sections stop cutting it.
   */
  removeMesh(mesh: MeshHandle): void {
    this.#alive();
    this.#host.removeMesh(mesh);
    const entry = this.#meshes.get(mesh);
    if (entry !== undefined) {
      this.#meshes.delete(mesh);
      this.#pendingDisplay.delete(entry);
      this.#inflightMeshes.delete(entry);
      this.#unindexMesh(entry);
      this.#dirtyMeshes.add(mesh);
      const display = `display:${entry.id}`;
      const field = entry.field === undefined ? undefined : `field:${entry.field.id}`;
      this.#host.forgetErrors((k) => k === display || k === field);
    }
    if (this.#selection === mesh) {
      this.#selection = null;
    }
    for (const other of this.#meshes.values()) {
      const owned = other.field;
      if (owned === undefined || !owned.to.includes(mesh)) {
        continue;
      }
      if (owned.to.length > 1) {
        // The scene keeps the field without the mesh, and the colouring.
        const at = owned.to.indexOf(mesh);
        other.field = {
          ...owned,
          to: owned.to.filter((_, k) => k !== at),
          toIds: owned.toIds.filter((_, k) => k !== at),
        };
        const prune = (d: DisplayState): DisplayState => {
          const c = withoutTarget(d.coloring, mesh);
          return c === d.coloring ? d : Object.freeze({ ...d, coloring: c });
        };
        other.display = prune(other.display);
        other.desired = prune(other.desired);
        other.sent = withoutTarget(other.sent, mesh);
        if (other.inflight !== undefined) {
          other.inflight.value = withoutTarget(other.inflight.value, mesh);
        }
        // Re-derived, which also re-indexes it under its remaining targets' keys.
        this.#dirtyMeshes.add(other.handle);
        continue;
      }
      // Gone from the scene with the mesh, and so is a colouring by it.
      const fieldKey = `field:${owned.id}`;
      this.#host.forgetErrors((k) => k === fieldKey);
      other.field = undefined;
      other.sent = null;
      other.inflight = undefined;
      this.#dirtyMeshes.add(other.handle);
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
      const { applied, sent, desired } = s;
      s.applied = this.#withMeshes(s.applied, without(s.applied.meshes));
      s.sent = this.#withMeshes(s.sent, without(s.sent.meshes));
      s.desired = this.#withMeshes(s.desired, without(s.desired.meshes));
      if (s.applied !== applied || s.sent !== sent || s.desired !== desired) {
        this.#dirtySections.add(s.handle);
      }
      const inflight = s.inflight.meshes;
      if (inflight !== undefined) {
        inflight.value = without(inflight.value);
      }
    }
    for (const [id, g] of this.#groups) {
      if (!g.meshes.includes(mesh)) {
        continue;
      }
      const pairs = g.pairs;
      const left: ContactGroup = Object.freeze({
        ...g,
        meshes: without(g.meshes),
        pairs:
          pairs === undefined || pairs === "all"
            ? pairs
            : Object.freeze(pairs.filter(([p, q]) => p !== mesh && q !== mesh)),
      });
      this.#groups.set(id, left);
      // A member left without partners stops being coloured.
      this.#applyGroup(left);
    }
    this.#update();
  }

  /** `parts` with `meshes`, the same object when they are. */
  #withMeshes(parts: SectionParts, meshes: readonly MeshHandle[]): SectionParts {
    return meshes === parts.meshes ? parts : Object.freeze({ ...parts, meshes });
  }

  /** Sets a mesh's placement (one undo step). */
  setPlacement(mesh: MeshHandle, placement: PlacementInput): void {
    this.#alive();
    this.#host.setPlacement(mesh, placement);
    this.#update();
  }

  /** Starts a drag of a mesh; see the scene's `drag()`. Ending or cancelling it updates the snapshot. */
  drag(mesh: MeshHandle): DragSession {
    this.#alive();
    const session = this.#host.scene.drag(mesh);
    const after = (): void => {
      if (!this.#disposed) {
        this.#update();
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
    this.#update();
    const after = (): void => {
      if (!this.#disposed) {
        this.#update();
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
    this.#update();
  }

  /** Redoes the latest undone step. */
  redo(): void {
    this.#alive();
    this.#host.redo();
    this.#update();
  }

  // ---- Display ----

  /**
   * Changes how a mesh is shown. `desired` changes at once; `display` follows
   * at the next `commitFrame()` (or a microtask without a frame loop). A
   * patch equal to the current request changes nothing.
   *
   * A `coloring` colours the mesh by distance to `distanceTo` (one mesh or
   * several) through a field the store owns (one per mesh, named
   * `fast-mesh-store:<n>`, kept while colouring is off and replaced only
   * when the set of meshes in `distanceTo`, `maxDistance` or `sign`
   * changes). `display.coloring` is the colouring
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
    this.#dirtyMeshes.add(mesh);
    this.#schedule();
    this.#update();
  }

  // ---- Contact groups ----

  /**
   * Sets contact group `id`, or removes it with `null`. Every member is
   * coloured by `group.coloring` with its partners (the members it is paired
   * with) as `distanceTo`, through one field per member measured to all of
   * its partners: each vertex shows the smallest signed distance over them
   * (the nearest surface, or the deepest penetration). Moving a member then
   * recomputes only its own field and those of its partners. A member
   * without partners shows no colouring.
   *
   * Members are coloured through {@link setDisplay}, so `desired` and
   * `display` show the colourings, and a later `setDisplay` colouring of a
   * member overrides the group's until the group is set again. Members
   * dropped from the group, or all of them when it is removed, stop being
   * coloured. A removed mesh leaves its groups (and their pairs).
   *
   * @throws `UNKNOWN_MESH` for a mesh not in the store; `INVALID_ARGUMENT`
   *   for a mesh listed twice, a mesh already in another group, a pair that
   *   is not two different members, or a bad colouring.
   */
  setContactGroup(id: string, group: ContactGroup | null): void {
    this.#alive();
    if (typeof id !== "string") {
      throw invalid("a contact group id must be a string");
    }
    const old = this.#groups.get(id);
    if (group === null) {
      if (old !== undefined) {
        this.#groups.delete(id);
        for (const m of old.meshes) {
          this.setDisplay(m, { coloring: null });
        }
      }
      return;
    }
    if (typeof group !== "object" || !Array.isArray(group.meshes)) {
      throw invalid("a contact group needs a list of meshes");
    }
    const meshes: readonly MeshHandle[] = group.meshes;
    meshes.forEach((m, k) => {
      this.#mesh(m);
      if (meshes.indexOf(m) !== k) {
        throw invalid("a contact group lists a mesh twice");
      }
      for (const [other, g] of this.#groups) {
        if (other !== id && g.meshes.includes(m)) {
          throw invalid(`a mesh is already in contact group "${other}"`);
        }
      }
    });
    const pairs = group.pairs;
    if (pairs !== undefined && pairs !== "all") {
      if (!Array.isArray(pairs)) {
        throw invalid('pairs must be a list of mesh pairs or "all"');
      }
      for (const pair of pairs) {
        if (!Array.isArray(pair) || pair.length !== 2 || pair[0] === pair[1]) {
          throw invalid("a pair needs two different meshes");
        }
        if (!meshes.includes(pair[0]) || !meshes.includes(pair[1])) {
          throw invalid("a pair's meshes must be members of the group");
        }
      }
    }
    if (typeof group.coloring !== "object" || group.coloring === null) {
      throw invalid("a contact group needs a coloring");
    }
    const next: ContactGroup = frozenCopy({ meshes, pairs, coloring: group.coloring });
    this.#groups.set(id, next);
    for (const m of old?.meshes ?? []) {
      if (!next.meshes.includes(m)) {
        this.setDisplay(m, { coloring: null });
      }
    }
    this.#applyGroup(next);
  }

  /** Colours every member of `g` by distance to its partners (none: no colouring). */
  #applyGroup(g: ContactGroup): void {
    for (const m of g.meshes) {
      const partners = g.meshes.filter(
        (p) =>
          p !== m &&
          (g.pairs === undefined ||
            g.pairs === "all" ||
            g.pairs.some(([a, b]) => (a === m && b === p) || (a === p && b === m))),
      );
      this.setDisplay(m, {
        coloring: partners.length === 0 ? null : { ...g.coloring, distanceTo: partners },
      });
    }
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
    this.#dirtySections.add(handle);
    this.#update();
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
    this.#dirtySections.add(section);
    this.#schedule();
    this.#update();
  }

  /** Removes a section; it leaves the snapshot at once. */
  removeSection(section: SectionHandle): void {
    this.#alive();
    this.#host.removeSection(section);
    const entry = this.#sections.get(section);
    if (entry !== undefined) {
      this.#sections.delete(section);
      this.#pendingSections.delete(entry);
      this.#inflightSections.delete(entry);
      this.#unindexSection(entry);
      this.#dirtySections.add(section);
      const own = `section:${entry.id}`;
      this.#host.forgetErrors((k) => k === own || k.startsWith(`${own}:`));
    }
    this.#update();
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
    this.#update();
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
        this.#inflightMeshes.add(entry);
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
   * Removes the distance fields the store owns (`fast-mesh-store:<n>`, and
   * so the colourings by them) and stops listening to the scene. The rest of
   * the scene is left as it is (it is the app's). On a Worker scene the
   * removals are queued like any call, and sent with the scene's next
   * `commitFrame()`; errors from a scene that can no longer take them (a
   * poisoned Worker scene, for one) are ignored. Later actions throw
   * `INVALID_ARGUMENT`; the last snapshot stays readable.
   */
  dispose(): void {
    if (this.#disposed) {
      return;
    }
    this.#disposed = true;
    for (const entry of this.#meshes.values()) {
      const field = entry.field;
      if (field === undefined) {
        continue;
      }
      entry.field = undefined;
      try {
        this.#host.removeField(field.handle);
      } catch {
        // A scene that can no longer take it (a poisoned or terminated
        // Worker scene, a freed Scene): nothing left to clean up there.
      }
    }
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
      this.#update();
    });
  }

  /**
   * Applies the declarative changes made since the last flush, the latest
   * value of each. Never throws: a scene call that fails puts its key (and
   * so the mesh or section) in `"error"`.
   */
  #flush(): void {
    for (const entry of this.#pendingDisplay) {
      // Its pending state ends here, whatever the flush does.
      this.#dirtyMeshes.add(entry.handle);
      const d = entry.display;
      const want = entry.desired;
      if (d.opacity !== want.opacity || d.visible !== want.visible) {
        entry.display = Object.freeze({ ...d, opacity: want.opacity, visible: want.visible });
      }
      this.#flushColoring(entry);
    }
    this.#pendingDisplay.clear();
    for (const entry of this.#pendingSections) {
      this.#dirtySections.add(entry.handle);
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
        this.#inflightMeshes.add(entry);
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
    // The targets as a set: by ascending id (ids unknown to the scene last).
    const order = (m: MeshHandle): number => this.#idOf(m) ?? Number.POSITIVE_INFINITY;
    const to = [...targetsOf(want.distanceTo)].sort((a, b) => order(a) - order(b));
    const old = entry.field;
    if (
      old !== undefined &&
      old.to.length === to.length &&
      old.to.every((m, k) => m === to[k]) &&
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
        to,
        maxDistance: want.maxDistance,
        sign,
      },
      key,
    );
    const field: OwnedField = {
      handle,
      id: this.#host.idOf(handle),
      to: Object.freeze(to),
      toIds: Object.freeze(to.map((m) => this.#idOf(m))),
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
    this.#inflightSections.add(entry);
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

  /**
   * Applies the colouring and section aspects whose requests are done, and
   * marks the states that change dirty. Only the entries that may have one
   * in flight are visited (all of them under the test-only full refresh).
   */
  #applyDone(): void {
    for (const entry of this.#fullRefresh ? [...this.#meshes.values()] : this.#inflightMeshes) {
      const wait = entry.inflight;
      if (wait === undefined) {
        this.#inflightMeshes.delete(entry);
        continue;
      }
      const done = this.#outcome(`display:${entry.id}`, wait);
      if (done !== undefined) {
        entry.inflight = undefined;
        this.#inflightMeshes.delete(entry);
        if (done === "ready" && entry.display.coloring !== wait.value) {
          entry.display = Object.freeze({ ...entry.display, coloring: wait.value });
          this.#dirtyMeshes.add(entry.handle);
        }
      }
    }
    for (const entry of this.#fullRefresh ? [...this.#sections.values()] : this.#inflightSections) {
      let waiting = false;
      for (const a of ASPECTS) {
        const wait = entry.inflight[a];
        if (wait === undefined) {
          continue;
        }
        const done = this.#outcome(`section:${entry.id}:${a}`, wait);
        if (done === undefined) {
          waiting = true;
          continue;
        }
        delete entry.inflight[a];
        if (done === "ready" && entry.applied[a] !== wait.value) {
          entry.applied = Object.freeze({ ...entry.applied, [a]: wait.value });
          this.#dirtySections.add(entry.handle);
        }
      }
      if (!waiting) {
        this.#inflightSections.delete(entry);
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
   * rejects its promise) and are ignored here. `history` is a dependent key
   * of every state. `pending` (a request waiting for the next flush that
   * differs from what was sent) also makes it `"computing"`.
   */
  #status(own: readonly string[], dependent: readonly string[], pending: boolean): Combined {
    const parts = own.map((k) => this.#key(k));
    if (
      pending ||
      dependent.some((k) => this.#host.status(k).status === "computing") ||
      this.#host.status("history").status === "computing"
    ) {
      parts.push({ status: "computing" });
    }
    return combine(parts);
  }

  /**
   * A mesh's status: its own `mesh:` and `display:` keys and the `field:`
   * key of its store field, made `"computing"` also by its `placement:` and
   * `edit:` keys, those of the meshes it is coloured by distance to, and the
   * history, and while a requested colouring waits for the next flush.
   * (Opacity and visibility are store-only: they never make it pending.)
   * Indexes the mesh under the keys it read.
   */
  #meshStatus(entry: MeshEntry): Combined {
    const id = entry.id;
    const own = [`mesh:${id}`, `display:${id}`];
    const dependent = [`placement:${id}`, `edit:${id}`];
    const field = entry.field;
    if (field !== undefined) {
      own.push(`field:${field.id}`);
      if (entry.sent !== null) {
        for (const to of field.toIds) {
          if (to !== undefined && to !== id) {
            dependent.push(`placement:${to}`, `edit:${to}`);
          }
        }
      }
    }
    const keys = [...own, ...dependent];
    if (entry.keys === undefined || !sameKeys(entry.keys, keys)) {
      this.#unindexMesh(entry);
      addDeps(this.#meshDeps, entry, keys);
      entry.keys = keys;
    }
    return this.#status(own, dependent, this.#pendingDisplay.has(entry) && !sameData(entry.desired.coloring, entry.sent));
  }

  /**
   * A section's status: its own `section:` keys, made `"computing"` also by
   * the `placement:` and `edit:` keys of its meshes and the history, and
   * while a requested aspect waits for the next flush. Indexes the section
   * under the keys it read.
   */
  #sectionStatus(entry: SectionEntry): Combined {
    const own = [`section:${entry.id}`, ...ASPECT_KEYS.map((a) => `section:${entry.id}:${a}`)];
    const dependent: string[] = [];
    for (const m of entry.applied.meshes) {
      const mesh = this.#meshes.get(m);
      if (mesh !== undefined) {
        dependent.push(`placement:${mesh.id}`, `edit:${mesh.id}`);
      }
    }
    const keys = [...own, ...dependent];
    if (entry.keys === undefined || !sameKeys(entry.keys, keys)) {
      this.#unindexSection(entry);
      addDeps(this.#sectionDeps, entry, keys);
      entry.keys = keys;
    }
    const pending = this.#pendingSections.has(entry) && ASPECTS.some((a) => !sameData(entry.desired[a], entry.sent[a]));
    return this.#status(own, dependent, pending);
  }

  #unindexMesh(entry: MeshEntry): void {
    if (entry.keys !== undefined) {
      dropDeps(this.#meshDeps, entry, entry.keys);
      entry.keys = undefined;
    }
  }

  #unindexSection(entry: SectionEntry): void {
    if (entry.keys !== undefined) {
      dropDeps(this.#sectionDeps, entry, entry.keys);
      entry.keys = undefined;
    }
  }

  /**
   * Marks dirty the states that read a key whose status changed since the
   * last refresh, and clears the host's set; every state when `history`
   * starts or stops computing.
   */
  #takeChanged(): void {
    const changed = this.#host.changed;
    for (const key of changed) {
      for (const entry of this.#meshDeps.get(key) ?? []) {
        this.#dirtyMeshes.add(entry.handle);
      }
      for (const entry of this.#sectionDeps.get(key) ?? []) {
        this.#dirtySections.add(entry.handle);
      }
    }
    changed.clear();
    // Every state reads only whether `history` is "computing": when that
    // flips (an undo or redo sent, or done), every state is re-derived.
    const historyBusy = this.#host.status("history").status === "computing";
    if (historyBusy !== this.#historyBusy) {
      this.#historyBusy = historyBusy;
      this.#allDirty = true;
    }
  }

  /** Counts `state` (`by` 1) in, or (`-1`) out of, the busy and failed states. */
  #count(state: { status: string }, by: number): void {
    if (state.status === "computing") {
      this.#busy += by;
    } else if (state.status === "error") {
      this.#failed += by;
      this.#failedMoved = true;
    }
  }

  /** Refreshes the snapshot, then calls the listeners when it changed. */
  #update(): void {
    this.#refresh();
    this.#notify();
  }

  /** Queues an {@link #update} on a microtask (once). */
  #queueUpdate(): void {
    if (this.#updateQueued) {
      return;
    }
    this.#updateQueued = true;
    queueMicrotask(() => {
      this.#updateQueued = false;
      // Nothing left to do when a refresh in the meantime took the event (a
      // store `commitFrame()` takes its own scene's): no scene call then, so
      // a scene freed in the same task is never touched.
      if (!this.#disposed && (this.#stale || this.#host.changed.size > 0 || this.#owed)) {
        this.#update();
      }
    });
  }

  /**
   * Re-derives the dirty states from the store and the scene, sharing every
   * other part of the previous snapshot; publishes a new snapshot only when
   * something changed (its listeners are called by {@link #notify}).
   */
  #refresh(): void {
    this.#stale = false;
    this.#applyDone();
    this.#takeChanged();
    const prev = this.#snapshot;
    const revision = prev.revision + 1;
    let changed = false;
    if (this.#allDirty || this.#fullRefresh) {
      this.#allDirty = false;
      for (const h of this.#meshes.keys()) {
        this.#dirtyMeshes.add(h);
      }
      for (const h of prev.meshes.keys()) {
        this.#dirtyMeshes.add(h);
      }
      for (const h of this.#sections.keys()) {
        this.#dirtySections.add(h);
      }
      for (const h of prev.sections.keys()) {
        this.#dirtySections.add(h);
      }
    }

    // Copied from the previous snapshot's map on the first change: order
    // kept, removed states deleted, new ones appended (as in `#meshes`).
    let meshes: Map<MeshHandle, MeshState> | undefined;
    let added = 0;
    for (const handle of this.#dirtyMeshes) {
      const entry = this.#meshes.get(handle);
      const old = prev.meshes.get(handle);
      if (entry === undefined) {
        if (old !== undefined) {
          meshes ??= new Map(prev.meshes);
          meshes.delete(handle);
          this.#count(old, -1);
        }
        continue;
      }
      const s = this.#meshStatus(entry);
      if (
        old !== undefined &&
        old.display === entry.display &&
        old.desired === entry.desired &&
        old.status === s.status &&
        old.error === s.error
      ) {
        continue;
      }
      const state: MeshState = { display: entry.display, desired: entry.desired, status: s.status, revision };
      if (s.error !== undefined) {
        state.error = s.error;
      }
      meshes ??= new Map(prev.meshes);
      if (old === undefined) {
        added++;
      } else {
        this.#count(old, -1);
      }
      meshes.set(handle, Object.freeze(state));
      this.#count(state, 1);
    }
    this.#dirtyMeshes.clear();
    if (meshes !== undefined) {
      changed = true;
      if (added > 1) {
        // Several added at once: put them in the order they were added.
        const m = meshes;
        meshes = new Map([...this.#meshes.keys()].map((h) => [h, m.get(h)!]));
      }
    }

    let sections: Map<SectionHandle, SectionState> | undefined;
    added = 0;
    for (const handle of this.#dirtySections) {
      const entry = this.#sections.get(handle);
      const old = prev.sections.get(handle);
      if (entry === undefined) {
        if (old !== undefined) {
          sections ??= new Map(prev.sections);
          sections.delete(handle);
          this.#count(old, -1);
        }
        continue;
      }
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
        continue;
      }
      const state: SectionState = { ...a, desired: entry.desired, status: s.status, revision };
      if (s.error !== undefined) {
        state.error = s.error;
      }
      sections ??= new Map(prev.sections);
      if (old === undefined) {
        added++;
      } else {
        this.#count(old, -1);
      }
      sections.set(handle, Object.freeze(state));
      this.#count(state, 1);
    }
    this.#dirtySections.clear();
    if (sections !== undefined) {
      changed = true;
      if (added > 1) {
        const m = sections;
        sections = new Map([...this.#sections.keys()].map((h) => [h, m.get(h)!]));
      }
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
    // As `combine` over every mesh, then every section, then the history:
    // the first error wins, then any "computing"; "ready" reads "idle".
    const meshStates = meshes ?? prev.meshes;
    const sectionStates = sections ?? prev.sections;
    const historyKey = this.#key("history");
    let overall: Combined = { status: "idle" };
    if (this.#failedMoved) {
      this.#failedMoved = false;
      this.#firstFailed =
        this.#failed > 0 ? (firstError(meshStates.values()) ?? firstError(sectionStates.values())) : undefined;
    }
    if (this.#firstFailed !== undefined) {
      overall = this.#firstFailed;
    } else if (historyKey.status === "error") {
      overall = historyKey;
    } else if (this.#busy > 0 || historyKey.status === "computing") {
      overall = { status: "computing" };
    }
    const status = overall.status === "ready" ? "idle" : overall.status;
    if (status !== prev.status || overall.error !== prev.error) {
      changed = true;
    }
    if (!changed) {
      return;
    }
    const snap: SceneSnapshot = {
      revision,
      meshes: meshes === undefined ? prev.meshes : Object.freeze(meshes),
      history,
      sections: sections === undefined ? prev.sections : Object.freeze(sections),
      selection: this.#selection,
      status,
    };
    if (overall.error !== undefined) {
      snap.error = overall.error;
    }
    this.#snapshot = Object.freeze(snap);
    this.#owed = true;
  }

  /** Calls the listeners once for the snapshots published since they were last called. */
  #notify(): void {
    if (!this.#owed) {
      return;
    }
    this.#owed = false;
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
    setFrameLoopOf = (store, on) => {
      store.#frameLoop = on;
      if (!on && !store.#disposed && (store.#pendingDisplay.size > 0 || store.#pendingSections.size > 0)) {
        store.#schedule();
      }
    };
    setFullRefreshOf = (store, on) => {
      store.#fullRefresh = on;
      if (on && !store.#disposed) {
        store.#update();
      }
    };
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

/**
 * Internal, not exported from `fast-mesh-wasm/store`: switches the store's
 * flushing. `commitFrame()` switches it on (declarative changes wait for the
 * next frame); `false` switches it back to flushing on a microtask, and
 * flushes the changes waiting for a frame. The r3f frame driver calls it
 * when the store's last driver unmounts.
 */
export function setFrameLoop(store: SceneStore, on: boolean): void {
  setFrameLoopOf(store, on);
}

/**
 * Test-only, not exported from `fast-mesh-wasm/store`: with `true`, every
 * refresh re-derives every state (the full rebuild the incremental refresh
 * must match), starting with one at once; `false` goes back to re-deriving
 * only the dirty states.
 */
export function setFullRefresh(store: SceneStore, on: boolean): void {
  setFullRefreshOf(store, on);
}
