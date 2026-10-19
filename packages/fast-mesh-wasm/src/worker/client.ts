// The main-thread `WorkerScene`: a `SceneLike` whose engine runs in a Worker.
// Handles are allocated here, synchronously, with client ids. Requests queue
// per frame and go to the Worker as one batch on `commitFrame()` (or right
// away, on a microtask, when a call returns a promise), coalesced like the
// engine coalesces them. Results come back with the bytes of every dirty
// range, which are applied to main-thread mirrors of each mesh's buffers; the
// mirrors back `views()`, and placements and section results are mirrored
// too, so everything the adapters read stays synchronous. Drags are
// optimistic: a move shows at once and holds until the Worker has the drag.
import type { Indices, Placement, PlacementInput, Positions64, Vec3 } from "../api.js";
import type { Range } from "../changes.js";
import { FastMeshError, type FastMeshErrorCode } from "../core.js";
import {
  type AddMeshOptions,
  type Analysis,
  type AnalyzeOptions,
  type ClosestHit,
  type ClosestPointOptions,
  type ColorRule,
  type Contact,
  type DistanceFieldDesc,
  type DragSession,
  type EditOp,
  type EditResult,
  FieldHandle,
  type FieldStatus,
  type FindContactsOptions,
  type MeshChange,
  type MeshDistance,
  MeshHandle,
  type RaycastOptions,
  type RayHit,
  type RepairOp,
  type RepairReport,
  type RepairReportOf,
  rotationMatrix,
  type SectionDesc,
  type SectionGap,
  SectionHandle,
  type SectionPlane,
  type SectionResult,
} from "../scene.js";
import type { SceneLike } from "../scene-like.js";
import { type MeshViews, ViewGuard } from "../views.js";
import {
  type ClientId,
  type ColorRuleWire,
  type ErrorWire,
  type FieldDescWire,
  isLastWins,
  keyOf,
  type MeshChangeWire,
  type QueryMap,
  type Request,
  type Result,
  type SectionDescWire,
  supersedes,
  type WorkerMessage,
} from "./protocol.js";

/** What {@link WorkerScene} needs of a Worker; a browser `Worker` fits. */
export interface WorkerLike {
  postMessage(msg: unknown, transfer?: Transferable[]): void;
  /** `"message"` events carry the message in `data`. */
  addEventListener(type: "message" | "error", cb: (e: any) => void): void;
  terminate(): void;
}

/** Options of {@link createWorkerScene}. */
export interface WorkerSceneOptions {
  /**
   * The Worker script, loaded as a module Worker. Defaults to the package's
   * `worker-entry.js` next to this module, which bundlers pick up from the
   * `new Worker(new URL(...), { type: "module" })` pattern.
   */
  workerUrl?: string | URL;
  /** Creates the Worker instead (overrides `workerUrl`). */
  createWorker?: () => WorkerLike;
  /**
   * Development checks, off by default: render views are then `Proxy`
   * objects that warn once when read after a mirror they view was replaced
   * (a mesh's slot count changed, a buffer turned on or off, the mesh was
   * removed). See {@link MeshViews}.
   */
  debug?: boolean;
}

/** The state of a request key, from {@link WorkerScene.status}. */
export interface KeyStatus {
  /**
   * `"computing"` while a request with this key is queued or running,
   * `"ready"` or `"error"` once the latest one finished, `"idle"` for a key
   * never used.
   */
  status: "idle" | "computing" | "ready" | "error";
  /** The revision of the request the status is about (0 when idle). */
  revision: number;
  error?: FastMeshError;
}

/**
 * Starts a Worker running the scene engine and resolves once it is ready.
 * Rejects with `INSTANCE_POISONED` when the Worker could not load fast-mesh.
 */
export async function createWorkerScene(options: WorkerSceneOptions = {}): Promise<WorkerScene> {
  const worker =
    options.createWorker?.() ??
    ((options.workerUrl === undefined
      ? new Worker(new URL("./worker-entry.js", import.meta.url), { type: "module" })
      : new Worker(options.workerUrl, { type: "module" })) as unknown as WorkerLike);
  const scene = new WorkerScene(TOKEN, worker, options.debug ?? false);
  try {
    await scene[STARTED];
  } catch (e) {
    scene.terminate();
    throw e;
  }
  return scene;
}

/** Keeps the constructor to {@link createWorkerScene}. */
const TOKEN = Symbol("WorkerScene");
const STARTED = Symbol("started");

/** Elements per slot of each buffer. */
const STRIDE = { positions: 3, normals: 3, colors: 4, indices: 3 } as const;
type BufferName = keyof typeof STRIDE;
const BUFFERS = Object.keys(STRIDE) as BufferName[];
const MAKE = {
  positions: (b: ArrayBuffer | number) => new Float32Array(b as ArrayBuffer),
  normals: (b: ArrayBuffer | number) => new Float32Array(b as ArrayBuffer),
  colors: (b: ArrayBuffer | number) => new Uint8Array(b as ArrayBuffer),
  indices: (b: ArrayBuffer | number) => new Uint32Array(b as ArrayBuffer),
};

const IDENTITY = [1, 0, 0, 0, 1, 0, 0, 0, 1];

/** A placement with its column-major `matrix4`. */
interface Pose {
  rotation: number[];
  translation: Vec3;
  matrix4: Float64Array;
}

/** `placement · translate(origin)`, column-major, as `Scene.matrix4`. */
function poseOf(rotation: number[], translation: Vec3, origin: Vec3): Pose {
  const r = rotation;
  const o = origin;
  const m = new Float64Array(16);
  for (let row = 0; row < 3; row++) {
    for (let col = 0; col < 3; col++) {
      m[4 * col + row] = r[3 * row + col];
    }
    m[12 + row] = r[3 * row] * o[0] + r[3 * row + 1] * o[1] + r[3 * row + 2] * o[2] + translation[row];
  }
  m[15] = 1;
  return { rotation, translation, matrix4: m };
}

function samePose(a: Pose, b: Pose): boolean {
  return a.matrix4.every((v, k) => v === b.matrix4[k]) && a.translation.every((v, k) => v === b.translation[k]);
}

/**
 * The engine's default origin, the bounding-box centre, or `undefined` when
 * the positions are malformed (the engine then reports the error).
 */
function boxCentre(p: Float32Array | Float64Array): Vec3 | undefined {
  if (p.length % 3 !== 0) {
    return undefined;
  }
  const min = [Infinity, Infinity, Infinity];
  const max = [-Infinity, -Infinity, -Infinity];
  for (let k = 0; k < p.length; k++) {
    const v = p[k];
    if (!Number.isFinite(v)) {
      return undefined;
    }
    const axis = k % 3;
    min[axis] = Math.min(min[axis], v);
    max[axis] = Math.max(max[axis], v);
  }
  if (p.length === 0) {
    return [0, 0, 0];
  }
  return [(min[0] + max[0]) * 0.5, (min[1] + max[1]) * 0.5, (min[2] + max[2]) * 0.5];
}

/** Ascending, merged union of two range lists; `"all"` wins. */
function mergeRanges(a: Range[] | "all", b: Range[] | "all"): Range[] | "all" {
  if (a === "all" || b === "all") {
    return "all";
  }
  if (a.length === 0) {
    return b;
  }
  if (b.length === 0) {
    return a;
  }
  const sorted = [...a, ...b].sort((x, y) => x[0] - y[0]);
  const out: [number, number][] = [];
  for (const [start, count] of sorted) {
    const last = out[out.length - 1];
    if (last !== undefined && start <= last[0] + last[1]) {
      last[1] = Math.max(last[1], start + count - last[0]);
    } else {
      out.push([start, count]);
    }
  }
  return out;
}

function union<T>(a: T[], b: T[]): T[] {
  return b.every((x) => a.includes(x)) ? a : [...a, ...b.filter((x) => !a.includes(x))];
}

/**
 * Drops each last-wins request that a later queued request with the same key
 * supersedes, as the engine does: never edits, adds, removals or history.
 */
function coalesce(queue: Request[]): Request[] {
  return queue.filter((req, k) => {
    if (!isLastWins(req)) {
      return true;
    }
    const key = keyOf(req);
    for (let j = k + 1; j < queue.length; j++) {
      if (keyOf(queue[j]) === key) {
        return !supersedes(queue[j], req);
      }
    }
    return true;
  });
}

function invalid(message: string): FastMeshError {
  return new FastMeshError("INVALID_ARGUMENT", message);
}

function fromWire(e: ErrorWire): FastMeshError {
  return new FastMeshError(e.code as FastMeshErrorCode, e.message);
}

/** A mesh as mirrored on the main thread. */
interface MeshState {
  id: ClientId;
  handle: MeshHandle;
  origin: Vec3;
  positions: Float32Array;
  normals: Float32Array | null;
  colors: Uint8Array | null;
  indices: Uint32Array;
  /** The placement as the engine last reported it. */
  pose: Pose;
  /**
   * The optimistic pose of a drag, shown instead of `pose` until the result
   * of revision `localUntil` (the drag's end or cancel) is processed.
   */
  local: Pose | undefined;
  localUntil: number;
  /** `removeMesh` was called: calls on it give `UNKNOWN_MESH`. */
  removed: boolean;
  /** Its removal was delivered: the mirrors are gone too. */
  gone: boolean;
  /** The colouring last requested (kept for a restart). */
  coloring: ColorRule | null;
}

interface FieldState {
  id: ClientId;
  handle: FieldHandle;
  desc: FieldDescWire;
  removed: boolean;
}

interface SectionState {
  id: ClientId;
  handle: SectionHandle;
  plane: SectionPlane;
  /** The meshes cut, as last requested. */
  meshes: MeshState[];
  fill: boolean;
  removed: boolean;
  /** The latest result per mesh client id. */
  results: Map<ClientId, SectionResult>;
}

interface Pending {
  resolve(value: unknown): void;
  reject(error: unknown): void;
}

/** An active drag. */
interface DragState {
  mesh: MeshState;
  active: boolean;
  end(): void;
}

/**
 * A scene whose engine runs in a Worker, from {@link createWorkerScene}. It
 * mirrors `Scene` (always `"eager"`) with these differences:
 *
 * - Handles are returned synchronously; the Worker catches up in order.
 * - State changes return `void` and are tracked by key through
 *   {@link WorkerScene.status}; calls that return data return promises.
 * - Requests are sent once per frame by {@link WorkerScene.commitFrame}
 *   (calls returning a promise also send on the next microtask), and
 *   "last wins" changes (colouring, section plane/meshes/fill, drag moves)
 *   superseded within the batch are dropped.
 * - `views`, `placement`, `matrix4`, `section` and `sectionPlane` read
 *   main-thread mirrors, updated as results arrive and reported by the next
 *   `commitFrame()`. Views follow the same fresh-view rule as `Scene`: get
 *   them right before use. A mesh has 0 slots until its add is processed.
 * - Drags are optimistic: `move` shows the pose at once, `end` and `cancel`
 *   are synchronous here, and the pose is held until the Worker has the drag.
 */
export class WorkerScene implements SceneLike {
  readonly #worker: WorkerLike;
  readonly #guard: ViewGuard | undefined;
  #rev = 0;
  #qid = 0;
  #nextMesh = 0;
  #nextField = 0;
  #nextSection = 0;
  readonly #meshes = new Map<ClientId, MeshState>();
  readonly #byMesh = new Map<MeshHandle, MeshState>();
  readonly #fields = new Map<ClientId, FieldState>();
  readonly #byField = new Map<FieldHandle, FieldState>();
  readonly #sections = new Map<ClientId, SectionState>();
  readonly #bySection = new Map<SectionHandle, SectionState>();
  /** This frame's requests, not sent yet. */
  #queue: Request[] = [];
  #flushScheduled = false;
  /** Sent batches awaiting their result (the engine posts one per batch, in order). */
  readonly #inflight: Request[][] = [];
  /**
   * Requests that did not apply because the engine is poisoned (from a
   * poisoned result, a Worker error, or sent after): for `restart()`.
   */
  readonly #failed: Request[] = [];
  readonly #replies = new Map<number, Pending>();
  /** Change records of results received since the last `commitFrame`, merged per mesh. */
  readonly #changes = new Map<MeshState, MeshChange>();
  /** Meshes whose shown pose changed here (drags), for the next `commitFrame`. */
  readonly #optimistic = new Set<MeshState>();
  readonly #listeners = new Set<(changes: MeshChange[]) => void>();
  readonly #statusListeners = new Set<() => void>();
  readonly #status = new Map<string, KeyStatus>();
  /** The latest revision requested per key. */
  readonly #latest = new Map<string, number>();
  #statusScheduled = false;
  #history = { canUndo: false, canRedo: false };
  #drag: DragState | undefined;
  #poison: FastMeshError | undefined;
  #terminated = false;
  #ready!: { resolve(): void; reject(e: unknown): void; settled: boolean };
  /** @internal Resolves when the Worker is ready. */
  readonly [STARTED]: Promise<void>;

  /** @internal Use {@link createWorkerScene}. */
  constructor(token: typeof TOKEN, worker: WorkerLike, debug: boolean) {
    if (token !== TOKEN) {
      throw new TypeError("use createWorkerScene() to create a WorkerScene");
    }
    this.#worker = worker;
    this.#guard = debug ? new ViewGuard() : undefined;
    this[STARTED] = new Promise<void>((resolve, reject) => {
      this.#ready = { resolve, reject, settled: false };
    });
    worker.addEventListener("message", (e: { data: WorkerMessage }) => this.#receive(e.data));
    worker.addEventListener("error", (e: { message?: string }) => {
      this.#poisonAll({
        code: "INSTANCE_POISONED",
        message: `the Worker failed${e?.message ? ` (${e.message})` : ""}; the Worker scene can no longer be used`,
      });
    });
  }

  // ---- Plumbing ----

  #alive(): void {
    if (this.#terminated) {
      throw invalid("the Worker scene is terminated");
    }
  }

  /** Queues a request; a keyed one puts its key into "computing". Returns its revision. */
  #push(req: Request): number {
    this.#queue.push(req);
    const key = keyOf(req);
    if (key !== undefined) {
      this.#latest.set(key, req.rev);
      this.#status.set(key, { status: "computing", revision: req.rev });
      this.#statusChanged();
    }
    return req.rev;
  }

  /** Queues a request with a reply and sends soon; resolves with `map(value)`. */
  #ask<T>(make: (rev: number, qid: number) => Request, map: (value: unknown) => T): Promise<T> {
    return new Promise<T>((resolve, reject) => {
      this.#alive();
      const qid = ++this.#qid;
      const req = make(++this.#rev, qid);
      this.#replies.set(qid, {
        resolve: (v) => {
          try {
            resolve(map(v));
          } catch (e) {
            reject(e);
          }
        },
        reject,
      });
      this.#push(req);
      if (!this.#flushScheduled) {
        this.#flushScheduled = true;
        queueMicrotask(() => {
          this.#flushScheduled = false;
          this.#flush();
        });
      }
    });
  }

  /** A query by name, with its reply mapped. */
  #query<N extends keyof QueryMap, T>(
    name: N,
    args: () => QueryMap[N]["args"],
    map: (value: QueryMap[N]["value"]) => T,
  ): Promise<T> {
    return this.#ask(
      (rev, qid) => ({ kind: "query", rev, qid, name, args: args() }),
      (v) => map(v as QueryMap[N]["value"]),
    );
  }

  /** Sends the queued requests as one batch, coalesced. */
  #flush(): void {
    if (this.#queue.length === 0 || this.#terminated) {
      return;
    }
    const batch = coalesce(this.#queue);
    this.#queue = [];
    if (this.#poison !== undefined) {
      this.#fail(batch, this.#poison);
      this.#statusChanged();
      return;
    }
    const transfer: Transferable[] = [];
    for (const req of batch) {
      if (req.kind === "addMesh") {
        transfer.push(req.positions, req.indices);
      }
    }
    this.#inflight.push(batch);
    this.#worker.postMessage(batch, transfer);
  }

  /** Marks requests failed with `error`: their statuses and promises. */
  #fail(batch: Request[], error: FastMeshError): void {
    for (const req of batch) {
      this.#failed.push(req);
      const key = keyOf(req);
      if (key !== undefined && this.#latest.get(key) === req.rev) {
        this.#status.set(key, { status: "error", revision: req.rev, error });
      }
      const qid = (req as { qid?: number }).qid;
      if (qid !== undefined) {
        this.#replies.get(qid)?.reject(error);
        this.#replies.delete(qid);
      }
    }
  }

  /** The engine is gone (a Worker error, or it could not start): fail everything in flight. */
  #poisonAll(error: ErrorWire): void {
    const e = fromWire(error);
    this.#poison ??= e;
    if (!this.#ready.settled) {
      this.#ready.settled = true;
      this.#ready.reject(e);
    }
    for (const batch of this.#inflight.splice(0)) {
      this.#fail(batch, e);
    }
    this.#statusChanged();
  }

  #receive(msg: WorkerMessage): void {
    if (this.#terminated) {
      return;
    }
    if ("kind" in msg) {
      if (msg.kind === "ready") {
        this.#ready.settled = true;
        this.#ready.resolve();
      } else {
        this.#poisonAll(msg.error);
      }
      return;
    }
    const batch = this.#inflight.shift() ?? [];
    this.#apply(msg, batch);
  }

  /** Applies one batch's result: mirrors, change records, statuses, replies. */
  #apply(r: Result, batch: Request[]): void {
    if (r.poisoned) {
      // Nothing in the batch applied; every status and reply is an error.
      const first = r.status[0]?.error ?? r.replies[0]?.error;
      this.#poison ??= fromWire(first ?? { code: "INSTANCE_POISONED", message: "the Worker engine stopped" });
      this.#failed.push(...batch);
      this.#applyStatus(r);
      this.#applyReplies(r);
      // Any promise of the batch still pending fails the same way.
      for (const req of batch) {
        const qid = (req as { qid?: number }).qid;
        if (qid !== undefined) {
          this.#replies.get(qid)?.reject(this.#poison);
          this.#replies.delete(qid);
        }
      }
      this.#statusChanged();
      return;
    }
    for (const c of r.changes) {
      this.#applyChange(c);
    }
    for (const p of r.placements) {
      const st = this.#meshes.get(p.id);
      if (st !== undefined) {
        st.pose = { rotation: p.rotation, translation: p.translation as Vec3, matrix4: Float64Array.from(p.matrix4) };
      }
    }
    for (const s of r.sections) {
      const st = this.#sections.get(s.section);
      if (st === undefined) {
        continue;
      }
      if (s.result === null) {
        st.results.delete(s.mesh);
      } else {
        st.results.set(s.mesh, s.result);
      }
    }
    // Drags whose end or cancel is processed: the engine's pose shows again.
    for (const st of this.#meshes.values()) {
      if (st.local !== undefined && r.rev >= st.localUntil) {
        const shown = st.local;
        st.local = undefined;
        if (!samePose(shown, st.pose)) {
          this.#optimistic.add(st);
        }
      }
    }
    if (r.history.canUndo !== this.#history.canUndo || r.history.canRedo !== this.#history.canRedo) {
      this.#history = { ...r.history };
      this.#statusChanged();
    }
    this.#applyStatus(r);
    this.#applyReplies(r);
  }

  #applyStatus(r: Result): void {
    for (const s of r.status) {
      const latest = this.#latest.get(s.key) ?? 0;
      if (s.rev < latest) {
        // A newer request for the key is still pending.
        continue;
      }
      this.#status.set(
        s.key,
        s.status === "ready"
          ? { status: "ready", revision: s.rev }
          : { status: "error", revision: s.rev, error: fromWire(s.error as ErrorWire) },
      );
      this.#statusChanged();
    }
  }

  #applyReplies(r: Result): void {
    for (const reply of r.replies) {
      const p = this.#replies.get(reply.qid);
      if (p === undefined) {
        continue;
      }
      this.#replies.delete(reply.qid);
      if (reply.ok) {
        p.resolve(reply.value);
      } else {
        p.reject(fromWire(reply.error as ErrorWire));
      }
    }
  }

  /** Applies a change record's bytes to the mesh's mirrors and merges the record for delivery. */
  #applyChange(c: MeshChangeWire): void {
    const st = this.#meshes.get(c.id);
    if (st === undefined) {
      return;
    }
    if (c.removed) {
      st.positions = new Float32Array(0);
      st.indices = new Uint32Array(0);
      st.normals = null;
      st.colors = null;
      this.#guard?.bump();
    } else {
      for (const name of BUFFERS) {
        this.#applyBuffer(st, name, c);
      }
    }
    const record: MeshChange = {
      mesh: st.handle,
      added: c.added,
      removed: c.removed,
      placementChanged: c.placementChanged,
      capacityChanged: c.capacityChanged,
      positions: c.positions,
      normals: c.normals,
      colors: c.colors,
      indices: c.indices,
      fieldsChanged: c.fields.flatMap((id) => this.#fields.get(id)?.handle ?? []),
      sectionsChanged: c.sections.flatMap((id) => this.#sections.get(id)?.handle ?? []),
    };
    const prev = this.#changes.get(st);
    this.#changes.set(st, prev === undefined ? record : merge(prev, record));
  }

  #applyBuffer(st: MeshState, name: BufferName, c: MeshChangeWire): void {
    const bytes = c.bytes[name];
    const ranges = c[name];
    if (bytes === null) {
      if ((name === "normals" || name === "colors") && st[name] !== null) {
        st[name] = null;
        this.#guard?.bump();
      }
      return;
    }
    const src = MAKE[name](bytes);
    const length = STRIDE[name] * (name === "indices" ? c.triangleSlots : c.vertexSlots);
    let dst = st[name] as Float32Array | Uint8Array | Uint32Array | null;
    if (ranges === "all" && src.length === length) {
      if (dst !== null && dst.length === length) {
        dst.set(src as ArrayLike<number>);
      } else {
        // Adopt the transferred buffer: a new mirror.
        (st as unknown as Record<BufferName, typeof src>)[name] = src;
        this.#guard?.bump();
      }
      return;
    }
    if (dst === null || dst.length !== length) {
      const grown = MAKE[name](length);
      if (dst !== null) {
        grown.set(dst.subarray(0, Math.min(dst.length, length)) as ArrayLike<number>);
      }
      dst = grown;
      (st as unknown as Record<BufferName, typeof grown>)[name] = grown;
      this.#guard?.bump();
    }
    if (ranges === "all") {
      dst.set(src.subarray(0, Math.min(src.length, length)) as ArrayLike<number>);
      return;
    }
    const stride = STRIDE[name];
    let at = 0;
    for (const [start, count] of ranges) {
      const n = count * stride;
      dst.set(src.subarray(at, at + n) as ArrayLike<number>, start * stride);
      at += n;
    }
  }

  #statusChanged(): void {
    if (this.#statusScheduled) {
      return;
    }
    this.#statusScheduled = true;
    queueMicrotask(() => {
      this.#statusScheduled = false;
      let failure: { error: unknown } | undefined;
      for (const listener of [...this.#statusListeners]) {
        try {
          listener();
        } catch (error) {
          failure ??= { error };
        }
      }
      if (failure !== undefined) {
        throw failure.error;
      }
    });
  }

  // ---- Ids ----

  /** The state of a mesh handle of this scene (removed ones included). */
  #state(mesh: MeshHandle): MeshState {
    const st = mesh instanceof MeshHandle ? this.#byMesh.get(mesh) : undefined;
    if (st === undefined) {
      throw invalid("not a mesh handle of this Worker scene");
    }
    return st;
  }

  /** A mesh that may still be changed: not removed. */
  #live(mesh: MeshHandle): MeshState {
    this.#alive();
    const st = this.#state(mesh);
    if (st.removed) {
      throw new FastMeshError("UNKNOWN_MESH", "the mesh was removed");
    }
    return st;
  }

  /** A mesh whose mirrors may be read: its removal not delivered yet. */
  #mirrored(mesh: MeshHandle): MeshState {
    const st = this.#state(mesh);
    if (st.gone) {
      throw new FastMeshError("UNKNOWN_MESH", "the mesh was removed");
    }
    return st;
  }

  #field(field: FieldHandle): FieldState {
    this.#alive();
    const st = field instanceof FieldHandle ? this.#byField.get(field) : undefined;
    if (st === undefined) {
      throw invalid("not a field handle of this Worker scene");
    }
    if (st.removed) {
      throw new FastMeshError("UNKNOWN_FIELD", "the field was removed");
    }
    return st;
  }

  #section(section: SectionHandle): SectionState {
    const st = section instanceof SectionHandle ? this.#bySection.get(section) : undefined;
    if (st === undefined) {
      throw invalid("not a section handle of this Worker scene");
    }
    if (st.removed) {
      throw new FastMeshError("UNKNOWN_SECTION", "the section was removed");
    }
    return st;
  }

  #meshList(meshes: readonly MeshHandle[]): MeshState[] {
    if (!Array.isArray(meshes)) {
      throw invalid("meshes must be an array of mesh handles");
    }
    return meshes.map((m) => this.#live(m));
  }

  /** A shown pose: the drag's, or the engine's. */
  #shown(st: MeshState): Pose {
    return st.local ?? st.pose;
  }

  // ---- Meshes ----

  /**
   * Adds a mesh from world positions and returns its handle at once. The
   * origin defaults to the bounding-box centre, computed here. The mesh has
   * no slots until the Worker has added it; a failed add shows as an error
   * on `status("mesh:<id>")`, and later calls on it fail with `UNKNOWN_MESH`.
   */
  addMesh(
    positions: Positions64,
    indices: Indices,
    options: Pick<AddMeshOptions, "origin" | "normals"> = {},
  ): MeshHandle {
    this.#alive();
    // Copies: the buffers are transferred.
    const p = positions instanceof Float32Array ? positions.slice() : Float64Array.from(positions);
    const i = Uint32Array.from(indices);
    const origin: Vec3 | undefined =
      options.origin === undefined ? boxCentre(p) : [options.origin[0], options.origin[1], options.origin[2]];
    const id = ++this.#nextMesh;
    const handle = new MeshHandle();
    const known = origin ?? [0, 0, 0];
    const st: MeshState = {
      id,
      handle,
      origin: known,
      positions: new Float32Array(0),
      normals: null,
      colors: null,
      indices: new Uint32Array(0),
      pose: poseOf(IDENTITY, [0, 0, 0], known),
      local: undefined,
      localUntil: 0,
      removed: false,
      gone: false,
      coloring: null,
    };
    this.#meshes.set(id, st);
    this.#byMesh.set(handle, st);
    this.#push({
      kind: "addMesh",
      rev: ++this.#rev,
      id,
      positions: p.buffer as ArrayBuffer,
      f64: !(p instanceof Float32Array),
      indices: i.buffer as ArrayBuffer,
      options: { origin, normals: options.normals ?? true },
    });
    return handle;
  }

  /**
   * Removes a mesh, ending an active drag of it first. Later calls on it give
   * `UNKNOWN_MESH`; its mirrors stay readable until the removal is delivered
   * by `commitFrame()`.
   */
  removeMesh(mesh: MeshHandle): void {
    const st = this.#live(mesh);
    if (this.#drag?.mesh === st) {
      this.#drag.end();
    }
    st.removed = true;
    for (const f of this.#fields.values()) {
      if (f.desc.on === st.id || f.desc.to === st.id) {
        f.removed = true;
      }
    }
    for (const s of this.#sections.values()) {
      s.meshes = s.meshes.filter((m) => m !== st);
    }
    this.#push({ kind: "removeMesh", rev: ++this.#rev, id: st.id });
  }

  /** Sets a mesh's placement (one undo step); `placement()` follows when the result arrives. */
  setPlacement(mesh: MeshHandle, placement: PlacementInput): void {
    const st = this.#live(mesh);
    const rotation = rotationMatrix(placement.rotation);
    this.#push({
      kind: "setPlacement",
      rev: ++this.#rev,
      id: st.id,
      rotation,
      translation: Array.from(placement.translation),
      key: `placement:${st.id}`,
    });
  }

  /** The mesh's placement as mirrored (a drag's pose while it is held). */
  placement(mesh: MeshHandle): Placement {
    const p = this.#shown(this.#mirrored(mesh));
    return { rotation: [...p.rotation], translation: [...p.translation] };
  }

  /** The column-major world matrix of the mesh's local render coordinates, as mirrored. */
  matrix4(mesh: MeshHandle): Float64Array {
    return this.#shown(this.#mirrored(mesh)).matrix4.slice();
  }

  /**
   * The mirrors of the mesh's render buffers, at full slot length. Each
   * getter returns the current mirror; a mirror is replaced when the slot
   * count changes or a buffer turns on or off, so get views right before use.
   */
  views(mesh: MeshHandle): MeshViews {
    const st = this.#mirrored(mesh);
    const guard = this.#guard;
    const wrap = <T extends ArrayBufferView>(v: T): T => (guard === undefined ? v : guard.wrap(v));
    return {
      positions: () => wrap(st.positions),
      normals: () => (st.normals === null ? null : wrap(st.normals)),
      colors: () => (st.colors === null ? null : wrap(st.colors)),
      indices: () => wrap(st.indices),
    };
  }

  /**
   * Starts an optimistic drag. `move` shows the pose here at once and sends
   * at most one move per batch; `end` and `cancel` are synchronous here.
   * `cancel` puts the start pose back and records nothing.
   */
  drag(mesh: MeshHandle): DragSession {
    const st = this.#live(mesh);
    if (this.#drag?.active) {
      throw invalid("a drag is already active; end or cancel it first");
    }
    const start = this.#shown(st);
    const key = `placement:${st.id}`;
    this.#push({ kind: "drag", rev: ++this.#rev, id: st.id, phase: "begin", key });
    // Hold the pose from now on: older placement results do not snap it back.
    st.local = start;
    st.localUntil = Number.POSITIVE_INFINITY;
    const finish = (phase: "end" | "cancel"): boolean => {
      if (!d.active) {
        return false;
      }
      d.active = false;
      this.#drag = undefined;
      if (this.#terminated) {
        return true;
      }
      st.localUntil = this.#push({ kind: "drag", rev: ++this.#rev, id: st.id, phase, key });
      return true;
    };
    const d: DragState = {
      mesh: st,
      active: true,
      end: () => {
        finish("end");
      },
    };
    this.#drag = d;
    return {
      get active() {
        return d.active;
      },
      move: (placement: PlacementInput) => {
        if (!d.active) {
          throw invalid("the drag is over");
        }
        this.#alive();
        const rotation = rotationMatrix(placement.rotation);
        const t = placement.translation;
        if (t?.length !== 3) {
          throw invalid("translation must be 3 numbers");
        }
        const translation: Vec3 = [t[0], t[1], t[2]];
        st.local = poseOf(rotation, translation, st.origin);
        this.#optimistic.add(st);
        this.#push({ kind: "drag", rev: ++this.#rev, id: st.id, phase: "move", rotation, translation, key });
      },
      end: d.end,
      cancel: () => {
        if (finish("cancel")) {
          st.local = start;
          this.#optimistic.add(st);
        }
      },
    };
  }

  /** Edits a mesh's geometry (one undo step); resolves with the ids added. */
  edit(mesh: MeshHandle, ops: readonly EditOp[]): Promise<EditResult> {
    return this.#ask(
      (rev, qid) => ({ kind: "edit", rev, id: this.#live(mesh).id, payload: ops, qid }),
      (v) => v as EditResult,
    );
  }

  /** Compacts a mesh's slots (its slot counts change). */
  compact(mesh: MeshHandle): void {
    const st = this.#live(mesh);
    this.#push({ kind: "compact", rev: ++this.#rev, id: st.id });
  }

  /** A report of the mesh's defects; see `Scene.analyze`. */
  analyze(mesh: MeshHandle, options?: AnalyzeOptions): Promise<Analysis> {
    return this.#query(
      "analyze",
      () => ({ mesh: this.#live(mesh).id, options }),
      (v) => v as Analysis,
    );
  }

  /** Runs one repair operation, or a list as one transaction; see `Scene.repair`. */
  repair<const O extends RepairOp>(mesh: MeshHandle, op: O): Promise<RepairReportOf<O>>;
  repair<const O extends readonly RepairOp[]>(
    mesh: MeshHandle,
    ops: O,
  ): Promise<{ -readonly [K in keyof O]: RepairReportOf<O[K]> }>;
  repair(mesh: MeshHandle, op: RepairOp | readonly RepairOp[]): Promise<RepairReport | RepairReport[]> {
    return this.#ask(
      (rev, qid) => ({ kind: "repair", rev, id: this.#live(mesh).id, payload: op, qid }),
      (v) => v as RepairReport | RepairReport[],
    );
  }

  // ---- History ----

  /** Undoes the latest step; an active drag is ended first. */
  undo(): void {
    this.#alive();
    this.#drag?.end();
    this.#push({ kind: "undo", rev: ++this.#rev });
  }

  /** Redoes the latest undone step; an active drag is ended first. */
  redo(): void {
    this.#alive();
    this.#drag?.end();
    this.#push({ kind: "redo", rev: ++this.#rev });
  }

  /** Whether there is a step to undo, as of the latest result. */
  get canUndo(): boolean {
    return this.#history.canUndo;
  }

  /** Whether there is a step to redo, as of the latest result. */
  get canRedo(): boolean {
    return this.#history.canRedo;
  }

  // ---- Fields and colours ----

  /** Adds a signed distance field; see `Scene.addDistanceField`. */
  addDistanceField(desc: DistanceFieldDesc): FieldHandle {
    this.#alive();
    const desc2: FieldDescWire = {
      name: desc.name,
      on: this.#live(desc.on).id,
      to: this.#live(desc.to).id,
      maxDistance: desc.maxDistance,
      sign: desc.sign,
    };
    const id = ++this.#nextField;
    const handle = new FieldHandle();
    const st: FieldState = { id, handle, desc: desc2, removed: false };
    this.#fields.set(id, st);
    this.#byField.set(handle, st);
    this.#push({ kind: "field", rev: ++this.#rev, field: id, op: "add", desc: desc2 });
    return handle;
  }

  /** Removes a field, and a colouring by it. */
  removeField(field: FieldHandle): void {
    const st = this.#field(field);
    st.removed = true;
    for (const m of this.#meshes.values()) {
      if (m.coloring?.field === field) {
        m.coloring = null;
      }
    }
    this.#push({ kind: "field", rev: ++this.#rev, field: st.id, op: "remove" });
  }

  /**
   * Colours a mesh from one of its fields, or turns colouring off with
   * `null`. Last wins: of several calls for a mesh in one frame, only the
   * last is sent. Tracked by `status("display:<id>")`.
   */
  setColoring(mesh: MeshHandle, rule: ColorRule | null): void {
    const st = this.#live(mesh);
    const wire: ColorRuleWire | null = rule === null ? null : { ...rule, field: this.#field(rule.field).id };
    st.coloring = rule;
    this.#push({ kind: "coloring", rev: ++this.#rev, id: st.id, rule: wire, key: `display:${st.id}` });
  }

  /** A copy of the field's values as of its last update, one per vertex slot. */
  fetchFieldValues(field: FieldHandle): Promise<Float32Array> {
    return this.#ask(
      (rev, qid) => ({ kind: "fetchFieldValues", rev, qid, field: this.#field(field).id }),
      (v) => v as Float32Array,
    );
  }

  /** Whether the field is current, and what its sign data found; see `Scene.fieldStatus`. */
  fieldStatus(field: FieldHandle): Promise<FieldStatus> {
    return this.#query(
      "fieldStatus",
      () => ({ field: this.#field(field).id }),
      (v) => v as FieldStatus,
    );
  }

  // ---- Sections ----

  /** Adds a plane section; its results arrive with the next results. */
  addSection(desc: SectionDesc): SectionHandle {
    this.#alive();
    const plane = checkPlane(desc.plane);
    const meshes = this.#meshList(desc.meshes);
    const id = ++this.#nextSection;
    const handle = new SectionHandle();
    const st: SectionState = {
      id,
      handle,
      plane: normalised(plane),
      meshes,
      fill: desc.fill ?? false,
      removed: false,
      results: new Map(),
    };
    this.#sections.set(id, st);
    this.#bySection.set(handle, st);
    const wire: SectionDescWire = { plane, meshes: meshes.map((m) => m.id), fill: st.fill };
    this.#push({ kind: "section", rev: ++this.#rev, section: id, op: "add", desc: wire });
    return handle;
  }

  #setSection(section: SectionHandle, desc: SectionDescWire, apply: (st: SectionState) => void): void {
    this.#alive();
    const st = this.#section(section);
    apply(st);
    this.#push({ kind: "section", rev: ++this.#rev, section: st.id, op: "set", desc });
  }

  /** Moves a section's plane; `sectionPlane()` returns it at once. Last wins per frame. */
  setSectionPlane(section: SectionHandle, plane: SectionPlane): void {
    const p = checkPlane(plane);
    this.#setSection(section, { plane: p }, (st) => {
      st.plane = normalised(p);
    });
  }

  /** Changes the meshes a section cuts. Last wins per frame. */
  setSectionMeshes(section: SectionHandle, meshes: readonly MeshHandle[]): void {
    const list = this.#meshList(meshes);
    this.#setSection(section, { meshes: list.map((m) => m.id) }, (st) => {
      st.meshes = list;
    });
  }

  /** Turns a section's fill on or off. Last wins per frame. */
  setSectionFill(section: SectionHandle, fill: boolean): void {
    this.#setSection(section, { fill }, (st) => {
      st.fill = fill;
    });
  }

  /** Removes a section. Its handle then gives `UNKNOWN_SECTION`. */
  removeSection(section: SectionHandle): void {
    this.#alive();
    const st = this.#section(section);
    st.removed = true;
    this.#push({ kind: "section", rev: ++this.#rev, section: st.id, op: "remove" });
  }

  /**
   * The section's latest result for a mesh as mirrored (empty and `"stale"`
   * until the first arrives). Do not modify it.
   *
   * @throws `INVALID_ARGUMENT` when the section does not cut `mesh`.
   */
  section(section: SectionHandle, mesh: MeshHandle): SectionResult {
    const s = this.#section(section);
    const m = this.#mirrored(mesh);
    const result = s.results.get(m.id);
    if (result !== undefined) {
      return result;
    }
    if (!s.meshes.includes(m)) {
      throw invalid("the section does not cut this mesh");
    }
    return { status: "stale", contours: [], length: 0, area: 0 };
  }

  /** The section's plane as last set here, the normal normalised. */
  sectionPlane(section: SectionHandle): SectionPlane {
    const p = this.#section(section).plane;
    return { origin: [...p.origin] as Vec3, normal: [...p.normal] as Vec3 };
  }

  /** The closest pair between two meshes' contours in a section; see `Scene.sectionGap`. */
  sectionGap(section: SectionHandle, a: MeshHandle, b: MeshHandle): Promise<SectionGap | null> {
    return this.#query(
      "sectionGap",
      () => ({ section: this.#section(section).id, a: this.#live(a).id, b: this.#live(b).id }),
      (v) => v,
    );
  }

  // ---- Queries ----

  /** The nearest hit of a world ray over every mesh, or `null`; see `Scene.raycast`. */
  raycast(origin: Readonly<Vec3>, direction: Readonly<Vec3>, options?: RaycastOptions): Promise<RayHit | null> {
    return this.#query(
      "raycast",
      () => ({ origin: [...origin] as Vec3, direction: [...direction] as Vec3, options }),
      (v) => (v === null ? null : { ...v, mesh: this.#handleOf(v.mesh) }),
    );
  }

  /** The surface point nearest to a world point, or `null`; see `Scene.closestPoint`. */
  closestPoint(point: Readonly<Vec3>, options: ClosestPointOptions = {}): Promise<ClosestHit | null> {
    return this.#query(
      "closestPoint",
      () => ({
        point: [...point] as Vec3,
        options: {
          maxDistance: options.maxDistance,
          mesh: options.mesh === undefined ? undefined : this.#live(options.mesh).id,
        },
      }),
      (v) => (v === null ? null : { ...v, mesh: this.#handleOf(v.mesh) }),
    );
  }

  /** The distance between two meshes' surfaces, or `null`; see `Scene.meshDistance`. */
  meshDistance(a: MeshHandle, b: MeshHandle, options?: { maxDistance?: number }): Promise<MeshDistance | null> {
    return this.#query(
      "meshDistance",
      () => ({ a: this.#live(a).id, b: this.#live(b).id, options }),
      (v) => v,
    );
  }

  /** Whether two meshes intersect or touch as placed. */
  intersects(a: MeshHandle, b: MeshHandle): Promise<boolean> {
    return this.#query(
      "intersects",
      () => ({ a: this.#live(a).id, b: this.#live(b).id }),
      (v) => v,
    );
  }

  /** Every mesh pair (or every pair of `pairs`) within `margin`; see `Scene.findContacts`. */
  findContacts(options: FindContactsOptions = {}): Promise<Contact[]> {
    return this.#query(
      "findContacts",
      () => ({
        margin: options.margin,
        pairs: options.pairs?.map(([x, y]): [ClientId, ClientId] => [this.#live(x).id, this.#live(y).id]),
      }),
      (v) => v.map((c) => ({ ...c, a: this.#handleOf(c.a), b: this.#handleOf(c.b) })),
    );
  }

  #handleOf(id: ClientId): MeshHandle {
    const st = this.#meshes.get(id);
    if (st === undefined) {
      throw new FastMeshError("UNKNOWN_MESH", `no mesh with client id ${id}`);
    }
    return st.handle;
  }

  // ---- Frames and status ----

  /** Registers a listener for {@link WorkerScene.commitFrame}; returns the function that removes it. */
  onChanges(listener: (changes: MeshChange[]) => void): () => void {
    this.#alive();
    this.#listeners.add(listener);
    return () => {
      this.#listeners.delete(listener);
    };
  }

  /**
   * Call once per animation frame. Sends this frame's requests to the Worker
   * as one batch, then delivers to every listener, and returns, the change
   * records of the results received since the last call (already applied to
   * the mirrors) plus the drags' pose changes made here. Every listener runs
   * even if one throws; the first error is rethrown afterwards.
   */
  commitFrame(): MeshChange[] {
    this.#alive();
    this.#flush();
    for (const st of this.#optimistic) {
      const record: MeshChange = {
        mesh: st.handle,
        added: false,
        removed: false,
        placementChanged: true,
        capacityChanged: false,
        positions: [],
        normals: [],
        colors: [],
        indices: [],
        fieldsChanged: [],
        sectionsChanged: [],
      };
      const prev = this.#changes.get(st);
      if (prev === undefined) {
        this.#changes.set(st, record);
      } else if (!prev.removed) {
        prev.placementChanged = true;
      }
    }
    this.#optimistic.clear();
    const changes = [...this.#changes.values()];
    this.#changes.clear();
    for (const c of changes) {
      if (c.removed) {
        const st = this.#byMesh.get(c.mesh);
        if (st !== undefined) {
          st.gone = true;
        }
      }
    }
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

  /**
   * The state of a request key: `mesh:<id>`, `placement:<id>`, `edit:<id>`,
   * `display:<id>`, `field:<id>`, `section:<id>` (add and remove),
   * `section:<id>:plane` / `:meshes` / `:fill`, `history`. A result sets
   * `"ready"` or `"error"` only when no newer request for the key is pending.
   */
  status(key: string): KeyStatus {
    return this.#status.get(key) ?? { status: "idle", revision: 0 };
  }

  /**
   * Registers a listener called (on a microtask) when any status or the
   * undo/redo state changes; returns the function that removes it.
   */
  onStatus(listener: () => void): () => void {
    this.#statusListeners.add(listener);
    return () => {
      this.#statusListeners.delete(listener);
    };
  }

  /** The client id behind a handle, as used in status keys. */
  idOf(handle: MeshHandle | FieldHandle | SectionHandle): number {
    if (handle instanceof MeshHandle) {
      return this.#state(handle).id;
    }
    const st = handle instanceof FieldHandle ? this.#byField.get(handle) : this.#bySection.get(handle);
    if (st === undefined) {
      throw invalid("not a handle of this Worker scene");
    }
    return st.id;
  }

  /**
   * Restarts the Worker and replays the scene from the mirrors. Not
   * available yet: rejects with `INVALID_ARGUMENT`.
   */
  restart(): Promise<void> {
    return Promise.reject(invalid("restart() is not implemented yet"));
  }

  /**
   * Stops the Worker. Pending promises reject with `INVALID_ARGUMENT`; later
   * calls throw it. Calling it again does nothing.
   */
  terminate(): void {
    if (this.#terminated) {
      return;
    }
    this.#terminated = true;
    this.#drag = undefined;
    this.#worker.terminate();
    const error = invalid("the Worker scene is terminated");
    if (!this.#ready.settled) {
      this.#ready.settled = true;
      this.#ready.reject(error);
    }
    for (const p of this.#replies.values()) {
      p.reject(error);
    }
    this.#replies.clear();
    this.#queue = [];
    this.#inflight.length = 0;
  }

  /** @internal Test-only: copies of the Worker engine's buffers of a mesh (an engine run with `debug`). */
  debugDump(mesh: MeshHandle): Promise<QueryMap["debugDump"]["value"]> {
    return this.#query(
      "debugDump",
      () => ({ mesh: this.#state(mesh).id }),
      (v) => v,
    );
  }
}

/** Merges a later record into an earlier one of the same mesh. */
function merge(a: MeshChange, b: MeshChange): MeshChange {
  if (b.removed) {
    return { ...b, added: false };
  }
  if (b.added) {
    return { ...b, fieldsChanged: union(a.fieldsChanged, b.fieldsChanged), sectionsChanged: union(a.sectionsChanged, b.sectionsChanged) };
  }
  if (a.removed) {
    return b;
  }
  return {
    mesh: a.mesh,
    added: a.added,
    removed: false,
    placementChanged: a.placementChanged || b.placementChanged,
    capacityChanged: a.capacityChanged || b.capacityChanged,
    positions: mergeRanges(a.positions, b.positions),
    normals: mergeRanges(a.normals, b.normals),
    colors: mergeRanges(a.colors, b.colors),
    indices: mergeRanges(a.indices, b.indices),
    fieldsChanged: union(a.fieldsChanged, b.fieldsChanged),
    sectionsChanged: union(a.sectionsChanged, b.sectionsChanged),
  };
}

function checkPlane(plane: SectionPlane): SectionPlane {
  if (plane?.origin?.length !== 3 || plane.normal?.length !== 3) {
    throw invalid("plane needs a 3-number origin and normal");
  }
  return { origin: [...plane.origin] as Vec3, normal: [...plane.normal] as Vec3 };
}

/** The plane with its normal normalised (left as is when zero: the engine reports it). */
function normalised(plane: SectionPlane): SectionPlane {
  const n = plane.normal;
  const len = Math.hypot(n[0], n[1], n[2]);
  if (!(len > 0) || !Number.isFinite(len)) {
    return plane;
  }
  return { origin: plane.origin, normal: [n[0] / len, n[1] / len, n[2] / len] };
}
