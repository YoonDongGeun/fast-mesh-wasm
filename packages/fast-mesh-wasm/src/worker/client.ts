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
import { FastMeshError, type FastMeshErrorCode, reportListenerError } from "../core.js";
import { FieldHandle, MeshHandle, rotationMatrix, SectionHandle, targetsOf } from "../handles.js";
import type {
  AddMeshOptions,
  Analysis,
  AnalyzeOptions,
  ClosestHit,
  ClosestPointOptions,
  ColorRule,
  Contact,
  DistanceFieldDesc,
  DragSession,
  EditOp,
  EditResult,
  FieldStatus,
  FindContactsOptions,
  MeshChange,
  MeshDistance,
  RaycastOptions,
  RayHit,
  RepairOp,
  RepairReport,
  RepairReportOf,
  SectionDesc,
  SectionGap,
  SectionPlane,
  SectionResult,
} from "../scene.js";
import { copyRule } from "../rule.js";
import type { SceneLike } from "../scene-like.js";
import { type MeshViews, ViewGuard } from "../views.js";
import {
  type ClientId,
  type ColorRuleWire,
  type ErrorWire,
  type FieldDescWire,
  type InitMessage,
  isLastWins,
  keyOf,
  type MeshChangeWire,
  type PoolCrash,
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
  addEventListener(type: "message" | "error" | "messageerror", cb: (e: any) => void): void;
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
   * removed). See {@link MeshViews}. Also logs, once per scene, why the
   * engine runs single-threaded when more threads were asked for (as in
   * development builds, where `import.meta.env.DEV` is set).
   */
  debug?: boolean;
  /**
   * The threads the engine computes with: its WebAssembly runs on a pool of
   * this many Workers sharing its memory. Defaults to the logical cores
   * (`navigator.hardwareConcurrency`), at most 8; `1` runs the
   * single-threaded engine. Threads need a cross-origin isolated page
   * (COOP/COEP headers); elsewhere, or when the threaded engine fails to
   * start, the scene runs single-threaded. {@link WorkerScene.threads} tells
   * which. An integer, clamped to 1 to {@link MAX_THREADS}.
   */
  threads?: number;
}

/** The state of a request key, from {@link WorkerScene.status}. */
export interface KeyStatus {
  /**
   * `"computing"` while a request with this key is queued or running,
   * `"ready"` or `"error"` once the latest one finished, `"idle"` for a key
   * never used, or one of a removed mesh, field or section once its removal
   * has settled (its state is then forgotten).
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
  if (options.threads !== undefined && !Number.isInteger(options.threads)) {
    throw invalid(`threads must be an integer (got ${options.threads})`);
  }
  const threads = options.threads === undefined ? undefined : Math.min(Math.max(options.threads, 1), MAX_THREADS);
  // Called again by `restart()`.
  const make = (): WorkerLike =>
    options.createWorker?.() ??
    ((options.workerUrl === undefined
      ? new Worker(new URL("./worker-entry.js", import.meta.url), { type: "module" })
      : new Worker(options.workerUrl, { type: "module" })) as unknown as WorkerLike);
  const dev = (import.meta as { env?: { DEV?: boolean } }).env?.DEV === true;
  const [scene, started] = createScene(make, {
    debug: options.debug ?? false,
    init: { kind: "init", threads, debug: (options.debug ?? false) || dev },
  });
  try {
    await started;
  } catch (e) {
    scene.terminate();
    throw e;
  }
  return scene;
}

/** The most threads {@link WorkerSceneOptions.threads} asks for; more are clamped. */
export const MAX_THREADS = 64;

/**
 * How long a promise-returning call waits for a `commitFrame()` before it is
 * sent anyway, once a frame loop runs (queries never hang when frames stop).
 */
export const FALLBACK_FLUSH_MS = 100;

/**
 * Set by `WorkerScene`'s static block: makes a scene (the constructor is
 * private) and returns it with its first Worker's start.
 */
let createScene: (
  make: () => WorkerLike,
  options: { debug: boolean; init: InitMessage },
) => [WorkerScene, Promise<void>];

/** Elements per slot of each buffer. */
const STRIDE = { positions: 3, normals: 3, colors: 4, values: 1, indices: 3 } as const;
type BufferName = keyof typeof STRIDE;
const BUFFERS = Object.keys(STRIDE) as BufferName[];
const MAKE = {
  positions: (b: ArrayBuffer | number) => new Float32Array(b as ArrayBuffer),
  normals: (b: ArrayBuffer | number) => new Float32Array(b as ArrayBuffer),
  colors: (b: ArrayBuffer | number) => new Uint8Array(b as ArrayBuffer),
  values: (b: ArrayBuffer | number) => new Float32Array(b as ArrayBuffer),
  indices: (b: ArrayBuffer | number) => new Uint32Array(b as ArrayBuffer),
};

const IDENTITY = [1, 0, 0, 0, 1, 0, 0, 0, 1];

/** A Worker giving back batches after its thread pool failed (see `WorkerScene`'s handoff). */
interface Handoff {
  old: WorkerLike;
  /** Batches sent to it not given back yet. */
  remaining: number;
  /** Batches sent since, held to keep the order. */
  held: { batch: Request[]; transfer: Transferable[] }[];
}

/** The buffers a batch transfers: those of its mesh adds. */
function transferOf(batch: Request[]): Transferable[] {
  const transfer: Transferable[] = [];
  for (const req of batch) {
    if (req.kind === "addMesh") {
      transfer.push(req.positions, req.indices);
    }
  }
  return transfer;
}

/** The most failed requests kept for `restart()`. */
const FAILED_LOG_LIMIT = 10_000;

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
  /** The values behind the colours: mirrored over the colours' ranges. */
  values: Float32Array | null;
  indices: Uint32Array;
  /** The placement as the engine last reported it. */
  pose: Pose;
  /**
   * The optimistic pose of a drag, shown instead of `pose` until the result
   * of revision `localUntil` (the drag's end or cancel) is processed.
   */
  local: Pose | undefined;
  localUntil: number;
  /**
   * The latest `setPlacement` not yet processed (until the result of
   * revision `requestedUntil`): a drag starts from it.
   */
  requested: Pose | undefined;
  requestedUntil: number;
  /** `removeMesh` was called: calls on it give `UNKNOWN_MESH`. */
  removed: boolean;
  /** Its removal was delivered: the mirrors are gone too. */
  gone: boolean;
  /** The colouring last requested (kept for a restart). */
  coloring: ColorRule | null;
  /**
   * The colouring the mirrors show: the requested one once the batch that
   * applied it is in (`coloring(mesh)` returns it).
   */
  shown: ColorRule | null;
  /** Colouring requests not applied yet, by revision. */
  pendingRules: Map<number, ColorRule | null>;
  /** The engine has the mesh, as of the latest result (its mirrors are its data). */
  live: boolean;
  /** Normals were asked for at the add. */
  normalsOn: boolean;
  /**
   * The add request until a result reports the mesh added; a restart can
   * replay it as long as it was never posted (`sent`: its buffers then went
   * to the Worker).
   */
  add: AddRequest | undefined;
  sent: boolean;
}

type AddRequest = Extract<Request, { kind: "addMesh" }>;

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

/** How a restart treats a request that never applied (see `sortOf`). */
interface Sorted {
  kind: "add" | "removal" | "declarative" | "lost";
  key: string | undefined;
  rev: number;
  /** For a declarative request: the keys its outcome depends on. */
  covers: string[];
}

function sortOf(req: Request): Sorted {
  const key = keyOf(req);
  const base = { key, rev: req.rev, covers: [] as string[] };
  if (req.kind === "addMesh") {
    return { ...base, kind: "add" };
  }
  if (req.kind === "removeMesh" || ((req.kind === "field" || req.kind === "section") && req.op === "remove")) {
    return { ...base, kind: "removal" };
  }
  if (
    req.kind === "setPlacement" ||
    req.kind === "drag" ||
    req.kind === "coloring" ||
    req.kind === "field" ||
    req.kind === "section"
  ) {
    return { ...base, kind: "declarative", covers: coversOf(req) };
  }
  return { ...base, kind: "lost" };
}

/** Keys a restart settles with its replay's result, each with the keys it depends on. */
interface Settle {
  rev: number;
  keys: Map<string, string[]>;
}

/**
 * The keys a declarative request's outcome depends on after a restart: the
 * mesh or section the replay re-creates for it.
 */
function coversOf(req: Request): string[] {
  switch (req.kind) {
    case "setPlacement":
    case "drag":
    case "coloring":
      return [`mesh:${req.id}`];
    case "section":
      return [`section:${req.section}`];
    case "field":
      return req.desc === undefined ? [] : [`mesh:${req.desc.on}`, ...req.desc.to.map((m) => `mesh:${m}`)];
    default:
      return [];
  }
}

/** A stand-in for a retired Worker: nothing sent to it goes anywhere. */
const INERT: WorkerLike = { postMessage() {}, addEventListener() {}, terminate() {} };

/** Crash channels made so far (part of each name, which must be unique). */
let crashChannels = 0;

/** A Worker's start: settled by its `ready` or `fatal` message, or an error. */
interface Ready {
  promise: Promise<void>;
  resolve(): void;
  reject(e: unknown): void;
  settled: boolean;
}

/** An active drag. */
interface DragState {
  mesh: MeshState;
  active: boolean;
  end(): void;
}

/** Set by `WorkerScene`'s static block: the test-only {@link debugDump}. */
let debugSizesOf: (scene: WorkerScene) => Record<string, number>;
let debugDumpOf: (scene: WorkerScene, mesh: MeshHandle) => Promise<QueryMap["debugDump"]["value"]>;
/** Set by `WorkerScene`'s static block: the internal {@link addFieldKeyed}. */
let addFieldKeyedOf: (scene: WorkerScene, desc: DistanceFieldDesc, key: string) => FieldHandle;
/** Set by `WorkerScene`'s static block: the internal {@link watchStatusKeys}. */
let watchStatusKeysOf: (scene: WorkerScene, sink: (key: string) => void) => () => void;

/**
 * A scene whose engine runs in a Worker, from {@link createWorkerScene}. It
 * mirrors `Scene` (always `"eager"`) with these differences:
 *
 * - Handles are returned synchronously; the Worker catches up in order.
 * - State changes return `void` and are tracked by key through
 *   {@link WorkerScene.status}; calls that return data return promises.
 * - Requests are sent once per frame by {@link WorkerScene.commitFrame},
 *   in call order, and "last wins" changes (colouring, section
 *   plane/meshes/fill, drag moves) superseded within the batch are dropped.
 *   Calls returning a promise wait for the next `commitFrame()` too, or are
 *   sent on their own after 100 ms when no frame comes;
 *   before the first `commitFrame()` (no frame loop) they are sent on the
 *   next microtask.
 * - `views`, `placement`, `matrix4`, `section` and `sectionPlane` read
 *   main-thread mirrors, updated as results arrive and reported by the next
 *   `commitFrame()`. Views follow the same fresh-view rule as `Scene`: get
 *   them right before use. A mesh has 0 slots until its add is processed.
 * - Drags are optimistic: `move` shows the pose at once, `end` and `cancel`
 *   are synchronous here, and the pose is held until the Worker has the drag.
 * - {@link WorkerScene.restart} replaces the Worker and replays the scene
 *   into it (the way back from `INSTANCE_POISONED`); handles keep working.
 *   Unlike other promise calls, its replay is sent at once, not with the
 *   next `commitFrame()`.
 */
export class WorkerScene implements SceneLike {
  /** Creates a Worker: the first, and one per `restart()`. */
  readonly #make: () => WorkerLike;
  /** The current Worker; only its messages are heard. */
  #worker: WorkerLike;
  readonly #guard: ViewGuard | undefined;
  #rev = 0;
  #qid = 0;
  #nextMesh = 0;
  #nextField = 0;
  #nextSection = 0;
  readonly #meshes = new Map<ClientId, MeshState>();
  readonly #byMesh = new Map<MeshHandle, MeshState>();
  /** Handles of removed meshes whose state was pruned: they give `UNKNOWN_MESH`. */
  readonly #tombstones = new WeakSet<MeshHandle>();
  /** Handles of removed fields and sections whose state was pruned. */
  readonly #fieldTombstones = new WeakSet<FieldHandle>();
  readonly #sectionTombstones = new WeakSet<SectionHandle>();
  /**
   * Removed fields and sections to forget once the result of their removal
   * (revision `rev`) is applied, or at a restart.
   */
  #prunes: { rev: number; run: () => void }[] = [];
  readonly #fields = new Map<ClientId, FieldState>();
  readonly #byField = new Map<FieldHandle, FieldState>();
  readonly #sections = new Map<ClientId, SectionState>();
  readonly #bySection = new Map<SectionHandle, SectionState>();
  /** This frame's requests, not sent yet. */
  #queue: Request[] = [];
  #flushScheduled = false;
  /** `commitFrame()` has been called: promise calls wait for the next frame. */
  #frameLoop = false;
  /** Sends queued promise calls when no frame comes. */
  #fallback: ReturnType<typeof setTimeout> | undefined;
  /** Sent batches awaiting their result (the engine posts one per batch, in order). */
  readonly #inflight: Request[][] = [];
  /**
   * Requests that did not apply because the engine is poisoned (from a
   * poisoned result, a Worker error, or failed here without being sent),
   * until `restart()`, which clears it. They have surfaced already (statuses
   * and rejected promises). Only never-sent `addMesh` requests are replayable
   * as they are: a sent one had its buffers transferred (they are detached
   * here). `restart()` finds those through each mesh's `add`, and rebuilds
   * acknowledged meshes from the mirrors instead. Capped at
   * {@link FAILED_LOG_LIMIT} entries, oldest dropped (`#failedDropped`
   * counts them).
   */
  readonly #failed: Request[] = [];
  #failedDropped = 0;
  /**
   * The keys of removals and declarative requests dropped from `#failed`
   * (latest per key), so a restart still settles them.
   */
  readonly #failedKeys = new Map<string, Omit<Sorted, "key">>();
  /** Keys a failed replay left in error, for the next restart to settle. */
  readonly #unsettled = new Map<string, Omit<Sorted, "key">>();
  /** The poses a failed replay had, for its retry. */
  #retryPoses: Map<MeshState, Pose> | undefined;
  readonly #replies = new Map<number, Pending>();
  /** Change records of results received since the last `commitFrame`, merged per mesh. */
  readonly #changes = new Map<MeshState, MeshChange>();
  /** Meshes whose shown pose changed here (drags), for the next `commitFrame`. */
  readonly #optimistic = new Set<MeshState>();
  readonly #listeners = new Set<(changes: MeshChange[]) => void>();
  readonly #statusListeners = new Set<() => void>();
  readonly #status = new Map<string, KeyStatus>();
  /** Called with each key whose status is set or forgotten, at once (see {@link watchStatusKeys}). */
  readonly #statusSinks = new Set<(key: string) => void>();
  /** The latest revision requested per key. */
  readonly #latest = new Map<string, number>();
  #statusScheduled = false;
  #history = { canUndo: false, canRedo: false };
  /** The values mirror is on (a `values()` read asked for it). */
  #values = false;
  #drag: DragState | undefined;
  #poison: FastMeshError | undefined;
  #terminated = false;
  /** The current Worker's start. */
  #ready!: Ready;
  /** Batches whose result settles a promise: a restart's replay. */
  readonly #batchDone = new Map<Request[], Pending>();
  /** The restart under way, until its replay's result is applied. */
  #restarting: Promise<void> | undefined;
  /**
   * The message starting each Worker's engine. Its `threads` becomes 1 once a
   * Worker's thread pool failed to start: later Workers (restarts) never try
   * again.
   */
  #init: InitMessage;
  /** The current Worker's thread count, from its ready message (1 until then). */
  #threads = 1;
  /**
   * A Worker whose thread pool failed to start, while it still gives back
   * batches sent to it (`remaining` more): batches sent meanwhile are held
   * (`held`, cloned with their buffers) and go to the current Worker after
   * those, keeping the order.
   */
  #handoff: Handoff | undefined;
  /** Whether the start-up warning was logged (once per scene). */
  #warned = false;
  /** The current Worker's crash channel (see `#watch`), while it may run threaded. */
  #crashWatch: BroadcastChannel | undefined;
  /** Use {@link createWorkerScene}. */
  private constructor(make: () => WorkerLike, options: { debug: boolean; init: InitMessage }) {
    this.#make = make;
    this.#guard = options.debug ? new ViewGuard() : undefined;
    this.#init = options.init;
    this.#worker = this.#attach(make());
  }

  /**
   * Listens to a new Worker and sends it the init message. Only the current
   * Worker is heard, and a handed-off one (its batches given back; its
   * errors poison the scene): a replaced one's late messages and errors are
   * ignored. `keepReady` keeps the start
   * under way (a Worker replacing one whose thread pool failed).
   */
  #attach(worker: WorkerLike, keepReady = false): WorkerLike {
    this.#threads = 1;
    if (!keepReady) {
      let ready!: Pick<Ready, "resolve" | "reject">;
      const promise = new Promise<void>((resolve, reject) => {
        ready = { resolve, reject };
      });
      // A restart's Worker failing to start is reported through its batches.
      promise.catch(() => {});
      this.#ready = { ...ready, promise, settled: false };
    }
    worker.addEventListener("message", (e: { data: WorkerMessage }) => {
      const msg = e.data;
      if ("kind" in msg && msg.kind === "fallback") {
        this.#onFallback(worker, msg);
      } else if (worker === this.#worker) {
        this.#receive(msg);
      }
    });
    worker.addEventListener("error", (e: { message?: string }) => {
      // A handed-off Worker too: the batches held for it would wait forever.
      if (worker !== this.#worker && worker !== this.#handoff?.old) {
        return;
      }
      this.#detach();
      this.#poisonAll({
        code: "INSTANCE_POISONED",
        message: `the Worker failed${e?.message ? ` (${e.message})` : ""}; the Worker scene can no longer be used`,
      });
    });
    // A result that cannot be deserialised is lost, and with it the order of
    // results: fatal, like a Worker error.
    worker.addEventListener("messageerror", () => {
      // A handed-off Worker too: the batches held for it would wait forever.
      if (worker !== this.#worker && worker !== this.#handoff?.old) {
        return;
      }
      this.#detach();
      this.#poisonAll({
        code: "INSTANCE_POISONED",
        message: "a message from the Worker could not be read; the Worker scene can no longer be used",
      });
    });
    // The init message goes first: the engine starts on it, and batches wait.
    this.#unwatch();
    try {
      worker.postMessage(this.#watch(worker));
    } catch (e) {
      // It never starts; a restart's replay fails to post the same way.
      const cause = e instanceof Error ? `${e.name}: ${e.message}` : String(e);
      this.#ready.settled = true;
      this.#ready.reject(new FastMeshError("INSTANCE_POISONED", `the Worker could not be started (${cause})`));
    }
    return worker;
  }

  /**
   * The init message for `worker`, naming a crash channel when its engine
   * may run threaded (more than one thread asked for, on a cross-origin
   * isolated page). A panic or trap on a pool thread kills only that nested
   * Worker; the engine Worker stays blocked waiting for it and never
   * answers. So the pool's threads report the crash on this channel, and it
   * poisons the scene as a Worker error does. Nothing is listened to on
   * other pages (single-threaded engines report their own traps).
   */
  #watch(worker: WorkerLike): InitMessage {
    const isolated = (globalThis as { crossOriginIsolated?: boolean }).crossOriginIsolated === true;
    if (this.#init.threads === 1 || !isolated || typeof BroadcastChannel !== "function") {
      return this.#init;
    }
    const name = `fast-mesh-wasm:pool-crash:${Date.now().toString(36)}:${Math.random().toString(36).slice(2)}:${++crashChannels}`;
    const channel = new BroadcastChannel(name);
    channel.onmessage = (e: MessageEvent) => {
      const msg = e.data as PoolCrash | null;
      if (msg?.kind !== "poolCrash" || worker !== this.#worker) {
        return;
      }
      this.#detach();
      this.#poisonAll({
        code: "INSTANCE_POISONED",
        message: `a thread of the Worker's pool crashed (${String(msg.detail)}); the Worker scene can no longer be used`,
      });
    };
    this.#crashWatch = channel;
    return { ...this.#init, crashChannel: name };
  }

  /** Stops listening to the current Worker's crash channel. */
  #unwatch(): void {
    this.#crashWatch?.close();
    this.#crashWatch = undefined;
  }

  /**
   * Stops hearing the current Worker and terminates it: after an error its
   * late results (for batches already failed here) must not apply.
   */
  #detach(): void {
    const old = this.#worker;
    this.#worker = INERT;
    old.terminate();
    this.#unwatch();
    this.#dropHandoff();
  }

  /**
   * A Worker's thread pool failed to start: it leaves a threaded instance
   * and nested Workers behind, so it is terminated and replaced by a
   * single-threaded Worker (this scene's later ones too), which gets the
   * batches it gives back, in order. The scene's start (or restart) goes on
   * with the new Worker.
   */
  #onFallback(worker: WorkerLike, msg: Extract<WorkerMessage, { kind: "fallback" }>): void {
    if (this.#terminated) {
      return;
    }
    if (worker === this.#worker) {
      if (msg.warning !== undefined && !this.#warned) {
        this.#warned = true;
        console.warn(msg.warning);
      }
      this.#init = { ...this.#init, threads: 1 };
      let next: WorkerLike;
      try {
        next = this.#make();
      } catch (e) {
        const cause = e instanceof Error ? `${e.name}: ${e.message}` : String(e);
        this.#detach();
        this.#poisonAll({ code: "INSTANCE_POISONED", message: `a new Worker could not be created (${cause})` });
        return;
      }
      // Every batch in flight went to the old Worker (none is answered before
      // a start): those not given back yet follow.
      this.#handoff = { old: worker, remaining: this.#inflight.length, held: [] };
      this.#worker = this.#attach(next, true);
    } else if (worker !== this.#handoff?.old) {
      return;
    }
    const handoff = this.#handoff as Handoff;
    handoff.remaining -= msg.batches.length;
    for (const batch of msg.batches) {
      this.#post(batch, transferOf(batch));
    }
    if (handoff.remaining <= 0 && this.#handoff === handoff) {
      this.#handoff = undefined;
      handoff.old.terminate();
      for (const { batch, transfer } of handoff.held) {
        this.#post(batch, transfer);
      }
    }
  }

  /** Posts a batch already counted in flight; one that cannot go poisons the scene. */
  #post(batch: Request[], transfer: Transferable[]): void {
    try {
      this.#worker.postMessage(batch, transfer);
    } catch (e) {
      const cause = e instanceof Error ? `${e.name}: ${e.message}` : String(e);
      this.#detach();
      this.#poisonAll({
        code: "INSTANCE_POISONED",
        message: `requests could not be passed to the single-threaded Worker (${cause})`,
      });
    }
  }

  /** Ends a handoff (restart, error, termination): the old Worker goes. */
  #dropHandoff(): void {
    this.#handoff?.old.terminate();
    this.#handoff = undefined;
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
      this.#setStatus(key, { status: "computing", revision: req.rev });
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
      this.#scheduleFlush();
    });
  }

  /**
   * Makes sure a queued promise call gets sent: with a frame loop, by the
   * next `commitFrame()` or else the fallback timer, so pointer-rate queries
   * never send the frame's moves early; without one, on a microtask.
   */
  #scheduleFlush(): void {
    if (this.#frameLoop) {
      this.#fallback ??= setTimeout(() => {
        this.#fallback = undefined;
        this.#flush();
      }, FALLBACK_FLUSH_MS);
      return;
    }
    if (!this.#flushScheduled) {
      this.#flushScheduled = true;
      queueMicrotask(() => {
        this.#flushScheduled = false;
        this.#flush();
      });
    }
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

  /**
   * Sends the queued requests as one batch, coalesced; returns it when it
   * went to the Worker. A batch that cannot be posted (`postMessage` threw,
   * say a `DataCloneError`) never reached the Worker: it fails here with
   * `INVALID_ARGUMENT`, and nothing else changes.
   */
  #flush(): Request[] | undefined {
    if (this.#queue.length === 0 || this.#terminated) {
      return undefined;
    }
    const batch = coalesce(this.#queue);
    this.#queue = [];
    if (this.#poison !== undefined) {
      this.#fail(batch, this.#poison);
      this.#statusChanged();
      return undefined;
    }
    const transfer: Transferable[] = [];
    for (const req of batch) {
      if (req.kind === "addMesh") {
        transfer.push(req.positions, req.indices);
      }
    }
    try {
      if (this.#handoff !== undefined) {
        // Held until the old Worker gave back everything; cloned now (moving
        // the buffers), so what cannot be posted still fails here.
        const held = structuredClone(batch, { transfer });
        this.#handoff.held.push({ batch: held, transfer: transferOf(held) });
      } else {
        this.#worker.postMessage(batch, transfer);
      }
    } catch (e) {
      const cause = e instanceof Error ? `${e.name}: ${e.message}` : String(e);
      this.#fail(batch, invalid(`the requests could not be sent to the Worker (${cause})`), false);
      this.#statusChanged();
      return undefined;
    }
    for (const req of batch) {
      if (req.kind === "addMesh") {
        const st = this.#meshes.get(req.id);
        if (st?.add === req) {
          st.sent = true;
        }
      }
    }
    this.#inflight.push(batch);
    return batch;
  }

  /** Marks requests failed with `error`: their statuses and promises (logged for `restart()` when `log`). */
  #fail(batch: Request[], error: FastMeshError, log = true): void {
    for (const req of batch) {
      if (log) {
        this.#logFailed(req);
      }
      const key = keyOf(req);
      if (key !== undefined && this.#latest.get(key) === req.rev) {
        this.#setStatus(key, { status: "error", revision: req.rev, error });
      }
      const qid = (req as { qid?: number }).qid;
      if (qid !== undefined) {
        this.#replies.get(qid)?.reject(error);
        this.#replies.delete(qid);
      }
    }
  }

  #logFailed(req: Request): void {
    this.#failed.push(req);
    if (this.#failed.length > FAILED_LOG_LIMIT) {
      // Drop the oldest half at once: amortised O(1) per request.
      const drop = this.#failed.length - FAILED_LOG_LIMIT / 2;
      for (const req of this.#failed.splice(0, drop)) {
        const { key, ...rest } = sortOf(req);
        if (key !== undefined && (rest.kind === "removal" || rest.kind === "declarative")) {
          this.#failedKeys.set(key, rest);
        }
      }
      this.#failedDropped += drop;
    }
  }

  /**
   * Poisoned: drops the drag hold and pending placements, so the shown poses
   * are the engine's last known ones (a restart begins clean).
   */
  #releaseHolds(): void {
    if (this.#drag !== undefined) {
      this.#drag.active = false;
      this.#drag = undefined;
    }
    for (const st of this.#meshes.values()) {
      if (st.local !== undefined && !samePose(st.local, st.pose)) {
        this.#optimistic.add(st);
      }
      st.local = undefined;
      st.localUntil = 0;
      st.requested = undefined;
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
      this.#batchDone.get(batch)?.reject(e);
      this.#batchDone.delete(batch);
    }
    this.#releaseHolds();
    this.#statusChanged();
  }

  #receive(msg: WorkerMessage): void {
    if (this.#terminated) {
      return;
    }
    if ("kind" in msg) {
      if (msg.kind === "ready") {
        this.#threads = msg.threads ?? 1;
        if (this.#threads === 1) {
          // It fell back to the single-threaded engine: no pool to hear.
          this.#unwatch();
        }
        if (msg.warning !== undefined && !this.#warned) {
          this.#warned = true;
          console.warn(msg.warning);
        }
        this.#ready.settled = true;
        this.#ready.resolve();
      } else if (msg.kind === "fatal") {
        this.#poisonAll(msg.error);
      }
      return;
    }
    const batch = this.#inflight.shift() ?? [];
    this.#apply(msg, batch);
    this.#runPrunes(msg.rev);
    const done = this.#batchDone.get(batch);
    if (done !== undefined) {
      this.#batchDone.delete(batch);
      if (msg.poisoned) {
        done.reject(this.#poison);
      } else {
        done.resolve(undefined);
      }
    }
  }

  /** Applies one batch's result: mirrors, change records, statuses, replies. */
  #apply(r: Result, batch: Request[]): void {
    if (r.poisoned) {
      // Nothing in the batch applied; every status and reply is an error.
      const first = r.status[0]?.error ?? r.replies[0]?.error;
      this.#poison ??= fromWire(first ?? { code: "INSTANCE_POISONED", message: "the Worker engine stopped" });
      for (const req of batch) {
        this.#logFailed(req);
      }
      this.#releaseHolds();
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
    // The rules of the colourings this batch applied, before their records.
    for (const { id, rev } of r.colorings) {
      const st = this.#meshes.get(id);
      if (st === undefined) {
        continue;
      }
      if (st.pendingRules.has(rev)) {
        st.shown = st.pendingRules.get(rev) ?? null;
      }
      for (const k of st.pendingRules.keys()) {
        if (k <= rev) {
          st.pendingRules.delete(k);
        }
      }
    }
    for (const c of r.changes) {
      this.#applyChange(c);
    }
    // No colours: no colouring (its field removed, or the mesh).
    for (const st of this.#meshes.values()) {
      if (st.colors === null) {
        st.shown = null;
      }
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
      if (st.requested !== undefined && r.rev >= st.requestedUntil) {
        st.requested = undefined;
      }
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
      this.#setStatus(
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
      st.live = false;
      this.#clearMirrors(st);
    } else {
      if (c.added) {
        st.live = true;
        st.add = undefined;
      }
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

  #clearMirrors(st: MeshState): void {
    st.positions = new Float32Array(0);
    st.indices = new Uint32Array(0);
    st.normals = null;
    st.colors = null;
    st.values = null;
    st.shown = null;
    this.#guard?.bump();
  }

  #applyBuffer(st: MeshState, name: BufferName, c: MeshChangeWire): void {
    const bytes = c.bytes[name];
    // The values change with the colours.
    const ranges = c[name === "values" ? "colors" : name];
    if (bytes === null) {
      if ((name === "normals" || name === "colors" || name === "values") && st[name] !== null) {
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

  /** Sets a key's status and tells the sinks. */
  #setStatus(key: string, status: KeyStatus): void {
    this.#status.set(key, status);
    for (const sink of this.#statusSinks) {
      sink(key);
    }
  }

  /** Forgets a key's status and tells the sinks. */
  #deleteStatus(key: string): void {
    if (this.#status.delete(key)) {
      for (const sink of this.#statusSinks) {
        sink(key);
      }
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
        reportListenerError(failure.error);
      }
    });
  }

  // ---- Ids ----

  /** The state of a mesh handle of this scene (removed ones included). */
  #state(mesh: MeshHandle): MeshState {
    const st = mesh instanceof MeshHandle ? this.#byMesh.get(mesh) : undefined;
    if (st === undefined) {
      if (mesh instanceof MeshHandle && this.#tombstones.has(mesh)) {
        throw new FastMeshError("UNKNOWN_MESH", "the mesh was removed");
      }
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
      if (field instanceof FieldHandle && this.#fieldTombstones.has(field)) {
        throw new FastMeshError("UNKNOWN_FIELD", "the field was removed");
      }
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
      if (section instanceof SectionHandle && this.#sectionTombstones.has(section)) {
        throw new FastMeshError("UNKNOWN_SECTION", "the section was removed");
      }
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
    const p = copyArray(positions, "positions", (x) =>
      x instanceof Float32Array ? x.slice() : Float64Array.from(x as ArrayLike<number>),
    );
    const i = copyArray(indices, "indices", (x) => Uint32Array.from(x as ArrayLike<number>));
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
      values: null,
      shown: null,
      pendingRules: new Map(),
      indices: new Uint32Array(0),
      pose: poseOf(IDENTITY, [0, 0, 0], known),
      local: undefined,
      localUntil: 0,
      requested: undefined,
      requestedUntil: 0,
      removed: false,
      gone: false,
      coloring: null,
      live: false,
      normalsOn: options.normals ?? true,
      add: undefined,
      sent: false,
    };
    this.#meshes.set(id, st);
    this.#byMesh.set(handle, st);
    st.add = {
      kind: "addMesh",
      rev: ++this.#rev,
      id,
      positions: p.buffer as ArrayBuffer,
      f64: !(p instanceof Float32Array),
      indices: i.buffer as ArrayBuffer,
      options: { origin, normals: st.normalsOn },
    };
    this.#push(st.add);
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
    const rev = ++this.#rev;
    for (const f of this.#fields.values()) {
      if (f.removed) {
        continue;
      }
      if (f.desc.on === st.id || (f.desc.to.length === 1 && f.desc.to[0] === st.id)) {
        // The engine drops the field, and colourings by it, with the mesh.
        this.#fieldRemoved(f, rev);
      } else if (f.desc.to.includes(st.id)) {
        // It keeps its other targets (a new object: sent requests keep theirs).
        f.desc = { ...f.desc, to: f.desc.to.filter((m) => m !== st.id) };
      }
    }
    for (const s of this.#sections.values()) {
      if (!s.removed) {
        s.meshes = s.meshes.filter((m) => m !== st);
      }
    }
    this.#push({ kind: "removeMesh", rev, id: st.id });
  }

  /** Sets a mesh's placement (one undo step); `placement()` follows when the result arrives. */
  setPlacement(mesh: MeshHandle, placement: PlacementInput): void {
    const st = this.#live(mesh);
    const rotation = rotationMatrix(placement.rotation);
    const t = placement.translation;
    if (t?.length !== 3) {
      throw invalid("translation must be 3 numbers");
    }
    const translation: Vec3 = [t[0], t[1], t[2]];
    st.requested = poseOf(rotation, translation, st.origin);
    st.requestedUntil = this.#push({
      kind: "setPlacement",
      rev: ++this.#rev,
      id: st.id,
      rotation,
      translation,
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
      values: () => {
        this.#wantValues();
        return st.values === null ? null : wrap(st.values);
      },
      indices: () => wrap(st.indices),
    };
  }

  /**
   * Turns the values mirror on, at the first `values()` read: until then
   * results carry no value bytes, so scenes drawn with vertex colours pay
   * nothing for it. The mirror fills with the next result.
   */
  #wantValues(): void {
    if (!this.#values) {
      this.#values = true;
      this.#push({ kind: "values", rev: ++this.#rev });
    }
  }

  /**
   * The rule the mirrored colours and values are made by (a frozen copy of
   * the one given to `setColoring`, the same object until another rule is
   * applied), or `null` without colouring. It follows the
   * results: a new rule shows with the batch that applies it, a rejected one
   * never, and a removed field's colouring goes when the removal is applied.
   */
  coloring(mesh: MeshHandle): ColorRule | null {
    return this.#mirrored(mesh).shown;
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
    // From where the mesh will be: the newest of a held drag pose and a
    // queued or in-flight setPlacement, by revision; else the engine's pose.
    const start =
      st.requested !== undefined && (st.local === undefined || st.requestedUntil > st.localUntil)
        ? st.requested
        : (st.local ?? st.pose);
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
      (rev, qid) => ({ kind: "edit", rev, id: this.#live(mesh).id, payload: copyInput(ops, "ops"), qid }),
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
      (rev, qid) => ({ kind: "repair", rev, id: this.#live(mesh).id, payload: copyInput(op, "op"), qid }),
      (v) => v as RepairReport | RepairReport[],
    );
  }

  // ---- History ----

  /** Undoes the latest step; an active drag is ended first. */
  undo(): void {
    this.#alive();
    this.#drag?.end();
    this.#forgetRequested();
    this.#push({ kind: "undo", rev: ++this.#rev });
  }

  /** Redoes the latest undone step; an active drag is ended first. */
  redo(): void {
    this.#alive();
    this.#drag?.end();
    this.#forgetRequested();
    this.#push({ kind: "redo", rev: ++this.#rev });
  }

  /**
   * An undo or redo is queued: requested placements and the poses held for
   * ended drags no longer say where the meshes will be. The engine's pose is
   * shown (and a new drag starts from it) until the results arrive.
   */
  #forgetRequested(): void {
    for (const st of this.#meshes.values()) {
      st.requested = undefined;
      if (st.local !== undefined && st.localUntil !== Number.POSITIVE_INFINITY) {
        if (!samePose(st.local, st.pose)) {
          this.#optimistic.add(st);
        }
        st.local = undefined;
        st.localUntil = 0;
      }
    }
  }

  /**
   * The threads the engine computes with: more than 1 when the threaded
   * engine runs (see {@link WorkerSceneOptions.threads}), 1 when
   * single-threaded. As of the current Worker's start (1 until it is ready).
   */
  get threads(): number {
    return this.#threads;
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
    return this.#addField(desc, undefined);
  }

  /** A field add, under `key` when given (see {@link addFieldKeyed}). */
  #addField(desc: DistanceFieldDesc, key: string | undefined): FieldHandle {
    this.#alive();
    const desc2: FieldDescWire = {
      name: desc.name,
      on: this.#live(desc.on).id,
      to: targetsOf(desc.to).map((m) => this.#live(m).id),
      maxDistance: desc.maxDistance,
      sign: desc.sign,
    };
    const id = ++this.#nextField;
    const handle = new FieldHandle();
    const st: FieldState = { id, handle, desc: desc2, removed: false };
    this.#fields.set(id, st);
    this.#byField.set(handle, st);
    this.#push(
      key === undefined
        ? { kind: "field", rev: ++this.#rev, field: id, op: "add", desc: desc2 }
        : { kind: "field", rev: ++this.#rev, field: id, op: "add", desc: desc2, key },
    );
    return handle;
  }

  /** Removes a field, and a colouring by it. */
  removeField(field: FieldHandle): void {
    const st = this.#field(field);
    const rev = ++this.#rev;
    this.#fieldRemoved(st, rev);
    this.#push({ kind: "field", rev, field: st.id, op: "remove" });
  }

  /** A field is removed (by request `rev`): its colourings go, and its state once that is applied. */
  #fieldRemoved(st: FieldState, rev: number): void {
    st.removed = true;
    for (const m of this.#meshes.values()) {
      if (m.coloring?.field === st.handle) {
        m.coloring = null;
      }
    }
    this.#prunes.push({
      rev,
      run: () => {
        this.#fields.delete(st.id);
        this.#byField.delete(st.handle);
        this.#fieldTombstones.add(st.handle);
        this.#forgetKeys([`field:${st.id}`], []);
      },
    });
  }

  /**
   * Runs the prunes whose removal is applied (revision `rev` or older).
   * Their keys are forgotten: `status()` gives `"idle"` for them.
   */
  #runPrunes(rev: number): void {
    if (this.#prunes.length === 0) {
      return;
    }
    const due = this.#prunes.filter((p) => p.rev <= rev);
    if (due.length === 0) {
      return;
    }
    this.#prunes = this.#prunes.filter((p) => p.rev > rev);
    for (const p of due) {
      p.run();
    }
    this.#statusChanged();
  }

  /** Forgets the statuses of `exact` keys and of keys starting with a `prefixes` entry. */
  #forgetKeys(exact: string[], prefixes: string[]): void {
    for (const key of exact) {
      this.#deleteStatus(key);
      this.#latest.delete(key);
    }
    if (prefixes.length > 0) {
      for (const key of [...this.#latest.keys()]) {
        if (prefixes.some((p) => key.startsWith(p))) {
          this.#deleteStatus(key);
          this.#latest.delete(key);
        }
      }
    }
  }

  /**
   * Colours a mesh from one of its fields, or turns colouring off with
   * `null`. Last wins: of several calls for a mesh in one frame, only the
   * last is sent. Tracked by `status("display:<id>")`.
   */
  setColoring(mesh: MeshHandle, rule: ColorRule | null): void {
    const st = this.#live(mesh);
    // A copy: later changes to the caller's object do not leak in.
    const copy = rule === null ? null : copyRule(rule);
    const wire: ColorRuleWire | null = copy === null ? null : { ...copy, field: this.#field(copy.field).id };
    st.coloring = copy;
    const rev = ++this.#rev;
    st.pendingRules.set(rev, copy);
    this.#push({ kind: "coloring", rev, id: st.id, rule: wire, key: `display:${st.id}` });
  }

  /** A copy of the field's values as of its last update, one per vertex slot. */
  fetchFieldValues(field: FieldHandle): Promise<Float32Array> {
    return this.#ask(
      (rev, qid) => ({ kind: "fetchFieldValues", rev, qid, field: this.#field(field).id }),
      (v) => v as Float32Array,
    );
  }

  /**
   * The field's nearest-target indices as of its last update (the index in
   * its `to`, `0xffffffff` for none); see `Scene.fieldNearest`. A target
   * removed since shifts the later indices down.
   */
  fetchFieldNearest(field: FieldHandle): Promise<Uint32Array> {
    return this.#query(
      "fieldNearest",
      () => ({ field: this.#field(field).id }),
      (v) => v as Uint32Array,
    );
  }

  /** What the sign data found out about each of the field's targets; see `Scene.fieldTargetStatus`. */
  fieldTargetStatus(field: FieldHandle): Promise<FieldStatus["sign"][]> {
    return this.#query(
      "fieldTargetStatus",
      () => ({ field: this.#field(field).id }),
      (v) => v as FieldStatus["sign"][],
    );
  }

  /**
   * Whether the field is current, and what its sign data found out about the
   * targets together; see `Scene.fieldStatus`.
   */
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
    // Its results are gone at once (no result may come while poisoned).
    st.results.clear();
    const rev = ++this.#rev;
    this.#prunes.push({
      rev,
      run: () => {
        this.#sections.delete(st.id);
        this.#bySection.delete(st.handle);
        this.#sectionTombstones.add(st.handle);
        this.#forgetKeys([`section:${st.id}`], [`section:${st.id}:`]);
      },
    });
    this.#push({ kind: "section", rev, section: st.id, op: "remove" });
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
    this.#frameLoop = true;
    if (this.#fallback !== undefined) {
      clearTimeout(this.#fallback);
      this.#fallback = undefined;
    }
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
        if (st === undefined) {
          continue;
        }
        st.gone = true;
        if (st.removed) {
          // Removed for good (unlike a removal by the engine, which a redo
          // may revert): keep only a tombstone.
          this.#meshes.delete(st.id);
          this.#byMesh.delete(st.handle);
          this.#tombstones.add(st.handle);
          const id = st.id;
          this.#forgetKeys([`mesh:${id}`, `placement:${id}`, `edit:${id}`, `display:${id}`], []);
          this.#statusChanged();
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
   * A status covers only requests made under its key: a drag or edit does
   * not mark the `display:` or `section:` keys of the colours and sections
   * that catch up after it. Returns a copy.
   */
  status(key: string): KeyStatus {
    const st = this.#status.get(key);
    return st === undefined ? { status: "idle", revision: 0 } : { ...st };
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
      if (handle instanceof FieldHandle && this.#fieldTombstones.has(handle)) {
        throw new FastMeshError("UNKNOWN_FIELD", "the field was removed");
      }
      if (handle instanceof SectionHandle && this.#sectionTombstones.has(handle)) {
        throw new FastMeshError("UNKNOWN_SECTION", "the section was removed");
      }
      throw invalid("not a handle of this Worker scene");
    }
    return st.id;
  }

  /**
   * Stops the Worker, starts a new one and replays the scene into it; this
   * is how a scene recovers after `INSTANCE_POISONED`. Resolves once the
   * replay's results are applied (reported by the next `commitFrame()`).
   *
   * The replay is built here, from what the main thread has:
   * - every mesh the engine had, re-added from its mirrors: live triangles
   *   only, compacted (slot ids change; handles do not), with its origin;
   * - meshes whose add was never sent (still queued, or failed here after a
   *   poisoning), from their own data;
   * - placements: the engine's, or, when the scene was not poisoned, the
   *   latest requested one (a `setPlacement` or drag not acknowledged yet);
   * - fields, colourings and sections as last requested (removed ones are
   *   not re-created).
   *
   * History is cleared (`canUndo`/`canRedo` are false). The replay is sent
   * at once, not with the next frame. Every replayed key is `"computing"`
   * until the replay's results arrive, and so is the key of every pending
   * declarative request (placement, drag, colouring, field, section), also
   * one that failed after a poisoning (its call was rejected then): it
   * settles with the replay (`"ready"`, or `"error"` when the replay failed
   * for it), which applies the newest value; after a poisoning, placements
   * replay the engine's pose. A colouring whose field cannot be replayed
   * (its mesh was lost) is dropped, with `UNKNOWN_FIELD` on its display key.
   * Removals count as done: field and section removal keys read `"idle"`
   * at once (their state is forgotten), mesh removal keys after the next
   * `commitFrame()` delivers the removal. When the replay fails as a whole
   * (no Worker, it could not be posted, a trap), the scene is poisoned and
   * the next `restart()` replays the same intent and settles those keys.
   *
   * What cannot be replayed surfaces as errors (`INSTANCE_POISONED`): edits,
   * repairs, compaction, undo/redo and queries sent and not answered, or not
   * sent yet (their promises reject), and meshes whose add was sent but
   * never acknowledged (their buffers went to the old Worker: they stay
   * failed, and calls on them give `UNKNOWN_MESH`). An active drag ends.
   * Mirrors, section results included, stay readable throughout. Calling it
   * again before it settles returns the same promise. When no Worker can be
   * created, the promise rejects and the scene stays poisoned (a later
   * `restart()` tries again).
   */
  restart(): Promise<void> {
    if (this.#terminated) {
      return Promise.reject(invalid("the Worker scene is terminated"));
    }
    if (this.#restarting !== undefined) {
      return this.#restarting;
    }
    const settle = this.#restart();
    // The replay goes now, not with the next frame: nothing else is queued.
    let worker: WorkerLike;
    try {
      worker = this.#make();
    } catch (e) {
      // No Worker: poisoned, the replay fails here (a later restart retries).
      const cause = e instanceof Error ? `${e.name}: ${e.message}` : String(e);
      const error = { code: "INSTANCE_POISONED", message: `a new Worker could not be created (${cause})` };
      this.#poisonAll(error);
      this.#flush();
      this.#settle(settle, fromWire(error));
      return Promise.reject(fromWire(error));
    }
    this.#worker = this.#attach(worker);
    const sent = this.#flush();
    if (sent === undefined) {
      // It could not be posted (failed here, see #flush): the new Worker has
      // nothing, so the scene is poisoned until a restart works.
      const error = {
        code: "INSTANCE_POISONED",
        message: "the restart's replay could not be sent to the Worker; restart again",
      };
      this.#poisonAll(error);
      this.#settle(settle, fromWire(error));
      return Promise.reject(fromWire(error));
    }
    const done = new Promise<void>((resolve, reject) => {
      this.#batchDone.set(sent, {
        resolve: () => {
          // The intent is applied: a later restart starts from the engine.
          this.#retryPoses = undefined;
          this.#settle(settle);
          resolve();
        },
        reject: (e) => {
          this.#settle(settle, e as FastMeshError);
          reject(e);
        },
      });
    });
    this.#restarting = done;
    const clear = () => {
      if (this.#restarting === done) {
        this.#restarting = undefined;
      }
    };
    done.then(clear, clear);
    return done;
  }

  /**
   * Ends a restart for the keys it settles without a request of their own:
   * each is `"ready"`, or `"error"` with `error` (the replay failed) or with
   * the error of a key it depends on (the mesh or section it targets), unless
   * a newer request for it was made since. When the replay failed, the keys
   * are carried to the next restart, which settles them again.
   */
  #settle(settle: Settle, error?: FastMeshError): void {
    for (const [key, covers] of settle.keys) {
      if (this.#latest.get(key) !== settle.rev) {
        continue;
      }
      if (error !== undefined) {
        this.#unsettled.set(key, { kind: "declarative", rev: settle.rev, covers });
      }
      const failed =
        error ?? covers.map((c) => this.#status.get(c)).find((s) => s?.status === "error")?.error;
      this.#setStatus(
        key,
        failed === undefined
          ? { status: "ready", revision: settle.rev }
          : { status: "error", revision: settle.rev, error: failed },
      );
    }
    this.#statusChanged();
  }

  /**
   * Retires the old Worker and queues the replay (the only thing queued).
   * Returns the keys of declarative requests the replay covers without a
   * request under their own key, to settle with the replay's result.
   */
  #restart(): Settle {
    const poisoned = this.#poison !== undefined;
    const old = this.#worker;
    // Its late messages are ignored from here on (see #attach).
    this.#worker = INERT;
    old.terminate();
    this.#unwatch();
    this.#dropHandoff();
    if (this.#fallback !== undefined) {
      clearTimeout(this.#fallback);
      this.#fallback = undefined;
    }
    if (!this.#ready.settled) {
      this.#ready.settled = true;
      this.#ready.reject(invalid("the Worker was restarted"));
    }

    // Where each mesh will be: with no poisoning, the newest of a held drag
    // pose and a requested placement; else the engine's. A retry after a
    // failed replay keeps the poses that replay had. Then the holds go.
    const poses = new Map<MeshState, Pose>();
    for (const st of this.#meshes.values()) {
      const intended =
        st.requested !== undefined && (st.local === undefined || st.requestedUntil > st.localUntil)
          ? st.requested
          : (st.local ?? st.pose);
      poses.set(st, this.#retryPoses?.get(st) ?? (poisoned ? st.pose : intended));
    }
    this.#retryPoses = poses;
    if (this.#drag !== undefined) {
      this.#drag.active = false;
      this.#drag = undefined;
    }
    for (const st of this.#meshes.values()) {
      if (!samePose(this.#shown(st), poses.get(st) as Pose)) {
        this.#optimistic.add(st);
      }
      st.local = undefined;
      st.localUntil = 0;
      st.requested = undefined;
      st.requestedUntil = 0;
    }

    // What was sent or queued, or failed since a poisoning, by kind (see
    // `sortOf`):
    // - adds: replayed below when never sent, else lost;
    // - removals: done (the new engine never gets what was removed);
    // - declarative state (placements and drags, colourings, fields,
    //   sections): settles with the replay, which applies the value it
    //   replays (after a poisoning, the engine's pose for placements);
    // - everything else (edits, repairs, compaction, undo/redo, queries)
    //   is lost: it fails if it had not already.
    // Keys a failed replay left unsettled, and those of failed requests
    // dropped from the capped log, are settled again here.
    const lost = new FastMeshError("INSTANCE_POISONED", "the Worker was restarted before this request ran");
    const pending = [...this.#inflight.splice(0).flat(), ...this.#queue];
    this.#queue = [];
    const settleKeys = new Map<string, string[]>();
    const apply = (sorted: Sorted) => {
      if (sorted.key === undefined || this.#latest.get(sorted.key) !== sorted.rev) {
        return;
      }
      if (sorted.kind === "removal") {
        this.#setStatus(sorted.key, { status: "ready", revision: sorted.rev });
      } else if (sorted.kind === "declarative") {
        settleKeys.set(sorted.key, sorted.covers);
      }
    };
    for (const [key, entry] of [...this.#unsettled, ...this.#failedKeys]) {
      apply({ key, ...entry });
    }
    this.#unsettled.clear();
    this.#failedKeys.clear();
    for (const req of this.#failed) {
      apply(sortOf(req));
    }
    for (const req of pending) {
      const sorted = sortOf(req);
      if (sorted.kind === "lost") {
        this.#fail([req], lost, false);
      } else {
        apply(sorted);
      }
    }
    for (const p of this.#replies.values()) {
      p.reject(lost);
    }
    this.#replies.clear();
    this.#poison = undefined;
    this.#failed.length = 0;
    this.#failedDropped = 0;
    this.#history = { canUndo: false, canRedo: false };
    this.#statusChanged();

    // The replay, in dependency order: meshes, placements, fields,
    // colourings, sections; then the history goes.
    const first = this.#rev + 1;
    if (this.#values) {
      this.#push({ kind: "values", rev: ++this.#rev });
    }
    const replayed = new Set<ClientId>();
    for (const st of this.#meshes.values()) {
      if (st.removed) {
        if (!st.gone) {
          // Its removal never comes now: report it here (it is then pruned).
          st.live = false;
          st.add = undefined;
          this.#clearMirrors(st);
          this.#removedHere(st);
        }
        continue;
      }
      if (st.live) {
        const { positions, indices } = liveGeometry(st);
        this.#push({
          kind: "addMesh",
          rev: ++this.#rev,
          id: st.id,
          positions: positions.buffer as ArrayBuffer,
          f64: true,
          indices: indices.buffer as ArrayBuffer,
          options: { origin: st.origin, normals: st.normalsOn },
        });
      } else if (st.add !== undefined && !st.sent) {
        // Never sent: its buffers are still here.
        st.add = { ...st.add, rev: ++this.#rev };
        this.#push(st.add);
      } else {
        const key = `mesh:${st.id}`;
        if (st.add !== undefined && this.status(key).status !== "error") {
          this.#setStatus(key, { status: "error", revision: this.#latest.get(key) ?? 0, error: lost });
        }
        st.add = undefined;
        continue;
      }
      replayed.add(st.id);
    }
    for (const st of this.#meshes.values()) {
      const pose = poses.get(st) as Pose;
      if (!replayed.has(st.id) || samePose(pose, poseOf(IDENTITY, [0, 0, 0], st.origin))) {
        continue;
      }
      st.requested = pose;
      st.requestedUntil = ++this.#rev;
      if (!samePose(pose, st.pose)) {
        // Held until the replay places it, so it does not flicker back.
        st.local = pose;
        st.localUntil = st.requestedUntil;
      }
      this.#push({
        kind: "setPlacement",
        rev: st.requestedUntil,
        id: st.id,
        rotation: pose.rotation,
        translation: pose.translation,
        key: `placement:${st.id}`,
      });
    }
    const fields = new Set<ClientId>();
    for (const f of this.#fields.values()) {
      if (!f.removed && replayed.has(f.desc.on) && f.desc.to.every((m) => replayed.has(m))) {
        fields.add(f.id);
        this.#push({ kind: "field", rev: ++this.#rev, field: f.id, op: "add", desc: f.desc });
      }
    }
    // Colourings whose field cannot be replayed are dropped, as errors.
    const dropped = new Map<string, FastMeshError>();
    for (const st of this.#meshes.values()) {
      const rule = st.coloring;
      if (rule === null || !replayed.has(st.id)) {
        continue;
      }
      const f = this.#byField.get(rule.field);
      if (f !== undefined && fields.has(f.id)) {
        const rev = ++this.#rev;
        st.pendingRules.set(rev, rule);
        this.#push({ kind: "coloring", rev, id: st.id, rule: { ...rule, field: f.id }, key: `display:${st.id}` });
      } else {
        st.coloring = null;
        dropped.set(
          `display:${st.id}`,
          new FastMeshError("UNKNOWN_FIELD", "the colouring's field could not be replayed (its mesh was lost)"),
        );
      }
    }
    for (const s of this.#sections.values()) {
      if (s.removed) {
        continue;
      }
      // Results stay readable until the replay's replace them; only those of
      // meshes not replayed go.
      for (const id of [...s.results.keys()]) {
        if (!replayed.has(id)) {
          s.results.delete(id);
        }
      }
      const desc: SectionDescWire = {
        plane: s.plane,
        meshes: s.meshes.filter((m) => replayed.has(m.id)).map((m) => m.id),
        fill: s.fill,
      };
      this.#push({ kind: "section", rev: ++this.#rev, section: s.id, op: "add", desc });
    }
    const rev = this.#push({ kind: "clearHistory", rev: ++this.#rev });
    // Declarative keys the replay covers without a request of their own
    // compute until its result.
    const keys = new Map<string, string[]>();
    for (const [key, covers] of settleKeys) {
      if ((this.#latest.get(key) ?? 0) < first && !dropped.has(key)) {
        keys.set(key, covers);
        this.#latest.set(key, rev);
        this.#setStatus(key, { status: "computing", revision: rev });
      }
    }
    for (const [key, error] of dropped) {
      this.#latest.set(key, rev);
      this.#setStatus(key, { status: "error", revision: rev, error });
    }
    // Removed fields and sections are not in the new engine: forget them.
    this.#runPrunes(Number.POSITIVE_INFINITY);
    return { rev, keys };
  }

  /** Records a mesh's removal for the next `commitFrame()`, as a result would. */
  #removedHere(st: MeshState): void {
    const record: MeshChange = {
      mesh: st.handle,
      added: false,
      removed: true,
      placementChanged: false,
      capacityChanged: false,
      positions: [],
      normals: [],
      colors: [],
      indices: [],
      fieldsChanged: [],
      sectionsChanged: [],
    };
    const prev = this.#changes.get(st);
    this.#changes.set(st, prev === undefined ? record : merge(prev, record));
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
    if (this.#fallback !== undefined) {
      clearTimeout(this.#fallback);
      this.#fallback = undefined;
    }
    if (this.#drag !== undefined) {
      this.#drag.active = false;
      this.#drag = undefined;
    }
    this.#worker.terminate();
    this.#unwatch();
    this.#dropHandoff();
    const error = invalid("the Worker scene is terminated");
    if (!this.#ready.settled) {
      this.#ready.settled = true;
      this.#ready.reject(error);
    }
    for (const p of [...this.#replies.values(), ...this.#batchDone.values()]) {
      p.reject(error);
    }
    this.#replies.clear();
    this.#batchDone.clear();
    this.#queue = [];
    this.#inflight.length = 0;
  }

  static {
    debugSizesOf = (scene) => ({
      meshes: scene.#meshes.size,
      byMesh: scene.#byMesh.size,
      fields: scene.#fields.size,
      byField: scene.#byField.size,
      sections: scene.#sections.size,
      bySection: scene.#bySection.size,
      status: scene.#status.size,
      latest: scene.#latest.size,
      prunes: scene.#prunes.length,
    });
    createScene = (make, options) => {
      const scene = new WorkerScene(make, options);
      return [scene, scene.#ready.promise];
    };
    addFieldKeyedOf = (scene, desc, key) => scene.#addField(desc, key);
    watchStatusKeysOf = (scene, sink) => {
      scene.#statusSinks.add(sink);
      return () => {
        scene.#statusSinks.delete(sink);
      };
    };
    debugDumpOf = (scene, mesh) =>
      scene.#query(
        "debugDump",
        () => ({ mesh: scene.#state(mesh).id }),
        (v) => v,
      );
  }
}

/**
 * Test-only, not exported from `fast-mesh-wasm/worker`: copies of the Worker
 * engine's buffers, placement and matrix of a mesh (the engine must run with
 * `debug`, as the Node harness does).
 */
export function debugDump(scene: WorkerScene, mesh: MeshHandle): Promise<QueryMap["debugDump"]["value"]> {
  return debugDumpOf(scene, mesh);
}

/**
 * Internal, not exported from `fast-mesh-wasm/worker`: adds a distance field
 * under `key`, the key of the display it is built for (`display:<meshId>`),
 * so the engine abandons the build when the colouring it feeds is superseded
 * before the build starts (the field is then built when first needed). Its
 * status is reported under `key`. Used by the store (`fast-mesh-wasm/store`).
 */
export function addFieldKeyed(scene: WorkerScene, desc: DistanceFieldDesc, key: string): FieldHandle {
  return addFieldKeyedOf(scene, desc, key);
}

/**
 * Internal, not exported from `fast-mesh-wasm/worker`: calls `sink` at once
 * with each key whose status is set or forgotten (before the `onStatus`
 * listeners run, on a microtask), so the store re-derives only the states
 * that read those keys. Returns the function that stops it.
 */
export function watchStatusKeys(scene: WorkerScene, sink: (key: string) => void): () => void {
  return watchStatusKeysOf(scene, sink);
}

/** Test-only, not exported from `fast-mesh-wasm/worker`: the sizes of the scene's internal maps. */
export function debugSizes(scene: WorkerScene): Record<string, number> {
  return debugSizesOf(scene);
}

/**
 * A deep copy of a caller's request input (edit ops, repair ops), taken at
 * call time so later changes to it do not leak into the request.
 */
function copyInput<T>(value: T, what: string): T {
  try {
    return structuredClone(value);
  } catch (e) {
    throw invalid(`${what} could not be copied (${e instanceof Error ? e.message : String(e)})`);
  }
}

/** Copies an array-like input; anything else is `INVALID_ARGUMENT`, as `Scene` reports bad input. */
function copyArray<T>(value: unknown, what: string, copy: (x: unknown) => T): T {
  if (typeof value !== "object" || value === null || typeof (value as { length?: unknown }).length !== "number") {
    throw invalid(`${what} must be an array or typed array of numbers`);
  }
  try {
    return copy(value);
  } catch (e) {
    throw invalid(`${what} could not be read as numbers (${e instanceof Error ? e.message : String(e)})`);
  }
}

/**
 * A mesh's live geometry from its mirrors, compacted: the live triangles (a
 * removed one reads `0, 0, 0`) in slot order, over the vertices they use in
 * slot order, as world positions before placement (local plus origin, in
 * `f64`, so the engine gets the same `f32` local positions back).
 */
function liveGeometry(st: MeshState): { positions: Float64Array; indices: Uint32Array } {
  const idx = st.indices;
  const p = st.positions;
  const slots = p.length / 3;
  const index = new Int32Array(slots).fill(-1);
  let triangles = 0;
  for (let t = 0; t < idx.length; t += 3) {
    const a = idx[t];
    const b = idx[t + 1];
    const c = idx[t + 2];
    if ((a === 0 && b === 0 && c === 0) || a >= slots || b >= slots || c >= slots) {
      continue;
    }
    index[a] = index[b] = index[c] = 0;
    triangles++;
  }
  let vertices = 0;
  for (let v = 0; v < slots; v++) {
    if (index[v] === 0) {
      index[v] = vertices++;
    }
  }
  const positions = new Float64Array(3 * vertices);
  const [ox, oy, oz] = st.origin;
  for (let v = 0; v < slots; v++) {
    const k = index[v];
    if (k >= 0) {
      positions[3 * k] = p[3 * v] + ox;
      positions[3 * k + 1] = p[3 * v + 1] + oy;
      positions[3 * k + 2] = p[3 * v + 2] + oz;
    }
  }
  const indices = new Uint32Array(3 * triangles);
  let at = 0;
  for (let t = 0; t < idx.length; t += 3) {
    const a = idx[t];
    const b = idx[t + 1];
    const c = idx[t + 2];
    if ((a === 0 && b === 0 && c === 0) || a >= slots || b >= slots || c >= slots) {
      continue;
    }
    indices[at++] = index[a];
    indices[at++] = index[b];
    indices[at++] = index[c];
  }
  return { positions, indices };
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

/**
 * A copy of a valid plane; throws `INVALID_ARGUMENT` (as `Scene` and the
 * engine do) for a malformed, non-finite or zero-normal plane, so a mirrored
 * plane is never invalid.
 */
function checkPlane(plane: SectionPlane): SectionPlane {
  if (plane?.origin?.length !== 3 || plane.normal?.length !== 3) {
    throw invalid("plane needs a 3-number origin and normal");
  }
  const origin: Vec3 = [plane.origin[0], plane.origin[1], plane.origin[2]];
  const normal: Vec3 = [plane.normal[0], plane.normal[1], plane.normal[2]];
  if (![...origin, ...normal].every((v) => typeof v === "number" && Number.isFinite(v))) {
    throw invalid("plane origin and normal must be finite");
  }
  if (normal[0] === 0 && normal[1] === 0 && normal[2] === 0) {
    throw invalid("plane normal must not be zero");
  }
  return { origin, normal };
}

/**
 * A checked plane with its normal normalised. Scaled by its largest
 * component first, so a huge finite normal (whose length overflows) or a
 * tiny one (whose length underflows) still normalises.
 */
function normalised(plane: SectionPlane): SectionPlane {
  const m = Math.max(Math.abs(plane.normal[0]), Math.abs(plane.normal[1]), Math.abs(plane.normal[2]));
  const n = [plane.normal[0] / m, plane.normal[1] / m, plane.normal[2] / m];
  const len = Math.hypot(n[0], n[1], n[2]);
  return { origin: plane.origin, normal: [n[0] / len, n[1] / len, n[2] / len] };
}
