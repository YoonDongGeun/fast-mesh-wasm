// The Worker-side engine: a sync `Scene` behind a keyed request queue. It maps
// client ids to the scene's handles, runs requests in arrival order in chunks
// (a request, a step of a field build, an operation of a repair list),
// yielding to the message loop between chunks so newer batches arrive
// meanwhile: superseded "last wins" requests are dropped, and field builds
// for a superseded display are deferred, before they start or at a step
// boundary. Field updates are deferred: after a batch's requests, the fields
// they left due are computed in steps, and a recompute made stale by a queued
// request is abandoned at a step boundary (§6.1 of the drag-speed design). After each batch it posts one `Result` with the frame's change
// records, the bytes of their dirty ranges, placements, section results,
// statuses and replies. A large mesh is added in the background: its inputs
// upload into wasm memory in chunks, then its store, BVH and normals are
// built in steps outside the scene, and its render buffers copied out in
// steps, while later requests that do not depend on it run;
// it lands in the frame of the batch under way then (or in an `update`
// message when the Worker is idle). Pure: it talks only through
// `Port`, so it runs in a browser Worker or in Node `worker_threads`.
import type { Range } from "../changes.js";
import { FastMeshError } from "../core.js";
import { decodeHeightMapFrame, decodeHeightMapOptions } from "../height-map.js";
import { generateCheckedHeightfield } from "../height-map-wasm.js";
import { generateEncodedPrimitive } from "../primitives-wasm.js";
import {
  type EditOp,
  type FieldHandle,
  fieldSteps,
  geodesicSteps,
  type JoinOptions,
  type RegisterOptions,
  meshAdds,
  type MeshChange,
  type MeshHandle,
  type RepairOp,
  Scene,
  type SectionHandle,
  type SectionPlane,
} from "../scene.js";
import type { MeshViews } from "../views.js";
import type { CollisionStepReport } from "../wave6.js";
import {
  type ClientId,
  type ColorRuleWire,
  type DrainHoleReportWire,
  type ErrorWire,
  type FieldDescWire,
  type InitMessage,
  isInit,
  isLastWins,
  keyOf,
  type MeshChangeWire,
  type PermanentStepWire,
  type PlaceMeshReportWire,
  type QueryMap,
  type Request,
  type Result,
  type SplitFacesReportWire,
  supersedes,
  type WorkerMessage,
} from "./protocol.js";

import type { WorkerSceneOptions } from "./client.js";
import { PoolStartError } from "./load.js";

// The artifact selection, for the Node `worker_threads` harness (dist/worker-engine.js),
// and the probe, for the browser check page.
export { loadEngine, startEngine } from "./load.js";
export { probeSharedGrowth } from "./probe.js";

/** The engine's side of the message channel. */
export interface Port {
  post(msg: Result, transfer: Transferable[]): void;
  /**
   * Posts a result that belongs to no batch (`{ kind: "update" }` on the
   * wire): a background add landed while the engine was idle. Without it,
   * meshes are never added in the background.
   */
  postUpdate?(msg: Result, transfer: Transferable[]): void;
  /** `cb` gets each batch of requests, in the order they were sent. */
  onMessage(cb: (batch: Request[]) => void): void;
}

/** Options of {@link runEngine}. */
export interface EngineOptions {
  /**
   * Test-only: answers the `debugDump` query (`{ mesh }`) with copies of the
   * mesh's render buffers, placement and matrix, for comparing mirrors.
   */
  debug?: boolean;
  /**
   * Test-only: called after each chunk of work: every request run (`kind`
   * is the request's), every field computed (`"fieldBuild"`, `id` is the
   * field's client id), every step of a field add's build
   * (`"fieldBuildStep"`, with the id) and every such build abandoned for a
   * superseded display (`"fieldBuildAbandon"`), every background add started
   * (`"addMeshBegin"`), stepped (`"addMeshStep"`), copied out
   * (`"addMeshPack"`) and landed (`"addMesh"`), with the mesh's id, every operation of a repair
   * list run in steps (`"repairOp"`), every field step (`"fieldStep"`) and
   * every abandoned field step unit (`"fieldStepAbandon"`), every step of a
   * geodesic query (`"geodesicStep"`) and every query cancelled
   * (`"queryCancel"`).
   */
  onChunk?: (chunk: { kind: string; id?: ClientId }) => void;
  /**
   * How long the engine runs chunks before it yields to the message loop,
   * in milliseconds: {@link YIELD_MS} by default. A yield costs a message
   * round trip, so thousands of small requests run back to back, while a
   * long operation lets newer batches in (and be coalesced) after each
   * chunk past the budget. `0` yields after every chunk (tests).
   */
  yieldMs?: number;
  /**
   * How many vertices of the due fields one field step computes:
   * {@link STEP_POINTS} by default. Smaller steps notice a stale recompute
   * sooner, at the cost of more step boundaries (tests use a few, so that a
   * unit spans many steps). Rounded down and clamped to at least 1.
   */
  stepPoints?: number;
  /**
   * The starvation guard, in milliseconds: {@link STALE_GUARD_MS} by
   * default. Once this long has passed since the last field step unit
   * completed, a queued drag move no longer abandons the unit: it completes
   * and is published, and the newest pose follows. So a sustained drag whose
   * recompute outlasts the gaps between moves still updates its colours
   * about every guard plus one recompute, each update exactly one pose. At
   * the default of 0 a move never abandons a unit: every recompute that
   * starts is shown, so its time alone sets the update rate.
   * Edits, repairs, compactions, removals, placements, undo, redo and drag
   * cancels always abandon, and so do drag moves once the drag's `end` is
   * queued (the final pose then follows within one recompute; a pointer
   * that pauses without ending may wait up to two).
   */
  staleGuardMs?: number;
  /**
   * Meshes of at least this many triangles are added in the background
   * ({@link BACKGROUND_ADD_TRIANGLES} by default): see
   * {@link Engine.#beginAdd}. `0` adds every mesh so (tests);
   * `Infinity` none. NaN keeps the default; a negative value counts as 0.
   * The init message's `backgroundAddTriangles`
   * ({@link WorkerSceneOptions.backgroundAddTriangles}) overrides it in
   * {@link serveEngine}.
   */
  backgroundAddTriangles?: number;
  /**
   * The work of one step of a background add ({@link ADD_STEP_WORK} by
   * default): 64 bytes of the upload of its positions and indices into
   * wasm memory; a vertex or triangle of its store build; a triangle's
   * box, or a triangle visited per tree level, of its BVH build; 4
   * triangles or 16 vertices of its normals; 64 bytes of the copy of its
   * render buffers. Clamped to 1 to 2^32 − 1; NaN or less than 1 keeps
   * the default.
   */
  addStepWork?: number;
  /**
   * The work of one step of a geodesic query
   * ({@link GEODESIC_STEP_WORK} by default): a triangle or vertex visited
   * by a pass over the mesh, or a vertex by an iteration of a solve. A step
   * is at least one of either, so on a large mesh this only tells how many
   * small ones run together. Rounded down and clamped to 1 to 2^32 − 1; NaN
   * keeps the default.
   */
  geodesicStepWork?: number;
  /**
   * The threads the engine computes with (1 by default; {@link serveEngine}
   * passes what the start reports). With more than one, a request waiting
   * on a background add lands it in one call (the parallel one-call
   * build, several times faster than steps); with one, it keeps stepping
   * with yields, which costs the same, so newer batches still arrive.
   */
  threads?: number;
  /**
   * Test-only: never drop a landed add's copied-out buffers early (see
   * {@link Engine.#unpack}), so that only the check at packing time guards
   * them.
   */
  keepCopies?: boolean;
  /**
   * The history budget in bytes (`SceneOptions.historyBudget`). The init
   * message's overrides it in {@link serveEngine}.
   */
  historyBudget?: number;
}

/** The default {@link EngineOptions.yieldMs}. */
export const YIELD_MS = 4;

/** The default {@link EngineOptions.staleGuardMs}. */
export const STALE_GUARD_MS = 0;

/** The default {@link EngineOptions.stepPoints}. */
export const STEP_POINTS = 4096;

/**
 * The default {@link EngineOptions.backgroundAddTriangles}: smaller meshes
 * add in well under a frame, in place.
 */
export const BACKGROUND_ADD_TRIANGLES = 65_536;

/** The default {@link EngineOptions.addStepWork}: a few milliseconds of BVH build. */
export const ADD_STEP_WORK = 100_000;

/** The default {@link EngineOptions.geodesicStepWork}: a few milliseconds of a solve. */
export const GEODESIC_STEP_WORK = 100_000;

/** A step budget that computes a whole unit at once. */
const ALL_POINTS = 0xffff_ffff;

/** Bytes of a background add's render buffers copied per unit of {@link EngineOptions.addStepWork}. */
const PACK_BYTES_PER_UNIT = 64;

/** Bytes of a background add's positions and indices uploaded per unit of {@link EngineOptions.addStepWork}. */
const UPLOAD_BYTES_PER_UNIT = 64;

const now: () => number =
  typeof performance !== "undefined" && typeof performance.now === "function"
    ? () => performance.now()
    : () => Date.now();

/**
 * Runs the engine on `port` until the instance traps. The WebAssembly module
 * must already be initialised.
 */
export function runEngine(port: Port, options: EngineOptions = {}): void {
  const engine = new Engine(port, options);
  port.onMessage((batch) => engine.receive(batch));
}

/** A Worker's side of its channel: raw messages in, messages out. */
export interface Endpoint {
  post(msg: WorkerMessage, transfer: Transferable[]): void;
  /** `cb` gets each message's data, in order. */
  onMessage(cb: (data: unknown) => void): void;
}

/** What a Worker's engine start reports in its ready message. */
export interface Started {
  /** The threads the engine computes with: 1 when single-threaded. */
  threads: number;
  /** Development only: why it is single-threaded though more threads were asked for. */
  warning?: string;
  /** The browser cannot run the threaded engine (its probe failed): the scene stops asking for threads. */
  probeFailed?: true;
}

/**
 * Serves the engine on a Worker's endpoint once it starts. `ready` is the
 * start itself (the WebAssembly module's initialisation), or a function
 * starting it for the Worker's first {@link InitMessage} (as the browser
 * Worker script does); with a promise, init messages are ignored. Batches
 * arriving before are kept and run in order; then `{ kind: "ready" }` (with
 * the {@link Started} report, 1 thread by default) is posted first. When the
 * start fails, `{ kind: "fatal" }` is posted instead and nothing runs. When
 * it fails with a {@link PoolStartError} (the threaded pool did not start),
 * `{ kind: "fallback" }` is posted instead, giving back every batch received
 * (and, one message each, every batch received later), their buffers
 * transferred: the scene replaces this Worker with a single-threaded one and
 * sends them there.
 */
export function serveEngine(
  endpoint: Endpoint,
  ready: Promise<unknown> | ((init: InitMessage) => Promise<Started>),
  options: EngineOptions = {},
): void {
  let deliver: ((batch: Request[]) => void) | undefined;
  let started = typeof ready !== "function";
  /** The engine's options, with the scene's from its init message. */
  let engineOptions = options;
  const early: Request[][] = [];
  /** Set once the start failed with a `PoolStartError`: batches go back. */
  let giveBack = false;
  const back = (batches: Request[][], warning?: string) => {
    const transfer: Transferable[] = [];
    for (const batch of batches) {
      for (const req of batch) {
        if (req.kind === "addMesh") {
          transfer.push(req.positions, req.indices);
        }
      }
    }
    endpoint.post(
      warning === undefined ? { kind: "fallback", batches } : { kind: "fallback", batches, warning },
      transfer,
    );
  };
  const serve = (start: Promise<unknown>) =>
    start.then(
      (info) => {
        const { threads = 1, warning, probeFailed } = (info ?? {}) as Partial<Started>;
        endpoint.post(
          {
            kind: "ready",
            threads,
            ...(warning === undefined ? {} : { warning }),
            ...(probeFailed === true ? { probeFailed } : {}),
          },
          [],
        );
        runEngine(
          {
            post: (msg, transfer) => endpoint.post(msg, transfer),
            postUpdate: (result, transfer) => endpoint.post({ kind: "update", result }, transfer),
            onMessage: (cb) => {
              deliver = cb;
              for (const batch of early.splice(0)) {
                cb(batch);
              }
            },
          },
          { threads, ...engineOptions },
        );
      },
      (e: unknown) => {
        if (e instanceof PoolStartError) {
          giveBack = true;
          back(early.splice(0), e.warning);
          return;
        }
        const cause = e instanceof Error ? `${e.name}: ${e.message}` : String(e);
        endpoint.post(
          {
            kind: "fatal",
            error: { code: "INSTANCE_POISONED", message: `the Worker could not load fast-mesh (${cause})` },
          },
          [],
        );
      },
    );
  // Listen right away: messages sent while the module loads are not lost.
  endpoint.onMessage((data) => {
    if (isInit(data)) {
      // Only the first starts the engine (and its thread pool, once per Worker).
      if (!started && typeof ready === "function") {
        started = true;
        const { backgroundAddTriangles, historyBudget } = data;
        engineOptions = {
          ...options,
          ...(backgroundAddTriangles === undefined ? {} : { backgroundAddTriangles }),
          ...(historyBudget === undefined ? {} : { historyBudget }),
        };
        serve(new Promise<Started>((resolve) => resolve(ready(data))));
      }
    } else if (giveBack) {
      back([data as Request[]]);
    } else if (deliver === undefined) {
      early.push(data as Request[]);
    } else {
      deliver(data as Request[]);
    }
  });
  if (typeof ready !== "function") {
    serve(ready);
  }
}

function unknown(code: "UNKNOWN_MESH" | "UNKNOWN_FIELD" | "UNKNOWN_SECTION", id: ClientId): FastMeshError {
  return new FastMeshError(code, `no ${code.slice(8).toLowerCase()} with client id ${id}`);
}

function invalid(message: string): FastMeshError {
  return new FastMeshError("INVALID_ARGUMENT", message);
}

function isPoison(e: unknown): boolean {
  return (
    (e instanceof FastMeshError && e.code === "INSTANCE_POISONED") || e instanceof WebAssembly.RuntimeError
  );
}

/**
 * An error as it crosses the boundary. A `FastMeshError` keeps its code;
 * anything else (a malformed request) is reported as `INVALID_ARGUMENT`.
 */
function toWire(e: unknown): ErrorWire {
  if (e instanceof FastMeshError) {
    return { code: e.code, message: e.message };
  }
  const message = e instanceof Error ? `${e.name}: ${e.message}` : String(e);
  return { code: "INVALID_ARGUMENT", message };
}

const POISONED: ErrorWire = {
  code: "INSTANCE_POISONED",
  message: "an earlier fast-mesh call panicked; the WebAssembly instance can no longer be used",
};

/** The poisoned error after packing or sending a result failed with `e`. */
function packingFailed(e: unknown): ErrorWire {
  const cause = e instanceof Error ? `${e.name}: ${e.message}` : String(e);
  return {
    code: "INSTANCE_POISONED",
    message: `packing or sending the Worker result failed (${cause}); the Worker scene can no longer be used`,
  };
}

/** Elements per slot of each buffer. */
const STRIDE = { positions: 3, normals: 3, colors: 4, values: 1, uvs: 2, indices: 3 } as const;

type BufferName = keyof typeof STRIDE;

/** A buffer's dirty ranges: the values change with the colours. */
const rangesOf = (c: MeshChange, name: BufferName): Range[] | "all" => c[name === "values" ? "colors" : name] ?? [];

/** Copies the dirty elements of `view` into a fresh buffer: the whole view for `"all"`. */
function copyRanges(
  view: Float32Array | Uint8Array | Uint32Array,
  ranges: Range[] | "all",
  stride: number,
): ArrayBuffer {
  if (ranges === "all") {
    return view.slice().buffer as ArrayBuffer;
  }
  let total = 0;
  for (const [, count] of ranges) {
    total += count * stride;
  }
  const out = new (view.constructor as Float32ArrayConstructor)(total) as typeof view;
  let at = 0;
  for (const [start, count] of ranges) {
    out.set(view.subarray(start * stride, (start + count) * stride), at);
    at += count * stride;
  }
  return out.buffer as ArrayBuffer;
}

/** What one batch collects for its `Result`. */
class Batch {
  rev = 0;
  readonly status: Result["status"] = [];
  readonly replies: Result["replies"] = [];
  readonly colorings: Result["colorings"] = [];
  readonly transfer: Transferable[] = [];

  ready(key: string | undefined, rev: number): void {
    if (key !== undefined) {
      this.status.push({ key, rev, status: "ready" });
    }
  }

  error(key: string | undefined, rev: number, error: ErrorWire): void {
    if (key !== undefined) {
      this.status.push({ key, rev, status: "error", error });
    }
  }
}

/** Marks the end of a received batch in the queue: its result is posted there. */
const END = Symbol("end of batch");
type Item = Request | typeof END;

/** A mesh being added in the background (see {@link Engine.#beginAdd}). */
interface PendingAdd {
  id: ClientId;
  /** The build's token (`meshAdds`). */
  token: number;
  /** The add's status key and revision, held until it lands. */
  key: string | undefined;
  rev: number;
  /** Built, waiting to be copied out and land (not while a field step unit is open). */
  built: boolean;
  /** Its render buffers being copied out, once built (see {@link Engine.#pack}). */
  packed?: Packed;
  /** The copy failed (not a trap): it lands without one, the frame copies. */
  uncopied?: true;
}

/**
 * Copies of a background add's render buffers, made in steps before it
 * lands, and sent as its first record's bytes in place of copies made then
 * (see {@link Engine.#pack}).
 */
interface Packed {
  positions: ArrayBuffer;
  normals: ArrayBuffer | null;
  indices: ArrayBuffer;
  /** Bytes copied so far, over the three in that order. */
  done: number;
  /**
   * The landed mesh's content stamp (`meshAdds.stamp`) when it landed: the
   * copies are sent only while the mesh still has it.
   */
  stamp?: number;
}

/** Whether every byte of a {@link Packed} copy is copied. */
const packedAll = (p: Packed | undefined): p is Packed =>
  p !== undefined && p.done === p.positions.byteLength + (p.normals?.byteLength ?? 0) + p.indices.byteLength;

/**
 * Yields to the message loop, so batches sent meanwhile are received:
 * `setImmediate` where there is one (Node: a chain of `MessageChannel` ticks
 * after chunks of a millisecond or more lets no `parentPort` message in,
 * measured), else a `MessageChannel` tick (browsers interleave it with the
 * Worker's messages, checked in headless Chrome; nested timers are clamped
 * to 4 ms there), else `setTimeout(0)`.
 */
function messageTick(): () => Promise<void> {
  const immediate = (globalThis as { setImmediate?: (cb: () => void) => unknown }).setImmediate;
  if (typeof immediate === "function") {
    return () => new Promise((resolve) => immediate(resolve));
  }
  if (typeof MessageChannel === "undefined") {
    return () => new Promise((resolve) => setTimeout(resolve, 0));
  }
  const channel = new MessageChannel();
  const waiting: (() => void)[] = [];
  channel.port1.onmessage = () => waiting.shift()?.();
  // Node: the port alone must not keep a process alive.
  (channel.port1 as { unref?: () => void }).unref?.();
  return () =>
    new Promise((resolve) => {
      waiting.push(resolve);
      channel.port2.postMessage(null);
    });
}

class Engine {
  readonly #port: Port;
  readonly #scene = new Scene();
  /** The scene's stepped field updates (internal API). */
  readonly #steps = fieldSteps(this.#scene);
  /** The scene's stepped mesh adds (internal API). */
  readonly #meshAdds = meshAdds(this.#scene);
  /** Geodesic queries run in steps. */
  readonly #geodesics = geodesicSteps(this.#scene);
  readonly #geodesicStepWork: number;
  /** Meshes being added in the background, in arrival order. */
  readonly #adds: PendingAdd[] = [];
  /**
   * Landed background adds whose render buffers were copied out before:
   * their record in the next frame sends these copies, unless a request
   * changed the mesh since (see {@link Engine.#unpack}).
   */
  readonly #packed = new Map<MeshHandle, Packed>();
  readonly #backgroundAdd: number;
  readonly #addStepWork: number;
  /** More than one thread: a waiting request finishes adds in one call. */
  readonly #threaded: boolean;
  readonly #keepCopies: boolean;
  /** The revision of the last result posted, for `update` messages. */
  #lastRev = 0;
  /** A received batch is being run: its first request started, its result is not posted yet. */
  #batchOpen = false;
  /**
   * The open batch's end marker is consumed (its fields are being stepped
   * or its result packed): a poisoning then must not take the next
   * batch's requests for this one's.
   */
  #endConsumed = false;
  /** Requests received and not run yet, each batch followed by {@link END}. */
  readonly #queue: Item[] = [];
  /** A drain is under way (possibly waiting at a yield). */
  #running = false;
  /** What the batch being run collects for its result. */
  #batch = new Batch();
  #poisoned = false;
  /** A `values` request turned the values mirror on. */
  #mirrorValues = false;
  /** The next result carries every coloured mesh's colours and values whole. */
  #valuesFull = false;
  readonly #meshes = new Map<ClientId, MeshHandle>();
  readonly #fields = new Map<ClientId, FieldHandle>();
  /** Fields whose build was abandoned: built when a request first needs them. */
  readonly #deferred = new Map<ClientId, FieldDescWire>();
  readonly #sections = new Map<ClientId, SectionHandle>();
  /** Engine handles back to client ids, for change records and query results. */
  readonly #meshIds = new Map<MeshHandle, ClientId>();
  readonly #fieldIds = new Map<FieldHandle, ClientId>();
  readonly #sectionIds = new Map<SectionHandle, ClientId>();
  /**
   * Handles removed this batch: their reverse ids are dropped once the frame
   * (which may still report them) is packed.
   */
  readonly #removed: MeshHandle[] = [];
  readonly #removedFields: FieldHandle[] = [];
  readonly #removedSections: SectionHandle[] = [];
  /** Each field's meshes, `[on, to]`: removing either drops the field. */
  /** Per field: `[on, ...to]`. */
  readonly #fieldMeshes = new Map<ClientId, ClientId[]>();
  /** Each section's meshes, for the entries a mesh removal ends. */
  readonly #sectionMeshes = new Map<ClientId, ClientId[]>();
  /** Section entries ended by mesh removals this batch, reported as `null`. */
  readonly #goneEntries: { section: ClientId; mesh: ClientId }[] = [];
  /**
   * Meshes out of the scene that undo or redo can bring back. Their ids stay
   * bound (requests on them fail in the scene), and the fields on them stay
   * known: they come back with the mesh.
   */
  readonly #parked = new Set<ClientId>();
  /** Parked meshes the client removed for good meanwhile: they go when they come back. */
  readonly #doomed = new Set<ClientId>();
  /** Meshes parked, restored or dropped by the history in this batch (see `Result.presence`). */
  readonly #moved = new Set<ClientId>();
  /** What those moves left of fields (see `Result.fieldTargets`). */
  readonly #fieldTargets = new Map<ClientId, ClientId[] | null>();
  /** The sections those moves took meshes from (see `Result.sectionMeshes`). */
  readonly #sectionsLost = new Set<ClientId>();
  /** The steps the history could not keep, for the next result. */
  readonly #permanent: PermanentStepWire[] = [];
  /** The active drag, by the dragged mesh's client id. */
  #drag: { id: ClientId; session: ReturnType<Scene["drag"]>; collisionActive: boolean } | undefined;
  /** Meshes whose authoritative placement must be emitted in the next result even if delta is 0. */
  readonly #forcePlacements = new Set<ClientId>();
  /** Mirrors `this.#scene`'s open step depth so step/drag/inSteps invariants stay aligned. */
  #stepDepth = 0;
  readonly #debug: boolean;
  readonly #onChunk: EngineOptions["onChunk"];
  readonly #tick = messageTick();
  readonly #yieldMs: number;
  /** When the engine last yielded (or started draining). */
  #yielded = 0;
  readonly #stepPoints: number;
  /**
   * The `display:` statuses of batches whose field step unit was abandoned:
   * they go with the result of the batch whose unit completes next, so the
   * keys read "computing" until then.
   */
  readonly #held: Result["status"] = [];
  /**
   * The meshes of the last abandoned unit, until a unit completes: its
   * fields are still due, so a unit opened while a request queued makes it
   * stale would be abandoned too.
   */
  #lastAbandoned: Set<ClientId> | undefined;
  readonly #staleGuardMs: number;
  /**
   * When the last field step unit completed (the engine's start before
   * the first): drag moves abandon units only within the guard after it.
   */
  #published = now();

  constructor(port: Port, options: EngineOptions) {
    this.#port = port;
    this.#debug = options.debug ?? false;
    this.#onChunk = options.onChunk;
    this.#yieldMs = options.yieldMs ?? YIELD_MS;
    // At least one vertex a step (NaN too); at most what one call takes.
    const points = Math.floor(options.stepPoints ?? STEP_POINTS);
    this.#stepPoints = points >= 1 ? Math.min(points, ALL_POINTS) : 1;
    this.#staleGuardMs = options.staleGuardMs ?? STALE_GUARD_MS;
    // NaN keeps the default, a negative threshold counts as 0, Infinity turns it off.
    const bg = options.backgroundAddTriangles;
    this.#backgroundAdd = bg === undefined || Number.isNaN(bg) ? BACKGROUND_ADD_TRIANGLES : Math.max(0, bg);
    // At least one unit a step (NaN or below 1: the default); at most what one call takes.
    const work = Math.floor(options.addStepWork ?? ADD_STEP_WORK);
    this.#addStepWork = work >= 1 ? Math.min(work, ALL_POINTS) : ADD_STEP_WORK;
    const geodesic = Math.floor(options.geodesicStepWork ?? GEODESIC_STEP_WORK);
    this.#geodesicStepWork = Number.isNaN(geodesic) ? GEODESIC_STEP_WORK : Math.min(Math.max(geodesic, 1), ALL_POINTS);
    this.#threaded = (options.threads ?? 1) > 1;
    this.#keepCopies = options.keepCopies ?? false;
    // Field updates wait for the end of the batch, to be stepped there.
    this.#steps.setDefer(true);
    if (options.historyBudget !== undefined) {
      this.#scene.setHistoryBudget(options.historyBudget);
    }
    // Client ids now: a mesh that is gone loses its own with the next result.
    const ids = (meshes: readonly MeshHandle[]) => meshes.flatMap((h) => this.#meshIds.get(h) ?? []);
    this.#scene.onPermanent(({ lost, meshes, gone, bytes }) => {
      this.#permanent.push({ lost, meshes: ids(meshes), gone: ids(gone), bytes });
    });
  }

  /**
   * {@link Engine.#maybeYield}, first advancing the oldest background add by
   * one step when the time budget is spent, so that adds go on while the
   * engine is busy (about half the time at most). `false` once poisoned.
   */
  async #yieldWithAdds(land = true): Promise<boolean> {
    if (this.#adds.length > 0 && now() - this.#yielded >= this.#yieldMs && !(await this.#advanceAdd(land))) {
      return false;
    }
    await this.#maybeYield();
    return true;
  }

  /**
   * Yields to the message loop once the time budget since the last yield is
   * spent; batches sent meanwhile are received there, and the next chunk's
   * cancellation check sees them.
   */
  async #maybeYield(): Promise<void> {
    if (now() - this.#yielded >= this.#yieldMs) {
      await this.#tick();
      this.#yielded = now();
    }
  }

  receive(batch: Request[]): void {
    if (this.#poisoned) {
      this.#postPoisoned(batch, new Batch());
      return;
    }
    this.#queue.push(...batch, END);
    if (!this.#running) {
      void this.#drain();
    }
  }

  /**
   * Whether a later queued request (from index `from` on) with the same key
   * supersedes the last-wins request `req`.
   */
  #superseded(req: Request, key: string | undefined, from = 0): boolean {
    if (key === undefined || !isLastWins(req)) {
      return false;
    }
    for (let k = from; k < this.#queue.length; k++) {
      const later = this.#queue[k];
      if (later !== END && keyOf(later) === key) {
        return supersedes(later, req);
      }
    }
    return false;
  }

  /**
   * Cooperative cancellation of a field build: a field `add` keyed like the
   * display it is built for is abandoned when the request it feeds (the next
   * queued last-wins request with that key) is already superseded. The field
   * is then kept as deferred and built when first needed.
   */
  #abandoned(req: Request, key: string | undefined): boolean {
    if (req.kind !== "field" || req.op !== "add" || key === undefined || req.desc === undefined) {
      return false;
    }
    if (this.#fields.has(req.field) || this.#deferred.has(req.field)) {
      // Run it: it fails as "already in use".
      return false;
    }
    if (!this.#feedsSuperseded(key)) {
      return false;
    }
    this.#deferred.set(req.field, req.desc);
    this.#fieldMeshes.set(req.field, [req.desc.on, ...req.desc.to]);
    return true;
  }

  /** Whether the next queued last-wins request keyed `key` is already superseded. */
  #feedsSuperseded(key: string | undefined): boolean {
    if (key === undefined) {
      return false;
    }
    const at = this.#queue.findIndex((x) => x !== END && keyOf(x) === key && isLastWins(x));
    return at >= 0 && this.#superseded(this.#queue[at] as Request, key, at + 1);
  }

  /**
   * Runs the queue in arrival order, one chunk (a request, or one operation
   * of a repair list) at a time, yielding to the message loop between
   * chunks once the time budget is spent, so that newer batches are
   * received meanwhile: superseded "last
   * wins" requests are then dropped before they start, and abandoned field
   * builds are deferred. Each batch's result is posted when its last
   * request is done, so results stay one per batch, in order.
   */
  async #drain(): Promise<void> {
    this.#running = true;
    this.#yielded = now();
    try {
      while (this.#queue.length > 0 || this.#adds.length > 0) {
        if (this.#queue.length === 0) {
          // Idle but for background adds: one step, then let messages in.
          if (!(await this.#advanceAdd())) {
            return;
          }
          await this.#tick();
          this.#yielded = now();
          continue;
        }
        const item = this.#queue.shift() as Item;
        if (item === END) {
          this.#batchOpen = true;
          this.#endConsumed = true;
          if (!(await this.#stepFields()) || !this.#finish()) {
            return;
          }
          this.#batchOpen = false;
          this.#endConsumed = false;
        } else {
          this.#batchOpen = true;
          if (!(await this.#step(item))) {
            return;
          }
        }
        if (this.#queue.length > 0 && this.#queue[0] !== END && !(await this.#yieldWithAdds())) {
          return;
        }
      }
    } finally {
      this.#running = false;
    }
  }

  /** Runs one request; `false` once the engine is poisoned. */
  async #step(req: Request): Promise<boolean> {
    const b = this.#batch;
    b.rev = Math.max(b.rev, req.rev);
    const key = keyOf(req);
    if (this.#superseded(req, key) || this.#abandoned(req, key)) {
      return true;
    }
    if (req.kind === "query" && this.#cancelled(req.qid)) {
      this.#replyCancelled(req.qid, b);
      return true;
    }
    // A trap while finishing them poisons with this request among the failed.
    if (this.#adds.length > 0 && this.#dependsOnAdds(req) && !(await this.#finishAdds(req))) {
      return false;
    }
    this.#unpack(req);
    try {
      if (req.kind === "addMesh" && this.#inBackground(req)) {
        await this.#beginAdd(req);
        // Its status is held until it lands.
        this.#onChunk?.({ kind: "addMeshBegin", id: req.id });
        return true;
      }
      if (req.kind === "repair" && this.#inSteps(req)) {
        await this.#repairInSteps(req, b);
      } else if (req.kind === "field" && req.op === "add") {
        if (!(await this.#addFieldInSteps(req, key))) {
          // Abandoned for a superseded display: no status, as before it starts.
          return true;
        }
      } else if (req.kind === "query" && (req.name === "geodesicDistances" || req.name === "shortestSurfacePath")) {
        await this.#geodesicInSteps(req, b);
      } else {
        this.#run(req, b);
      }
      b.ready(key, req.rev);
      this.#onChunk?.({ kind: req.kind, id: (req as { id?: ClientId }).id });
    } catch (e) {
      if (isPoison(e)) {
        this.#poison(b, POISONED, req);
        return false;
      }
      const error = toWire(e);
      b.error(key, req.rev, error);
      const qid = (req as { qid?: number }).qid;
      if (qid !== undefined) {
        b.replies.push({ qid, ok: false, error });
      }
    }
    return true;
  }

  // ---- Cancelled queries ----

  /** Whether a queued request cancels the query `qid`. */
  #cancelled(qid: number): boolean {
    return this.#queue.some((item) => item !== END && item.kind === "cancel" && item.qid === qid);
  }

  #replyCancelled(qid: number, b: Batch): void {
    b.replies.push({ qid, ok: false, error: { code: "CANCELLED", message: "the query was cancelled" } });
    this.#onChunk?.({ kind: "queryCancel" });
  }

  /**
   * Runs a query of geodesic distances, or of a path (its distances, then
   * the path in one go), in steps of {@link EngineOptions.geodesicStepWork},
   * yielding between them once the time budget is spent: a queued `cancel`
   * of it then ends it. Other requests wait for it, so its mesh does not
   * change meanwhile.
   */
  async #geodesicInSteps(req: Extract<Request, { kind: "query" }>, b: Batch): Promise<void> {
    const args = req.args as QueryMap["geodesicDistances"]["args"] & Partial<QueryMap["shortestSurfacePath"]["args"]>;
    const mesh = this.#mesh(args.mesh);
    let distances = req.name === "shortestSurfacePath" ? args.options?.distances : undefined;
    if (distances === undefined) {
      const steps = this.#geodesics;
      const token = steps.begin(mesh, args.sources, args.options);
      try {
        while (!steps.step(token, this.#geodesicStepWork)) {
          this.#onChunk?.({ kind: "geodesicStep", id: args.mesh });
          // Only receives: nothing else runs before this request is done.
          await this.#maybeYield();
          if (this.#cancelled(req.qid)) {
            this.#replyCancelled(req.qid, b);
            return;
          }
        }
        distances = steps.finish(token);
      } finally {
        // (Nothing left to drop once it is finished or failed.)
        steps.drop(token);
      }
    }
    if (req.name === "geodesicDistances") {
      b.replies.push({ qid: req.qid, ok: true, value: distances });
      b.transfer.push(distances.buffer as ArrayBuffer);
    } else {
      const value = this.#scene.shortestSurfacePath(mesh, args.sources, args.targetVertex as number, {
        ...args.options,
        distances,
      });
      b.replies.push({ qid: req.qid, ok: true, value });
    }
  }

  // ---- Background adds ----

  /**
   * Whether an add runs in the background: the port can post updates, the
   * mesh is large enough, and the add is not asked to run in place (nor
   * undoable: it must land in the step it was made in).
   */
  #inBackground(req: Extract<Request, { kind: "addMesh" }>): boolean {
    return (
      this.#port.postUpdate !== undefined &&
      req.inline !== true &&
      req.options.undoable !== true &&
      req.indices.byteLength / 12 >= this.#backgroundAdd
    );
  }

  /**
   * Starts adding a mesh in the background: its inputs upload into wasm
   * memory in chunks, yielding between them, then the mesh, its BVH and
   * its normals are built outside the scene, a step at a time between
   * other chunks and while the Worker is idle, and it lands in the scene
   * when built, in arrival order with the other background adds. Until
   * then its status is held, and a request that depends on it
   * ({@link Engine.#dependsOnAdds}) first finishes every background add;
   * other requests run meanwhile.
   */
  async #beginAdd(req: Extract<Request, { kind: "addMesh" }>): Promise<void> {
    if (this.#meshes.has(req.id) || this.#adds.some((a) => a.id === req.id)) {
      throw invalid(`mesh client id ${req.id} is already in use`);
    }
    const positions = req.f64 ? new Float64Array(req.positions) : new Float32Array(req.positions);
    const indices = new Uint32Array(req.indices);
    if (positions.length % 3 !== 0) {
      throw invalid("positions length must be a multiple of 3");
    }
    if (indices.length % 3 !== 0) {
      throw invalid("indices length must be a multiple of 3");
    }
    const uvs = req.uvs === undefined ? undefined : new Float32Array(req.uvs);
    if (uvs !== undefined && uvs.length !== (2 * positions.length) / 3) {
      throw invalid("uvs need 2 values per vertex");
    }
    const token = this.#meshAdds.beginUpload(positions.length / 3, indices.length / 3, {
      ...req.options,
      mode: "eager",
    });
    try {
      const bytes = this.#addStepWork * UPLOAD_BYTES_PER_UNIT;
      const p = Math.max(1, Math.floor(bytes / positions.BYTES_PER_ELEMENT));
      const n = Math.max(1, Math.floor(bytes / indices.BYTES_PER_ELEMENT));
      for (let at = 0; at < positions.length; at += p) {
        const chunk = positions.subarray(at, at + p);
        this.#meshAdds.pushPositions(token, chunk instanceof Float64Array ? chunk : Float64Array.from(chunk));
        await this.#maybeYield();
      }
      for (let at = 0; at < indices.length; at += n) {
        this.#meshAdds.pushIndices(token, indices.subarray(at, at + n));
        await this.#maybeYield();
      }
      if (uvs !== undefined) {
        const u = Math.max(1, Math.floor(bytes / uvs.BYTES_PER_ELEMENT));
        for (let at = 0; at < uvs.length; at += u) {
          this.#meshAdds.pushUvs(token, uvs.subarray(at, at + u));
          await this.#maybeYield();
        }
      }
    } catch (e) {
      try {
        this.#meshAdds.drop(token);
      } catch {
        // Gone already.
      }
      throw e;
    }
    this.#adds.push({ id: req.id, token, key: keyOf(req), rev: req.rev, built: false });
  }

  /** Whether a mesh of `ids` is being added in the background. */
  #pending(...ids: unknown[]): boolean {
    return this.#adds.some((a) => ids.includes(a.id));
  }

  /**
   * Whether the field `id` is deferred (not built yet) and measured on or
   * to a mesh being added in the background: building it needs that mesh.
   */
  #fieldPending(id: unknown): boolean {
    const desc = this.#deferred.get(id as ClientId);
    return desc !== undefined && this.#pending(desc.on, ...desc.to);
  }

  /**
   * Whether `req` must see the background adds landed first: it names one
   * of their meshes, or a deferred field measured on or to one (building
   * it needs the mesh), adds or removes a mesh (slots are allocated in
   * arrival order), or queries every mesh (`raycast`, `closestPoint`
   * without a mesh, `findContacts` without pairs).
   */
  #dependsOnAdds(req: Request): boolean {
    switch (req.kind) {
      case "addMesh":
      case "removeMesh":
      case "booleanBatch":
      case "cutByPlane":
      case "extractSelectedFaces":
      case "splitSelectedFaces":
      case "reconstructPointCloud":
      case "addHeightfield":
      case "convexHull":
      case "convexHullPoints":
      case "wrap":
      case "punchDrainHole":
      case "placeMesh":
      case "moveWithCollision":
      case "commitCollisionInteraction":
        return true;
      case "compact":
      case "setPlacement":
      case "drag":
      case "edit":
      case "repair":
      case "trimByCurve":
      case "fillHoleContinuity":
      case "fillAllHolesContinuity":
      case "bridgePartialArcs":
      case "applyDyntopo":
      case "dyntopoDab":
      case "deleteSelectedFaces":
      case "extrudeFaces":
      case "extrudeBase":
      case "applyFfd":
      case "applyArap":
      case "applyKelvinlet":
      case "applyXpbd":
      case "alignToFittedPrimitive":
      case "hollowMesh":
      case "latticeFill":
      case "denoise":
      case "delaunayFlips":
        return this.#pending(req.id);
      case "join":
      case "boolean":
        return this.#pending(req.target, req.source);
      case "setCollisionRule":
      case "clearCollisionRule":
        return this.#pending(req.a, req.b);
      case "setDefaultCollisionRule":
        return false;
      case "register":
      case "alignByLandmarks":
      case "registerMultistage":
        return this.#pending(req.moving, req.fixed);
      case "coloring":
        return this.#pending(req.id) || (req.rule !== null && this.#fieldPending(req.rule?.field));
      case "fetchFieldValues":
        return this.#fieldPending(req.field);
      case "field":
        return req.desc !== undefined ? this.#pending(req.desc.on, ...req.desc.to) : this.#fieldPending(req.field);
      case "section": {
        const meshes = req.desc?.meshes;
        return Array.isArray(meshes) && this.#pending(...meshes);
      }
      case "query": {
        const a = (req.args ?? {}) as {
          a?: ClientId;
          b?: ClientId;
          mesh?: ClientId;
          source?: ClientId;
          target?: ClientId;
          field?: ClientId;
          options?: { mesh?: ClientId };
          pairs?: [ClientId, ClientId][];
        };
        switch (req.name) {
          case "raycast":
            return true;
          case "fieldStatus":
          case "fieldNearest":
          case "fieldTargetStatus":
            return this.#fieldPending(a.field);
          case "closestPoint":
            return a.options?.mesh === undefined || this.#pending(a.options.mesh);
          case "findContacts":
            return a.pairs === undefined || this.#pending(...a.pairs.flat());
          case "wrapPlan":
          case "heightMapFrame":
          case "heightMap":
            return this.#pending(...(req.args as { meshes: ClientId[] }).meshes);
          default:
            return this.#pending(a.a, a.b, a.mesh, a.source, a.target);
        }
      }
      default:
        return false;
    }
  }

  /**
   * Advances the oldest background add by one step: of its build (BVH and
   * normals), or once built, of the copy of its render buffers
   * ({@link Engine.#pack}); with `land`, lands it once copied (landing
   * changes the scene, which would drop an open field step unit: the
   * unit's loop steps without landing; copying changes nothing). A build
   * that fails is dropped, with an error status for its add. Without
   * `copy`, a built add lands at once and the frame copies its buffers.
   * `false` once poisoned.
   */
  async #advanceAdd(land = true, failed?: Request, copy = true): Promise<boolean> {
    const a = this.#adds[0];
    if (a === undefined) {
      return true;
    }
    if (a.built && (!copy || a.uncopied === true) && land) {
      // The frame copies the buffers: land as built.
      a.packed = undefined;
      return this.#land(a, failed);
    }
    if (!a.built) {
      try {
        a.built = this.#meshAdds.step(a.token, this.#addStepWork);
      } catch (e) {
        return this.#addFailed(a, e, false, failed);
      }
      this.#onChunk?.({ kind: "addMeshStep", id: a.id });
    } else if (!packedAll(a.packed) && a.uncopied !== true) {
      try {
        this.#pack(a);
      } catch (e) {
        if (isPoison(e)) {
          return this.#addFailed(a, e, false, failed);
        }
        // No memory for the copy (say): the frame copies the buffers.
        a.packed = undefined;
        a.uncopied = true;
      }
      this.#onChunk?.({ kind: "addMeshPack", id: a.id });
    }
    return (packedAll(a.packed) || a.uncopied === true) && land ? this.#land(a, failed) : true;
  }

  /**
   * Whether `req` may change the positions, normals or triangles of the
   * mesh `id`: an edit, repair, compaction, join or removal of it, or any
   * undo or redo.
   */
  #changesMesh(req: Request, id: ClientId): boolean {
    switch (req.kind) {
      case "edit":
      case "repair":
      case "compact":
      case "cutByPlane":
      case "trimByCurve":
      case "fillHoleContinuity":
      case "fillAllHolesContinuity":
      case "bridgePartialArcs":
      case "applyDyntopo":
      case "dyntopoDab":
      case "deleteSelectedFaces":
      case "extrudeFaces":
      case "extrudeBase":
      case "splitSelectedFaces":
      case "applyFfd":
      case "applyArap":
      case "applyKelvinlet":
      case "applyXpbd":
      case "hollowMesh":
      case "latticeFill":
      case "denoise":
      case "delaunayFlips":
      case "punchDrainHole":
      case "removeMesh":
        return req.id === id;
      case "boolean":
      case "placeMesh":
        return req.target === id;
      case "join":
        return req.target === id || req.source === id;
      case "moveWithCollision":
      case "commitCollisionInteraction":
      case "undo":
      case "redo":
        return true;
      default:
        return false;
    }
  }

  /**
   * Copies about {@link EngineOptions.addStepWork} × 64 more bytes of a
   * built add's render buffers (positions, normals, indices) out of wasm
   * memory, so that the frame it lands in sends them instead of copying
   * the whole mesh then. The build's buffers are the added mesh's, moved
   * into the scene unchanged, so the copies are what the frame would copy,
   * bit for bit, unless a request changes the mesh first
   * ({@link Engine.#unpack}).
   */
  #pack(a: PendingAdd): void {
    // Fresh views, read with no scene call in between.
    const views = this.#meshAdds.buffers(a.token);
    const p = (a.packed ??= {
      positions: new ArrayBuffer(views.positions.byteLength),
      normals: views.normals === null ? null : new ArrayBuffer(views.normals.byteLength),
      indices: new ArrayBuffer(views.indices.byteLength),
      done: 0,
    });
    const parts: [ArrayBufferView | null, ArrayBuffer | null][] = [
      [views.positions, p.positions],
      [views.normals, p.normals],
      [views.indices, p.indices],
    ];
    let budget = this.#addStepWork * PACK_BYTES_PER_UNIT;
    let start = 0;
    for (const [view, out] of parts) {
      if (view === null || out === null) {
        continue;
      }
      const end = start + out.byteLength;
      if (budget > 0 && p.done < end) {
        const from = p.done - start;
        const n = Math.min(out.byteLength - from, budget);
        new Uint8Array(out, from, n).set(new Uint8Array(view.buffer, view.byteOffset + from, n));
        p.done += n;
        budget -= n;
      }
      start = end;
    }
  }

  /**
   * Forgets the copied-out buffers of a landed mesh that `req` may change
   * (an edit, repair or compaction of it, an undo or redo), to free them
   * early. Correctness does not depend on it: the copies are sent only
   * while the mesh's content stamp is the one it landed with.
   */
  #unpack(req: Request): void {
    if (this.#packed.size === 0 || this.#keepCopies) {
      return;
    }
    for (const h of [...this.#packed.keys()]) {
      if (this.#changesMesh(req, this.#meshId(h))) {
        this.#packed.delete(h);
      }
    }
  }

  /**
   * Finishes every background add, in arrival order, for `failed`, the
   * request that waits on them (it is among the failed requests if they
   * poison the engine). Threaded, each lands in one call, which builds
   * what is left of its BVH at once with the parallel one-call build
   * (several times faster than steps). Single-threaded the two cost the
   * same, so it keeps stepping, yielding (to receive only) between steps,
   * so that newer batches still arrive and coalesce; it skips the copy of
   * an add's buffers when `failed` changes that mesh (the frame copies
   * them anyway).
   */
  async #finishAdds(failed?: Request): Promise<boolean> {
    while (this.#adds.length > 0) {
      if (this.#threaded) {
        if (!this.#land(this.#adds[0], failed)) {
          return false;
        }
        continue;
      }
      // No copy of an add's buffers when the waiting request changes it:
      // the frame would copy them again.
      const copy = failed === undefined || !this.#changesMesh(failed, this.#adds[0].id);
      if (!(await this.#advanceAdd(true, failed, copy))) {
        return false;
      }
      if (this.#adds.length > 0) {
        await this.#maybeYield();
      }
    }
    return true;
  }

  /**
   * Adds a built mesh to the scene. Its status goes with the batch under
   * way, or, when the Worker is idle (no batch received and not finished),
   * in an `update` message with the frame it lands in.
   */
  #land(a: PendingAdd, failed?: Request): boolean {
    this.#adds.shift();
    let h: MeshHandle;
    try {
      h = this.#meshAdds.finish(a.token);
    } catch (e) {
      return this.#addFailed(a, e, true, failed);
    }
    this.#meshes.set(a.id, h);
    this.#meshIds.set(h, a.id);
    if (packedAll(a.packed)) {
      try {
        a.packed.stamp = this.#meshAdds.stamp(h);
        this.#packed.set(h, a.packed);
      } catch (e) {
        if (isPoison(e)) {
          return this.#addFailed(a, e, true, failed);
        }
        // Not kept: the frame copies the buffers.
      }
    }
    this.#onChunk?.({ kind: "addMesh", id: a.id });
    this.#settle((b) => b.ready(a.key, a.rev));
    return true;
  }

  /**
   * Drops a failed background add; `false` (poisoned) on a trap, with
   * `failed` (the request waiting on it, if any) among the failed requests.
   */
  #addFailed(a: PendingAdd, e: unknown, taken = false, failed?: Request): boolean {
    if (!taken) {
      this.#adds.shift();
      try {
        this.#meshAdds.drop(a.token);
      } catch {
        // Gone already.
      }
    }
    if (isPoison(e)) {
      if (this.#batchOpen) {
        this.#batch.error(a.key, a.rev, POISONED);
        this.#poison(this.#batch, POISONED, failed, this.#endConsumed);
      } else {
        const b = new Batch();
        b.error(a.key, a.rev, POISONED);
        this.#poisonIdle(b, POISONED);
      }
      return false;
    }
    const error = toWire(e);
    this.#settle((b) => b.error(a.key, a.rev, error));
    return true;
  }

  /**
   * Records a background add's outcome with `put`: in the batch under way,
   * or, when the Worker is idle, in an `update` message with the frame.
   */
  #settle(put: (b: Batch) => void): void {
    if (this.#batchOpen) {
      put(this.#batch);
      return;
    }
    const b = new Batch();
    b.rev = this.#lastRev;
    put(b);
    let result: Result;
    try {
      result = this.#result(b);
    } catch (e) {
      this.#poisonIdle(b, isPoison(e) ? POISONED : packingFailed(e));
      return;
    }
    try {
      this.#port.postUpdate?.(result, b.transfer);
    } catch (e) {
      this.#poisonIdle(b, packingFailed(e));
    }
  }

  /**
   * Brings the fields the batch left due up to date in steps of
   * {@link EngineOptions.stepPoints} vertices, yielding between them once
   * the time budget is spent. At each step boundary, when a queued request
   * makes the unit stale ({@link Engine.#makesStale}), the unit is
   * abandoned: the batch's result then carries none of its values or
   * colours, and the `display:` statuses of its meshes are held back until a
   * later unit completes. No unit is opened while a request queued makes
   * the last abandoned unit stale (its fields are still due, so the new unit
   * would be stale too). Past the starvation guard
   * ({@link EngineOptions.staleGuardMs}) drag moves no longer abandon, so a
   * sustained drag still publishes. Other requests wait for the unit. A field error
   * ends the unit like a failed eager update: swallowed, the failing fields
   * stay stale. `false` once the engine is poisoned.
   */
  async #stepFields(): Promise<boolean> {
    const steps = this.#steps;
    const b = this.#batch;
    if (this.#lastAbandoned !== undefined && this.#staleQueued(this.#lastAbandoned)) {
      this.#hold(b, this.#lastAbandoned);
      return true;
    }
    let meshes: Set<ClientId> | undefined;
    for (;;) {
      let state: "idle" | "pending" | "done";
      try {
        state = steps.step(this.#stepPoints);
        if (state === "pending") {
          meshes ??= new Set(steps.meshes().map((h) => this.#meshId(h)));
        }
      } catch (e) {
        let error = e;
        if (!isPoison(error)) {
          // The unit is over (the fields that could be applied were), or
          // reading its meshes failed: leave none open.
          try {
            steps.abandon();
          } catch (e2) {
            error = e2;
          }
        }
        if (isPoison(error)) {
          b.status.push(...this.#held.splice(0));
          this.#poison(b, POISONED, undefined, true);
          return false;
        }
        state = "done";
      }
      this.#onChunk?.({ kind: "fieldStep" });
      if (state !== "pending") {
        if (state !== "idle") {
          this.#published = now();
        }
        this.#lastAbandoned = undefined;
        b.status.push(...this.#held.splice(0));
        return true;
      }
      if (!(await this.#yieldWithAdds(false))) {
        return false;
      }
      if (meshes !== undefined && this.#staleQueued(meshes)) {
        steps.abandon();
        this.#onChunk?.({ kind: "fieldStepAbandon" });
        this.#lastAbandoned = meshes;
        this.#hold(b, meshes);
        return true;
      }
    }
  }

  /**
   * Whether a queued request makes a unit over `meshes` stale. Drag moves
   * count only within the starvation guard after the last completed unit,
   * or when the drag's `end` is queued too: on release the newest pose
   * follows within one recompute, never after a kept older one. (An `end`
   * with no move queued before it leaves the unit alone: the unit is then
   * at the final pose.)
   */
  #staleQueued(meshes: Set<ClientId>): boolean {
    const moves =
      now() - this.#published <= this.#staleGuardMs ||
      this.#queue.some((item) => item !== END && item.kind === "drag" && item.phase === "end" && meshes.has(item.id));
    return this.#queue.some((item) => item !== END && this.#makesStale(item, meshes, moves));
  }

  /** Moves the batch's `display:` statuses of `meshes` to the held ones. */
  #hold(b: Batch, meshes: Set<ClientId>): void {
    const kept: Result["status"] = [];
    for (const st of b.status.splice(0)) {
      const held = st.key.startsWith("display:") && meshes.has(Number(st.key.slice("display:".length)));
      (held ? this.#held : kept).push(st);
    }
    b.status.push(...kept);
  }

  /**
   * Whether the queued `req` makes a field step unit over `meshes` (the
   * meshes its fields are on and measured to) stale (§6.1): a drag move
   * (only with `moves`: within the starvation guard), a drag cancel or a
   * placement, a registration of one of them, an edit, repair, compaction,
   * join or removal of one of them,
   * an undo or redo, or the removal of a field between two of them.
   */
  #makesStale(req: Request, meshes: Set<ClientId>, moves: boolean): boolean {
    switch (req.kind) {
      case "drag":
        // A move, an end with a last move, or a cancel (back to the start pose).
        return (
          meshes.has(req.id) &&
          (req.phase === "cancel" || (moves && (req.phase === "move" || (req.phase === "end" && req.rotation !== undefined))))
        );
      case "fieldInteraction":
        return true;
      case "setPlacement":
      case "edit":
      case "repair":
      case "compact":
      case "cutByPlane":
      case "trimByCurve":
      case "fillHoleContinuity":
      case "fillAllHolesContinuity":
      case "bridgePartialArcs":
      case "applyDyntopo":
      case "dyntopoDab":
      case "deleteSelectedFaces":
      case "extrudeFaces":
      case "extrudeBase":
      case "splitSelectedFaces":
      case "applyFfd":
      case "applyArap":
      case "applyKelvinlet":
      case "applyXpbd":
      case "alignToFittedPrimitive":
      case "hollowMesh":
      case "latticeFill":
      case "denoise":
      case "delaunayFlips":
      case "punchDrainHole":
      case "removeMesh":
        return meshes.has(req.id);
      case "boolean":
      case "placeMesh":
        return meshes.has(req.target);
      case "join":
        return meshes.has(req.target) || meshes.has(req.source);
      case "register":
      case "alignByLandmarks":
      case "registerMultistage":
        return meshes.has(req.moving);
      case "moveWithCollision":
      case "commitCollisionInteraction":
      case "undo":
      case "redo":
        return true;
      case "field": {
        const on = this.#fieldMeshes.get(req.field);
        return req.op === "remove" && on !== undefined && on.every((m) => meshes.has(m));
      }
      default:
        return false;
    }
  }

  /**
   * Brings every due field up to date at once, for a request that reads
   * fields (their values, status or colours). A field error is swallowed
   * like a failed eager update's; a trap is rethrown.
   */
  #settleFields(): void {
    try {
      // One call computes the whole unit.
      while (this.#steps.step(ALL_POINTS) === "pending") {}
    } catch (e) {
      if (isPoison(e)) {
        throw e;
      }
    }
  }

  /** Posts the finished batch's result; `false` when packing failed (poisoned). */
  #finish(): boolean {
    const b = this.#batch;
    this.#batch = new Batch();
    let result: Result;
    try {
      result = this.#result(b);
    } catch (e) {
      // Packing failed: the frame's changes are taken and lost, so the
      // mirrors can no longer follow. Stop like a trap.
      this.#poison(b, isPoison(e) ? POISONED : packingFailed(e), undefined, true);
      return false;
    }
    this.#lastRev = Math.max(this.#lastRev, result.rev);
    try {
      this.#port.post(result, b.transfer);
    } catch (e) {
      // Not sent (a value that cannot be cloned, say): the frame's changes
      // are lost, as when packing fails. Its poisoned result goes instead,
      // so every batch still gets exactly one result.
      this.#poison(b, packingFailed(e), undefined, true);
      return false;
    }
    return true;
  }

  /**
   * Stops the engine and posts the batches as failed. In `b`'s result every
   * status and reply already collected, `failed` and the rest of its batch
   * (none when `ended`: its end was reached) get `error`; every later batch
   * already received gets a poisoned result of its own.
   */
  #poison(b: Batch, error: ErrorWire, failed?: Request, ended = false): void {
    this.#poisoned = true;
    this.#failAdds(b, error);
    const rest = this.#queue.splice(0);
    for (const st of b.status) {
      st.status = "error";
      st.error = error;
    }
    for (const reply of b.replies) {
      reply.ok = false;
      reply.error = error;
      delete reply.value;
    }
    b.transfer.length = 0;
    let end = ended ? -1 : rest.indexOf(END);
    if (end < 0 && !ended) {
      end = rest.length;
    }
    const own = rest.slice(0, Math.max(end, 0)) as Request[];
    this.#postPoisoned(failed === undefined ? own : [failed, ...own], b, error);
    for (let start = end + 1; start < rest.length; ) {
      let next = rest.indexOf(END, start);
      if (next < 0) {
        next = rest.length;
      }
      this.#postPoisoned(rest.slice(start, next) as Request[], new Batch(), error);
      start = next + 1;
    }
  }

  /**
   * Stops the engine while no batch is open (a background add trapped or
   * could not be posted while the Worker was idle): `b`'s outcome, every
   * pending add's key failed with it, goes out as a poisoned `update`,
   * which belongs to no batch; batches received but not started (none,
   * normally) get poisoned results of their own, as later ones will.
   */
  #poisonIdle(b: Batch, error: ErrorWire): void {
    this.#poisoned = true;
    this.#failAdds(b, error);
    for (const st of b.status) {
      st.status = "error";
      st.error = error;
    }
    const result: Result = {
      rev: this.#lastRev,
      changes: [],
      placements: [],
      sections: [],
      history: { canUndo: false, canRedo: false },
      status: b.status,
      replies: [],
      colorings: [],
      poisoned: true,
    };
    try {
      this.#port.postUpdate?.(result, []);
    } catch {
      // The channel itself is gone: nothing more can be sent.
    }
    const rest = this.#queue.splice(0);
    for (let start = 0; start < rest.length; ) {
      let next = rest.indexOf(END, start);
      if (next < 0) {
        next = rest.length;
      }
      this.#postPoisoned(rest.slice(start, next) as Request[], new Batch(), error);
      start = next + 1;
    }
  }

  /** Fails the status of every background add still pending into `b`, and forgets them. */
  #failAdds(b: Batch, error: ErrorWire): void {
    for (const a of this.#adds.splice(0)) {
      b.error(a.key, a.rev, error);
    }
  }

  #postPoisoned(failed: Request[], b: Batch, error: ErrorWire = POISONED): void {
    for (const req of failed) {
      b.rev = Math.max(b.rev, req.rev);
      b.error(keyOf(req), req.rev, error);
      const qid = (req as { qid?: number }).qid;
      if (qid !== undefined) {
        b.replies.push({ qid, ok: false, error });
      }
    }
    const result: Result = {
      rev: b.rev,
      changes: [],
      placements: [],
      sections: [],
      history: { canUndo: false, canRedo: false },
      status: b.status,
      replies: b.replies,
      colorings: [],
      poisoned: true,
    };
    try {
      this.#port.post(result, b.transfer);
    } catch {
      // Something in it cannot be sent: send the bare outcome (keys,
      // revisions, qids and the error only), so the batch still gets its
      // one result.
      try {
        this.#port.post(
          {
            ...result,
            status: b.status.map((st) => ({ key: st.key, rev: st.rev, status: "error" as const, error })),
            replies: b.replies.map((r) => ({ qid: r.qid, ok: false, error })),
          },
          [],
        );
      } catch {
        // The channel itself is gone: nothing more can be sent.
      }
    }
  }

  /**
   * Whether a repair list runs one operation per chunk, inside a history
   * step: one undo step, reverted when an operation fails, like the list
   * call. Only where that is the same as the list call: more than one
   * operation, no drag holding a step open, and no redo steps (the first
   * recorded operation would drop them even if a later one fails).
   */
  #inSteps(req: Extract<Request, { kind: "edit" | "repair" }>): boolean {
    return (
      Array.isArray(req.payload) &&
      req.payload.length > 1 &&
      !req.payload.some((o) => {
        const name = (o as { op?: unknown } | null)?.op;
        return name === "captureRestSizes" || name === "captureStrokeState";
      }) &&
      this.#drag === undefined &&
      this.#stepDepth === 0 &&
      !this.#scene.canRedo()
    );
  }

  async #repairInSteps(req: Extract<Request, { kind: "edit" | "repair" }>, b: Batch): Promise<void> {
    const s = this.#scene;
    const mesh = this.#mesh(req.id);
    const ops = req.payload as RepairOp[];
    const reports: unknown[] = [];
    s.beginStep();
    try {
      for (let k = 0; k < ops.length; k++) {
        if (k > 0) {
          // Only receives: nothing else runs before this request is done.
          await this.#maybeYield();
        }
        reports.push(s.repair(mesh, ops[k]));
        this.#onChunk?.({ kind: "repairOp", id: req.id });
      }
    } catch (e) {
      if (!isPoison(e)) {
        s.cancelStep();
      }
      throw e;
    }
    s.endStep();
    if (req.qid !== undefined) {
      b.replies.push({ qid: req.qid, ok: true, value: reports });
    }
  }

  /**
   * Adds a field and builds it in steps of {@link EngineOptions.stepPoints}
   * (its sign data, near search and values), yielding between them once the
   * time budget is spent. At each step boundary the 9a rule applies: when
   * the display request the add feeds is superseded by a queued one, the
   * build is abandoned and the field, never computed, is withdrawn without a
   * report and kept as deferred (built when first needed), as if abandoned
   * before it started; `false` then. Other requests wait for the build. A
   * build that fails withdraws the field and throws, so the add fails and
   * the scene is as before.
   */
  async #addFieldInSteps(req: Extract<Request, { kind: "field" }>, key: string | undefined): Promise<boolean> {
    if (this.#fields.has(req.field) || this.#deferred.has(req.field)) {
      throw invalid(`field client id ${req.field} is already in use`);
    }
    const desc = req.desc;
    if (desc === undefined) {
      throw invalid("adding a field needs a desc");
    }
    const steps = this.#steps;
    const h = steps.addDeferred({ ...desc, on: this.#mesh(desc.on), to: desc.to.map((m) => this.#mesh(m)) });
    const forget = () => {
      // Never computed and colouring nothing, so the scene takes it back
      // without a report. Should it ever refuse, remove it the usual way:
      // that reports the removal, so the handle's client id stays until the
      // frame is packed (the cleanup after packing drops it).
      if (steps.withdraw(h)) {
        this.#fieldIds.delete(h);
      } else {
        this.#scene.removeField(h);
        this.#removedFields.push(h);
      }
      this.#fields.delete(req.field);
      this.#fieldMeshes.delete(req.field);
    };
    this.#fields.set(req.field, h);
    this.#fieldIds.set(h, req.field);
    this.#fieldMeshes.set(req.field, [desc.on, ...desc.to]);
    for (;;) {
      let state: "idle" | "pending" | "done";
      try {
        state = steps.stepField(h, this.#stepPoints);
      } catch (e) {
        if (!isPoison(e)) {
          forget();
        }
        throw e;
      }
      this.#onChunk?.({ kind: "fieldBuildStep", id: req.field });
      if (state !== "pending") {
        break;
      }
      await this.#maybeYield();
      if (this.#feedsSuperseded(key)) {
        steps.abandon();
        forget();
        this.#deferred.set(req.field, desc);
        this.#fieldMeshes.set(req.field, [desc.on, ...desc.to]);
        this.#onChunk?.({ kind: "fieldBuildAbandon", id: req.field });
        return false;
      }
    }
    this.#onChunk?.({ kind: "fieldBuild", id: req.field });
    return true;
  }

  // ---- Ids ----

  #mesh(id: ClientId): MeshHandle {
    const h = this.#meshes.get(id);
    if (h === undefined) {
      throw unknown("UNKNOWN_MESH", id);
    }
    return h;
  }

  /** A field's handle; a deferred field is built first. */
  #field(id: ClientId): FieldHandle {
    const h = this.#fields.get(id);
    if (h !== undefined) {
      return h;
    }
    const desc = this.#deferred.get(id);
    if (desc === undefined) {
      throw unknown("UNKNOWN_FIELD", id);
    }
    return this.#buildField(id, desc);
  }

  /**
   * Computes a field (one chunk). Built late, a field has the values it
   * would have had (eager fields follow every edit and placement); only its
   * errors (a taken name, say) surface late, on the request that needs it.
   * A failed build keeps the field deferred: the next request that needs it
   * tries again (and reports the error again), and removing it still works.
   */
  #buildField(id: ClientId, desc: FieldDescWire): FieldHandle {
    const h = this.#scene.addDistanceField({ ...desc, on: this.#mesh(desc.on), to: desc.to.map((m) => this.#mesh(m)) });
    this.#deferred.delete(id);
    this.#fields.set(id, h);
    this.#fieldIds.set(h, id);
    this.#fieldMeshes.set(id, [desc.on, ...desc.to]);
    this.#onChunk?.({ kind: "fieldBuild", id });
    return h;
  }

  #section(id: ClientId): SectionHandle {
    const h = this.#sections.get(id);
    if (h === undefined) {
      throw unknown("UNKNOWN_SECTION", id);
    }
    return h;
  }

  #meshList(ids: unknown): MeshHandle[] {
    if (!Array.isArray(ids)) {
      throw invalid("meshes must be an array of client ids");
    }
    return ids.map((id: ClientId) => this.#mesh(id));
  }

  /** The client id of an engine mesh handle (a mesh of this engine always has one). */
  #meshId(h: MeshHandle): ClientId {
    return this.#meshIds.get(h) as ClientId;
  }

  /**
   * Drops engine-side references of a removed mesh: fields on it (or only
   * to it) and its section entries. The scene did the same with the mesh. A
   * `"parked"` mesh keeps the fields on it, and leaves those of other parked
   * meshes alone: they come back together. What is left of the fields is
   * reported when the client did not ask for the removal, or may see it in
   * another order (`"parked"` and `"gone"`).
   */
  #dropMeshRefs(id: ClientId, why: "removed" | "parked" | "gone" = "removed"): void {
    for (const [field, [on, ...to]] of this.#fieldMeshes) {
      if (why === "parked" && this.#parked.has(on)) {
        continue;
      }
      if (on === id || (to.length === 1 && to[0] === id)) {
        this.#dropField(field);
        if (why !== "removed") {
          this.#fieldTargets.set(field, null);
        }
      } else if (to.includes(id)) {
        // The field keeps its other targets; so does a deferred build.
        const left = to.filter((m) => m !== id);
        this.#fieldMeshes.set(field, [on, ...left]);
        if (why !== "removed") {
          this.#fieldTargets.set(field, left);
        }
        const desc = this.#deferred.get(field);
        if (desc !== undefined) {
          this.#deferred.set(field, { ...desc, to: desc.to.filter((m) => m !== id) });
        }
      }
    }
    for (const [section, meshes] of this.#sectionMeshes) {
      if (meshes.includes(id)) {
        this.#sectionMeshes.set(
          section,
          meshes.filter((m) => m !== id),
        );
        this.#goneEntries.push({ section, mesh: id });
        if (why !== "removed") {
          this.#sectionsLost.add(section);
        }
      }
    }
  }

  /**
   * After a request that may park or restore meshes (an undoable removal, a
   * join, an undo, a redo, a cancelled step): brings the engine's books in
   * line with where the scene has each mesh.
   */
  #syncPresence(): void {
    const left: ClientId[] = [];
    for (const [id, h] of this.#meshes) {
      const live = this.#scene.has(h);
      if (live !== this.#parked.has(id)) {
        continue;
      }
      this.#moved.add(id);
      if (live) {
        this.#parked.delete(id);
        if (this.#doomed.delete(id)) {
          this.#scene.removeMesh(h);
          this.#forgetMesh(id, h);
        }
      } else {
        this.#parked.add(id);
        left.push(id);
      }
    }
    for (const id of left) {
      if (this.#drag?.id === id) {
        this.#drag = undefined;
      }
      this.#dropMeshRefs(id, "parked");
    }
  }

  /** Forgets a mesh removed for good. */
  #forgetMesh(id: ClientId, h: MeshHandle): void {
    this.#meshes.delete(id);
    // `#meshIds` keeps the handle until its removal record is sent.
    this.#removed.push(h);
    if (this.#drag?.id === id) {
      this.#drag = undefined;
    }
    this.#dropMeshRefs(id);
  }

  /** Forgets the parked meshes the history has dropped for good. */
  #dropGone(): void {
    for (const id of [...this.#parked]) {
      const h = this.#meshes.get(id) as MeshHandle;
      if (this.#scene.isRestorable(h)) {
        continue;
      }
      this.#parked.delete(id);
      this.#doomed.delete(id);
      this.#moved.add(id);
      this.#meshes.delete(id);
      // `#meshIds` keeps the handle until its removal record is sent.
      this.#removed.push(h);
      this.#dropMeshRefs(id, "gone");
    }
  }

  // ---- Requests ----

  #run(req: Request, b: Batch): void {
    const s = this.#scene;
    switch (req.kind) {
      case "addMesh": {
        if (this.#meshes.has(req.id) || this.#adds.some((a) => a.id === req.id)) {
          throw invalid(`mesh client id ${req.id} is already in use`);
        }
        const positions = req.f64 ? new Float64Array(req.positions) : new Float32Array(req.positions);
        const h = s.addMesh(positions, new Uint32Array(req.indices), {
          ...req.options,
          mode: "eager",
          ...(req.uvs === undefined ? {} : { uvs: new Float32Array(req.uvs) }),
        });
        this.#meshes.set(req.id, h);
        this.#meshIds.set(h, req.id);
        return;
      }
      case "removeMesh": {
        if (req.undoable === true) {
          // The result says where it is, whatever comes of this.
          this.#moved.add(req.id);
        }
        if (this.#parked.has(req.id)) {
          // An undo or redo took it out before the client knew.
          if (req.undoable !== true) {
            this.#doomed.add(req.id);
          }
          return;
        }
        const h = this.#mesh(req.id);
        if (req.undoable === true) {
          s.removeMesh(h, { undoable: true });
          this.#syncPresence();
          return;
        }
        s.removeMesh(h);
        this.#forgetMesh(req.id, h);
        return;
      }
      case "join": {
        const target = this.#mesh(req.target);
        const source = this.#mesh(req.source);
        const value = s.join(target, source, req.payload as JoinOptions);
        // The source is parked: the result says so.
        this.#syncPresence();
        if (req.qid !== undefined) {
          b.replies.push({ qid: req.qid, ok: true, value });
        }
        return;
      }
      case "register": {
        const value = s.register(this.#mesh(req.moving), this.#mesh(req.fixed), req.payload as RegisterOptions);
        b.replies.push({ qid: req.qid, ok: true, value });
        return;
      }
      case "boolean": {
        s.boolean(this.#mesh(req.target), this.#mesh(req.source), req.payload.op, req.payload);
        b.replies.push({ qid: req.qid, ok: true, value: undefined });
        return;
      }
      case "booleanBatch": {
        if (this.#meshes.has(req.addedId) || this.#adds.some((a) => a.id === req.addedId)) {
          throw invalid(`mesh client id ${req.addedId} is already in use`);
        }
        const added = s.booleanBatch(this.#meshList(req.meshes), req.tree, req.options);
        this.#meshes.set(req.addedId, added);
        this.#meshIds.set(added, req.addedId);
        b.replies.push({ qid: req.qid, ok: true, value: undefined });
        return;
      }
      case "cutByPlane": {
        if (this.#meshes.has(req.addedId) || this.#adds.some((a) => a.id === req.addedId)) {
          throw invalid(`mesh client id ${req.addedId} is already in use`);
        }
        const r = s.cutByPlane(this.#mesh(req.id), req.plane, { cap: req.cap });
        this.#meshes.set(req.addedId, r.removed);
        this.#meshIds.set(r.removed, req.addedId);
        b.replies.push({
          qid: req.qid,
          ok: true,
          value: {
            keptTriangles: r.keptTriangles,
            removedTriangles: r.removedTriangles,
            capTriangles: r.capTriangles,
          },
        });
        return;
      }
      case "trimByCurve": {
        const r = s.trimByCurve(this.#mesh(req.id), req.payload);
        b.replies.push({
          qid: req.qid,
          ok: true,
          value: {
            trianglesAdded: r.trianglesAdded,
            trianglesRemoved: r.trianglesRemoved,
            imprintTriangles: r.imprintTriangles,
            cutTriangles: r.cutTriangles,
            removedArea: r.removedArea,
            verticesAdded: r.verticesAdded,
            seamEdges: r.seamEdges,
          },
        });
        return;
      }
      case "fillHoleContinuity": {
        const value = s.fillHoleContinuity(this.#mesh(req.id), req.loopIndex, req.options);
        b.replies.push({ qid: req.qid, ok: true, value });
        return;
      }
      case "fillAllHolesContinuity": {
        const value = s.fillAllHolesContinuity(this.#mesh(req.id), req.options);
        b.replies.push({ qid: req.qid, ok: true, value });
        return;
      }
      case "bridgePartialArcs": {
        const value = s.bridgePartialArcs(this.#mesh(req.id), req.loopA, req.loopB, req.options);
        b.replies.push({ qid: req.qid, ok: true, value });
        return;
      }
      case "applyDyntopo": {
        const value = s.applyDyntopo(this.#mesh(req.id), req.center, req.radius, req.options, req.pinned);
        b.replies.push({ qid: req.qid, ok: true, value });
        if (value.createdVertices.buffer instanceof ArrayBuffer) {
          b.transfer.push(value.createdVertices.buffer);
        }
        if (value.removedVertices.buffer instanceof ArrayBuffer) {
          b.transfer.push(value.removedVertices.buffer);
        }
        return;
      }
      case "dyntopoDab": {
        const value = s.dyntopoDab(this.#mesh(req.id), req.dab, req.options);
        b.replies.push({ qid: req.qid, ok: true, value });
        if (value.createdVertices.buffer instanceof ArrayBuffer) {
          b.transfer.push(value.createdVertices.buffer);
        }
        if (value.removedVertices.buffer instanceof ArrayBuffer) {
          b.transfer.push(value.removedVertices.buffer);
        }
        return;
      }
      case "alignByLandmarks": {
        const value = s.alignByLandmarks(this.#mesh(req.moving), this.#mesh(req.fixed), req.pairs, req.allowScaling);
        b.replies.push({ qid: req.qid, ok: true, value });
        return;
      }
      case "registerMultistage": {
        const value = s.registerMultistage(this.#mesh(req.moving), this.#mesh(req.fixed), req.payload);
        b.replies.push({ qid: req.qid, ok: true, value });
        return;
      }
      case "deleteSelectedFaces": {
        const value = s.deleteSelectedFaces(this.#mesh(req.id), req.selectedTris);
        b.replies.push({ qid: req.qid, ok: true, value });
        return;
      }
      case "extrudeFaces": {
        const value = s.extrudeFaces(this.#mesh(req.id), req.triangles, req.options);
        b.replies.push({ qid: req.qid, ok: true, value });
        return;
      }
      case "extrudeBase": {
        const value = s.extrudeBase(this.#mesh(req.id), req.options);
        b.replies.push({ qid: req.qid, ok: true, value });
        return;
      }
      case "extractSelectedFaces": {
        this.#ensureUnused(req.addedId);
        const added = s.extractSelectedFaces(this.#mesh(req.id), req.selectedTris, { normals: req.normals });
        this.#bindAdded(req.addedId, added);
        b.replies.push({ qid: req.qid, ok: true, value: undefined });
        return;
      }
      case "splitSelectedFaces": {
        this.#ensureUnused(req.addedId);
        const r = s.splitSelectedFaces(this.#mesh(req.id), req.selectedTris, { normals: req.normals });
        this.#bindAdded(req.addedId, r.detached);
        const value: SplitFacesReportWire = {
          detached: req.addedId,
          removedTriangles: r.removedTriangles,
          removedVertices: r.removedVertices,
        };
        b.replies.push({ qid: req.qid, ok: true, value });
        return;
      }
      case "applyFfd": {
        const value = s.applyFfd(this.#mesh(req.id), req.options);
        b.replies.push({ qid: req.qid, ok: true, value });
        return;
      }
      case "applyArap": {
        const value = s.applyArap(this.#mesh(req.id), req.options);
        b.replies.push({ qid: req.qid, ok: true, value });
        return;
      }
      case "applyKelvinlet": {
        const value = s.applyKelvinlet(this.#mesh(req.id), req.options);
        b.replies.push({ qid: req.qid, ok: true, value });
        return;
      }
      case "applyXpbd": {
        const value = s.applyXpbd(this.#mesh(req.id), req.options);
        b.replies.push({ qid: req.qid, ok: true, value });
        return;
      }
      case "setDefaultCollisionRule":
        s.setDefaultCollisionRule(req.rule);
        return;
      case "setCollisionRule":
        s.setCollisionRule(this.#mesh(req.a), this.#mesh(req.b), req.rule);
        return;
      case "clearCollisionRule":
        s.clearCollisionRule(this.#mesh(req.a), this.#mesh(req.b));
        return;
      case "moveWithCollision": {
        const r = s.moveWithCollision(this.#mesh(req.id), req.placement, req.isRelease);
        b.replies.push({ qid: req.qid, ok: true, value: this.#mapCollisionReport(r) });
        return;
      }
      case "commitCollisionInteraction": {
        const r = s.commitCollisionInteraction(this.#mesh(req.id));
        b.replies.push({ qid: req.qid, ok: true, value: this.#mapCollisionReport(r) });
        return;
      }
      case "alignToFittedPrimitive": {
        const value = s.alignToFittedPrimitive(this.#mesh(req.id), req.primitive, req.targetAxis, req.targetOrigin);
        b.replies.push({ qid: req.qid, ok: true, value });
        return;
      }
      case "reconstructPointCloud": {
        this.#ensureUnused(req.addedId);
        const added = s.reconstructPointCloud(req.cloud, req.options);
        this.#bindAdded(req.addedId, added);
        b.replies.push({ qid: req.qid, ok: true, value: undefined });
        return;
      }
      case "addHeightfield": {
        this.#ensureUnused(req.addedId);
        const { normals, ...placement } = req.placement;
        const { mesh, ...value } = s.addHeightfield(req.field, { placement, normals });
        this.#bindAdded(req.addedId, mesh);
        b.replies.push({ qid: req.qid, ok: true, value });
        return;
      }
      case "convexHull":
      case "convexHullPoints": {
        this.#ensureUnused(req.addedId);
        const options = { normals: req.normals };
        const r =
          req.kind === "convexHull"
            ? s.convexHull(this.#meshList(req.meshes), options)
            : s.convexHullOfPoints(req.points, options);
        this.#bindAdded(req.addedId, r.mesh);
        const value = { inputPoints: r.inputPoints, vertices: r.vertices, triangles: r.triangles };
        b.replies.push({ qid: req.qid, ok: true, value });
        return;
      }
      case "denoise": {
        const value = s.denoise(this.#mesh(req.id), req.options);
        b.replies.push({ qid: req.qid, ok: true, value });
        return;
      }
      case "delaunayFlips": {
        const value = s.delaunayFlips(this.#mesh(req.id), req.options);
        b.replies.push({ qid: req.qid, ok: true, value });
        return;
      }
      case "wrap": {
        this.#ensureUnused(req.addedId);
        const { mesh, ...value } = s.wrap(this.#meshList(req.meshes), req.options);
        this.#bindAdded(req.addedId, mesh);
        b.replies.push({ qid: req.qid, ok: true, value });
        return;
      }
      case "hollowMesh": {
        const value = s.hollowMesh(this.#mesh(req.id), req.options);
        b.replies.push({ qid: req.qid, ok: true, value });
        return;
      }
      case "latticeFill": {
        const value = s.latticeFill(this.#mesh(req.id), req.options);
        b.replies.push({ qid: req.qid, ok: true, value });
        return;
      }
      case "punchDrainHole": {
        if (req.plugId !== undefined) {
          this.#ensureUnused(req.plugId);
        }
        const r = s.punchDrainHole(this.#mesh(req.id), req.options);
        const plugIds: ClientId[] = [];
        if (r.plugs.length > 0) {
          if (r.plugs.length > 1 || req.plugId === undefined) {
            for (const h of r.plugs) {
              s.removeMesh(h);
            }
            throw invalid("punchDrainHole produced unexpected plug count");
          }
          this.#bindAdded(req.plugId, r.plugs[0]);
          plugIds.push(req.plugId);
        }
        const value: DrainHoleReportWire = {
          holesPunched: r.holesPunched,
          keptTriangles: r.keptTriangles,
          usedFallback: r.usedFallback,
          plugs: plugIds,
          methodUsed: r.methodUsed,
          uvsDropped: r.uvsDropped,
        };
        b.replies.push({ qid: req.qid, ok: true, value });
        return;
      }
      case "placeMesh": {
        if (req.addedId !== undefined) {
          this.#ensureUnused(req.addedId);
        }
        const r = s.placeMesh(this.#mesh(req.target), this.#mesh(req.stamp), req.options);
        let addedId: ClientId | null = null;
        if (r.added !== null) {
          if (req.addedId === undefined) {
            s.removeMesh(r.added);
            throw invalid("placeMesh produced an unexpected detached mesh");
          }
          this.#bindAdded(req.addedId, r.added);
          addedId = req.addedId;
        }
        const value: PlaceMeshReportWire = {
          transform: r.transform,
          keptTriangles: r.keptTriangles,
          usedFallback: r.usedFallback,
          conformMaxDisplacement: r.conformMaxDisplacement,
          added: addedId,
          uvsDropped: r.uvsDropped,
        };
        b.replies.push({ qid: req.qid, ok: true, value });
        return;
      }
      case "fieldInteraction":
        s.setFieldInteraction(req.points);
        return;
      case "cancel":
        // Its query is done, or was dropped.
        return;
      case "compact":
        s.compact(this.#mesh(req.id));
        return;
      case "setPlacement":
        s.setPlacement(this.#mesh(req.id), {
          rotation: req.rotation,
          translation: req.translation as [number, number, number],
          ...(req.scale === undefined ? {} : { scale: req.scale as [number, number, number] }),
        });
        return;
      case "drag":
        this.#runDrag(req);
        return;
      case "edit": {
        const value = s.edit(this.#mesh(req.id), req.payload as EditOp[]);
        if (req.qid !== undefined) {
          b.replies.push({ qid: req.qid, ok: true, value });
        }
        return;
      }
      case "repair": {
        const value = s.repair(this.#mesh(req.id), req.payload as RepairOp);
        if (req.qid !== undefined) {
          b.replies.push({ qid: req.qid, ok: true, value });
        }
        return;
      }
      case "undo":
        s.undo();
        this.#stepDepth = 0;
        if (this.#drag !== undefined && !this.#drag.session.active) {
          this.#drag = undefined;
        }
        this.#syncPresence();
        return;
      case "redo":
        s.redo();
        this.#stepDepth = 0;
        if (this.#drag !== undefined && !this.#drag.session.active) {
          this.#drag = undefined;
        }
        this.#syncPresence();
        return;
      case "clearHistory":
        if (this.#drag !== undefined) {
          // Closing the history ends the drag's step: end the drag first.
          this.#drag.session.end();
          this.#drag = undefined;
        }
        s.clearHistory();
        this.#stepDepth = 0;
        return;
      case "historyBudget":
        s.setHistoryBudget(req.bytes);
        return;
      case "step":
        switch (req.phase) {
          case "begin":
            s.beginStep();
            this.#stepDepth++;
            return;
          case "end":
            s.endStep();
            this.#stepDepth = Math.max(0, this.#stepDepth - 1);
            if (this.#drag !== undefined && !this.#drag.session.active) {
              this.#drag = undefined;
            }
            return;
          case "cancel":
            s.cancelStep();
            this.#stepDepth = 0;
            if (this.#drag !== undefined && !this.#drag.session.active) {
              this.#drag = undefined;
            }
            this.#syncPresence();
            return;
        }
        throw invalid(`unknown step phase ${(req as { phase: unknown }).phase}`);
      case "coloring":
        s.setColoring(this.#mesh(req.id), req.rule === null ? null : this.#rule(req.rule));
        b.colorings.push({ id: req.id, rev: req.rev });
        return;
      case "values":
        this.#valuesFull ||= !this.#mirrorValues;
        this.#mirrorValues = true;
        return;
      case "field":
        this.#runField(req);
        return;
      case "section":
        this.#runSection(req);
        return;
      case "query": {
        if (
          req.name === "fieldStatus" ||
          req.name === "fieldNearest" ||
          req.name === "fieldTargetStatus" ||
          req.name === "debugDump"
        ) {
          this.#settleFields();
        }
        const value = this.#query(req.name, req.args);
        b.replies.push({ qid: req.qid, ok: true, value });
        const pushOwned = (view: unknown) => {
          if (
            ArrayBuffer.isView(view) &&
            view.byteOffset === 0 &&
            view.byteLength === view.buffer.byteLength &&
            view.buffer instanceof ArrayBuffer
          ) {
            b.transfer.push(view.buffer);
          }
        };
        if (
          req.name === "verticesInSphere" ||
          req.name === "selfIntersectingTriangles" ||
          req.name === "selectFacesScreen" ||
          req.name === "selectFacesInSphere" ||
          req.name === "selectByCreaseAngle" ||
          req.name === "selectConnectedShell" ||
          req.name === "selectSmallFaceIslands" ||
          req.name === "growFaceSelection" ||
          req.name === "shrinkFaceSelection" ||
          req.name === "fillSelectionHoles" ||
          req.name === "smoothSelectionBoundary" ||
          req.name === "faceSelectionToVertexMask" ||
          req.name === "vertexMaskToFaceSelection"
        ) {
          pushOwned(value);
        } else if (req.name === "latticePreview") {
          const preview = value as QueryMap["latticePreview"]["value"];
          pushOwned(preview.positions);
          pushOwned(preview.indices);
        } else if (req.name === "buildFfdLattice") {
          pushOwned((value as QueryMap["buildFfdLattice"]["value"]).offsets);
        } else if (req.name === "fitPrimitive") {
          pushOwned((value as QueryMap["fitPrimitive"]["value"]).residuals);
        } else if (req.name === "sliceContours") {
          for (const loop of (value as QueryMap["sliceContours"]["value"]).loops) {
            pushOwned(loop.points2d);
            pushOwned(loop.points3d);
          }
        } else if (req.name === "sliceContourStack") {
          for (const slice of value as QueryMap["sliceContourStack"]["value"]) {
            for (const loop of slice.loops) {
              pushOwned(loop.points2d);
              pushOwned(loop.points3d);
            }
          }
        } else if (req.name === "overhangs") {
          pushOwned((value as QueryMap["overhangs"]["value"]).vertexAnglesDeg);
        } else if (req.name === "triangleQuality") {
          pushOwned((value as QueryMap["triangleQuality"]["value"]).vertexQuality);
        } else if (req.name === "meshDeviation") {
          pushOwned((value as QueryMap["meshDeviation"]["value"]).vertexDeviations);
        } else if (req.name === "heightMap") {
          pushOwned((value as QueryMap["heightMap"]["value"]).heights);
        } else if (req.name === "generatePrimitive" || req.name === "generateHeightfield") {
          const mesh = value as QueryMap["generatePrimitive"]["value"];
          pushOwned(mesh.positions);
          pushOwned(mesh.indices);
        }
        return;
      }
      case "fetchFieldValues": {
        this.#settleFields();
        const values = s.fieldValues(this.#field(req.field)).slice();
        b.replies.push({ qid: req.qid, ok: true, value: values });
        b.transfer.push(values.buffer);
        return;
      }
      default:
        throw invalid(`unknown request kind ${(req as { kind: unknown }).kind}`);
    }
  }

  #runDrag(req: Extract<Request, { kind: "drag" }>): void {
    const pose = () => {
      if (req.rotation === undefined || req.translation === undefined) {
        throw invalid("a drag move needs a rotation and a translation");
      }
      return {
        rotation: req.rotation,
        translation: req.translation as [number, number, number],
        ...(req.scale === undefined ? {} : { scale: req.scale as [number, number, number] }),
      };
    };
    if (req.phase === "begin") {
      const handle = this.#mesh(req.id);
      const session = this.#scene.drag(handle);
      const collisionActive = this.#scene.hasActiveCollisionResponse(handle);
      this.#drag = { id: req.id, session, collisionActive };
      return;
    }
    this.#mesh(req.id);
    const drag = this.#drag;
    if (drag === undefined || drag.id !== req.id || !drag.session.active) {
      throw invalid("no active drag of this mesh");
    }
    switch (req.phase) {
      case "move":
        drag.session.move(pose());
        if (drag.collisionActive) {
          this.#forcePlacements.add(req.id);
        }
        return;
      case "end":
        this.#drag = undefined;
        if (req.rotation !== undefined) {
          try {
            drag.session.move(pose());
          } finally {
            drag.session.end();
          }
        } else {
          drag.session.end();
        }
        if (drag.collisionActive) {
          this.#forcePlacements.add(req.id);
        }
        return;
      case "cancel":
        this.#drag = undefined;
        drag.session.cancel();
        return;
    }
  }

  #rule(rule: ColorRuleWire) {
    return { ...rule, field: this.#field(rule.field) };
  }

  /** A field removal (adds run in steps: {@link Engine.#addFieldInSteps}). */
  #runField(req: Extract<Request, { kind: "field" }>): void {
    if (this.#deferred.has(req.field)) {
      // Never built: nothing to remove from the scene.
      this.#dropField(req.field);
      return;
    }
    this.#scene.removeField(this.#field(req.field));
    this.#dropField(req.field);
  }

  /** Forgets a field the scene no longer has; its reverse id goes after packing. */
  #dropField(field: ClientId): void {
    const h = this.#fields.get(field);
    if (h !== undefined) {
      this.#removedFields.push(h);
    }
    this.#fields.delete(field);
    this.#deferred.delete(field);
    this.#fieldMeshes.delete(field);
  }

  #runSection(req: Extract<Request, { kind: "section" }>): void {
    const s = this.#scene;
    const desc = req.desc ?? {};
    switch (req.op) {
      case "add": {
        if (this.#sections.has(req.section)) {
          throw invalid(`section client id ${req.section} is already in use`);
        }
        const h = s.addSection({
          plane: desc.plane as SectionPlane,
          meshes: this.#meshList(desc.meshes),
          fill: desc.fill,
        });
        this.#sections.set(req.section, h);
        this.#sectionIds.set(h, req.section);
        this.#sectionMeshes.set(req.section, [...(desc.meshes as ClientId[])]);
        return;
      }
      case "set": {
        const h = this.#section(req.section);
        if (desc.plane !== undefined) {
          s.setSectionPlane(h, desc.plane);
        }
        if (desc.meshes !== undefined) {
          s.setSectionMeshes(h, this.#meshList(desc.meshes));
          this.#sectionMeshes.set(req.section, [...desc.meshes]);
        }
        if (desc.fill !== undefined) {
          s.setSectionFill(h, desc.fill);
        }
        return;
      }
      case "remove": {
        const h = this.#section(req.section);
        s.removeSection(h);
        this.#sections.delete(req.section);
        this.#sectionMeshes.delete(req.section);
        // Its reverse id goes after packing (the frame still reports it).
        this.#removedSections.push(h);
        return;
      }
    }
  }

  #query(name: string, args: unknown): unknown {
    const s = this.#scene;
    if (typeof args !== "object" || args === null) {
      throw invalid("query args must be an object");
    }
    switch (name) {
      case "raycast": {
        const a = args as QueryMap["raycast"]["args"];
        const hit = s.raycast(a.origin, a.direction, a.options);
        return hit === null ? null : { ...hit, mesh: this.#meshId(hit.mesh) };
      }
      case "closestPoint": {
        const a = args as QueryMap["closestPoint"]["args"];
        const mesh = a.options?.mesh === undefined ? undefined : this.#mesh(a.options.mesh);
        const hit = s.closestPoint(a.point, { maxDistance: a.options?.maxDistance, mesh });
        return hit === null ? null : { ...hit, mesh: this.#meshId(hit.mesh) };
      }
      case "meshDistance": {
        const a = args as QueryMap["meshDistance"]["args"];
        return s.meshDistance(this.#mesh(a.a), this.#mesh(a.b), a.options);
      }
      case "intersects": {
        const a = args as QueryMap["intersects"]["args"];
        return s.intersects(this.#mesh(a.a), this.#mesh(a.b));
      }
      case "selfIntersectingTriangles": {
        const a = args as QueryMap["selfIntersectingTriangles"]["args"];
        return s.selfIntersectingTriangles(this.#mesh(a.mesh));
      }
      case "wrapPlan": {
        const a = args as QueryMap["wrapPlan"]["args"];
        return s.wrapPlan(this.#meshList(a.meshes), a.options);
      }
      case "removalBytes":
        return s.removalBytes(this.#mesh((args as QueryMap["removalBytes"]["args"]).mesh));
      case "latticePlan": {
        const a = args as QueryMap["latticePlan"]["args"];
        return s.latticePlan(this.#mesh(a.mesh), a.options);
      }
      case "latticePreview": {
        const a = args as QueryMap["latticePreview"]["args"];
        return s.latticePreview(this.#mesh(a.mesh), a.options);
      }
      case "findContacts": {
        const a = args as QueryMap["findContacts"]["args"];
        const pairs = a.pairs?.map(([x, y]): [MeshHandle, MeshHandle] => [this.#mesh(x), this.#mesh(y)]);
        return s
          .findContacts({ margin: a.margin, pairs })
          .map((c) => ({ ...c, a: this.#meshId(c.a), b: this.#meshId(c.b) }));
      }
      case "analyze": {
        const a = args as QueryMap["analyze"]["args"];
        return s.analyze(this.#mesh(a.mesh), a.options);
      }
      case "measure": {
        const a = args as QueryMap["measure"]["args"];
        return s.measure(this.#mesh(a.mesh));
      }
      case "verticesInSphere": {
        const a = args as QueryMap["verticesInSphere"]["args"];
        return s.verticesInSphere(this.#mesh(a.mesh), a.center, a.radius);
      }
      case "fieldTargetStatus": {
        const a = args as QueryMap["fieldTargetStatus"]["args"];
        return s.fieldTargetStatus(this.#field(a.field));
      }
      case "fieldNearest": {
        const a = args as QueryMap["fieldNearest"]["args"];
        return s.fieldNearest(this.#field(a.field));
      }
      case "fieldStatus": {
        const a = args as QueryMap["fieldStatus"]["args"];
        return s.fieldStatus(this.#field(a.field));
      }
      case "sectionGap": {
        const a = args as QueryMap["sectionGap"]["args"];
        return s.sectionGap(this.#section(a.section), this.#mesh(a.a), this.#mesh(a.b));
      }
      case "overhangs": {
        const a = args as QueryMap["overhangs"]["args"];
        return s.overhangs(this.#mesh(a.mesh), a.options);
      }
      case "triangleQuality": {
        const a = args as QueryMap["triangleQuality"]["args"];
        return s.triangleQuality(this.#mesh(a.mesh), a.metric, a.sliverThreshold);
      }
      case "meshDeviation": {
        const a = args as QueryMap["meshDeviation"]["args"];
        return s.meshDeviation(this.#mesh(a.source), this.#mesh(a.target), a.options);
      }
      case "releaseGeodesics":
        s.releaseGeodesics();
        return null;
      case "detectSymmetryPlane": {
        const a = args as QueryMap["detectSymmetryPlane"]["args"];
        return s.detectSymmetryPlane(this.#mesh(a.mesh));
      }
      case "selectFacesScreen": {
        const a = args as QueryMap["selectFacesScreen"]["args"];
        return s.selectFacesScreen(this.#mesh(a.mesh), a.shape, a.options);
      }
      case "selectFacesInSphere": {
        const a = args as QueryMap["selectFacesInSphere"]["args"];
        return s.selectFacesInSphere(this.#mesh(a.mesh), a.center, a.radius, a.normalCone);
      }
      case "selectByCreaseAngle": {
        const a = args as QueryMap["selectByCreaseAngle"]["args"];
        return s.selectByCreaseAngle(this.#mesh(a.mesh), a.seedTri, a.maxDihedralRad);
      }
      case "selectConnectedShell": {
        const a = args as QueryMap["selectConnectedShell"]["args"];
        return s.selectConnectedShell(this.#mesh(a.mesh), a.seedTri);
      }
      case "selectSmallFaceIslands": {
        const a = args as QueryMap["selectSmallFaceIslands"]["args"];
        return s.selectSmallFaceIslands(this.#mesh(a.mesh), a.maxFaceCount, a.maxArea);
      }
      case "growFaceSelection": {
        const a = args as QueryMap["growFaceSelection"]["args"];
        return s.growFaceSelection(this.#mesh(a.mesh), a.selectedTris, a.rings, a.adjacency);
      }
      case "shrinkFaceSelection": {
        const a = args as QueryMap["shrinkFaceSelection"]["args"];
        return s.shrinkFaceSelection(this.#mesh(a.mesh), a.selectedTris, a.rings, a.adjacency);
      }
      case "fillSelectionHoles": {
        const a = args as QueryMap["fillSelectionHoles"]["args"];
        return s.fillSelectionHoles(this.#mesh(a.mesh), a.selectedTris, a.maxHoleTris);
      }
      case "smoothSelectionBoundary": {
        const a = args as QueryMap["smoothSelectionBoundary"]["args"];
        return s.smoothSelectionBoundary(this.#mesh(a.mesh), a.selectedTris, a.iterations);
      }
      case "faceSelectionToVertexMask": {
        const a = args as QueryMap["faceSelectionToVertexMask"]["args"];
        return s.faceSelectionToVertexMask(this.#mesh(a.mesh), a.selectedTris);
      }
      case "vertexMaskToFaceSelection": {
        const a = args as QueryMap["vertexMaskToFaceSelection"]["args"];
        return s.vertexMaskToFaceSelection(this.#mesh(a.mesh), a.mask, a.threshold, a.requireAllVertices);
      }
      case "buildFfdLattice": {
        const a = args as QueryMap["buildFfdLattice"]["args"];
        return s.buildFfdLattice(this.#mesh(a.mesh), a.dims, a.basis, a.padding);
      }
      case "fitPrimitive": {
        const a = args as QueryMap["fitPrimitive"]["args"];
        return s.fitPrimitive(this.#mesh(a.mesh), a.options);
      }
      case "sliceContours": {
        const a = args as QueryMap["sliceContours"]["args"];
        return s.sliceContours(this.#mesh(a.mesh), a.origin, a.normal);
      }
      case "sliceContourStack": {
        const a = args as QueryMap["sliceContourStack"]["args"];
        return s.sliceContourStack(this.#mesh(a.mesh), a.origin, a.normal, a.spacing, a.count);
      }
      case "heightMapFrame": {
        const a = args as QueryMap["heightMapFrame"]["args"];
        return s.heightMapFrame(this.#meshList(a.meshes), decodeHeightMapOptions(a.options));
      }
      case "heightMap": {
        const a = args as QueryMap["heightMap"]["args"];
        const frame =
          a.frame === undefined ? decodeHeightMapOptions(a.options ?? []) : decodeHeightMapFrame(a.frame);
        return s.heightMap(this.#meshList(a.meshes), frame, a.rows);
      }
      case "generateHeightfield":
        return generateCheckedHeightfield(args as QueryMap["generateHeightfield"]["args"]);
      case "generatePrimitive":
        return generateEncodedPrimitive(args as QueryMap["generatePrimitive"]["args"]);
      case "debugDump": {
        if (!this.#debug) {
          throw invalid(`unknown query ${name}`);
        }
        const h = this.#mesh((args as { mesh: ClientId }).mesh);
        const v = s.views(h);
        return {
          positions: v.positions().slice(),
          normals: v.normals()?.slice() ?? null,
          colors: v.colors()?.slice() ?? null,
          values: v.values?.()?.slice() ?? null,
          uvs: v.uvs()?.slice() ?? null,
          indices: v.indices().slice(),
          placement: s.placement(h),
          matrix4: Array.from(s.matrix4(h)),
        };
      }
      default:
        throw invalid(`unknown query ${name}`);
    }
  }

  // ---- Results ----

  /** Commits the frame and packs everything that changed. */
  #result(b: Batch): Result {
    const s = this.#scene;
    this.#dropGone();
    let frame = s.commitFrame();
    if (this.#valuesFull) {
      this.#valuesFull = false;
      frame = this.#withValues(frame);
    }
    const changes: MeshChangeWire[] = [];
    const placements: Result["placements"] = [];
    const sections: Result["sections"] = [];
    const forced = new Set(this.#forcePlacements);
    this.#forcePlacements.clear();
    for (const c of frame) {
      const id = this.#meshId(c.mesh);
      changes.push(this.#change(c, id, b));
      if (c.removed) {
        forced.delete(id);
        continue;
      }
      const isForced = forced.delete(id);
      if (c.added || c.placementChanged || isForced) {
        const p = s.placement(c.mesh);
        placements.push({
          id,
          rotation: p.rotation,
          translation: p.translation,
          scale: p.scale as number[],
          matrix4: Array.from(s.matrix4(c.mesh)),
          ...(isForced ? { authoritative: true } : {}),
        });
      }
      for (const sh of c.sectionsChanged) {
        const section = this.#sectionIds.get(sh) as ClientId;
        const result = this.#sectionResult(sh, c.mesh);
        if (result !== null) {
          for (const contour of result.contours) {
            b.transfer.push(contour.points.buffer);
          }
          if (result.fill !== undefined) {
            b.transfer.push(result.fill.positions.buffer, result.fill.indices.buffer);
          }
        }
        sections.push({ section, mesh: id, result });
      }
    }
    for (const id of forced) {
      const h = this.#meshes.get(id);
      if (h === undefined || this.#parked.has(id)) {
        continue;
      }
      const p = s.placement(h);
      placements.push({
        id,
        rotation: p.rotation,
        translation: p.translation,
        scale: p.scale as number[],
        matrix4: Array.from(s.matrix4(h)),
        authoritative: true,
      });
    }
    // A removed mesh's record carries nothing else: report its entries gone.
    for (const { section, mesh } of this.#goneEntries.splice(0)) {
      sections.push({ section, mesh, result: null });
    }
    const presence: NonNullable<Result["presence"]> = [...this.#moved].map((id) => ({
      id,
      state: this.#doomed.has(id) ? "gone" : this.#parked.has(id) ? "parked" : this.#meshes.has(id) ? "live" : "gone",
    }));
    this.#moved.clear();
    const fieldTargets = [...this.#fieldTargets].map(([field, to]) => ({ field, to }));
    this.#fieldTargets.clear();
    const sectionMeshes = [...this.#sectionsLost].flatMap((section) => {
      const meshes = this.#sectionMeshes.get(section);
      return meshes === undefined ? [] : [{ section, meshes }];
    });
    this.#sectionsLost.clear();
    const history = { canUndo: s.canUndo(), canRedo: s.canRedo() };
    // The frame's commit had the scene report the last of them.
    const permanent = this.#permanent.splice(0);
    // Every landed add's record is in this frame.
    this.#packed.clear();
    // Every removal is in this frame, or was never reported (added and
    // removed in between): the removed handles' ids can go.
    for (const h of this.#removed.splice(0)) {
      this.#meshIds.delete(h);
    }
    for (const h of this.#removedFields.splice(0)) {
      this.#fieldIds.delete(h);
    }
    for (const h of this.#removedSections.splice(0)) {
      this.#sectionIds.delete(h);
    }
    return {
      rev: b.rev,
      changes,
      placements,
      sections,
      history,
      historyBytes: s.historyBytes(),
      ...(presence.length === 0 ? {} : { presence }),
      ...(fieldTargets.length === 0 ? {} : { fieldTargets }),
      ...(sectionMeshes.length === 0 ? {} : { sectionMeshes }),
      ...(permanent.length === 0 ? {} : { permanent }),
      status: b.status,
      replies: b.replies,
      colorings: b.colorings,
    };
  }

  /**
   * `frame` with every coloured mesh's colours "all" (a record added where it
   * has none), so the values mirror just turned on starts whole.
   */
  #withValues(frame: MeshChange[]): MeshChange[] {
    const out = [...frame];
    for (const [id, h] of this.#meshes) {
      if (this.#parked.has(id) || (this.#scene.views(h).values?.() ?? null) === null) {
        continue;
      }
      const at = out.findIndex((c) => c.mesh === h);
      if (at === -1) {
        out.push({
          mesh: h,
          added: false,
          removed: false,
          placementChanged: false,
          capacityChanged: false,
          positions: [],
          normals: [],
          colors: "all",
          indices: [],
          fieldsChanged: [],
          sectionsChanged: [],
        });
      } else if (!out[at].removed) {
        out[at] = { ...out[at], colors: "all" };
      }
    }
    return out;
  }

  /** The section's result for a mesh, or `null` when the section or the entry is gone. */
  #sectionResult(section: SectionHandle, mesh: MeshHandle) {
    try {
      return this.#scene.section(section, mesh);
    } catch (e) {
      if (
        e instanceof FastMeshError &&
        (e.code === "INVALID_ARGUMENT" || e.code === "UNKNOWN_MESH" || e.code === "UNKNOWN_SECTION")
      ) {
        return null;
      }
      throw e;
    }
  }

  #change(c: MeshChange, id: ClientId, b: Batch): MeshChangeWire {
    const wire: MeshChangeWire = {
      id,
      added: c.added,
      removed: c.removed,
      ...(c.restorable === true && !this.#doomed.has(id) ? { restorable: true as const } : {}),
      placementChanged: c.placementChanged,
      capacityChanged: c.capacityChanged,
      positions: c.positions,
      normals: c.normals,
      colors: c.colors,
      indices: c.indices,
      ...(c.uvs === undefined ? {} : { uvs: c.uvs }),
      fields: c.fieldsChanged.map((f) => this.#fieldIds.get(f) as ClientId),
      sections: c.sectionsChanged.map((x) => this.#sectionIds.get(x) as ClientId),
      vertexSlots: 0,
      triangleSlots: 0,
      bytes: { positions: null, normals: null, colors: null, values: null, indices: null },
    };
    if (c.removed) {
      return wire;
    }
    if (c.added) {
      wire.origin = this.#scene.origin(c.mesh);
    }
    wire.vertexSlots = this.#scene.vertexSlots(c.mesh);
    wire.triangleSlots = this.#scene.triangleSlots(c.mesh);
    if (
      c.added ||
      c.capacityChanged ||
      c.positions === "all" ||
      c.positions.length > 0 ||
      c.indices === "all" ||
      c.indices.length > 0
    ) {
      wire.bounds = this.#scene.bounds(c.mesh);
    }
    // A background add's buffers copied out before it landed, while its
    // content stamp says the mesh is unchanged since: sent as they are.
    let packed = c.added ? this.#packed.get(c.mesh) : undefined;
    if (packed !== undefined && packed.stamp !== this.#meshAdds.stamp(c.mesh)) {
      packed = undefined;
    }
    // Fresh views, read with no scene call in between.
    const views: MeshViews = this.#scene.views(c.mesh);
    for (const name of Object.keys(STRIDE) as BufferName[]) {
      // The values only once the client turned their mirror on.
      const view = name === "values" && !this.#mirrorValues ? null : (views[name]?.() ?? null);
      if (view !== null) {
        const ranges = rangesOf(c, name);
        const copy = name === "positions" || name === "normals" || name === "indices" ? packed?.[name] : undefined;
        const bytes =
          ranges === "all" && copy != null && copy.byteLength === view.byteLength
            ? copy
            : copyRanges(view, ranges, STRIDE[name]);
        wire.bytes[name] = bytes;
        b.transfer.push(bytes);
      }
    }
    return wire;
  }

  #ensureUnused(id: ClientId): void {
    if (this.#meshes.has(id) || this.#adds.some((a) => a.id === id)) {
      throw invalid(`mesh client id ${id} is already in use`);
    }
  }

  #bindAdded(id: ClientId, handle: MeshHandle): void {
    this.#ensureUnused(id);
    this.#meshes.set(id, handle);
    this.#meshIds.set(handle, id);
  }

  #mapCollisionReport(r: ReturnType<Scene["moveWithCollision"]>): CollisionStepReport<ClientId> {
    return {
      appliedPlacement: r.appliedPlacement,
      appliedQuaternion: r.appliedQuaternion,
      activePairs: r.activePairs.map((p) => ({
        ...p,
        other: this.#meshId(p.other),
      })),
      modifiedMeshes: r.modifiedMeshes.map((m) => this.#meshId(m)),
      minClearance: r.minClearance,
      cutsExecuted: r.cutsExecuted,
      softVerticesDeformed: r.softVerticesDeformed,
      elasticRecoveryCount: r.elasticRecoveryCount,
      selfIntersectionGuardTriggered: r.selfIntersectionGuardTriggered,
    };
  }
}
