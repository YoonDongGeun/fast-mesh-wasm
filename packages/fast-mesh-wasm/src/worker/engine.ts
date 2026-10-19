// The Worker-side engine: a sync `Scene` behind a keyed request queue. It maps
// client ids to the scene's handles, runs requests in arrival order in chunks
// (a request, a field build, an operation of a repair list), yielding to the
// message loop between chunks so newer batches arrive meanwhile: superseded
// "last wins" requests are dropped and abandoned field builds deferred before
// they start. After each batch it posts one `Result` with the frame's change
// records, the bytes of their dirty ranges, placements, section results,
// statuses and replies. Pure: it talks only through `Port`, so it runs in a
// browser Worker or in Node `worker_threads`.
import type { Range } from "../changes.js";
import { FastMeshError } from "../core.js";
import {
  type EditOp,
  type FieldHandle,
  type MeshChange,
  type MeshHandle,
  type RepairOp,
  Scene,
  type SectionHandle,
  type SectionPlane,
} from "../scene.js";
import type { MeshViews } from "../views.js";
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
  supersedes,
  type WorkerMessage,
} from "./protocol.js";

/** The engine's side of the message channel. */
export interface Port {
  post(msg: Result, transfer: Transferable[]): void;
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
   * field's client id) and every operation of a repair list run in steps
   * (`"repairOp"`).
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
}

/** The default {@link EngineOptions.yieldMs}. */
export const YIELD_MS = 4;

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

/**
 * Serves the engine on a Worker's endpoint once `ready` (the WebAssembly
 * module's initialisation) resolves. Batches arriving before are kept and
 * run in order; then `{ kind: "ready" }` is posted first. When `ready`
 * rejects, `{ kind: "fatal" }` is posted instead and nothing runs.
 */
export function serveEngine(endpoint: Endpoint, ready: Promise<unknown>, options: EngineOptions = {}): void {
  let deliver: ((batch: Request[]) => void) | undefined;
  const early: Request[][] = [];
  // Listen right away: messages sent while the module loads are not lost.
  endpoint.onMessage((data) => {
    if (deliver === undefined) {
      early.push(data as Request[]);
    } else {
      deliver(data as Request[]);
    }
  });
  ready.then(
    () => {
      endpoint.post({ kind: "ready" }, []);
      runEngine(
        {
          post: (msg, transfer) => endpoint.post(msg, transfer),
          onMessage: (cb) => {
            deliver = cb;
            for (const batch of early.splice(0)) {
              cb(batch);
            }
          },
        },
        options,
      );
    },
    (e: unknown) => {
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
const STRIDE = { positions: 3, normals: 3, colors: 4, indices: 3 } as const;

type BufferName = keyof typeof STRIDE;

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

/**
 * Yields to the message loop, so batches sent meanwhile are received: a
 * `MessageChannel` tick (nested timers are clamped to 4 ms in browsers),
 * else `setTimeout(0)`.
 */
function messageTick(): () => Promise<void> {
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
  /** Requests received and not run yet, each batch followed by {@link END}. */
  readonly #queue: Item[] = [];
  /** A drain is under way (possibly waiting at a yield). */
  #running = false;
  /** What the batch being run collects for its result. */
  #batch = new Batch();
  #poisoned = false;
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
  readonly #fieldMeshes = new Map<ClientId, [ClientId, ClientId]>();
  /** Each section's meshes, for the entries a mesh removal ends. */
  readonly #sectionMeshes = new Map<ClientId, ClientId[]>();
  /** Section entries ended by mesh removals this batch, reported as `null`. */
  readonly #goneEntries: { section: ClientId; mesh: ClientId }[] = [];
  /** The active drag, by the dragged mesh's client id. */
  #drag: { id: ClientId; session: ReturnType<Scene["drag"]> } | undefined;
  readonly #debug: boolean;
  readonly #onChunk: EngineOptions["onChunk"];
  readonly #tick = messageTick();
  readonly #yieldMs: number;
  /** When the engine last yielded (or started draining). */
  #yielded = 0;

  constructor(port: Port, options: EngineOptions) {
    this.#port = port;
    this.#debug = options.debug ?? false;
    this.#onChunk = options.onChunk;
    this.#yieldMs = options.yieldMs ?? YIELD_MS;
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
    const at = this.#queue.findIndex((x) => x !== END && keyOf(x) === key && isLastWins(x));
    if (at < 0 || !this.#superseded(this.#queue[at] as Request, key, at + 1)) {
      return false;
    }
    this.#deferred.set(req.field, req.desc);
    this.#fieldMeshes.set(req.field, [req.desc.on, req.desc.to]);
    return true;
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
      while (this.#queue.length > 0) {
        const item = this.#queue.shift() as Item;
        if (item === END) {
          if (!this.#finish()) {
            return;
          }
        } else if (!(await this.#step(item))) {
          return;
        }
        if (this.#queue.length > 0 && this.#queue[0] !== END) {
          await this.#maybeYield();
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
    try {
      if (req.kind === "repair" && this.#inSteps(req)) {
        await this.#repairInSteps(req, b);
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
      Array.isArray(req.payload) && req.payload.length > 1 && this.#drag === undefined && !this.#scene.canRedo()
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
   * A failed build forgets the field.
   */
  #buildField(id: ClientId, desc: FieldDescWire): FieldHandle {
    this.#deferred.delete(id);
    let h: FieldHandle;
    try {
      h = this.#scene.addDistanceField({ ...desc, on: this.#mesh(desc.on), to: this.#mesh(desc.to) });
    } catch (e) {
      this.#fieldMeshes.delete(id);
      throw e;
    }
    this.#fields.set(id, h);
    this.#fieldIds.set(h, id);
    this.#fieldMeshes.set(id, [desc.on, desc.to]);
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

  // ---- Requests ----

  #run(req: Request, b: Batch): void {
    const s = this.#scene;
    switch (req.kind) {
      case "addMesh": {
        if (this.#meshes.has(req.id)) {
          throw invalid(`mesh client id ${req.id} is already in use`);
        }
        const positions = req.f64 ? new Float64Array(req.positions) : new Float32Array(req.positions);
        const h = s.addMesh(positions, new Uint32Array(req.indices), { ...req.options, mode: "eager" });
        this.#meshes.set(req.id, h);
        this.#meshIds.set(h, req.id);
        return;
      }
      case "removeMesh": {
        const h = this.#mesh(req.id);
        s.removeMesh(h);
        this.#meshes.delete(req.id);
        // `#meshIds` keeps the handle until its removal record is sent.
        this.#removed.push(h);
        if (this.#drag?.id === req.id) {
          this.#drag = undefined;
        }
        // The scene drops the fields on or to the mesh, and its section entries.
        for (const [field, [on, to]] of this.#fieldMeshes) {
          if (on === req.id || to === req.id) {
            this.#dropField(field);
          }
        }
        for (const [section, meshes] of this.#sectionMeshes) {
          if (meshes.includes(req.id)) {
            this.#sectionMeshes.set(
              section,
              meshes.filter((m) => m !== req.id),
            );
            this.#goneEntries.push({ section, mesh: req.id });
          }
        }
        return;
      }
      case "compact":
        s.compact(this.#mesh(req.id));
        return;
      case "setPlacement":
        s.setPlacement(this.#mesh(req.id), {
          rotation: req.rotation,
          translation: req.translation as [number, number, number],
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
        return;
      case "redo":
        s.redo();
        return;
      case "clearHistory":
        if (this.#drag !== undefined) {
          // Closing the history ends the drag's step: end the drag first.
          this.#drag.session.end();
          this.#drag = undefined;
        }
        s.clearHistory();
        return;
      case "coloring":
        s.setColoring(this.#mesh(req.id), req.rule === null ? null : this.#rule(req.rule));
        return;
      case "field":
        this.#runField(req);
        return;
      case "section":
        this.#runSection(req);
        return;
      case "query":
        b.replies.push({ qid: req.qid, ok: true, value: this.#query(req.name, req.args) });
        return;
      case "fetchFieldValues": {
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
      return { rotation: req.rotation, translation: req.translation as [number, number, number] };
    };
    if (req.phase === "begin") {
      const session = this.#scene.drag(this.#mesh(req.id));
      this.#drag = { id: req.id, session };
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
        return;
      case "end":
        this.#drag = undefined;
        if (req.rotation !== undefined) {
          try {
            drag.session.move(pose());
          } finally {
            drag.session.end();
          }
          return;
        }
        drag.session.end();
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

  #runField(req: Extract<Request, { kind: "field" }>): void {
    if (req.op === "add") {
      if (this.#fields.has(req.field) || this.#deferred.has(req.field)) {
        throw invalid(`field client id ${req.field} is already in use`);
      }
      const desc = req.desc;
      if (desc === undefined) {
        throw invalid("adding a field needs a desc");
      }
      this.#buildField(req.field, desc);
      return;
    }
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
      case "fieldStatus": {
        const a = args as QueryMap["fieldStatus"]["args"];
        return s.fieldStatus(this.#field(a.field));
      }
      case "sectionGap": {
        const a = args as QueryMap["sectionGap"]["args"];
        return s.sectionGap(this.#section(a.section), this.#mesh(a.a), this.#mesh(a.b));
      }
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
    const frame = s.commitFrame();
    const changes: MeshChangeWire[] = [];
    const placements: Result["placements"] = [];
    const sections: Result["sections"] = [];
    for (const c of frame) {
      const id = this.#meshId(c.mesh);
      changes.push(this.#change(c, id, b));
      if (c.removed) {
        continue;
      }
      if (c.added || c.placementChanged) {
        const p = s.placement(c.mesh);
        placements.push({
          id,
          rotation: p.rotation,
          translation: p.translation,
          matrix4: Array.from(s.matrix4(c.mesh)),
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
    // A removed mesh's record carries nothing else: report its entries gone.
    for (const { section, mesh } of this.#goneEntries.splice(0)) {
      sections.push({ section, mesh, result: null });
    }
    const history = { canUndo: s.canUndo(), canRedo: s.canRedo() };
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
      status: b.status,
      replies: b.replies,
    };
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
      placementChanged: c.placementChanged,
      capacityChanged: c.capacityChanged,
      positions: c.positions,
      normals: c.normals,
      colors: c.colors,
      indices: c.indices,
      fields: c.fieldsChanged.map((f) => this.#fieldIds.get(f) as ClientId),
      sections: c.sectionsChanged.map((x) => this.#sectionIds.get(x) as ClientId),
      vertexSlots: 0,
      triangleSlots: 0,
      bytes: { positions: null, normals: null, colors: null, indices: null },
    };
    if (c.removed) {
      return wire;
    }
    wire.vertexSlots = this.#scene.vertexSlots(c.mesh);
    wire.triangleSlots = this.#scene.triangleSlots(c.mesh);
    // Fresh views, read with no scene call in between.
    const views: MeshViews = this.#scene.views(c.mesh);
    for (const name of Object.keys(STRIDE) as BufferName[]) {
      const view = views[name]();
      if (view !== null) {
        const bytes = copyRanges(view, c[name], STRIDE[name]);
        wire.bytes[name] = bytes;
        b.transfer.push(bytes);
      }
    }
    return wire;
  }
}
