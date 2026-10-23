// Messages between the main-thread `WorkerScene` and the Worker-side engine.
// Everything here survives structured clone; bulk data travels as transferred
// `ArrayBuffer`s. Meshes, fields and sections are named by client ids, which
// the main side allocates synchronously; the engine maps them to its handles.
import type { Vec3 } from "../api.js";
import type { Range } from "../changes.js";
import type {
  AddMeshOptions,
  AnalyzeOptions,
  ColorRule,
  DistanceFieldDesc,
  EditOp,
  RaycastOptions,
  RepairOp,
  SectionPlane,
  SectionResult,
} from "../scene.js";

/** A mesh, field or section id allocated on the main side. Unique per kind. */
export type ClientId = number;

/** A {@link ColorRule} naming its field by client id. */
export type ColorRuleWire = Omit<ColorRule, "field"> & { field: ClientId };

/** A {@link DistanceFieldDesc} naming its meshes by client id, its targets as a list. */
export type FieldDescWire = Omit<DistanceFieldDesc, "on" | "to"> & { on: ClientId; to: ClientId[] };

/**
 * A section description by client ids. For `op: "add"`, `plane` and `meshes`
 * are required; for `op: "set"`, each field present is applied (plane, then
 * meshes, then fill).
 */
export interface SectionDescWire {
  plane?: SectionPlane;
  meshes?: ClientId[];
  fill?: boolean;
}

/** A {@link SectionResult}; its typed arrays are transferred. */
export type SectionResultWire = SectionResult;

/**
 * A request. `rev` is the main side's revision, increasing per request.
 * `key` names the state the request targets (see {@link keyOf} for the
 * defaults); requests whose kind is "last wins" ({@link isLastWins}) are
 * dropped when a later queued request with the same key supersedes them.
 */
export type Request =
  | {
      kind: "addMesh";
      rev: number;
      id: ClientId;
      /** `Float64Array` bytes when `f64`, else `Float32Array` bytes: world positions. */
      positions: ArrayBuffer;
      f64: boolean;
      /** `Uint32Array` bytes. */
      indices: ArrayBuffer;
      /** `mode` is ignored: the Worker scene is always `"eager"`. */
      options: AddMeshOptions;
      key?: string;
    }
  | { kind: "removeMesh" | "compact"; rev: number; id: ClientId; key?: string }
  | { kind: "setPlacement"; rev: number; id: ClientId; rotation: number[]; translation: number[]; key: string }
  | {
      kind: "drag";
      rev: number;
      id: ClientId;
      phase: "begin" | "move" | "end" | "cancel";
      /** For `move`, and optionally `end` (a last move before ending). */
      rotation?: number[];
      translation?: number[];
      key: string;
    }
  | {
      kind: "edit" | "repair";
      rev: number;
      id: ClientId;
      /** `EditOp[]` for `edit`; `RepairOp` or `RepairOp[]` for `repair`. */
      payload: unknown;
      /** When set, the result (an `EditResult` or the repair report(s)) is replied under it. */
      qid?: number;
      key?: string;
    }
  | { kind: "undo" | "redo"; rev: number; key?: string }
  /** Drops every undo and redo step (the end of a `restart()` replay). */
  | { kind: "clearHistory"; rev: number; key?: string }
  | { kind: "coloring"; rev: number; id: ClientId; rule: ColorRuleWire | null; key: string }
  /**
   * Turns on the mirror of the colours' values (`views(mesh).values()`):
   * until it, results carry no value bytes. The next result then carries every
   * coloured mesh's colours and values whole.
   */
  | { kind: "values"; rev: number }
  /**
   * A field `add` may carry the key of the display it is built for (such as
   * `display:<meshId>`). When the request with that key it feeds (the next
   * queued last-wins one) is superseded before the build starts, the build is
   * abandoned: no status is reported, and the field is built when a later
   * request first needs it (a colouring, `fetchFieldValues`, `fieldStatus`).
   */
  | { kind: "field"; rev: number; field: ClientId; op: "add" | "remove"; desc?: FieldDescWire; key?: string }
  | {
      kind: "section";
      rev: number;
      section: ClientId;
      op: "add" | "set" | "remove";
      desc?: SectionDescWire;
      key?: string;
    }
  | { kind: "query"; rev: number; qid: number; name: string; args: unknown }
  | { kind: "fetchFieldValues"; rev: number; qid: number; field: ClientId };

/** The queries of `{ kind: "query" }`: their `args` and reply `value`, by name. */
export interface QueryMap {
  raycast: {
    args: { origin: Vec3; direction: Vec3; options?: RaycastOptions };
    value: { mesh: ClientId; triangle: number; point: Vec3; distance: number; barycentric: Vec3 } | null;
  };
  closestPoint: {
    args: { point: Vec3; options?: { maxDistance?: number; mesh?: ClientId } };
    value: { mesh: ClientId; triangle: number; point: Vec3; distance: number } | null;
  };
  meshDistance: {
    args: { a: ClientId; b: ClientId; options?: { maxDistance?: number } };
    value: { distance: number; a: Vec3; b: Vec3; triangleA: number; triangleB: number } | null;
  };
  intersects: { args: { a: ClientId; b: ClientId }; value: boolean };
  findContacts: {
    args: { margin?: number; pairs?: [ClientId, ClientId][] };
    value: {
      a: ClientId;
      b: ClientId;
      intersecting: boolean;
      distance: number;
      trianglesA: Uint32Array;
      trianglesB: Uint32Array;
    }[];
  };
  analyze: { args: { mesh: ClientId; options?: AnalyzeOptions }; value: unknown };
  fieldStatus: { args: { field: ClientId }; value: unknown };
  fieldNearest: { args: { field: ClientId }; value: unknown };
  fieldTargetStatus: { args: { field: ClientId }; value: unknown };
  sectionGap: {
    args: { section: ClientId; a: ClientId; b: ClientId };
    value: { distance: number; a: Vec3; b: Vec3 } | null;
  };
  /** Test-only (an engine run with `debug`): copies of a mesh's buffers. */
  debugDump: {
    args: { mesh: ClientId };
    value: {
      positions: Float32Array;
      normals: Float32Array | null;
      colors: Uint8Array | null;
      values: Float32Array | null;
      indices: Uint32Array;
      placement: { rotation: number[]; translation: Vec3 };
      matrix4: number[];
    };
  };
}

/** `EditOp[]`, the `payload` of an `edit` request. */
export type EditPayload = EditOp[];
/** The `payload` of a `repair` request. */
export type RepairPayload = RepairOp | RepairOp[];

/**
 * One mesh's change record (a `MeshChange` of the engine's scene) with
 * client ids and the bytes of its dirty ranges.
 *
 * `bytes` holds, per buffer, every dirty range's elements concatenated in
 * range order (positions and normals: 3 `f32` per vertex slot; colours: 4
 * bytes per vertex slot; values: 1 `f32` per vertex slot, over the colours'
 * ranges, once a `values` request turned them on; indices: 3 `u32` per triangle slot), or the whole buffer when the
 * ranges are `"all"`. `null` for a buffer that is off (normals, colours and
 * values), and for every buffer of a removed mesh.
 */
export interface MeshChangeWire {
  id: ClientId;
  added: boolean;
  removed: boolean;
  placementChanged: boolean;
  capacityChanged: boolean;
  positions: Range[] | "all";
  normals: Range[] | "all";
  colors: Range[] | "all";
  indices: Range[] | "all";
  fields: ClientId[];
  sections: ClientId[];
  /** Slot counts after the change; 0 for a removed mesh. */
  vertexSlots: number;
  triangleSlots: number;
  bytes: {
    positions: ArrayBuffer | null;
    normals: ArrayBuffer | null;
    colors: ArrayBuffer | null;
    values: ArrayBuffer | null;
    indices: ArrayBuffer | null;
  };
}

/** An error crossing the boundary: a `FastMeshError` code and message. */
export interface ErrorWire {
  code: string;
  message: string;
}

/**
 * What the engine posts after each processed batch: one per batch received,
 * in order (a batch whose requests were all dropped still gets one).
 */
export interface Result {
  /** The highest revision of the batch processed (or dropped as superseded or abandoned). */
  rev: number;
  /** Change records, ranges applied in order; never drop them. */
  changes: MeshChangeWire[];
  /** Placement and `matrix4` (column-major) of every mesh added or moved. */
  placements: { id: ClientId; rotation: number[]; translation: number[]; matrix4: number[] }[];
  /**
   * Every changed section entry. `result` is `null` once the entry is gone:
   * the section was removed, no longer cuts the mesh, or the mesh was
   * removed (one entry per section that cut it).
   */
  sections: { section: ClientId; mesh: ClientId; result: SectionResultWire | null }[];
  history: { canUndo: boolean; canRedo: boolean };
  /**
   * The outcome of every keyed request run in this batch, in order (a
   * request dropped as superseded, or an abandoned field build, has none).
   */
  status: { key: string; rev: number; status: "ready" | "error"; error?: ErrorWire }[];
  replies: { qid: number; ok: boolean; value?: unknown; error?: ErrorWire }[];
  /**
   * The colouring requests applied in this batch (mesh, revision), in order:
   * the colours and values in `changes` are made by these rules. A rejected
   * or superseded colouring has none. (Their `display:` statuses may come
   * later, with the batch that completes the fields.)
   */
  colorings: { id: ClientId; rev: number }[];
  /**
   * The engine stopped: the WebAssembly instance trapped, or packing the
   * result failed. The whole batch failed: its changes are lost, and every
   * status and reply in this result is an `INSTANCE_POISONED` error (the
   * message names the cause). Later batches fail the same way without
   * running. Recreate the Worker.
   */
  poisoned?: boolean;
}

/**
 * The first message to a Worker, before any batch: how to start the engine.
 * `threads` is the pool size asked for (`undefined`: the Worker's default,
 * see `defaultThreads`); `debug` asks for the development warnings.
 * `crashChannel` (threaded engines only) names the `BroadcastChannel` the
 * pool's threads report a crash on: the engine Worker is then blocked and
 * cannot, so the scene listens there itself (see {@link PoolCrash}).
 */
export interface InitMessage {
  kind: "init";
  threads?: number;
  debug: boolean;
  crashChannel?: string;
}

/**
 * What a pool thread posts on the init message's `crashChannel` when it
 * panics or traps (`detail`: the panic message, or the error). The thread
 * is gone and the engine Worker waits on it forever: the scene poisons.
 */
export interface PoolCrash {
  kind: "poolCrash";
  detail: string;
}

/** Whether a message to a Worker is its {@link InitMessage} (else a batch). */
export function isInit(data: unknown): data is InitMessage {
  return !Array.isArray(data) && (data as { kind?: unknown } | null)?.kind === "init";
}

/**
 * What a Worker posts: a {@link Result} per batch, after `{ kind: "ready" }`
 * once the engine runs, or `{ kind: "fatal" }` when it could not start. The
 * ready message reports the threads the engine computes with (1 when
 * single-threaded) and, in development, why it is single-threaded when more
 * were asked for (`warning`, for the main side to log once per scene).
 * `probeFailed` says this browser cannot run the threaded engine at all: the
 * scene then asks its later Workers, and the page's later scenes, for one
 * thread.
 * `{ kind: "fallback" }` replaces the ready message when the threaded pool
 * did not start: the Worker gives back the batches it received (later ones
 * follow, one message each) and expects to be replaced by a single-threaded
 * Worker, which gets them.
 */
export type WorkerMessage =
  | Result
  | { kind: "ready"; threads?: number; warning?: string; probeFailed?: true }
  | { kind: "fallback"; batches: Request[][]; warning?: string }
  | { kind: "fatal"; error: ErrorWire };

/**
 * The key of a request: its own `key`, or a default naming what it targets
 * (`mesh:<id>` for add and remove, `edit:<id>` for edits, repairs and
 * compaction, `field:<id>`, `section:<id>` for section add and remove,
 * `section:<id>:<aspects>` for a section `set`, such as `section:3:plane` or
 * `section:3:plane+fill`, `history`). Queries have none.
 */
export function keyOf(req: Request): string | undefined {
  if ("key" in req && req.key !== undefined) {
    return req.key;
  }
  switch (req.kind) {
    case "addMesh":
    case "removeMesh":
      return `mesh:${req.id}`;
    case "compact":
    case "edit":
    case "repair":
      return `edit:${req.id}`;
    case "undo":
    case "redo":
    case "clearHistory":
      return "history";
    case "field":
      return `field:${req.field}`;
    case "section":
      return req.op === "set"
        ? `section:${req.section}:${sectionAspects(req.desc).join("+") || "none"}`
        : `section:${req.section}`;
    default:
      return undefined;
  }
}

/** The aspects a section `set` changes, in application order. */
export function sectionAspects(desc: SectionDescWire | undefined): ("plane" | "meshes" | "fill")[] {
  return (["plane", "meshes", "fill"] as const).filter((a) => desc?.[a] !== undefined);
}

/**
 * Whether only the latest of consecutive same-key requests of this kind
 * matters: colouring, section plane/meshes/fill changes and drag moves. The
 * engine drops such a request only when the next queued request with the
 * same key is of the same kind, for the same target, and (for a section
 * `set`) changes every aspect the earlier one changed; so a reused key never
 * drops another target's or another aspect's change.
 */
export function isLastWins(req: Request): boolean {
  return (
    req.kind === "coloring" ||
    (req.kind === "section" && req.op === "set") ||
    (req.kind === "drag" && req.phase === "move")
  );
}

/**
 * Whether `later` makes the last-wins request `earlier` redundant: same kind,
 * same target (mesh or section) and, for a section `set`, every aspect of
 * `earlier` changed again. Keys are not compared here.
 */
export function supersedes(later: Request, earlier: Request): boolean {
  if (!isLastWins(earlier) || !isLastWins(later) || later.kind !== earlier.kind) {
    return false;
  }
  switch (earlier.kind) {
    case "coloring":
    case "drag":
      return (later as typeof earlier).id === earlier.id;
    case "section": {
      const l = later as typeof earlier;
      if (l.section !== earlier.section) {
        return false;
      }
      const changed = sectionAspects(l.desc);
      return sectionAspects(earlier.desc).every((a) => changed.includes(a));
    }
    default:
      return false;
  }
}
