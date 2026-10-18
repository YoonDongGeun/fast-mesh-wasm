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

/** A {@link DistanceFieldDesc} naming its meshes by client id. */
export type FieldDescWire = Omit<DistanceFieldDesc, "on" | "to"> & { on: ClientId; to: ClientId };

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
  | { kind: "coloring"; rev: number; id: ClientId; rule: ColorRuleWire | null; key: string }
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
  sectionGap: {
    args: { section: ClientId; a: ClientId; b: ClientId };
    value: { distance: number; a: Vec3; b: Vec3 } | null;
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
 * bytes per vertex slot; indices: 3 `u32` per triangle slot), or the whole
 * buffer when the ranges are `"all"`. `null` for a buffer that is off
 * (normals or colours), and for every buffer of a removed mesh.
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
    indices: ArrayBuffer | null;
  };
}

/** An error crossing the boundary: a `FastMeshError` code and message. */
export interface ErrorWire {
  code: string;
  message: string;
}

/** What the engine posts after each processed batch. */
export interface Result {
  /** The highest revision processed (or dropped as superseded). */
  rev: number;
  /** Change records, ranges applied in order; never drop them. */
  changes: MeshChangeWire[];
  /** Placement and `matrix4` (column-major) of every mesh added or moved. */
  placements: { id: ClientId; rotation: number[]; translation: number[]; matrix4: number[] }[];
  /** Every changed section entry; `result` is `null` once the section or the entry is gone. */
  sections: { section: ClientId; mesh: ClientId; result: SectionResultWire | null }[];
  history: { canUndo: boolean; canRedo: boolean };
  /**
   * The outcome of every keyed request run in this batch, in order (a
   * request dropped as superseded has none).
   */
  status: { key: string; rev: number; status: "ready" | "error"; error?: ErrorWire }[];
  replies: { qid: number; ok: boolean; value?: unknown; error?: ErrorWire }[];
  /**
   * The WebAssembly instance trapped: every call from now on fails with
   * `INSTANCE_POISONED`, and the changes of this batch are lost. Recreate the
   * Worker.
   */
  poisoned?: boolean;
}

/**
 * The key of a request: its own `key`, or a default naming what it targets
 * (`mesh:<id>` for add and remove, `edit:<id>` for edits, repairs and
 * compaction, `field:<id>`, `section:<id>`, `history`). Queries have none.
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
      return "history";
    case "field":
      return `field:${req.field}`;
    case "section":
      return `section:${req.section}`;
    default:
      return undefined;
  }
}

/**
 * Whether only the latest of consecutive same-key requests of this kind
 * matters: colouring, section plane/meshes/fill changes and drag moves. Give
 * section changes of different aspects (plane, meshes, fill) different keys.
 */
export function isLastWins(req: Request): boolean {
  return (
    req.kind === "coloring" ||
    (req.kind === "section" && req.op === "set") ||
    (req.kind === "drag" && req.phase === "move")
  );
}
