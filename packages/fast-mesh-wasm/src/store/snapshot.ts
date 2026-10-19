// The store's observable state: immutable, frozen snapshots whose unchanged
// parts are shared between versions, so `===` tells a selector whether its
// slice changed. Types only from the scene modules: no WebAssembly here.
import type { FastMeshError } from "../core.js";
import type {
  Colormap,
  MeshHandle,
  OutOfRange,
  Rgba,
  SectionHandle,
  SectionPlane,
  SignMode,
} from "../scene.js";

/**
 * The state of a mesh's or section's work: `"computing"` while it, or a
 * result it depends on, is pending; `"ready"` once applied; `"error"` when
 * the latest request failed; `"idle"` for nothing requested (or removed).
 */
export type Status = "idle" | "computing" | "ready" | "error";

/**
 * Declarative distance colouring of a mesh: its vertices coloured by their
 * signed distance to `distanceTo`. The store owns the distance field behind
 * it. The colour fields are those of the scene's `ColorRule`.
 */
export interface DistanceColoring {
  /** The mesh the distances are measured to. */
  distanceTo: MeshHandle;
  /** Largest distance computed (inclusive); may be `Infinity`. */
  maxDistance: number;
  /** Defaults to `"auto"`. */
  sign?: SignMode;
  map: Colormap;
  /** `[lo, hi]` mapped onto the colormap; finite, `lo < hi`. */
  range: readonly [lo: number, hi: number];
  below?: OutOfRange;
  above?: OutOfRange;
  center?: number;
  base?: Rgba;
  nan?: Rgba;
}

/** How a mesh is shown. `opacity` and `visible` live only in the store. */
export interface DisplayState {
  /** Distance colouring, or `null` for none. */
  coloring: DistanceColoring | null;
  /** In [0, 1]; defaults to 1. */
  opacity: number;
  /** Defaults to `true`. */
  visible: boolean;
}

/** A mesh in a {@link SceneSnapshot}. Frozen. */
export interface MeshState {
  /** What is applied (shown). */
  display: DisplayState;
  /** What was last requested; equals `display` once applied. */
  desired: DisplayState;
  status: Status;
  /** The snapshot revision at which this state last changed. */
  revision: number;
  error?: FastMeshError;
}

/** The declarative parts of a section. */
export interface SectionParts {
  plane: SectionPlane;
  meshes: readonly MeshHandle[];
  fill: boolean;
}

/** A section in a {@link SceneSnapshot}. Frozen. */
export interface SectionState extends SectionParts {
  /** What was last requested; the top-level fields are what is applied. */
  desired: SectionParts;
  status: Status;
  /** The snapshot revision at which this state last changed. */
  revision: number;
  error?: FastMeshError;
}

/** An immutable view of a store's scene; replaced (never mutated) on change. */
export interface SceneSnapshot {
  /** Increases with every new snapshot. */
  revision: number;
  meshes: ReadonlyMap<MeshHandle, MeshState>;
  history: { canUndo: boolean; canRedo: boolean };
  sections: ReadonlyMap<SectionHandle, SectionState>;
  selection: MeshHandle | null;
  /** `"error"` when any mesh, section or the history has an error, else `"computing"` while any is pending. */
  status: "idle" | "computing" | "error";
  /** The first error, when `status` is `"error"`. */
  error?: FastMeshError;
}

/** A plain-data value deep-copied and deep-frozen; handles and other class instances are kept as they are. */
export function frozenCopy<T>(value: T): T {
  if (typeof value !== "object" || value === null) {
    return value;
  }
  if (Array.isArray(value)) {
    return Object.freeze(value.map((v) => frozenCopy(v))) as T;
  }
  if (Object.getPrototypeOf(value) !== Object.prototype) {
    return value;
  }
  const out: Record<string, unknown> = {};
  for (const [k, v] of Object.entries(value)) {
    if (v !== undefined) {
      out[k] = frozenCopy(v);
    }
  }
  return Object.freeze(out) as T;
}

/** Deep equality of plain data (arrays, plain objects, primitives); class instances compare by identity. */
export function sameData(a: unknown, b: unknown): boolean {
  if (Object.is(a, b)) {
    return true;
  }
  if (typeof a !== "object" || typeof b !== "object" || a === null || b === null) {
    return false;
  }
  if (Array.isArray(a) || Array.isArray(b)) {
    if (!Array.isArray(a) || !Array.isArray(b) || a.length !== b.length) {
      return false;
    }
    return a.every((v, k) => sameData(v, b[k]));
  }
  if (Object.getPrototypeOf(a) !== Object.prototype || Object.getPrototypeOf(b) !== Object.prototype) {
    return false;
  }
  const ka = Object.keys(a).filter((k) => (a as Record<string, unknown>)[k] !== undefined);
  const kb = Object.keys(b).filter((k) => (b as Record<string, unknown>)[k] !== undefined);
  if (ka.length !== kb.length) {
    return false;
  }
  return ka.every((k) => sameData((a as Record<string, unknown>)[k], (b as Record<string, unknown>)[k]));
}

/** The default display: no colouring, opaque, visible. */
export const DEFAULT_DISPLAY: DisplayState = Object.freeze({ coloring: null, opacity: 1, visible: true });

/** The state of a mesh not in a snapshot (removed, or never added): `"idle"`, default display. */
export const REMOVED_MESH_STATE: MeshState = Object.freeze({
  display: DEFAULT_DISPLAY,
  desired: DEFAULT_DISPLAY,
  status: "idle",
  revision: 0,
});

/** The snapshot of an empty scene; what the server renders (SSR). */
export const EMPTY_SNAPSHOT: SceneSnapshot = Object.freeze({
  revision: 0,
  meshes: Object.freeze(new Map<MeshHandle, MeshState>()),
  history: Object.freeze({ canUndo: false, canRedo: false }),
  sections: Object.freeze(new Map<SectionHandle, SectionState>()),
  selection: null,
  status: "idle",
});

/** A mesh's state in `snap`, or {@link REMOVED_MESH_STATE} when it is not there. */
export function meshState(snap: SceneSnapshot, mesh: MeshHandle): MeshState {
  return snap.meshes.get(mesh) ?? REMOVED_MESH_STATE;
}

/** A status with its error. */
export interface Combined {
  status: Status;
  error?: FastMeshError;
}

/**
 * Combines statuses: `"error"` wins over `"computing"`, which wins over
 * `"ready"`, which wins over `"idle"`. The first error is kept.
 */
export function combine(parts: Iterable<Combined>): Combined {
  let out: Combined = { status: "idle" };
  const rank = { idle: 0, ready: 1, computing: 2, error: 3 } as const;
  for (const p of parts) {
    if (rank[p.status] > rank[out.status]) {
      out = p.status === "error" ? { status: "error", error: p.error } : { status: p.status };
    }
  }
  return out;
}
