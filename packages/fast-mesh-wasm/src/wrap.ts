// Wraps: the types `Scene.wrap` and `Scene.wrapPlan` (and their `WorkerScene`
// twins) take and return, and the argument checks and decoding they share.
import { FastMeshError } from "./core.js";
import type { MeshHandle } from "./scene.js";

/** The default `detail` of a wrap: voxels along the longest side of the meshes' bounds. */
export const DEFAULT_WRAP_DETAIL = 128;
/** The default `maxVoxels` of a wrap: 256 x 256 x 256. */
export const DEFAULT_WRAP_MAX_VOXELS = 16_777_216;

const MIN_DETAIL = 8;
const MAX_DETAIL = 4096;

function invalid(message: string): FastMeshError {
  return new FastMeshError("INVALID_ARGUMENT", message);
}

/** Options for `wrapPlan`: the grid's part of {@link WrapOptions}. */
export interface WrapGridOptions {
  /**
   * The widest gap, slot or hole that gets closed, in world units: openings
   * narrower than this are bridged (at least two voxels are). Finite and not
   * negative; default `0`.
   */
  gap?: number;
  /** The voxel size in world units. Without it, `detail` sets it. */
  voxelSize?: number;
  /**
   * Voxels along the longest side of the meshes' bounds when `voxelSize` is
   * not given: a whole number from 8 to 4096, default
   * {@link DEFAULT_WRAP_DETAIL}.
   */
  detail?: number;
  /**
   * The largest voxel grid accepted, a whole number of at least 64, default
   * {@link DEFAULT_WRAP_MAX_VOXELS}. A wrap over it is refused before
   * anything is allocated.
   */
  maxVoxels?: number;
}

/** Options for `wrap`. */
export interface WrapOptions extends WrapGridOptions {
  /**
   * Move the wrap's vertices that lie within one voxel of the meshes onto
   * them, wherever that keeps the surface free of self-intersections.
   * Recovers detail and sharp features the voxels round off. Default `false`.
   */
  project?: boolean;
  /** Turn on the new mesh's smooth vertex normals, as `addMesh` does (default `true`). */
  normals?: boolean;
}

/** The voxel grid of a wrap, known before it runs. */
export interface WrapPlan {
  /** Grid nodes along x, y and z. */
  dims: [number, number, number];
  /** The voxel size in world units. */
  voxelSize: number;
  /** The closing radius in world units: half the gap, at least one voxel. */
  radius: number;
  /** Grid nodes in all. */
  voxels: number;
  /**
   * The estimated peak memory of the grid, in bytes. The meshes, what is
   * extracted from the grid and their search trees come on top.
   */
  bytes: number;
  /** The limit the plan was checked against. */
  maxVoxels: number;
  /** Whether `voxels <= maxVoxels`; `wrap` refuses the plan otherwise. */
  fits: boolean;
}

/** What `wrap` added. */
export interface WrapResult {
  /** The new mesh holding the wrap: closed, manifold, outward-wound, one undo step. */
  mesh: MeshHandle;
  /** Vertices of the wrap. */
  vertices: number;
  /** Triangles of the wrap. */
  triangles: number;
  /** Vertices of the wrap moved onto the meshes (`0` without `project`). */
  projected: number;
  /** The voxel grid used. */
  plan: WrapPlan;
}

/** Internal: {@link WrapOptions} with every default filled in. */
export type CheckedWrapOptions = Required<Omit<WrapOptions, "voxelSize">> & Pick<WrapOptions, "voxelSize">;

/** Internal: one mesh or several, as a non-empty list. */
export function wrapMeshList(meshes: MeshHandle | readonly MeshHandle[]): readonly MeshHandle[] {
  const list = Array.isArray(meshes) ? (meshes as readonly MeshHandle[]) : [meshes as MeshHandle];
  if (list.length === 0) {
    throw invalid("wrap needs at least one mesh");
  }
  return list;
}

function flag(value: unknown, name: string, fallback: boolean): boolean {
  const v = value ?? fallback;
  if (typeof v !== "boolean") {
    throw invalid(`options.${name} must be a boolean`);
  }
  return v;
}

/** Internal: the options with their defaults, checked; a plain object, safe to post to a Worker. */
export function checkWrapOptions(options: WrapOptions | undefined): CheckedWrapOptions {
  if (options !== undefined && (options === null || typeof options !== "object")) {
    throw invalid("options must be an object");
  }
  const gap = options?.gap ?? 0;
  if (typeof gap !== "number" || !Number.isFinite(gap) || gap < 0) {
    throw invalid("options.gap must be a finite number and not negative");
  }
  const voxelSize = options?.voxelSize;
  if (voxelSize !== undefined && (typeof voxelSize !== "number" || !Number.isFinite(voxelSize) || voxelSize <= 0)) {
    throw invalid("options.voxelSize must be a positive finite number");
  }
  const detail = options?.detail ?? DEFAULT_WRAP_DETAIL;
  if (!Number.isInteger(detail) || detail < MIN_DETAIL || detail > MAX_DETAIL) {
    throw invalid(`options.detail must be a whole number from ${MIN_DETAIL} to ${MAX_DETAIL}`);
  }
  const maxVoxels = options?.maxVoxels ?? DEFAULT_WRAP_MAX_VOXELS;
  if (!Number.isInteger(maxVoxels) || maxVoxels < 64 || maxVoxels > 0xffff_ffff) {
    throw invalid("options.maxVoxels must be a whole number from 64 to 4294967295");
  }
  const project = flag(options?.project, "project", false);
  const normals = flag(options?.normals, "normals", true);
  return { gap, ...(voxelSize === undefined ? {} : { voxelSize }), detail, project, maxVoxels, normals };
}

/** Internal: checked options as the binding takes them. */
export function encodeWrapOptions(o: CheckedWrapOptions): Float64Array {
  return Float64Array.of(o.gap, o.voxelSize ?? 0, o.detail, o.project ? 1 : 0, o.maxVoxels);
}

/** Internal: a plan from the binding's `[nx, ny, nz, voxelSize, radius, voxels, bytes, maxVoxels, fits]`. */
export function decodeWrapPlan(raw: ArrayLike<number>, at = 0): WrapPlan {
  return {
    dims: [raw[at], raw[at + 1], raw[at + 2]],
    voxelSize: raw[at + 3],
    radius: raw[at + 4],
    voxels: raw[at + 5],
    bytes: raw[at + 6],
    maxVoxels: raw[at + 7],
    fits: raw[at + 8] !== 0,
  };
}

/** Internal: the counts and plan of a binding's `[index, generation, vertices, triangles, projected, ...plan]`. */
export function decodeWrapCounts(raw: ArrayLike<number>): Omit<WrapResult, "mesh"> {
  return { vertices: raw[2], triangles: raw[3], projected: raw[4], plan: decodeWrapPlan(raw, 5) };
}
