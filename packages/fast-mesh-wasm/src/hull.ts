// Convex hulls: the types `Scene.convexHull` and `Scene.convexHullOfPoints`
// (and their `WorkerScene` twins) take and return, and the argument checks and
// decoding they share.
import type { Positions64 } from "./api.js";
import { FastMeshError } from "./core.js";
import type { MeshHandle } from "./scene.js";
import { cloudOrigin, type PackedPointCloud } from "./wave6.js";

function invalid(message: string): FastMeshError {
  return new FastMeshError("INVALID_ARGUMENT", message);
}

/** Options for `convexHull` and `convexHullOfPoints`. */
export interface ConvexHullOptions {
  /** Turn on the new mesh's smooth vertex normals, as `addMesh` does (default `true`). */
  normals?: boolean;
}

/**
 * Points for `convexHullOfPoints`: flat world positions `[x0, y0, z0, x1, ...]`
 * (`Float32Array`, `Float64Array` or numbers), or a {@link PackedPointCloud},
 * whose positions are used (its normals and colours are ignored), around its
 * `origin` if it has one. A bare
 * `Float32Array` is always read as flat positions, never as a packed cloud.
 */
export type ConvexHullPoints = Positions64 | PackedPointCloud;

/** What `convexHull` or `convexHullOfPoints` added. */
export interface ConvexHullResult {
  /** The new mesh holding the hull: closed, outward-wound, one undo step. */
  mesh: MeshHandle;
  /** Points the hull was built from; for meshes, the vertices their live triangles use. */
  inputPoints: number;
  /** Vertices of the hull: input points, including every corner. */
  vertices: number;
  /** Triangles of the hull. */
  triangles: number;
}

/** One mesh or several, as a non-empty list. */
export function hullMeshList(meshes: MeshHandle | readonly MeshHandle[]): readonly MeshHandle[] {
  const list = Array.isArray(meshes) ? (meshes as readonly MeshHandle[]) : [meshes as MeshHandle];
  if (list.length === 0) {
    throw invalid("convexHull needs at least one mesh");
  }
  return list;
}

/** The `normals` option, checked (default `true`). */
export function hullNormals(options: ConvexHullOptions | undefined): boolean {
  if (options !== undefined && (options === null || typeof options !== "object")) {
    throw invalid("options must be an object");
  }
  const normals = options?.normals ?? true;
  if (typeof normals !== "boolean") {
    throw invalid("options.normals must be a boolean");
  }
  return normals;
}

function isPackedCloud(points: ConvexHullPoints): points is PackedPointCloud {
  return (
    typeof points === "object" &&
    points !== null &&
    !ArrayBuffer.isView(points) &&
    "data" in points &&
    (points as PackedPointCloud).data instanceof Float32Array
  );
}

/**
 * The points as a fresh flat `Float64Array` (safe to transfer): every value a
 * finite number, the length a multiple of 3.
 */
export function hullPointsBuffer(points: ConvexHullPoints): Float64Array {
  if (points == null || typeof points !== "object") {
    throw invalid("points must be flat positions or a PackedPointCloud");
  }
  let flat: ArrayLike<number>;
  let origin: readonly number[] = [0, 0, 0];
  if (isPackedCloud(points)) {
    origin = cloudOrigin(points) ?? origin;
    const count = points.data[0];
    if (!Number.isInteger(count) || count < 0 || points.data.length < 3 + 3 * count) {
      throw invalid("packed point cloud buffer length mismatch");
    }
    flat = points.data.subarray(3, 3 + 3 * count);
  } else {
    flat = points as ArrayLike<number>;
  }
  if (typeof flat.length !== "number" || flat.length % 3 !== 0) {
    throw invalid("points length must be a multiple of 3");
  }
  const out = new Float64Array(flat.length);
  for (let i = 0; i < flat.length; i++) {
    const v = flat[i];
    if (typeof v !== "number" || !Number.isFinite(v)) {
      throw invalid(`points[${i}] must be a finite number`);
    }
    out[i] = v + origin[i % 3];
  }
  return out;
}

/** The counts of a binding's `[index, generation, inputPoints, vertices, triangles]`. */
export function decodeHullCounts(raw: ArrayLike<number>): Omit<ConvexHullResult, "mesh"> {
  return { inputPoints: raw[2], vertices: raw[3], triangles: raw[4] };
}
