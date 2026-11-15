// Standalone WebAssembly helpers for Wave 6 contour export and point-cloud processing.
// Kept separate from `wave6.ts` so `WorkerScene`, `domains/*`, and `store/*` stay Wasm-free.
import type { Vec3 } from "./api.js";
import { call, ensureInitialized, FastMeshError } from "./core.js";
import { checkU32Int, codeIn, normalizeVec3 } from "./options.js";
import * as wasm from "./wasm/fast_mesh_wasm.js";
import {
  checkPositiveArg,
  cloudBuffer,
  CONTOUR_EXPORT_CODES,
  type ContourExportFormat,
  type ContourSlice,
  encodeContourSlices,
  type PackedPointCloud,
  POINT_CLOUD_FORMAT_CODES,
  type PointCloudFilterResult,
  type PointCloudFormat,
  wrapPackedPointCloud,
} from "./wave6.js";

function invalid(message: string): FastMeshError {
  return new FastMeshError("INVALID_ARGUMENT", message);
}

function decodeOutlierResult(raw: Float64Array): PointCloudFilterResult {
  return {
    report: {
      keptCount: raw[0],
      removedCount: raw[1],
      meanDistance: raw[2],
      stdDev: raw[3],
      threshold: raw[4],
    },
    cloud: wrapPackedPointCloud(Float32Array.from(raw.subarray(5))),
  };
}

/** Serializes contour slices to SVG, JSON, or CSV. */
export function sliceContoursExport(
  slices: ContourSlice | readonly ContourSlice[],
  format: ContourExportFormat,
): string {
  ensureInitialized("sliceContoursExport()");
  const list = Array.isArray(slices) ? slices : [slices];
  const packed = encodeContourSlices(list);
  const code = codeIn(CONTOUR_EXPORT_CODES, format, "format");
  return call(() => wasm.sliceContoursExport(packed, code));
}

/** Parses ASCII point cloud text (`"xyz"`, `"pcd"`, or `"ply"`). */
export function parsePointCloudText(text: string, format: PointCloudFormat = "xyz"): PackedPointCloud {
  ensureInitialized("parsePointCloudText()");
  if (typeof text !== "string") {
    throw invalid("text must be a string");
  }
  const code = codeIn(POINT_CLOUD_FORMAT_CODES, format, "format");
  return wrapPackedPointCloud(call(() => wasm.parsePointCloudText(text, code)));
}

/** Parses binary/ASCII `.ply` point cloud bytes. */
export function parsePointCloudBytes(bytes: Uint8Array): PackedPointCloud {
  ensureInitialized("parsePointCloudBytes()");
  if (!(bytes instanceof Uint8Array)) {
    throw invalid("bytes must be a Uint8Array");
  }
  return wrapPackedPointCloud(call(() => wasm.parsePointCloudBytes(bytes)));
}

/** Runs Statistical Outlier Removal (SOR) on a point cloud. */
export function pointCloudSor(
  cloud: PackedPointCloud | Float32Array,
  kNeighbors = 16,
  stdRatio = 2.0,
): PointCloudFilterResult {
  ensureInitialized("pointCloudSor()");
  const raw = call(() =>
    wasm.pointCloudSor(
      cloudBuffer(cloud),
      checkU32Int(kNeighbors, "kNeighbors"),
      checkPositiveArg(stdRatio, "stdRatio"),
    ),
  );
  return decodeOutlierResult(raw);
}

/** Runs Radius Outlier Removal (ROR) on a point cloud. */
export function pointCloudRor(
  cloud: PackedPointCloud | Float32Array,
  radius: number,
  minNeighbors = 4,
): PointCloudFilterResult {
  ensureInitialized("pointCloudRor()");
  const raw = call(() =>
    wasm.pointCloudRor(
      cloudBuffer(cloud),
      checkPositiveArg(radius, "radius"),
      checkU32Int(minNeighbors, "minNeighbors"),
    ),
  );
  return decodeOutlierResult(raw);
}

/** Voxel-grid downsamples a point cloud. */
export function pointCloudVoxelDownsample(
  cloud: PackedPointCloud | Float32Array,
  voxelSize: number,
): PackedPointCloud {
  ensureInitialized("pointCloudVoxelDownsample()");
  return wrapPackedPointCloud(
    call(() => wasm.pointCloudVoxelDownsample(cloudBuffer(cloud), checkPositiveArg(voxelSize, "voxelSize"))),
  );
}

/** Estimates and orients normals on a point cloud. */
export function pointCloudEstimateNormals(
  cloud: PackedPointCloud | Float32Array,
  kNeighbors = 12,
  cameraHint?: Readonly<Vec3>,
): PackedPointCloud {
  ensureInitialized("pointCloudEstimateNormals()");
  const hint = cameraHint === undefined ? undefined : Float64Array.from(normalizeVec3(cameraHint, "cameraHint"));
  return wrapPackedPointCloud(
    call(() => wasm.pointCloudEstimateNormals(cloudBuffer(cloud), checkU32Int(kNeighbors, "kNeighbors"), hint)),
  );
}
