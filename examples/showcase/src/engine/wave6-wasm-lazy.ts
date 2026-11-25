// Pure-TS point-cloud buffer pack/unpack helpers + lazy Wasm loader for Showcase.
// Keeping this wrapper free of top-level `"fast-mesh-wasm"` value imports prevents
// eager Wasm instantiation on boot in Worker mode and in jsdom component tests.
import type { PackedPointCloud, Positions, Vec3 } from "fast-mesh-wasm";

function toF32(src: Positions, label: string, origin?: Readonly<Vec3>): Float32Array {
  if (src instanceof Float32Array && origin === undefined) {
    return src;
  }
  const out = new Float32Array(src.length);
  for (let i = 0; i < src.length; i++) {
    const v = src[i];
    if (typeof v !== "number" || !Number.isFinite(v)) {
      throw new RangeError(`${label}[${i}] must be a finite number`);
    }
    out[i] = origin === undefined ? v : v - origin[i % 3];
  }
  return out;
}

/**
 * Packs position (and optional normal/color) buffers into a {@link PackedPointCloud} without loading Wasm.
 * With `origin`, the (world) positions are stored relative to it, as the SDK's `packPointCloud` does.
 */
export function packPointCloudLocal(
  positions: Positions,
  normals?: Positions,
  colors?: Positions,
  origin?: Readonly<Vec3>,
): PackedPointCloud {
  const pos = toF32(positions, "positions", origin);
  if (pos.length % 3 !== 0) {
    throw new RangeError("positions length must be a multiple of 3");
  }
  const count = pos.length / 3;
  const norm = normals === undefined ? undefined : toF32(normals, "normals");
  const col = colors === undefined ? undefined : toF32(colors, "colors");
  if (norm !== undefined && norm.length !== pos.length) {
    throw new RangeError("normals length must match positions length");
  }
  if (col !== undefined && col.length !== pos.length) {
    throw new RangeError("colors length must match positions length");
  }
  const hasNormals = norm !== undefined;
  const hasColors = col !== undefined;
  const data = new Float32Array(3 + count * (3 + (hasNormals ? 3 : 0) + (hasColors ? 3 : 0)));
  data[0] = count;
  data[1] = hasNormals ? 1 : 0;
  data[2] = hasColors ? 1 : 0;
  let offset = 3;
  data.set(pos, offset);
  offset += pos.length;
  if (norm !== undefined) {
    data.set(norm, offset);
    offset += norm.length;
  }
  if (col !== undefined) {
    data.set(col, offset);
  }
  return { data, count, hasNormals, hasColors, ...(origin === undefined ? {} : { origin }) };
}

/** Unpacks a {@link PackedPointCloud} into separate TypedArrays without loading Wasm. Positions are relative to `origin`. */
export function unpackPointCloudLocal(cloud: PackedPointCloud): {
  count: number;
  positions: Float32Array;
  normals: Float32Array | null;
  colors: Float32Array | null;
  origin: Readonly<Vec3>;
} {
  const buf = cloud.data;
  const count = Math.max(0, Math.round(buf[0] ?? 0));
  const hasNormals = (buf[1] ?? 0) !== 0;
  const hasColors = (buf[2] ?? 0) !== 0;
  let at = 3;
  const positions = Float32Array.from(buf.subarray(at, at + count * 3));
  at += count * 3;
  const normals = hasNormals ? Float32Array.from(buf.subarray(at, at + count * 3)) : null;
  if (hasNormals) {
    at += count * 3;
  }
  const colors = hasColors ? Float32Array.from(buf.subarray(at, at + count * 3)) : null;
  return { count, positions, normals, colors, origin: cloud.origin ?? [0, 0, 0] };
}

type FastMeshModule = typeof import("fast-mesh-wasm");
let wasmPromise: Promise<FastMeshModule> | null = null;

/** Lazily loads and initializes the standalone `fast-mesh-wasm` module when invoked by user actions. */
export function ensureWave6Wasm(): Promise<FastMeshModule> {
  if (wasmPromise === null) {
    wasmPromise = import("fast-mesh-wasm").then(async (mod) => {
      await mod.init();
      return mod;
    });
  }
  return wasmPromise;
}
