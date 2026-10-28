// The VTK sample reader: vtk.js-format volume descriptors (`index.json`
// alongside the scalar blob) into float volumes, resampled axis-aligned,
// then stacked into axial slices for the DICOM pipeline.
import type { DicomSlice } from "./dicom";

/** The bundled sample CTs (head + chest). */
export type SampleKind = "headsq" | "lidc2";

export const SAMPLE_NAMES: Record<SampleKind, string> = {
  headsq: "CT head",
  lidc2: "CT chest",
};

/** A parsed vtk.js volume descriptor (dims, mm geometry, scalar ref). */
export interface VtkJsImage {
  dims: [number, number, number];
  spacing: [number, number, number];
  origin: [number, number, number];
  /** Row-major 3x3 direction cosines, or null when axis-aligned. */
  direction: number[] | null;
  dataType: "int16" | "uint8";
  refId: string;
}

function isNumberArray(v: unknown, length: number): v is number[] {
  return Array.isArray(v) && v.length === length && v.every((n) => typeof n === "number" && Number.isFinite(n));
}

function fail(): never {
  throw new Error("not a vtk.js image descriptor");
}

/** Reads a vtk.js `index.json` descriptor (throws on garbage). */
export function parseVtkJsImage(descriptor: unknown): VtkJsImage {
  if (typeof descriptor !== "object" || descriptor === null) fail();
  const d = descriptor as Record<string, unknown>;
  const origin = d["origin"];
  if (!isNumberArray(origin, 3)) fail();
  const spacing = d["spacing"];
  if (!isNumberArray(spacing, 3)) fail();
  const extent = d["extent"];
  if (!isNumberArray(extent, 6)) fail();
  if (!extent.every(Number.isInteger)) fail();
  const dims: [number, number, number] = [
    extent[1] - extent[0] + 1,
    extent[3] - extent[2] + 1,
    extent[5] - extent[4] + 1,
  ];
  if (dims.some((n) => n <= 0)) fail();
  const pointData = d["pointData"] as { arrays?: { data?: Record<string, unknown> }[] } | undefined;
  const data = pointData?.arrays?.[0]?.data;
  if (typeof data !== "object" || data === null) fail();
  const dataType = (data as Record<string, unknown>)["dataType"];
  if (dataType !== "Int16Array" && dataType !== "Uint8Array") fail();
  const size = (data as Record<string, unknown>)["size"];
  if (size !== dims[0] * dims[1] * dims[2]) fail();
  const ref = (data as Record<string, unknown>)["ref"] as { id?: unknown } | undefined;
  if (typeof ref?.id !== "string") fail();
  const direction = d["direction"];
  if (direction !== undefined && !isNumberArray(direction, 9)) fail();
  return {
    dims,
    spacing: [spacing[0], spacing[1], spacing[2]],
    origin: [origin[0], origin[1], origin[2]],
    direction:
      direction === undefined || isIdentity(direction as number[]) ? null : (direction as number[]).slice(),
    dataType: dataType === "Int16Array" ? "int16" : "uint8",
    refId: ref!.id as string,
  };
}

function isIdentity(d: number[]): boolean {
  for (let r = 0; r < 3; r++) {
    for (let c = 0; c < 3; c++) {
      if (d[r * 3 + c] !== (r === c ? 1 : 0)) return false;
    }
  }
  return true;
}

/** Reads little-endian scalars into a float volume. */
export function toFloatVolume(bytes: Uint8Array, dataType: "int16" | "uint8"): Float32Array {
  if (dataType === "uint8") return Float32Array.from(bytes);
  if (bytes.byteLength % 2 !== 0) throw new Error("odd-length int16 scalars");
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const out = new Float32Array(bytes.byteLength / 2);
  for (let i = 0; i < out.length; i++) out[i] = view.getInt16(i * 2, true);
  return out;
}

export interface AlignedVolume {
  values: Float32Array;
  dims: [number, number, number];
  spacing: [number, number, number];
  origin: [number, number, number];
}

function invert3(m: number[]): number[] {
  const [a, b, c, d, e, f, g, h, i] = m;
  const det = a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g);
  if (det === 0) throw new Error("singular volume direction");
  const s = 1 / det;
  return [
    (e * i - f * h) * s,
    (c * h - b * i) * s,
    (b * f - c * e) * s,
    (f * g - d * i) * s,
    (a * i - c * g) * s,
    (c * d - a * f) * s,
    (d * h - e * g) * s,
    (b * g - a * h) * s,
    (a * e - b * d) * s,
  ];
}

/**
 * Resamples a rotated volume onto its axis-aligned bounding box
 * (trilinear, clamped; null direction passes through untouched).
 */
export function alignVolume(
  values: Float32Array,
  dims: readonly [number, number, number],
  spacing: readonly [number, number, number],
  origin: readonly [number, number, number],
  direction: number[] | null,
): AlignedVolume {
  if (direction === null) {
    return {
      values,
      dims: [dims[0], dims[1], dims[2]],
      spacing: [spacing[0], spacing[1], spacing[2]],
      origin: [origin[0], origin[1], origin[2]],
    };
  }
  const [nx, ny, nz] = dims;
  // The rotated corners bound the output grid.
  const min = [Infinity, Infinity, Infinity];
  const max = [-Infinity, -Infinity, -Infinity];
  for (const cx of [0, nx - 1]) {
    for (const cy of [0, ny - 1]) {
      for (const cz of [0, nz - 1]) {
        for (let r = 0; r < 3; r++) {
          const w =
            origin[r] +
            direction[r * 3] * cx * spacing[0] +
            direction[r * 3 + 1] * cy * spacing[1] +
            direction[r * 3 + 2] * cz * spacing[2];
          if (w < min[r]) min[r] = w;
          if (w > max[r]) max[r] = w;
        }
      }
    }
  }
  const inv = invert3(direction);
  const outDims: [number, number, number] = [0, 1, 2].map(
    (r) => Math.round((max[r] - min[r]) / spacing[r]) + 1,
  ) as [number, number, number];
  const out = new Float32Array(outDims[0] * outDims[1] * outDims[2]);
  const at = (x: number, y: number, z: number): number => {
    const cx = Math.min(nx - 1, Math.max(0, x));
    const cy = Math.min(ny - 1, Math.max(0, y));
    const cz = Math.min(nz - 1, Math.max(0, z));
    const x0 = Math.floor(cx);
    const y0 = Math.floor(cy);
    const z0 = Math.floor(cz);
    const x1 = Math.min(nx - 1, x0 + 1);
    const y1 = Math.min(ny - 1, y0 + 1);
    const z1 = Math.min(nz - 1, z0 + 1);
    const fx = cx - x0;
    const fy = cy - y0;
    const fz = cz - z0;
    const idx = (x: number, y: number, z: number): number => values[(z * ny + y) * nx + x];
    return (
      idx(x0, y0, z0) * (1 - fx) * (1 - fy) * (1 - fz) +
      idx(x1, y0, z0) * fx * (1 - fy) * (1 - fz) +
      idx(x0, y1, z0) * (1 - fx) * fy * (1 - fz) +
      idx(x1, y1, z0) * fx * fy * (1 - fz) +
      idx(x0, y0, z1) * (1 - fx) * (1 - fy) * fz +
      idx(x1, y0, z1) * fx * (1 - fy) * fz +
      idx(x0, y1, z1) * (1 - fx) * fy * fz +
      idx(x1, y1, z1) * fx * fy * fz
    );
  };
  for (let z = 0; z < outDims[2]; z++) {
    for (let y = 0; y < outDims[1]; y++) {
      for (let x = 0; x < outDims[0]; x++) {
        // The output voxel's world point, back in input voxel units.
        const px = min[0] + x * spacing[0] - origin[0];
        const py = min[1] + y * spacing[1] - origin[1];
        const pz = min[2] + z * spacing[2] - origin[2];
        out[(z * outDims[1] + y) * outDims[0] + x] = at(
          (inv[0] * px + inv[1] * py + inv[2] * pz) / spacing[0],
          (inv[3] * px + inv[4] * py + inv[5] * pz) / spacing[1],
          (inv[6] * px + inv[7] * py + inv[8] * pz) / spacing[2],
        );
      }
    }
  }
  return {
    values: out,
    dims: outDims,
    spacing: [spacing[0], spacing[1], spacing[2]],
    origin: [min[0], min[1], min[2]],
  };
}

/**
 * A sample's HU mapping, series identity, mesh isosurface and suggested
 * volume preset (the descriptor's extras; a null iso takes Otsu's
 * threshold, a missing preset takes Bone).
 */
export interface SampleMeta {
  slope: number;
  intercept: number;
  seriesUid: string;
  description: string;
  iso: number | null;
  preset: string | null;
}

/** Reads a sample descriptor's HU mapping, series identity, iso and preset. */
export function parseSampleMeta(descriptor: unknown): SampleMeta {
  if (typeof descriptor !== "object" || descriptor === null) fail();
  const d = descriptor as Record<string, unknown>;
  const hu = d["hu"] as { slope?: unknown; intercept?: unknown } | undefined;
  if (typeof hu?.slope !== "number" || typeof hu?.intercept !== "number") fail();
  if (typeof d["seriesUid"] !== "string" || typeof d["description"] !== "string") fail();
  if (d["iso"] !== null && typeof d["iso"] !== "number") fail();
  if (d["preset"] !== undefined && d["preset"] !== null && typeof d["preset"] !== "string") fail();
  return {
    slope: hu!.slope as number,
    intercept: hu!.intercept as number,
    seriesUid: d["seriesUid"] as string,
    description: d["description"] as string,
    iso: d["iso"] as number | null,
    preset: (d["preset"] as string | null | undefined) ?? null,
  };
}

async function gunzip(bytes: Uint8Array): Promise<Uint8Array> {
  const stream = new ReadableStream<BufferSource>({
    start: (controller) => {
      controller.enqueue(bytes.slice());
      controller.close();
    },
  }).pipeThrough(new DecompressionStream("gzip"));
  return new Uint8Array(await new Response(stream).arrayBuffer());
}

/** A loaded sample: its axial slices, the iso to mesh (null: Otsu's) and its suggested preset. */
export interface LoadedSample {
  slices: DicomSlice[];
  iso: number | null;
  preset: string | null;
}

/**
 * Fetches a bundled sample (`baseUrl` carries the trailing slash),
 * gunzips its scalars and stacks them into axial slices.
 */
export async function loadSample(
  sample: SampleKind,
  baseUrl: string,
  fetchImpl: typeof fetch = fetch,
): Promise<LoadedSample> {
  const root = `${baseUrl}samples/${sample}`;
  const index = await fetchImpl(`${root}/index.json`);
  if (!index.ok) throw new Error(`sample ${sample}: index.json answered ${index.status}`);
  const descriptor: unknown = await index.json();
  const image = parseVtkJsImage(descriptor);
  const meta = parseSampleMeta(descriptor);
  const blob = await fetchImpl(`${root}/data/${image.refId}`);
  if (!blob.ok) throw new Error(`sample ${sample}: scalars answered ${blob.status}`);
  const compressed = new Uint8Array(await blob.arrayBuffer());
  if (compressed.length < 2 || compressed[0] !== 0x1f || compressed[1] !== 0x8b) {
    throw new Error(`sample ${sample}: scalars are not gzip (a server decoded them?)`);
  }
  const raw = await gunzip(compressed);
  const floats = toFloatVolume(raw, image.dataType);
  if (floats.length !== image.dims[0] * image.dims[1] * image.dims[2]) {
    throw new Error(`sample ${sample}: scalars hold ${floats.length} voxels, not ${image.dims.join("x")}`);
  }
  const aligned = alignVolume(floats, image.dims, image.spacing, image.origin, image.direction);
  return {
    slices: volumeToSlices(aligned.values, aligned.dims, aligned.spacing, aligned.origin, meta),
    iso: meta.iso,
    preset: meta.preset,
  };
}

/** Stacks an aligned volume into axial slices (values scaled to HU). */
export function volumeToSlices(
  values: Float32Array,
  dims: readonly [number, number, number],
  spacing: readonly [number, number, number],
  origin: readonly [number, number, number],
  meta: { seriesUid: string; description: string; slope: number; intercept: number },
): DicomSlice[] {
  const [nx, ny, nz] = dims;
  const out: DicomSlice[] = [];
  for (let z = 0; z < nz; z++) {
    const pixels = new Int16Array(nx * ny);
    for (let i = 0; i < nx * ny; i++) {
      const hu = values[z * ny * nx + i] * meta.slope + meta.intercept;
      pixels[i] = Math.min(32767, Math.max(-32768, Math.round(hu)));
    }
    out.push({
      seriesUid: meta.seriesUid,
      instanceUid: `${meta.seriesUid}.${z}`,
      description: meta.description,
      rows: ny,
      columns: nx,
      bitsAllocated: 16,
      pixelRepresentation: 1,
      rescaleSlope: 1,
      rescaleIntercept: 0,
      pixelSpacing: [spacing[1], spacing[0]],
      position: [origin[0], origin[1], origin[2] + z * spacing[2]],
      orientation: [1, 0, 0, 0, 1, 0],
      pixels,
      inverted: false,
    });
  }
  return out;
}
