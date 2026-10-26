// The series stack: parsed slices grouped by series into regular float
// volumes with their patient geometry. Slices sort along the slice normal;
// tilted or irregularly spaced series fail (a regular grid would lie).
import type { DicomSlice } from "./dicom";

export interface DicomVolume {
  seriesUid: string;
  /** The slices' SeriesDescription (the first slice's; series agree). */
  description: string;
  columns: number;
  rows: number;
  slices: number;
  /** Rescaled values (Hounsfield units for CT), slice-major: value[z][y][x]. */
  values: Float32Array;
  /** Voxel spacing in mm: [dx, dy, dz]. */
  spacing: [number, number, number];
  /** The first slice's first voxel, in mm (patient coordinates). */
  origin: [number, number, number];
  /** The +x (column), +y (row) and +z (slice) steps in mm. */
  xStep: [number, number, number];
  yStep: [number, number, number];
  sliceStep: [number, number, number];
  inverted: boolean;
}

const dot = (a: readonly [number, number, number], b: readonly [number, number, number]): number =>
  a[0] * b[0] + a[1] * b[1] + a[2] * b[2];

function cross(a: readonly [number, number, number], b: readonly [number, number, number]): [number, number, number] {
  return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
}

/** Stacks every series found; `@throws` on lone, inconsistent or tilted series. */
export function stackSeries(slices: readonly DicomSlice[]): DicomVolume[] {
  const groups = new Map<string, DicomSlice[]>();
  for (const s of slices) {
    const g = groups.get(s.seriesUid);
    if (g === undefined) {
      groups.set(s.seriesUid, [s]);
    } else {
      g.push(s);
    }
  }
  return [...groups.values()].map(stackOne);
}

function stackOne(slices: DicomSlice[]): DicomVolume {
  const [first, ...rest] = slices;
  if (rest.length === 0) {
    throw new Error(
      `DICOM series ${first.seriesUid} has a single slice: drop the whole series (one file per slice) to build a volume`,
    );
  }
  const { rows, columns } = first;
  const [rx, ry, rz, cx, cy, cz] = first.orientation;
  const normal = cross([rx, ry, rz], [cx, cy, cz]);
  const norm = Math.hypot(normal[0], normal[1], normal[2]);
  const n: [number, number, number] = [normal[0] / norm, normal[1] / norm, normal[2] / norm];
  for (const s of rest) {
    if (s.rows !== rows || s.columns !== columns) {
      throw new Error(
        `DICOM series ${first.seriesUid} mixes frame sizes (${rows}x${columns} and ${s.rows}x${s.columns})`,
      );
    }
    if (s.orientation.some((o, i) => Math.abs(o - first.orientation[i]) > 1e-6)) {
      throw new Error(`DICOM series ${first.seriesUid} mixes orientations (a tilted acquisition cannot stack)`);
    }
    if (s.inverted !== first.inverted) {
      throw new Error(`DICOM series ${first.seriesUid} mixes photometric interpretations`);
    }
    if (Math.abs(s.pixelSpacing[0] - first.pixelSpacing[0]) > 1e-9 || Math.abs(s.pixelSpacing[1] - first.pixelSpacing[1]) > 1e-9) {
      throw new Error(`DICOM series ${first.seriesUid} mixes pixel spacings`);
    }
  }
  const along = slices.map((s) => dot(s.position, n)).sort((a, b) => a - b);
  const ordered = [...slices].sort((a, b) => dot(a.position, n) - dot(b.position, n));
  const gaps = along.slice(1).map((p, i) => p - along[i]);
  const dz = gaps.slice().sort((a, b) => a - b)[Math.floor(gaps.length / 2)];
  if (!(dz > 0)) {
    throw new Error(`DICOM series ${first.seriesUid} has coincident slices`);
  }
  for (const g of gaps) {
    if (Math.abs(g - dz) > 1e-3 * dz) {
      throw new Error(`DICOM series ${first.seriesUid} has irregular slice spacing (gaps differ)`);
    }
  }
  // Tilted (gantry) series shift in-plane between slices: a regular grid
  // would shear the anatomy.
  const planar = Math.min(first.pixelSpacing[0], first.pixelSpacing[1]);
  for (let i = 1; i < ordered.length; i++) {
    const dx = ordered[i].position[0] - ordered[i - 1].position[0] - n[0] * gaps[i - 1];
    const dy = ordered[i].position[1] - ordered[i - 1].position[1] - n[1] * gaps[i - 1];
    const dzr = ordered[i].position[2] - ordered[i - 1].position[2] - n[2] * gaps[i - 1];
    if (Math.hypot(dx, dy, dzr) > 1e-3 * planar) {
      throw new Error(`DICOM series ${first.seriesUid} is tilted (slices shift in-plane): cannot stack`);
    }
  }
  const values = new Float32Array(columns * rows * ordered.length);
  ordered.forEach((s, z) => {
    const at = z * columns * rows;
    for (let i = 0; i < columns * rows; i++) {
      values[at + i] = s.pixels[i] * s.rescaleSlope + s.rescaleIntercept;
    }
  });
  const dx = first.pixelSpacing[1];
  const dy = first.pixelSpacing[0];
  return {
    seriesUid: first.seriesUid,
    description: first.description,
    columns,
    rows,
    slices: ordered.length,
    values,
    spacing: [dx, dy, dz],
    origin: [...ordered[0].position],
    xStep: [rx * dx, ry * dx, rz * dx],
    yStep: [cx * dy, cy * dy, cz * dy],
    sliceStep: [n[0] * dz, n[1] * dz, n[2] * dz],
    inverted: first.inverted,
  };
}
