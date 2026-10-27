// The MPR slices: plane extraction over the x-fastest volume, with a
// fixed orientation (up is +Y for axial, +Z otherwise), and window/level
// to gray.
export type SliceAxis = "axial" | "coronal" | "sagittal";

export type SliceImage = { data: Float32Array; width: number; height: number };

/** The plane at `index` along `axis`, top row first. */
export function extractSlice(
  values: Float32Array,
  columns: number,
  rows: number,
  slices: number,
  axis: SliceAxis,
  index: number,
): SliceImage {
  if (axis === "axial") {
    const data = new Float32Array(columns * rows);
    for (let j = 0; j < rows; j++) {
      for (let i = 0; i < columns; i++) {
        data[(rows - 1 - j) * columns + i] = values[(index * rows + j) * columns + i];
      }
    }
    return { data, width: columns, height: rows };
  }
  if (axis === "coronal") {
    const data = new Float32Array(columns * slices);
    for (let k = 0; k < slices; k++) {
      for (let i = 0; i < columns; i++) {
        data[(slices - 1 - k) * columns + i] = values[(k * rows + index) * columns + i];
      }
    }
    return { data, width: columns, height: slices };
  }
  const data = new Float32Array(rows * slices);
  for (let k = 0; k < slices; k++) {
    for (let j = 0; j < rows; j++) {
      data[(slices - 1 - k) * rows + j] = values[(k * rows + j) * columns + index];
    }
  }
  return { data, width: rows, height: slices };
}

export type VolumeDims = { columns: number; rows: number; slices: number };

/**
 * A pointer at (`dx`, `dy`) within the view's box to the slice indices it
 * picks on the other two axes. The image is `object-fit: contain`, so the
 * box may letterbox it; out-of-image points clamp to the edge.
 */
export function pointerSlice(
  axis: SliceAxis,
  dims: VolumeDims,
  canvas: { width: number; height: number },
  rect: { width: number; height: number },
  dx: number,
  dy: number,
): Partial<Record<SliceAxis, number>> {
  const scale = Math.min(rect.width / canvas.width, rect.height / canvas.height);
  const ox = (rect.width - canvas.width * scale) / 2;
  const oy = (rect.height - canvas.height * scale) / 2;
  const pick = (p: number, count: number): number => Math.min(count - 1, Math.max(0, Math.floor(p)));
  const px = pick((dx - ox) / scale, canvas.width);
  const top = pick((dy - oy) / scale, canvas.height);
  if (axis === "axial") {
    return { sagittal: px, coronal: dims.rows - 1 - top };
  }
  if (axis === "coronal") {
    return { sagittal: px, axial: dims.slices - 1 - top };
  }
  return { coronal: px, axial: dims.slices - 1 - top };
}

/** The values mapped through the window to opaque gray RGBA. */
export function windowToGray(data: Float32Array, window: number, level: number): Uint8ClampedArray<ArrayBuffer> {
  const out = new Uint8ClampedArray(data.length * 4);
  const lo = level - window / 2;
  for (let i = 0; i < data.length; i++) {
    const g = Math.round(Math.min(1, Math.max(0, (data[i] - lo) / window)) * 255);
    out[4 * i] = g;
    out[4 * i + 1] = g;
    out[4 * i + 2] = g;
    out[4 * i + 3] = 255;
  }
  return out;
}
