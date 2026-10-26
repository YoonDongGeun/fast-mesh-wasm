// The volume LUT: a transfer preset sampled into 8-bit RGBA entries
// over the data range, plus the volume normalized to 8-bit indices for
// the 3D texture.
import type { VolumePreset } from "./volume-presets";

export const LUT_SIZE = 256;

function ramp(points: Array<readonly [number, number]>, v: number): number {
  if (v <= points[0][0]) {
    return points[0][1];
  }
  for (let i = 1; i < points.length; i++) {
    if (v <= points[i][0]) {
      const [x0, y0] = points[i - 1];
      const [x1, y1] = points[i];
      const t = x1 === x0 ? 0 : (v - x0) / (x1 - x0);
      return y0 + (y1 - y0) * t;
    }
  }
  return points[points.length - 1][1];
}

const clamp01 = (x: number): number => Math.min(1, Math.max(0, x));

/** The preset sampled into `size` RGBA entries over [min, max]. */
export function presetLut(preset: VolumePreset, min: number, max: number, size = LUT_SIZE): Uint8Array {
  const lut = new Uint8Array(size * 4);
  const span = max - min === 0 ? 1 : max - min;
  const red = preset.color.map(([v, r]) => [v, r] as const);
  const green = preset.color.map(([v, , g]) => [v, g] as const);
  const blue = preset.color.map(([v, , , b]) => [v, b] as const);
  for (let i = 0; i < size; i++) {
    const v = min + (i / (size - 1)) * span;
    lut[4 * i] = Math.round(clamp01(ramp(red, v)) * 255);
    lut[4 * i + 1] = Math.round(clamp01(ramp(green, v)) * 255);
    lut[4 * i + 2] = Math.round(clamp01(ramp(blue, v)) * 255);
    lut[4 * i + 3] = Math.round(clamp01(ramp(preset.opacity, v)) * 255);
  }
  return lut;
}

/** The values mapped to 8-bit over [min, max], clamped. */
export function normalizeVolume(values: Float32Array, min: number, max: number): Uint8Array {
  const out = new Uint8Array(values.length);
  const span = max - min === 0 ? 1 : max - min;
  for (let i = 0; i < values.length; i++) {
    out[i] = Math.round(clamp01((values[i] - min) / span) * 255);
  }
  return out;
}

/** The [min, max] of the values. */
export function dataRange(values: Float32Array): [number, number] {
  let min = Number.POSITIVE_INFINITY;
  let max = Number.NEGATIVE_INFINITY;
  for (let i = 0; i < values.length; i++) {
    if (values[i] < min) min = values[i];
    if (values[i] > max) max = values[i];
  }
  return [min, max];
}
