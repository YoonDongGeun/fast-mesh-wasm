// The isosurface: marching tetrahedra (a Freudenthal 6-tet split, so shared
// cube faces always agree) with edge-keyed vertex sharing, plus Otsu's
// threshold for a modality-agnostic default.
import { describe, expect, it } from "vitest";
import { isosurface, otsu } from "../src/load/isosurface";

const box = (columns: number, rows: number, slices: number) => ({
  columns,
  rows,
  slices,
  origin: [0, 0, 0] as const,
});
const BOX = box(2, 2, 2);

function vol(dims: readonly [number, number, number], f: (x: number, y: number, z: number) => number): Float32Array {
  const [nx, ny, nz] = dims;
  const out = new Float32Array(nx * ny * nz);
  for (let z = 0; z < nz; z++) {
    for (let y = 0; y < ny; y++) {
      for (let x = 0; x < nx; x++) {
        out[z * ny * nx + y * nx + x] = f(x, y, z);
      }
    }
  }
  return out;
}

/** Every directed edge has its reverse exactly once (closed, no cracks, no fins). */
function expectManifold(positions: Float64Array, indices: Uint32Array): void {
  expect(indices.length % 3).toBe(0);
  expect(indices.length).toBeGreaterThan(0);
  const seen = new Map<number, number>();
  const key = (a: number, b: number) => a * 4294967296 + b;
  for (let t = 0; t < indices.length; t += 3) {
    const [a, b, c] = [indices[t], indices[t + 1], indices[t + 2]];
    expect(new Set([a, b, c]).size).toBe(3);
    for (const [u, v] of [[a, b], [b, c], [c, a]]) {
      seen.set(key(u, v), (seen.get(key(u, v)) ?? 0) + 1);
    }
  }
  for (const [k, n] of seen) {
    const a = Math.floor(k / 4294967296);
    const b = k % 4294967296;
    expect(n).toBe(1);
    expect(seen.get(key(b, a))).toBe(1);
  }
}

describe("isosurface", () => {
  it("emits_nothing_outside_the_range", () => {
    const values = vol([2, 2, 2], () => 0);
    expect(isosurface(values, BOX, 0.5, false).indices.length).toBe(0);
    expect(isosurface(values, BOX, -0.5, false).indices.length).toBe(0);
  });

  it("cuts_one_hot_corner_into_six_tris", () => {
    const values = vol([2, 2, 2], (x, y, z) => (x === 0 && y === 0 && z === 0 ? 1 : 0));
    const { positions, indices } = isosurface(values, BOX, 0.5, false);
    expect(indices.length / 3).toBe(6);
    for (const p of positions) {
      expect(p).toBeGreaterThanOrEqual(0);
      expect(p).toBeLessThanOrEqual(1);
    }
  });

  it("closes_a_blob", () => {
    const values = vol([3, 3, 3], (x, y, z) => (x === 1 && y === 1 && z === 1 ? 1 : 0));
    const { positions, indices } = isosurface(values, box(3, 3, 3), 0.5, false);
    expectManifold(positions, indices);
  });

  it("cuts_a_plane_exactly", () => {
    const values = vol([4, 4, 4], (x) => x);
    const { positions, indices } = isosurface(values, box(4, 4, 4), 1.5, false);
    expect(indices.length).toBeGreaterThan(0);
    for (let i = 0; i < positions.length; i += 3) {
      expect(positions[i]).toBeCloseTo(1.5, 9);
    }
  });

  it("flip_reverses_the_solid", () => {
    const values = vol([3, 3, 3], (x, y, z) => (x === 1 && y === 1 && z === 1 ? 1 : 0));
    const a = isosurface(values, box(3, 3, 3), 0.5, false);
    const b = isosurface(values, box(3, 3, 3), 0.5, true);
    expect(a.indices.length).toBe(b.indices.length);
    const signed = (p: Float64Array, idx: Uint32Array): number => {
      let v = 0;
      for (let t = 0; t < idx.length; t += 3) {
        const [i, j, k] = [3 * idx[t], 3 * idx[t + 1], 3 * idx[t + 2]];
        v +=
          p[i] * (p[j + 1] * p[k + 2] - p[j + 2] * p[k + 1]) -
          p[i + 1] * (p[j] * p[k + 2] - p[j + 2] * p[k]) +
          p[i + 2] * (p[j] * p[k + 1] - p[j + 1] * p[k]);
      }
      return v / 6;
    };
    expect(signed(a.positions, a.indices)).toBeGreaterThan(0);
    expect(signed(b.positions, b.indices)).toBeLessThan(0);
  });
});

describe("otsu", () => {
  it("splits_bimodal_values", () => {
    const values = new Float32Array([...new Array(100).fill(0), ...new Array(100).fill(10)]);
    const t = otsu(values);
    expect(t).toBeGreaterThan(0);
    expect(t).toBeLessThan(10);
  });

  it("returns_the_value_when_uniform", () => {
    expect(otsu(new Float32Array([5, 5, 5]))).toBe(5);
  });
});
