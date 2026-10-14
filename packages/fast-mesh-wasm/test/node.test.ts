// Imports the package by its own name so the test goes through the published
// `exports` map and the built dist/ files, exactly like a user install.
import { describe, expect, it } from "vitest";
import { computeBounds, computeVertexNormals, init } from "fast-mesh-wasm";

// Unit quad on the XY plane, two counter-clockwise triangles facing +Z.
const positions = new Float32Array([0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0]);
const indices = new Uint32Array([0, 1, 2, 0, 2, 3]);
const up = [0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1];

describe("fast-mesh-wasm (node entry)", () => {
  it("init resolves and accepts the same arguments as in the browser", async () => {
    await expect(init()).resolves.toBeUndefined();
    await expect(init(new Uint8Array())).resolves.toBeUndefined();
  });

  it("computes vertex normals", () => {
    const normals = computeVertexNormals(positions, indices);
    expect(normals).toBeInstanceOf(Float32Array);
    expect(Array.from(normals)).toEqual(up);
  });

  it("accepts Uint16Array indices, plain arrays and Float64Array positions", () => {
    expect(Array.from(computeVertexNormals(positions, new Uint16Array(indices)))).toEqual(up);
    expect(Array.from(computeVertexNormals(Array.from(positions), Array.from(indices)))).toEqual(up);
    expect(Array.from(computeVertexNormals(new Float64Array(positions), indices))).toEqual(up);
  });

  it("does not modify its inputs", () => {
    const pos = Array.from(positions);
    const idx = new Uint16Array(indices);
    computeVertexNormals(pos, idx);
    expect(pos).toEqual(Array.from(positions));
    expect(Array.from(idx)).toEqual(Array.from(indices));
  });

  it("throws a readable error for bad input", () => {
    expect(() => computeVertexNormals(positions, new Uint32Array([0, 1, 9]))).toThrow(
      "index 9 is out of range for a mesh with 4 vertices",
    );
  });

  it("returns bounds as a plain object", () => {
    const bounds = computeBounds(positions);
    expect(bounds).toEqual({
      min: [0, 0, 0],
      max: [1, 1, 0],
      center: [0.5, 0.5, 0],
      size: [1, 1, 0],
    });
    expect(Object.getPrototypeOf(bounds)).toBe(Object.prototype);
    expect(computeBounds([0, 0, 0, 2, 4, 6]).size).toEqual([2, 4, 6]);
  });
});
