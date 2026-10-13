// Imports the package by its own name so the test goes through the published
// `exports` map and the built dist/ files, exactly like a user install.
import { describe, expect, it } from "vitest";
import { computeBounds, computeVertexNormals, init } from "fast-mesh-wasm";

// Unit quad on the XY plane, two counter-clockwise triangles facing +Z.
const positions = new Float32Array([0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0]);
const indices = new Uint32Array([0, 1, 2, 0, 2, 3]);

describe("fast-mesh-wasm (node entry)", () => {
  it("init resolves", async () => {
    await expect(init()).resolves.toBeUndefined();
  });

  it("computes vertex normals", () => {
    const normals = computeVertexNormals(positions, indices);
    expect(normals).toBeInstanceOf(Float32Array);
    expect(Array.from(normals)).toEqual([0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1]);
  });

  it("throws a readable error for bad input", () => {
    expect(() => computeVertexNormals(positions, new Uint32Array([0, 1, 9]))).toThrow(
      "index 9 is out of range for a mesh with 4 vertices",
    );
  });

  it("computes bounds and frees memory with `using`", () => {
    using bounds = computeBounds(positions);
    expect(Array.from(bounds.min)).toEqual([0, 0, 0]);
    expect(Array.from(bounds.max)).toEqual([1, 1, 0]);
    expect(Array.from(bounds.center)).toEqual([0.5, 0.5, 0]);
    expect(Array.from(bounds.size)).toEqual([1, 1, 0]);
  });
});
