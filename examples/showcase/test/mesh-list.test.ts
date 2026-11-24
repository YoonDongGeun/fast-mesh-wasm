// The mesh list's pure parts: the triangle count of an index buffer, the
// count as printed, each mesh's group folder and the heavy meshes.
import { describe, expect, it } from "vitest";
import { estimateMeshBytes } from "../src/io/big-file";
import { countTriangles, formatCount, heavyMeshes, meshGroupIndex } from "../src/features/home/ux/mesh-list";

describe("mesh list", () => {
  it("counts_triangles_skipping_removed_ones", () => {
    expect(countTriangles(new Uint32Array([0, 1, 2, 0, 0, 0, 2, 3, 0]))).toBe(2);
    expect(countTriangles(new Uint32Array([]))).toBe(0);
    expect(countTriangles(new Uint32Array([0, 0, 0]))).toBe(0);
  });

  it("formats_counts_in_millions_past_a_million", () => {
    expect(formatCount(12_288)).toBe("12,288");
    expect(formatCount(999_999)).toBe("999,999");
    expect(formatCount(1_050_000)).toBe("1.05 M");
  });

  it("indexes_each_member_to_its_group", () => {
    const index = meshGroupIndex([
      { id: 0, name: "G1", members: ["a", "b"] },
      { id: 1, name: "G2", members: ["b", "c"] },
    ]);
    expect(index.get("a")).toEqual({ id: 0, name: "G1" });
    // In two groups: the last listed wins.
    expect(index.get("b")).toEqual({ id: 1, name: "G2" });
    expect(index.get("d")).toBeUndefined();
  });

  it("lists_heavy_meshes_and_their_memory", () => {
    const items = [{ triangles: 10 }, { triangles: 2_000_000 }, { triangles: 500_000 }];
    const { heavy, bytes } = heavyMeshes(items, 500_000);
    expect(heavy).toEqual([items[1], items[2]]);
    expect(bytes).toBe(estimateMeshBytes(2_000_000) + estimateMeshBytes(500_000));
    expect(heavyMeshes(items, 3_000_000)).toEqual({ heavy: [], bytes: 0 });
  });
});
