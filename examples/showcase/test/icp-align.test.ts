// The align panel's fixed-mesh pick: the current pick while it is loaded
// and differs from the moving mesh, else the first other mesh.
import type { MeshHandle } from "fast-mesh-wasm";
import { describe, expect, it } from "vitest";
import type { Item } from "../src/app/state/app-state";
import { pickDistinctMesh } from "../src/features/transform/ux/use-icp-align";

const [a, b, c, gone] = [1, 2, 3, 4].map((n) => n as unknown as MeshHandle);
const items = [a, b, c].map((mesh) => ({ mesh }) as unknown as Item);

describe("pickDistinctMesh", () => {
  it("keeps_a_loaded_distinct_pick", () => {
    expect(pickDistinctMesh(items, a, c)).toBe(c);
  });

  it("falls_back_to_the_first_other_mesh", () => {
    expect(pickDistinctMesh(items, a, null)).toBe(b);
    expect(pickDistinctMesh(items, a, a)).toBe(b);
    expect(pickDistinctMesh(items, b, gone)).toBe(a);
  });

  it("is_null_without_another_mesh", () => {
    expect(pickDistinctMesh(items.slice(0, 1), a, null)).toBeNull();
    expect(pickDistinctMesh([], null, null)).toBeNull();
  });
});
