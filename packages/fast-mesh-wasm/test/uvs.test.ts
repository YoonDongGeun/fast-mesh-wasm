// The UV channel through the package's scene: add, views, operations, undo and
// change records. The per-operation table is tested in Rust; these cover the
// bindings on top of it.
import { describe, expect, it } from "vitest";
import { FastMeshError, Scene } from "../src/node.js";

/** A closed unit box at `x`. */
const box = (x = 0) => ({
  positions: Float64Array.from(
    [0, 1].flatMap((z) => [
      [x, 0, z],
      [x + 1, 0, z],
      [x + 1, 1, z],
      [x, 1, z],
    ]).flat(),
  ),
  indices: Uint32Array.from([0, 2, 1, 0, 3, 2, 4, 5, 6, 4, 6, 7, 0, 1, 5, 0, 5, 4, 2, 3, 7, 2, 7, 6, 1, 2, 6, 1, 6, 5, 3, 0, 4, 3, 4, 7]),
});
const BOX_UVS = Float32Array.from({ length: 16 }, (_, i) => i / 16);

/** Two quads with split vertices along the shared edge; `seam` makes their UVs differ. */
const quads = (seam: boolean) => ({
  positions: Float64Array.from([0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0, 0, 1, 1, 0, 2, 0, 0, 2, 1, 0]),
  indices: Uint32Array.from([0, 1, 2, 0, 2, 3, 4, 6, 7, 4, 7, 5]),
  uvs: Float32Array.from([0, 0, 0.5, 0, 0.5, 1, 0, 1, seam ? 0.75 : 0.5, 0, seam ? 0.75 : 0.5, 1, 1, 0, 1, 1]),
});

describe("addMesh with uvs", () => {
  it("is zero-copy readable and null without them", () => {
    const s = new Scene();
    const { positions, indices } = box();
    const plain = s.addMesh(positions, indices);
    const textured = s.addMesh(positions, indices, { uvs: BOX_UVS });
    expect(s.views(plain).uvs()).toBeNull();
    const view = s.views(textured).uvs();
    expect(view).toEqual(BOX_UVS);
    expect(view?.buffer).toBe(s.views(textured).positions().buffer);
    const records = s.takeChanges();
    expect(records.find((c) => c.mesh === plain)).not.toHaveProperty("uvs");
    expect(records.find((c) => c.mesh === textured)?.uvs).toBe("all");
    s.free();
  });

  it("takes a plain array", () => {
    const s = new Scene();
    const { positions, indices } = box();
    const m = s.addMesh(positions, indices, { uvs: Array.from(BOX_UVS) });
    expect(s.views(m).uvs()).toEqual(BOX_UVS);
    s.free();
  });

  it("rejects a wrong length or non-finite values and adds nothing", () => {
    const s = new Scene();
    const { positions, indices } = box();
    const code = (uvs: Float32Array) => {
      try {
        s.addMesh(positions, indices, { uvs });
      } catch (e) {
        return (e as FastMeshError).code;
      }
      return "none";
    };
    expect(code(new Float32Array(15))).toBe("INVALID_ARGUMENT");
    const bad = BOX_UVS.slice();
    bad[3] = Number.NaN;
    expect(code(bad)).toBe("INVALID_ARGUMENT");
    expect(s.takeChanges()).toEqual([]);
    s.free();
  });
});

describe("operations", () => {
  it("weld keeps a seam and merges equal UVs", () => {
    const s = new Scene();
    const seam = quads(true);
    const a = s.addMesh(seam.positions, seam.indices, { uvs: seam.uvs });
    expect(s.repair(a, { op: "weld", tolerance: 0.01 }).mergedVertices).toBe(0);
    const same = quads(false);
    const b = s.addMesh(same.positions, same.indices, { uvs: same.uvs });
    expect(s.repair(b, { op: "weld", tolerance: 0.01 }).mergedVertices).toBe(2);
    const c = s.addMesh(same.positions, same.indices);
    expect(s.repair(c, { op: "weld", tolerance: 0.01 }).mergedVertices).toBe(2);
    s.free();
  });

  it("subdivide reports the new vertices as dirty and undo restores the buffer", () => {
    const s = new Scene();
    const { positions, indices } = box();
    const m = s.addMesh(positions, indices, { uvs: BOX_UVS });
    s.takeChanges();
    s.repair(m, { op: "subdivide" });
    const uvs = s.views(m).uvs();
    expect(uvs?.length).toBe(2 * s.vertexSlots(m));
    expect(uvs?.subarray(0, 16)).toEqual(BOX_UVS);
    expect(s.takeChanges().find((c) => c.mesh === m)?.uvs).toBe("all");
    s.undo();
    expect(s.views(m).uvs()?.subarray(0, 16)).toEqual(BOX_UVS);
    s.free();
  });

  it("a boolean drops them, reports it, and undo and redo follow", () => {
    const s = new Scene();
    const a = box();
    const b = box(0.5);
    const ma = s.addMesh(a.positions, a.indices, { uvs: BOX_UVS });
    const mb = s.addMesh(b.positions, b.indices);
    s.takeChanges();
    s.boolean(ma, mb, "union");
    expect(s.views(ma).uvs()).toBeNull();
    expect(s.takeChanges().find((c) => c.mesh === ma)?.uvs).toBe("all");
    s.undo();
    expect(s.views(ma).uvs()?.subarray(0, 16)).toEqual(BOX_UVS);
    expect(s.takeChanges().find((c) => c.mesh === ma)?.uvs).toBe("all");
    s.redo();
    expect(s.views(ma).uvs()).toBeNull();
    s.free();
  });

  it("a cut gives the new mesh its UVs", () => {
    const s = new Scene();
    const { positions, indices } = box();
    const m = s.addMesh(positions, indices, { uvs: BOX_UVS });
    const { removed } = s.cutByPlane(m, { origin: [0.5, 0, 0], normal: [1, 0, 0] });
    expect(s.views(removed).uvs()?.length).toBe(2 * s.vertexSlots(removed));
    expect(s.views(m).uvs()?.length).toBe(2 * s.vertexSlots(m));
    s.free();
  });
});

describe("cutByPlane across a UV seam", () => {
  /** The box with its +x face its own island: slots 8 to 11 repeat corners 1, 2, 6 and 5. */
  const seamedBox = (x = 0) => {
    const { positions, indices } = box(x);
    const split = new Map([[1, 8], [2, 9], [6, 10], [5, 11]]);
    const extra = [1, 2, 6, 5].flatMap((v) => Array.from(positions.subarray(3 * v, 3 * v + 3)));
    const faces = Uint32Array.from(indices);
    for (let i = 24; i < 30; i++) {
      faces[i] = split.get(faces[i]) as number;
    }
    return {
      positions: Float64Array.from([...positions, ...extra]),
      indices: faces,
      uvs: Float32Array.from({ length: 24 }, (_, i) => i / 24),
    };
  };
  const plane = { origin: [0, 0.5, 0] as [number, number, number], normal: [0, 1, 0] as [number, number, number] };

  it("caps both sides, reports it and undoes", () => {
    const s = new Scene();
    const { positions, indices, uvs } = seamedBox();
    const m = s.addMesh(positions, indices, { uvs, normals: false });
    const r = s.cutByPlane(m, plane);
    expect(r.capTriangles).toBeGreaterThanOrEqual(4);
    expect(s.triangleCount(m)).toBe(r.keptTriangles);
    expect(s.triangleCount(r.removed)).toBe(r.removedTriangles);
    expect(s.views(m).uvs()?.length).toBe(2 * s.vertexSlots(m));
    expect(s.views(r.removed).uvs()?.length).toBe(2 * s.vertexSlots(r.removed));
    // Closed by position: a boolean checks that first, and rejects an open mesh.
    // Far apart, so this tests the check and not the intersection.
    const moved = seamedBox(5);
    const tool = s.addMesh(moved.positions, moved.indices, { uvs: moved.uvs, normals: false });
    expect(() => s.boolean(r.removed, tool, "union")).not.toThrow();
    s.undo();
    s.undo();
    expect(s.triangleCount(m)).toBe(12);
    expect(s.vertexCount(m)).toBe(12);
    s.free();
  });

  it("leaves it open when asked, with nothing to report", () => {
    const s = new Scene();
    const { positions, indices, uvs } = seamedBox();
    const m = s.addMesh(positions, indices, { uvs, normals: false });
    expect(s.cutByPlane(m, plane, { cap: false }).capTriangles).toBe(0);
    s.free();
  });
});

describe("edit", () => {
  it("setUvs writes added vertices through refs, undoably", () => {
    const s = new Scene();
    const { positions, indices } = box();
    const m = s.addMesh(positions, indices, { uvs: BOX_UVS });
    s.edit(m, [
      { op: "addVertices", positions: [5, 5, 5] },
      { op: "setUvs", vertices: [{ ref: [0, 0] }, 1], uvs: [0.25, 0.75, 0.5, 0.5] },
    ]);
    const uvs = s.views(m).uvs() as Float32Array;
    expect(Array.from(uvs.subarray(16, 18))).toEqual([0.25, 0.75]);
    expect(Array.from(uvs.subarray(2, 4))).toEqual([0.5, 0.5]);
    s.undo();
    expect(s.views(m).uvs()?.subarray(0, 16)).toEqual(BOX_UVS);
    s.free();
  });

  it("addVertices gives the default UV; setUvs needs UVs and the right length", () => {
    const s = new Scene();
    const { positions, indices } = box();
    const m = s.addMesh(positions, indices, { uvs: BOX_UVS });
    s.edit(m, [{ op: "addVertices", positions: [5, 5, 5] }]);
    expect(Array.from((s.views(m).uvs() as Float32Array).subarray(16, 18))).toEqual([0, 0]);
    expect(() => s.edit(m, [{ op: "setUvs", vertices: [0], uvs: [0] }])).toThrow(FastMeshError);
    expect(() => s.edit(m, [{ op: "setUvs", vertices: [0], uvs: [Number.NaN, 0] }])).toThrow(FastMeshError);
    const plain = s.addMesh(positions, indices);
    expect(() => s.edit(plain, [{ op: "setUvs", vertices: [0], uvs: [0, 0] }])).toThrow(FastMeshError);
    s.free();
  });
});
