// bindMesh and UVs: the `uv` attribute follows the mesh's channel through edits,
// drops and undo, and a texture goes on the material only while there are UVs.
import * as THREE from "three";
import { describe, expect, it } from "vitest";
import { type MeshChange, type MeshHandle, Scene } from "../src/node.js";
import { bindMesh } from "../src/three/index.js";

const box = (x = 0) => ({
  positions: Float64Array.from([0, 1].flatMap((z) => [x, 0, z, x + 1, 0, z, x + 1, 1, z, x, 1, z])),
  indices: Uint32Array.from([0, 2, 1, 0, 3, 2, 4, 5, 6, 4, 6, 7, 0, 1, 5, 0, 5, 4, 2, 3, 7, 2, 7, 6, 1, 2, 6, 1, 6, 5, 3, 0, 4, 3, 4, 7]),
});
const UVS = Float32Array.from({ length: 16 }, (_, i) => i / 16);

const uvAttribute = (o: THREE.Mesh) => o.geometry.getAttribute("uv") as THREE.BufferAttribute | undefined;

function textured(s: Scene): MeshHandle {
  const { positions, indices } = box();
  return s.addMesh(positions, indices, { uvs: UVS });
}

describe("bindMesh with UVs", () => {
  it("has a uv attribute holding a mirror of the view", () => {
    const s = new Scene();
    const b = bindMesh(s, textured(s));
    const attribute = uvAttribute(b.object);
    expect(attribute?.itemSize).toBe(2);
    expect(attribute?.array).toEqual(UVS);
    expect(attribute?.array.buffer).not.toBe(s.views(b.mesh).uvs()?.buffer);
    b.dispose();
    s.free();
  });

  it("has no uv attribute without UVs, and leaves the material alone", () => {
    const s = new Scene();
    const { positions, indices } = box();
    const m = s.addMesh(positions, indices);
    const material = new THREE.MeshBasicMaterial();
    const version = material.version;
    const b = bindMesh(s, m, { material });
    expect(uvAttribute(b.object)).toBeUndefined();
    expect(material.version).toBe(version);
    s.repair(m, { op: "subdivide" });
    s.commitFrame();
    expect(uvAttribute(b.object)).toBeUndefined();
    b.dispose();
    s.free();
  });

  it("uploads only the dirty UVs of an edit", () => {
    const s = new Scene();
    const m = textured(s);
    const b = bindMesh(s, m);
    s.commitFrame();
    const attribute = uvAttribute(b.object) as THREE.BufferAttribute;
    attribute.clearUpdateRanges();
    const version = attribute.version;
    s.edit(m, [{ op: "setUvs", vertices: [3], uvs: [0.25, 0.75] }]);
    s.commitFrame();
    expect(attribute.version).toBe(version + 1);
    expect(attribute.updateRanges).toEqual([{ start: 6, count: 2 }]);
    expect(Array.from(attribute.array.subarray(6, 8))).toEqual([0.25, 0.75]);
    b.dispose();
    s.free();
  });

  it("resizes with the slots and stays aligned", () => {
    const s = new Scene();
    const m = textured(s);
    const b = bindMesh(s, m);
    s.commitFrame();
    s.repair(m, { op: "subdivide" });
    s.commitFrame();
    expect(uvAttribute(b.object)?.array).toEqual(s.views(m).uvs());
    s.undo();
    s.commitFrame();
    expect(uvAttribute(b.object)?.array).toEqual(s.views(m).uvs());
    b.dispose();
    s.free();
  });

  it("takes the map off while a boolean has dropped the UVs, and undo puts it back", () => {
    const s = new Scene();
    const m = textured(s);
    const tool = box(0.5);
    const other = s.addMesh(tool.positions, tool.indices);
    const map = new THREE.Texture();
    const material = new THREE.MeshStandardMaterial();
    const b = bindMesh(s, m, { material, map });
    expect(material.map).toBe(map);
    s.commitFrame();

    s.boolean(m, other, "union");
    const seen: MeshChange[] = s.commitFrame();
    expect(seen.find((c) => c.mesh === m)?.uvs).toBe("all");
    expect(uvAttribute(b.object)).toBeUndefined();
    expect(material.map).toBeNull();

    const before = material.version;
    s.undo();
    s.commitFrame();
    expect(uvAttribute(b.object)?.array).toEqual(s.views(m).uvs());
    expect(material.map).toBe(map);
    expect(material.version).toBeGreaterThan(before);
    b.dispose();
    map.dispose();
    s.free();
  });

  it("draws a given material's own map, and restores it on dispose", () => {
    const s = new Scene();
    const map = new THREE.Texture();
    const material = new THREE.MeshBasicMaterial({ map });
    const { positions, indices } = box();
    const plain = s.addMesh(positions, indices);
    const b = bindMesh(s, plain, { material });
    expect(material.map).toBeNull();
    b.dispose();
    expect(material.map).toBe(map);
    const c = bindMesh(s, textured(s), { material });
    expect(material.map).toBe(map);
    c.dispose();
    expect(material.map).toBe(map);
    s.free();
  });

  it("makes the default material white under a map, grey otherwise", () => {
    const s = new Scene();
    const map = new THREE.Texture();
    const white = bindMesh(s, textured(s), { map });
    const material = white.object.material as THREE.MeshStandardMaterial;
    expect(material.color.getHex()).toBe(0xffffff);
    expect(material.map).toBe(map);
    const grey = bindMesh(s, textured(s));
    expect((grey.object.material as THREE.MeshStandardMaterial).color.getHex()).toBe(0xcccccc);
    white.dispose();
    grey.dispose();
    s.free();
  });

  it("works with fragment shading", () => {
    const s = new Scene();
    const material = new THREE.MeshStandardMaterial();
    const b = bindMesh(s, textured(s), { material, shading: "fragment" });
    expect(b.shading).toBe("fragment");
    expect(uvAttribute(b.object)?.array).toEqual(UVS);
    b.dispose();
    s.free();
  });
});
