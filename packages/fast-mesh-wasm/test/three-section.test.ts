// bindSection in Node: clipping planes on the bound meshes' materials,
// contour lines and fill following the scene's section results.
import * as THREE from "three";
import { afterEach, describe, expect, it } from "vitest";
import { type MeshHandle, Scene, type SectionHandle, type Vec3 } from "../src/node.js";
import { bindMesh, bindSection, type BoundSection } from "../src/three/index.js";

/** A closed UV sphere (even stacks: equator vertices on `z = center.z`). */
function sphere(center: Vec3, radius = 1, stacks = 16, slices = 32) {
  const positions: number[] = [center[0], center[1], center[2] + radius];
  for (let i = 1; i < stacks; i++) {
    const phi = (Math.PI * i) / stacks;
    for (let j = 0; j < slices; j++) {
      const theta = (2 * Math.PI * j) / slices;
      positions.push(
        center[0] + radius * Math.sin(phi) * Math.cos(theta),
        center[1] + radius * Math.sin(phi) * Math.sin(theta),
        center[2] + radius * Math.cos(phi),
      );
    }
  }
  positions.push(center[0], center[1], center[2] - radius);
  const south = positions.length / 3 - 1;
  const ring = (i: number, j: number) => 1 + (i - 1) * slices + (j % slices);
  const indices: number[] = [];
  for (let j = 0; j < slices; j++) {
    indices.push(0, ring(1, j), ring(1, j + 1));
    indices.push(south, ring(stacks - 1, j + 1), ring(stacks - 1, j));
  }
  for (let i = 1; i < stacks - 1; i++) {
    for (let j = 0; j < slices; j++) {
      const a = ring(i, j);
      const b = ring(i, j + 1);
      const c = ring(i + 1, j + 1);
      const d = ring(i + 1, j);
      indices.push(a, d, c, a, c, b);
    }
  }
  return { positions: Float64Array.from(positions), indices: Uint32Array.from(indices) };
}

let scenes: Scene[] = [];
afterEach(() => {
  for (const s of scenes) {
    s.free();
  }
  scenes = [];
});

function setup() {
  const s = new Scene();
  scenes.push(s);
  const ga = sphere([0, 0, 0]);
  const gb = sphere([3, 0, 0]);
  const a = s.addMesh(ga.positions, ga.indices);
  const b = s.addMesh(gb.positions, gb.indices);
  const ba = bindMesh(s, a);
  const bb = bindMesh(s, b);
  const sec = s.addSection({ plane: { origin: [0, 0, 0], normal: [0, 0, 1] }, meshes: [a, b], fill: true });
  s.commitFrame();
  return { s, a, b, ba, bb, sec };
}

/** The expected segment endpoints of a mesh's contours. */
function expectedLines(s: Scene, sec: SectionHandle, m: MeshHandle): number[] {
  const out: number[] = [];
  for (const { points, closed } of s.section(sec, m).contours) {
    const n = points.length / 3;
    for (let k = 0; k < (closed ? n : n - 1); k++) {
      const j = (k + 1) % n;
      out.push(...points.subarray(3 * k, 3 * k + 3), ...points.subarray(3 * j, 3 * j + 3));
    }
  }
  return out;
}

function linesOf(bs: BoundSection, k: number): THREE.LineSegments {
  return bs.group.children.filter((c) => c instanceof THREE.LineSegments)[k] as THREE.LineSegments;
}

function fillOf(bs: BoundSection, k: number): THREE.Mesh {
  return bs.group.children.filter((c) => !(c instanceof THREE.LineSegments))[k] as THREE.Mesh;
}

function positionsOf(o: THREE.Object3D & { geometry: THREE.BufferGeometry }): number[] {
  return Array.from(o.geometry.getAttribute("position")?.array ?? []);
}

describe("bindSection", () => {
  it("section_clipping_and_lines", () => {
    const { s, a, b, ba, bb, sec } = setup();
    const bs = bindSection(s, sec, [ba, bb], { lineColor: 0xff0000 });
    const ma = ba.object.material as THREE.Material;
    const mb = bb.object.material as THREE.Material;
    // The plane is the section's; it is on both materials.
    expect(bs.plane.normal.toArray()).toEqual([0, 0, 1]);
    expect(bs.plane.constant).toBe(-0);
    expect(ma.clippingPlanes).toEqual([bs.plane]);
    expect(mb.clippingPlanes![0]).toBe(bs.plane);
    // Lines and fill match the scene's results.
    expect(bs.group.children.length).toBe(4);
    const la = linesOf(bs, 0);
    expect(la).toBeInstanceOf(THREE.LineSegments);
    expect((la.material as THREE.LineBasicMaterial).color.getHex()).toBe(0xff0000);
    expect(positionsOf(la)).toEqual(expectedLines(s, sec, a));
    expect(positionsOf(la).length).toBeGreaterThanOrEqual(6 * 32);
    expect(positionsOf(linesOf(bs, 1))).toEqual(expectedLines(s, sec, b));
    const fill = s.section(sec, a).fill!;
    expect(positionsOf(fillOf(bs, 0))).toEqual(Array.from(fill.positions));
    expect(Array.from(fillOf(bs, 0).geometry.getIndex()!.array)).toEqual(Array.from(fill.indices));

    // A plane move shows on the next frame: plane, lines and fill.
    s.setSectionPlane(sec, { origin: [0, 0, 0.5], normal: [0, 0, -2] });
    expect(bs.plane.constant).toBe(-0);
    const before = la.geometry;
    s.commitFrame();
    expect(bs.plane.normal.toArray()).toEqual([0, 0, -1]);
    expect(bs.plane.constant).toBeCloseTo(0.5, 12);
    expect(bs.plane.distanceToPoint(new THREE.Vector3(0, 0, 0))).toBeCloseTo(0.5, 12);
    expect(linesOf(bs, 0).geometry).not.toBe(before);
    const moved = expectedLines(s, sec, a);
    expect(positionsOf(linesOf(bs, 0))).toEqual(moved);
    expect(moved.filter((_, k) => k % 3 === 2).every((z) => Math.abs(z - 0.5) < 1e-6)).toBe(true);
    expect(positionsOf(fillOf(bs, 0))).toEqual(Array.from(s.section(sec, a).fill!.positions));

    // Moving a mesh rebuilds its lines.
    s.setPlacement(b, { rotation: [1, 0, 0, 0, 1, 0, 0, 0, 1], translation: [0, 5, 0] });
    s.commitFrame();
    expect(positionsOf(linesOf(bs, 1))).toEqual(expectedLines(s, sec, b));
    expect(positionsOf(linesOf(bs, 1))[1]).toBeGreaterThan(3);

    // Dispose removes the planes and the geometries.
    let disposed = 0;
    for (const c of bs.group.children) {
      (c as THREE.Mesh).geometry.addEventListener("dispose", () => disposed++);
    }
    bs.dispose();
    bs.dispose();
    expect(disposed).toBe(4);
    expect(ma.clippingPlanes).toBeNull();
    expect(mb.clippingPlanes).toBeNull();
    // No longer following.
    const after = linesOf(bs, 0).geometry;
    s.setSectionPlane(sec, { origin: [0, 0, 0], normal: [0, 0, 1] });
    s.commitFrame();
    expect(linesOf(bs, 0).geometry).toBe(after);
  });

  it("clip_options_and_shared_planes", () => {
    const { s, ba, bb, sec } = setup();
    // An existing clipping plane is kept; a shared material gets the plane once.
    const own = new THREE.Plane(new THREE.Vector3(1, 0, 0), 2);
    const material = ba.object.material as THREE.Material;
    material.clippingPlanes = [own];
    bb.object.material = material;
    const bs = bindSection(s, sec, [ba, bb]);
    expect(material.clippingPlanes).toEqual([own, bs.plane]);
    bs.dispose();
    expect(material.clippingPlanes).toEqual([own]);

    // clip: false leaves the materials alone.
    const plain = bindSection(s, sec, [ba], { clip: false });
    expect(material.clippingPlanes).toEqual([own]);
    plain.dispose();
    expect(material.clippingPlanes).toEqual([own]);
  });

  it("material_arrays_are_each_clipped", () => {
    const { s, ba, sec } = setup();
    const first = new THREE.MeshStandardMaterial();
    const second = new THREE.MeshStandardMaterial();
    ba.object.material = [first, second];
    const bs = bindSection(s, sec, [ba]);
    expect(first.clippingPlanes).toEqual([bs.plane]);
    expect(second.clippingPlanes).toEqual([bs.plane]);
    bs.dispose();
    expect(first.clippingPlanes).toBeNull();
    expect(second.clippingPlanes).toBeNull();
  });

  it("plane_follows_set_section_plane_for_lazy_meshes", () => {
    const { s, a, b, ba, bb, sec } = setup();
    s.setMode(a, "lazy");
    s.setMode(b, "lazy");
    s.commitFrame();
    const bs = bindSection(s, sec, [ba, bb]);
    const before = positionsOf(linesOf(bs, 0));
    s.setSectionPlane(sec, { origin: [0, 0, 0.25], normal: [0, 0, 1] });
    s.commitFrame();
    // The plane moves on the next frame; the lazy results (and lines) lag.
    expect(bs.plane.constant).toBeCloseTo(-0.25, 12);
    expect(s.section(sec, a).status).toBe("stale");
    expect(positionsOf(linesOf(bs, 0))).toEqual(before);
    // Once computed, the lines follow.
    s.updateSections();
    s.commitFrame();
    expect(positionsOf(linesOf(bs, 0))).toEqual(expectedLines(s, sec, a));
    expect(positionsOf(linesOf(bs, 0))).not.toEqual(before);
    bs.dispose();
  });

  it("uncut_or_removed_meshes_and_sections_draw_nothing", () => {
    const { s, a, b, ba, bb, sec } = setup();
    s.setSectionMeshes(sec, [a]);
    s.commitFrame();
    const bs = bindSection(s, sec, [ba, bb]);
    expect(positionsOf(linesOf(bs, 0)).length).toBeGreaterThan(0);
    expect(positionsOf(linesOf(bs, 1))).toEqual([]);
    // A removed mesh empties its drawing.
    s.removeMesh(a);
    s.commitFrame();
    expect(positionsOf(linesOf(bs, 0))).toEqual([]);
    // A removed section empties everything and stops following.
    s.setSectionMeshes(sec, [b]);
    s.commitFrame();
    expect(positionsOf(linesOf(bs, 1)).length).toBeGreaterThan(0);
    s.removeSection(sec);
    s.commitFrame();
    expect(positionsOf(linesOf(bs, 1))).toEqual([]);
    // Its clipping goes with it.
    expect((bb.object.material as THREE.Material).clippingPlanes).toBeNull();
    bs.dispose();
    expect((bb.object.material as THREE.Material).clippingPlanes).toBeNull();
    ba.dispose();
    bb.dispose();
  });
});
