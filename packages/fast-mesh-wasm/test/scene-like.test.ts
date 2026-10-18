// The three.js adapters depend only on SceneLike: a hand-written fake scene
// (in-memory buffers and a change emitter) drives bindMesh and bindSection.
import { describe, expect, expectTypeOf, it } from "vitest";
import type { MeshChange, SceneLike } from "../src/node.js";
import { MeshHandle, Scene, SectionHandle } from "../src/scene.js";
import type { DragSession, SectionPlane, SectionResult } from "../src/scene.js";
import type { BufferAttribute, LineSegments } from "three";
import { bindMesh, bindSection } from "../src/three/index.js";

const IDENTITY = [1, 0, 0, 0, 1, 0, 0, 0, 1];

function change(mesh: MeshHandle, over: Partial<MeshChange> = {}): MeshChange {
  return {
    mesh,
    added: false,
    removed: false,
    placementChanged: false,
    capacityChanged: false,
    positions: [],
    normals: [],
    colors: [],
    indices: [],
    fieldsChanged: [],
    sectionsChanged: [],
    ...over,
  };
}

/** One triangle in memory, one section, and a manual change emitter. */
class FakeScene implements SceneLike {
  readonly mesh = new MeshHandle();
  readonly section_ = new SectionHandle();
  readonly positions = Float32Array.from([0, 0, 0, 1, 0, 0, 0, 1, 0]);
  readonly normals = Float32Array.from([0, 0, 1, 0, 0, 1, 0, 0, 1]);
  readonly indices = Uint32Array.from([0, 1, 2]);
  plane: SectionPlane = { origin: [0, 0, 0], normal: [0, 0, 1] };
  result: SectionResult = {
    status: "current",
    contours: [{ points: Float32Array.from([0, 0, 0, 1, 0, 0]), closed: false }],
    length: 1,
    area: 0,
  };
  pending: MeshChange[] = [];
  listeners = new Set<(c: MeshChange[]) => void>();

  views() {
    return {
      positions: () => this.positions,
      normals: () => this.normals,
      colors: () => null,
      indices: () => this.indices,
    };
  }
  matrix4() {
    return Float64Array.from([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]);
  }
  placement() {
    return { rotation: IDENTITY, translation: [0, 0, 0] as [number, number, number] };
  }
  section() {
    return this.result;
  }
  sectionPlane() {
    return this.plane;
  }
  onChanges(listener: (c: MeshChange[]) => void) {
    this.listeners.add(listener);
    return () => void this.listeners.delete(listener);
  }
  commitFrame() {
    const changes = this.pending;
    this.pending = [];
    for (const l of this.listeners) {
      l(changes);
    }
    return changes;
  }
  drag(): DragSession {
    throw new Error("unused");
  }
}

describe("SceneLike", () => {
  it("Scene is assignable to SceneLike", () => {
    expectTypeOf<Scene>().toMatchTypeOf<SceneLike>();
  });

  it("adapters_accept_scene_like", () => {
    const fake = new FakeScene();
    const bound = bindMesh(fake, fake.mesh);
    const geometry = bound.object.geometry;
    expect(geometry.getAttribute("position").count).toBe(3);
    expect(geometry.getIndex()?.count).toBe(3);
    expect(fake.listeners.size).toBe(1);

    const section = bindSection(fake, fake.section_, [bound]);
    expect(section.plane.normal.toArray()).toEqual([0, 0, 1]);
    expect(section.group.children.length).toBe(2);
    const lines = section.group.children[0] as LineSegments;
    expect(lines.geometry.getAttribute("position").count).toBe(2);

    // A moved vertex arrives as a dirty range.
    fake.positions[3] = 5;
    fake.pending = [change(fake.mesh, { positions: [[1, 1]] })];
    fake.commitFrame();
    expect((geometry.getAttribute("position") as BufferAttribute).getX(1)).toBe(5);

    section.dispose();
    bound.dispose();
    expect(fake.listeners.size).toBe(0);
  });
});
