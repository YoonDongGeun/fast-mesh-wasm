// The object info's ux logic (#257) against the real engine: the counts,
// edges and components of one pass over the index buffer agree with the
// SDK's `triangleCount`, `vertexCount` and `analyze`; the boxes with its
// `bounds`, origin and placement; area and volume are its `measure` (on a
// local scene and through promises, as a WorkerScene answers); the world
// box tracker walks the vertices only when the matrix's linear part changes;
// and which change records count as geometry, placement or removal.
import { init, type MeshChange, type MeshHandle, Scene } from "fast-mesh-wasm";
import { afterEach, beforeAll, describe, expect, it } from "vitest";
import {
  boxCenter,
  boxSize,
  type InfoScene,
  linearBox,
  meanEdgeLength,
  meshTopology,
  offsetBox,
  readMeshInfo,
  WorldBoxTracker,
} from "../src/features/inspect/ux/object-info";
import { meshChangeKind } from "../src/features/inspect/ux/use-object-info";

beforeAll(async () => {
  await init();
});

const scenes: Scene[] = [];
afterEach(() => {
  for (const s of scenes.splice(0)) {
    s.free();
  }
});

/** An axis-aligned box from `lo` to `hi`, outward-wound: 8 vertices, 12 triangles. */
function box(lo: [number, number, number], hi: [number, number, number], base = 0): { positions: number[]; indices: number[] } {
  const p: number[] = [];
  for (const z of [lo[2], hi[2]]) {
    for (const y of [lo[1], hi[1]]) {
      for (const x of [lo[0], hi[0]]) {
        p.push(x, y, z);
      }
    }
  }
  const quads = [
    [0, 2, 3, 1],
    [4, 5, 7, 6],
    [0, 1, 5, 4],
    [2, 6, 7, 3],
    [0, 4, 6, 2],
    [1, 3, 7, 5],
  ];
  const t: number[] = [];
  for (const [a, b, c, d] of quads) {
    t.push(a + base, b + base, c + base, a + base, c + base, d + base);
  }
  return { positions: p, indices: t };
}

function add(scene: Scene, positions: number[], indices: number[], origin: [number, number, number] = [0, 0, 0]): MeshHandle {
  const m = scene.addMesh(Float64Array.from(positions), Uint32Array.from(indices), { origin });
  scene.commitFrame();
  return m;
}

function newScene(): Scene {
  const scene = new Scene();
  scenes.push(scene);
  return scene;
}

function topologyOf(scene: Scene, mesh: MeshHandle) {
  const views = scene.views(mesh);
  return meshTopology(views.indices(), views.positions().length / 3);
}

/** The scene with `measure` answering through a promise, as a WorkerScene does. */
function asyncScene(scene: Scene): InfoScene {
  return {
    views: (m) => scene.views(m),
    matrix4: (m) => scene.matrix4(m),
    origin: (m) => scene.origin(m),
    inspect: { measure: (m) => Promise.resolve(scene.inspect.measure(m)) },
  };
}

describe("meshTopology", () => {
  it("counts a closed box like the SDK: 12 triangles, 8 vertices, 18 edges, one component", () => {
    const scene = newScene();
    const c = box([-1, -1, -1], [1, 1, 1]);
    const m = add(scene, c.positions, c.indices);
    const topo = topologyOf(scene, m);
    expect(topo.triangles).toBe(scene.triangleCount(m));
    expect(topo.used.length).toBe(scene.vertexCount(m));
    expect(topo.edges.length / 2).toBe(18);
    expect(topo.components).toBe(scene.inspect.analyze(m).components.length);
    expect(topo.components).toBe(1);
    // Every edge once, lower vertex first, ascending.
    for (let i = 0; i < topo.edges.length; i += 2) {
      expect(topo.edges[i]).toBeLessThan(topo.edges[i + 1]);
    }
  });

  it("counts components, open and non-manifold edges the way analyze does", () => {
    const scene = newScene();
    // Two separate boxes, an open fan, and three triangles on one edge.
    const a = box([0, 0, 0], [1, 1, 1]);
    const b = box([3, 0, 0], [4, 1, 1], 8);
    const positions = [...a.positions, ...b.positions, 0, 0, 5, 1, 0, 5, 0, 1, 5, 1, 1, 5, 0, 0, 8, 1, 0, 8, 0.5, 1, 8, 0.5, -1, 8, 0.5, 0, 9];
    const indices = [...a.indices, ...b.indices, 16, 17, 18, 17, 19, 18, 20, 21, 22, 21, 20, 23, 20, 21, 24];
    const m = add(scene, positions, indices);
    const topo = topologyOf(scene, m);
    const analysis = scene.inspect.analyze(m);
    expect(analysis.nonManifoldEdges).toBe(1);
    expect(topo.components).toBe(analysis.components.length);
    expect(topo.components).toBe(4);
    expect(topo.triangles).toBe(scene.triangleCount(m));
    expect(topo.used.length).toBe(scene.vertexCount(m));
    // 18 + 18 for the boxes, 5 for the fan, 7 for the three triangles sharing an edge.
    expect(topo.edges.length / 2).toBe(18 + 18 + 5 + 7);
  });

  it("skips removed triangle slots and the vertices only they used", () => {
    const scene = newScene();
    const a = box([0, 0, 0], [1, 1, 1]);
    const b = box([3, 0, 0], [4, 1, 1], 8);
    const m = add(scene, [...a.positions, ...b.positions], [...a.indices, ...b.indices]);
    // Remove the second box's twelve triangles: their slots read 0, 0, 0.
    scene.edit(m, [{ op: "removeTriangles", triangles: Uint32Array.from({ length: 12 }, (_, i) => 12 + i) }]);
    scene.commitFrame();
    const topo = topologyOf(scene, m);
    expect(topo.triangles).toBe(scene.triangleCount(m));
    expect(topo.triangles).toBe(12);
    expect(topo.components).toBe(scene.inspect.analyze(m).components.length);
    expect(topo.used.length).toBe(8);
    expect(topo.edges.length / 2).toBe(18);
    // The box is the SDK's bounds: the dead slots' stale positions do not count.
    const views = scene.views(m);
    const local = linearBox(views.positions(), topo.used, null);
    const bounds = scene.bounds(m);
    expect(local).toEqual({ min: bounds?.min, max: bounds?.max });
  });

  it("handles an empty index buffer", () => {
    const topo = meshTopology(new Uint32Array(0), 0);
    expect(topo).toEqual({ triangles: 0, used: new Uint32Array(0), edges: new Uint32Array(0), components: 0 });
    expect(linearBox(new Float32Array(0), topo.used, null)).toBeNull();
    expect(meanEdgeLength(new Float32Array(0), topo.edges, null)).toBe(0);
  });
});

describe("boxes and edge lengths", () => {
  it("reads the local box in the mesh's frame and the world box through its placement", async () => {
    const scene = newScene();
    const c = box([1, 2, 3], [3, 6, 4]);
    // Positions are stored relative to the origin: the local box adds it back.
    const m = add(scene, c.positions, c.indices, [2, 4, 3.5]);
    const info = await readMeshInfo(scene, m);
    expect(info.local).toEqual({ min: [1, 2, 3], max: [3, 6, 4] });
    expect(info.world).toEqual(info.local);
    const bounds = scene.bounds(m);
    expect(info.local).toEqual(offsetBox({ min: bounds!.min, max: bounds!.max }, scene.origin(m)));

    // A quarter turn about z and a move: the world box turns, the local one stays.
    scene.setPlacement(m, { rotation: [0, -1, 0, 1, 0, 0, 0, 0, 1], translation: [10, 0, 0] });
    scene.commitFrame();
    const moved = await readMeshInfo(scene, m);
    expect(moved.local).toEqual(info.local);
    expect(boxSize(moved.world!)).toEqual([4, 2, 1].map((x) => expect.closeTo(x, 9)));
    // world = R (origin + p) + t: the centre (2, 4, 3.5) goes to (-4 + 10, 2, 3.5).
    expect(boxCenter(moved.world!)).toEqual([6, 2, 3.5].map((x) => expect.closeTo(x, 9)));
  });

  it("averages the unique edges, with the placement's scale", async () => {
    const scene = newScene();
    const c = box([-1, -1, -1], [1, 1, 1]);
    const m = add(scene, c.positions, c.indices);
    const expected = (12 * 2 + 6 * 2 * Math.SQRT2) / 18;
    const info = await readMeshInfo(scene, m);
    expect(info.meanEdge).toBeCloseTo(expected, 6);
    expect(info.edges).toBe(18);

    scene.setPlacement(m, { rotation: [1, 0, 0, 0, 1, 0, 0, 0, 1], translation: [5, 5, 5], scale: [2, 2, 2] });
    scene.commitFrame();
    const scaled = await readMeshInfo(scene, m);
    expect(scaled.meanEdge).toBeCloseTo(2 * expected, 6);
    // The SDK measures scaled too.
    expect(scaled.area).toBeCloseTo(scene.inspect.measure(m).area, 9);
    expect(scaled.area).toBeCloseTo(4 * 24, 4);
  });

  it("tracks the world box: a translation shifts it, a rotation walks the vertices again", () => {
    const scene = newScene();
    const c = box([0, 0, 0], [2, 1, 1]);
    const m = add(scene, c.positions, c.indices);
    const views = scene.views(m);
    const positions = Float32Array.from(views.positions());
    const topo = meshTopology(views.indices(), positions.length / 3);
    const tracker = new WorldBoxTracker(positions, topo.used);
    const at = (t: [number, number, number], r = [1, 0, 0, 0, 1, 0, 0, 0, 1]) => {
      scene.setPlacement(m, { rotation: r, translation: t });
      return scene.matrix4(m);
    };
    const first = tracker.box(at([0, 0, 0]));
    expect(first).toEqual({ min: [0, 0, 0], max: [2, 1, 1] });
    const shifted = tracker.box(at([1, 2, 3]));
    expect(shifted).toEqual({ min: [1, 2, 3], max: [3, 3, 4] });
    expect(tracker.walks).toBe(1);
    const turnedM = at([0, 0, 0], [0, -1, 0, 1, 0, 0, 0, 0, 1]);
    const turned = tracker.box(turnedM);
    expect(tracker.walks).toBe(2);
    const exact = linearBox(positions, topo.used, turnedM)!;
    expect(turned).toEqual(offsetBox(exact, [turnedM[12], turnedM[13], turnedM[14]]));
    expect(boxSize(turned!)).toEqual([1, 2, 1].map((x) => expect.closeTo(x, 9)));
  });
});

describe("readMeshInfo", () => {
  it("is the SDK's measure for area and volume, locally and through promises", async () => {
    const scene = newScene();
    const c = box([-1, -1, -1], [1, 1, 1]);
    const m = add(scene, c.positions, c.indices);
    const sdk = scene.inspect.measure(m);
    for (const s of [scene, asyncScene(scene)]) {
      const info = await readMeshInfo(s, m);
      expect(info.area).toBe(sdk.area);
      expect(info.volume).toBe(sdk.volume);
      expect(info.volumeReason).toBeNull();
      expect(info).toMatchObject({ triangles: 12, vertices: 8, edges: 18, components: 1 });
    }
    expect(sdk.volume).toBeCloseTo(8, 6);
  });

  it("has no volume for an open mesh, and says why", async () => {
    const scene = newScene();
    const c = box([-1, -1, -1], [1, 1, 1]);
    const m = add(scene, c.positions, c.indices.slice(0, 30));
    const info = await readMeshInfo(asyncScene(scene), m);
    expect(info.volume).toBeNull();
    expect(info.volumeReason).toBe("open");
    expect(info.area).toBe(scene.inspect.measure(m).area);
    expect(info.triangles).toBe(10);
  });

  it("reads no measurement when the query fails, and throws when the mesh is gone", async () => {
    const scene = newScene();
    const c = box([-1, -1, -1], [1, 1, 1]);
    const m = add(scene, c.positions, c.indices);
    const failing: InfoScene = { ...asyncScene(scene), inspect: { measure: () => Promise.reject(new Error("gone")) } };
    const info = await readMeshInfo(failing, m);
    expect(info).toMatchObject({ area: null, volume: null, volumeReason: null, triangles: 12 });
    scene.removeMesh(m);
    scene.commitFrame();
    await expect(readMeshInfo(scene, m)).rejects.toThrow();
  });
});

describe("meshChangeKind", () => {
  const record = (mesh: MeshHandle, patch: Partial<MeshChange>): MeshChange => ({
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
    ...patch,
  });

  it("tells geometry, placement and removal of the watched mesh apart", () => {
    const scene = newScene();
    const c = box([0, 0, 0], [1, 1, 1]);
    const a = add(scene, c.positions, c.indices);
    const b = add(scene, c.positions, c.indices);
    expect(meshChangeKind([record(a, { placementChanged: true })], a)).toBe("placement");
    expect(meshChangeKind([record(a, { positions: [{ start: 0, end: 1 }] as never })], a)).toBe("geometry");
    expect(meshChangeKind([record(a, { indices: "all" })], a)).toBe("geometry");
    expect(meshChangeKind([record(a, { placementChanged: true }), record(a, { capacityChanged: true })], a)).toBe("geometry");
    expect(meshChangeKind([record(a, { colors: "all" })], a)).toBeNull();
    expect(meshChangeKind([record(b, { indices: "all" })], a)).toBeNull();
    expect(meshChangeKind([record(a, { indices: "all" }), record(a, { removed: true })], a)).toBe("removed");
  });
});
