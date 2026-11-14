import { afterEach, describe, expect, it } from "vitest";
import { type AutoRepairPreset, Scene } from "../src/node.js";

const scenes: Scene[] = [];
function createScene(): Scene {
  const s = new Scene();
  scenes.push(s);
  return s;
}

afterEach(() => {
  for (const s of scenes.splice(0)) {
    s.free();
  }
});

function boxMesh(min: [number, number, number], max: [number, number, number]) {
  const [x0, y0, z0] = min;
  const [x1, y1, z1] = max;
  const positions = [
    x0, y0, z0, x1, y0, z0, x1, y1, z0, x0, y1, z0,
    x0, y0, z1, x1, y0, z1, x1, y1, z1, x0, y1, z1,
  ];
  const indices = [
    0, 2, 1, 0, 3, 2,
    4, 5, 6, 4, 6, 7,
    0, 1, 5, 0, 5, 4,
    2, 3, 7, 2, 7, 6,
    0, 4, 7, 0, 7, 3,
    1, 2, 6, 1, 6, 5,
  ];
  return { positions, indices };
}

function uvSphere(stacks: number, slices: number, radius = 1) {
  const positions = [0, 0, radius];
  for (let i = 1; i < stacks; i++) {
    const phi = (Math.PI * i) / stacks;
    for (let j = 0; j < slices; j++) {
      const theta = (2 * Math.PI * j) / slices;
      positions.push(
        radius * Math.sin(phi) * Math.cos(theta),
        radius * Math.sin(phi) * Math.sin(theta),
        radius * Math.cos(phi),
      );
    }
  }
  positions.push(0, 0, -radius);
  const south = positions.length / 3 - 1;
  const ring = (i: number, j: number) => 1 + (i - 1) * slices + (j % slices);
  const indices: number[] = [];
  for (let j = 0; j < slices; j++) {
    indices.push(0, ring(1, j), ring(1, j + 1));
    indices.push(south, ring(stacks - 1, j + 1), ring(stacks - 1, j));
    for (let i = 1; i < stacks - 1; i++) {
      const a = ring(i, j);
      const b = ring(i + 1, j);
      const c = ring(i + 1, j + 1);
      const d = ring(i, j + 1);
      indices.push(a, b, c, a, c, d);
    }
  }
  return { positions, indices };
}

describe("auto repair presets", () => {
  it("general preset preserves backward-compatible five stages and UV seams", () => {
    const s = createScene();
    const sph = uvSphere(8, 12, 1);
    const keep = sph.indices.slice(0, -6);
    const m = s.addMesh(sph.positions, keep);
    const rep = s.repair(m, { kind: "auto", preset: "general" });
    expect(Object.keys(rep)).toEqual([
      "weld",
      "removeDegenerate",
      "orient",
      "removeSmallComponents",
      "fillHoles",
    ]);
    expect(rep.fillHoles.filled.length).toBe(1);
    expect(s.measure(m).closed).toBe(true);
  });

  it("scan3d preset heals non-manifold fins, slivers, outlier debris, and holes in one undo step", () => {
    const s = createScene();
    const box = boxMesh([0, 0, 0], [2, 2, 2]);
    const extraOffset = box.positions.length / 3;
    const positions = [
      ...box.positions,
      0.5, 2.5, 0.5,
      10, 10, 10, 10.2, 10, 10, 10, 10.2, 10,
    ];
    const indices = [
      ...box.indices.slice(0, -3),
      6, 7, extraOffset,
      extraOffset + 1, extraOffset + 2, extraOffset + 3,
    ];
    const m = s.addMesh(positions, indices);
    const rep = s.repair(m, { kind: "auto", preset: "scan3d" });
    expect(rep.makeManifold).toBeDefined();
    expect(rep.cleanupSlivers).toBeDefined();
    expect(rep.makeManifold!.prunedFinTriangles).toBeGreaterThanOrEqual(1);
    expect(rep.removeSmallComponents.removedComponents).toBeGreaterThanOrEqual(1);
    expect(rep.fillHoles.filled.length).toBe(1);
    const a = s.analyze(m);
    expect(a.loops).toEqual([]);
    expect(a.nonManifoldEdges).toBe(0);
    expect(a.components.length).toBe(1);
    expect(s.measure(m).closed).toBe(true);

    expect(s.undo()).toBe(true);
    expect(s.canUndo()).toBe(false);
    expect(s.analyze(m).nonManifoldEdges).toBeGreaterThan(0);
  });

  it("print3d preset turns self-intersecting surface into watertight zero-SI solid and drops UVs only when volumetric fallback runs", () => {
    const s = createScene();
    const sph = uvSphere(10, 16, 1);
    for (let v = 0; v < sph.positions.length; v += 3) {
      if (sph.positions[v + 2] > 0.65) {
        sph.positions[v + 2] = -sph.positions[v + 2] * 1.4;
      }
    }
    const uvs = Array.from({ length: (sph.positions.length / 3) * 2 }, () => 0.5);
    const m = s.addMesh(sph.positions, sph.indices, { uvs });

    const rep = s.repair(m, { kind: "auto", preset: "print3d" });
    expect(rep.resolveSelfIntersections).toBeDefined();
    expect(rep.resolveSelfIntersections!.remainingPairs).toBe(0);
    expect(s.measure(m).closed).toBe(true);
    const a = s.analyze(m, { selfIntersection: true });
    expect(a.nonManifoldEdges).toBe(0);
    expect(a.selfIntersects).toBe(false);
    if (!rep.resolveSelfIntersections!.uvsDropped) {
      expect(s.views(m).uvs()).not.toBeNull();
    }

    expect(s.undo()).toBe(true);
    const repVox = s.repair(m, {
      kind: "auto",
      preset: "print3d",
      selfIntersectionMode: "voxelHeal",
      voxelSize: 0.12,
    });
    expect(repVox.resolveSelfIntersections!.modeUsed).toBe("voxelHeal");
    expect(repVox.resolveSelfIntersections!.uvsDropped).toBe(true);
    expect(s.views(m).uvs()).toBeNull();
    expect(s.measure(m).closed).toBe(true);
    expect(s.undo()).toBe(true);
    expect(s.views(m).uvs()).not.toBeNull();
  });

  it("cadImport preset preserves open boundaries by default while cleaning duplicates and non-manifold topology", () => {
    const s = createScene();
    const box = boxMesh([0, 0, 0], [4, 4, 4]);
    const finVert = box.positions.length / 3;
    const positions = [...box.positions, 2, 2, 5];
    const openIdx = [
      ...box.indices.slice(6),
      box.indices[6], box.indices[7], box.indices[8],
      4, 5, finVert,
    ];
    const m = s.addMesh(positions, openIdx);
    const rep = s.repair(m, { kind: "auto", preset: "cadImport" });
    expect(rep.removeDegenerate.removedTriangles).toBe(1);
    expect(rep.makeManifold).toBeDefined();
    expect(rep.makeManifold!.prunedFinTriangles).toBeGreaterThanOrEqual(1);
    expect(rep.cleanupSlivers).toBeDefined();
    const a = s.analyze(m);
    expect(a.duplicate).toBe(0);
    expect(a.nonManifoldEdges).toBe(0);
    expect(a.loops.length).toBe(1);
    expect(s.measure(m).closed).toBe(false);
  });

  it("validates preset names", () => {
    const s = createScene();
    const box = boxMesh([0, 0, 0], [1, 1, 1]);
    const m = s.addMesh(box.positions, box.indices);
    expect(() => s.repair(m, { kind: "auto", preset: "bogus" as AutoRepairPreset })).toThrow(/preset/);
  });
});
