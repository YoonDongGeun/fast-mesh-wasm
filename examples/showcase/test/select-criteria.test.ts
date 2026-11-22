// Face selection by criteria (#258), the main-thread tests: each criterion
// on a mesh built so the expected faces are known, edge tests checked
// against a brute-force edge count, and the slicing that keeps a million
// triangles interactive.
import { afterEach, describe, expect, it, vi } from "vitest";
import {
  buildEdgeAdjacency,
  expandToComponents,
  IDENTITY_MATRIX,
  type MeshGeometry,
  type Pause,
  selectBoundaryFaces,
  selectByArea,
  selectFacingCamera,
  selectInconsistentOrientation,
  selectOverhangs,
  selectSharpFaces,
  selectSlivers,
  unionSlots,
} from "../src/features/select/ux/face-criteria";
import { buildAxisVector } from "../src/features/select/ux/select-criteria";

const noPause: Pause = () => Promise.resolve();

afterEach(() => {
  vi.restoreAllMocks();
});

function geometry(positions: number[], indices: number[], matrix: ArrayLike<number> = IDENTITY_MATRIX): MeshGeometry {
  return { positions: Float32Array.from(positions), indices: Uint32Array.from(indices), matrix };
}

/** The closed box [-1, 1]^3 wound outward: 12 triangles, two per face in the order -Z, +Z, -Y, +Y, -X, +X. */
function box(): MeshGeometry {
  return geometry(
    [-1, -1, -1, 1, -1, -1, 1, 1, -1, -1, 1, -1, -1, -1, 1, 1, -1, 1, 1, 1, 1, -1, 1, 1],
    [0, 2, 1, 0, 3, 2, 4, 5, 6, 4, 6, 7, 0, 1, 5, 0, 5, 4, 2, 3, 7, 2, 7, 6, 0, 4, 7, 0, 7, 3, 1, 2, 6, 1, 6, 5],
  );
}

/** An n x n grid of unit quads in z = 0, each split along its (i, j)-(i+1, j+1) diagonal, wound towards +z. */
function grid(n: number, lift?: (x: number, y: number) => number): MeshGeometry {
  const positions: number[] = [];
  for (let j = 0; j <= n; j++) {
    for (let i = 0; i <= n; i++) {
      positions.push(i, j, lift ? lift(i, j) : 0);
    }
  }
  const v = (i: number, j: number) => j * (n + 1) + i;
  const indices: number[] = [];
  for (let j = 0; j < n; j++) {
    for (let i = 0; i < n; i++) {
      indices.push(v(i, j), v(i + 1, j), v(i + 1, j + 1), v(i, j), v(i + 1, j + 1), v(i, j + 1));
    }
  }
  return geometry(positions, indices);
}

/** The edges of every live triangle with how many triangles use them, counted with a map. */
function edgeUse(indices: Uint32Array): Map<string, number> {
  const uses = new Map<string, number>();
  for (let t = 0; t < indices.length / 3; t++) {
    const tri = [indices[3 * t], indices[3 * t + 1], indices[3 * t + 2]];
    if (tri[0] === tri[1] || tri[1] === tri[2] || tri[0] === tri[2]) continue;
    for (let k = 0; k < 3; k++) {
      const a = tri[k];
      const b = tri[(k + 1) % 3];
      const key = a < b ? `${a},${b}` : `${b},${a}`;
      uses.set(key, (uses.get(key) ?? 0) + 1);
    }
  }
  return uses;
}

function bruteBoundaryFaces(indices: Uint32Array): number[] {
  const uses = edgeUse(indices);
  const out: number[] = [];
  for (let t = 0; t < indices.length / 3; t++) {
    const tri = [indices[3 * t], indices[3 * t + 1], indices[3 * t + 2]];
    if (tri[0] === tri[1] || tri[1] === tri[2] || tri[0] === tri[2]) continue;
    const open = [0, 1, 2].some((k) => {
      const a = tri[k];
      const b = tri[(k + 1) % 3];
      return uses.get(a < b ? `${a},${b}` : `${b},${a}`) === 1;
    });
    if (open) out.push(t);
  }
  return out;
}

function normalOf(g: MeshGeometry, t: number): [number, number, number] {
  const p = (k: number) => {
    const o = g.indices[3 * t + k] * 3;
    return [g.positions[o], g.positions[o + 1], g.positions[o + 2]];
  };
  const [a, b, c] = [p(0), p(1), p(2)];
  const u = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
  const w = [c[0] - a[0], c[1] - a[1], c[2] - a[2]];
  const n: [number, number, number] = [u[1] * w[2] - u[2] * w[1], u[2] * w[0] - u[0] * w[2], u[0] * w[1] - u[1] * w[0]];
  const len = Math.hypot(...n);
  return [n[0] / len, n[1] / len, n[2] / len];
}

/** Faces with an edge whose two faces' normals are at least `deg` apart, every pair of faces tested. */
function bruteSharpFaces(g: MeshGeometry, deg: number): number[] {
  const tris = g.indices.length / 3;
  const hit = new Set<number>();
  for (let s = 0; s < tris; s++) {
    for (let t = s + 1; t < tris; t++) {
      const vs = [g.indices[3 * s], g.indices[3 * s + 1], g.indices[3 * s + 2]];
      const shared = [g.indices[3 * t], g.indices[3 * t + 1], g.indices[3 * t + 2]].filter((v) => vs.includes(v));
      if (shared.length !== 2) continue;
      const [n, m] = [normalOf(g, s), normalOf(g, t)];
      const angle = (Math.acos(Math.min(1, Math.max(-1, n[0] * m[0] + n[1] * m[1] + n[2] * m[2]))) * 180) / Math.PI;
      if (angle >= deg) {
        hit.add(s);
        hit.add(t);
      }
    }
  }
  return [...hit].sort((a, b) => a - b);
}

describe("area", () => {
  // Three triangles of area 0.5, 2 and 8, a dead slot, and a collinear one (area 0).
  const g = geometry(
    [0, 0, 0, 1, 0, 0, 0, 1, 0, 2, 0, 0, 0, 2, 0, 4, 0, 0, 0, 4, 0, 8, 0, 0],
    [0, 1, 2, 0, 3, 4, 0, 0, 0, 0, 5, 6, 0, 1, 7],
  );

  it("selects faces smaller or larger than the threshold, skipping dead slots", async () => {
    expect([...(await selectByArea(g, { mode: "below", threshold: 1 }, noPause))]).toEqual([0, 4]);
    expect([...(await selectByArea(g, { mode: "above", threshold: 1 }, noPause))]).toEqual([1, 3]);
    expect([...(await selectByArea(g, { mode: "above", threshold: 2 }, noPause))]).toEqual([3]);
    expect([...(await selectByArea(g, { mode: "below", threshold: 0 }, noPause))]).toEqual([]);
  });

  it("measures in the world, through the placement", async () => {
    // Scaled by 2 along x and y: every area times 4.
    const scaled = { ...g, matrix: [2, 0, 0, 0, 0, 2, 0, 0, 0, 0, 1, 0, 5, 6, 7, 1] };
    expect([...(await selectByArea(scaled, { mode: "above", threshold: 1.5 }, noPause))]).toEqual([0, 1, 3]);
    expect([...(await selectByArea(scaled, { mode: "below", threshold: 1.5 }, noPause))]).toEqual([4]);
    expect([...(await selectByArea(scaled, { mode: "below", threshold: 2.5 }, noPause))]).toEqual([0, 4]);
  });
});

describe("facing the camera", () => {
  it("selects faces within the angle of an orthographic view direction", async () => {
    const g = box();
    const camera = { eye: [0, 0, 10] as [number, number, number], forward: [0, 0, -1] as [number, number, number], orthographic: true };
    expect([...(await selectFacingCamera(g, camera, 10, noPause))]).toEqual([2, 3]);
    // Walls are at exactly 90°.
    expect([...(await selectFacingCamera(g, camera, 89, noPause))]).toEqual([2, 3]);
    expect((await selectFacingCamera(g, camera, 91, noPause)).length).toBe(10);
    expect((await selectFacingCamera(g, camera, 180, noPause)).length).toBe(12);
  });

  it("takes the direction to the eye per face for a perspective camera", async () => {
    const g = box();
    // From the +x side, above: +X faces it squarely, the top only at a slant
    // (about 79° from each top face's centre), the other walls face away.
    const camera = { eye: [10, 0, 3] as [number, number, number], forward: [-1, 0, 0] as [number, number, number], orthographic: false };
    expect([...(await selectFacingCamera(g, camera, 30, noPause))]).toEqual([10, 11]);
    expect([...(await selectFacingCamera(g, camera, 75, noPause))]).toEqual([10, 11]);
    expect([...(await selectFacingCamera(g, camera, 89.9, noPause))]).toEqual([2, 3, 10, 11]);
  });

  it("follows the placement", async () => {
    // Rotated a quarter turn about y (+Z goes to +X): the old +X faces now face -Z.
    const g = { ...box(), matrix: [0, 0, -1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1] };
    const camera = { eye: [0, 0, -10] as [number, number, number], forward: [0, 0, 1] as [number, number, number], orthographic: true };
    expect([...(await selectFacingCamera(g, camera, 10, noPause))]).toEqual([10, 11]);
  });
});

describe("overhangs", () => {
  it("selects faces past the critical angle below horizontal", async () => {
    const g = box();
    expect([...(await selectOverhangs(g, [0, 0, 1], 45, noPause))]).toEqual([0, 1]);
    expect([...(await selectOverhangs(g, [0, 0, 1], 89.9, noPause))]).toEqual([0, 1]);
    // Strictly past: a flat ceiling is 90°.
    expect([...(await selectOverhangs(g, [0, 0, 1], 90, noPause))]).toEqual([]);
    expect([...(await selectOverhangs(g, [0, 0, -1], 45, noPause))]).toEqual([2, 3]);
    expect([...(await selectOverhangs(g, buildAxisVector("-x"), 10, noPause))]).toEqual([10, 11]);
  });

  it("reads tilted faces by their angle past horizontal", async () => {
    // Normals 30° and 60° below horizontal (down-facing ramps), and one 30° above.
    const ramp = (deg: number) => {
      const r = (deg * Math.PI) / 180;
      // The normal (cos r, 0, -sin r) for a positive r.
      return [0, 0, 0, 0, 1, 0, Math.sin(r), 0, Math.cos(r)];
    };
    const g = geometry([...ramp(30), ...ramp(60), ...ramp(-30)], [0, 1, 2, 3, 4, 5, 6, 7, 8]);
    expect([...(await selectOverhangs(g, [0, 0, 1], 45, noPause))]).toEqual([1]);
    expect([...(await selectOverhangs(g, [0, 0, 1], 20, noPause))]).toEqual([0, 1]);
    await expect(selectOverhangs(g, [0, 0, 0], 45, noPause)).rejects.toThrow(/direction/);
    await expect(selectOverhangs(g, [0, 0, 1], 200, noPause)).rejects.toThrow(/critical angle/);
  });
});

describe("slivers", () => {
  it("selects faces whose circumradius is many inradii, leaving zero-area ones out", async () => {
    const h = Math.sqrt(3) / 2;
    const g = geometry(
      [0, 0, 0, 1, 0, 0, 0.5, h, 0, 0, 0, 0, 10, 0, 0, 5, 0.05, 0, 0, 0, 0, 1, 0, 0, 2, 0, 0],
      [0, 1, 2, 3, 4, 5, 6, 7, 8],
    );
    expect([...(await selectSlivers(g, 20, noPause))]).toEqual([1]);
    // The equilateral one reads 1 (up to the f32 corners).
    expect([...(await selectSlivers(g, 0.99, noPause))]).toEqual([0, 1]);
  });
});

describe("edge adjacency", () => {
  it("names each edge's other face, boundary and non-manifold edges, and components", async () => {
    // A fin: three triangles on the edge 0-1, and a separate triangle.
    const g = geometry(
      [0, 0, 0, 1, 0, 0, 0, 1, 0, 0, -1, 0, 0, 0, 1, 5, 5, 5, 6, 5, 5, 5, 6, 5],
      [0, 1, 2, 1, 0, 3, 0, 1, 4, 5, 6, 7, 0, 0, 0],
    );
    const adj = await buildEdgeAdjacency(g.indices, 8, noPause);
    expect([...adj.proper]).toEqual([1, 1, 1, 1, 0]);
    expect(adj.across[0]).toBe(-2);
    expect(adj.across[3]).toBe(-2);
    expect(adj.across[1]).toBe(-1);
    expect([...adj.component.subarray(0, 4)]).toEqual([0, 0, 0, 3]);
    expect(adj.open[0]).toBe(1);
    expect(adj.open[3]).toBe(1);

    const closed = await buildEdgeAdjacency(box().indices, 8, noPause);
    expect(closed.open[0]).toBe(0);
    for (let c = 0; c < 36; c++) {
      expect(closed.across[c]).toBeGreaterThanOrEqual(0);
    }
  });
});

describe("boundary-adjacent", () => {
  it("matches a brute-force edge count on an open grid", async () => {
    const g = grid(3);
    const adj = await buildEdgeAdjacency(g.indices, 16, noPause);
    const byEdge = await selectBoundaryFaces(g.indices, 16, adj, "edge", noPause);
    expect([...byEdge]).toEqual(bruteBoundaryFaces(g.indices));
    // Per quad, one triangle holds the bottom and right edges, the other the top and left.
    expect(byEdge.length).toBe(10);
    // Touching a boundary vertex: all but the centre quad.
    const byVertex = await selectBoundaryFaces(g.indices, 16, adj, "vertex", noPause);
    expect(byVertex.length).toBe(16);
    expect([...byVertex]).not.toContain(8);
    expect([...byVertex]).not.toContain(9);
  });

  it("finds the rim of a hole in a closed box, and nothing on the closed box", async () => {
    const g = box();
    const closed = await buildEdgeAdjacency(g.indices, 8, noPause);
    expect((await selectBoundaryFaces(g.indices, 8, closed, "edge", noPause)).length).toBe(0);
    // Remove the +Z face (slots 2 and 3 become dead).
    const holed = g.indices.slice();
    holed.fill(0, 6, 12);
    const adj = await buildEdgeAdjacency(holed, 8, noPause);
    const rim = await selectBoundaryFaces(holed, 8, adj, "edge", noPause);
    expect([...rim]).toEqual(bruteBoundaryFaces(holed));
    // One triangle of each wall holds a top edge.
    expect(rim.length).toBe(4);
  });
});

describe("sharp edges", () => {
  it("selects the faces along creases past the angle", async () => {
    const g = box();
    const adj = await buildEdgeAdjacency(g.indices, 8, noPause);
    expect((await selectSharpFaces(g, adj, 45, noPause)).length).toBe(12);
    expect((await selectSharpFaces(g, adj, 91, noPause)).length).toBe(0);
  });

  it("matches a brute-force dihedral test on a bumped grid", async () => {
    const g = grid(6, (x, y) => (x === 3 && y === 3 ? 1.2 : x === 1 && y === 4 ? 0.3 : 0));
    const adj = await buildEdgeAdjacency(g.indices, 49, noPause);
    for (const deg of [5, 20, 40, 60]) {
      expect([...(await selectSharpFaces(g, adj, deg, noPause))]).toEqual(bruteSharpFaces(g, deg));
    }
    expect((await selectSharpFaces(grid(4), await buildEdgeAdjacency(grid(4).indices, 25, noPause), 1, noPause)).length).toBe(0);
  });
});

describe("inconsistent orientation", () => {
  async function wrong(g: MeshGeometry): Promise<number[]> {
    const adj = await buildEdgeAdjacency(g.indices, g.positions.length / 3, noPause);
    return [...(await selectInconsistentOrientation(g, adj, noPause))];
  }

  function flipped(g: MeshGeometry, tris: number[]): MeshGeometry {
    const indices = g.indices.slice();
    for (const t of tris) {
      const b = indices[3 * t + 1];
      indices[3 * t + 1] = indices[3 * t + 2];
      indices[3 * t + 2] = b;
    }
    return { ...g, indices };
  }

  it("is empty on a consistent mesh", async () => {
    expect(await wrong(box())).toEqual([]);
    expect(await wrong(grid(4))).toEqual([]);
  });

  it("selects the smaller patch of an open mesh", async () => {
    // Flip 3 of the 32 faces: they become the minority patches.
    expect(await wrong(flipped(grid(4), [5, 6, 20]))).toEqual([5, 6, 20]);
    // Flip most of it: the unflipped rest is now the minority.
    const most = Array.from({ length: 32 }, (_, t) => t).filter((t) => t !== 0 && t !== 1);
    expect(await wrong(flipped(grid(4), most))).toEqual([0, 1]);
  });

  it("turns a closed shell outward", async () => {
    // One face of a box flipped.
    expect(await wrong(flipped(box(), [7]))).toEqual([7]);
    // The whole box inside out: every face is wrong.
    const all = Array.from({ length: 12 }, (_, t) => t);
    expect(await wrong(flipped(box(), all))).toEqual(all);
    // Inside out but for one face: that face is the only one facing out, and stays.
    expect(await wrong(flipped(box(), all.filter((t) => t !== 4)))).toEqual(all.filter((t) => t !== 4));
  });
});

describe("components and unions", () => {
  it("grows a selection to the components it touches", () => {
    const components = [{ triangles: Uint32Array.of(0, 1, 2) }, { triangles: Uint32Array.of(3, 4) }, { triangles: Uint32Array.of(5) }];
    expect([...expandToComponents(components, [1], 6)]).toEqual([0, 1, 2]);
    expect([...expandToComponents(components, [5, 3], 6)]).toEqual([3, 4, 5]);
    expect([...expandToComponents(components, [], 6)]).toEqual([]);
    expect([...unionSlots([[4, 1], Uint32Array.of(1, 2), [9]], 6)]).toEqual([1, 2, 4]);
  });
});

describe("slicing", () => {
  it("yields between slices once a slice's time budget is used", async () => {
    // Every clock read moves 5 ms: a slice ends at its second check.
    let clock = 0;
    vi.spyOn(performance, "now").mockImplementation(() => (clock += 5));
    let pauses = 0;
    const counting: Pause = () => {
      pauses++;
      return Promise.resolve();
    };
    const g = grid(100);
    const result = await selectByArea(g, { mode: "above", threshold: 0.25 }, counting);
    expect(result.length).toBe(20_000);
    // 20,000 triangles at a check per 4,096: about two pauses.
    expect(pauses).toBeGreaterThanOrEqual(2);
  });

  it("keeps every slice short on a million triangles", async () => {
    const n = 708; // 1,002,528 triangles
    const g = grid(n, (x, y) => ((x * 7 + y * 13) % 11 === 0 ? 0.5 : 0));
    let longest = 0;
    let last = performance.now();
    let pauses = 0;
    const timed: Pause = () => {
      const now = performance.now();
      longest = Math.max(longest, now - last);
      pauses++;
      return new Promise((resolve) =>
        setTimeout(() => {
          last = performance.now();
          resolve();
        }, 0),
      );
    };
    const adj = await buildEdgeAdjacency(g.indices, (n + 1) * (n + 1), timed);
    const sharp = await selectSharpFaces(g, adj, 10, timed);
    const wrongWound = await selectInconsistentOrientation(g, adj, timed);
    longest = Math.max(longest, performance.now() - last);
    expect(sharp.length).toBeGreaterThan(0);
    expect(wrongWound.length).toBe(0);
    expect(pauses).toBeGreaterThan(10);
    // The budget is 8 ms; leave room for a slow or busy machine.
    expect(longest).toBeLessThan(150);
  }, 60_000);
});
