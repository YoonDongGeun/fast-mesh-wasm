// The angle measure and the object overlays' geometry (#257): the angle at
// the middle point, the click sequence, the arc and the label anchor; the
// bounding box's edges and label anchors; face and vertex normal lines
// (outward, the asked length, capped for large meshes, from the engine's
// own vertex normals when it has them); and the used vertices.
import { init, Scene } from "fast-mesh-wasm";
import { afterEach, beforeAll, describe, expect, it } from "vitest";
import {
  angleArc,
  angleDegrees,
  angleLabelAnchor,
  arcRadius,
  formatAngle,
  nextAnglePoints,
} from "../src/features/inspect/ux/angle-measure";
import { boxSize } from "../src/features/inspect/ux/object-info";
import {
  accumulateVertexNormals,
  boxEdges,
  boxLabelAnchors,
  faceNormalLines,
  MAX_NORMAL_LINES,
  usedVertices,
  vertexNormalLines,
} from "../src/features/inspect/ux/overlay-geometry";

beforeAll(async () => {
  await init();
});

const scenes: Scene[] = [];
afterEach(() => {
  for (const s of scenes.splice(0)) {
    s.free();
  }
});

/** The unit cube `[-1, 1]^3`, outward-wound. */
function cube(): { positions: Float64Array; indices: Uint32Array } {
  const p: number[] = [];
  for (const z of [-1, 1]) {
    for (const y of [-1, 1]) {
      for (const x of [-1, 1]) {
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
    t.push(a, b, c, a, c, d);
  }
  return { positions: Float64Array.from(p), indices: Uint32Array.from(t) };
}

/** An `n` by `n` grid of quads in the z = 0 plane, facing +z: `2 n²` triangles. */
function grid(n: number): { positions: Float32Array; indices: Uint32Array } {
  const p = new Float32Array((n + 1) * (n + 1) * 3);
  for (let j = 0; j <= n; j++) {
    for (let i = 0; i <= n; i++) {
      p.set([i, j, 0], 3 * (j * (n + 1) + i));
    }
  }
  const t: number[] = [];
  for (let j = 0; j < n; j++) {
    for (let i = 0; i < n; i++) {
      const a = j * (n + 1) + i;
      t.push(a, a + 1, a + n + 2, a, a + n + 2, a + n + 1);
    }
  }
  return { positions: p, indices: Uint32Array.from(t) };
}

/** The lines as `[from, to]` pairs. */
function segments(lines: Float32Array): [number[], number[]][] {
  const out: [number[], number[]][] = [];
  for (let i = 0; i < lines.length; i += 6) {
    out.push([Array.from(lines.subarray(i, i + 3)), Array.from(lines.subarray(i + 3, i + 6))]);
  }
  return out;
}

const len = (a: number[], b: number[]) => Math.hypot(b[0] - a[0], b[1] - a[1], b[2] - a[2]);

describe("angle measure", () => {
  it("measures the angle at the middle point", () => {
    expect(angleDegrees([1, 0, 0], [0, 0, 0], [0, 1, 0])).toBeCloseTo(90, 12);
    expect(angleDegrees([1, 0, 0], [0, 0, 0], [-1, 0, 0])).toBeCloseTo(180, 12);
    expect(angleDegrees([1, 0, 0], [0, 0, 0], [5, 0, 0])).toBeCloseTo(0, 12);
    expect(angleDegrees([0, 0, 0], [1, 0, 0], [0.5, Math.sqrt(3) / 2, 0])).toBeCloseTo(60, 12);
    // Away from the origin, in any plane.
    expect(angleDegrees([3, 2, 1 + 1], [3, 2, 1], [3, 2 + 1, 1 + 1])).toBeCloseTo(45, 12);
    expect(angleDegrees([0, 0, 0], [0, 0, 0], [1, 0, 0])).toBeNull();
    expect(formatAngle(42.46)).toBe("42.5°");
  });

  it("appends clicks, and a fourth starts a new angle", () => {
    let pts = nextAnglePoints([], [1, 0, 0]);
    pts = nextAnglePoints(pts, [0, 0, 0]);
    pts = nextAnglePoints(pts, [0, 1, 0]);
    expect(pts).toEqual([
      [1, 0, 0],
      [0, 0, 0],
      [0, 1, 0],
    ]);
    expect(nextAnglePoints(pts, [5, 5, 5])).toEqual([[5, 5, 5]]);
  });

  it("draws the arc between the arms at its radius, and the label on the bisector", () => {
    const a = [4, 0, 0];
    const v = [0, 0, 0];
    const c = [0, 2, 0];
    const r = arcRadius(a, v, c);
    expect(r).toBeCloseTo(2 / 3, 12);
    const arc = angleArc(a, v, c, r, 8);
    expect(arc.length).toBe(27);
    expect(Array.from(arc.subarray(0, 3))).toEqual([r, 0, 0].map((x) => expect.closeTo(x, 6)));
    expect(Array.from(arc.subarray(24, 27))).toEqual([0, r, 0].map((x) => expect.closeTo(x, 6)));
    for (let i = 0; i < arc.length; i += 3) {
      expect(Math.hypot(arc[i], arc[i + 1], arc[i + 2])).toBeCloseTo(r, 6);
    }
    const at = angleLabelAnchor(a, v, c, r);
    expect(at[0]).toBeCloseTo(at[1], 12);
    expect(at[2]).toBe(0);
    // No single arc between opposite arms; none for an arm without length.
    expect(angleArc([1, 0, 0], v, [-1, 0, 0], 1).length).toBe(0);
    expect(angleArc(v, v, c, 1).length).toBe(0);
  });
});

describe("bounding box geometry", () => {
  it("has twelve axis-aligned edges, four along each axis at the box's size", () => {
    const b = { min: [1, 2, 3] as [number, number, number], max: [4, 4, 7] as [number, number, number] };
    const segs = segments(boxEdges(b));
    expect(segs).toHaveLength(12);
    const size = boxSize(b);
    for (let axis = 0; axis < 3; axis++) {
      const along = segs.filter(([p, q]) => p[axis] !== q[axis]);
      expect(along).toHaveLength(4);
      for (const [p, q] of along) {
        expect(len(p, q)).toBeCloseTo(size[axis], 6);
      }
    }
    const [x, y, z] = boxLabelAnchors(b);
    expect(x).toEqual([2.5, 2, 3]);
    expect(y).toEqual([1, 3, 3]);
    expect(z).toEqual([1, 2, 5]);
  });
});

describe("normal lines", () => {
  it("draws one outward face normal per face, at the asked length", () => {
    const c = cube();
    const lines = faceNormalLines(Float32Array.from(c.positions), c.indices, 0.5);
    const segs = segments(lines);
    expect(segs).toHaveLength(12);
    for (const [from, to] of segs) {
      expect(len(from, to)).toBeCloseTo(0.5, 6);
      // Outward: the tip is farther from the centre than the face.
      expect(Math.hypot(...to)).toBeGreaterThan(Math.hypot(...from));
      // From a point on a face (one coordinate at ±1).
      expect(from.some((x) => Math.abs(Math.abs(x) - 1) < 1e-6)).toBe(true);
    }
  });

  it("caps the lines on a large mesh by sampling every k-th face or vertex", () => {
    const g = grid(150); // 45,000 triangles, 22,801 vertices
    const faces = segments(faceNormalLines(g.positions, g.indices, 1));
    expect(faces.length).toBeLessThanOrEqual(MAX_NORMAL_LINES);
    expect(faces.length).toBeGreaterThan(MAX_NORMAL_LINES / 3);
    for (const [from, to] of faces) {
      expect(to[2] - from[2]).toBeCloseTo(1, 6);
    }
    const used = usedVertices(g.indices, g.positions.length / 3);
    const verts = segments(vertexNormalLines(g.positions, g.indices, used, null, 2));
    expect(verts.length).toBeLessThanOrEqual(MAX_NORMAL_LINES);
    expect(verts.length).toBeGreaterThan(MAX_NORMAL_LINES / 3);
    for (const [from, to] of verts) {
      expect(to).toEqual([from[0], from[1], from[2] + 2].map((x) => expect.closeTo(x, 5)));
    }
    expect(segments(faceNormalLines(g.positions, g.indices, 1, 100)).length).toBeLessThanOrEqual(100);
  });

  it("uses the engine's vertex normals when the mesh has them, else area-weighted ones", () => {
    const scene = new Scene();
    scenes.push(scene);
    const c = cube();
    const m = scene.addMesh(c.positions, c.indices, { origin: [0, 0, 0], normals: true });
    scene.commitFrame();
    const views = scene.views(m);
    const positions = views.positions();
    const indices = views.indices();
    const normals = views.normals();
    expect(normals).not.toBeNull();
    const used = usedVertices(indices, positions.length / 3);
    const fromEngine = segments(vertexNormalLines(positions, indices, used, normals, 1));
    const own = segments(vertexNormalLines(positions, indices, used, null, 1));
    expect(fromEngine).toHaveLength(8);
    for (let i = 0; i < 8; i++) {
      const [from, to] = fromEngine[i];
      expect(len(from, to)).toBeCloseTo(1, 5);
      // A corner's normal points out of the cube, along its diagonal.
      const d = [to[0] - from[0], to[1] - from[1], to[2] - from[2]];
      expect(d[0] * from[0] + d[1] * from[1] + d[2] * from[2]).toBeGreaterThan(0);
      expect(own[i][0]).toEqual(from);
    }
    const acc = accumulateVertexNormals(positions, indices);
    expect(acc.length).toBe(positions.length);
  });

  it("walks only the vertices live triangles use", () => {
    // Slot 3 is unused; the removed triangle (0, 0, 0) uses nothing.
    const used = usedVertices(Uint32Array.from([0, 0, 0, 1, 2, 4]), 5);
    expect(Array.from(used)).toEqual([1, 2, 4]);
  });
});
