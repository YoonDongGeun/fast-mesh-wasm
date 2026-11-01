// The export reports (issue #127): the plane projection is orthonormal, SVG
// and DXF carry the contours (closed loops flagged), the collision CSV has
// its header and escapes names, and the panel's row/contour readers gather
// the live state (skipping meshes gone meanwhile).
import { describe, expect, it } from "vitest";
import {
  buildCollisionRows,
  readSectionContours,
} from "../src/panels/ExportPanel";
import {
  collisionRowsToCsv,
  contoursToDxf,
  contoursToSvg,
  planeBasis,
  projectContours,
  reportFileName,
} from "../src/load/report";

const ORIGIN: [number, number, number] = [0, 0, 0];
const Z: [number, number, number] = [0, 0, 1];

// A unit square in the z=0 plane, closed; plus an open segment.
const SQUARE = { points: Float32Array.from([0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0]), closed: true };
const SEGMENT = { points: Float32Array.from([2, 3, 0, 4, 5, 0]), closed: false };

describe("plane projection", () => {
  it("gives an orthonormal basis of the normal", () => {
    for (const n of [
      [0, 0, 1],
      [1, 0, 0],
      [1, 2, 3],
    ] as [number, number, number][]) {
      const { u, v } = planeBasis(n);
      const len = (a: readonly number[]) => Math.hypot(a[0], a[1], a[2]);
      expect(len(u)).toBeCloseTo(1, 12);
      expect(len(v)).toBeCloseTo(1, 12);
      expect(u[0] * v[0] + u[1] * v[1] + u[2] * v[2]).toBeCloseTo(0, 10);
      expect(u[0] * n[0] + u[1] * n[1] + u[2] * n[2]).toBeCloseTo(0, 10);
    }
  });

  it("projects world points onto the plane about the origin", () => {
    const [square] = projectContours([SQUARE], ORIGIN, Z);
    expect(square.closed).toBe(true);
    expect(square.points).toHaveLength(4);
    const xs = square.points.map(([x]) => x);
    const ys = square.points.map(([, y]) => y);
    expect(Math.min(...xs)).toBeCloseTo(0, 12);
    expect(Math.max(...xs)).toBeCloseTo(1, 12);
    expect(Math.min(...ys)).toBeCloseTo(0, 12);
    expect(Math.max(...ys)).toBeCloseTo(1, 12);
  });
});

describe("section files", () => {
  it("writes closed contours as polygons and open ones as polylines", () => {
    const svg = contoursToSvg([SQUARE, SEGMENT], ORIGIN, Z);
    expect(svg).toContain("<svg");
    expect(svg).toContain("<polygon");
    expect(svg).toContain("<polyline");
    expect(svg).toContain("</svg>");
  });

  it("writes DXF polylines with the closed flag", () => {
    const dxf = contoursToDxf([SQUARE, SEGMENT], ORIGIN, Z);
    expect(dxf).toContain("ENTITIES");
    expect(dxf).toContain("LWPOLYLINE");
    expect(dxf).toContain("EOF");
    const flags = dxf.split("\n").filter((l) => l === "0" || l === "1");
    // Both flag values appear: the square closed (1), the segment open (0).
    expect(flags).toContain("1");
    expect(flags).toContain("0");
  });

  it("stays valid with no contours", () => {
    expect(contoursToSvg([], ORIGIN, Z)).toContain("<svg");
    const dxf = contoursToDxf([], ORIGIN, Z);
    expect(dxf).toContain("ENTITIES");
    expect(dxf).not.toContain("LWPOLYLINE");
  });
});

describe("collision CSV", () => {
  it("writes the header and one row per pair", () => {
    const csv = collisionRowsToCsv([
      { a: "A", b: "B", gap: 0.5, intersects: false, depth: null },
      { a: "B", b: "C", gap: 0, intersects: true, depth: 1.25 },
    ]);
    expect(csv.split("\n")).toEqual([
      "mesh_a,mesh_b,min_gap,intersects,max_depth",
      "A,B,0.5,no,",
      "B,C,0,yes,1.25",
      "",
    ]);
  });

  it("escapes commas and quotes in mesh names", () => {
    const csv = collisionRowsToCsv([{ a: 'we,ird "mesh"', b: "plain", gap: null, intersects: false, depth: null }]);
    expect(csv).toContain('"we,ird ""mesh""",plain,,no,');
  });

  it("is header-only with no rows", () => {
    expect(collisionRowsToCsv([])).toBe("mesh_a,mesh_b,min_gap,intersects,max_depth\n");
  });

  it("sanitizes download names", () => {
    expect(reportFileName("my mesh", "csv")).toBe("my mesh.csv");
    expect(reportFileName("a/b:c", "dxf")).toBe("a_b_c.dxf");
    expect(reportFileName("  ", "svg")).toBe("export.svg");
  });
});

describe("panel readers", () => {
  const meshA = { id: 1, name: "A", mesh: "hA" };
  const meshB = { id: 2, name: "B", mesh: "hB" };
  const items = [meshA, meshB] as unknown as Parameters<typeof buildCollisionRows>[0];
  const hA = "hA" as never;
  const hB = "hB" as never;

  it("builds the pair row under the pair as picked now", () => {
    const rows = buildCollisionRows(
      items,
      { a: hA, b: hB },
      { readouts: { penetration: { depth: 0.5, atLeast: false }, bound: Infinity, gap: 0, intersects: true, contacts: { a: 3, b: 4 } } },
      { readouts: null },
    );
    expect(rows).toEqual([{ a: "A", b: "B", gap: 0, intersects: true, depth: 0.5 }]);
  });

  it("builds one row per group pair and none before readouts land", () => {
    expect(buildCollisionRows(items, { a: null, b: null }, { readouts: null }, { readouts: null })).toEqual([]);
    const rows = buildCollisionRows(
      items,
      { a: null, b: null },
      { readouts: null },
      {
        readouts: {
          bound: Infinity,
          pairs: [{ a: hA, b: hB, gap: 2, intersects: false, depth: null, atLeast: false }],
        },
      },
    );
    expect(rows).toEqual([{ a: "A", b: "B", gap: 2, intersects: false, depth: null }]);
  });

  it("reads every cut mesh's contours, skipping meshes gone meanwhile", () => {
    const handle = { section: true };
    const store = {
      getSnapshot: () => ({ sections: new Map([[handle, { desired: { meshes: ["m1", "m2"] } }]]) }),
    };
    const scene = {
      section: (_s: unknown, m: string) => {
        if (m === "m2") {
          throw new Error("UNKNOWN_MESH");
        }
        return { contours: [{ points: Float32Array.from([0, 0, 0, 1, 0, 0]), closed: false }] };
      },
    };
    const read = readSectionContours(
      store as never,
      scene as never,
      handle,
      { origin: [0, 0, 0], normal: [0, 0, 1] },
    );
    expect(read?.contours).toHaveLength(1);
    expect(read?.contours[0].closed).toBe(false);
  });
});
