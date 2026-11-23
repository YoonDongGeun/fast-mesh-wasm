// Solid text (#260): the bundled font, glyph outlines and their nesting,
// and the text mesh: closed and consistently wound (checked here and by the
// engine's own analysis), welded, outward, with one tunnel per counter (the
// holes of A, B and 8), sized and spaced as asked.
import { Scene } from "fast-mesh-wasm";
import { afterEach, beforeAll, describe, expect, it } from "vitest";
import { glyphContours, loadTextFont, type TypefaceFont } from "../src/features/mesh-edit/ux/text-font";
import {
  buildTextMesh,
  capHeight,
  DEFAULT_TEXT_PARAMS,
  glyphRegions,
  layoutText,
  MAX_TEXT_CHARS,
  refineAcross,
  repairOutline,
  signedArea,
  type TextMesh,
  type TextParams,
  textOutlineSegments,
  textParamsProblem,
} from "../src/features/mesh-edit/ux/text-mesh";

let font: TypefaceFont;
beforeAll(async () => {
  font = await loadTextFont();
});

const scenes: Scene[] = [];
afterEach(() => {
  for (const s of scenes.splice(0)) s.free();
});

const params = (patch: Partial<TextParams> = {}): TextParams => ({ ...DEFAULT_TEXT_PARAMS, ...patch });

/** Edge use counts: every directed edge once, and its reverse once. */
function edgeReport(indices: Uint32Array) {
  const directed = new Map<string, number>();
  for (let t = 0; t < indices.length; t += 3) {
    for (let e = 0; e < 3; e++) {
      const k = `${indices[t + e]}>${indices[t + ((e + 1) % 3)]}`;
      directed.set(k, (directed.get(k) ?? 0) + 1);
    }
  }
  let unpaired = 0;
  let repeated = 0;
  for (const [k, c] of directed) {
    if (c > 1) repeated++;
    const [a, b] = k.split(">");
    if (!directed.has(`${b}>${a}`)) unpaired++;
  }
  return { edges: directed.size / 2, unpaired, repeated };
}

/** The signed volume (positive when wound outward). */
function signedVolume(m: Pick<TextMesh, "positions" | "indices">): number {
  const p = m.positions;
  let v = 0;
  for (let t = 0; t < m.indices.length; t += 3) {
    const a = 3 * m.indices[t];
    const b = 3 * m.indices[t + 1];
    const c = 3 * m.indices[t + 2];
    v +=
      p[a] * (p[b + 1] * p[c + 2] - p[b + 2] * p[c + 1]) -
      p[a + 1] * (p[b] * p[c + 2] - p[b + 2] * p[c]) +
      p[a + 2] * (p[b] * p[c + 1] - p[b + 1] * p[c]);
  }
  return v / 6;
}

/** Connected pieces of triangles sharing vertices. */
function components(m: Pick<TextMesh, "indices" | "positions">): number {
  const parent = Array.from({ length: m.positions.length / 3 }, (_, i) => i);
  const find = (x: number): number => (parent[x] === x ? x : (parent[x] = find(parent[x])));
  for (let t = 0; t < m.indices.length; t += 3) {
    parent[find(m.indices[t])] = find(m.indices[t + 1]);
    parent[find(m.indices[t + 1])] = find(m.indices[t + 2]);
  }
  const roots = new Set<number>();
  for (let t = 0; t < m.indices.length; t++) roots.add(find(m.indices[t]));
  return roots.size;
}

/** Closed, consistently wound, outward and welded; returns V - E + F. */
function expectSolid(m: TextMesh, label: string): number {
  const { edges, unpaired, repeated } = edgeReport(m.indices);
  expect(unpaired, `${label}: edges without a reverse`).toBe(0);
  expect(repeated, `${label}: edges used twice the same way`).toBe(0);
  expect(signedVolume(m), `${label}: outward volume`).toBeGreaterThan(0);
  const keys = new Set<string>();
  for (let v = 0; v < m.positions.length; v += 3) {
    keys.add(`${m.positions[v]},${m.positions[v + 1]},${m.positions[v + 2]}`);
  }
  expect(keys.size, `${label}: welded`).toBe(m.positions.length / 3);
  return m.positions.length / 3 - edges + m.indices.length / 3;
}

/** The engine's own reading of the mesh: closed, manifold, consistently wound, not self-intersecting. */
function expectEngineSolid(m: TextMesh, label: string): void {
  const scene = new Scene();
  scenes.push(scene);
  const mesh = scene.addMesh(Float64Array.from(m.positions), m.indices);
  const measure = scene.inspect.measure(mesh);
  expect(measure.closed, `${label}: closed`).toBe(true);
  expect(measure.orientationConflicts, `${label}: winding`).toBe(0);
  expect(measure.volume, `${label}: volume`).toBeGreaterThan(0);
  const a = scene.analyze(mesh, { selfIntersection: true });
  expect(a.nonManifoldEdges, `${label}: non-manifold edges`).toBe(0);
  expect(a.coincidentGroups, `${label}: coincident vertices`).toBe(0);
  expect(a.selfIntersects, `${label}: self-intersection`).toBe(false);
}

describe("the bundled font", () => {
  it("loads once, lazily, and covers printable ASCII", async () => {
    expect(loadTextFont()).toBe(loadTextFont());
    expect(font.resolution).toBeGreaterThan(0);
    for (let c = 32; c < 127; c++) {
      expect(font.glyphs[String.fromCharCode(c)], String.fromCharCode(c)).toBeDefined();
    }
    expect(capHeight(font)).toBeGreaterThan(0.5 * font.resolution);
  });

  it("flattens outline curves into the asked number of pieces", () => {
    const glyph = { ha: 10, o: "m 0 0 l 10 0 q 10 10 20 5 l 0 10" };
    const [c4] = glyphContours(glyph, 4);
    expect(c4.length / 2).toBe(2 + 4 + 1);
    // The curve ends where it is told to, through its control point's pull.
    expect([c4[10], c4[11]]).toEqual([10, 10]);
    expect(c4[6]).toBeGreaterThan(10);
    expect(glyphContours(glyph, 1)[0].length / 2).toBe(4);
    expect(() => glyphContours({ ha: 1, o: "l 1 1" }, 2)).toThrow();
    expect(glyphContours({ ha: 1 }, 2)).toEqual([]);
  });

  it("finds holes by nesting, whatever the winding", () => {
    const square = (s: number, ccw: boolean) => {
      const c = [-s, -s, s, -s, s, s, -s, s];
      return ccw ? c : [-s, s, s, s, s, -s, -s, -s];
    };
    // An outer square, a hole in it, and an island in the hole: two regions.
    for (const ccw of [true, false]) {
      const regions = glyphRegions([square(1, !ccw), square(3, ccw), square(2, ccw)]);
      expect(regions.length).toBe(2);
      const big = regions.find((r) => Math.abs(signedArea(r.outer)) === 36);
      expect(big?.holes.length).toBe(1);
      expect(signedArea(big!.outer)).toBeGreaterThan(0);
      expect(signedArea(big!.holes[0])).toBeLessThan(0);
      expect(regions.find((r) => r !== big)?.holes.length).toBe(0);
    }
  });
});

describe("cap triangulation", () => {
  it("puts back an outline point the triangulation left out, splitting the triangle over it", () => {
    // A square with a point halfway along its bottom, filled without it.
    const pts = [0, 0, 1, 0, 2, 0, 2, 2, 0, 2];
    const ring: [number, number][] = [
      [0, 1],
      [1, 2],
      [2, 3],
      [3, 4],
      [4, 0],
    ];
    const tris = [
      [0, 2, 3],
      [0, 3, 4],
    ];
    expect(repairOutline(pts, tris, ring)).toBe(true);
    expect(tris).toEqual([
      [0, 1, 3],
      [0, 3, 4],
      [1, 2, 3],
    ]);
    // A gap no triangle runs over cannot be mended.
    expect(repairOutline(pts, [[0, 2, 3]], ring)).toBe(false);
  });
});

describe("refining a stamp to follow a curve", () => {
  it("cuts every edge down across, not up the walls, and keeps the solid", () => {
    const p = params({ text: "AB8\nR2", size: 14, depth: 5 });
    const mesh = buildTextMesh(font, p);
    const fine = refineAcross(mesh, 14 / 8);
    expect(expectSolid(fine as TextMesh, "refined")).toBe(expectSolid(mesh, "plain"));
    expectEngineSolid(fine as TextMesh, "refined");
    expect(signedVolume(fine)).toBeCloseTo(signedVolume(mesh), 2);
    let longest = 0;
    for (let t = 0; t < fine.indices.length; t += 3) {
      for (let e = 0; e < 3; e++) {
        const a = 3 * fine.indices[t + e];
        const b = 3 * fine.indices[t + ((e + 1) % 3)];
        longest = Math.max(longest, Math.hypot(fine.positions[a] - fine.positions[b], fine.positions[a + 1] - fine.positions[b + 1]));
      }
    }
    expect(longest).toBeLessThanOrEqual(14 / 8);
    // The walls' upright edges stay whole: no vertex halfway up over an outline point.
    const corners = new Set<string>();
    for (let v = 0; v < mesh.positions.length; v += 3) corners.add(`${mesh.positions[v]},${mesh.positions[v + 1]}`);
    let halfway = 0;
    for (let v = 0; v < fine.positions.length; v += 3) {
      const z = fine.positions[v + 2];
      if (z > 0 && z < 5 && corners.has(`${fine.positions[v]},${fine.positions[v + 1]}`)) halfway++;
    }
    expect(halfway).toBe(0);
    expect(fine.indices.length).toBeGreaterThan(mesh.indices.length);
    // The vertices it had stay where they were.
    expect(Array.from(fine.positions.slice(0, mesh.positions.length))).toEqual(Array.from(mesh.positions));
    // The same every time, and nothing to do when every edge is short already.
    expect(refineAcross(mesh, 14 / 8)).toEqual(fine);
    expect(refineAcross(mesh, 1000)).toEqual({ positions: mesh.positions, indices: mesh.indices });
    expect(() => refineAcross(mesh, 0)).toThrow();
  });
});

describe("text meshes", () => {
  it("are closed and consistently oriented, with a tunnel per counter (A, B, 8)", () => {
    // V - E + F = 2 (shells) - 2 (tunnels): A has one counter, B and 8 two.
    const expected: Record<string, number> = { A: 0, B: -2, "8": -2, H: 2, AB8: 2 * 3 - 2 * 5 };
    for (const [text, euler] of Object.entries(expected)) {
      const m = buildTextMesh(font, params({ text }));
      expect(m.skipped, text).toEqual([]);
      expect(expectSolid(m, text), `${text}: V - E + F`).toBe(euler);
      expect(components(m), `${text}: pieces`).toBe(Array.from(text).length);
      expectEngineSolid(m, text);
    }
  });

  it("build every printable ASCII character closed, alone and in a sample string", () => {
    for (let c = 33; c < 127; c++) {
      const ch = String.fromCharCode(c);
      const m = buildTextMesh(font, params({ text: ch, curveSegments: 3 }));
      expect(m.skipped, ch).toEqual([]);
      expect(m.missing, ch).toEqual([]);
      expectSolid(m, ch);
    }
    const sample = buildTextMesh(font, params({ text: "Sample 8 BAD@0%&\nquick (fox) #42?", spacing: 0.2, curveSegments: 8 }));
    expect(sample.skipped).toEqual([]);
    expectSolid(sample, "sample");
    expectEngineSolid(sample, "sample");
  });

  it("encloses the outline's area times the depth", () => {
    const p = params({ text: "B8", depth: 2 });
    const layout = layoutText(font, p);
    let area = 0;
    for (const g of layout.glyphs) {
      for (const r of glyphRegions(g.contours)) {
        area += signedArea(r.outer) + r.holes.reduce((s, h) => s + signedArea(h), 0);
      }
    }
    expect(signedVolume(buildTextMesh(font, p))).toBeCloseTo(area * 2, 3);
  });

  it("are as tall as the letter height and as deep as the depth, centred", () => {
    const m = buildTextMesh(font, params({ text: "H", size: 12, depth: 3 }));
    const ys = Array.from({ length: m.positions.length / 3 }, (_, i) => m.positions[3 * i + 1]);
    const zs = Array.from({ length: m.positions.length / 3 }, (_, i) => m.positions[3 * i + 2]);
    expect(Math.max(...ys) - Math.min(...ys)).toBeCloseTo(12, 4);
    expect(Math.max(...ys) + Math.min(...ys)).toBeCloseTo(0, 4);
    expect([Math.min(...zs), Math.max(...zs)]).toEqual([0, 3]);
    expect(m.size[1]).toBeCloseTo(12, 4);
    expect(m.size[2]).toBe(3);
    const centred = buildTextMesh(font, params({ text: "H", depth: 3 }), { centreZ: true });
    const cz = Array.from({ length: centred.positions.length / 3 }, (_, i) => centred.positions[3 * i + 2]);
    expect([Math.min(...cz), Math.max(...cz)]).toEqual([-1.5, 1.5]);
  });

  it("spread by the letter spacing, between letters only", () => {
    const tight = buildTextMesh(font, params({ text: "IIII" }));
    const loose = buildTextMesh(font, params({ text: "IIII", spacing: 2 }));
    expect(loose.size[0] - tight.size[0]).toBeCloseTo(3 * 2, 4);
    const single = buildTextMesh(font, params({ text: "I" })).size[0];
    expect(buildTextMesh(font, params({ text: "I", spacing: 5 })).size[0]).toBeCloseTo(single, 6);
  });

  it("stack lines, draw missing characters as ? and skip blanks", () => {
    const one = buildTextMesh(font, params({ text: "AB" }));
    const two = buildTextMesh(font, params({ text: "AB\r\nAB" }));
    expect(two.size[1]).toBeGreaterThan(2 * one.size[1]);
    expect(two.indices.length).toBe(2 * one.indices.length);
    const odd = buildTextMesh(font, params({ text: "aé" }));
    expect(odd.missing).toEqual(["é"]);
    expect(odd.indices.length).toBe(buildTextMesh(font, params({ text: "a?" })).indices.length);
    const blank = buildTextMesh(font, params({ text: "   " }));
    expect([blank.positions.length, blank.indices.length]).toEqual([0, 0]);
  });

  it("checks their parameters", () => {
    expect(textParamsProblem(DEFAULT_TEXT_PARAMS)).toBeNull();
    expect(textParamsProblem(params({ text: " \n " }))).toMatch(/Type some text/);
    expect(textParamsProblem(params({ text: "x".repeat(MAX_TEXT_CHARS + 1) }))).toMatch(/At most/);
    expect(textParamsProblem(params({ size: 0 }))).toMatch(/height/);
    expect(textParamsProblem(params({ depth: Number.NaN }))).toMatch(/depth/);
    expect(textParamsProblem(params({ spacing: -1 }))).toMatch(/spacing/);
    expect(textParamsProblem(params({ curveSegments: 0 }))).toMatch(/Curve/);
    expect(textParamsProblem(params({ curveSegments: 2.5 }))).toMatch(/Curve/);
  });

  it("outline as line segments around the mesh's footprint", () => {
    const p = params({ text: "O" });
    const { segments, radius } = textOutlineSegments(font, p);
    const m = buildTextMesh(font, p);
    // One segment per wall: two triangles each (the ones not level in z).
    let wallTriangles = 0;
    for (let t = 0; t < m.indices.length; t += 3) {
      const z = [0, 1, 2].map((k) => m.positions[3 * m.indices[t + k] + 2]);
      if (z[0] !== z[1] || z[1] !== z[2]) wallTriangles++;
    }
    expect(segments.length / 6).toBeGreaterThan(8);
    expect(wallTriangles).toBe(2 * (segments.length / 6));
    expect(segments.every((v, i) => i % 3 !== 2 || v === 0)).toBe(true);
    expect(radius).toBeCloseTo(Math.hypot(m.size[0], m.size[1]) / 2, 4);
  });

  it("build long strings quickly", () => {
    const text = "The quick brown fox jumps over the lazy dog 0123456789 ".repeat(3).slice(0, MAX_TEXT_CHARS);
    const t0 = performance.now();
    const m = buildTextMesh(font, params({ text, curveSegments: 12 }));
    const ms = performance.now() - t0;
    expect(m.skipped).toEqual([]);
    expectSolid(m, "long");
    expect(ms).toBeLessThan(2000);
  });
});
