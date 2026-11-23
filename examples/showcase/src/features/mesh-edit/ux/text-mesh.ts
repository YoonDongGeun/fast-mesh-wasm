// Solid text (#260): a string laid out in the bundled font, each glyph's
// outline split into filled regions (an outer contour and the holes inside
// it: the counters of A, B, 8), each region's caps triangulated and its side
// walls built between them, as one closed, welded, outward-wound mesh. The
// text lies in the XY plane, reading along +X with +Y up, and is extruded
// along +Z.
//
// Lengths are millimetres: the letter height is a capital's (the height of
// H), the depth is the extrusion, and the letter spacing is added between
// neighbouring letters. Lines (split at newlines) are centred on each other,
// and the text's box on the origin.
//
// The caps are triangulated with three.js's ear clipping (`ShapeUtils`),
// which takes holes. Ear clipping may leave out a point that lies exactly
// on a straight run of the outline, which would leave a gap beside the
// wall; such points are first removed from the outline, and any left out
// later are put back by splitting the triangle across them. A glyph that
// still does not close is left out and reported, so the mesh always closes.
import { ShapeUtils, Vector2 } from "three";
import { cleanContour, type Contour, glyphContours, type TypefaceFont } from "./text-font";

/** What text to build and how big, in millimetres. */
export interface TextParams {
  /** The text; newlines start new lines. */
  text: string;
  /** The letter height: a capital's height. */
  size: number;
  /** The extrusion depth. */
  depth: number;
  /** Extra space between neighbouring letters (0 or more). */
  spacing: number;
  /** Straight pieces per outline curve, 1 to {@link MAX_CURVE_SEGMENTS}. */
  curveSegments: number;
}

/** The most characters the text tool builds. */
export const MAX_TEXT_CHARS = 200;
/** The most straight pieces per outline curve. */
export const MAX_CURVE_SEGMENTS = 32;

/** The default text: a word with holed letters, 5 mm tall and 1.5 mm deep. */
export const DEFAULT_TEXT_PARAMS: TextParams = {
  text: "Text",
  size: 5,
  depth: 1.5,
  spacing: 0,
  curveSegments: 6,
};

/**
 * Why `p` cannot be built, in words, or `null` when it can: some text to
 * draw, at most {@link MAX_TEXT_CHARS} characters, positive size and depth,
 * spacing of 0 or more, and a whole number of curve pieces in range.
 */
export function textParamsProblem(p: TextParams): string | null {
  if (p.text.trim() === "") {
    return "Type some text.";
  }
  if (Array.from(p.text).length > MAX_TEXT_CHARS) {
    return `At most ${MAX_TEXT_CHARS} characters.`;
  }
  if (!(p.size > 0) || !Number.isFinite(p.size)) {
    return "The letter height must be above zero.";
  }
  if (!(p.depth > 0) || !Number.isFinite(p.depth)) {
    return "The depth must be above zero.";
  }
  if (!(p.spacing >= 0) || !Number.isFinite(p.spacing)) {
    return "The letter spacing must be zero or more.";
  }
  if (!Number.isInteger(p.curveSegments) || p.curveSegments < 1 || p.curveSegments > MAX_CURVE_SEGMENTS) {
    return `Curve segments must be a whole number from 1 to ${MAX_CURVE_SEGMENTS}.`;
  }
  return null;
}

/** The height of a capital in font units: the top of H, or 70% of the ascender without one. */
export function capHeight(font: TypefaceFont): number {
  const h = font.glyphs.H;
  if (h !== undefined) {
    let top = -Infinity;
    for (const c of glyphContours(h, 1)) {
      for (let i = 1; i < c.length; i += 2) {
        top = Math.max(top, c[i]);
      }
    }
    if (top > 0) {
      return top;
    }
  }
  return 0.7 * font.ascender;
}

/** One glyph laid out: its character and its contours in millimetres (`f32` values, cleaned). */
export interface LaidOutGlyph {
  char: string;
  contours: Contour[];
}

/** A string laid out: the glyphs with outlines, the characters the font lacks, and the box. */
export interface TextLayout {
  glyphs: LaidOutGlyph[];
  /** Characters drawn as `?` because the font has no glyph for them, each once. */
  missing: string[];
  /** The box of every outline point: `[minX, minY, maxX, maxY]`, centred on the origin. */
  box: [number, number, number, number];
}

/**
 * Lays out `p.text` in `font`: each line's glyphs side by side (advance plus
 * spacing), lines one line height apart and centred on each other, then the
 * whole box centred on the origin. A character the font lacks is drawn as
 * `?` (and listed); a tab is a space.
 */
export function layoutText(font: TypefaceFont, p: Pick<TextParams, "text" | "size" | "spacing" | "curveSegments">): TextLayout {
  const scale = p.size / capHeight(font);
  const lineAdvance = (font.lineHeight ?? font.ascender - font.descender) * scale;
  const missing = new Set<string>();
  const raw: { char: string; contours: Contour[] }[] = [];
  const lines = p.text.replace(/\r\n?/g, "\n").split("\n");
  lines.forEach((line, row) => {
    const placed: { char: string; contours: Contour[] }[] = [];
    let pen = 0;
    const chars = Array.from(line.replace(/\t/g, " "));
    chars.forEach((ch, k) => {
      let glyph = font.glyphs[ch];
      if (glyph === undefined) {
        missing.add(ch);
        glyph = font.glyphs["?"] ?? { ha: font.resolution / 2 };
      }
      const y0 = -row * lineAdvance;
      const contours = glyphContours(glyph, p.curveSegments).map((c) => {
        const out: Contour = new Array<number>(c.length);
        for (let i = 0; i < c.length; i += 2) {
          out[i] = pen + c[i] * scale;
          out[i + 1] = y0 + c[i + 1] * scale;
        }
        return out;
      });
      if (contours.length > 0) {
        placed.push({ char: ch, contours });
      }
      pen += glyph.ha * scale + (k < chars.length - 1 ? p.spacing : 0);
    });
    // Centre the line on x = 0 by its advance.
    const shift = -pen / 2;
    for (const g of placed) {
      for (const c of g.contours) {
        for (let i = 0; i < c.length; i += 2) {
          c[i] += shift;
        }
      }
    }
    raw.push(...placed);
  });
  let minX = Infinity;
  let minY = Infinity;
  let maxX = -Infinity;
  let maxY = -Infinity;
  for (const g of raw) {
    for (const c of g.contours) {
      for (let i = 0; i < c.length; i += 2) {
        minX = Math.min(minX, c[i]);
        maxX = Math.max(maxX, c[i]);
        minY = Math.min(minY, c[i + 1]);
        maxY = Math.max(maxY, c[i + 1]);
      }
    }
  }
  if (!(maxX >= minX)) {
    return { glyphs: [], missing: [...missing], box: [0, 0, 0, 0] };
  }
  const cx = (minX + maxX) / 2;
  const cy = (minY + maxY) / 2;
  // Centred, then rounded to f32 (what the mesh stores) before cleaning, so
  // that the triangulation sees the very points the mesh is made of.
  const glyphs: LaidOutGlyph[] = [];
  for (const g of raw) {
    const contours: Contour[] = [];
    for (const c of g.contours) {
      const moved: Contour = new Array<number>(c.length);
      for (let i = 0; i < c.length; i += 2) {
        moved[i] = Math.fround(c[i] - cx);
        moved[i + 1] = Math.fround(c[i + 1] - cy);
      }
      const clean = cleanContour(moved);
      if (clean.length >= 6 && signedArea(clean) !== 0) {
        contours.push(clean);
      }
    }
    if (contours.length > 0) {
      glyphs.push({ char: g.char, contours });
    }
  }
  return { glyphs, missing: [...missing], box: [minX - cx, minY - cy, maxX - cx, maxY - cy] };
}

/** The signed area of a closed polyline (positive counter-clockwise). */
export function signedArea(c: Contour): number {
  let a = 0;
  const n = c.length;
  for (let i = 0; i < n; i += 2) {
    const j = (i + 2) % n;
    a += c[i] * c[j + 1] - c[j] * c[i + 1];
  }
  return a / 2;
}

/** Whether `(x, y)` is inside the closed polyline `c` (even-odd). */
function inside(c: Contour, x: number, y: number): boolean {
  let hit = false;
  const n = c.length;
  for (let i = 0, j = n - 2; i < n; j = i, i += 2) {
    const yi = c[i + 1];
    const yj = c[j + 1];
    if (yi > y !== yj > y && x < ((c[j] - c[i]) * (y - yi)) / (yj - yi) + c[i]) {
      hit = !hit;
    }
  }
  return hit;
}

/** Whether contour `a` lies inside contour `b`: most of three of its points do. */
function contourInside(a: Contour, b: Contour): boolean {
  const m = a.length / 2;
  let votes = 0;
  for (let k = 0; k < 3; k++) {
    const i = Math.floor((k * m) / 3);
    if (inside(b, a[2 * i], a[2 * i + 1])) {
      votes++;
    }
  }
  return votes >= 2;
}

/** A filled region: an outer contour (counter-clockwise) and its holes (clockwise). */
export interface TextRegion {
  outer: Contour;
  holes: Contour[];
}

/**
 * The filled regions of one glyph's contours, by nesting: a contour inside
 * an even number of others is an outer one, inside an odd number a hole of
 * the smallest outer one around it. Font winding is not relied on.
 */
export function glyphRegions(contours: readonly Contour[]): TextRegion[] {
  const n = contours.length;
  const area = contours.map((c) => Math.abs(signedArea(c)));
  const within: number[][] = contours.map(() => []);
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      if (i !== j && area[j] > area[i] && contourInside(contours[i], contours[j])) {
        within[i].push(j);
      }
    }
  }
  const depth = within.map((w) => w.length);
  const regions = new Map<number, TextRegion>();
  const oriented = (c: Contour, ccw: boolean): Contour => (signedArea(c) > 0 === ccw ? c : reverse(c));
  for (let i = 0; i < n; i++) {
    if (depth[i] % 2 === 0) {
      regions.set(i, { outer: oriented(contours[i], true), holes: [] });
    }
  }
  for (let i = 0; i < n; i++) {
    if (depth[i] % 2 === 1) {
      let parent = -1;
      for (const j of within[i]) {
        if (depth[j] === depth[i] - 1 && (parent < 0 || area[j] < area[parent])) {
          parent = j;
        }
      }
      regions.get(parent)?.holes.push(oriented(contours[i], false));
    }
  }
  return [...regions.values()];
}

function reverse(c: Contour): Contour {
  const out: Contour = new Array<number>(c.length);
  const m = c.length / 2;
  for (let i = 0; i < m; i++) {
    out[2 * i] = c[2 * (m - 1 - i)];
    out[2 * i + 1] = c[2 * (m - 1 - i) + 1];
  }
  return out;
}

/**
 * Counter-clockwise triangles filling `region`, numbering its points outer
 * first, then each hole: every outline edge is an edge of exactly one
 * triangle, in the outline's direction. `null` when that cannot be had.
 */
export function triangulateRegion(region: TextRegion): number[][] | null {
  const rings = [region.outer, ...region.holes];
  const pts: number[] = [];
  const ringEdges: [number, number][] = [];
  for (const ring of rings) {
    const start = pts.length / 2;
    const m = ring.length / 2;
    for (let i = 0; i < m; i++) {
      ringEdges.push([start + i, start + ((i + 1) % m)]);
    }
    pts.push(...ring);
  }
  const vec = (c: Contour) => {
    const out: Vector2[] = [];
    for (let i = 0; i < c.length; i += 2) {
      out.push(new Vector2(c[i], c[i + 1]));
    }
    return out;
  };
  const tris = ShapeUtils.triangulateShape(vec(region.outer), region.holes.map(vec));
  let total = 0;
  for (const [a, b, c] of tris) {
    total += cross(pts, a, b, c);
  }
  if (total < 0) {
    for (const t of tris) {
      [t[1], t[2]] = [t[2], t[1]];
    }
  }
  return repairOutline(pts, tris, ringEdges) ? tris : null;
}

/** Twice the signed area of triangle `a b c` over the flat points `p`. */
function cross(p: readonly number[], a: number, b: number, c: number): number {
  return (p[2 * b] - p[2 * a]) * (p[2 * c + 1] - p[2 * a + 1]) - (p[2 * b + 1] - p[2 * a + 1]) * (p[2 * c] - p[2 * a]);
}

/** Whether point `v` lies on segment `p q`, strictly between its ends. */
function onSegment(pts: readonly number[], p: number, q: number, v: number): boolean {
  const dx = pts[2 * q] - pts[2 * p];
  const dy = pts[2 * q + 1] - pts[2 * p + 1];
  const vx = pts[2 * v] - pts[2 * p];
  const vy = pts[2 * v + 1] - pts[2 * p + 1];
  const len2 = dx * dx + dy * dy;
  const t = vx * dx + vy * dy;
  return Math.abs(dx * vy - dy * vx) <= 1e-9 * len2 && t > 0 && t < len2;
}

/**
 * Puts back outline points the triangulation left out: while an outline
 * edge is missing, the triangle edge running straight over one of its ends
 * is split there. True when every outline edge is then present.
 */
export function repairOutline(pts: readonly number[], tris: number[][], ringEdges: readonly [number, number][]): boolean {
  const n = pts.length / 2;
  const key = (a: number, b: number) => a * n + b;
  for (let guard = 0; guard <= ringEdges.length; guard++) {
    const have = new Set<number>();
    for (const [a, b, c] of tris) {
      have.add(key(a, b));
      have.add(key(b, c));
      have.add(key(c, a));
    }
    const gap = ringEdges.find(([u, w]) => !have.has(key(u, w)));
    if (gap === undefined) {
      return true;
    }
    let split = false;
    for (let t = 0; t < tris.length && !split; t++) {
      for (let e = 0; e < 3 && !split; e++) {
        const p = tris[t][e];
        const q = tris[t][(e + 1) % 3];
        const r = tris[t][(e + 2) % 3];
        for (const v of gap) {
          if (v !== p && v !== q && onSegment(pts, p, q, v)) {
            tris[t] = [p, v, r];
            tris.push([v, q, r]);
            split = true;
            break;
          }
        }
      }
    }
    if (!split) {
      return false;
    }
  }
  return false;
}

/** A text mesh: welded, closed, outward-wound triangles, with what the layout found. */
export interface TextMesh {
  positions: Float32Array;
  indices: Uint32Array;
  /** Characters drawn as `?`. */
  missing: string[];
  /** Characters left out because their outline could not be closed (none in the bundled font's ASCII). */
  skipped: string[];
  /** The text's size: width, height and depth. */
  size: [number, number, number];
}

/**
 * The closed mesh of `layout` extruded by `depth`: each glyph's regions
 * capped above and below and walled between, z from `z0` to `z0 + depth`,
 * then welded (equal positions share a vertex; triangles a weld collapses
 * are dropped).
 */
export function extrudeLayout(layout: TextLayout, depth: number, z0 = 0): TextMesh {
  const positions: number[] = [];
  const indices: number[] = [];
  const skipped: string[] = [];
  const z1 = Math.fround(z0 + depth);
  const zb = Math.fround(z0);
  for (const glyph of layout.glyphs) {
    const startV = positions.length;
    const startT = indices.length;
    let ok = true;
    for (const region of glyphRegions(glyph.contours)) {
      const tris = triangulateRegion(region);
      if (tris === null) {
        ok = false;
        break;
      }
      const rings = [region.outer, ...region.holes];
      const base = positions.length / 3;
      let count = 0;
      for (const ring of rings) {
        count += ring.length / 2;
      }
      const bottom = (i: number) => base + i;
      const top = (i: number) => base + count + i;
      for (const z of [zb, z1]) {
        for (const ring of rings) {
          for (let i = 0; i < ring.length; i += 2) {
            positions.push(ring[i], ring[i + 1], z);
          }
        }
      }
      for (const [a, b, c] of tris) {
        indices.push(top(a), top(b), top(c), bottom(a), bottom(c), bottom(b));
      }
      let start = 0;
      for (const ring of rings) {
        const m = ring.length / 2;
        for (let i = 0; i < m; i++) {
          const u = start + i;
          const w = start + ((i + 1) % m);
          indices.push(bottom(u), bottom(w), top(w), bottom(u), top(w), top(u));
        }
        start += m;
      }
    }
    if (!ok || !isClosed(indices.slice(startT))) {
      positions.length = startV;
      indices.length = startT;
      if (!skipped.includes(glyph.char)) {
        skipped.push(glyph.char);
      }
    }
  }
  const welded = weld(positions, indices);
  const [minX, minY, maxX, maxY] = layout.box;
  return {
    ...welded,
    missing: layout.missing,
    skipped,
    size: [maxX - minX, maxY - minY, layout.glyphs.length > 0 ? depth : 0],
  };
}

/**
 * Whether `indices` close up: every edge is used once each way (so every
 * edge has two triangles, wound consistently).
 */
export function isClosed(indices: ArrayLike<number>): boolean {
  let top = 0;
  for (let i = 0; i < indices.length; i++) {
    top = Math.max(top, indices[i]);
  }
  const n = top + 1;
  const count = new Map<number, number>();
  for (let t = 0; t < indices.length; t += 3) {
    for (let e = 0; e < 3; e++) {
      const k = indices[t + e] * n + indices[t + ((e + 1) % 3)];
      count.set(k, (count.get(k) ?? 0) + 1);
    }
  }
  for (const [k, c] of count) {
    const a = Math.floor(k / n);
    const b = k - a * n;
    if (c !== 1 || count.get(b * n + a) !== 1) {
      return false;
    }
  }
  return true;
}

/** `positions` and `indices` with equal positions sharing one vertex, collapsed triangles dropped. */
export function weld(positions: readonly number[], indices: readonly number[]): { positions: Float32Array; indices: Uint32Array } {
  // `+ 0` turns -0 into 0, so that both weld.
  const f32 = Float32Array.from(positions, (v) => v + 0);
  const bits = new Uint32Array(f32.buffer);
  const vertexCount = f32.length / 3;
  // Buckets by a hash of the f32 bits, compared exactly.
  const buckets = new Map<number, number[]>();
  const remap = new Uint32Array(vertexCount);
  const out: number[] = [];
  const firstOf: number[] = [];
  for (let v = 0; v < vertexCount; v++) {
    const x = bits[3 * v];
    const y = bits[3 * v + 1];
    const z = bits[3 * v + 2];
    const h = (Math.imul(x, 73856093) ^ Math.imul(y, 19349663) ^ Math.imul(z, 83492791)) >>> 0;
    let bucket = buckets.get(h);
    if (bucket === undefined) {
      bucket = [];
      buckets.set(h, bucket);
    }
    let id = -1;
    for (const cand of bucket) {
      const w = firstOf[cand];
      if (bits[3 * w] === x && bits[3 * w + 1] === y && bits[3 * w + 2] === z) {
        id = cand;
        break;
      }
    }
    if (id < 0) {
      id = firstOf.length;
      firstOf.push(v);
      bucket.push(id);
      out.push(f32[3 * v], f32[3 * v + 1], f32[3 * v + 2]);
    }
    remap[v] = id;
  }
  const tris: number[] = [];
  for (let t = 0; t < indices.length; t += 3) {
    const a = remap[indices[t]];
    const b = remap[indices[t + 1]];
    const c = remap[indices[t + 2]];
    if (a !== b && b !== c && c !== a) {
      tris.push(a, b, c);
    }
  }
  return { positions: Float32Array.from(out), indices: Uint32Array.from(tris) };
}

/**
 * The closed mesh of `p` in `font`, its box centred on the origin in X and
 * Y; z from 0 to the depth, or centred too with `centreZ`.
 */
export function buildTextMesh(font: TypefaceFont, p: TextParams, options: { centreZ?: boolean } = {}): TextMesh {
  const layout = layoutText(font, p);
  return extrudeLayout(layout, p.depth, options.centreZ === true ? -p.depth / 2 : 0);
}

/**
 * The outline of `p` as line segments at z = 0 (`[x0, y0, 0, x1, y1, 0, ...]`
 * per segment), centred as the mesh, and the radius of its box: what the
 * viewport shows of a text stamp before it is placed.
 */
export function textOutlineSegments(font: TypefaceFont, p: TextParams): { segments: Float32Array; radius: number } {
  const layout = layoutText(font, p);
  const out: number[] = [];
  for (const g of layout.glyphs) {
    for (const c of g.contours) {
      for (let i = 0; i < c.length; i += 2) {
        const j = (i + 2) % c.length;
        out.push(c[i], c[i + 1], 0, c[j], c[j + 1], 0);
      }
    }
  }
  const [minX, minY, maxX, maxY] = layout.box;
  return { segments: Float32Array.from(out), radius: Math.hypot(maxX - minX, maxY - minY) / 2 };
}
