// The Text tool's typeface (#260): the bundled bold sans-serif font as
// typeface JSON (see `../fonts/LICENSE.txt`), loaded on first use as its own
// chunk, so the page pays for it only once a text tool opens; and each
// glyph's outline as closed polylines, its curves flattened.
//
// A glyph outline is a string of commands in font units: `m x y` starts a
// contour, `l x y` draws a line, `q x y cx cy` a quadratic curve to `(x, y)`
// through control `(cx, cy)`, and `b x y c1x c1y c2x c2y` a cubic one.
// Contours close back to their first point.

/** One glyph: its advance width, and its outline (none for a space). */
export interface TypefaceGlyph {
  /** The advance width, in font units. */
  ha: number;
  /** The outline commands; missing or empty for a blank glyph. */
  o?: string;
}

/** A typeface JSON font: its glyphs by character and its metrics, in font units. */
export interface TypefaceFont {
  glyphs: Readonly<Record<string, TypefaceGlyph>>;
  /** Font units per em. */
  resolution: number;
  ascender: number;
  descender: number;
  /** Baseline to baseline; `ascender - descender` when missing. */
  lineHeight?: number;
}

/** A closed polyline in font units, `[x0, y0, x1, y1, ...]`, without its first point repeated. */
export type Contour = number[];

let fontLoad: Promise<TypefaceFont> | null = null;

/**
 * The bundled font, imported on first call (a chunk of its own) and kept;
 * later calls share it. A failed load is forgotten, so the next call tries
 * again.
 */
export function loadTextFont(): Promise<TypefaceFont> {
  fontLoad ??= import("../fonts/sans-bold.typeface.json?raw").then(
    (m) => parseTypeface(m.default),
    (e: unknown) => {
      fontLoad = null;
      throw e;
    },
  );
  return fontLoad;
}

/** A typeface JSON font from its text, checked for glyphs and a positive resolution. */
export function parseTypeface(json: string): TypefaceFont {
  const data = JSON.parse(json) as Partial<TypefaceFont>;
  if (typeof data.glyphs !== "object" || data.glyphs === null || !(Number(data.resolution) > 0)) {
    throw new Error("not a typeface font: it needs glyphs and a positive resolution");
  }
  return data as TypefaceFont;
}

/**
 * The contours of `glyph`'s outline in font units, each curve cut into
 * `curveSegments` straight pieces (at least 1), as drawn: the caller cleans
 * them ({@link cleanContour}) once they are where they will be built.
 */
export function glyphContours(glyph: TypefaceGlyph, curveSegments: number): Contour[] {
  const n = Math.max(1, Math.floor(curveSegments));
  const out: Contour[] = [];
  const tokens = (glyph.o ?? "").trim().split(/\s+/);
  let cur: Contour | null = null;
  let x = 0;
  let y = 0;
  const num = (i: number): number => {
    const v = Number(tokens[i]);
    if (!Number.isFinite(v)) {
      throw new Error(`bad glyph outline number "${tokens[i]}"`);
    }
    return v;
  };
  for (let i = 0; i < tokens.length; ) {
    const cmd = tokens[i++];
    if (cmd === "") {
      continue;
    }
    if (cmd === "m") {
      x = num(i++);
      y = num(i++);
      cur = [x, y];
      out.push(cur);
      continue;
    }
    if (cur === null) {
      throw new Error(`glyph outline draws ("${cmd}") before it moves`);
    }
    if (cmd === "l") {
      x = num(i++);
      y = num(i++);
      cur.push(x, y);
    } else if (cmd === "q") {
      const ex = num(i++);
      const ey = num(i++);
      const cx = num(i++);
      const cy = num(i++);
      for (let k = 1; k <= n; k++) {
        const t = k / n;
        const u = 1 - t;
        cur.push(u * u * x + 2 * u * t * cx + t * t * ex, u * u * y + 2 * u * t * cy + t * t * ey);
      }
      x = ex;
      y = ey;
    } else if (cmd === "b") {
      const ex = num(i++);
      const ey = num(i++);
      const c1x = num(i++);
      const c1y = num(i++);
      const c2x = num(i++);
      const c2y = num(i++);
      for (let k = 1; k <= n; k++) {
        const t = k / n;
        const u = 1 - t;
        const a = u * u * u;
        const b = 3 * u * u * t;
        const c = 3 * u * t * t;
        const d = t * t * t;
        cur.push(a * x + b * c1x + c * c2x + d * ex, a * y + b * c1y + c * c2y + d * ey);
      }
      x = ex;
      y = ey;
    } else {
      throw new Error(`unknown glyph outline command "${cmd}"`);
    }
  }
  return out;
}

/** `c` without repeated points (the closing one too) or points exactly in line with their neighbours. */
export function cleanContour(c: Contour): Contour {
  // Repeated points, cyclically.
  let pts: number[] = [];
  for (let i = 0; i < c.length; i += 2) {
    const n = pts.length;
    if (n === 0 || pts[n - 2] !== c[i] || pts[n - 1] !== c[i + 1]) {
      pts.push(c[i], c[i + 1]);
    }
  }
  while (pts.length >= 4 && pts[0] === pts[pts.length - 2] && pts[1] === pts[pts.length - 1]) {
    pts.length -= 2;
  }
  // Points in line with their neighbours, until none is left: removing one
  // can put its neighbour in line.
  for (let changed = true; changed && pts.length >= 6; ) {
    changed = false;
    const kept: number[] = [];
    const m = pts.length / 2;
    for (let i = 0; i < m; i++) {
      const p = (i + m - 1) % m;
      const q = (i + 1) % m;
      const ax = pts[2 * p];
      const ay = pts[2 * p + 1];
      const bx = pts[2 * i];
      const by = pts[2 * i + 1];
      const cx = pts[2 * q];
      const cy = pts[2 * q + 1];
      if ((by - ay) * (cx - bx) - (bx - ax) * (cy - by) === 0) {
        changed = true;
        // Drop this one; its neighbours are judged against the points kept.
        for (let j = i + 1; j < m; j++) {
          kept.push(pts[2 * j], pts[2 * j + 1]);
        }
        break;
      }
      kept.push(bx, by);
    }
    pts = kept;
  }
  return pts;
}
