// Fragment shading of a colour rule, for `bindMesh({ shading: "fragment" })`:
// the lookup texture's data, the vertex value encoding, the GLSL, and a
// JavaScript reference of the shader's mapping that the tests check against
// the engine's `color_of`. Pure data and arithmetic: no three.js, no
// WebAssembly.
//
// The shader colours each fragment from its interpolated value the way the
// engine colours a vertex (crates/fast-mesh-fields/src/color.rs): `t` from
// the range (piecewise linear with a centre), the 256-entry table of a
// built-in map (or the custom stops) interpolated linearly in sRGB bytes and
// rounded, then sRGB to linear through the same 16-bit table as the vertex
// colours. So at a vertex the fragment colour is the vertex colour, while
// across a triangle the colour follows the value per pixel, and a range
// edge is an exact iso-line of the interpolated value.
//
// Encoding (one `vec2` attribute per vertex: value, NaN flag):
// - a finite value is passed as is, so the iso-lines between finite
//   vertices (the range ends, the contact band's edges) are exact;
// - `+Infinity` (a distance field's "far") becomes `hi + FAR_SPAN · (hi - lo)`
//   and `-Infinity` becomes `lo - FAR_SPAN · (hi - lo)`: finite values
//   beyond the range, which colour like the infinity at the vertex (`above`
//   / `below`) and interpolate towards it, so a far vertex tints nothing
//   inside the range beyond the iso-line `v = hi`;
// - NaN sets the flag (and takes the far value): fragments where the
//   interpolated flag exceeds 1/2 show the rule's NaN colour, so a NaN vertex
//   colours at most half of its triangles.
//
// Softness: with `edgeSoftness` s > 0 a fragment shows the mean colour of
// the values in [v - s/2, v + s/2] (a box filter in distance units, on sRGB
// bytes before linearisation). Linear stretches of the colormap stay as
// they are; a step (a range edge to the base colour, the contact band's
// edges) becomes a ramp s wide. At s = 0 an edge is a hard step.
import type { ColorRule, Rgba } from "./scene.js";

/** The rule's `base` colour when it gives none. */
export const DEFAULT_BASE: Rgba = [200, 200, 200, 255];
/** The rule's `nan` colour when it gives none. */
export const DEFAULT_NAN: Rgba = [128, 128, 128, 255];

/**
 * sRGB byte to linear, as a normalized `Uint16` (0 to 65535). The engine's
 * colour maps are sRGB-encoded bytes; three treats vertex colours as linear
 * working-space values and encodes its output to sRGB, so bytes passed
 * through would be encoded twice and look washed out. Sixteen bits keep the
 * dark tones that eight linear bits would band. The fragment shading
 * linearises through the same table.
 */
export const SRGB_TO_LINEAR16: Uint16Array = (() => {
  const table = new Uint16Array(256);
  for (let i = 0; i < 256; i++) {
    const c = i / 255;
    const linear = c <= 0.04045 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4;
    table[i] = Math.round(linear * 65535);
  }
  return table;
})();

/** Texels per row of the lookup texture; a custom map with more stops widens it. */
export const LUT_WIDTH = 256;
/**
 * Rows of the lookup texture: 0 the map's colours (sRGB bytes as floats: a
 * built-in map's table, or the custom stops), 1 the custom stops' positions
 * (`.r`), 2 a built-in table's running integral, 3 the sRGB-to-linear table
 * (`.r`, {@link SRGB_TO_LINEAR16} / 65535).
 */
export const LUT_ROWS = 4;
/**
 * A softness window over more built-in table entries than this integrates
 * through the running integral (row 2); a narrower one sums its entries.
 */
export const PREFIX_MIN_ENTRIES = 16;
/** `+Infinity` encodes as `hi + FAR_SPAN · (hi - lo)`, `-Infinity` as `lo - FAR_SPAN · (hi - lo)`. */
export const FAR_SPAN = 1;

/**
 * The engine's built-in tables (RGB; alpha is 255), base64: Turbo, Viridis
 * and Coolwarm as `color_of` samples them (the tests check them against the
 * engine). Grayscale is `(i, i, i)`.
 */
const TABLES: Record<"turbo" | "viridis" | "coolwarm", string> = {
  turbo:
    "IxcbJxooKxw0Lx4/MyBKNiNVOSVfOyhpPipyQCx7Qi+ERDGMRTSURzebSDmiSTypST6vSkG1SkS7S0bBS0nGS0zLS0/PSlHT" +
    "SlTXSlfbSVnfSVziSF/lR2LoRmXqRWftRGrvQ23wQnDyQXL0QHX1P3j2Pnv3PX34O4D4OoP5OYb5OIj5N4v5NY75NJD4M5P4" +
    "Mpb3MZj2MJv2L571LqD0LaPyLKXxK6jwKqruKq3tKa/rKLLqKLToJ7bmJ7nkJrviJr3gJcDeJcLcJcTaJcbXJcjVJcrTJc3R" +
    "Jc/OJtHMJtLJJtTHJ9bEJ9jCKNq/Kdy9Kt26K9+4LOG1LeKyLuSwL+WtMOerMeioM+qmNOujNuygN+6eOe+bO/CZPfGWP/KU" +
    "QfORQ/SPRfWMR/aKSfeHS/iFTvmDUPmAUvp+Vfp8V/t5Wvt3Xfx1X/xzYv1xZf1uaP1sav1qbf5ocP5mc/5kdv5ief5gfP1e" +
    "f/1dgv1bhf1ZiPxXi/xWjvxUkftSlftRmPpPm/lOnvlMofhLpPdJp/ZIqvZGrfVFsPREs/NCtvJBufBAvO8/v+4+wu08xes7" +
    "yOo6y+k5zec40OY30+Q21eM12OE029803d4z39wy4tox5Ngw5tYw6dQv69Iu7dAt784t8cws88or9Mgr9sYq+MQq+cEp+78o" +
    "/L0o/bon/7gn/7Um/7Mm/7El/64l/6wk/6kk/6Yj/6Qj/6Ei/58i/5wi/5kh/5ch/5Qg/5Eg/44f/4wf/4ke/4Ye/4Me/4Ed" +
    "/34d/3sc/3gc/3Ub/3Mb/3Aa/m0a/Goa+2gZ+WUZ+GIY9l8Y9FwX81oX8VcW71QW7VIV608U6UwU5koT5EcT4kUS4EIS3UAR" +
    "2z0Q2DsQ1jgP0zYP0TQOzjENyy8NyS0MxisLxCkLwScKviUKvCMJuSEItx8ItB0HsRwGrxoGrBgFqhcEqBYEpRQDoxMCoRIC" +
    "nxEBnRAAmw8Amg4AmA4Alg0AlQwAlAwAkwwAkgwAkQsAkQwAkAwAkAwAkAwAkA0A",
  viridis:
    "RwFVRwNXRwRYRwZZRwdbRwhcRwpdRwtfSA1gSA5hSA9jSBFkSBJlSBRnSBVoSBZpSBhqSBlsSBptSBxuSB1vSB9wSCBxSCFy" +
    "SCN0SCR1SCV2SCd3Ryh4Ryl5Ryp5Ryx6Ry17Ry58RzB9RjF+RjJ/RjN/RjWARjaBRTeBRTiCRTqDRTuDRDyERD2FRD6FRD+G" +
    "Q0GGQ0KHQ0OHQkSIQkWIQUaIQUiJQUmJQEqKQEuKP0yKP02LP06LPk+LPlCLPVGMPVKMPFSMPFWMO1aMO1eNOliNOlmNOVqN" +
    "OVuNOFyNOF2NN16ONl+ONmCONWGONWKONGOONGSOM2WOMmaOMmeOMWiOMWmOMGqOMGuOL2yOLm2OLm6OLW+OLXCOLHGOLHKO" +
    "K3OOK3SOKnSOKXWOKXaOKHeOKHiOJ3mOJ3qOJnuOJnyNJX2NJX6NJX+NJICNJIGNI4KNI4ONIoSNIoWNIoaNIYaNIYeMIYiM" +
    "IYmMIIqMIIuMIIyMII2MH46MH4+LH5CLH5GLH5KLH5OLH5SLH5SKH5WKH5aKH5eKH5iJH5mJH5qJH5uJIJyIIJ2IIJ6IIJ+I" +
    "IaCHIaGHIaKHIqKGIqOGI6SFI6WFJKaFJaeEJaiEJqmDJ6qDJ6uCKKyCKayBKq2AK66AK69/LLB/LbF+LrJ9MLN9MbR8MrR7" +
    "M7V6NLZ6Nbd5N7h4OLl3Orp2O7p1PLt0PrxzP71yQb5xQ79wRL9vRsBuSMFtSsJsS8NrTcNpT8RoUcVnU8ZmVcZkV8djWchi" +
    "W8lgXslfYMpeYstcZMxbZ8xZac1Ya85Wbs5VcM9Tc9BSddBQeNFOetJNfdJLf9NKgtNIhNRGh9VFitVDjdZBj9ZAktc+ldc9" +
    "mNg7mtk5ndk4oNo2o9o0ptszqNsxq9wwrtwusdwttN0rt90qut4pvN4nv98mwt8lxd8kyOAjyuAhzeEg0OEg0uEf1eIe2OId" +
    "2uId3eMc4OMc4uMb5OQb5+Qb6eQb7OUb7uUb8OUc8uYc9OYd9uYe+Ocf+ucg/Och",
  coolwarm:
    "O0zAPE7CPVDDPlHFQFPGQVXIQlbJQ1jLRFrMRVzOR13PSF/RSWHSSmPTS2TVTWbWTmjXT2nZUGvaUm3bU27dVHDeVXLfV3Pg" +
    "WHXhWXfiWnjkXHrlXXzmXn3nX3/oYYDpYoLqY4TrZYXsZoftZ4juaYrvaovva43wbY7xbpDyb5HzcZPzcpT0c5b1dZf2dpn2" +
    "d5r3eZz4ep34e5/5faD5fqH6gKP6gaT7gqb7hKf8haj8h6r8iKv9iaz9i639jK/+jbD+j7H+kLL+krT+k7X/lLb/lrf/l7j/" +
    "mLn/mrv/m7z/nb3/nr7/n7//ocD/osH/o8L+pcP+psT+qMX+qcb+qsf9rMj9rcn9rsn8sMr8scv8ssz7s837tc76ts76t8/5" +
    "udD5utH4u9H4vNL3vtP2v9P2wNT1wdT0w9X0xNbzxdbyxtfxyNfxydjwytjvy9juzNntzdnsztrr0Nrq0drp0tvo09vn1Nvm" +
    "1dvl1tzk19zj2Nzi2dzh2tzg293e3N3d3dzc3tzb39zZ4dvY4trW49rV5NnT5dnS5djR5tjP59fO6NbM6dbL6tXJ69TI69PG" +
    "7NPF7dLD7tHC7tDA78+/78698M688c268cy48su38sq188m088iy9Mex9Mav9MWu9cSs9cOq9cGp9sCn9r+m9r6k9r2i97yh" +
    "97qf97me97ic97eb97WZ97SX97OW97KU97CT96+R962Q96yO96uM96mL96iJ96aI9qWG9qOF9qKD9qCB9Z+A9Z1+9Zx99Jp7" +
    "9Jl69Jd485V385R18pJ08pBy8Y9x8Y1v8Itu8Ips74hr74Zp7oVo7YNm7YFl7H9j631i6nxg6npf6Xhd6HZc53Ra5nJZ5XBY" +
    "5W9W5G1V42tT4mlS4WdR4GVP32NO3mFN3V9L3F1K21tJ2llH2FZG11RF1lJD1VBC1E5B00xA0Uk+0Ec9z0U8zkM7zEA5yz44" +
    "yjs3yDk2xzY0xjQzxDEywy4xwSswwCgvvyUuvSIsvB4ruhoquRYptxEotgsntAQm",
};

const decoded = new Map<string, Uint8Array>();

/** A built-in map's 256 RGB entries. */
function table(map: "turbo" | "viridis" | "coolwarm" | "grayscale"): Uint8Array {
  let t = decoded.get(map);
  if (t === undefined) {
    if (map === "grayscale") {
      t = new Uint8Array(768);
      for (let i = 0; i < 256; i++) {
        t.fill(i, 3 * i, 3 * i + 3);
      }
    } else {
      const text = atob(TABLES[map]);
      t = new Uint8Array(text.length);
      for (let i = 0; i < text.length; i++) {
        t[i] = text.charCodeAt(i);
      }
    }
    decoded.set(map, t);
  }
  return t;
}

/** Everything the shader needs from a rule: the texture's data and the uniforms. */
export interface Shading {
  /** `width × LUT_ROWS` RGBA floats (see {@link LUT_ROWS}). */
  lut: Float32Array;
  width: number;
  /** Entries in row 0: 256 for a built-in map, the stop count for a custom one. */
  count: number;
  /** A built-in map: entries evenly spaced at `i / 255`. */
  builtin: boolean;
  /** `[lo, hi]` as `f32`. */
  range: readonly [number, number];
  /** The centre as `f32`, or NaN for none. */
  center: number;
  belowBase: boolean;
  aboveBase: boolean;
  base: Rgba;
  nan: Rgba;
}

const f = Math.fround;

/** The shading data of a valid colour rule. */
export function shadingOf(rule: ColorRule): Shading {
  const map = rule.map;
  const builtin = typeof map === "string";
  const count = builtin ? 256 : map.stops.length;
  const width = Math.max(LUT_WIDTH, count);
  const lut = new Float32Array(width * LUT_ROWS * 4);
  const row = (r: number, i: number) => (r * width + i) * 4;
  if (builtin) {
    const t = table(map);
    for (let i = 0; i < 256; i++) {
      lut.set([t[3 * i], t[3 * i + 1], t[3 * i + 2], 255], row(0, i));
    }
    // The running integral of the table, linear between entries.
    const sum = [0, 0, 0, 0];
    for (let i = 0; i < 256; i++) {
      if (i > 0) {
        for (let k = 0; k < 4; k++) {
          sum[k] += (lut[row(0, i - 1) + k] + lut[row(0, i) + k]) / (2 * 255);
        }
      }
      lut.set(sum, row(2, i));
    }
  } else {
    map.stops.forEach(([t, color], i) => {
      lut.set(color, row(0, i));
      lut[row(1, i)] = t;
    });
  }
  for (let i = 0; i < 256; i++) {
    lut[row(3, i)] = SRGB_TO_LINEAR16[i] / 65535;
  }
  return {
    lut,
    width,
    count,
    builtin,
    range: [f(rule.range[0]), f(rule.range[1])],
    center: rule.center === undefined ? Number.NaN : f(rule.center),
    belowBase: (rule.below ?? "clamp") === "base",
    aboveBase: (rule.above ?? "base") === "base",
    base: rule.base ?? DEFAULT_BASE,
    nan: rule.nan ?? DEFAULT_NAN,
  };
}

/**
 * Writes the attribute of `values[from, to)` (see the notes at the top) into
 * `out` from element `2 · from`: `(value, NaN flag)` per vertex.
 */
export function encodeValues(
  s: Pick<Shading, "range">,
  values: ArrayLike<number>,
  out: Float32Array,
  from: number,
  to: number,
): void {
  const [lo, hi] = s.range;
  const span = FAR_SPAN * (hi - lo);
  for (let i = from; i < to; i++) {
    const v = values[i];
    const nan = Number.isNaN(v);
    out[2 * i] = nan || v === Infinity ? hi + span : v === -Infinity ? lo - span : v;
    out[2 * i + 1] = nan ? 1 : 0;
  }
}

// ---- The JavaScript reference of the shader (mirrors SHADING_GLSL) ----

/** Four channels. */
export type Vec4 = [number, number, number, number];

function texel(s: Shading, r: number, i: number): Vec4 {
  const at = (r * s.width + i) * 4;
  return [s.lut[at], s.lut[at + 1], s.lut[at + 2], s.lut[at + 3]];
}

const lerp4 = (a: Vec4, b: Vec4, x: number): Vec4 => [
  f(a[0] + f(f(b[0] - a[0]) * x)),
  f(a[1] + f(f(b[1] - a[1]) * x)),
  f(a[2] + f(f(b[2] - a[2]) * x)),
  f(a[3] + f(f(b[3] - a[3]) * x)),
];

/** The map's colour at `t` in [0, 1] in sRGB bytes, unrounded (`sample` before rounding). */
function colorAt(s: Shading, t: number): Vec4 {
  if (s.builtin) {
    const x = f(t * 255);
    const i = Math.min(Math.floor(x), 254);
    return lerp4(texel(s, 0, i), texel(s, 0, i + 1), f(x - i));
  }
  const last = s.count - 1;
  if (t <= texel(s, 1, 0)[0]) {
    return texel(s, 0, 0);
  }
  if (t >= texel(s, 1, last)[0]) {
    return texel(s, 0, last);
  }
  for (let k = 1; k <= last; k++) {
    const p1 = texel(s, 1, k)[0];
    if (t <= p1) {
      const p0 = texel(s, 1, k - 1)[0];
      const span = f(p1 - p0);
      return lerp4(texel(s, 0, k - 1), texel(s, 0, k), span > 0 ? f(f(t - p0) / span) : 1);
    }
  }
  return texel(s, 0, last);
}

/** Rounds half away from zero (Rust's `f32::round`) and clamps to a byte. */
const roundByte = (c: number): number => {
  const r = Math.floor(c);
  return Math.min(Math.max(c - r >= 0.5 ? r + 1 : r, 0), 255);
};

/** `t` of a value within the range, in `f32` as the engine computes it. */
function tOf(s: Shading, v: number): number {
  const [lo, hi] = s.range;
  const c = s.center;
  if (Number.isNaN(c)) {
    return f(f(v - lo) / f(hi - lo));
  }
  if (v === c) {
    return 0.5;
  }
  return v < c ? f(f(0.5 * f(v - lo)) / f(c - lo)) : f(0.5 + f(f(0.5 * f(v - c)) / f(hi - c)));
}

/** The colour of a finite value as the engine's `color_of` gives it, in sRGB bytes. */
export function exactBytes(s: Shading, v: number): Vec4 {
  const [lo, hi] = s.range;
  let c: Vec4;
  if (v < lo) {
    c = s.belowBase ? [...s.base] : colorAt(s, 0);
  } else if (v > hi) {
    c = s.aboveBase ? [...s.base] : colorAt(s, 1);
  } else {
    c = colorAt(s, Math.min(Math.max(tOf(s, v), 0), 1));
  }
  return [roundByte(c[0]), roundByte(c[1]), roundByte(c[2]), roundByte(c[3])];
}

/** A built-in table's running integral at `t`: row 2, then within the entry. */
function prefix(s: Shading, t: number): Vec4 {
  const i = Math.min(Math.floor(t * 255), 254);
  const d = t - i / 255;
  const g = texel(s, 2, i);
  const a = texel(s, 0, i);
  const b = texel(s, 0, i + 1);
  const at = (k: number) => g[k] + d * (a[k] + 0.5 * d * 255 * (b[k] - a[k]));
  return [at(0), at(1), at(2), at(3)];
}

/** The integral of the map's colour over `[t1, t2]` (`0 <= t1 <= t2 <= 1`). */
function integral(s: Shading, t1: number, t2: number): Vec4 {
  if (s.builtin && (t2 - t1) * 255 > PREFIX_MIN_ENTRIES) {
    const g1 = prefix(s, t1);
    const g2 = prefix(s, t2);
    return [g2[0] - g1[0], g2[1] - g1[1], g2[2] - g1[2], g2[3] - g1[3]];
  }
  // Trapezoids between the knots inside the window: exact, since the map is
  // linear between knots.
  const out: Vec4 = [0, 0, 0, 0];
  let prev = t1;
  let cp = colorAt(s, t1);
  const add = (p: number, c: Vec4) => {
    for (let k = 0; k < 4; k++) {
      out[k] += ((p - prev) * (cp[k] + c[k])) / 2;
    }
    prev = p;
    cp = c;
  };
  if (s.builtin) {
    const last = Math.ceil(t2 * 255) - 1;
    for (let k = Math.floor(t1 * 255) + 1; k <= last; k++) {
      add(k / 255, texel(s, 0, k));
    }
  } else {
    for (let k = 0; k < s.count; k++) {
      const p = texel(s, 1, k)[0];
      if (p > t1 && p < t2) {
        add(p, texel(s, 0, k));
      }
    }
  }
  add(t2, colorAt(s, t2));
  return out;
}

/** The mean colour of the values in `[v - h, v + h]` (`h > 0`), in unrounded sRGB bytes. */
export function softBytes(s: Shading, v: number, h: number): Vec4 {
  const [lo, hi] = s.range;
  const c = s.center;
  const a = v - h;
  const b = v + h;
  const sum: Vec4 = [0, 0, 0, 0];
  const acc = (w: number, col: Vec4) => {
    for (let k = 0; k < 4; k++) {
      sum[k] += w * col[k];
    }
  };
  acc(Math.max(0, Math.min(b, lo) - a), s.belowBase ? [...s.base] : colorAt(s, 0));
  acc(Math.max(0, b - Math.max(a, hi)), s.aboveBase ? [...s.base] : colorAt(s, 1));
  const ia = Math.max(a, lo);
  const ib = Math.min(b, hi);
  if (ib > ia) {
    if (Number.isNaN(c)) {
      acc(hi - lo, integral(s, (ia - lo) / (hi - lo), (ib - lo) / (hi - lo)));
    } else {
      const lowEnd = Math.min(ib, c);
      if (lowEnd > ia) {
        acc(2 * (c - lo), integral(s, (0.5 * (ia - lo)) / (c - lo), (0.5 * (lowEnd - lo)) / (c - lo)));
      }
      const highStart = Math.max(ia, c);
      if (ib > highStart) {
        acc(2 * (hi - c), integral(s, 0.5 + (0.5 * (highStart - c)) / (hi - c), 0.5 + (0.5 * (ib - c)) / (hi - c)));
      }
    }
  }
  return [sum[0] / (b - a), sum[1] / (b - a), sum[2] / (b - a), sum[3] / (b - a)];
}

/** sRGB bytes (fractional ones too) to linear RGBA, through row 3. */
function linear(s: Shading, c: Vec4): Vec4 {
  const out: Vec4 = [0, 0, 0, c[3] / 255];
  for (let k = 0; k < 3; k++) {
    const x = Math.min(Math.max(c[k], 0), 255);
    const i = Math.floor(x);
    const lo = texel(s, 3, i)[0];
    const hi = texel(s, 3, Math.min(i + 1, 255))[0];
    out[k] = lo + (hi - lo) * (x - i);
  }
  return out;
}

/**
 * The linear RGBA the shader gives the interpolated attribute
 * `(value, nanFlag)` at edge softness `softness`: the reference the GLSL
 * mirrors. With softness 0 at a vertex it is that vertex's colour in vertex
 * shading (`SRGB_TO_LINEAR16[color_of] / 65535`, alpha `/ 255`).
 */
export function shadeReference(s: Shading, value: number, nanFlag: number, softness: number): Vec4 {
  if (nanFlag > 0.5) {
    return linear(s, [...s.nan]);
  }
  return linear(s, softness > 0 ? softBytes(s, value, softness / 2) : exactBytes(s, value));
}

// ---- The GLSL (GLSL ES 3.0, as three compiles its materials) ----

/** Declarations for the vertex shader: the attribute and its varying. */
export const SHADING_VERTEX = /* glsl */ `
attribute vec2 fmValue;
varying vec2 vFmValue;
`;

/**
 * Declarations and functions for the fragment shader. `fmShade(vFmValue)`
 * is the linear RGBA of the fragment; each function mirrors the reference
 * function of the same name above (`fmColorAt` = `colorAt`, ...).
 */
export const SHADING_FRAGMENT = /* glsl */ `
varying vec2 vFmValue;
uniform highp sampler2D fmLut;
uniform int fmCount;
uniform bool fmBuiltin;
uniform vec2 fmRange;
uniform float fmCenter;
uniform bool fmHasCenter;
uniform bool fmBelowBase;
uniform bool fmAboveBase;
uniform vec4 fmBase;
uniform vec4 fmNan;
uniform float fmSoftness;
uniform bool fmOn;

vec4 fmTexel(int r, int i) {
  return texelFetch(fmLut, ivec2(i, r), 0);
}

vec4 fmLerp(vec4 a, vec4 b, float x) {
  return a + (b - a) * x;
}

vec4 fmColorAt(float t) {
  if (fmBuiltin) {
    float x = t * 255.0;
    int i = min(int(floor(x)), 254);
    return fmLerp(fmTexel(0, i), fmTexel(0, i + 1), x - float(i));
  }
  int last = fmCount - 1;
  if (t <= fmTexel(1, 0).r) {
    return fmTexel(0, 0);
  }
  if (t >= fmTexel(1, last).r) {
    return fmTexel(0, last);
  }
  for (int k = 1; k < fmCount; k++) {
    float p1 = fmTexel(1, k).r;
    if (t <= p1) {
      float p0 = fmTexel(1, k - 1).r;
      float span = p1 - p0;
      return fmLerp(fmTexel(0, k - 1), fmTexel(0, k), span > 0.0 ? (t - p0) / span : 1.0);
    }
  }
  return fmTexel(0, last);
}

vec4 fmRound(vec4 c) {
  vec4 r = floor(c);
  return clamp(r + step(0.5, c - r), 0.0, 255.0);
}

float fmT(float v) {
  float lo = fmRange.x;
  float hi = fmRange.y;
  float c = fmCenter;
  if (!fmHasCenter) {
    return (v - lo) / (hi - lo);
  }
  if (v == c) {
    return 0.5;
  }
  return v < c ? 0.5 * (v - lo) / (c - lo) : 0.5 + 0.5 * (v - c) / (hi - c);
}

vec4 fmExact(float v) {
  vec4 c;
  if (v < fmRange.x) {
    c = fmBelowBase ? fmBase : fmColorAt(0.0);
  } else if (v > fmRange.y) {
    c = fmAboveBase ? fmBase : fmColorAt(1.0);
  } else {
    c = fmColorAt(clamp(fmT(v), 0.0, 1.0));
  }
  return fmRound(c);
}

vec4 fmPrefix(float t) {
  int i = min(int(floor(t * 255.0)), 254);
  float d = t - float(i) / 255.0;
  vec4 a = fmTexel(0, i);
  return fmTexel(2, i) + d * (a + 0.5 * d * 255.0 * (fmTexel(0, i + 1) - a));
}

vec4 fmIntegral(float t1, float t2) {
  if (fmBuiltin && (t2 - t1) * 255.0 > ${PREFIX_MIN_ENTRIES}.0) {
    return fmPrefix(t2) - fmPrefix(t1);
  }
  vec4 sum = vec4(0.0);
  float prev = t1;
  vec4 cp = fmColorAt(t1);
  if (fmBuiltin) {
    int first = int(floor(t1 * 255.0)) + 1;
    int last = int(ceil(t2 * 255.0)) - 1;
    for (int j = 0; j <= ${PREFIX_MIN_ENTRIES + 1}; j++) {
      int k = first + j;
      if (k > last) {
        break;
      }
      float p = float(k) / 255.0;
      vec4 c = fmTexel(0, k);
      sum += (p - prev) * (cp + c) * 0.5;
      prev = p;
      cp = c;
    }
  } else {
    for (int k = 0; k < fmCount; k++) {
      float p = fmTexel(1, k).r;
      if (p > t1 && p < t2) {
        vec4 c = fmTexel(0, k);
        sum += (p - prev) * (cp + c) * 0.5;
        prev = p;
        cp = c;
      }
    }
  }
  return sum + (t2 - prev) * (cp + fmColorAt(t2)) * 0.5;
}

vec4 fmSoft(float v, float h) {
  float lo = fmRange.x;
  float hi = fmRange.y;
  float c = fmCenter;
  float a = v - h;
  float b = v + h;
  vec4 sum = max(0.0, min(b, lo) - a) * (fmBelowBase ? fmBase : fmColorAt(0.0));
  sum += max(0.0, b - max(a, hi)) * (fmAboveBase ? fmBase : fmColorAt(1.0));
  float ia = max(a, lo);
  float ib = min(b, hi);
  if (ib > ia) {
    if (!fmHasCenter) {
      sum += (hi - lo) * fmIntegral((ia - lo) / (hi - lo), (ib - lo) / (hi - lo));
    } else {
      float lowEnd = min(ib, c);
      if (lowEnd > ia) {
        sum += 2.0 * (c - lo) * fmIntegral(0.5 * (ia - lo) / (c - lo), 0.5 * (lowEnd - lo) / (c - lo));
      }
      float highStart = max(ia, c);
      if (ib > highStart) {
        sum += 2.0 * (hi - c) * fmIntegral(0.5 + 0.5 * (highStart - c) / (hi - c), 0.5 + 0.5 * (ib - c) / (hi - c));
      }
    }
  }
  return sum / (b - a);
}

vec4 fmLinear(vec4 c) {
  vec4 o = vec4(0.0, 0.0, 0.0, c.a / 255.0);
  for (int k = 0; k < 3; k++) {
    float x = clamp(c[k], 0.0, 255.0);
    int i = int(floor(x));
    o[k] = mix(fmTexel(3, i).r, fmTexel(3, min(i + 1, 255)).r, x - float(i));
  }
  return o;
}

vec4 fmShade(vec2 value) {
  if (value.y > 0.5) {
    return fmLinear(fmNan);
  }
  return fmLinear(fmSoftness > 0.0 ? fmSoft(value.x, 0.5 * fmSoftness) : fmExact(value.x));
}
`;
