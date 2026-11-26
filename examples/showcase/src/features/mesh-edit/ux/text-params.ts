// What the text tool builds (#260): its parameters, their defaults and their
// checks. Apart from text-mesh.ts: the app state keeps the text, and needs its
// defaults before the Mesh Edit chunk is here.

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
