// Colour rules for the engine and for keeping: the binding's encoding of a
// `ColorRule` and a deep copy of one. Internal: not exported from the package.
import { DEFAULT_BASE, DEFAULT_NAN } from "./colormap.js";
import { FastMeshError } from "./core.js";
import type { ColorRule, OutOfRange } from "./scene.js";

const MAPS = { turbo: 0, viridis: 1, coolwarm: 2, grayscale: 3 } as const;
const OUT_OF_RANGE: Record<OutOfRange, number> = { clamp: 0, base: 1 };

function invalid(message: string): FastMeshError {
  return new FastMeshError("INVALID_ARGUMENT", message);
}

/** The code of a string option in `table`, or `INVALID_ARGUMENT` naming `what`. */
function codeIn(table: Record<string, number>, value: string, what: string): number {
  if (!Object.hasOwn(table, value)) {
    throw invalid(`${what} must be one of ${Object.keys(table).join(", ")}`);
  }
  return table[value];
}

/** A colour rule as the binding's `(map, rule)`: see `parse_rule` in the Rust crate. */
export function encodeRule(rule: ColorRule): { map: number; values: Float32Array } {
  if (rule.range?.length !== 2) {
    throw invalid("range must be [lo, hi]");
  }
  // NaN is the binding's "no centre": reject it here rather than drop it.
  if (rule.center !== undefined && Number.isNaN(rule.center)) {
    throw invalid("color center must lie within the range");
  }
  const base = rule.base ?? DEFAULT_BASE;
  const nan = rule.nan ?? DEFAULT_NAN;
  if (base.length !== 4) {
    throw invalid("base must have 4 channels");
  }
  if (nan.length !== 4) {
    throw invalid("nan must have 4 channels");
  }
  const values = [
    rule.range[0],
    rule.range[1],
    rule.center ?? Number.NaN,
    codeIn(OUT_OF_RANGE, rule.below ?? "clamp", "below"),
    codeIn(OUT_OF_RANGE, rule.above ?? "base", "above"),
    ...base,
    ...nan,
  ];
  let map: number;
  if (typeof rule.map === "string") {
    map = codeIn(MAPS, rule.map, "map");
  } else if (typeof rule.map === "object" && rule.map !== null && Array.isArray(rule.map.stops)) {
    map = 4;
    for (const stop of rule.map.stops as unknown[]) {
      if (!Array.isArray(stop) || stop.length !== 2) {
        throw invalid("each stop must be [t, rgba]");
      }
      const [t, color] = stop as [number, ArrayLike<number> | null];
      if (typeof color !== "object" || color === null || color.length !== 4) {
        throw invalid("stop colours must have 4 channels");
      }
      values.push(t, ...Array.from(color));
    }
  } else {
    throw invalid('map must be "turbo", "viridis", "coolwarm", "grayscale" or { stops }');
  }
  return { map, values: Float32Array.from(values) };
}

/**
 * A deep, frozen copy of a colour rule (the field handle kept, not frozen):
 * its range and colours, and a custom map's stops with their colours, so
 * later changes to the caller's object do not leak in and callers of
 * `coloring()` cannot change it. Colours given as typed arrays become plain
 * arrays (a typed array cannot be frozen). Copies what it is given; the
 * engine checks it.
 */
export function copyRule(rule: ColorRule): ColorRule {
  const frozen = (x: unknown): unknown =>
    Array.isArray(x) || ArrayBuffer.isView(x) ? Object.freeze(Array.from(x as ArrayLike<unknown>)) : x;
  const copy = { ...rule } as Record<string, unknown>;
  for (const [k, v] of Object.entries(copy)) {
    if (k !== "field") {
      copy[k] = frozen(v);
    }
  }
  const stops = (rule.map as { stops?: unknown } | null)?.stops;
  if (Array.isArray(stops)) {
    copy.map = Object.freeze({
      stops: Object.freeze(
        stops.map((stop: unknown) => (Array.isArray(stop) ? Object.freeze(stop.map(frozen)) : stop)),
      ),
    });
  }
  return Object.freeze(copy) as unknown as ColorRule;
}
