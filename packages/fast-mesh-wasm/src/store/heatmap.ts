// `contactHeatmap`: a preset `DistanceColoring` (without its target) for
// showing how two meshes touch: penetration in reds, contact in white, the gap
// in green to blue. Pure data: no scene, no WebAssembly.
import { FastMeshError } from "../core.js";
import type { Rgba, SignMode } from "../scene.js";
import type { DistanceColoring } from "./snapshot.js";

/** Options of {@link contactHeatmap}. Distances are in the meshes' units. */
export interface ContactHeatmapOptions {
  /** The penetration depth shown darkest (`deep`); deeper clamps to it. `> 0`. */
  depth: number;
  /**
   * The gap shown `far`; farther shows the base colour. `>= 0`: at 0 only
   * penetration (and the contact band) is coloured.
   */
  gap: number;
  /**
   * Distances in `[-tolerance, +tolerance]` are contact (`contact`). In
   * `[0, min(depth, gap))`, or `[0, depth)` when `gap` is 0; defaults to 0.
   */
  tolerance?: number;
  /**
   * `true` computes every vertex's distance (`maxDistance: Infinity`), so any
   * penetration deeper than `depth` shows `deep`; every update recomputes the
   * whole field. `false` (the default) bounds the field at
   * `2 · max(depth, gap)`, the fast contact path:
   * vertices beyond it are far and show the base colour, deep penetration
   * included. The bound sits past the range ends, so the values just beyond
   * the coloured area are still known and its outer edge is a value
   * boundary (which `bindMesh`'s fragment shading softens with
   * `edgeSoftness`); only an edge from a coloured vertex to one beyond the
   * bound (a deep interior, or a triangle longer than the margin) switches
   * hard, half way along.
   */
  exactDepth?: boolean;
  /** Defaults to `"auto"`. */
  sign?: SignMode;
  /** Overrides of the default colours. */
  colors?: { deep?: Rgba; shallow?: Rgba; contact?: Rgba; near?: Rgba; far?: Rgba };
}

/** The default colours: `#7f0000`, `#ff6b6b`, `#ffffff`, `#4caf50`, `#1e3a8a`. */
const DEFAULT_COLORS = {
  deep: [0x7f, 0x00, 0x00, 255],
  shallow: [0xff, 0x6b, 0x6b, 255],
  contact: [0xff, 0xff, 0xff, 255],
  near: [0x4c, 0xaf, 0x50, 255],
  far: [0x1e, 0x3a, 0x8a, 255],
} as const satisfies Record<string, Rgba>;

/**
 * Fast mode bounds the field at this multiple of `max(depth, gap)`: past the
 * range ends, so the coloured area's outer edge is a value boundary.
 */
export const FAST_BOUND_FACTOR = 2;

/** How far (in t) the `shallow` and `near` stops sit from the contact band's edges. */
const EDGE = 1e-4;

function invalid(message: string): FastMeshError {
  return new FastMeshError("INVALID_ARGUMENT", message);
}

/**
 * A contact heatmap: spread it into a colouring with its target,
 * `store.setDisplay(a, { coloring: { distanceTo: b, ...contactHeatmap({ depth, gap }) } })`.
 *
 * The range is `[-depth, gap]` (below clamps to `deep`, above shows the base
 * colour). From `-depth` the colour runs from `deep` to `shallow` just below
 * the contact band, is `contact` across `[-tolerance, +tolerance]` (a single
 * stop at 0 when the tolerance is 0), then runs from `near` just above the
 * band to `far` at `gap`. With `gap` 0 there is no gap ramp: the range is
 * `[-depth, tolerance]` and anything beyond the band shows the base colour.
 * A change of `tolerance` or `colors` alone keeps `maxDistance` and `sign`,
 * so the store reuses its field and only recolours.
 *
 * @throws `INVALID_ARGUMENT` unless `depth > 0`, `gap >= 0` (both finite) and
 * `0 ≤ tolerance < min(depth, gap)` (`< depth` when `gap` is 0).
 */
export function contactHeatmap(options: ContactHeatmapOptions): Omit<DistanceColoring, "distanceTo"> {
  const { depth, gap, tolerance = 0, exactDepth = false, sign = "auto" } = options;
  if (typeof depth !== "number" || !Number.isFinite(depth) || !(depth > 0)) {
    throw invalid("depth must be a finite number > 0");
  }
  if (typeof gap !== "number" || !Number.isFinite(gap) || !(gap >= 0)) {
    throw invalid("gap must be a finite number >= 0");
  }
  // gap 0 colours penetration only: the band is then bounded by depth alone.
  const ramp = gap > 0;
  if (typeof tolerance !== "number" || !(tolerance >= 0 && tolerance < (ramp ? Math.min(depth, gap) : depth))) {
    throw invalid("tolerance must be a number in [0, min(depth, gap)), or [0, depth) when gap is 0");
  }
  const o = options.colors;
  const c: Record<keyof typeof DEFAULT_COLORS, Rgba> = {
    deep: o?.deep ?? DEFAULT_COLORS.deep,
    shallow: o?.shallow ?? DEFAULT_COLORS.shallow,
    contact: o?.contact ?? DEFAULT_COLORS.contact,
    near: o?.near ?? DEFAULT_COLORS.near,
    far: o?.far ?? DEFAULT_COLORS.far,
  };
  // The range's top: `gap`, or the band's upper edge without a gap ramp.
  const top = ramp ? gap : tolerance;
  const t = (v: number): number => (v + depth) / (depth + top);
  const lo = t(-tolerance);
  // Without a gap ramp the band ends the range (exactly 1, for the end stop).
  const hi = ramp ? t(tolerance) : 1;
  // The band's lower edge is inside (0, 1]: tolerance < depth. With a gap
  // ramp its upper edge is inside (0, 1) too: tolerance < gap.
  const below = Math.min(EDGE, lo / 2);
  const above = Math.min(EDGE, (1 - hi) / 2);
  const band: [number, Rgba][] = tolerance === 0 ? [[lo, c.contact]] : [[lo, c.contact], [hi, c.contact]];
  const wanted: [number, Rgba][] = [
    [0, c.deep],
    [lo - below, c.shallow],
    ...band,
    ...(ramp
      ? ([
          [hi + above, c.near],
          [1, c.far],
        ] as [number, Rgba][])
      : []),
  ];
  // Stops reach the engine as f32 and must stay strictly increasing there:
  // drop a stop that rounds onto the one before it (only with a band edge
  // closer to an end than f32 resolves). The ends are always kept.
  const stops: (readonly [number, Rgba])[] = [];
  for (const [pos, color] of wanted) {
    const prev = stops[stops.length - 1];
    if (prev !== undefined && !(Math.fround(pos) > Math.fround(prev[0]))) {
      if (pos !== 1) {
        continue;
      }
      stops.pop();
    }
    stops.push(Object.freeze([pos, color] as const));
  }
  return {
    maxDistance: exactDepth ? Infinity : FAST_BOUND_FACTOR * Math.max(depth, gap),
    sign,
    map: { stops: Object.freeze(stops) },
    range: Object.freeze([-depth, top] as const),
    below: "clamp",
    above: "base",
  };
}
