// The collision panel's arithmetic: slider scales and the tolerance limit
// (contactHeatmap's validation), the deepest-penetration readout from the
// fields' values, and how distances are printed.

/** The smallest of 1, 2 or 5 times a power of ten that is at least `r` (1 for a non-positive `r`). */
export function niceScale(r: number): number {
  if (!(r > 0) || !Number.isFinite(r)) {
    return 1;
  }
  const p = 10 ** Math.floor(Math.log10(r));
  for (const m of [1, 2, 5, 10]) {
    // A relative slack: 2 must not round up to 5 through floating point.
    if (m * p >= r * (1 - 1e-12)) {
      return m * p;
    }
  }
  return 10 * p;
}

/** The tolerance must stay below this: `min(depth, gap)`, or `depth` when the gap ("colour from") is 0. */
export function toleranceLimit(depth: number, gap: number): number {
  return gap > 0 ? Math.min(depth, gap) : depth;
}

/** `tolerance` clamped into `[0, limit - step]` (0 when that is empty), so it stays strictly below the limit. */
export function clampTolerance(tolerance: number, depth: number, gap: number, step: number): number {
  const max = toleranceLimit(depth, gap) - step;
  return max <= 0 ? 0 : Math.min(Math.max(tolerance, 0), max);
}

/**
 * The deepest penetration over the meshes' field values: `-min` over the
 * finite negative values (0 when none is negative), or `null` when no mesh
 * has values. `atBound` when that value sits at the field's `bound`
 * (`maxDistance`; values are f32): the penetration may go deeper.
 */
export function deepestPenetration(
  values: readonly (Float32Array | null)[],
  bound: number,
): { depth: number; atBound: boolean } | null {
  let min = Infinity;
  let any = false;
  for (const v of values) {
    if (v === null) {
      continue;
    }
    any = true;
    for (let i = 0; i < v.length; i++) {
      const x = v[i];
      if (x < min) {
        min = x; // NaN never compares below; Infinity never below a finite min
      }
    }
  }
  if (!any) {
    return null;
  }
  const depth = min < 0 ? -min : 0;
  const atBound = Number.isFinite(bound) && depth > 0 && depth >= Math.fround(bound) * (1 - 1e-6);
  return { depth, atBound };
}

/** Three significant digits; exponent form below 0.001. */
export function formatDistance(x: number): string {
  if (x === 0) {
    return "0";
  }
  const a = Math.abs(x);
  if (a < 1e-3) {
    const [m, e] = x.toExponential(2).split("e");
    return `${m}e${Number(e)}`;
  }
  return a >= 100 ? x.toFixed(0) : String(Number(x.toPrecision(3)));
}
