// The surface cleanup section's pure parts (#264): how the denoise and
// Delaunay flip fields parse, the face region they run on, the report lines
// and the before/after readouts. No React, no engine calls.
import type { DelaunayFlipReport, DenoiseReport, Placement, SurfaceStats } from "fast-mesh-wasm";
import { formatLength, type LengthUnit } from "../../../shared/ux/units";
import { sanitizeSelectedTriangles } from "../../select/ux/face-sets";
import { plural } from "./repair-summary";

export { SURFACE_CLEANUP_COMMANDS } from "./surface-cleanup-commands";

/** The engine's limit on denoise rounds. */
export const MAX_DENOISE_ITERATIONS = 100;
/** The engine's defaults, which the fields start at. */
export const DEFAULT_DENOISE_ITERATIONS = 5;
export const DEFAULT_DENOISE_STRENGTH = 0.5;
export const DEFAULT_FEATURE_ANGLE_DEG = 45;
export const DEFAULT_DIHEDRAL_LIMIT_DEG = 10;

/** A number field clamped to `min ..= max`; an empty or invalid entry reads as `fallback`. */
function clampedInput(text: string, min: number, max: number, fallback: number): number {
  const n = Number(text);
  if (text.trim() === "" || !Number.isFinite(n)) {
    return fallback;
  }
  return Math.max(min, Math.min(max, n));
}

/** The rounds field: a whole number in `1 ..= 100`. */
export function denoiseIterationsInput(text: string): number {
  return Math.round(clampedInput(text, 1, MAX_DENOISE_ITERATIONS, DEFAULT_DENOISE_ITERATIONS));
}

/** The strength slider, in `0.05 ..= 1` (0 would do nothing). */
export function denoiseStrengthInput(text: string): number {
  return clampedInput(text, 0.05, 1, DEFAULT_DENOISE_STRENGTH);
}

/** The feature angle field, in degrees `1 ..= 180`. */
export function featureAngleInput(text: string): number {
  return clampedInput(text, 1, 180, DEFAULT_FEATURE_ANGLE_DEG);
}

/** The dihedral limit field, in degrees `0 ..= 90`. */
export function dihedralLimitInput(text: string): number {
  return clampedInput(text, 0, 90, DEFAULT_DIHEDRAL_LIMIT_DEG);
}

/**
 * The triangles to run on: the mesh's live selected faces while the section
 * is limited to them, else `undefined` for the whole mesh.
 */
export function cleanupRegion(
  indices: ArrayLike<number>,
  selected: Uint32Array | undefined,
  selectedOnly: boolean,
): Uint32Array | undefined {
  return selectedOnly ? sanitizeSelectedTriangles(indices, selected ?? []) : undefined;
}

/** The largest per-axis scale of a placement (1 without one): how much longer a local length can be in the world. */
export function largestScale(placement: Pick<Placement, "scale">): number {
  return placement.scale === undefined ? 1 : Math.max(...placement.scale);
}

/** An angle in degrees, to two decimals. */
export function formatDegrees(deg: number): string {
  return `${deg.toFixed(2)}°`;
}

/** One before/after readout. */
export interface SurfaceReadout {
  key: "normalDeviation" | "minAngle";
  label: string;
  before: string;
  after: string;
}

/** The before/after readouts of either op's report. */
export function surfaceReadouts(r: { before: SurfaceStats; after: SurfaceStats }): SurfaceReadout[] {
  return [
    {
      key: "normalDeviation",
      label: "Mean normal deviation",
      before: formatDegrees(r.before.meanNormalDeviation),
      after: formatDegrees(r.after.meanNormalDeviation),
    },
    {
      key: "minAngle",
      label: "Minimum angle",
      before: formatDegrees(r.before.minAngle),
      after: formatDegrees(r.after.minAngle),
    },
  ];
}

function readoutText(r: { before: SurfaceStats; after: SurfaceStats }): string {
  return surfaceReadouts(r)
    .map((x) => `${x.label.toLowerCase()} ${x.before} → ${x.after}`)
    .join(", ");
}

/** The denoise report line; `scale` turns the engine's local displacement into a world length. */
export function describeDenoise(r: DenoiseReport, unit: LengthUnit, scale = 1): string {
  if (r.moved === 0) {
    return "Denoise: nothing to smooth";
  }
  const kept = r.reverted > 0 ? `, ${plural(r.reverted, "vertex", "vertices")} left in place` : "";
  return (
    `Denoise: moved ${plural(r.moved, "vertex", "vertices")} ` +
    `(max ${formatLength(r.maxDisplacement * scale, unit)})${kept}, ${readoutText(r)}`
  );
}

/** The Delaunay flips report line. */
export function describeDelaunayFlips(r: DelaunayFlipReport): string {
  if (r.flips === 0) {
    return "Delaunay flips: nothing to flip";
  }
  const stopped = r.converged ? "" : " (stopped at the pass limit)";
  return `Delaunay flips: ${plural(r.flips, "edge")} in ${plural(r.passes, "pass", "passes")}${stopped}, ${readoutText(r)}`;
}
