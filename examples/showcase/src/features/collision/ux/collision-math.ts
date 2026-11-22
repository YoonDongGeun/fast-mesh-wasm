// The collision panel's arithmetic: slider scales and the tolerance limit
// (contactHeatmap's validation), the deepest-penetration readout from the
// fields' values, a target's topology from its analysis, and how distances
// are printed.
import type {
  Analysis,
  CollisionMode,
  CollisionRule,
  CollisionStepReport,
  MeshHandle,
} from "fast-mesh-wasm";
import type { Wave6CollisionState } from "../../../app/state/wave6-state";

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

/** A bounded field's deepest value within this fraction of the bound may hide a deeper one. */
const NEAR_BOUND = 0.1;

/**
 * The deepest penetration over the meshes' field values: `-min` over the
 * finite negative values (0 when none is negative), or `null` when no mesh
 * has values.
 *
 * A field bounded at `bound` (`maxDistance`, fast mode) stores Infinity
 * beyond it, deep interiors included, so the true penetration can be
 * deeper than what it holds. `atLeast` says so: when the meshes intersect
 * and the deepest value is within 10% of the bound, or there is no finite
 * negative value at all. An unbounded field (`Infinity`) is exact.
 *
 * `bound` is the field's own `maxDistance` (fast mode: `2 · max(depth, gap)`),
 * not the depth slider: a value between `depth` and the bound is measured
 * exactly (its colour clamps to the deepest red), so only a value near the
 * bound can hide a deeper one.
 */
export function deepestPenetration(
  values: readonly (Float32Array | null)[],
  bound: number,
  intersects: boolean,
): { depth: number; atLeast: boolean } | null {
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
  return { depth, atLeast: mayBeDeeper(depth, bound, intersects) };
}

/**
 * Whether a penetration `depth` read from fields bounded at `bound` may hide
 * a deeper one: the bound is finite (fast mode), the meshes intersect, and
 * the depth is within 10% of the bound, or 0 (no vertex inside the other
 * within the bound). See `deepestPenetration`.
 */
export function mayBeDeeper(depth: number, bound: number, intersects: boolean): boolean {
  return Number.isFinite(bound) && intersects && (depth === 0 || depth >= bound * (1 - NEAR_BOUND));
}

/**
 * What an analysis says about a mesh as a distance target: `open` with any
 * boundary loop (holes and open chains) or a component that is not closed;
 * `inconsistent` with orientation conflicts. Either way a field's sign
 * follows the surface normals.
 */
export function topologyOf(a: Pick<Analysis, "loops" | "components" | "orientationConflicts">): {
  open: boolean;
  inconsistent: boolean;
} {
  return {
    open: a.loops.length > 0 || a.components.some((c) => !c.closed),
    inconsistent: a.orientationConflicts > 0,
  };
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

export const DEFAULT_COLLISION_STIFFNESS = 0.5;
export const DEFAULT_CONTACT_MARGIN = 0.0;

function clampNonNegative(value: number, fallback: number): number {
  return Number.isFinite(value) ? Math.max(0, value) : fallback;
}

function clampUnitInterval(value: number, fallback: number): number {
  return Number.isFinite(value) ? Math.min(1, Math.max(0, value)) : fallback;
}

/** Build a full `CollisionRule` from showcase `Wave6CollisionState`, mapping `cutBackend` to `cutMethod`. */
export function buildCollisionRule(
  collision: Wave6CollisionState,
  overrides?: {
    mode?: CollisionMode;
    stiffness?: number;
    contactMargin?: number;
  },
): CollisionRule {
  return {
    mode: overrides?.mode ?? collision.mode,
    role: collision.role,
    distanceBackend: collision.distanceBackend,
    cutMethod: collision.cutBackend,
    cutTrigger: collision.cutTrigger,
    softSolver: collision.softSolver,
    materialMemory: collision.materialMemory,
    compliance: clampNonNegative(collision.compliance, Math.fround(0.05)),
    stiffness: clampNonNegative(
      overrides?.stiffness ?? DEFAULT_COLLISION_STIFFNESS,
      DEFAULT_COLLISION_STIFFNESS,
    ),
    recoveryRate: clampUnitInterval(collision.recoveryRate, 1.0),
    contactMargin: clampNonNegative(
      overrides?.contactMargin ?? DEFAULT_CONTACT_MARGIN,
      DEFAULT_CONTACT_MARGIN,
    ),
    selfCollisionGuard: collision.preventSelfIntersection,
    rigidSlide: collision.rigidSlide,
  };
}

/** Stable string key for comparing two normalized `CollisionRule` payloads. */
export function collisionRuleSignature(rule: CollisionRule): string {
  return JSON.stringify([
    rule.mode,
    rule.role,
    rule.distanceBackend,
    rule.cutMethod,
    rule.cutTrigger,
    rule.softSolver,
    rule.materialMemory,
    rule.compliance,
    rule.stiffness,
    rule.recoveryRate,
    rule.contactMargin,
    rule.selfCollisionGuard,
    rule.rigidSlide,
  ]);
}

/** Human-readable summary of a `CollisionStepReport` for UI readouts and `wave6.collision.lastStepSummary`. */
export function summarizeCollisionStepReport(report: CollisionStepReport<MeshHandle>): string {
  const parts: string[] = [];
  if (Number.isFinite(report.minClearance)) {
    parts.push(`clearance ${formatDistance(report.minClearance)}`);
  }
  parts.push(`${report.activePairs.length} active pair${report.activePairs.length === 1 ? "" : "s"}`);
  if (report.cutsExecuted > 0) {
    parts.push(`${report.cutsExecuted} cut${report.cutsExecuted === 1 ? "" : "s"}`);
  }
  if (report.softVerticesDeformed > 0) {
    parts.push(`${report.softVerticesDeformed} soft vert${report.softVerticesDeformed === 1 ? "" : "s"}`);
  }
  if (report.elasticRecoveryCount > 0) {
    parts.push(`${report.elasticRecoveryCount} recovered`);
  }
  if (report.selfIntersectionGuardTriggered) {
    parts.push("self-guard triggered");
  }
  return parts.join(" · ");
}
