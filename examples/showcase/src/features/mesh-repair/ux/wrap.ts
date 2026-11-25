// The wrap section's pure parts (#263): its settings and their checks, the
// gap a selection starts with, the engine options they make, the new mesh's
// name, and the lines for the planned grid and the finished wrap.
import type { SceneLike, WrapOptions, WrapPlan } from "fast-mesh-wasm";
import type { Item } from "../../../app/state/app-state";
import { formatLength, type LengthUnit } from "../../../shared/ux/units";

/** How the voxel size is set: by voxels along the longest side, or directly. */
export type WrapSizing = "detail" | "voxel";

export interface WrapSettings {
  /** The widest gap that closes, in mm; `null` for the selection's default. */
  gap: number | null;
  sizing: WrapSizing;
  /** Voxels along the longest side. */
  detail: number;
  /** The voxel size in mm; `null` for the one the detail gives. */
  voxelSize: number | null;
  project: boolean;
}

export const DEFAULT_WRAP_SETTINGS: WrapSettings = {
  gap: null,
  sizing: "detail",
  detail: 128,
  voxelSize: null,
  project: true,
};

/** The engine's range for `detail`. */
export const MIN_WRAP_DETAIL = 8;
export const MAX_WRAP_DETAIL = 4096;

/** The default gap, as a share of the selection's longest side. */
const DEFAULT_GAP_SHARE = 0.02;

/** A typed detail: a whole number in the engine's range. */
export function wrapDetailInput(text: string): number {
  const n = Math.round(Number(text));
  if (!Number.isFinite(n)) {
    return DEFAULT_WRAP_SETTINGS.detail;
  }
  return Math.min(MAX_WRAP_DETAIL, Math.max(MIN_WRAP_DETAIL, n));
}

/** The longest side of the world box around `sources` (from their boxes' corners); 0 without geometry. */
export function worldExtent(scene: SceneLike, sources: readonly Item[]): number {
  const lo = [Infinity, Infinity, Infinity];
  const hi = [-Infinity, -Infinity, -Infinity];
  for (const { mesh } of sources) {
    const b = scene.bounds?.(mesh);
    if (!b) {
      continue;
    }
    const m = scene.matrix4(mesh);
    for (let i = 0; i < 8; i++) {
      const p = [i & 1 ? b.max[0] : b.min[0], i & 2 ? b.max[1] : b.min[1], i & 4 ? b.max[2] : b.min[2]];
      for (let k = 0; k < 3; k++) {
        const w = m[k] * p[0] + m[4 + k] * p[1] + m[8 + k] * p[2] + m[12 + k];
        lo[k] = Math.min(lo[k], w);
        hi[k] = Math.max(hi[k], w);
      }
    }
  }
  return lo[0] > hi[0] ? 0 : Math.max(hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]);
}

/** The gap a selection of that size starts with: two percent of it, to two digits. */
export function defaultWrapGap(extent: number): number {
  return extent > 0 ? Number((extent * DEFAULT_GAP_SHARE).toPrecision(2)) : 0;
}

/** The voxel size that `detail` gives a selection of that size. */
export function detailVoxelSize(extent: number, detail: number): number {
  return extent / detail;
}

/** The engine options for `settings` on a selection with the default gap `autoGap`. */
export function wrapOptions(settings: WrapSettings, autoGap: number): WrapOptions {
  const options: WrapOptions = { gap: Math.max(0, settings.gap ?? autoGap), project: settings.project };
  if (settings.sizing === "voxel" && settings.voxelSize !== null && settings.voxelSize > 0) {
    options.voxelSize = settings.voxelSize;
  } else {
    options.detail = settings.detail;
  }
  return options;
}

/** The new mesh's name: after its one source, or the number of sources. */
export function wrapMeshName(sources: readonly Item[]): string {
  return sources.length === 1 ? `${sources[0].name} wrap` : `Wrap of ${sources.length} meshes`;
}

/** A voxel count, in thousands or millions from there on. */
function formatCount(n: number): string {
  if (n < 1e4) return n.toLocaleString("en");
  if (n < 1e6) return `${(n / 1e3).toFixed(0)}k`;
  if (n < 1e9) return `${(n / 1e6).toFixed(1)}M`;
  return `${(n / 1e9).toFixed(1)}G`;
}

/** Bytes in MB or GB, at least 1 MB. */
export function formatBytes(bytes: number): string {
  const mb = bytes / 2 ** 20;
  return mb < 1024 ? `${Math.max(1, Math.round(mb))} MB` : `${(mb / 1024).toFixed(1)} GB`;
}

/** The planned grid: its nodes, voxel size and memory, and the limit when it is over it. */
export function describeWrapPlan(plan: WrapPlan, unit: LengthUnit): string {
  const grid =
    `Grid ${plan.dims.join(" × ")}: ${formatCount(plan.voxels)} voxels of ${formatLength(plan.voxelSize, unit)}, ` +
    `about ${formatBytes(plan.bytes)}`;
  return plan.fits
    ? `${grid}. Closes gaps under ${formatLength(2 * plan.radius, unit)}.`
    : `${grid}, over the limit of ${formatCount(plan.maxVoxels)} voxels. Use a larger voxel size or less detail.`;
}

/** The report line for a finished wrap. */
export function describeWrap(r: {
  name: string;
  vertices: number;
  triangles: number;
  projected: number;
  plan: WrapPlan;
}): string {
  const onto = r.projected > 0 ? `, ${r.projected.toLocaleString("en")} vertices on the input` : "";
  return (
    `${r.name}: ${r.vertices.toLocaleString("en")} vertices, ${r.triangles.toLocaleString("en")} triangles${onto} ` +
    `(grid ${r.plan.dims.join(" × ")}, ${formatBytes(r.plan.bytes)})`
  );
}
