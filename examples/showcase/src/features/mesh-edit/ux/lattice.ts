// The lattice section's pure parts (#265): its settings, the lengths a part
// starts with, the engine options they make, and the lines for the planned
// grid, the preview and the finished fill. Lengths are in millimetres (the
// engine's world unit); text follows the unit setting.
import type { LatticeKind, LatticeOptions, LatticePlan, LatticeRegion, LatticeReport } from "fast-mesh-wasm";
import { formatLength, formatVolume, type LengthUnit } from "../../../shared/ux/units";
import { formatBytes } from "../../mesh-repair/ux/wrap";

/** The lattice types, their labels and what they are. */
export const LATTICE_KINDS: readonly (readonly [LatticeKind, string, string])[] = [
  ["gyroid", "Gyroid", "About as stiff in every direction"],
  ["schwarzP", "Schwarz P", "Round chambers joined by necks, in a cubic grid"],
  ["diamond", "Diamond", "The most wall for the same cell and wall thickness"],
];

export interface LatticeSettings {
  kind: LatticeKind;
  /** The cell size in mm; `null` for the part's default. */
  cell: number | null;
  /** The wall thickness in mm; `null` for the cell's default. */
  wall: number | null;
  /** Whether a solid shell is left, and how thick in mm; `null` for the wall's thickness. */
  shell: boolean;
  shellThickness: number | null;
  region: LatticeRegion;
  /** The voxel size in mm; `null` for the one the lengths give. */
  voxel: number | null;
}

export const DEFAULT_LATTICE_SETTINGS: LatticeSettings = {
  kind: "gyroid",
  cell: null,
  wall: null,
  shell: true,
  shellThickness: null,
  region: "auto",
  voxel: null,
};

/** Cells along the longest side of a part, by default. */
const DEFAULT_CELLS = 6;
/** A wall's share of the cell, by default, and at most. */
const DEFAULT_WALL_SHARE = 0.15;
const MAX_WALL_SHARE = 1 / 3;

/** Two digits: lengths a person would type. */
function round2(mm: number): number {
  return Number(mm.toPrecision(2));
}

/** The lengths in mm the settings come to on a part of that size (its longest side). */
export interface LatticeLengths {
  cell: number;
  wall: number;
  /** `0` without a shell. */
  shell: number;
}

/** The typed lengths, and the defaults for the rest: a sixth of the part, 15 % of the cell, the wall. */
export function latticeLengths(settings: LatticeSettings, extent: number): LatticeLengths {
  const cell = settings.cell ?? round2(extent / DEFAULT_CELLS);
  const wall = settings.wall ?? round2(cell * DEFAULT_WALL_SHARE);
  const shell = settings.shell ? (settings.shellThickness ?? wall) : 0;
  return { cell, wall, shell };
}

/** Why the lengths cannot make a lattice, or `null`. */
export function latticeProblem(lengths: LatticeLengths): string | null {
  if (!(lengths.cell > 0)) return "Set a cell size.";
  if (!(lengths.wall > 0)) return "Set a wall thickness.";
  if (lengths.wall > lengths.cell * MAX_WALL_SHARE * (1 + 1e-6)) {
    return "Walls can be at most a third of the cell size thick.";
  }
  return null;
}

/** The engine options for `settings` and the lengths they come to. */
export function latticeOptions(settings: LatticeSettings, lengths: LatticeLengths): LatticeOptions {
  return {
    kind: settings.kind,
    cellSize: lengths.cell,
    wallThickness: lengths.wall,
    shellThickness: lengths.shell,
    region: settings.region,
    ...(settings.voxel !== null && settings.voxel > 0 ? { voxelSize: settings.voxel } : {}),
  };
}

/** A count, in thousands or millions from ten thousand on. */
function formatCount(n: number): string {
  if (n < 1e4) return Math.round(n).toLocaleString("en");
  if (n < 1e6) return `${(n / 1e3).toFixed(0)}k`;
  if (n < 1e9) return `${(n / 1e6).toFixed(1)}M`;
  return `${(n / 1e9).toFixed(1)}G`;
}

const percent = (share: number) => `${(100 * share).toFixed(0)}%`;

/** The planned grid, what it makes, and the limit when it is over it. */
export function describeLatticePlan(plan: LatticePlan, unit: LengthUnit): string {
  const grid =
    `Grid ${plan.dims.join(" × ")}: ${formatCount(plan.voxels)} voxels of ${formatLength(plan.voxelSize, unit)}, ` +
    `about ${formatCount(plan.triangles)} triangles and ${formatBytes(plan.bytes)}`;
  if (!plan.fits) {
    return `${grid}, over the limit of ${formatCount(plan.maxVoxels)} voxels. Use larger cells, thicker walls or a larger voxel size.`;
  }
  const region = plan.region === "cavity" ? "the cavity" : "the inside";
  return `${grid}. Fills ${region} (${formatVolume(plan.regionVolume, unit)}); the walls take ${percent(plan.solidFraction)}.`;
}

/** What the preview shows, and how it differs from the lattice. */
export function describeLatticePreview(plan: LatticePlan, triangles: number, wall: number, unit: LengthUnit): string {
  if (triangles === 0) {
    return "Preview: no walls at these sizes.";
  }
  const notes = [`${formatCount(triangles)} triangles on ${formatCount(plan.voxels)} voxels`];
  if (plan.wallThickness > wall * 1.001) {
    notes.push(`walls drawn ${formatLength(plan.wallThickness, unit)} thick`);
  }
  if (plan.cropped) {
    notes.push("the middle of the part only");
  }
  return `Preview: ${notes.join(", ")}.`;
}

/** The report line of a finished fill; `shell` is the thickness it left. */
export function describeLattice(
  name: string,
  kind: LatticeKind,
  r: LatticeReport,
  shell: number,
  unit: LengthUnit,
): string {
  const label = LATTICE_KINDS.find(([id]) => id === kind)?.[1] ?? kind;
  const left = percent(r.solidVolume / r.plan.regionVolume);
  const region = r.plan.region === "cavity" ? "cavity" : "volume";
  // Bits of wall of an open lattice, and of the space between them otherwise.
  const open = r.plan.region === "inside" && shell === 0;
  const pieces = `${r.fragments.toLocaleString("en")} ${open ? "loose piece" : "small pocket"}${r.fragments === 1 ? "" : "s"}`;
  const loose = r.fragments > 0 ? ` ${pieces} ${open ? "left out" : "filled"}.` : "";
  return (
    `${name}: ${label} lattice, ${r.triangles.toLocaleString("en")} triangles; ` +
    `${left} of the ${region} is solid (${formatVolume(r.solidVolume, unit)}).${loose}`
  );
}
