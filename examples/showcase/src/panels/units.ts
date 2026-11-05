// Units (issue #145). The engine is unitless; this app reads its world unit
// as the millimetre, the unit DICOM speaks. Everything stored stays in it:
// mesh positions, heatmap sliders, colorings, placements, project files.
//
// Two different things convert, at the two boundaries:
// - The display unit changes text only. A readout converts on its way out,
//   an input also on its way back; nothing stored moves.
// - The import scale changes the geometry: a file's numbers are multiplied
//   once, when it loads, and the mesh is in millimetres from then on. It
//   rides along as provenance (the project file too), never to be re-applied.
import { formatCoord } from "../viewport/inspect-tools";
import { formatDistance } from "./collision-math";

/** A display unit: what readouts are shown in. */
export type LengthUnit = "mm" | "in";

/** Millimetres per inch, exactly. */
const MM_PER_INCH = 25.4;

/** Millimetres per one of `unit`. */
function mmPer(unit: LengthUnit): number {
  return unit === "in" ? MM_PER_INCH : 1;
}

/** A length in millimetres, in `unit`. */
export function toDisplayLength(mm: number, unit: LengthUnit): number {
  return unit === "mm" ? mm : mm / mmPer(unit);
}

/** A length typed or dragged in `unit`, in millimetres. */
export function fromDisplayLength(value: number, unit: LengthUnit): number {
  return unit === "mm" ? value : value * mmPer(unit);
}

/** An area in square millimetres, in square `unit`: the factor squared. */
export function toDisplayArea(mm2: number, unit: LengthUnit): number {
  return unit === "mm" ? mm2 : mm2 / mmPer(unit) ** 2;
}

/** A volume in cubic millimetres, in cubic `unit`: the factor cubed. */
export function toDisplayVolume(mm3: number, unit: LengthUnit): number {
  return unit === "mm" ? mm3 : mm3 / mmPer(unit) ** 3;
}

/** `formatDistance`'s digits for a length, without the unit (for a label that names it). */
export function formatLengthValue(mm: number, unit: LengthUnit): string {
  return formatDistance(toDisplayLength(mm, unit));
}

/** A length with its unit: `25.4 mm`, `1 in`. */
export function formatLength(mm: number, unit: LengthUnit): string {
  return `${formatLengthValue(mm, unit)} ${unit}`;
}

/** An area with its unit: `645 mm²`, `1 in²`. */
export function formatArea(mm2: number, unit: LengthUnit): string {
  return `${formatDistance(toDisplayArea(mm2, unit))} ${unit}²`;
}

/** A volume with its unit: `16387 mm³`, `1 in³`. */
export function formatVolume(mm3: number, unit: LengthUnit): string {
  return `${formatDistance(toDisplayVolume(mm3, unit))} ${unit}³`;
}

/** A coordinate with its unit, in `formatCoord`'s digits. */
export function formatCoordLength(mm: number, unit: LengthUnit): string {
  return `${formatCoord(toDisplayLength(mm, unit))} ${unit}`;
}

/** A point, its unit named once: `(1, 0, -2) in`. Never for a direction: normals have no unit. */
export function formatPoint(p: readonly number[], unit: LengthUnit): string {
  return `(${p.map((x) => formatCoord(toDisplayLength(x, unit))).join(", ")}) ${unit}`;
}

/** A range input's numbers: all four in the same unit. */
export interface LengthRange {
  min: number;
  max: number;
  step: number;
  value: number;
}

/** A range in millimetres, as its input shows it in `unit`; read a change back with `fromDisplayLength`. */
export function rangeToDisplay(range: LengthRange, unit: LengthUnit): LengthRange {
  return {
    min: toDisplayLength(range.min, unit),
    max: toDisplayLength(range.max, unit),
    step: toDisplayLength(range.step, unit),
    value: toDisplayLength(range.value, unit),
  };
}

/**
 * What a typed length input holds, for a stored `mm`. Rounded past float
 * noise when converting, so a typed `0.1` in reads back as `0.1` (the
 * controlled input would otherwise rewrite it under the caret).
 */
export function lengthInputValue(mm: number, unit: LengthUnit): number {
  return unit === "mm" ? mm : Number(toDisplayLength(mm, unit).toPrecision(12));
}

/** The units a file can be assumed to be in, on load. */
export type ImportUnit = "mm" | "cm" | "m" | "in";

/** Millimetres per file unit. */
export const IMPORT_FACTORS: Readonly<Record<ImportUnit, number>> = { mm: 1, cm: 10, m: 1000, in: MM_PER_INCH };

/** The assumed units in the order the panel lists them. */
export const IMPORT_UNITS: readonly ImportUnit[] = ["mm", "cm", "m", "in"];

/** Whether `value` names an import unit. */
export function isImportUnit(value: unknown): value is ImportUnit {
  return typeof value === "string" && (IMPORT_UNITS as readonly string[]).includes(value);
}

/**
 * How a file's numbers became millimetres: multiplied by `factor` once, on
 * load. `units` is the unit assumed, `null` for a custom factor.
 */
export interface ImportScale {
  factor: number;
  units: ImportUnit | null;
}

/** A file already in the world unit: its numbers stay as they are. */
export const NO_IMPORT_SCALE: ImportScale = { factor: 1, units: "mm" };

/** The scale of a file assumed to be in `units`. */
export function importScaleOf(units: ImportUnit): ImportScale {
  return { factor: IMPORT_FACTORS[units], units };
}

/** The scale of a typed factor: the unit it matches, else custom. */
export function importScaleFromFactor(factor: number): ImportScale {
  return { factor, units: IMPORT_UNITS.find((u) => IMPORT_FACTORS[u] === factor) ?? null };
}

/** A typed factor, or `null` for one that would destroy the geometry (not finite and positive). */
export function parseImportScale(text: string): number | null {
  const trimmed = text.trim();
  if (trimmed === "") {
    return null;
  }
  const factor = Number(trimmed);
  return Number.isFinite(factor) && factor > 0 ? factor : null;
}

/** Applies an import scale in place (off the page: the loader Worker owns the array). */
export function scalePositions(positions: Float64Array, factor: number): Float64Array {
  if (factor !== 1) {
    for (let i = 0; i < positions.length; i++) {
      positions[i] *= factor;
    }
  }
  return positions;
}
