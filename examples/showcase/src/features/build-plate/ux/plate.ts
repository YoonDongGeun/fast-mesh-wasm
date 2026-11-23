// The build plate (#266): a rectangle on the floor (the plane where the up
// axis is 0), its size, the corner it starts from, and how Arrange fills it.
// Lengths are millimetres, the app's world unit. Display settings, so they
// persist and are not undoable; read tolerantly from storage. Pure (no DOM,
// no three, no runtime imports), for settings.ts and the tests.
import type { UpAxis } from "../../transform/ux/placement-math";

export interface BuildPlate {
  /** Drawn in the viewport. */
  visible: boolean;
  /** Along X. */
  width: number;
  /** Along the floor's other axis: Y with Z up, Z with Y up. */
  depth: number;
  /** The corner with the smallest coordinates, along the same two axes. */
  origin: readonly [number, number];
  /** The axis that points away from the plate. */
  up: UpAxis;
  /** The least gap Arrange leaves between two footprints. */
  spacing: number;
  /** Arrange may turn meshes 90° about the up axis to fit more of them. */
  turn: boolean;
  /** Arrange keeps off the visible meshes it does not move. */
  keepClear: boolean;
}

/** 200 mm square around the world origin, 2 mm between meshes, Z up, hidden. */
export const DEFAULT_BUILD_PLATE: BuildPlate = {
  visible: false,
  width: 200,
  depth: 200,
  origin: [-100, -100],
  up: "z",
  spacing: 2,
  turn: true,
  keepClear: true,
};

/** World axis indices: `u` along the plate's width, `v` along its depth, `up` away from it. */
export interface PlateAxes {
  u: 0;
  v: 1 | 2;
  up: 1 | 2;
}

export function plateAxes(up: UpAxis): PlateAxes {
  return up === "y" ? { u: 0, v: 2, up: 1 } : { u: 0, v: 1, up: 2 };
}

/** The name of the plate's depth axis. */
export function depthAxisName(up: UpAxis): "Y" | "Z" {
  return up === "y" ? "Z" : "Y";
}

/** Whether the plate has a size and a place to be drawn at. */
export function plateHasSize(plate: Pick<BuildPlate, "width" | "depth" | "origin">): boolean {
  const { width, depth, origin } = plate;
  return width > 0 && depth > 0 && [width, depth, ...origin].every(Number.isFinite);
}

/** Why meshes cannot be arranged on the plate, in words, or `null` when they can. */
export function plateProblem(plate: BuildPlate): string | null {
  if (!plateHasSize(plate)) {
    return "The plate's width and depth must be above zero.";
  }
  if (!(plate.spacing >= 0 && Number.isFinite(plate.spacing))) {
    return "The spacing must be zero or more.";
  }
  return null;
}

/** Reads a stored plate: a missing or mistyped field takes its default. */
export function parseBuildPlate(value: unknown): BuildPlate {
  const d = DEFAULT_BUILD_PLATE;
  const doc = (typeof value === "object" && value !== null ? value : {}) as Record<string, unknown>;
  const flag = (v: unknown, fallback: boolean) => (typeof v === "boolean" ? v : fallback);
  const num = (v: unknown, fallback: number, min = -Infinity) =>
    typeof v === "number" && Number.isFinite(v) && v >= min ? v : fallback;
  const origin: unknown[] = Array.isArray(doc.origin) && doc.origin.length === 2 ? doc.origin : [];
  return {
    visible: flag(doc.visible, d.visible),
    width: num(doc.width, d.width, Number.MIN_VALUE),
    depth: num(doc.depth, d.depth, Number.MIN_VALUE),
    origin: [num(origin[0], d.origin[0]), num(origin[1], d.origin[1])],
    up: doc.up === "y" ? "y" : "z",
    spacing: num(doc.spacing, d.spacing, 0),
    turn: flag(doc.turn, d.turn),
    keepClear: flag(doc.keepClear, d.keepClear),
  };
}
