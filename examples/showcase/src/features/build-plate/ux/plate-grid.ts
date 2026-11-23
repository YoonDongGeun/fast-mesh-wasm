// The build plate's lines (#266), in world millimetres: its outline, and a
// grid from its corner in steps of 1, 2 or 5 times a power of ten (about
// twenty along the longer side). Pure, for the viewport layer and the tests.
import { niceSize } from "../../home/ux/create-primitive";
import { type BuildPlate, plateAxes } from "./plate";

export interface PlateGrid {
  /** The grid's step. */
  step: number;
  /** The inner grid lines, two points each. */
  lines: Float32Array;
  /** The four corners in order around the plate, counter-clockwise seen from above. */
  corners: Float32Array;
}

/** The lines of a plate with a size (`plateHasSize`). */
export function plateGrid(plate: Pick<BuildPlate, "width" | "depth" | "origin" | "up">): PlateGrid {
  const { width, depth, origin, up } = plate;
  const { u, v } = plateAxes(up);
  const step = niceSize(Math.max(width, depth) / 20);
  const put = (out: number[], a: number, b: number) => {
    const p = [0, 0, 0];
    p[u] = origin[0] + a;
    p[v] = origin[1] + b;
    out.push(...p);
  };
  const points: number[] = [];
  // A line within a hair of the far edge is the edge.
  for (let k = 1; k * step < width * (1 - 1e-6); k++) {
    put(points, k * step, 0);
    put(points, k * step, depth);
  }
  for (let k = 1; k * step < depth * (1 - 1e-6); k++) {
    put(points, 0, k * step);
    put(points, width, k * step);
  }
  const around: [number, number][] = [
    [0, 0],
    [width, 0],
    [width, depth],
    [0, depth],
  ];
  // X to Z turns clockwise seen from +Y.
  if (up === "y") {
    around.reverse();
  }
  const corners: number[] = [];
  for (const [a, b] of around) {
    put(corners, a, b);
  }
  return { step, lines: Float32Array.from(points), corners: Float32Array.from(corners) };
}
