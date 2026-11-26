// The shapes the Create dialog offers (#259): each one's parameters (lengths
// as multiples of the dialog's base size, resolutions as counts) and their
// ranges. Apart from create-primitive.ts: the command palette lists the
// shapes before the dialog's chunk is here.
import type { PrimitiveShapeKind } from "fast-mesh-wasm";

/** A length (millimetres) or a resolution (a whole number). */
export interface ParamSpec {
  key: string;
  label: string;
  type: "length" | "count";
  /** The default: a length as a multiple of the dialog's base size, a count as is. */
  value: number;
  /** Lengths: whether 0 is allowed (a cone's apex). Counts: the smallest value. */
  min: number;
  /** Counts only: the largest value. */
  max?: number;
}

/** A shape button: its kind, label and parameters, in the order they show. */
export interface ShapeSpec {
  kind: PrimitiveShapeKind;
  label: string;
  params: readonly ParamSpec[];
}

const length = (key: string, label: string, value: number, min = 1): ParamSpec => ({
  key,
  label,
  type: "length",
  value,
  min,
});
const count = (key: string, label: string, value: number, min: number, max: number): ParamSpec => ({
  key,
  label,
  type: "count",
  value,
  min,
  max,
});

/** A length that may be 0 (`min: 0`); others must be positive (`min: 1`). */
export const ZERO_OK = 0;
const SIDES = 8192;
const BANDS = 4096;

/** The shapes the dialog offers, in button order. */
export const CREATE_SHAPES: readonly ShapeSpec[] = [
  {
    kind: "box",
    label: "Box",
    params: [
      length("sizeX", "Width (X)", 1),
      length("sizeY", "Height (Y)", 1),
      length("sizeZ", "Depth (Z)", 1),
      count("segX", "Segments X", 1, 1, 2048),
      count("segY", "Segments Y", 1, 1, 2048),
      count("segZ", "Segments Z", 1, 1, 2048),
    ],
  },
  {
    kind: "cylinder",
    label: "Cylinder",
    params: [
      length("radius", "Radius", 0.5),
      length("height", "Height", 1),
      count("segments", "Sides", 48, 3, SIDES),
      count("rings", "Height segments", 1, 1, BANDS),
      count("capRings", "Cap rings", 1, 1, BANDS),
    ],
  },
  {
    kind: "cone",
    label: "Cone",
    params: [
      length("radius", "Base radius", 0.5, ZERO_OK),
      length("topRadius", "Top radius", 0, ZERO_OK),
      length("height", "Height", 1),
      count("segments", "Sides", 48, 3, SIDES),
      count("rings", "Height segments", 1, 1, BANDS),
      count("capRings", "Cap rings", 1, 1, BANDS),
    ],
  },
  {
    kind: "uvSphere",
    label: "Sphere",
    params: [length("radius", "Radius", 0.5), count("segments", "Segments", 48, 3, SIDES), count("rings", "Rings", 24, 2, BANDS)],
  },
  {
    kind: "icosphere",
    label: "Icosphere",
    params: [length("radius", "Radius", 0.5), count("subdivisions", "Subdivisions", 3, 0, 9)],
  },
  {
    kind: "torus",
    label: "Torus",
    params: [
      length("majorRadius", "Ring radius", 0.375),
      length("minorRadius", "Tube radius", 0.125),
      count("segments", "Ring segments", 48, 3, SIDES),
      count("tubeSegments", "Tube segments", 24, 3, SIDES),
    ],
  },
  {
    kind: "arrow",
    label: "Arrow",
    params: [
      length("length", "Length", 1),
      length("shaftRadius", "Shaft radius", 0.05),
      length("headRadius", "Head radius", 0.1),
      length("headLength", "Head length", 0.25),
      count("segments", "Sides", 32, 3, SIDES),
    ],
  },
  {
    kind: "prism",
    label: "Prism",
    params: [count("sides", "Sides", 6, 3, 1024), length("radius", "Radius", 0.5), length("height", "Height", 1)],
  },
  {
    kind: "plane",
    label: "Plane",
    params: [
      length("width", "Width (X)", 1),
      length("depth", "Depth (Z)", 1),
      count("segX", "Segments X", 10, 1, BANDS),
      count("segZ", "Segments Z", 10, 1, BANDS),
    ],
  },
];
