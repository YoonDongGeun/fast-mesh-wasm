// Kept polylines as SVG and DXF (#268), through the contour exporter: the
// curves are drawn in one plane, the one that fits them best or one of the
// world's, in millimetres. What does not lie in the plane is flattened onto
// it, and the caller is told how far it was.
import type { Vec3 } from "../../../io/mesh-data";
import { contoursToDxf, contoursToSvg } from "../../../io/report";
import { AXIS_NORMAL, type Axis } from "../../inspect/ux/section-clip";
import { fitPlane, planarTolerance } from "./planar";
import type { PolylineCurve } from "./polyline";

/** The plane an export draws in: the best fit, or the one facing a world axis. */
export type ExportPlane = "fit" | Axis;

/** The export planes, in menu order. */
export const EXPORT_PLANES: readonly { plane: ExportPlane; label: string }[] = [
  { plane: "fit", label: "Their own plane" },
  { plane: "z", label: "XY (top)" },
  { plane: "y", label: "XZ (front)" },
  { plane: "x", label: "YZ (side)" },
];

/** An exported file's text, and how far the curves strayed from its plane (0 within the planar tolerance). */
export interface PolylineFile {
  text: string;
  flattened: number;
}

/** The curves as an SVG or DXF document. */
export function polylinesFile(curves: readonly PolylineCurve[], format: "svg" | "dxf", plane: ExportPlane): PolylineFile {
  const fit = fitPlane(curves);
  const origin: Vec3 = plane === "fit" && fit !== null ? fit.plane.origin : [0, 0, 0];
  const normal: Vec3 = plane === "fit" ? (fit?.plane.normal ?? [0, 0, 1]) : AXIS_NORMAL[plane];
  let min = Infinity;
  let max = -Infinity;
  for (const { points } of curves) {
    for (let i = 0; i + 2 < points.length; i += 3) {
      const d = (points[i] - origin[0]) * normal[0] + (points[i + 1] - origin[1]) * normal[1] + (points[i + 2] - origin[2]) * normal[2];
      min = Math.min(min, d);
      max = Math.max(max, d);
    }
  }
  const depth = max > min ? max - min : 0;
  const draw = format === "svg" ? contoursToSvg : contoursToDxf;
  return { text: draw(curves, origin, normal), flattened: depth > 2 * planarTolerance(curves) ? depth : 0 };
}
