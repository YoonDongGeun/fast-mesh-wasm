// The kept polylines in the viewport (#268): each visible one as a line of
// its colour, a few pixels wide and just in front of the surface it lies on,
// with its hidden parts faint through the meshes; the selected ones thicker.
// Pointer rays pass through them: picking a mesh or a brush under a line
// works as if it were not there.
import { useEffect, useLayoutEffect, useMemo } from "react";
import * as THREE from "three";
import { LineMaterial } from "three/addons/lines/LineMaterial.js";
import { LineSegments2 } from "three/addons/lines/LineSegments2.js";
import { LineSegmentsGeometry } from "three/addons/lines/LineSegmentsGeometry.js";
import { type AppState, useApp } from "../../../app/state/app-state";
import { curveSegments, curvesSphere, type Polyline } from "../ux/polyline";

const LINE_PX = 2.5;
const SELECTED_PX = 4.5;
const THROUGH = 0.25;
const SELECTED_THROUGH = 0.5;

/** One polyline's lines: the visible part, and the whole faintly through everything. */
interface LineParts {
  group: THREE.Group;
  line: LineSegments2;
  through: LineSegments2;
  points: Float64Array;
  closed: boolean;
}

/** The polylines' lines, made and dropped as the list changes. */
export class PolylineLines {
  readonly group = new THREE.Group();
  readonly #parts = new Map<number, LineParts>();

  /** Draws `polylines`, with the `selected` ones thicker. */
  sync(polylines: readonly Polyline[], selected: readonly number[]): void {
    const ids = new Set(polylines.map((p) => p.id));
    const picked = new Set(selected);
    for (const [id, parts] of this.#parts) {
      if (!ids.has(id)) {
        dispose(parts);
        this.#parts.delete(id);
      }
    }
    for (const p of polylines) {
      let parts = this.#parts.get(p.id);
      if (parts !== undefined && (parts.points !== p.points || parts.closed !== p.closed)) {
        dispose(parts);
        this.#parts.delete(p.id);
        parts = undefined;
      }
      if (parts === undefined) {
        parts = build(p);
        this.#parts.set(p.id, parts);
        this.group.add(parts.group);
      }
      const on = picked.has(p.id);
      parts.group.visible = p.visible;
      parts.line.material.color.setHex(p.color);
      parts.through.material.color.setHex(p.color);
      parts.line.material.linewidth = on ? SELECTED_PX : LINE_PX;
      parts.through.material.linewidth = on ? SELECTED_PX : LINE_PX;
      parts.through.material.opacity = on ? SELECTED_THROUGH : THROUGH;
    }
  }

  /** The number of polylines with lines. */
  get size(): number {
    return this.#parts.size;
  }

  dispose(): void {
    for (const parts of this.#parts.values()) {
      dispose(parts);
    }
    this.#parts.clear();
    this.group.removeFromParent();
  }
}

/** Rays pass through. */
function noRaycast(): void {}

function build(p: Polyline): LineParts {
  const center = curvesSphere([p])?.center ?? [0, 0, 0];
  const geometry = new LineSegmentsGeometry();
  geometry.setPositions(curveSegments(p, center));
  const line = new LineSegments2(
    geometry,
    new LineMaterial({
      toneMapped: false,
      // In front of the surface it lies on.
      polygonOffset: true,
      polygonOffsetFactor: -4,
      polygonOffsetUnits: -4,
    }),
  );
  line.renderOrder = 7;
  const through = new LineSegments2(
    geometry,
    new LineMaterial({ toneMapped: false, transparent: true, depthTest: false, depthWrite: false }),
  );
  through.renderOrder = 7;
  line.raycast = noRaycast;
  through.raycast = noRaycast;
  const group = new THREE.Group();
  group.name = `polyline:${p.id}`;
  group.position.set(center[0], center[1], center[2]);
  group.add(line, through);
  return { group, line, through, points: p.points, closed: p.closed };
}

function dispose(parts: LineParts): void {
  parts.group.removeFromParent();
  parts.line.geometry.dispose();
  parts.line.material.dispose();
  parts.through.material.dispose();
}

export function PolylineLayer(props: { app: AppState }) {
  const { app } = props;
  const polylines = useApp(app, (s) => s.polylines);
  const selection = useApp(app, (s) => s.polylineSelection);
  const lines = useMemo(() => new PolylineLines(), []);
  useEffect(() => () => lines.dispose(), [lines]);
  useLayoutEffect(() => lines.sync(polylines, selection), [lines, polylines, selection]);
  return <primitive object={lines.group} />;
}
