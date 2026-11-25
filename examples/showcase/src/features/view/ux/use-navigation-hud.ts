// What the navigation HUD follows each frame: the cube turned with the
// camera (its CSS transform written straight to the element), and the XYZ
// triad and the scale bar re-rendered only when what they show changes.
// Clicking a cube face, edge or corner snaps the camera; its arrows orbit
// and roll it (#255).
import { type RefObject, useCallback, useEffect, useState } from "react";
import * as THREE from "three";
import type { AppState } from "../../../app/state/app-state";
import type { LengthUnit } from "../../../shared/ux/units";
import { view } from "../../../viewport/ux/view";
import {
  buildViewCubeMatrixCss,
  computeMetricScaleBar,
  projectBasisTriad,
  type ScaleBarReadout,
  type TriadAxisProjection,
  type ViewCubeTarget,
} from "./navigation-hud";
import {
  ORBIT_FINE_STEP_DEG,
  ORBIT_STEP_DEG,
  type OrbitDirection,
  ROLL_STEP_DEG,
  type RollDirection,
} from "./view-steps";

/** The triad as drawn, at half-pixel steps: a new key means a visible change. */
export function triadSummaryKey(axes: readonly TriadAxisProjection[]): string {
  return axes.map((a) => `${a.axis}:${Math.round(a.x * 2)},${Math.round(a.y * 2)}`).join("|");
}

/**
 * Follows the viewport camera every animation frame while mounted: turns
 * `cubeRef`'s element with it, and returns the triad and the scale bar (in
 * `units`), whose state changes only when their drawing would.
 */
export function useNavigationHud(
  units: LengthUnit,
  cubeRef: RefObject<HTMLElement | null>,
): { triad: TriadAxisProjection[]; scaleBar: ScaleBarReadout } {
  const [triad, setTriad] = useState<TriadAxisProjection[]>(() =>
    projectBasisTriad(new THREE.Quaternion(), 20),
  );
  const [scaleBar, setScaleBar] = useState<ScaleBarReadout>({
    label: units === "in" ? "1 in" : "10 mm",
    widthPx: 80,
    valueInUnits: 10,
    mmLength: 10,
  });

  useEffect(() => {
    let raf = 0;
    let lastTriadKey = "";
    let lastScaleKey = "";
    const targetVec = new THREE.Vector3();

    const tick = () => {
      const cam = view.camera;
      const canvas = view.canvas;
      if (cam !== null) {
        if (cubeRef.current !== null) {
          cubeRef.current.style.transform = buildViewCubeMatrixCss(cam.quaternion, -26);
        }
        const nextTriad = projectBasisTriad(cam.quaternion, 20);
        const tKey = triadSummaryKey(nextTriad);
        if (tKey !== lastTriadKey) {
          lastTriadKey = tKey;
          setTriad(nextTriad);
        }
        if (canvas !== null && canvas.clientHeight > 0) {
          if (view.orbit !== null) {
            targetVec.copy(view.orbit.target);
          } else {
            targetVec.set(0, 0, 0);
          }
          const dist = cam.position.distanceTo(targetVec);
          const nextScale = computeMetricScaleBar({
            camera: cam,
            viewportHeightPx: canvas.clientHeight,
            targetDistanceWorld: dist,
            units,
          });
          const sKey = `${nextScale.label}|${Math.round(nextScale.widthPx)}`;
          if (sKey !== lastScaleKey) {
            lastScaleKey = sKey;
            setScaleBar(nextScale);
          }
        }
      }
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [units]);

  return { triad, scaleBar };
}

/** Snaps the camera to a cube target: its named preset, else its direction and up. */
export function snapView(app: AppState, target: ViewCubeTarget): void {
  if (target.preset !== undefined) {
    app.requestCameraPreset(target.preset);
  } else {
    app.requestCameraDirection(target.dir, target.up);
  }
}

/** Orbits the view one step the way `direction` points: a quarter turn, or 15° when `fine` (Shift held). */
export function orbitView(app: AppState, direction: OrbitDirection, fine = false): void {
  app.requestCameraStep({ kind: "orbit", direction, deg: fine ? ORBIT_FINE_STEP_DEG : ORBIT_STEP_DEG });
}

/** Rolls the view a quarter turn about the viewing direction, the picture turning `direction`. */
export function rollView(app: AppState, direction: RollDirection): void {
  app.requestCameraStep({ kind: "roll", direction, deg: ROLL_STEP_DEG });
}

/**
 * A phone-sized touch screen: there the cube box starts folded to its
 * toggle, leaving the small canvas to the fingers.
 */
const compactTouch =
  typeof matchMedia === "function" &&
  matchMedia("(pointer: coarse) and (max-width: 600px), (pointer: coarse) and (max-height: 600px)").matches;

/** Whether the cube box is open (folded on a phone at first), and its toggle. */
export function useViewCubeOpen(): [boolean, () => void] {
  const [open, setOpen] = useState(!compactTouch);
  const toggle = useCallback(() => setOpen((o) => !o), []);
  return [open, toggle];
}
