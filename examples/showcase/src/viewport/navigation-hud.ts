// Pure math and layout helpers for the ViewCube, XYZ basis triad, and live scale bar (#232).
import * as THREE from "three";
import type { LengthUnit } from "../panels/units";
import type { CameraPreset } from "../scene/app-state";

export interface ViewCubeTarget {
  id: string;
  label: string;
  title: string;
  kind: "face" | "edge" | "corner";
  preset?: CameraPreset;
  dir: readonly [number, number, number];
  up: readonly [number, number, number];
}

function norm3(x: number, y: number, z: number): [number, number, number] {
  const len = Math.hypot(x, y, z) || 1;
  return [x / len, y / len, z / len];
}

function upForDirection(dx: number, dy: number, dz: number): [number, number, number] {
  const fwd = new THREE.Vector3(dx, dy, dz).normalize();
  const refUp = Math.abs(fwd.y) > 0.98 ? new THREE.Vector3(0, 0, fwd.y > 0 ? -1 : 1) : new THREE.Vector3(0, 1, 0);
  const right = new THREE.Vector3().crossVectors(refUp, fwd).normalize();
  const up = new THREE.Vector3().crossVectors(fwd, right).normalize();
  return [up.x, up.y, up.z];
}

export const VIEW_CUBE_FACES: readonly ViewCubeTarget[] = [
  { id: "front", label: "Front", title: "Front (+Z) view", kind: "face", preset: "front", dir: [0, 0, 1], up: [0, 1, 0] },
  { id: "back", label: "Back", title: "Back (-Z) view", kind: "face", preset: "back", dir: [0, 0, -1], up: [0, 1, 0] },
  { id: "left", label: "Left", title: "Left (-X) view", kind: "face", preset: "left", dir: [-1, 0, 0], up: [0, 1, 0] },
  { id: "right", label: "Right", title: "Right (+X) view", kind: "face", preset: "right", dir: [1, 0, 0], up: [0, 1, 0] },
  { id: "top", label: "Top", title: "Top (+Y) view", kind: "face", preset: "top", dir: [0, 1, 0], up: [0, 0, -1] },
  { id: "bottom", label: "Bottom", title: "Bottom (-Y) view", kind: "face", preset: "bottom", dir: [0, -1, 0], up: [0, 0, 1] },
];

export const VIEW_CUBE_EDGES: readonly ViewCubeTarget[] = [
  { id: "top-front", label: "Top-Front", title: "Top-Front isometric edge", kind: "edge", dir: norm3(0, 1, 1), up: upForDirection(0, 1, 1) },
  { id: "top-back", label: "Top-Back", title: "Top-Back isometric edge", kind: "edge", dir: norm3(0, 1, -1), up: upForDirection(0, 1, -1) },
  { id: "top-left", label: "Top-Left", title: "Top-Left isometric edge", kind: "edge", dir: norm3(-1, 1, 0), up: upForDirection(-1, 1, 0) },
  { id: "top-right", label: "Top-Right", title: "Top-Right isometric edge", kind: "edge", dir: norm3(1, 1, 0), up: upForDirection(1, 1, 0) },
  { id: "bottom-front", label: "Bottom-Front", title: "Bottom-Front edge", kind: "edge", dir: norm3(0, -1, 1), up: upForDirection(0, -1, 1) },
  { id: "bottom-back", label: "Bottom-Back", title: "Bottom-Back edge", kind: "edge", dir: norm3(0, -1, -1), up: upForDirection(0, -1, -1) },
  { id: "bottom-left", label: "Bottom-Left", title: "Bottom-Left edge", kind: "edge", dir: norm3(-1, -1, 0), up: upForDirection(-1, -1, 0) },
  { id: "bottom-right", label: "Bottom-Right", title: "Bottom-Right edge", kind: "edge", dir: norm3(1, -1, 0), up: upForDirection(1, -1, 0) },
  { id: "front-left", label: "Front-Left", title: "Front-Left edge", kind: "edge", dir: norm3(-1, 0, 1), up: upForDirection(-1, 0, 1) },
  { id: "front-right", label: "Front-Right", title: "Front-Right edge", kind: "edge", dir: norm3(1, 0, 1), up: upForDirection(1, 0, 1) },
  { id: "back-left", label: "Back-Left", title: "Back-Left edge", kind: "edge", dir: norm3(-1, 0, -1), up: upForDirection(-1, 0, -1) },
  { id: "back-right", label: "Back-Right", title: "Back-Right edge", kind: "edge", dir: norm3(1, 0, -1), up: upForDirection(1, 0, -1) },
];

export const VIEW_CUBE_CORNERS: readonly ViewCubeTarget[] = [
  { id: "top-front-right", label: "TFR", title: "Iso Top-Front-Right", kind: "corner", dir: norm3(1, 1, 1), up: upForDirection(1, 1, 1) },
  { id: "top-front-left", label: "TFL", title: "Iso Top-Front-Left", kind: "corner", dir: norm3(-1, 1, 1), up: upForDirection(-1, 1, 1) },
  { id: "top-back-right", label: "TBR", title: "Iso Top-Back-Right", kind: "corner", dir: norm3(1, 1, -1), up: upForDirection(1, 1, -1) },
  { id: "top-back-left", label: "TBL", title: "Iso Top-Back-Left", kind: "corner", dir: norm3(-1, 1, -1), up: upForDirection(-1, 1, -1) },
  { id: "bottom-front-right", label: "BFR", title: "Iso Bottom-Front-Right", kind: "corner", dir: norm3(1, -1, 1), up: upForDirection(1, -1, 1) },
  { id: "bottom-front-left", label: "BFL", title: "Iso Bottom-Front-Left", kind: "corner", dir: norm3(-1, -1, 1), up: upForDirection(-1, -1, 1) },
  { id: "bottom-back-right", label: "BBR", title: "Iso Bottom-Back-Right", kind: "corner", dir: norm3(1, -1, -1), up: upForDirection(1, -1, -1) },
  { id: "bottom-back-left", label: "BBL", title: "Iso Bottom-Back-Left", kind: "corner", dir: norm3(-1, -1, -1), up: upForDirection(-1, -1, -1) },
];

/**
 * Computes a CSS `matrix3d(...)` string transforming a CSS-3D cube (+X right, +Y down, +Z viewer)
 * to match Three.js `cameraQuaternion` cleanly.
 */
export function buildViewCubeMatrix4(cameraQuaternion: THREE.Quaternion): number[] {
  const qInv = cameraQuaternion.clone().invert();
  const rot = new THREE.Matrix4().makeRotationFromQuaternion(qInv);
  // Conjugate by S_flipY = diag(1, -1, 1, 1): flips signs of elements involving row 1 XOR col 1.
  const e = [...rot.elements];
  e[1] = -e[1];
  e[4] = -e[4];
  e[6] = -e[6];
  e[9] = -e[9];
  return e;
}

export function buildViewCubeMatrixCss(cameraQuaternion: THREE.Quaternion, translateZPx = -32): string {
  const e = buildViewCubeMatrix4(cameraQuaternion);
  return `translateZ(${translateZPx}px) matrix3d(${e.map((v) => v.toFixed(6)).join(",")})`;
}

export interface TriadAxisProjection {
  axis: "X" | "Y" | "Z";
  /** Screen horizontal offset from center in CSS px (+ right). */
  x: number;
  /** Screen vertical offset from center in CSS px (+ down). */
  y: number;
  /** View-space depth (+ toward viewer, used for back-to-front SVG painter order). */
  z: number;
  color: string;
}

/** Projects world +X, +Y, +Z basis vectors into 2D HUD coordinates sorted back-to-front. */
export function projectBasisTriad(cameraQuaternion: THREE.Quaternion, armPx = 22): TriadAxisProjection[] {
  const qInv = cameraQuaternion.clone().invert();
  const axes: { axis: "X" | "Y" | "Z"; vec: THREE.Vector3; color: string }[] = [
    { axis: "X", vec: new THREE.Vector3(1, 0, 0), color: "#f87171" },
    { axis: "Y", vec: new THREE.Vector3(0, 1, 0), color: "#4ade80" },
    { axis: "Z", vec: new THREE.Vector3(0, 0, 1), color: "#60a5fa" },
  ];
  return axes
    .map(({ axis, vec, color }) => {
      const v = vec.clone().applyQuaternion(qInv);
      return {
        axis,
        x: v.x * armPx,
        y: -v.y * armPx,
        z: v.z,
        color,
      };
    })
    .sort((a, b) => a.z - b.z);
}

export interface ScaleBarReadout {
  label: string;
  widthPx: number;
  valueInUnits: number;
  mmLength: number;
}

function niceStep(target: number): number {
  if (!Number.isFinite(target) || target <= 0) {
    return 1;
  }
  const exp = Math.floor(Math.log10(target));
  const pow = Math.pow(10, exp);
  const norm = target / pow;
  const step = norm <= 1.5 ? 1 : norm <= 3.5 ? 2 : norm <= 7.5 ? 5 : 10;
  return step * pow;
}

function formatScaleNumber(value: number): string {
  if (Math.abs(value) >= 100) {
    return String(Math.round(value));
  }
  return String(Number(value.toPrecision(3)));
}

/**
 * Computes a clean `1 / 2 / 5 * 10^n` metric/imperial scale bar matching live camera zoom/distance (#232).
 */
export function computeMetricScaleBar(params: {
  camera: THREE.Camera;
  viewportHeightPx: number;
  targetDistanceWorld: number;
  units: LengthUnit;
  targetBarPx?: number;
}): ScaleBarReadout {
  const { camera, viewportHeightPx, targetDistanceWorld, units, targetBarPx = 96 } = params;
  const hPx = Math.max(viewportHeightPx, 1);
  let worldHeight = 10;
  if (camera instanceof THREE.OrthographicCamera) {
    worldHeight = Math.abs(camera.top - camera.bottom) / Math.max(camera.zoom, 1e-4);
  } else {
    const fov = camera instanceof THREE.PerspectiveCamera ? camera.fov : 45;
    const d = Math.max(targetDistanceWorld, 1e-4);
    worldHeight = 2 * Math.tan(THREE.MathUtils.degToRad(fov / 2)) * d;
  }
  const worldPerPx = Math.max(worldHeight / hPx, 1e-9);
  const desiredMm = worldPerPx * targetBarPx;

  if (units === "in") {
    const desiredIn = desiredMm / 25.4;
    const stepIn = niceStep(desiredIn);
    const mmLength = stepIn * 25.4;
    const widthPx = Math.max(24, Math.min(180, mmLength / worldPerPx));
    return {
      label: `${formatScaleNumber(stepIn)} in`,
      widthPx,
      valueInUnits: stepIn,
      mmLength,
    };
  }

  const stepMm = niceStep(desiredMm);
  const widthPx = Math.max(24, Math.min(180, stepMm / worldPerPx));
  if (stepMm >= 1000) {
    const m = stepMm / 1000;
    return {
      label: `${formatScaleNumber(m)} m`,
      widthPx,
      valueInUnits: stepMm,
      mmLength: stepMm,
    };
  }
  return {
    label: `${formatScaleNumber(stepMm)} mm`,
    widthPx,
    valueInUnits: stepMm,
    mmLength: stepMm,
  };
}
