// Screen-space face selection (#233): pointer pixels to normalized device
// coordinates, the box and lasso shapes `repair.selectFacesScreen` tests
// triangles against, and the camera it projects them with.
import type { ScreenSelectOptions, ScreenSelectionShape, Vec3 } from "fast-mesh-wasm";
import * as THREE from "three";

/** A box dragged less than this far (in pixels) counts as a click and gets a small pick pad. */
export const CLICK_DRAG_THRESHOLD_PX = 3;

type PixelRect = Pick<DOMRect, "width" | "height">;

/** Converts client pixel coordinates inside `rect` to normalized device coordinates `[-1, 1]`. */
export function clientPointToNdc(
  clientX: number,
  clientY: number,
  rect: Pick<DOMRect, "left" | "top" | "width" | "height">,
): [number, number] {
  const width = Math.max(1, rect.width);
  const height = Math.max(1, rect.height);
  const ndcX = ((clientX - rect.left) / width) * 2 - 1;
  const ndcY = 1 - ((clientY - rect.top) / height) * 2;
  return [ndcX, ndcY];
}

/** Converts canvas-local pixels (origin top left) to normalized device coordinates. */
export function localPointToNdc(x: number, y: number, width: number, height: number): [number, number] {
  const w = Math.max(1, width);
  const h = Math.max(1, height);
  return [(x / w) * 2 - 1, 1 - (y / h) * 2];
}

/** The client point in canvas-local pixels, clamped into the canvas. */
export function clientToLocalPixels(
  clientX: number,
  clientY: number,
  rect: Pick<DOMRect, "left" | "top" | "width" | "height">,
): [number, number] {
  return [
    Math.max(0, Math.min(rect.width, clientX - rect.left)),
    Math.max(0, Math.min(rect.height, clientY - rect.top)),
  ];
}

/** The world-space pick ray through a client point. */
export function rayFromClient(
  camera: THREE.Camera,
  clientX: number,
  clientY: number,
  rect: Pick<DOMRect, "left" | "top" | "width" | "height">,
): { origin: Vec3; direction: Vec3 } {
  const [nx, ny] = clientPointToNdc(clientX, clientY, rect);
  const raycaster = new THREE.Raycaster();
  raycaster.setFromCamera(new THREE.Vector2(nx, ny), camera);
  return {
    origin: [raycaster.ray.origin.x, raycaster.ray.origin.y, raycaster.ray.origin.z],
    direction: [raycaster.ray.direction.x, raycaster.ray.direction.y, raycaster.ray.direction.z],
  };
}

/** The NDC rectangle spanned by two canvas-local corners, grown by `padPx` pixels on every side. */
export function rectShapeFromPixels(
  x0: number,
  y0: number,
  x1: number,
  y1: number,
  rect: PixelRect,
  padPx: number,
): ScreenSelectionShape {
  const a = localPointToNdc(Math.min(x0, x1) - padPx, Math.min(y0, y1) - padPx, rect.width, rect.height);
  const b = localPointToNdc(Math.max(x0, x1) + padPx, Math.max(y0, y1) + padPx, rect.width, rect.height);
  return {
    kind: "rect",
    minNdc: [Math.min(a[0], b[0]), Math.min(a[1], b[1])],
    maxNdc: [Math.max(a[0], b[0]), Math.max(a[1], b[1])],
  };
}

/** A finished box drag's shape: a click (under the drag threshold) picks a 4-pixel pad around the point. */
export function boxShapeFromPixels(x0: number, y0: number, x1: number, y1: number, rect: PixelRect): ScreenSelectionShape {
  const pad = Math.hypot(x1 - x0, y1 - y0) < CLICK_DRAG_THRESHOLD_PX ? 4 : 0;
  return rectShapeFromPixels(x0, y0, x1, y1, rect, pad);
}

/** A finished lasso's shape: its polygon, or a 4-pixel pad around the first point when it has fewer than three. */
export function lassoShapeFromPixels(
  points: readonly (readonly [number, number])[],
  rect: PixelRect,
): ScreenSelectionShape {
  return points.length >= 3
    ? {
        kind: "lasso",
        polygonNdc: points.map(([px, py]) => localPointToNdc(px, py, rect.width, rect.height)),
      }
    : rectShapeFromPixels(
        points[0]?.[0] ?? 0,
        points[0]?.[1] ?? 0,
        points[0]?.[0] ?? 0,
        points[0]?.[1] ?? 0,
        rect,
        4,
      );
}

/** Extracts `ScreenSelectOptions` from a Three.js camera for `repair.selectFacesScreen`. */
export function buildScreenSelectOptions(
  camera: THREE.Camera,
  visibleOnly: boolean,
  rejectBackfaces: boolean,
  includeTouching = true,
): ScreenSelectOptions {
  camera.updateMatrixWorld();
  camera.matrixWorldInverse.copy(camera.matrixWorld).invert();
  const viewProjMatrix = new THREE.Matrix4().multiplyMatrices(
    camera.projectionMatrix,
    camera.matrixWorldInverse,
  );
  const origin = new THREE.Vector3();
  const forward = new THREE.Vector3();
  camera.getWorldPosition(origin);
  camera.getWorldDirection(forward);
  if (forward.lengthSq() === 0) {
    forward.set(0, 0, -1);
  }
  return {
    viewProj: viewProjMatrix.toArray(),
    cameraOrigin: [origin.x, origin.y, origin.z],
    cameraForward: [forward.x, forward.y, forward.z],
    isOrthographic: Boolean((camera as THREE.OrthographicCamera).isOrthographicCamera),
    visibleOnly,
    rejectBackfaces,
    includeTouching,
  };
}
