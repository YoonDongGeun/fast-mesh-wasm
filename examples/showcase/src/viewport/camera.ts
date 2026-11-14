// Camera presets, framing bounds, and Perspective <-> Orthographic projection math (#195).
import type { Scene, SceneLike } from "fast-mesh-wasm";
import * as THREE from "three";
import type { AppState, CameraPreset, CameraProjection, Item } from "../scene/app-state";

export interface SceneCameraBounds {
  center: [number, number, number];
  radius: number;
}

export interface CameraPresetPose {
  position: [number, number, number];
  target: [number, number, number];
  up: [number, number, number];
  distance: number;
  near: number;
  far: number;
  halfHeight: number;
}

const PRESET_AXES: Record<CameraPreset, { dir: [number, number, number]; up: [number, number, number] }> = {
  front: { dir: [0, 0, 1], up: [0, 1, 0] },
  back: { dir: [0, 0, -1], up: [0, 1, 0] },
  left: { dir: [-1, 0, 0], up: [0, 1, 0] },
  right: { dir: [1, 0, 0], up: [0, 1, 0] },
  top: { dir: [0, 1, 0], up: [0, 0, -1] },
  bottom: { dir: [0, -1, 0], up: [0, 0, 1] },
};

/** Computes the combined world-space bounding sphere over all visible meshes in `items`. */
export function computeSceneBounds(
  items: readonly Item[],
  scene: SceneLike,
  app?: AppState,
  hiddenMeshes?: ReadonlyMap<number | object, boolean>,
): SceneCameraBounds {
  const box = new THREE.Box3();
  const pt = new THREE.Vector3();
  const mat = new THREE.Matrix4();
  const minVec = new THREE.Vector3();
  const maxVec = new THREE.Vector3();

  for (const item of items) {
    const isHidden = hiddenMeshes?.get(item.mesh as unknown as object) ?? false;
    if (isHidden) {
      continue;
    }
    try {
      const m = scene.matrix4(item.mesh);
      mat.fromArray(m);
      const b = typeof (scene as Scene).bounds === "function" ? (scene as Scene).bounds(item.mesh) : undefined;
      if (b !== undefined && b !== null) {
        minVec.set(b.min[0], b.min[1], b.min[2]);
        maxVec.set(b.max[0], b.max[1], b.max[2]);
        for (let ix = 0; ix < 2; ix++) {
          for (let iy = 0; iy < 2; iy++) {
            for (let iz = 0; iz < 2; iz++) {
              pt.set(
                ix === 0 ? minVec.x : maxVec.x,
                iy === 0 ? minVec.y : maxVec.y,
                iz === 0 ? minVec.z : maxVec.z,
              ).applyMatrix4(mat);
              box.expandByPoint(pt);
            }
          }
        }
      } else {
        const p = scene.views(item.mesh).positions();
        const t = scene.views(item.mesh).indices();
        let any = false;
        for (let k = 0; k + 2 < t.length; k += 3) {
          if (t[k] !== 0 || t[k + 1] !== 0 || t[k + 2] !== 0) {
            any = true;
            for (let c = 0; c < 3; c++) {
              const v = t[k + c];
              pt.set(p[3 * v], p[3 * v + 1], p[3 * v + 2]).applyMatrix4(mat);
              box.expandByPoint(pt);
            }
          }
        }
        if (!any && item.radius > 0) {
          pt.set(item.center[0], item.center[1], item.center[2]).applyMatrix4(mat);
          box.expandByPoint(pt);
        }
      }
    } catch {
      // Fallback to cached center & radius if mesh views are temporarily unavailable
      pt.set(item.center[0], item.center[1], item.center[2]);
      box.expandByPoint(pt);
    }
  }

  void app;
  if (box.isEmpty()) {
    return { center: [0, 0, 0], radius: 3 };
  }
  const center = new THREE.Vector3();
  box.getCenter(center);
  const radius = Math.max(box.min.distanceTo(box.max) / 2, 0.25);
  return { center: [center.x, center.y, center.z], radius };
}

/** Computes camera pose framing `bounds` from one of the 6 standard axis presets. */
export function presetPose(
  preset: CameraPreset,
  bounds: SceneCameraBounds,
  fovDeg = 45,
): CameraPresetPose {
  const { dir, up } = PRESET_AXES[preset];
  const r = Math.max(bounds.radius, 0.25);
  const halfFov = THREE.MathUtils.degToRad(fovDeg / 2);
  const distance = (r * 2.4) / Math.sin(halfFov);
  const halfHeight = Math.tan(halfFov) * distance;
  return {
    position: [
      bounds.center[0] + dir[0] * distance,
      bounds.center[1] + dir[1] * distance,
      bounds.center[2] + dir[2] * distance,
    ],
    target: [...bounds.center],
    up: [...up],
    distance,
    near: Math.max(distance / 1000, 1e-4),
    far: Math.max(distance * 100, 100),
    halfHeight,
  };
}

/** Computes OrthographicCamera half-extents matching a PerspectiveCamera at `distance`. */
export function perspectiveToOrthoHalfExtents(
  fovDeg: number,
  distance: number,
  aspect: number,
): { halfHeight: number; halfWidth: number } {
  const d = Math.max(distance, 1e-4);
  const safeAspect = Number.isFinite(aspect) && aspect > 0 ? aspect : 1;
  const halfHeight = Math.tan(THREE.MathUtils.degToRad(fovDeg / 2)) * d;
  return { halfHeight, halfWidth: halfHeight * safeAspect };
}

/** Computes equivalent PerspectiveCamera distance matching an OrthographicCamera frustum height and zoom. */
export function orthoToPerspectiveDistance(
  orthoTop: number,
  orthoBottom: number,
  zoom: number,
  fovDeg: number,
): number {
  const effectiveHalfH = Math.abs(orthoTop - orthoBottom) / (2 * Math.max(zoom, 1e-4));
  const tanHalf = Math.tan(THREE.MathUtils.degToRad(fovDeg / 2));
  return Math.max(effectiveHalfH / Math.max(tanHalf, 1e-4), 1e-2);
}

/** Applies a camera pose + projection settings cleanly onto either PerspectiveCamera or OrthographicCamera. */
export function applyCameraPose(
  camera: THREE.Camera,
  pose: CameraPresetPose,
  aspect: number,
  projection: CameraProjection,
): void {
  camera.position.set(pose.position[0], pose.position[1], pose.position[2]);
  camera.up.set(pose.up[0], pose.up[1], pose.up[2]);
  if (projection === "perspective" && camera instanceof THREE.PerspectiveCamera) {
    camera.near = pose.near;
    camera.far = pose.far;
    camera.aspect = Number.isFinite(aspect) && aspect > 0 ? aspect : 1;
    camera.updateProjectionMatrix();
  } else if (camera instanceof THREE.OrthographicCamera) {
    const safeAspect = Number.isFinite(aspect) && aspect > 0 ? aspect : 1;
    const h = pose.halfHeight;
    const w = h * safeAspect;
    camera.left = -w;
    camera.right = w;
    camera.top = h;
    camera.bottom = -h;
    camera.zoom = 1;
    camera.near = -Math.max(pose.distance * 50, 500);
    camera.far = Math.max(pose.distance * 50, 500);
    camera.updateProjectionMatrix();
  }
}
