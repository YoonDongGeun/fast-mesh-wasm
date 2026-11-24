// The viewport's camera: a trackball (free flipping in every direction, no
// poles) over a perspective or orthographic camera, switched in place keeping
// the pose and framing, and the requests that move it: reset view, presets,
// ViewCube directions, Mesh Healer hotspots and the focus on a mesh or box.
import { useFrame, useThree } from "@react-three/fiber";
import type { MeshHandle } from "fast-mesh-wasm";
import { useScene } from "fast-mesh-wasm/react";
import { type RefObject, useEffect, useRef, useState } from "react";
import * as THREE from "three";
import { TrackballControls } from "three/addons/controls/TrackballControls.js";
import type { FocusRequest } from "../../app/App";
import { type AppState, type CameraProjection, type Item, useApp } from "../../app/state/app-state";
import {
  applyCameraPose,
  computeSceneBounds,
  directionPose,
  orthoToPerspectiveDistance,
  perspectiveToOrthoHalfExtents,
  presetPose,
} from "../../features/view/ux/camera";
import { watchFingers } from "./touch";
import { view } from "./view";

/**
 * The trackball camera in `cameraProjection`, published as `view.camera`,
 * `view.canvas` and `view.orbit`. `orbit` is null until the trackball is up;
 * `orbitTargetRef` keeps its target across a camera switch, for the requests.
 */
export function useOrbitCamera(cameraProjection: CameraProjection): {
  orbit: TrackballControls | null;
  orbitTargetRef: RefObject<THREE.Vector3>;
} {
  const camera = useThree((s) => s.camera);
  const setThree = useThree((s) => s.set);
  const gl = useThree((s) => s.gl);
  const size = useThree((s) => s.size);
  const [orbit, setOrbit] = useState<TrackballControls | null>(null);
  const perspCamRef = useRef<THREE.PerspectiveCamera | null>(null);
  const orthoCamRef = useRef<THREE.OrthographicCamera | null>(null);
  const orbitTargetRef = useRef(new THREE.Vector3(-0.6, 0.9, 0));

  if (camera instanceof THREE.PerspectiveCamera) {
    perspCamRef.current = camera;
  } else if (camera instanceof THREE.OrthographicCamera) {
    orthoCamRef.current = camera;
  }

  // Switch seamlessly between PerspectiveCamera and OrthographicCamera while keeping pose & framing.
  useEffect(() => {
    const aspect = size.height > 0 ? size.width / size.height : 1;
    const target = orbit?.target ?? orbitTargetRef.current;
    orbitTargetRef.current.copy(target);
    if (cameraProjection === "orthographic" && camera instanceof THREE.PerspectiveCamera) {
      const dist = Math.max(camera.position.distanceTo(target), 1e-3);
      const { halfHeight, halfWidth } = perspectiveToOrthoHalfExtents(camera.fov, dist, aspect);
      const ortho =
        orthoCamRef.current ??
        new THREE.OrthographicCamera(
          -halfWidth,
          halfWidth,
          halfHeight,
          -halfHeight,
          -Math.max(dist * 50, 500),
          Math.max(dist * 50, 500),
        );
      ortho.left = -halfWidth;
      ortho.right = halfWidth;
      ortho.top = halfHeight;
      ortho.bottom = -halfHeight;
      ortho.zoom = 1;
      ortho.near = -Math.max(dist * 50, 500);
      ortho.far = Math.max(dist * 50, 500);
      ortho.position.copy(camera.position);
      ortho.up.copy(camera.up);
      ortho.lookAt(target);
      ortho.updateProjectionMatrix();
      orthoCamRef.current = ortho;
      setThree({ camera: ortho });
    } else if (cameraProjection === "perspective" && camera instanceof THREE.OrthographicCamera) {
      const persp = perspCamRef.current ?? new THREE.PerspectiveCamera(45, aspect, 0.01, 5000);
      const dist = orthoToPerspectiveDistance(camera.top, camera.bottom, camera.zoom, persp.fov);
      const dir = camera.position.clone().sub(target);
      if (dir.lengthSq() === 0) {
        dir.set(0, 0, 1);
      }
      dir.normalize();
      persp.position.copy(target).addScaledVector(dir, dist);
      persp.up.copy(camera.up);
      persp.aspect = aspect;
      persp.near = Math.max(dist / 1000, 1e-4);
      persp.far = Math.max(dist * 100, 100);
      persp.lookAt(target);
      persp.updateProjectionMatrix();
      perspCamRef.current = persp;
      setThree({ camera: persp });
    }
  }, [cameraProjection, camera, orbit, setThree, size.width, size.height]);

  useEffect(() => {
    view.camera = camera;
    view.canvas = gl.domElement;
    const unwatch = watchFingers(gl.domElement); // before the trackball: see touch.ts
    const o = new TrackballControls(camera, gl.domElement);
    o.target.copy(orbitTargetRef.current);
    // No inertia: the camera stops where the pointer lifts (deterministic).
    o.staticMoving = true;
    o.update();
    view.orbit = o;
    setOrbit(o);
    return () => {
      orbitTargetRef.current.copy(o.target);
      o.dispose();
      unwatch();
      setOrbit((c) => (c === o ? null : c));
      if (view.camera === camera) {
        view.camera = null;
        view.canvas = null;
      }
      if (view.orbit === o) {
        view.orbit = null;
      }
    };
  }, [camera, gl]);

  // The trackball caches the canvas's box; a rotated phone or a resized panel changes it.
  useEffect(() => {
    orbit?.handleResize();
    if (camera instanceof THREE.OrthographicCamera && size.width > 0 && size.height > 0) {
      const halfH = Math.abs(camera.top - camera.bottom) / 2;
      const halfW = halfH * (size.width / size.height);
      camera.left = -halfW;
      camera.right = halfW;
      camera.updateProjectionMatrix();
    }
  }, [orbit, size, camera]);

  useFrame(() => orbit?.update());

  return { orbit, orbitTargetRef };
}

/**
 * Moves the camera on each new request: `resetView` bumped, a camera preset,
 * a ViewCube direction, a Mesh Healer hotspot, or a `focus` that points it at
 * a mesh (or frames a world box). A new list of items is not a new request.
 */
export function useCameraRequests(params: {
  app: AppState;
  items: readonly Item[];
  orbit: TrackballControls | null;
  orbitTargetRef: RefObject<THREE.Vector3>;
  cameraProjection: CameraProjection;
  focus: FocusRequest | null;
  resetView: number;
  radiusOf: (mesh: MeshHandle) => number;
}): void {
  const { app, items, orbit, orbitTargetRef, cameraProjection, focus, resetView, radiusOf } = params;
  const store = useScene();
  const camera = useThree((s) => s.camera);
  const size = useThree((s) => s.size);
  const cameraPresetRequest = useApp(app, (s) => s.cameraPresetRequest);
  const cameraDirectionRequest = useApp(app, (s) => s.cameraDirectionRequest);
  const healerFocusRequest = useApp(app, (s) => s.healerFocusRequest);
  // Read when a focus/preset request arrives: a new list of items is not a new request.
  const radiusRef = useRef(radiusOf);
  radiusRef.current = radiusOf;
  const itemsRef = useRef(items);
  itemsRef.current = items;
  const lastResetViewRef = useRef(0);
  const lastFocusRef = useRef<FocusRequest | null>(null);
  const lastPresetSeqRef = useRef(0);
  const lastDirSeqRef = useRef(0);
  const lastHealerFocusSeqRef = useRef(0);

  // Reset view: the default pose (and an upright camera: the trackball
  // may have rolled it).
  useEffect(() => {
    if (orbit === null || resetView === 0 || resetView === lastResetViewRef.current) {
      return;
    }
    lastResetViewRef.current = resetView;
    camera.position.set(-0.6, 3.4, 8.2);
    camera.up.set(0, 1, 0);
    orbitTargetRef.current.set(-0.6, 0.9, 0);
    orbit.target.set(-0.6, 0.9, 0);
    if (camera instanceof THREE.PerspectiveCamera) {
      camera.near = 0.01;
      camera.far = 5000;
      camera.updateProjectionMatrix();
    } else if (camera instanceof THREE.OrthographicCamera) {
      const dist = camera.position.distanceTo(orbit.target);
      const aspect = size.height > 0 ? size.width / size.height : 1;
      const { halfHeight, halfWidth } = perspectiveToOrthoHalfExtents(45, dist, aspect);
      camera.left = -halfWidth;
      camera.right = halfWidth;
      camera.top = halfHeight;
      camera.bottom = -halfHeight;
      camera.zoom = 1;
      camera.near = -Math.max(dist * 50, 500);
      camera.far = Math.max(dist * 50, 500);
      camera.updateProjectionMatrix();
    }
    orbit.update();
  }, [orbit, resetView, camera, size.width, size.height]);

  // Camera presets: frame the scene bounds along Front / Back / Left / Right / Top / Bottom.
  useEffect(() => {
    if (orbit === null || cameraPresetRequest === null || cameraPresetRequest.n === lastPresetSeqRef.current) {
      return;
    }
    lastPresetSeqRef.current = cameraPresetRequest.n;
    const bounds = computeSceneBounds(itemsRef.current, store.scene, app);
    const pose = presetPose(cameraPresetRequest.preset, bounds);
    const aspect = size.height > 0 ? size.width / size.height : 1;
    applyCameraPose(camera, pose, aspect, cameraProjection);
    orbitTargetRef.current.set(pose.target[0], pose.target[1], pose.target[2]);
    orbit.target.set(pose.target[0], pose.target[1], pose.target[2]);
    orbit.update();
  }, [orbit, cameraPresetRequest, cameraProjection, store, app, camera, size.width, size.height]);

  // Arbitrary ViewCube edge/corner direction snapping (#232).
  useEffect(() => {
    if (orbit === null || cameraDirectionRequest === null || cameraDirectionRequest.n === lastDirSeqRef.current) {
      return;
    }
    lastDirSeqRef.current = cameraDirectionRequest.n;
    const bounds = computeSceneBounds(itemsRef.current, store.scene, app);
    const pose = directionPose(cameraDirectionRequest.dir, cameraDirectionRequest.up, bounds);
    const aspect = size.height > 0 ? size.width / size.height : 1;
    applyCameraPose(camera, pose, aspect, cameraProjection);
    orbitTargetRef.current.set(pose.target[0], pose.target[1], pose.target[2]);
    orbit.target.set(pose.target[0], pose.target[1], pose.target[2]);
    orbit.update();
  }, [orbit, cameraDirectionRequest, cameraProjection, store, app, camera, size.width, size.height]);

  // Mesh Healer defect-hotspot framing (#235).
  useEffect(() => {
    if (orbit === null || healerFocusRequest === null || healerFocusRequest.n === lastHealerFocusSeqRef.current) {
      return;
    }
    lastHealerFocusSeqRef.current = healerFocusRequest.n;
    const center = new THREE.Vector3(
      healerFocusRequest.center[0],
      healerFocusRequest.center[1],
      healerFocusRequest.center[2],
    );
    try {
      const m = store.scene.matrix4(healerFocusRequest.mesh);
      center.applyMatrix4(new THREE.Matrix4().fromArray(m));
    } catch {
      // Fall back to supplied coordinates if matrix is unavailable.
    }
    const radius = Math.max(healerFocusRequest.radius, 0.05);
    const dir = camera.position.clone().sub(orbit.target);
    if (dir.lengthSq() === 0) {
      dir.set(0, 0.35, 1);
    }
    const fov = camera instanceof THREE.PerspectiveCamera ? camera.fov : 45;
    const distance = (radius * 1.65) / Math.sin(THREE.MathUtils.degToRad(fov) / 2);
    orbitTargetRef.current.copy(center);
    orbit.target.copy(center);
    camera.position.copy(center).addScaledVector(dir.normalize(), distance);
    if (camera instanceof THREE.PerspectiveCamera) {
      camera.near = Math.max(distance / 1000, 1e-4);
      camera.far = Math.max(distance * 100, 100);
      camera.updateProjectionMatrix();
    } else if (camera instanceof THREE.OrthographicCamera) {
      const aspect = size.height > 0 ? size.width / size.height : 1;
      const { halfHeight, halfWidth } = perspectiveToOrthoHalfExtents(fov, distance, aspect);
      camera.left = -halfWidth;
      camera.right = halfWidth;
      camera.top = halfHeight;
      camera.bottom = -halfHeight;
      camera.zoom = 1;
      camera.near = -Math.max(distance * 50, 500);
      camera.far = Math.max(distance * 50, 500);
      camera.updateProjectionMatrix();
    }
    orbit.update();
  }, [orbit, healerFocusRequest, store, camera, size.width, size.height]);

  // Focus: the orbit target to the mesh's centre (its origin, where the
  // gizmo sits), the camera back along the requested (or current)
  // direction. A focus box frames that world box instead of the mesh.
  useEffect(() => {
    if (orbit === null || focus === null || focus === lastFocusRef.current) {
      return;
    }
    lastFocusRef.current = focus;
    let center: THREE.Vector3;
    let radius: number;
    if (focus.box !== undefined) {
      center = new THREE.Vector3(focus.box.center[0], focus.box.center[1], focus.box.center[2]);
      radius = Math.max(focus.box.radius, 1e-3);
    } else {
      let m: Float64Array;
      try {
        m = store.scene.matrix4(focus.mesh);
      } catch {
        return; // removed meanwhile
      }
      center = new THREE.Vector3(m[12], m[13], m[14]);
      radius = Math.max(radiusRef.current(focus.mesh), 1e-3);
    }
    const dir =
      focus.dir === undefined
        ? camera.position.clone().sub(orbit.target)
        : new THREE.Vector3(focus.dir[0], focus.dir[1], focus.dir[2]);
    if (dir.lengthSq() === 0) {
      dir.set(0, 0.4, 1);
    }
    const fov = camera instanceof THREE.PerspectiveCamera ? camera.fov : 45;
    // Frame individual meshes slightly tighter so shorter ribbon-hosted viewports keep brush drags local.
    const fitScale = focus.box !== undefined ? 1.15 : 1.0;
    const distance = (radius * fitScale) / Math.sin(THREE.MathUtils.degToRad(fov) / 2);
    orbitTargetRef.current.copy(center);
    orbit.target.copy(center);
    camera.position.copy(center).addScaledVector(dir.normalize(), distance);
    if (camera instanceof THREE.PerspectiveCamera) {
      camera.near = Math.max(distance / 1000, 1e-4);
      camera.far = Math.max(distance * 100, 100);
      camera.updateProjectionMatrix();
    } else if (camera instanceof THREE.OrthographicCamera) {
      const aspect = size.height > 0 ? size.width / size.height : 1;
      const { halfHeight, halfWidth } = perspectiveToOrthoHalfExtents(fov, distance, aspect);
      camera.left = -halfWidth;
      camera.right = halfWidth;
      camera.top = halfHeight;
      camera.bottom = -halfHeight;
      camera.zoom = 1;
      camera.near = -Math.max(distance * 50, 500);
      camera.far = Math.max(distance * 50, 500);
      camera.updateProjectionMatrix();
    }
    orbit.update();
  }, [orbit, focus, store, camera, size.width, size.height]);
}
