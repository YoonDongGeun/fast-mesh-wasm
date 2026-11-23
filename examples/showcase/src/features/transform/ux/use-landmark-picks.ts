// The landmark picks' viewport side (#253): while the align panel picks
// landmark pairs, a tap on the canvas (not a drag, not a pinch) raycasts the
// engine's BVH for the mesh the next point belongs on (the moving mesh, then
// the fixed one), looking past the others, and adds the hit in that mesh's
// local frame. The canvas listeners attach only while picking.
import { useThree } from "@react-three/fiber";
import type { MeshHandle, RayHit, Scene } from "fast-mesh-wasm";
import type { SceneStore } from "fast-mesh-wasm/store";
import type { WorkerScene } from "fast-mesh-wasm/worker";
import { useEffect } from "react";
import { type AppState, useApp } from "../../../app/state/app-state";
import { localHitPoint, rayThroughPointer, raycastVisibleSurface } from "../../../viewport/ux/surface-raycast";
import { tapSlop } from "../../../viewport/ux/touch";
import { view } from "../../../viewport/ux/view";
import { addLandmark, type LandmarkSet, nextLandmarkSide } from "./align-tools";

/** The mesh the next landmark click lands on. */
export function landmarkTarget(set: LandmarkSet): MeshHandle {
  return nextLandmarkSide(set) === "moving" ? set.moving : set.fixed;
}

/**
 * Lands an engine hit on the landmark pairs: adds its local point when it is
 * on the mesh the next point belongs on. Any other hit, or a hit while not
 * picking, does nothing.
 */
export function landLandmarkHit(
  app: AppState,
  store: SceneStore,
  hit: Pick<RayHit, "mesh" | "triangle" | "barycentric">,
): void {
  const set = app.getSnapshot().alignTool.landmarks;
  if (set === null || !set.picking || hit.mesh !== landmarkTarget(set)) {
    return;
  }
  let local: readonly [number, number, number];
  try {
    local = localHitPoint(store.scene as Scene | WorkerScene, hit);
  } catch {
    return; // removed meanwhile
  }
  app.setAlignTool({ landmarks: addLandmark(set, local) });
}

/** Takes the canvas's taps while landmark pairs are being picked (inside the Canvas). */
export function useLandmarkPickPointer(app: AppState, store: SceneStore): void {
  const armed = useApp(app, (s) => s.alignTool.landmarks?.picking === true);
  const { camera, gl, raycaster } = useThree();

  useEffect(() => {
    if (!armed) {
      return;
    }
    const dom = gl.domElement;
    let down: { x: number; y: number; id: number } | null = null;
    let busy = false;

    const onDown = (e: PointerEvent) => {
      down = e.button === 0 && !view.multiTouch ? { x: e.clientX, y: e.clientY, id: e.pointerId } : null;
    };
    const onUp = (e: PointerEvent) => {
      const at = down;
      down = null;
      if (at === null || at.id !== e.pointerId || e.button !== 0 || view.dragging || view.multiTouch || busy) {
        return;
      }
      if (Math.hypot(e.clientX - at.x, e.clientY - at.y) > tapSlop(e.pointerType, 4)) {
        return;
      }
      const set = app.getSnapshot().alignTool.landmarks;
      if (set === null || !set.picking) {
        return;
      }
      const target = landmarkTarget(set);
      const { origin, direction } = rayThroughPointer(raycaster, dom, camera, e.clientX, e.clientY);
      busy = true;
      void raycastVisibleSurface(
        store.scene as Scene | WorkerScene,
        [origin.x, origin.y, origin.z],
        [direction.x, direction.y, direction.z],
        (mesh) => mesh === target && (store.getSnapshot().meshes.get(mesh)?.desired.visible ?? true),
      )
        .then((hit) => {
          if (hit !== null) {
            landLandmarkHit(app, store, hit);
          }
        })
        .catch(() => {
          // Removed meanwhile: the click is dropped.
        })
        .finally(() => {
          busy = false;
        });
    };

    dom.addEventListener("pointerdown", onDown);
    dom.addEventListener("pointerup", onUp);
    return () => {
      dom.removeEventListener("pointerdown", onDown);
      dom.removeEventListener("pointerup", onUp);
    };
  }, [app, store, armed, camera, gl, raycaster]);
}
