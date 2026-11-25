// The geodesic and path picks' viewport side (#252): while a pick is armed, a
// tap on the canvas (not a drag, not a pinch: the camera still orbits, pans
// and zooms) raycasts the engine's BVH for the picked mesh only, looking
// past the others, and queues the hit to land as a source or a path point
// (see `surface-path-actions.ts`). The canvas listeners attach only while a
// pick is armed.
import { useThree } from "@react-three/fiber";
import type { Scene } from "fast-mesh-wasm";
import type { SceneStore } from "fast-mesh-wasm/store";
import type { WorkerScene } from "fast-mesh-wasm/worker";
import { useEffect } from "react";
import { type AppState, useApp } from "../../../app/state/app-state";
import { rayThroughPointer, raycastVisibleSurface } from "../../../viewport/ux/surface-raycast";
import { tapSlop } from "../../../viewport/ux/touch";
import { view } from "../../../viewport/ux/view";
import { queueSurfaceTap } from "./surface-path-actions";

/** Takes the canvas's taps while a geodesic or path pick is armed (inside the Canvas). */
export function useSurfacePickPointer(app: AppState, store: SceneStore): void {
  const armed = useApp(app, (s) => s.surfacePath.armed !== null);
  const { camera, gl, raycaster } = useThree();

  useEffect(() => {
    if (!armed) {
      return;
    }
    const dom = gl.domElement;
    let down: { x: number; y: number; id: number } | null = null;

    const onDown = (e: PointerEvent) => {
      down = e.button === 0 && !view.multiTouch ? { x: e.clientX, y: e.clientY, id: e.pointerId } : null;
    };
    const onUp = (e: PointerEvent) => {
      const at = down;
      down = null;
      if (at === null || at.id !== e.pointerId || e.button !== 0 || view.dragging || view.multiTouch) {
        return;
      }
      if (Math.hypot(e.clientX - at.x, e.clientY - at.y) > tapSlop(e.pointerType, 4)) {
        return;
      }
      const s = app.getSnapshot().surfacePath;
      if (s.armed === null || s.mesh === null) {
        return;
      }
      const target = s.mesh;
      const { origin, direction } = rayThroughPointer(raycaster, dom, camera, e.clientX, e.clientY);
      const hit = raycastVisibleSurface(
        store.scene as Scene | WorkerScene,
        [origin.x, origin.y, origin.z],
        [direction.x, direction.y, direction.z],
        (mesh) => mesh === target && (store.getSnapshot().meshes.get(mesh)?.desired.visible ?? true),
      );
      void queueSurfaceTap(app, store.scene, hit);
    };

    dom.addEventListener("pointerdown", onDown);
    dom.addEventListener("pointerup", onUp);
    return () => {
      dom.removeEventListener("pointerdown", onDown);
      dom.removeEventListener("pointerup", onUp);
    };
  }, [app, store, armed, camera, gl, raycaster]);
}
