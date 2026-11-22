// The placement picks' viewport side (#256): while a pick is armed, a tap
// on the canvas (not a drag, not a pinch) raycasts the engine's BVH for the
// armed mesh only, looking past the others, and lands the hit: its world
// point and its face's outward world normal. The last click of a pick sets
// the new placement (one undo step) and disarms it. The canvas listeners
// attach only while a pick is armed.
import { useThree } from "@react-three/fiber";
import type { MeshHandle, RayHit, Scene, Vec3 } from "fast-mesh-wasm";
import type { SceneStore } from "fast-mesh-wasm/store";
import type { WorkerScene } from "fast-mesh-wasm/worker";
import { useEffect } from "react";
import { type AppState, useApp } from "../../../app/state/app-state";
import { rayThroughPointer, raycastVisibleSurface } from "../../../viewport/ux/surface-raycast";
import { tapSlop } from "../../../viewport/ux/touch";
import { view } from "../../../viewport/ux/view";
import { placeMesh, worldTriangleNormal } from "./placement-helpers";
import { addPlacementHit, placementForPick, pickReport } from "./placement-picks";

/**
 * Lands an engine hit on the armed pick: adds it while more clicks are
 * needed, and sets the placement on the last one. A hit on another mesh, or
 * with nothing armed, does nothing.
 */
export function landPlacementHit(
  app: AppState,
  store: SceneStore,
  hit: Pick<RayHit, "mesh" | "triangle" | "point">,
): void {
  const pick = app.getSnapshot().placementTool.pick;
  if (pick === null || hit.mesh !== pick.mesh) {
    return;
  }
  let normal: Vec3 | null = null;
  try {
    normal = worldTriangleNormal(store.scene, hit.mesh, hit.triangle);
  } catch {
    return; // removed meanwhile
  }
  const { pick: next, complete } = addPlacementHit(pick, { point: [hit.point[0], hit.point[1], hit.point[2]], normal });
  if (!complete) {
    app.setPlacementTool({ pick: next });
    return;
  }
  try {
    const placed = placeMesh(store, pick.mesh, (mesh) => placementForPick(mesh, next));
    app.setPlacementTool({ pick: null, report: { text: placed === null ? "Already in place" : pickReport(next), error: false } });
  } catch (e) {
    app.setPlacementTool({ pick: null, report: { text: e instanceof Error ? e.message : String(e), error: true } });
  }
}

/** Takes the canvas's taps while a placement pick is armed (inside the Canvas). */
export function usePlacementPickPointer(app: AppState, store: SceneStore): void {
  const armed = useApp(app, (s) => s.placementTool.pick !== null);
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
      const pick = app.getSnapshot().placementTool.pick;
      if (pick === null) {
        return;
      }
      const target: MeshHandle = pick.mesh;
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
            landPlacementHit(app, store, hit);
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
