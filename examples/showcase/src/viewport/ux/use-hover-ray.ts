// The vertex under the pointer, for the Group mode's hover tip (`view.hover`).
import { useThree } from "@react-three/fiber";
import type { Scene } from "fast-mesh-wasm";
import type { WorkerScene } from "fast-mesh-wasm/worker";
import { useEffect } from "react";
import * as THREE from "three";
import { HOVER_PERIOD_MS, nearestCorner, rayThroughPointer } from "./surface-raycast";
import { isExtraFinger, tapSlop } from "./touch";
import { setHover, view } from "./view";

/**
 * The vertex under the pointer, into `view.hover`: the engine's raycast
 * (its BVH) from the camera through the pointer, the hit triangle's corner
 * with the largest barycentric weight. One raycast at a time, at most every
 * HOVER_PERIOD_MS, for the latest pointer position; none while a button is
 * down (orbiting, or the gizmo dragging). Cleared when the pointer leaves
 * the canvas and on unmount.
 */
export function useHoverRay(scene: Scene | WorkerScene): void {
  const camera = useThree((s) => s.camera);
  const gl = useThree((s) => s.gl);

  useEffect(() => {
    const canvas = gl.domElement;
    const caster = new THREE.Raycaster();
    let alive = true;
    let pending: { x: number; y: number } | null = null;
    let running = false;
    let last = -Infinity;
    let timer: ReturnType<typeof setTimeout> | undefined;
    // Touch has no hover: a tap asks for the point under it instead, and the
    // tip stays until the next touch (which clears it).
    let fingerAt: { x: number; y: number } | null = null;
    let fingerDown = false;
    const run = () => {
      if (!alive || running || pending === null) {
        return;
      }
      const wait = last + HOVER_PERIOD_MS - performance.now();
      if (wait > 0) {
        timer ??= setTimeout(() => {
          timer = undefined;
          run();
        }, wait);
        return;
      }
      const at = pending;
      pending = null;
      running = true;
      last = performance.now();
      const { origin, direction } = rayThroughPointer(caster, canvas, camera, at.x, at.y);
      Promise.resolve()
        .then(() => scene.raycast([origin.x, origin.y, origin.z], [direction.x, direction.y, direction.z]))
        .then(
          (hit) => {
            if (!alive || fingerDown) {
              return;
            }
            if (hit === null) {
              setHover(null);
              return;
            }
            setHover({ mesh: hit.mesh, slot: nearestCorner(scene, hit), x: at.x, y: at.y });
          },
          () => {
            // A mesh removed (or the scene restarted) meanwhile: the next move asks again.
          },
        )
        .finally(() => {
          running = false;
          run();
        });
    };
    const onMove = (e: PointerEvent) => {
      if (e.buttons !== 0 || view.dragging) {
        return;
      }
      pending = { x: e.clientX, y: e.clientY };
      run();
    };
    const onLeave = (e: PointerEvent) => {
      // A lifted finger leaves the canvas: that is when a tap's tip appears.
      if (e.pointerType === "touch") {
        return;
      }
      pending = null;
      setHover(null);
    };
    const onDown = (e: PointerEvent) => {
      if (e.pointerType !== "touch") {
        return;
      }
      fingerDown = true;
      fingerAt = isExtraFinger(e) ? null : { x: e.clientX, y: e.clientY };
      pending = null;
      setHover(null);
    };
    const onUp = (e: PointerEvent) => {
      if (e.pointerType !== "touch") {
        return;
      }
      fingerDown = false;
      const at = fingerAt;
      fingerAt = null;
      if (e.type === "pointerup" && at !== null && Math.hypot(e.clientX - at.x, e.clientY - at.y) <= tapSlop("touch", 0)) {
        pending = at;
        run();
      }
    };
    canvas.addEventListener("pointermove", onMove);
    canvas.addEventListener("pointerleave", onLeave);
    canvas.addEventListener("pointerdown", onDown);
    canvas.addEventListener("pointerup", onUp);
    canvas.addEventListener("pointercancel", onUp);
    return () => {
      alive = false;
      clearTimeout(timer);
      canvas.removeEventListener("pointermove", onMove);
      canvas.removeEventListener("pointerleave", onLeave);
      canvas.removeEventListener("pointerdown", onDown);
      canvas.removeEventListener("pointerup", onUp);
      canvas.removeEventListener("pointercancel", onUp);
      setHover(null);
    };
  }, [scene, camera, gl]);
}
