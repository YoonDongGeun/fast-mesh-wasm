// Dragging the section gizmo's handle ball (#110): the pointer ray meets a
// drag plane through the handle that faces the camera while it contains
// the section normal, and the hit's travel along the normal moves the
// plane. Engine commits through the store are throttled; the app state
// (and so the drawing) follows every move.
import type { SceneStore } from "fast-mesh-wasm/store";
import { useEffect, useRef } from "react";
import * as THREE from "three";
import type { AppState } from "../../../app/state/app-state";
import { dragPlaneNormal, planeHit } from "../../ct/ux/slice-drag";
import { isExtraFinger } from "../../../viewport/ux/touch";
import { setDragging, view } from "../../../viewport/ux/view";
import { normalTravel } from "./inspect-tools";

/** The drawn gizmo the drag reads: its group (for visibility) and the handle ball it grabs. */
export interface SectionGizmoTarget {
  group: THREE.Group;
  handle: THREE.Mesh;
}

/**
 * Listens on the canvas for presses on the gizmo's handle and moves the
 * section plane along its normal while the button is held. Hovering the
 * handle shows a grab cursor; the orbit is off for the drag.
 */
export function useSectionDrag(
  app: AppState,
  store: SceneStore,
  camera: THREE.Camera,
  gl: THREE.WebGLRenderer,
  gizmo: SectionGizmoTarget,
): void {
  const drag = useRef<{
    pointerId: number;
    grab: [number, number, number];
    planePoint: [number, number, number];
    planeNormal: [number, number, number];
    origin0: [number, number, number];
    normal: [number, number, number];
    lastCommit: number;
  } | null>(null);

  // Dragging the handle ball moves the plane along its normal.
  useEffect(() => {
    const canvas = gl.domElement;
    const caster = new THREE.Raycaster();
    const rayOf = (x: number, y: number): THREE.Ray => {
      const r = canvas.getBoundingClientRect();
      caster.setFromCamera(
        new THREE.Vector2(((x - r.left) / r.width) * 2 - 1, 1 - ((y - r.top) / r.height) * 2),
        camera,
      );
      return caster.ray;
    };
    const commit = (origin: [number, number, number], normal: [number, number, number], force: boolean) => {
      const snap = app.getSnapshot();
      const cur = snap.inspectSection;
      if (cur === null) {
        return;
      }
      const d = drag.current;
      const now = performance.now();
      if (!force && d !== null && now - d.lastCommit < 80) {
        return;
      }
      if (d !== null) {
        d.lastCommit = now;
      }
      try {
        store.setSection(cur.handle, { plane: { origin: [...origin], normal: [...normal] } });
      } catch {
        // Removed meanwhile: the gizmo cleans up.
      }
      app.setInspectSection({ handle: cur.handle, plane: { origin: [...origin], normal: [...normal] } });
    };
    const onDown = (e: PointerEvent) => {
      const snap = app.getSnapshot();
      const cur = snap.inspectSection;
      if (e.button !== 0 || isExtraFinger(e) || drag.current !== null || view.dragging || cur === null) {
        return;
      }
      if (snap.sculpt !== null || snap.brushPick !== null) {
        return;
      }
      const ray = rayOf(e.clientX, e.clientY);
      if (caster.intersectObject(gizmo.handle, false).length === 0) {
        return;
      }
      const normal = new THREE.Vector3(cur.plane.normal[0], cur.plane.normal[1], cur.plane.normal[2]);
      const seen = camera.getWorldDirection(new THREE.Vector3());
      const dp = dragPlaneNormal([normal.x, normal.y, normal.z], [seen.x, seen.y, seen.z]);
      if (dp === null) {
        return;
      }
      const center = gizmo.handle.getWorldPosition(new THREE.Vector3());
      const grab = planeHit(
        [ray.origin.x, ray.origin.y, ray.origin.z],
        [ray.direction.x, ray.direction.y, ray.direction.z],
        [center.x, center.y, center.z],
        dp,
      );
      if (grab === null) {
        return;
      }
      drag.current = {
        pointerId: e.pointerId,
        grab,
        planePoint: [center.x, center.y, center.z],
        planeNormal: dp,
        origin0: [...cur.plane.origin],
        normal: [cur.plane.normal[0], cur.plane.normal[1], cur.plane.normal[2]],
        lastCommit: 0,
      };
      try {
        canvas.setPointerCapture(e.pointerId);
      } catch {
        // Already released: the drag still tracks until the button lifts.
      }
      if (view.orbit !== null) {
        view.orbit.enabled = false;
      }
      setDragging(true);
      canvas.style.cursor = "grabbing";
    };
    const onMove = (e: PointerEvent) => {
      const d = drag.current;
      if (d === null) {
        if (e.buttons !== 0 || view.dragging) {
          return;
        }
        const cur = app.getSnapshot().inspectSection;
        if (cur === null || !gizmo.group.visible) {
          return;
        }
        rayOf(e.clientX, e.clientY);
        canvas.style.cursor = caster.intersectObject(gizmo.handle, false).length === 0 ? "" : "grab";
        return;
      }
      if (e.pointerId !== d.pointerId) {
        return;
      }
      const ray = rayOf(e.clientX, e.clientY);
      const now = planeHit(
        [ray.origin.x, ray.origin.y, ray.origin.z],
        [ray.direction.x, ray.direction.y, ray.direction.z],
        d.planePoint,
        d.planeNormal,
      );
      if (now === null) {
        return;
      }
      const t = normalTravel(d.grab, now, d.normal);
      commit(
        [d.origin0[0] + d.normal[0] * t, d.origin0[1] + d.normal[1] * t, d.origin0[2] + d.normal[2] * t],
        d.normal,
        false,
      );
    };
    const endDrag = (e: PointerEvent, commitLast: boolean) => {
      const d = drag.current;
      if (d === null || e.pointerId !== d.pointerId) {
        return;
      }
      if (commitLast) {
        const cur = app.getSnapshot().inspectSection;
        if (cur !== null) {
          commit(
            [...cur.plane.origin] as [number, number, number],
            [...cur.plane.normal] as [number, number, number],
            true,
          );
        }
      }
      drag.current = null;
      canvas.style.cursor = "";
      if (view.orbit !== null) {
        view.orbit.enabled = true;
      }
      setDragging(false);
    };
    const onUp = (e: PointerEvent) => endDrag(e, true);
    const onCancel = (e: PointerEvent) => endDrag(e, false);
    const onLeave = () => {
      if (drag.current === null) {
        canvas.style.cursor = "";
      }
    };
    canvas.addEventListener("pointerdown", onDown);
    canvas.addEventListener("pointermove", onMove);
    canvas.addEventListener("pointerup", onUp);
    canvas.addEventListener("pointercancel", onCancel);
    canvas.addEventListener("pointerleave", onLeave);
    return () => {
      canvas.removeEventListener("pointerdown", onDown);
      canvas.removeEventListener("pointermove", onMove);
      canvas.removeEventListener("pointerup", onUp);
      canvas.removeEventListener("pointercancel", onCancel);
      canvas.removeEventListener("pointerleave", onLeave);
      drag.current = null;
    };
  }, [app, store, camera, gl, gizmo]);
}
