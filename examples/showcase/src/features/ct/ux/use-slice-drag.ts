// Dragging the MPR slice quads through the main scene by their corner
// balls: the pointer handling that turns a press on a ball and the pull
// after it into slice steps on the linked MPR state.
import { useThree } from "@react-three/fiber";
import { useEffect, useRef } from "react";
import * as THREE from "three";
import type { AppState } from "../../../app/state/app-state";
import type { VolumePayload } from "../../../io/protocol";
import { isExtraFinger } from "../../../viewport/ux/touch";
import { view } from "../../../viewport/ux/view";
import type { SceneSliceQuads } from "../ui/SceneSlices";
import { clampSlice, dragPlaneNormal, planeHit, sliceSteps } from "./slice-drag";
import type { SliceAxis } from "./volume-slice";

type SliceDragState = {
  axis: SliceAxis;
  volumeId: string;
  index0: number;
  grab: [number, number, number];
  planePoint: [number, number, number];
  planeNormal: [number, number, number];
  /** The quad normal towards higher slices (world). */
  along: [number, number, number];
  spacing: number;
  count: number;
};

/**
 * Drags the slice `quads` by their corner balls: a press on one grabs the
 * stack, and pulling it moves the slice (the linked MPR state, so every
 * slice view follows). The orbit stands down for the gesture, like for a
 * sculpt stroke; the gizmo's own drags and the sculpt brush keep priority.
 * Listens on the canvas while mounted, reading the latest `payload` and
 * `active` as the pointer acts.
 */
export function useSliceDrag(app: AppState, quads: SceneSliceQuads, payload: VolumePayload | null, active: boolean): void {
  const camera = useThree((s) => s.camera);
  const gl = useThree((s) => s.gl);
  const payloadRef = useRef(payload);
  payloadRef.current = payload;
  const activeRef = useRef(active);
  activeRef.current = active;

  useEffect(() => {
    const canvas = gl.domElement;
    const caster = new THREE.Raycaster();
    let drag: SliceDragState | null = null;
    let pointerId = -1;

    const rayOf = (x: number, y: number): THREE.Ray => {
      const r = canvas.getBoundingClientRect();
      caster.setFromCamera(
        new THREE.Vector2(((x - r.left) / r.width) * 2 - 1, 1 - ((y - r.top) / r.height) * 2),
        camera,
      );
      return caster.ray;
    };

    const endDrag = (): void => {
      drag = null;
      pointerId = -1;
      canvas.style.cursor = "";
      if (view.orbit !== null) {
        view.orbit.enabled = true;
      }
    };

    const onDown = (e: PointerEvent): void => {
      if (e.button !== 0 || isExtraFinger(e) || drag !== null || view.dragging) {
        return;
      }
      const snap = app.getSnapshot();
      if (snap.sculpt !== null || snap.brushPick !== null) {
        return;
      }
      const payload = payloadRef.current;
      const mpr = snap.volumeMpr;
      if (!activeRef.current || payload === null || mpr === null || !quads.group.visible) {
        return;
      }
      const ray = rayOf(e.clientX, e.clientY);
      const hits = caster.intersectObjects(
        quads.handles.map((h) => h.object),
        false,
      );
      if (hits.length === 0) {
        return;
      }
      const axis = (hits[0].object.userData.sliceAxis ?? null) as SliceAxis | null;
      const quad = axis === null ? null : quads.quadMesh(axis);
      if (axis === null || quad === null) {
        return;
      }
      const normal = quad.getWorldDirection(new THREE.Vector3());
      const seen = camera.getWorldDirection(new THREE.Vector3());
      const plane = dragPlaneNormal([normal.x, normal.y, normal.z], [seen.x, seen.y, seen.z]);
      if (plane === null) {
        return;
      }
      const center = hits[0].object.getWorldPosition(new THREE.Vector3());
      const grab = planeHit(
        [ray.origin.x, ray.origin.y, ray.origin.z],
        [ray.direction.x, ray.direction.y, ray.direction.z],
        [center.x, center.y, center.z],
        plane,
      );
      if (grab === null) {
        return;
      }
      const sign = quads.axisSign(axis, normal);
      const spacing = axis === "axial" ? payload.spacing[2] : axis === "coronal" ? payload.spacing[1] : payload.spacing[0];
      const count = axis === "axial" ? payload.slices : axis === "coronal" ? payload.rows : payload.columns;
      drag = {
        axis,
        volumeId: mpr.volumeId,
        index0: mpr.slices[axis],
        grab,
        planePoint: [center.x, center.y, center.z],
        planeNormal: plane,
        along: [sign * normal.x, sign * normal.y, sign * normal.z],
        spacing,
        count,
      };
      pointerId = e.pointerId;
      view.pressTaken = true;
      try {
        canvas.setPointerCapture(e.pointerId);
      } catch {
        // Already released: the drag still tracks until the button lifts.
      }
      if (view.orbit !== null) {
        view.orbit.enabled = false;
      }
      canvas.style.cursor = "grabbing";
    };

    const onMove = (e: PointerEvent): void => {
      if (drag === null) {
        if (e.buttons !== 0 || view.dragging) {
          return;
        }
        const snap = app.getSnapshot();
        if (
          snap.sculpt !== null ||
          snap.brushPick !== null ||
          !activeRef.current ||
          !quads.group.visible
        ) {
          return;
        }
        rayOf(e.clientX, e.clientY);
        const hits = caster.intersectObjects(
          quads.handles.map((h) => h.object),
          false,
        );
        canvas.style.cursor = hits.length === 0 ? "" : "grab";
        return;
      }
      if (e.pointerId !== pointerId) {
        return;
      }
      const ray = rayOf(e.clientX, e.clientY);
      const now = planeHit(
        [ray.origin.x, ray.origin.y, ray.origin.z],
        [ray.direction.x, ray.direction.y, ray.direction.z],
        drag.planePoint,
        drag.planeNormal,
      );
      if (now === null) {
        return;
      }
      const index = clampSlice(drag.index0 + sliceSteps(drag.grab, now, drag.along, drag.spacing), drag.count);
      const cur = app.getSnapshot().volumeMpr;
      if (cur !== null && cur.volumeId === drag.volumeId && cur.slices[drag.axis] !== index) {
        app.setVolumeMpr({ ...cur, slices: { ...cur.slices, [drag.axis]: index } });
      }
    };

    const onUp = (e: PointerEvent): void => {
      if (e.pointerId === pointerId) {
        endDrag();
      }
    };

    const onLeave = (): void => {
      if (drag === null) {
        canvas.style.cursor = "";
      }
    };

    canvas.addEventListener("pointerdown", onDown);
    canvas.addEventListener("pointermove", onMove);
    canvas.addEventListener("pointerup", onUp);
    canvas.addEventListener("pointercancel", onUp);
    canvas.addEventListener("pointerleave", onLeave);
    return () => {
      canvas.removeEventListener("pointerdown", onDown);
      canvas.removeEventListener("pointermove", onMove);
      canvas.removeEventListener("pointerup", onUp);
      canvas.removeEventListener("pointercancel", onUp);
      canvas.removeEventListener("pointerleave", onLeave);
      endDrag();
    };
  }, [app, quads, camera, gl]);
}
