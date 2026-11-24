// The gizmo on the selected mesh: the mesh bindings it attaches to, as
// `<FastMesh onBind>` delivers them, and its drag wiring. A TransformControls
// gizmo moves the mesh through `attachTransformControls` (one undo step per
// drag); a scale drag lands as one `applyMeshScale` on release.
import { useThree } from "@react-three/fiber";
import type { MeshHandle } from "fast-mesh-wasm";
import { useScene } from "fast-mesh-wasm/react";
import { attachTransformControls, type MeshBinding } from "fast-mesh-wasm/three";
import { useCallback, useEffect, useState } from "react";
import type { TrackballControls } from "three/addons/controls/TrackballControls.js";
import { TransformControls } from "three/addons/controls/TransformControls.js";
import { applyMeshScale } from "../../features/transform/ux/transform-math";
import type { GizmoMode } from "../ui/Viewport";
import { isExtraFinger } from "./touch";
import { setDragging, view } from "./view";

/** Each mesh's binding, as `<FastMesh onBind>` delivers it (`onBind`): for the gizmo, and as `view.bindings`. */
export function useMeshBindings(): {
  bindings: ReadonlyMap<MeshHandle, MeshBinding>;
  onBind: (mesh: MeshHandle, bound: MeshBinding | null) => void;
} {
  const [bindings, setBindings] = useState<ReadonlyMap<MeshHandle, MeshBinding>>(() => new Map());
  const onBind = useCallback((mesh: MeshHandle, bound: MeshBinding | null) => {
    setBindings((prev) => {
      const next = new Map(prev);
      if (bound === null) {
        next.delete(mesh);
      } else {
        next.set(mesh, bound);
      }
      return next;
    });
  }, []);
  useEffect(() => {
    view.bindings = bindings;
  }, [bindings]);
  return { bindings, onBind };
}

/**
 * A TransformControls gizmo on `bound` in `mode`, while the trackball is up.
 * The trackball stands down while it drags (`view.dragging`), and it is
 * published as `view.gizmo`.
 */
export function useGizmoDrag(params: {
  orbit: TrackballControls | null;
  bound: MeshBinding | null;
  mode: GizmoMode;
}): void {
  const { orbit, bound, mode } = params;
  const store = useScene();
  const camera = useThree((s) => s.camera);
  const gl = useThree((s) => s.gl);
  const scene = useThree((s) => s.scene);
  const [gizmo, setGizmo] = useState<TransformControls | null>(null);

  const isScale = mode === "scale";
  useEffect(() => {
    if (orbit === null || bound === null) {
      return;
    }
    // The gizmo takes every pointer on the canvas, so a second finger would
    // grab a handle or move the one in hand, and its lift would end the drag.
    // It stands down for that finger's events: this runs before its listeners.
    const canvas = gl.domElement;
    const gate = (e: PointerEvent) => {
      controls.enabled = !isExtraFinger(e);
    };
    // It has no pointercancel of its own: a drag the system took over ends here.
    const onCancel = (e: PointerEvent) => {
      if (!isExtraFinger(e) && controls.dragging) {
        controls.pointerUp(null);
      }
    };
    const POINTER_EVENTS = ["pointerdown", "pointermove", "pointerup"] as const;
    for (const type of POINTER_EVENTS) {
      canvas.addEventListener(type, gate);
    }
    const controls = new TransformControls(camera, canvas);
    canvas.addEventListener("pointercancel", onCancel);
    const helper = controls.getHelper();
    scene.add(helper);
    controls.attach(bound.object);
    const detach = isScale ? null : attachTransformControls(controls, store.scene, bound);
    if (isScale) {
      controls.setMode("scale");
    }
    const onDragging = (e: { value: unknown }) => {
      const active = e.value === true;
      orbit.enabled = !active;
      setDragging(active);
      if (isScale && !active) {
        const sx = bound.object.scale.x;
        const sy = bound.object.scale.y;
        const sz = bound.object.scale.z;
        bound.object.scale.set(1, 1, 1);
        if (
          Number.isFinite(sx) &&
          Number.isFinite(sy) &&
          Number.isFinite(sz) &&
          sx > 0 &&
          sy > 0 &&
          sz > 0 &&
          (Math.abs(sx - 1) > 1e-4 || Math.abs(sy - 1) > 1e-4 || Math.abs(sz - 1) > 1e-4)
        ) {
          void applyMeshScale(store, bound.mesh, [sx, sy, sz]);
        }
      }
    };
    controls.addEventListener("dragging-changed", onDragging);
    view.gizmo = controls;
    setGizmo(controls);
    return () => {
      for (const type of POINTER_EVENTS) {
        canvas.removeEventListener(type, gate);
      }
      canvas.removeEventListener("pointercancel", onCancel);
      controls.removeEventListener("dragging-changed", onDragging);
      detach?.();
      bound.object.scale.set(1, 1, 1);
      controls.detach();
      scene.remove(helper);
      controls.dispose();
      orbit.enabled = true;
      setDragging(false);
      if (view.gizmo === controls) {
        view.gizmo = null;
      }
      setGizmo((g) => (g === controls ? null : g));
    };
  }, [store, camera, gl, scene, orbit, bound, isScale]);

  useEffect(() => {
    gizmo?.setMode(mode);
  }, [gizmo, mode]);
}
