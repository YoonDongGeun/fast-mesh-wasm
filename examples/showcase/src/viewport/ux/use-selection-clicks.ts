// Clicks on the canvas: a click on a mesh selects it (shift adds it), lands an
// armed inspect tool's pick or an armed brush-center pick; a tap on empty
// space clears the selection. Drags, orbits, pinches and presses a tool took
// are not clicks.
import { type ThreeEvent, useThree } from "@react-three/fiber";
import type { MeshHandle, Scene } from "fast-mesh-wasm";
import type { WorkerScene } from "fast-mesh-wasm/worker";
import { type RefObject, useCallback, useEffect, useRef } from "react";
import type { AppState, InspectTool } from "../../app/state/app-state";
import { pickVertex } from "../../features/inspect/ui/InspectLayers";
import { selectMesh } from "../../features/select/ux/selection";
import type { SculptBrush } from "../../features/sculpt/ux/sculpt";
import { localHitPoint } from "./surface-raycast";
import { isExtraFinger, tapSlop } from "./touch";
import { view } from "./view";

/**
 * The viewport's selection clicks: `onPick` selects a clicked mesh, and
 * `onMissed` (the canvas's missed click) clears the selection on a tap on
 * empty space, which `downAt` (fed by `usePressTracker`) tells from a drag.
 */
export function useSelectionClicks(params: {
  app: AppState;
  select: (mesh: MeshHandle | null) => void;
  selection: MeshHandle | null;
  sculpt: SculptBrush | null;
  maskPaint: boolean;
  pointerOwnedByWave6: boolean;
  picking: MeshHandle | null;
  inspectTool: InspectTool | null;
}): {
  downAt: RefObject<{ x: number; y: number } | null>;
  onMissed: (e: MouseEvent) => void;
  onPick: (mesh: MeshHandle, additive: boolean) => void;
} {
  const { app, select, selection, sculpt, maskPaint, pointerOwnedByWave6, picking, inspectTool } = params;
  const downAt = useRef<{ x: number; y: number } | null>(null);
  const lastDragEnd = useRef(0);
  useEffect(() => {
    const onDrag = (dragging: boolean) => {
      if (!dragging) {
        lastDragEnd.current = Date.now();
      }
    };
    view.dragListeners.add(onDrag);
    return () => {
      view.dragListeners.delete(onDrag);
    };
  }, []);

  // A tap on empty space clears the selection; sculpt, mask paint, and Wave 6
  // viewport tools own the pointer, and gizmo drags and orbits are not taps.
  // While an inspect tool is armed the selection stays put (the ruler and query
  // click meshes without selecting them).
  const onMissed = useCallback(
    (e: MouseEvent) => {
      if (
        sculpt !== null ||
        maskPaint ||
        pointerOwnedByWave6 ||
        picking !== null ||
        view.dragging ||
        view.multiTouch ||
        view.pressTaken ||
        selection === null
      ) {
        return;
      }
      if (inspectTool !== null) {
        return;
      }
      if (view.gizmo !== null && view.gizmo.axis !== null) {
        return;
      }
      if (Date.now() - lastDragEnd.current < 350) {
        return;
      }
      const down = downAt.current;
      if (down === null || Math.hypot(e.clientX - down.x, e.clientY - down.y) > tapSlop(view.pointerType, 6)) {
        return;
      }
      // Through clickMulti: a shift-click on nothing is ignored, like on a mesh.
      selectMesh(app, select, selection, null, e.shiftKey);
    },
    [app, select, selection, sculpt, maskPaint, pointerOwnedByWave6, picking, inspectTool],
  );
  const onPick = useCallback(
    (mesh: MeshHandle, additive: boolean) => selectMesh(app, select, selection, mesh, additive),
    [app, select, selection],
  );
  return { downAt, onMissed, onPick };
}

/** Records canvas pointerdowns into `track`, so a missed click can tell a drag from a tap and a finger from a pinch. */
export function usePressTracker(track: { current: { x: number; y: number } | null }): void {
  const gl = useThree((s) => s.gl);
  useEffect(() => {
    const el = gl.domElement;
    const onDown = (e: PointerEvent) => {
      view.pointerType = e.pointerType;
      view.multiTouch = isExtraFinger(e);
      if (!view.multiTouch) {
        view.pressTaken = false;
        track.current = e.button === 0 ? { x: e.clientX, y: e.clientY } : null;
      }
    };
    // Capture: before the tools' listeners, which may take this press.
    el.addEventListener("pointerdown", onDown, true);
    return () => el.removeEventListener("pointerdown", onDown, true);
  }, [gl, track]);
}

/**
 * A mesh's click: selects it, or lands an armed brush-center pick on it.
 * While an inspect tool is armed the ruler and query own the click and never
 * select. Not a click: the end of an orbit drag, a pinch's lift, a stroke's
 * release, or a press a tool took.
 */
export function useMeshClick(params: {
  app: AppState;
  mesh: MeshHandle;
  scene: Scene | WorkerScene;
  tool: InspectTool | null;
  picking: boolean;
  sculpted: boolean;
  suppressClick: boolean;
  onPick: (mesh: MeshHandle, additive: boolean) => void;
  onPickPoint: (mesh: MeshHandle, point: readonly [number, number, number]) => void;
}): (e: ThreeEvent<MouseEvent>) => void {
  const { app, mesh, scene, tool, picking, sculpted, suppressClick, onPick, onPickPoint } = params;
  return useCallback(
    (e: ThreeEvent<MouseEvent>) => {
      e.stopPropagation();
      // The end of an orbit drag that started on the mesh is not a click,
      // nor is a pinch's lift.
      if (e.delta > tapSlop(view.pointerType, 2) || view.multiTouch) {
        return;
      }
      // A stroke's release on the mesh is not a click either, nor on a mesh
      // behind it, which the sculpted mesh's missing handler lets the click reach.
      if (sculpted || suppressClick || view.pressTaken) {
        return;
      }
      // The inspect tools own the click while armed, and never select: the
      // ruler appends a world point (a third click restarts it), the query
      // records the snapped vertex. A hit on another mesh still lands.
      if (tool === "ruler" || tool === "query") {
        const { origin, direction } = e.ray;
        const at = app.getSnapshot();
        void Promise.resolve()
          .then(() =>
            pickVertex(scene, [origin.x, origin.y, origin.z], [direction.x, direction.y, direction.z]),
          )
          .then((picked) => {
            if (picked === null) {
              return;
            }
            if (tool === "ruler") {
              const ruler = at.ruler;
              if (ruler === null || ruler.b !== null) {
                app.setRuler({ a: picked.point, b: null });
              } else {
                app.setRuler({ a: ruler.a, b: picked.point });
              }
            } else {
              app.setInspectQuery({ mesh: picked.mesh, slot: picked.slot, point: picked.point });
            }
          })
          .catch(() => {
            // Removed meanwhile: the click is dropped.
          });
        return;
      }
      if (!picking) {
        onPick(mesh, e.nativeEvent.shiftKey);
        return;
      }
      // A pick click never selects: raycast and interpolate the hit in the
      // mesh's local positions, the frame the brush center is in. A hit on
      // another mesh (or nothing) leaves the pick armed.
      const { origin, direction } = e.ray;
      void Promise.resolve()
        .then(() => scene.raycast([origin.x, origin.y, origin.z], [direction.x, direction.y, direction.z]))
        .then((hit) => {
          if (hit === null || hit.mesh !== mesh) {
            return;
          }
          onPickPoint(mesh, localHitPoint(scene, hit));
        })
        .catch(() => {
          // Removed meanwhile: the pick stays armed.
        });
    },
    [mesh, onPick, picking, onPickPoint, scene, sculpted, suppressClick, tool, app],
  );
}
