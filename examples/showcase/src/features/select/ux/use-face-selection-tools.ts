// The face selection tools' pointer handling (#233): box and lasso drags
// publish the 2D marquee and select by screen shape on release, the brush
// paints sphere dabs along the stroke, and crease and shell select from a
// seed triangle under the press. Every tool combines with the current
// selection by the combine mode and keeps the orbit off while it owns the
// press.
import type { SceneStore } from "fast-mesh-wasm/store";
import { useEffect, useRef } from "react";
import type * as THREE from "three";
import type { AppState, Item } from "../../../app/state/app-state";
import type { Wave6SelectState } from "../../../app/state/wave6-state";
import { view } from "../../../viewport/ux/view";
import { setFaceSelectionMarquee } from "./face-selection-marquee";
import {
  applyScreenFaceSelection,
  raycastVisibleItem,
  resolveActiveSelectItem,
  writeMeshFaceSelection,
} from "./face-selection-ops";
import { combineFaceSelection, EMPTY_U32 } from "./face-sets";
import {
  boxShapeFromPixels,
  buildScreenSelectOptions,
  clientToLocalPixels,
  lassoShapeFromPixels,
  rayFromClient,
} from "./screen-select";

/** A lasso only records a new point once the pointer has moved this far (in pixels). */
const LASSO_MIN_STEP_PX = 3;

/**
 * Listens on the canvas while `activeTool` is armed (pointer down in the
 * capture phase, so the tool takes the press before the viewport does).
 * Disarming clears the marquee and gives the orbit back.
 */
export function useFaceSelectionTools(
  app: AppState,
  store: SceneStore,
  camera: THREE.Camera,
  gl: THREE.WebGLRenderer,
  activeTool: Wave6SelectState["activeTool"],
): void {
  const orbitLockRef = useRef(false);

  useEffect(() => {
    const dom = gl.domElement;

    const releaseOrbitLock = () => {
      if (orbitLockRef.current) {
        if (view.orbit !== null) {
          view.orbit.enabled = true;
        }
        view.pressTaken = false;
        orbitLockRef.current = false;
      }
    };

    if (!activeTool) {
      setFaceSelectionMarquee(null);
      releaseOrbitLock();
      return;
    }

    type DragState =
      | {
          kind: "box";
          pointerId: number;
          item: Item;
          rect: DOMRect;
          x0: number;
          y0: number;
          x1: number;
          y1: number;
        }
      | {
          kind: "lasso";
          pointerId: number;
          item: Item;
          rect: DOMRect;
          points: [number, number][];
        }
      | {
          kind: "brush";
          pointerId: number;
          preSelectionByItemId: Map<number, Uint32Array>;
          touchedByItemId: Map<number, Set<number>>;
        }
      | {
          kind: "seed";
          pointerId: number;
          tool: "crease" | "shell";
        };

    let drag: DragState | null = null;
    let brushBusy = false;
    let queuedBrushEvent: { clientX: number; clientY: number } | null = null;

    const localCoords = (e: PointerEvent, rect: DOMRect): [number, number] =>
      clientToLocalPixels(e.clientX, e.clientY, rect);

    const runBrushDab = async (clientX: number, clientY: number, brushDrag: Extract<DragState, { kind: "brush" }>) => {
      if (brushBusy) {
        queuedBrushEvent = { clientX, clientY };
        return;
      }
      brushBusy = true;
      try {
        const repair = store.scene.repair;
        if (!repair) {
          return;
        }
        const cfg = app.getSnapshot().wave6.select;
        const rect = dom.getBoundingClientRect();
        const ray = rayFromClient(camera, clientX, clientY, rect);
        const picked = await raycastVisibleItem(
          store,
          app.getSnapshot().items,
          ray.origin,
          ray.direction,
          cfg.backfaceCull,
        );
        if (!picked) {
          return;
        }
        const dabTris = await Promise.resolve(
          repair.selectFacesInSphere(picked.item.mesh, picked.hit.point, cfg.brushRadius),
        );
        let pre = brushDrag.preSelectionByItemId.get(picked.item.id);
        if (!pre) {
          pre = app.getSnapshot().wave6.select.selectedByMesh[picked.item.id] ?? EMPTY_U32;
          brushDrag.preSelectionByItemId.set(picked.item.id, pre);
        }
        let touched = brushDrag.touchedByItemId.get(picked.item.id);
        if (!touched) {
          touched = new Set<number>();
          brushDrag.touchedByItemId.set(picked.item.id, touched);
        }
        for (let i = 0; i < dabTris.length; i++) {
          touched.add(dabTris[i]);
        }
        const strokeUnion = Uint32Array.from(touched);
        const combined = combineFaceSelection(pre, strokeUnion, cfg.combineMode);
        writeMeshFaceSelection(app, picked.item.id, combined);
        if (store.getSnapshot().selection !== picked.item.mesh) {
          store.select(picked.item.mesh);
        }
      } finally {
        brushBusy = false;
        const pending = queuedBrushEvent;
        queuedBrushEvent = null;
        if (pending && drag?.kind === "brush") {
          void runBrushDab(pending.clientX, pending.clientY, drag);
        }
      }
    };

    const runSeedSelect = async (clientX: number, clientY: number, tool: "crease" | "shell") => {
      const repair = store.scene.repair;
      if (!repair) {
        return;
      }
      const cfg = app.getSnapshot().wave6.select;
      const rect = dom.getBoundingClientRect();
      const ray = rayFromClient(camera, clientX, clientY, rect);
      const picked = await raycastVisibleItem(
        store,
        app.getSnapshot().items,
        ray.origin,
        ray.direction,
        cfg.backfaceCull,
      );
      if (!picked) {
        return;
      }
      const incoming =
        tool === "crease"
          ? await Promise.resolve(
              repair.selectByCreaseAngle(
                picked.item.mesh,
                picked.hit.triangle,
                (cfg.creaseAngleDeg * Math.PI) / 180,
              ),
            )
          : await Promise.resolve(repair.selectConnectedShell(picked.item.mesh, picked.hit.triangle));
      const pre = app.getSnapshot().wave6.select.selectedByMesh[picked.item.id] ?? EMPTY_U32;
      const combined = combineFaceSelection(pre, incoming, cfg.combineMode);
      writeMeshFaceSelection(app, picked.item.id, combined);
      if (store.getSnapshot().selection !== picked.item.mesh) {
        store.select(picked.item.mesh);
      }
    };

    const acquireOrbitLock = (pointerId: number) => {
      if (view.orbit !== null) {
        view.orbit.enabled = false;
      }
      view.pressTaken = true;
      orbitLockRef.current = true;
      try {
        dom.setPointerCapture?.(pointerId);
      } catch {
        // Synthetic events outside a live browser pointer stream have no active pointer capture.
      }
    };

    const releaseCapture = (pointerId: number) => {
      try {
        dom.releasePointerCapture?.(pointerId);
      } catch {
        // Ignore if pointer capture was never registered or already lost.
      }
    };

    const finishGesture = () => {
      drag = null;
      queuedBrushEvent = null;
      setFaceSelectionMarquee(null);
      releaseOrbitLock();
    };

    const onPointerDown = (e: PointerEvent) => {
      if (e.button !== 0 || e.altKey || view.dragging || view.multiTouch || view.pressTaken) {
        return;
      }
      if (drag !== null) {
        // Second finger cancels active selection drag to let multi-touch take over.
        finishGesture();
        return;
      }
      const tool = app.getSnapshot().wave6.select.activeTool;
      if (!tool) {
        return;
      }

      if (tool === "box" || tool === "lasso") {
        const target = resolveActiveSelectItem(app, store);
        if (!target) {
          return;
        }
        const rect = dom.getBoundingClientRect();
        const [x, y] = localCoords(e, rect);
        acquireOrbitLock(e.pointerId);

        if (tool === "box") {
          drag = { kind: "box", pointerId: e.pointerId, item: target, rect, x0: x, y0: y, x1: x, y1: y };
          setFaceSelectionMarquee({
            kind: "box",
            x0: x,
            y0: y,
            x1: x,
            y1: y,
            width: rect.width,
            height: rect.height,
          });
        } else {
          drag = { kind: "lasso", pointerId: e.pointerId, item: target, rect, points: [[x, y]] };
          setFaceSelectionMarquee({
            kind: "lasso",
            points: [[x, y]],
            width: rect.width,
            height: rect.height,
          });
        }
        return;
      }

      acquireOrbitLock(e.pointerId);

      if (tool === "brush") {
        const brushDrag: Extract<DragState, { kind: "brush" }> = {
          kind: "brush",
          pointerId: e.pointerId,
          preSelectionByItemId: new Map(),
          touchedByItemId: new Map(),
        };
        drag = brushDrag;
        void runBrushDab(e.clientX, e.clientY, brushDrag);
        return;
      }

      if (tool === "crease" || tool === "shell") {
        drag = { kind: "seed", pointerId: e.pointerId, tool };
        void runSeedSelect(e.clientX, e.clientY, tool);
      }
    };

    const onPointerMove = (e: PointerEvent) => {
      if (!drag || e.pointerId !== drag.pointerId) {
        return;
      }
      if (drag.kind === "box") {
        const [x, y] = localCoords(e, drag.rect);
        drag.x1 = x;
        drag.y1 = y;
        setFaceSelectionMarquee({
          kind: "box",
          x0: drag.x0,
          y0: drag.y0,
          x1: x,
          y1: y,
          width: drag.rect.width,
          height: drag.rect.height,
        });
        return;
      }
      if (drag.kind === "lasso") {
        const [x, y] = localCoords(e, drag.rect);
        const last = drag.points[drag.points.length - 1];
        if (Math.hypot(x - last[0], y - last[1]) >= LASSO_MIN_STEP_PX) {
          drag.points.push([x, y]);
          setFaceSelectionMarquee({
            kind: "lasso",
            points: [...drag.points],
            width: drag.rect.width,
            height: drag.rect.height,
          });
        }
        return;
      }
      if (drag.kind === "brush") {
        void runBrushDab(e.clientX, e.clientY, drag);
      }
    };

    const onPointerUpOrCancel = (e: PointerEvent) => {
      if (!drag || e.pointerId !== drag.pointerId) {
        return;
      }
      const completed = drag;
      releaseCapture(e.pointerId);
      finishGesture();
      if (e.type === "pointercancel") {
        return;
      }
      const cfg = app.getSnapshot().wave6.select;
      const options = buildScreenSelectOptions(camera, cfg.visibleOnly, cfg.backfaceCull, true);

      if (completed.kind === "box") {
        void applyScreenFaceSelection({
          app,
          store,
          item: completed.item,
          shape: boxShapeFromPixels(completed.x0, completed.y0, completed.x1, completed.y1, completed.rect),
          options,
          combineMode: cfg.combineMode,
        });
        return;
      }

      if (completed.kind === "lasso") {
        void applyScreenFaceSelection({
          app,
          store,
          item: completed.item,
          shape: lassoShapeFromPixels(completed.points, completed.rect),
          options,
          combineMode: cfg.combineMode,
        });
      }
    };

    dom.addEventListener("pointerdown", onPointerDown, { capture: true });
    dom.addEventListener("pointermove", onPointerMove);
    dom.addEventListener("pointerup", onPointerUpOrCancel);
    dom.addEventListener("pointercancel", onPointerUpOrCancel);
    return () => {
      dom.removeEventListener("pointerdown", onPointerDown, { capture: true });
      dom.removeEventListener("pointermove", onPointerMove);
      dom.removeEventListener("pointerup", onPointerUpOrCancel);
      dom.removeEventListener("pointercancel", onPointerUpOrCancel);
      setFaceSelectionMarquee(null);
      releaseOrbitLock();
    };
  }, [app, store, camera, gl, activeTool]);
}
