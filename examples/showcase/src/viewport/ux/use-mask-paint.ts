// The mask paint: its pointer loop (press-and-hold paints the protected
// region into the sculpt mask). When it disarms, and the live-mask cache it
// shares with the sculpt stroke, are in use-mask-tracker.ts.
import { useThree } from "@react-three/fiber";
import type { MeshHandle, Scene } from "fast-mesh-wasm";
import { useScene, useSelection } from "fast-mesh-wasm/react";
import { meshState } from "fast-mesh-wasm/store";
import type { WorkerScene } from "fast-mesh-wasm/worker";
import { useEffect, useRef } from "react";
import * as THREE from "three";
import { inspect } from "../../app/state/readouts";
import { type AppState, useApp } from "../../app/state/app-state";
import { faceNormal, interpolateHit, reflectCenters, toSymmetry } from "../../features/sculpt/ux/sculpt";
import { easeNormal, placeBrushCursor } from "../../features/sculpt/ui/brush-cursor";
import { applyMaskPaintStep, type MaskLiveTracker } from "../../features/sculpt/ux/mask-protect";
import { applyAltPinClick, shouldAllowStrokeHit } from "./brush-target";
import { castVisibleSurface, HOVER_PERIOD_MS, STROKE_PERIOD_MS } from "./surface-raycast";
import { isExtraFinger } from "./touch";
import { view } from "./view";

/**
 * The mask paint on the selected mesh: hover shows the paint ball (blue,
 * against sculpt's amber), press-and-hold accumulates sphere-brush picks
 * into the mask. Painting never opens an undo step (the mask is not a mesh
 * edit); a new mesh replaces the mask, and a stale one fails loudly (see
 * `features/sculpt/ux/sculpt.ts`'s mask contract). One raycast at a time, through the
 * engine's BVH. `ball` is the paint cursor it places, owned by the caller.
 */
export function useMaskPaint(params: {
  app: AppState;
  scene: Scene | WorkerScene;
  maskTracker: MaskLiveTracker;
  ball: THREE.Group;
}): void {
  const { app, scene, maskTracker, ball } = params;
  const store = useScene();
  const [selection, select] = useSelection();
  const camera = useThree((s) => s.camera);
  const gl = useThree((s) => s.gl);
  const items = useApp(app, (s) => s.items);
  const pinToActive = useApp(app, (s) => s.sculptPinToActive);
  // The selection, items, pin toggle and radius drive the paint live, outside React.
  const selRef = useRef(selection);
  selRef.current = selection;
  const selectRef = useRef(select);
  selectRef.current = select;
  const itemsRef = useRef(items);
  itemsRef.current = items;
  const pinRef = useRef(pinToActive);
  pinRef.current = pinToActive;
  const radius = useApp(app, (s) => s.maskRadius);
  const radiusRef = useRef(radius);
  radiusRef.current = radius;

  useEffect(() => {
    const canvas = gl.domElement;
    let cursorNormal: [number, number, number] | null = null;
    /** The pointer whose press is waiting on its raycast. */
    let pressing = -1;
    const caster = new THREE.Raycaster();
    let alive = true;
    let running = false;
    let last = -Infinity;
    let timer: ReturnType<typeof setTimeout> | undefined;
    let pending: { x: number; y: number; period: number } | null = null;
    /** The painting pointer, or -1 when no paint accumulates. */
    let active = -1;
    /** The mesh the paint accumulates on (the selection at press). */
    let paintMesh: MeshHandle | null = null;
    /** Serializes async `verticesInSphere` picks so rapid moves never race `sculptMask`. */
    let paintTail: Promise<void> = Promise.resolve();

    const isVisibleMesh = (mesh: MeshHandle): boolean => {
      if (!itemsRef.current.some((i) => i.mesh === mesh)) {
        return false;
      }
      return meshState(store.getSnapshot(), mesh).desired.visible;
    };

    const isEligibleMesh = (mesh: MeshHandle): boolean => {
      const item = itemsRef.current.find((i) => i.mesh === mesh);
      return item !== undefined && !item.locked && meshState(store.getSnapshot(), mesh).desired.visible;
    };

    const canMaskTarget = (mesh: MeshHandle): boolean =>
      shouldAllowStrokeHit({
        hitMesh: mesh,
        strokeTargetMesh: paintMesh,
        pinToActive: pinRef.current,
        selection: selRef.current,
        fallbackMesh: null,
        items: itemsRef.current,
        storeSnapshot: store.getSnapshot(),
      });

    const cast = (x: number, y: number) => castVisibleSurface(scene, caster, canvas, camera, x, y, isVisibleMesh);

    /** The hit over the mesh's local views: its point, and the surface's way there for the cursor. */
    const localHit = (mesh: MeshHandle, hit: { triangle: number; barycentric: readonly [number, number, number] }) => {
      const v = scene.views(mesh);
      const indices = v.indices();
      const positions = v.positions();
      const corners = [indices[3 * hit.triangle], indices[3 * hit.triangle + 1], indices[3 * hit.triangle + 2]];
      cursorNormal = easeNormal(cursorNormal, faceNormal(positions, corners[0], corners[1], corners[2]));
      return interpolateHit(positions, corners, hit.barycentric);
    };

    const hideBall = () => {
      cursorNormal = null;
      ball.visible = false;
      inspect.sculpt.hovering = false;
    };

    /** The cursor on the surface point: a child of the mesh, in its local frame. */
    const showBall = (mesh: MeshHandle, point: readonly [number, number, number]) => {
      const binding = view.bindings.get(mesh);
      if (binding === undefined) {
        hideBall();
        return;
      }
      if (ball.parent !== binding.object) {
        binding.object.add(ball);
      }
      inspect.sculpt.cursorShows++;
      placeBrushCursor(ball, point, cursorNormal ?? [0, 0, 1], radiusRef.current);
    };

    /** Accumulates one BVH-localized sphere pick (plus active symmetry mirrors) into the mask. */
    const paintAt = async (mesh: MeshHandle, point: readonly [number, number, number]): Promise<void> => {
      const radius = radiusRef.current;
      const sym = toSymmetry(app.getSnapshot().sculptSymmetry);
      const centers = reflectCenters(point, sym);
      await applyMaskPaintStep({
        mesh,
        centers,
        radius,
        verticesInSphere: (m, c, r) => store.verticesInSphere(m, c, r),
        views: (m) => scene.views(m),
        tracker: maskTracker,
        isAlive: () => alive,
        getPrevMask: () => app.getSnapshot().sculptMask,
        setMask: (next) => app.setSculptMask(next),
        onStale: () => {
          app.setSculptMask(null);
          app.setSculptMaskNotice("The mask went stale after a topology edit and was cleared — repaint it.");
          inspect.sculpt.maskStale++;
        },
        clearNotice: () => {
          if (app.getSnapshot().sculptMaskNotice !== null) {
            app.setSculptMaskNotice(null);
          }
        },
      });
    };

    const enqueuePaint = (mesh: MeshHandle, point: readonly [number, number, number]): Promise<void> => {
      paintTail = paintTail.then(
        () => paintAt(mesh, point),
        () => paintAt(mesh, point),
      );
      return paintTail;
    };

    /** Ends the paint: the orbit goes back to the gesture. */
    const endPaint = () => {
      active = -1;
      paintMesh = null;
      if (view.orbit !== null) {
        view.orbit.enabled = true;
      }
    };

    const run = () => {
      if (!alive || running || pending === null) {
        return;
      }
      const wait = last + pending.period - performance.now();
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
      Promise.resolve()
        .then(() => cast(at.x, at.y))
        .then(async (hit) => {
          if (!alive) {
            return;
          }
          if (hit === null || !canMaskTarget(hit.mesh)) {
            if (active === -1) {
              hideBall();
            }
            return;
          }
          const mesh = paintMesh ?? hit.mesh;
          let point: [number, number, number];
          try {
            point = localHit(mesh, hit);
          } catch {
            // Removed meanwhile: the next move asks again.
            return;
          }
          inspect.sculpt.hovering = true;
          showBall(mesh, point);
          if (active !== -1 && paintMesh !== null) {
            await enqueuePaint(paintMesh, point);
          }
        })
        .catch(() => {
          // Removed meanwhile: the next move asks again.
        })
        .finally(() => {
          running = false;
          if (alive) {
            run();
          }
        });
    };

    const onMove = (e: PointerEvent) => {
      if (active !== -1 && e.pointerId !== active) {
        return;
      }
      pending = { x: e.clientX, y: e.clientY, period: active !== -1 ? STROKE_PERIOD_MS : HOVER_PERIOD_MS };
      run();
    };
    const onDown = (e: PointerEvent) => {
      if (e.button !== 0 || isExtraFinger(e) || active !== -1 || view.dragging) {
        return;
      }
      const altPressed = e.altKey;
      pressing = e.pointerId;
      // Immediate: the paint starts where the button lands, not a throttle later.
      Promise.resolve()
        .then(() => cast(e.clientX, e.clientY))
        .then(
          (hit) => {
            if (!alive || active !== -1 || view.dragging || view.multiTouch) {
              return;
            }
            if (hit === null) {
              return;
            }
            if (altPressed) {
              if (!isEligibleMesh(hit.mesh)) {
                return;
              }
              view.pressTaken = true;
              applyAltPinClick({
                app,
                select: selectRef.current,
                selection: selRef.current,
                hitMesh: hit.mesh,
                pinToActive: pinRef.current,
              });
              return;
            }
            if (!canMaskTarget(hit.mesh)) {
              return;
            }
            const mesh = hit.mesh;
            let point: [number, number, number];
            try {
              point = localHit(mesh, hit);
            } catch {
              return;
            }
            inspect.sculpt.hovering = true;
            showBall(mesh, point);
            // Released meanwhile (a long frame runs the down and the up together): the
            // press still paints where it landed, but nothing would end a paint left open.
            if (pressing !== e.pointerId) {
              void enqueuePaint(mesh, point);
              return;
            }
            active = e.pointerId;
            // Its release is not a click on whatever is behind the mesh.
            view.pressTaken = true;
            paintMesh = mesh;
            if (view.orbit !== null) {
              view.orbit.enabled = false;
            }
            void enqueuePaint(mesh, point);
          },
          () => {
            // Removed meanwhile: the next press asks again.
          },
        );
    };
    const onUp = (e: PointerEvent) => {
      pressing = -1;
      if (e.pointerId === active) {
        endPaint();
      }
    };
    const onLeave = (e: PointerEvent) => {
      pressing = -1;
      if (active !== -1) {
        if (e.pointerId === active) {
          endPaint();
        }
        return;
      }
      pending = null;
      hideBall();
    };
    canvas.addEventListener("pointermove", onMove);
    canvas.addEventListener("pointerdown", onDown);
    canvas.addEventListener("pointerup", onUp);
    canvas.addEventListener("pointercancel", onUp);
    canvas.addEventListener("pointerleave", onLeave);
    return () => {
      alive = false;
      clearTimeout(timer);
      canvas.removeEventListener("pointermove", onMove);
      canvas.removeEventListener("pointerdown", onDown);
      canvas.removeEventListener("pointerup", onUp);
      canvas.removeEventListener("pointercancel", onUp);
      canvas.removeEventListener("pointerleave", onLeave);
      endPaint();
      hideBall();
    };
  }, [scene, store, camera, gl, ball, app, maskTracker]);
}
