// The viewport deform tools' pointer handling (#193): dragging an FFD cage
// control point on a camera-facing plane, and the ARAP Pin / Handle, XPBD Pin
// and Kelvinlet Drag tools on the active mesh's surface. A drag solves once,
// on release. Presses that miss leave OrbitControls alone.
import type { RootState } from "@react-three/fiber";
import type { FfdLatticeData, MeshHandle } from "fast-mesh-wasm";
import type { SceneStore } from "fast-mesh-wasm/store";
import { type RefObject, useEffect, useState } from "react";
import * as THREE from "three";
import type { AppState } from "../../../app/state/app-state";
import type { Wave6DeformState } from "../../../app/state/wave6-state";
import { isExtraFinger } from "../../../viewport/ux/touch";
import { setDragging, view } from "../../../viewport/ux/view";
import { dragProtection } from "./deform-protection";
import {
  arapOptions,
  buildKelvinletLoadFromDrag,
  formatError,
  formatFreeformReport,
  kelvinletOptions,
  requireDeformDomain,
} from "./deform-solve";
import { applyDisplacementToLattice, resetLatticeOffsets, type Vec3Tuple } from "./ffd-lattice";
import type { SculptMask } from "./sculpt";
import { patchDeform, updateDeform } from "./use-deform-state";

/** The Kelvinlet pull being dragged, from the grab point to the pointer, in mesh-local space. */
export interface KelvinletPreview {
  start: Vec3Tuple;
  end: Vec3Tuple;
}

/** What the drags need from the R3F root. */
export type DeformDragThree = Pick<RootState, "camera" | "gl" | "raycaster" | "invalidate">;

/** What both drag hooks read: the layer's refs stay the layer's, read at event time. */
export interface DeformDragInput {
  app: AppState;
  store: SceneStore;
  activeTool: Wave6DeformState["activeTool"];
  activeMesh: MeshHandle | null;
  /** The current `wave6.deform`, refreshed every render. */
  deformRef: RefObject<Wave6DeformState>;
  /** The current sculpt mask, refreshed every render. */
  maskRef: RefObject<SculptMask | null>;
  three: DeformDragThree;
}

function setNdcFromPointer(ev: PointerEvent, element: HTMLElement, target: THREE.Vector2): void {
  const rect = element.getBoundingClientRect();
  const w = rect.width || 1;
  const h = rect.height || 1;
  target.set(((ev.clientX - rect.left) / w) * 2 - 1, -((ev.clientY - rect.top) / h) * 2 + 1);
}

function nearestTriangleVertex(
  hit: THREE.Intersection,
  positions: ArrayLike<number>,
  invMatrix: THREE.Matrix4,
): number | null {
  if (!hit.face) return null;
  const localHit = hit.point.clone().applyMatrix4(invMatrix);
  const candidates = [hit.face.a, hit.face.b, hit.face.c];
  let best = candidates[0];
  let bestDistSq = Infinity;
  for (const v of candidates) {
    const base = v * 3;
    if (base + 2 >= positions.length) continue;
    const dx = positions[base] - localHit.x;
    const dy = positions[base + 1] - localHit.y;
    const dz = positions[base + 2] - localHit.z;
    const d2 = dx * dx + dy * dy + dz * dz;
    if (d2 < bestDistSq) {
      bestDistSq = d2;
      best = v;
    }
  }
  return best;
}

/**
 * FFD Cage sphere pointer drag on camera-facing plane; preserves OrbitControls when inactive or clicking empty space.
 * Returns the hovered and the dragged control point, for highlighting.
 */
export function useFfdCageDrag(
  input: DeformDragInput & {
    hasLattice: boolean;
    /** The control point spheres, indexed like the lattice (`userData.cpIndex`). */
    cpMeshRefs: RefObject<Array<THREE.Mesh | null>>;
  },
): { hoveredCp: number | null; activeCp: number | null } {
  const { app, store, activeTool, activeMesh, hasLattice, cpMeshRefs, deformRef, maskRef } = input;
  const { camera, gl, raycaster, invalidate } = input.three;
  const [hoveredCp, setHoveredCp] = useState<number | null>(null);
  const [activeCp, setActiveCp] = useState<number | null>(null);

  useEffect(() => {
    if (activeTool !== "ffdCage" || activeMesh === null || !hasLattice) {
      setHoveredCp(null);
      setActiveCp(null);
      return;
    }
    const dom = gl.domElement;
    const ndc = new THREE.Vector2();
    const camDir = new THREE.Vector3();
    const hitWorld = new THREE.Vector3();
    const plane = new THREE.Plane();
    const invMatrix = new THREE.Matrix4();

    let draggingState: {
      cpIndex: number;
      startLattice: FfdLatticeData;
      startLocal: THREE.Vector3;
      invMat: THREE.Matrix4;
      dragPlane: THREE.Plane;
      lastDelta: Vec3Tuple;
      pointerId: number;
    } | null = null;
    let disposed = false;

    const releaseDrag = () => {
      if (view.orbit) {
        view.orbit.enabled = true;
      }
      setDragging(false);
      setActiveCp(null);
    };

    const onPointerDown = (e: PointerEvent) => {
      if (e.button !== 0 || isExtraFinger(e) || view.multiTouch) return;
      const targets = cpMeshRefs.current.filter((m): m is THREE.Mesh => m !== null);
      if (targets.length === 0) return;

      setNdcFromPointer(e, dom, ndc);
      raycaster.setFromCamera(ndc, camera);
      const hits = raycaster.intersectObjects(targets, false);
      if (hits.length === 0) return;

      const cpIndex = Number(hits[0].object.userData.cpIndex);
      if (!Number.isInteger(cpIndex)) return;
      const curLattice = deformRef.current.ffdLattice;
      if (!curLattice) return;

      e.stopPropagation();
      view.pressTaken = true;
      setDragging(true);
      setActiveCp(cpIndex);
      if (view.orbit) {
        view.orbit.enabled = false;
      }

      const startWorld = new THREE.Vector3();
      hits[0].object.getWorldPosition(startWorld);
      camera.getWorldDirection(camDir);
      plane.setFromNormalAndCoplanarPoint(camDir, startWorld);
      try {
        invMatrix.fromArray(store.scene.matrix4(activeMesh)).invert();
      } catch {
        invMatrix.identity();
      }
      const startLocal = startWorld.clone().applyMatrix4(invMatrix);

      draggingState = {
        cpIndex,
        startLattice: curLattice,
        startLocal,
        invMat: invMatrix.clone(),
        dragPlane: plane.clone(),
        lastDelta: [0, 0, 0],
        pointerId: e.pointerId,
      };
      dom.setPointerCapture?.(e.pointerId);
      invalidate();
    };

    const onPointerMove = (e: PointerEvent) => {
      if (!draggingState) {
        const targets = cpMeshRefs.current.filter((m): m is THREE.Mesh => m !== null);
        if (targets.length === 0) return;
        setNdcFromPointer(e, dom, ndc);
        raycaster.setFromCamera(ndc, camera);
        const hits = raycaster.intersectObjects(targets, false);
        const nextHover = hits.length > 0 ? Number(hits[0].object.userData.cpIndex) : null;
        setHoveredCp((prev) => (prev === nextHover ? prev : nextHover));
        return;
      }
      if (e.pointerId !== draggingState.pointerId) return;
      setNdcFromPointer(e, dom, ndc);
      raycaster.setFromCamera(ndc, camera);
      if (!raycaster.ray.intersectPlane(draggingState.dragPlane, hitWorld)) return;

      const hitLocal = hitWorld.clone().applyMatrix4(draggingState.invMat);
      const dx = hitLocal.x - draggingState.startLocal.x;
      const dy = hitLocal.y - draggingState.startLocal.y;
      const dz = hitLocal.z - draggingState.startLocal.z;
      draggingState.lastDelta = [dx, dy, dz];

      const baseIdx = draggingState.cpIndex * 3;
      const startOff: Vec3Tuple = [
        draggingState.startLattice.offsets[baseIdx] + dx,
        draggingState.startLattice.offsets[baseIdx + 1] + dy,
        draggingState.startLattice.offsets[baseIdx + 2] + dz,
      ];
      const nextLattice = applyDisplacementToLattice(draggingState.startLattice, draggingState.cpIndex, startOff, "set");
      patchDeform(app, { ffdLattice: nextLattice });
      invalidate();
    };

    const onPointerUp = (e: PointerEvent) => {
      if (!draggingState || e.pointerId !== draggingState.pointerId) return;
      const state = draggingState;
      draggingState = null;
      dom.releasePointerCapture?.(e.pointerId);
      releaseDrag();

      const [dx, dy, dz] = state.lastDelta;
      if (Math.hypot(dx, dy, dz) <= 1e-6) return;

      void (async () => {
        try {
          const domain = requireDeformDomain(store);
          const stepLattice = applyDisplacementToLattice(
            resetLatticeOffsets(state.startLattice),
            state.cpIndex,
            [dx, dy, dz],
            "set",
          );
          const protect = dragProtection(deformRef.current.xpbdProtectMask, maskRef.current, activeMesh);
          const report = await domain.applyFfd(activeMesh, {
            lattice: stepLattice,
            outsideMode: "clamp",
            protect,
          });
          store.commitFrame();
          if (!disposed) {
            patchDeform(app, { lastReport: formatFreeformReport("FFD Drag", report) });
          }
        } catch (err) {
          if (!disposed) {
            patchDeform(app, { lastReport: `FFD Drag failed: ${formatError(err)}` });
          }
        }
      })();
    };

    dom.addEventListener("pointerdown", onPointerDown, true);
    window.addEventListener("pointermove", onPointerMove);
    window.addEventListener("pointerup", onPointerUp);
    window.addEventListener("pointercancel", onPointerUp);
    return () => {
      disposed = true;
      dom.removeEventListener("pointerdown", onPointerDown, true);
      window.removeEventListener("pointermove", onPointerMove);
      window.removeEventListener("pointerup", onPointerUp);
      window.removeEventListener("pointercancel", onPointerUp);
      if (draggingState) {
        releaseDrag();
      }
    };
  }, [activeTool, activeMesh, hasLattice, app, store, gl, camera, raycaster, invalidate]);

  return { hoveredCp, activeCp };
}

/**
 * ARAP Pin / Handle / XPBD Pin / Kelvinlet Drag pointer tools on the active mesh surface.
 * Returns the Kelvinlet pull while one is dragged.
 */
export function useSurfaceDeformDrag(input: DeformDragInput): KelvinletPreview | null {
  const { app, store, activeTool, activeMesh, deformRef, maskRef } = input;
  const { camera, gl, raycaster, invalidate } = input.three;
  const [kelvinletPreview, setKelvinletPreview] = useState<KelvinletPreview | null>(null);

  useEffect(() => {
    const tool = activeTool;
    if (
      activeMesh === null ||
      (tool !== "arapPin" && tool !== "arapHandle" && tool !== "xpbdPin" && tool !== "kelvinletDrag")
    ) {
      setKelvinletPreview(null);
      return;
    }

    const dom = gl.domElement;
    const ndc = new THREE.Vector2();
    const camDir = new THREE.Vector3();
    const hitWorld = new THREE.Vector3();
    let disposed = false;

    let activeDrag:
      | {
          kind: "arapHandle";
          vertex: number;
          startLocal: Vec3Tuple;
          lastTarget: Vec3Tuple;
          invMat: THREE.Matrix4;
          plane: THREE.Plane;
          pointerId: number;
        }
      | {
          kind: "kelvinletDrag";
          startLocal: Vec3Tuple;
          lastLocal: Vec3Tuple;
          invMat: THREE.Matrix4;
          plane: THREE.Plane;
          pointerId: number;
        }
      | null = null;

    const releaseDrag = () => {
      if (view.orbit) {
        view.orbit.enabled = true;
      }
      setDragging(false);
    };

    const onPointerDown = (e: PointerEvent) => {
      if (e.button !== 0 || isExtraFinger(e) || view.multiTouch) return;
      const binding = view.bindings.get(activeMesh);
      if (!binding) return;

      setNdcFromPointer(e, dom, ndc);
      raycaster.setFromCamera(ndc, camera);
      const hits = raycaster.intersectObject(binding.object, false);
      if (hits.length === 0) return;

      const invMat = new THREE.Matrix4();
      try {
        invMat.fromArray(store.scene.matrix4(activeMesh)).invert();
      } catch {
        invMat.identity();
      }

      const firstHit = hits[0];
      const localHitVec = firstHit.point.clone().applyMatrix4(invMat);
      const startLocal: Vec3Tuple = [localHitVec.x, localHitVec.y, localHitVec.z];

      if (tool === "arapPin" || tool === "xpbdPin") {
        let positions: Float32Array;
        try {
          positions = store.scene.views(activeMesh).positions();
        } catch {
          return;
        }
        const slot = nearestTriangleVertex(firstHit, positions, invMat);
        if (slot === null) return;
        e.stopPropagation();
        view.pressTaken = true;
        updateDeform(app, (prev) => {
          const exists = prev.arapAnchors.includes(slot);
          const nextAnchors = exists ? prev.arapAnchors.filter((v) => v !== slot) : [...prev.arapAnchors, slot];
          return { arapAnchors: nextAnchors };
        });
        return;
      }

      if (tool === "arapHandle") {
        let positions: Float32Array;
        try {
          positions = store.scene.views(activeMesh).positions();
        } catch {
          return;
        }
        const slot = nearestTriangleVertex(firstHit, positions, invMat);
        if (slot === null) return;
        const b = slot * 3;
        const vPos: Vec3Tuple = [positions[b], positions[b + 1], positions[b + 2]];

        e.stopPropagation();
        view.pressTaken = true;
        setDragging(true);
        if (view.orbit) {
          view.orbit.enabled = false;
        }
        camera.getWorldDirection(camDir);
        const plane = new THREE.Plane().setFromNormalAndCoplanarPoint(camDir, firstHit.point);

        updateDeform(app, (prev) => {
          const rest = prev.arapHandles.filter((h) => h.vertex !== slot);
          return { arapHandles: [...rest, { vertex: slot, target: vPos }] };
        });

        activeDrag = {
          kind: "arapHandle",
          vertex: slot,
          startLocal: vPos,
          lastTarget: vPos,
          invMat,
          plane,
          pointerId: e.pointerId,
        };
        dom.setPointerCapture?.(e.pointerId);
        return;
      }

      // tool === "kelvinletDrag"
      e.stopPropagation();
      view.pressTaken = true;
      setDragging(true);
      if (view.orbit) {
        view.orbit.enabled = false;
      }
      camera.getWorldDirection(camDir);
      const plane = new THREE.Plane().setFromNormalAndCoplanarPoint(camDir, firstHit.point);
      activeDrag = {
        kind: "kelvinletDrag",
        startLocal,
        lastLocal: startLocal,
        invMat,
        plane,
        pointerId: e.pointerId,
      };
      setKelvinletPreview({ start: startLocal, end: startLocal });
      dom.setPointerCapture?.(e.pointerId);
    };

    const onPointerMove = (e: PointerEvent) => {
      if (!activeDrag || e.pointerId !== activeDrag.pointerId) return;
      setNdcFromPointer(e, dom, ndc);
      raycaster.setFromCamera(ndc, camera);
      if (!raycaster.ray.intersectPlane(activeDrag.plane, hitWorld)) return;

      const hitLocal = hitWorld.clone().applyMatrix4(activeDrag.invMat);
      const target: Vec3Tuple = [hitLocal.x, hitLocal.y, hitLocal.z];

      if (activeDrag.kind === "arapHandle") {
        activeDrag.lastTarget = target;
        const vertex = activeDrag.vertex;
        updateDeform(app, (prev) => ({
          arapHandles: prev.arapHandles.map((h) => (h.vertex === vertex ? { vertex, target } : h)),
        }));
      } else {
        activeDrag.lastLocal = target;
        setKelvinletPreview({ start: activeDrag.startLocal, end: target });
      }
      invalidate();
    };

    const onPointerUp = (e: PointerEvent) => {
      if (!activeDrag || e.pointerId !== activeDrag.pointerId) return;
      const drag = activeDrag;
      activeDrag = null;
      dom.releasePointerCapture?.(e.pointerId);
      releaseDrag();

      if (drag.kind === "arapHandle") {
        const dx = drag.lastTarget[0] - drag.startLocal[0];
        const dy = drag.lastTarget[1] - drag.startLocal[1];
        const dz = drag.lastTarget[2] - drag.startLocal[2];
        if (Math.hypot(dx, dy, dz) <= 1e-5) return;

        void (async () => {
          try {
            const domain = requireDeformDomain(store);
            const cur = deformRef.current;
            const protect = dragProtection(cur.xpbdProtectMask, maskRef.current, activeMesh);
            const report = await domain.applyArap(activeMesh, arapOptions(cur, 15, protect));
            store.commitFrame();
            if (!disposed) {
              patchDeform(app, { lastReport: formatFreeformReport("ARAP Drag", report) });
            }
          } catch (err) {
            if (!disposed) {
              patchDeform(app, { lastReport: `ARAP Drag failed: ${formatError(err)}` });
            }
          }
        })();
        return;
      }

      // Finish kelvinlet drag
      setKelvinletPreview(null);
      const delta: Vec3Tuple = [
        drag.lastLocal[0] - drag.startLocal[0],
        drag.lastLocal[1] - drag.startLocal[1],
        drag.lastLocal[2] - drag.startLocal[2],
      ];
      if (Math.hypot(delta[0], delta[1], delta[2]) <= 1e-5) return;

      void (async () => {
        try {
          const domain = requireDeformDomain(store);
          const cur = deformRef.current;
          const protect = dragProtection(cur.xpbdProtectMask, maskRef.current, activeMesh);
          const load = buildKelvinletLoadFromDrag(cur.kelvinletKind, delta);
          const report = await domain.applyKelvinlet(activeMesh, kelvinletOptions(cur, drag.startLocal, load, protect));
          store.commitFrame();
          if (!disposed) {
            patchDeform(app, { lastReport: formatFreeformReport("Kelvinlet Drag", report) });
          }
        } catch (err) {
          if (!disposed) {
            patchDeform(app, { lastReport: `Kelvinlet Drag failed: ${formatError(err)}` });
          }
        }
      })();
    };

    dom.addEventListener("pointerdown", onPointerDown, true);
    window.addEventListener("pointermove", onPointerMove);
    window.addEventListener("pointerup", onPointerUp);
    window.addEventListener("pointercancel", onPointerUp);
    return () => {
      disposed = true;
      dom.removeEventListener("pointerdown", onPointerDown, true);
      window.removeEventListener("pointermove", onPointerMove);
      window.removeEventListener("pointerup", onPointerUp);
      window.removeEventListener("pointercancel", onPointerUp);
      if (activeDrag) {
        releaseDrag();
      }
    };
  }, [activeTool, activeMesh, app, store, gl, camera, raycaster, invalidate]);

  return kelvinletPreview;
}
