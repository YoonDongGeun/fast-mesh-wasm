// The print-prep viewport tool (#236): while Drain Hole or Place Stamp is
// armed, hovering raycasts the visible, unlocked meshes (never the scene
// stamp itself) and stands the preview on the surface, and a tap (not a
// drag, not a pinch) punches the hole or places the stamp there, one op at
// a time. The canvas listeners attach only while a tool is armed; the
// pointer state lives in refs, so hovering never re-renders.
import type { MeshHandle, Scene, Vec3 } from "fast-mesh-wasm";
import type { SceneStore } from "fast-mesh-wasm/store";
import type { WorkerScene } from "fast-mesh-wasm/worker";
import { useThree } from "@react-three/fiber";
import { useEffect, useRef } from "react";
import * as THREE from "three";
import { type AppState, useApp } from "../../../app/state/app-state";
import type { Wave6PrintPrepState } from "../../../app/state/wave6-state";
import { view } from "../../../viewport/ux/view";
import { executeDrainHoleAtHit, executePlaceStampAtHit, getPlaceAlignToNormal } from "./print-prep-ops";
import { computeHitPointAndWorldNormal, orientationFromSurfaceNormal } from "./surface-orientation";

function tapSlop(pointerType: string, fallback = 4): number {
  return pointerType === "touch" ? 10 : fallback;
}

/** The viewport's surface raycast: the first hit on a mesh `accept` takes, past the others. */
export type SurfaceRaycast = (
  scene: Scene | WorkerScene,
  origin: readonly [number, number, number],
  direction: readonly [number, number, number],
  accept: (mesh: MeshHandle) => boolean,
) => Promise<{ mesh: MeshHandle; triangle: number; point: Readonly<Vec3> } | null>;

/** How the preview stands on the surface for the armed tool. */
export interface PrintPrepPreviewPose {
  position: Vec3;
  quaternion: [number, number, number, number];
  /** A drain hole (ring, bore and normal); otherwise a stamp (ring, normal and in-plane axes). */
  drain: boolean;
  /** The hole's radius, or the stamp's footprint radius. */
  ringRadius: number;
  /** The bore's depth (drain holes only). */
  depth: number;
  /** The normal line's length. */
  normalLength: number;
  /** The in-plane axis lines' length (stamps only). */
  axisLength: number;
  /** The stamp's uniform scale (1 for drain holes): a text stamp's outline is drawn at it. */
  stampScale: number;
}

/** The preview the tool drives: shown at a pose, or hidden. */
export interface PrintPrepPreviewTarget {
  show(pose: PrintPrepPreviewPose): void;
  hide(): void;
}

/**
 * The preview pose at a surface `point` with outward `normal`: a stamp
 * stands upright (+Z) when it does not align to the normal, and spins by
 * its twist; sizes follow the drain or place parameters.
 */
export function printPrepPreviewPose(
  prep: Wave6PrintPrepState,
  point: Vec3,
  normal: Vec3,
  alignToNormal: boolean,
): PrintPrepPreviewPose {
  const tool = prep.activeTool;
  const effectiveNormal: Vec3 = tool === "placeStamp" && !alignToNormal ? [0, 0, 1] : normal;
  const twistRad = tool === "placeStamp" ? (prep.placeParams.twistDeg * Math.PI) / 180 : 0;
  const orient = orientationFromSurfaceNormal(effectiveNormal, twistRad);
  if (tool === "drainHole") {
    const radius = Math.max(0.25, prep.drainParams.diameter * 0.5);
    return {
      position: point,
      quaternion: orient.quaternion,
      drain: true,
      ringRadius: radius,
      depth: Math.max(1, prep.drainParams.depth),
      normalLength: Math.max(2, radius * 1.5),
      axisLength: 0,
      stampScale: 1,
    };
  }
  const scale = Math.max(0.25, prep.placeParams.uniformScale);
  const ringR = 2 * scale;
  return {
    position: point,
    quaternion: orient.quaternion,
    drain: false,
    ringRadius: ringR,
    depth: 0,
    normalLength: Math.max(2.5, ringR * 1.25),
    axisLength: ringR * 0.85,
    stampScale: Math.max(0.01, prep.placeParams.uniformScale),
  };
}

export function usePrintPrepPointerTool(args: {
  app: AppState;
  store: SceneStore;
  printPrep: Wave6PrintPrepState;
  /** Whether a viewport tool (drain hole, place stamp) is armed. */
  active: boolean;
  /** Stable for the layer's lifetime. */
  preview: PrintPrepPreviewTarget;
  raycast: SurfaceRaycast;
}): void {
  const { app, store, printPrep, active, preview, raycast } = args;
  const items = useApp(app, (s) => s.items);
  const { camera, gl, raycaster } = useThree();

  const busyRef = useRef(false);
  const downRef = useRef<{ x: number; y: number; pointerId: number } | null>(null);
  const printPrepRef = useRef(printPrep);
  printPrepRef.current = printPrep;
  const itemsRef = useRef(items);
  itemsRef.current = items;

  useEffect(() => {
    if (!active) {
      preview.hide();
      return;
    }

    const dom = gl.domElement;

    const raycastAtPointer = async (clientX: number, clientY: number) => {
      const rect = dom.getBoundingClientRect();
      if (rect.width <= 0 || rect.height <= 0) {
        return null;
      }
      const ndc = new THREE.Vector2(
        ((clientX - rect.left) / rect.width) * 2 - 1,
        -(((clientY - rect.top) / rect.height) * 2 - 1),
      );
      raycaster.setFromCamera(ndc, camera);
      const o = raycaster.ray.origin;
      const d = raycaster.ray.direction;

      const curItems = itemsRef.current;
      const prep = printPrepRef.current;
      const itemsByMesh = new Map<MeshHandle, (typeof curItems)[number]>();
      for (const item of curItems) {
        itemsByMesh.set(item.mesh, item);
      }
      const excludeStampId =
        prep.activeTool === "placeStamp" &&
        prep.placeParams.stampSource === "sceneMesh"
          ? prep.placeParams.stampMeshId
          : null;

      const meshStates = store.getSnapshot().meshes;
      const hit = await raycast(
        store.scene as Scene | WorkerScene,
        [o.x, o.y, o.z],
        [d.x, d.y, d.z],
        (mesh) => {
          const item = itemsByMesh.get(mesh);
          if (!item || item.locked || (meshStates.get(mesh)?.desired.visible ?? true) === false) {
            return false;
          }
          if (excludeStampId !== null && item.id === excludeStampId) {
            return false;
          }
          return true;
        },
      );
      if (hit === null) {
        return null;
      }
      const resolved = computeHitPointAndWorldNormal(store.scene, hit);
      if (resolved === null) {
        return null;
      }
      const item = itemsByMesh.get(hit.mesh) ?? null;
      return { mesh: hit.mesh, item, point: resolved.point, normal: resolved.normal };
    };

    const onPointerDown = (e: PointerEvent) => {
      if (e.button !== 0 || view.multiTouch) {
        downRef.current = null;
        return;
      }
      downRef.current = { x: e.clientX, y: e.clientY, pointerId: e.pointerId };
    };

    const onPointerMove = (e: PointerEvent) => {
      if (view.dragging || view.multiTouch || !e.isPrimary) {
        preview.hide();
        return;
      }
      void raycastAtPointer(e.clientX, e.clientY).then((res) => {
        if (res === null) {
          preview.hide();
          return;
        }
        preview.show(printPrepPreviewPose(printPrepRef.current, res.point, res.normal, getPlaceAlignToNormal(app)));
      });
    };

    const onPointerUp = (e: PointerEvent) => {
      const down = downRef.current;
      downRef.current = null;
      if (
        down === null ||
        down.pointerId !== e.pointerId ||
        e.button !== 0 ||
        view.dragging ||
        view.multiTouch ||
        busyRef.current
      ) {
        return;
      }
      const maxMove = tapSlop(e.pointerType, 4);
      if (Math.hypot(e.clientX - down.x, e.clientY - down.y) > maxMove) {
        return;
      }
      busyRef.current = true;
      void (async () => {
        try {
          const res = await raycastAtPointer(e.clientX, e.clientY);
          if (res === null) {
            return;
          }
          const prep = printPrepRef.current;
          if (prep.activeTool === "drainHole") {
            await executeDrainHoleAtHit({
              store,
              app,
              target: res.mesh,
              targetName: res.item?.name,
              hitPoint: res.point,
              hitNormal: res.normal,
              drainParams: prep.drainParams,
              through: prep.drainParams.depth <= 0,
            });
          } else if (prep.activeTool === "placeStamp") {
            await executePlaceStampAtHit({
              store,
              app,
              target: res.mesh,
              targetName: res.item?.name,
              hitPoint: res.point,
              hitNormal: res.normal,
              placeParams: prep.placeParams,
              alignToNormal: getPlaceAlignToNormal(app),
            });
          }
        } catch (err) {
          console.error("Print prep viewport tool action failed:", err);
        } finally {
          busyRef.current = false;
        }
      })();
    };

    const onPointerLeave = () => {
      preview.hide();
    };

    dom.addEventListener("pointerdown", onPointerDown);
    dom.addEventListener("pointermove", onPointerMove);
    dom.addEventListener("pointerup", onPointerUp);
    dom.addEventListener("pointerleave", onPointerLeave);
    return () => {
      dom.removeEventListener("pointerdown", onPointerDown);
      dom.removeEventListener("pointermove", onPointerMove);
      dom.removeEventListener("pointerup", onPointerUp);
      dom.removeEventListener("pointerleave", onPointerLeave);
      preview.hide();
    };
  }, [app, camera, gl, active, preview, raycaster, store, raycast]);
}
