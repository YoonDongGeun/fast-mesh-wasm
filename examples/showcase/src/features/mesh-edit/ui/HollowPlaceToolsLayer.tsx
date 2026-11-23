// The print-prep viewport layer (#236): the drain hole or stamp preview
// that stands on the surface under the pointer while a tool is armed (a
// text stamp also shows its letters' outline, #260), and the center-surface
// hit Punch at Center uses. The pointer handling and the ops live in `ux/`;
// this layer hands them the viewport's surface raycast.
import type { MeshHandle, Scene, SceneLike, Vec3 } from "fast-mesh-wasm";
import type { SceneStore } from "fast-mesh-wasm/store";
import type { WorkerScene } from "fast-mesh-wasm/worker";
import { useEffect, useMemo } from "react";
import * as THREE from "three";
import { type AppState, useApp } from "../../../app/state/app-state";
import { raycastVisibleSurface } from "../../../viewport/ux/surface-raycast";
import { computeHitPointAndWorldNormal } from "../ux/surface-orientation";
import { loadTextFont } from "../ux/text-font";
import { textOutlineSegments, textParamsProblem } from "../ux/text-mesh";
import {
  type PrintPrepPreviewPose,
  type PrintPrepPreviewTarget,
  usePrintPrepPointerTool,
} from "../ux/use-print-prep-pointer";

export interface HollowPlaceToolsLayerProps {
  app: AppState;
  store: SceneStore;
}

/**
 * Raycasts the target mesh's +Z face center in world space, falling back to its top bounding center.
 */
export async function findCenterSurfaceHit(
  scene: SceneLike,
  mesh: MeshHandle,
): Promise<{ point: Vec3; normal: Vec3 } | null> {
  const views = scene.views(mesh);
  const positions = views.positions();
  const vCount = positions.length / 3;
  if (vCount === 0) {
    return null;
  }
  const m4 = new THREE.Matrix4().fromArray(Array.from(scene.matrix4(mesh)));
  const v = new THREE.Vector3();
  let minX = Infinity;
  let minY = Infinity;
  let minZ = Infinity;
  let maxX = -Infinity;
  let maxY = -Infinity;
  let maxZ = -Infinity;
  for (let i = 0; i < vCount; i++) {
    v.set(positions[i * 3], positions[i * 3 + 1], positions[i * 3 + 2]).applyMatrix4(m4);
    if (v.x < minX) minX = v.x;
    if (v.y < minY) minY = v.y;
    if (v.z < minZ) minZ = v.z;
    if (v.x > maxX) maxX = v.x;
    if (v.y > maxY) maxY = v.y;
    if (v.z > maxZ) maxZ = v.z;
  }
  const cx = (minX + maxX) * 0.5;
  const cy = (minY + maxY) * 0.5;
  const spanZ = Math.max(1, maxZ - minZ);
  const hit = await raycastVisibleSurface(
    scene as Scene | WorkerScene,
    [cx, cy, maxZ + spanZ],
    [0, 0, -1],
    (candidate) => candidate === mesh,
  );
  if (hit !== null) {
    const resolved = computeHitPointAndWorldNormal(scene, hit);
    if (resolved !== null) {
      return resolved;
    }
  }
  return { point: [cx, cy, maxZ], normal: [0, 0, 1] };
}

function createCircleLoopGeometry(radius: number, segments = 48): THREE.BufferGeometry {
  const pts: number[] = [];
  for (let i = 0; i < segments; i++) {
    const t = (2 * Math.PI * i) / segments;
    pts.push(radius * Math.cos(t), radius * Math.sin(t), 0);
  }
  const geo = new THREE.BufferGeometry();
  geo.setAttribute("position", new THREE.Float32BufferAttribute(pts, 3));
  return geo;
}

function createLineGeometry(points: number[]): THREE.BufferGeometry {
  const geo = new THREE.BufferGeometry();
  geo.setAttribute("position", new THREE.Float32BufferAttribute(points, 3));
  return geo;
}
/** The preview's three.js objects, and how a pose shows on them. */
interface PrintPrepPreview extends PrintPrepPreviewTarget {
  root: THREE.Group;
  /** A text stamp's outline (segments at z = 0, unscaled) and its radius, or none. */
  setOutline(outline: { segments: Float32Array; radius: number } | null): void;
  dispose(): void;
}

function createPrintPrepPreview(): PrintPrepPreview {
  const root = new THREE.Group();
  root.visible = false;
  root.renderOrder = 12;

  const ringGeo = createCircleLoopGeometry(1, 48);
  const ringMat = new THREE.LineBasicMaterial({
    color: "#38bdf8",
    depthTest: false,
    depthWrite: false,
  });
  const ring = new THREE.LineLoop(ringGeo, ringMat);
  ring.renderOrder = 12;
  root.add(ring);

  const boreGeo = new THREE.CylinderGeometry(1, 1, 1, 24, 1, true);
  boreGeo.rotateX(Math.PI / 2);
  boreGeo.translate(0, 0, -0.5);
  const boreMat = new THREE.MeshBasicMaterial({
    color: "#38bdf8",
    transparent: true,
    opacity: 0.25,
    side: THREE.DoubleSide,
    depthTest: false,
    depthWrite: false,
  });
  const bore = new THREE.Mesh(boreGeo, boreMat);
  bore.renderOrder = 12;
  root.add(bore);

  const normalGeo = createLineGeometry([0, 0, 0, 0, 0, 1]);
  const normalMat = new THREE.LineBasicMaterial({
    color: "#38bdf8",
    depthTest: false,
    depthWrite: false,
  });
  const normalLine = new THREE.Line(normalGeo, normalMat);
  normalLine.renderOrder = 12;
  root.add(normalLine);

  const xAxisGeo = createLineGeometry([0, 0, 0, 1, 0, 0]);
  const xAxisMat = new THREE.LineBasicMaterial({
    color: "#ef4444",
    depthTest: false,
    depthWrite: false,
  });
  const xAxis = new THREE.Line(xAxisGeo, xAxisMat);
  xAxis.renderOrder = 12;
  root.add(xAxis);

  const yAxisGeo = createLineGeometry([0, 0, 0, 0, 1, 0]);
  const yAxisMat = new THREE.LineBasicMaterial({
    color: "#22c55e",
    depthTest: false,
    depthWrite: false,
  });
  const yAxis = new THREE.Line(yAxisGeo, yAxisMat);
  yAxis.renderOrder = 12;
  root.add(yAxis);

  const outlineGeo = new THREE.BufferGeometry();
  const outlineMat = new THREE.LineBasicMaterial({
    color: "#f59e0b",
    depthTest: false,
    depthWrite: false,
  });
  const outline = new THREE.LineSegments(outlineGeo, outlineMat);
  outline.renderOrder = 12;
  outline.visible = false;
  root.add(outline);
  let outlineRadius: number | null = null;

  return {
    root,
    setOutline(next) {
      outlineGeo.setAttribute("position", new THREE.Float32BufferAttribute(next?.segments ?? new Float32Array(0), 3));
      outlineGeo.computeBoundingSphere();
      outlineRadius = next === null || next.segments.length === 0 ? null : next.radius;
      if (outlineRadius === null) {
        outline.visible = false;
      }
    },
    show(pose: PrintPrepPreviewPose) {
      root.position.set(pose.position[0], pose.position[1], pose.position[2]);
      root.quaternion.set(pose.quaternion[0], pose.quaternion[1], pose.quaternion[2], pose.quaternion[3]);
      ring.scale.set(pose.ringRadius, pose.ringRadius, 1);
      if (pose.drain) {
        bore.visible = true;
        bore.scale.set(pose.ringRadius, pose.ringRadius, pose.depth);
        normalLine.scale.set(1, 1, pose.normalLength);
        xAxis.visible = false;
        yAxis.visible = false;
        ringMat.color.set("#38bdf8");
        normalMat.color.set("#38bdf8");
      } else {
        bore.visible = false;
        normalLine.scale.set(1, 1, pose.normalLength);
        xAxis.visible = true;
        xAxis.scale.set(pose.axisLength, 1, 1);
        yAxis.visible = true;
        yAxis.scale.set(1, pose.axisLength, 1);
        ringMat.color.set("#f59e0b");
        normalMat.color.set("#f59e0b");
      }
      outline.visible = !pose.drain && outlineRadius !== null;
      if (outline.visible && outlineRadius !== null) {
        outline.scale.setScalar(pose.stampScale);
        ring.scale.set(outlineRadius * pose.stampScale, outlineRadius * pose.stampScale, 1);
        xAxis.scale.set(outlineRadius * pose.stampScale * 0.85, 1, 1);
        yAxis.scale.set(1, outlineRadius * pose.stampScale * 0.85, 1);
      }
      root.visible = true;
    },
    hide() {
      root.visible = false;
    },
    dispose() {
      ringGeo.dispose();
      ringMat.dispose();
      boreGeo.dispose();
      boreMat.dispose();
      normalGeo.dispose();
      normalMat.dispose();
      xAxisGeo.dispose();
      xAxisMat.dispose();
      yAxisGeo.dispose();
      yAxisMat.dispose();
      outlineGeo.dispose();
      outlineMat.dispose();
    },
  };
}

export function HollowPlaceToolsLayer({ app, store }: HollowPlaceToolsLayerProps) {
  const printPrep = useApp(app, (s) => s.wave6.printPrep);
  const activeTool = printPrep.activeTool;
  const isToolActive = activeTool === "drainHole" || activeTool === "placeStamp";

  const preview = useMemo(() => createPrintPrepPreview(), []);

  useEffect(() => {
    return () => {
      preview.dispose();
    };
  }, [preview]);

  // A text stamp shows its letters: the outline, once the font is in.
  const textStamp = activeTool === "placeStamp" && printPrep.placeParams.stampSource === "text";
  const text = printPrep.placeParams.text;
  useEffect(() => {
    if (!textStamp || textParamsProblem(text) !== null) {
      preview.setOutline(null);
      return;
    }
    let live = true;
    loadTextFont().then(
      (font) => {
        if (live) {
          preview.setOutline(textOutlineSegments(font, text));
        }
      },
      () => {
        if (live) {
          preview.setOutline(null);
        }
      },
    );
    return () => {
      live = false;
    };
  }, [preview, textStamp, text]);

  usePrintPrepPointerTool({ app, store, printPrep, active: isToolActive, preview, raycast: raycastVisibleSurface });

  return (
    <primitive
      object={preview.root}
      visible={isToolActive && printPrep.activeTool !== null && preview.root.visible}
    />
  );
}
