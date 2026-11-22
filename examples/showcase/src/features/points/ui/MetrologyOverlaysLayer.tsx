// 3D viewport overlays for #234: active packed point cloud (`THREE.Points`),
// sliced contour loops (`THREE.LineSegments`), and fitted primitive datum guides.
import "@react-three/fiber";
import type {
  ContourSlice,
  FittedPrimitiveReport,
  PackedPointCloud,
  Vec3,
} from "fast-mesh-wasm";
import type { SceneStore } from "fast-mesh-wasm/store";
import { useEffect, useLayoutEffect, useMemo } from "react";
import * as THREE from "three";
import { type AppState, useApp } from "../../../app/state/app-state";
import { unpackPointCloudLocal } from "../../../engine/wave6-wasm-lazy";
import { type Wave6MetrologyState } from "../../../app/state/wave6-state";

export interface MetrologyOverlaysLayerProps {
  app: AppState;
  store: SceneStore;
}

const INITIAL_METROLOGY: Wave6MetrologyState = {
  fittedPrimitive: null,
  showPrimitiveOverlay: true,
  contourSlices: null,
  showContoursOverlay: true,
  pointCloud: null,
  showPointCloudOverlay: true,
  pointCloudReport: null,
};
const RING_SEGMENTS = 32;

function emptyGeometry(): THREE.BufferGeometry {
  const geo = new THREE.BufferGeometry();
  geo.setAttribute("position", new THREE.Float32BufferAttribute(new Float32Array(0), 3));
  return geo;
}

function normalizeColorBuffer(raw: Float32Array): Float32Array {
  let maxVal = 0;
  for (let i = 0; i < raw.length; i++) {
    if (raw[i] > maxVal) {
      maxVal = raw[i];
    }
  }
  if (maxVal <= 1.01) {
    return raw;
  }
  const scaled = new Float32Array(raw.length);
  for (let i = 0; i < raw.length; i++) {
    scaled[i] = Math.min(1, Math.max(0, raw[i] / 255));
  }
  return scaled;
}

/** Builds a `THREE.BufferGeometry` holding positions (and optional normals/colors) for a `PackedPointCloud`. */
export function buildPointCloudGeometry(cloud: PackedPointCloud | null): THREE.BufferGeometry {
  if (cloud === null || cloud.count === 0) {
    return emptyGeometry();
  }
  const { positions, normals, colors } = unpackPointCloudLocal(cloud);
  const geo = new THREE.BufferGeometry();
  geo.setAttribute("position", new THREE.Float32BufferAttribute(positions, 3));
  if (normals !== null) {
    geo.setAttribute("normal", new THREE.Float32BufferAttribute(normals, 3));
  }
  if (colors !== null) {
    geo.setAttribute("color", new THREE.Float32BufferAttribute(normalizeColorBuffer(colors), 3));
  }
  geo.computeBoundingSphere();
  return geo;
}

/** Builds `THREE.LineSegments` geometry from one or more `ContourSlice` loop lists. */
export function buildContourSegmentsGeometry(slices: readonly ContourSlice[] | null): THREE.BufferGeometry {
  if (slices === null || slices.length === 0) {
    return emptyGeometry();
  }
  const verts: number[] = [];
  for (const slice of slices) {
    for (const loop of slice.loops) {
      const pts = loop.points3d;
      const n = Math.floor(pts.length / 3);
      if (n < 2) {
        continue;
      }
      for (let i = 0; i < n - 1; i++) {
        const a = 3 * i;
        const b = 3 * (i + 1);
        verts.push(pts[a], pts[a + 1], pts[a + 2], pts[b], pts[b + 1], pts[b + 2]);
      }
      if (loop.closed) {
        const last = 3 * (n - 1);
        verts.push(pts[last], pts[last + 1], pts[last + 2], pts[0], pts[1], pts[2]);
      }
    }
  }
  if (verts.length === 0) {
    return emptyGeometry();
  }
  const geo = new THREE.BufferGeometry();
  geo.setAttribute("position", new THREE.Float32BufferAttribute(Float32Array.from(verts), 3));
  geo.computeBoundingSphere();
  return geo;
}

function buildBasis(axis: Vec3): { u: THREE.Vector3; v: THREE.Vector3; n: THREE.Vector3 } {
  const n = new THREE.Vector3(axis[0], axis[1], axis[2]);
  if (n.lengthSq() < 1e-12) {
    n.set(0, 0, 1);
  } else {
    n.normalize();
  }
  const helper = Math.abs(n.y) < 0.9 ? new THREE.Vector3(0, 1, 0) : new THREE.Vector3(1, 0, 0);
  const u = new THREE.Vector3().crossVectors(helper, n).normalize();
  const v = new THREE.Vector3().crossVectors(n, u).normalize();
  return { u, v, n };
}

function pushSeg(out: number[], a: THREE.Vector3, b: THREE.Vector3): void {
  out.push(a.x, a.y, a.z, b.x, b.y, b.z);
}

function pushRing(
  out: number[],
  center: THREE.Vector3,
  u: THREE.Vector3,
  v: THREE.Vector3,
  radius: number,
  segments = RING_SEGMENTS,
): void {
  const r = Math.max(1e-4, radius);
  const prev = new THREE.Vector3();
  const next = new THREE.Vector3();
  for (let i = 0; i < segments; i++) {
    const t0 = (i / segments) * Math.PI * 2;
    const t1 = ((i + 1) / segments) * Math.PI * 2;
    prev.copy(center).addScaledVector(u, Math.cos(t0) * r).addScaledVector(v, Math.sin(t0) * r);
    next.copy(center).addScaledVector(u, Math.cos(t1) * r).addScaledVector(v, Math.sin(t1) * r);
    pushSeg(out, prev, next);
  }
}

/** Builds wireframe datum guide segments (`THREE.LineSegments`) for a `FittedPrimitiveReport`. */
export function buildPrimitiveGuideSegments(report: FittedPrimitiveReport | null): THREE.BufferGeometry {
  if (report === null) {
    return emptyGeometry();
  }
  const verts: number[] = [];
  const p = report.primitive;

  switch (p.kind) {
    case "plane": {
      const origin = new THREE.Vector3(p.origin[0], p.origin[1], p.origin[2]);
      const { u, v, n } = buildBasis(p.normal);
      const half = 1.0;
      const c00 = origin.clone().addScaledVector(u, -half).addScaledVector(v, -half);
      const c10 = origin.clone().addScaledVector(u, half).addScaledVector(v, -half);
      const c11 = origin.clone().addScaledVector(u, half).addScaledVector(v, half);
      const c01 = origin.clone().addScaledVector(u, -half).addScaledVector(v, half);
      pushSeg(verts, c00, c10);
      pushSeg(verts, c10, c11);
      pushSeg(verts, c11, c01);
      pushSeg(verts, c01, c00);
      pushSeg(verts, origin.clone().addScaledVector(u, -half), origin.clone().addScaledVector(u, half));
      pushSeg(verts, origin.clone().addScaledVector(v, -half), origin.clone().addScaledVector(v, half));
      pushSeg(verts, origin, origin.clone().addScaledVector(n, half));
      break;
    }
    case "sphere": {
      const center = new THREE.Vector3(p.center[0], p.center[1], p.center[2]);
      const r = Math.max(1e-4, p.radius);
      const x = new THREE.Vector3(1, 0, 0);
      const y = new THREE.Vector3(0, 1, 0);
      const z = new THREE.Vector3(0, 0, 1);
      pushRing(verts, center, x, y, r);
      pushRing(verts, center, x, z, r);
      pushRing(verts, center, y, z, r);
      const crossLen = r * 0.25;
      pushSeg(verts, center.clone().addScaledVector(x, -crossLen), center.clone().addScaledVector(x, crossLen));
      pushSeg(verts, center.clone().addScaledVector(y, -crossLen), center.clone().addScaledVector(y, crossLen));
      pushSeg(verts, center.clone().addScaledVector(z, -crossLen), center.clone().addScaledVector(z, crossLen));
      break;
    }
    case "cylinder": {
      const origin = new THREE.Vector3(p.axisOrigin[0], p.axisOrigin[1], p.axisOrigin[2]);
      const { u, v, n } = buildBasis(p.axisDir);
      const r = Math.max(1e-4, p.radius);
      const halfH = Math.max(r, p.height > 0 ? p.height * 0.5 : r);
      const bot = origin.clone().addScaledVector(n, -halfH);
      const top = origin.clone().addScaledVector(n, halfH);
      pushSeg(verts, bot, top);
      pushRing(verts, bot, u, v, r);
      pushRing(verts, top, u, v, r);
      for (const dir of [u, u.clone().negate(), v, v.clone().negate()]) {
        pushSeg(verts, bot.clone().addScaledVector(dir, r), top.clone().addScaledVector(dir, r));
      }
      break;
    }
    case "cone": {
      const apex = new THREE.Vector3(p.apex[0], p.apex[1], p.apex[2]);
      const { u, v, n } = buildBasis(p.axisDir);
      const h = Math.max(1e-3, p.height > 0 ? p.height : 1.0);
      const baseR = Math.max(1e-4, h * Math.tan(Math.min(Math.PI * 0.48, Math.max(1e-3, p.halfAngleRad))));
      const baseCenter = apex.clone().addScaledVector(n, h);
      pushSeg(verts, apex, baseCenter);
      pushRing(verts, baseCenter, u, v, baseR);
      for (const dir of [u, u.clone().negate(), v, v.clone().negate()]) {
        pushSeg(verts, apex, baseCenter.clone().addScaledVector(dir, baseR));
      }
      break;
    }
  }

  const geo = new THREE.BufferGeometry();
  geo.setAttribute("position", new THREE.Float32BufferAttribute(Float32Array.from(verts), 3));
  geo.computeBoundingSphere();
  return geo;
}

/** R3F viewport overlay layer for Wave 6 point clouds, contour slices, and fitted datum guides. */
export function MetrologyOverlaysLayer({ app, store }: MetrologyOverlaysLayerProps) {
  void store;
  const metrology = useApp(app, (s) => s.wave6?.metrology ?? INITIAL_METROLOGY);

  const pointsObj = useMemo(
    () =>
      new THREE.Points(
        emptyGeometry(),
        new THREE.PointsMaterial({ size: 0.04, sizeAttenuation: true, color: 0x38bdf8, vertexColors: false }),
      ),
    [],
  );
  const contourLinesObj = useMemo(
    () =>
      new THREE.LineSegments(
        emptyGeometry(),
        new THREE.LineBasicMaterial({ color: 0xf59e0b }),
      ),
    [],
  );
  const primitiveLinesObj = useMemo(
    () =>
      new THREE.LineSegments(
        emptyGeometry(),
        new THREE.LineBasicMaterial({ color: 0x22c55e }),
      ),
    [],
  );

  useEffect(
    () => () => {
      pointsObj.geometry.dispose();
      (pointsObj.material as THREE.Material).dispose();
      contourLinesObj.geometry.dispose();
      (contourLinesObj.material as THREE.Material).dispose();
      primitiveLinesObj.geometry.dispose();
      (primitiveLinesObj.material as THREE.Material).dispose();
    },
    [pointsObj, contourLinesObj, primitiveLinesObj],
  );

  useLayoutEffect(() => {
    const nextGeo = buildPointCloudGeometry(metrology.pointCloud);
    const prevGeo = pointsObj.geometry;
    pointsObj.geometry = nextGeo;
    prevGeo.dispose();

    const mat = pointsObj.material as THREE.PointsMaterial;
    const hasColors = Boolean(metrology.pointCloud?.hasColors);
    if (mat.vertexColors !== hasColors) {
      mat.vertexColors = hasColors;
      mat.color.setHex(hasColors ? 0xffffff : 0x38bdf8);
      mat.needsUpdate = true;
    }
    const radius = nextGeo.boundingSphere?.radius ?? 1;
    mat.size = Math.max(0.015, Math.min(2, radius * 0.025));
    pointsObj.visible = metrology.showPointCloudOverlay && metrology.pointCloud !== null && metrology.pointCloud.count > 0;
  }, [pointsObj, metrology.pointCloud, metrology.showPointCloudOverlay]);

  useLayoutEffect(() => {
    const nextGeo = buildContourSegmentsGeometry(metrology.contourSlices);
    const prevGeo = contourLinesObj.geometry;
    contourLinesObj.geometry = nextGeo;
    prevGeo.dispose();
    contourLinesObj.visible =
      metrology.showContoursOverlay && metrology.contourSlices !== null && metrology.contourSlices.length > 0;
  }, [contourLinesObj, metrology.contourSlices, metrology.showContoursOverlay]);

  useLayoutEffect(() => {
    const nextGeo = buildPrimitiveGuideSegments(metrology.fittedPrimitive);
    const prevGeo = primitiveLinesObj.geometry;
    primitiveLinesObj.geometry = nextGeo;
    prevGeo.dispose();
    primitiveLinesObj.visible = metrology.showPrimitiveOverlay && metrology.fittedPrimitive !== null;
  }, [primitiveLinesObj, metrology.fittedPrimitive, metrology.showPrimitiveOverlay]);

  return (
    <group name="metrology-overlays">
      <primitive object={pointsObj} />
      <primitive object={contourLinesObj} />
      <primitive object={primitiveLinesObj} />
    </group>
  );
}
