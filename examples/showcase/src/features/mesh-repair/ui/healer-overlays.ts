/**
 * Extended 3D visual defect overlays & click-to-focus defect hotspots for the
 * Mesh Healer (#235). Complements `repair-overlays.ts` with self-intersecting
 * triangles, inverted shells / orientation conflicts, sliver faces, and
 * spatially bounded defect hotspots.
 */
import type { Analysis } from "fast-mesh-wasm";
import * as THREE from "three";
import type { Vec3 } from "../../../io/mesh-data";
import type { HealerOverlayKind, RepairOverlayKind } from "../../../app/state/app-state";
import { buildRepairOverlayGroup } from "./repair-overlays";

export const HEALER_OVERLAY_KINDS: readonly HealerOverlayKind[] = [
  "selfIntersections",
  "inverted",
  "slivers",
];

export const HEALER_OVERLAY_COLORS: Record<HealerOverlayKind, number> = {
  selfIntersections: 0xf97316, // vivid orange
  inverted: 0xf43f5e, // crimson rose
  slivers: 0xeab308, // warm gold
};

export interface DefectHotspot {
  id: string;
  kind: RepairOverlayKind | HealerOverlayKind;
  label: string;
  count: number;
  center: Vec3;
  radius: number;
}

export interface HealerOverlayCounts {
  selfIntersections: number;
  inverted: number;
  slivers: number;
}

const SLIVER_ASPECT_THRESHOLD = 20;
const MAX_DETECT_TRIS = 60_000;
const MAX_SI_HOTSPOT_SAMPLES = 256;

function trianglePoint(positions: Float32Array, vi: number): [number, number, number] {
  const o = vi * 3;
  return [positions[o] ?? 0, positions[o + 1] ?? 0, positions[o + 2] ?? 0];
}

function triangleBounds(
  positions: Float32Array,
  triIndices: Iterable<number>,
  indices: Uint32Array,
): { center: Vec3; radius: number } {
  let minX = Infinity;
  let minY = Infinity;
  let minZ = Infinity;
  let maxX = -Infinity;
  let maxY = -Infinity;
  let maxZ = -Infinity;
  let count = 0;
  for (const tri of triIndices) {
    const base = tri * 3;
    for (let k = 0; k < 3; k++) {
      const vi = indices[base + k] ?? 0;
      const o = vi * 3;
      const x = positions[o] ?? 0;
      const y = positions[o + 1] ?? 0;
      const z = positions[o + 2] ?? 0;
      if (x < minX) minX = x;
      if (y < minY) minY = y;
      if (z < minZ) minZ = z;
      if (x > maxX) maxX = x;
      if (y > maxY) maxY = y;
      if (z > maxZ) maxZ = z;
      count++;
    }
  }
  if (count === 0) {
    return { center: [0, 0, 0], radius: 0.1 };
  }
  const cx = (minX + maxX) * 0.5;
  const cy = (minY + maxY) * 0.5;
  const cz = (minZ + maxZ) * 0.5;
  const dx = maxX - minX;
  const dy = maxY - minY;
  const dz = maxZ - minZ;
  const radius = Math.max(Math.hypot(dx, dy, dz) * 0.5, 0.05);
  return { center: [cx, cy, cz], radius };
}

function vertexListBounds(positions: Float32Array, vertices: Iterable<number>): { center: Vec3; radius: number } {
  let minX = Infinity;
  let minY = Infinity;
  let minZ = Infinity;
  let maxX = -Infinity;
  let maxY = -Infinity;
  let maxZ = -Infinity;
  let count = 0;
  for (const vi of vertices) {
    const o = vi * 3;
    const x = positions[o] ?? 0;
    const y = positions[o + 1] ?? 0;
    const z = positions[o + 2] ?? 0;
    if (x < minX) minX = x;
    if (y < minY) minY = y;
    if (z < minZ) minZ = z;
    if (x > maxX) maxX = x;
    if (y > maxY) maxY = y;
    if (z > maxZ) maxZ = z;
    count++;
  }
  if (count === 0) {
    return { center: [0, 0, 0], radius: 0.1 };
  }
  return {
    center: [(minX + maxX) * 0.5, (minY + maxY) * 0.5, (minZ + maxZ) * 0.5],
    radius: Math.max(Math.hypot(maxX - minX, maxY - minY, maxZ - minZ) * 0.5, 0.05),
  };
}

/**
 * Returns triangle indices with extreme circumradius-to-2-inradius aspect ratio
 * (`>= maxAspectRatio`), skipping exact zero-area degenerate triangles already
 * captured by `Analysis.degenerateList`.
 */
export function detectSliverTriangles(
  positions: Float32Array,
  indices: Uint32Array,
  maxAspectRatio = SLIVER_ASPECT_THRESHOLD,
): Uint32Array {
  const triCount = Math.floor(indices.length / 3);
  if (triCount === 0 || triCount > MAX_DETECT_TRIS * 4) return new Uint32Array(0);
  const out: number[] = [];
  for (let t = 0; t < triCount; t++) {
    const b = t * 3;
    const ia = indices[b]!;
    const ib = indices[b + 1]!;
    const ic = indices[b + 2]!;
    if (ia === ib || ib === ic || ic === ia) continue;
    const [ax, ay, az] = trianglePoint(positions, ia);
    const [bx, by, bz] = trianglePoint(positions, ib);
    const [cx, cy, cz] = trianglePoint(positions, ic);
    const ab = Math.hypot(bx - ax, by - ay, bz - az);
    const bc = Math.hypot(cx - bx, cy - by, cz - bz);
    const ca = Math.hypot(ax - cx, ay - cy, az - cz);
    const s = (ab + bc + ca) * 0.5;
    const ux = bx - ax;
    const uy = by - ay;
    const uz = bz - az;
    const vx = cx - ax;
    const vy = cy - ay;
    const vz = cz - az;
    const cxp = uy * vz - uz * vy;
    const cyp = uz * vx - ux * vz;
    const czp = ux * vy - uy * vx;
    const area2 = Math.hypot(cxp, cyp, czp);
    const area = area2 * 0.5;
    if (!(area > 1e-18) || !(s > 1e-18)) continue;
    const aspect = (ab * bc * ca * s) / (8 * area * area);
    if (aspect >= maxAspectRatio) {
      out.push(t);
    }
  }
  return Uint32Array.from(out);
}

/**
 * Finds interior edges whose incident triangles traverse them in the same
 * directed winding (`u -> v` twice), indicating flipped normals / orientation
 * conflicts.
 */
export function detectOrientationConflictEdges(indices: Uint32Array): Uint32Array {
  const triCount = Math.floor(indices.length / 3);
  if (triCount === 0 || triCount > MAX_DETECT_TRIS * 4) return new Uint32Array(0);
  // Map undirected edge `(lo * stride + hi)` -> bitmask of observed orientations.
  // Bit 1: saw `(lo -> hi)`, bit 2: saw `(hi -> lo)`, bit 4: duplicate same-direction.
  const maxVert = indices.reduce((m, v) => (v > m ? v : m), 0) + 1;
  const edgeDirs = new Map<number, number>();
  const conflicts: number[] = [];
  const seenConflict = new Set<number>();

  for (let t = 0; t < triCount; t++) {
    const b = t * 3;
    const vs = [indices[b]!, indices[b + 1]!, indices[b + 2]!];
    for (let e = 0; e < 3; e++) {
      const u = vs[e]!;
      const v = vs[(e + 1) % 3]!;
      if (u === v) continue;
      const forward = u < v;
      const lo = forward ? u : v;
      const hi = forward ? v : u;
      const key = lo * maxVert + hi;
      const prev = edgeDirs.get(key) ?? 0;
      const bit = forward ? 1 : 2;
      if ((prev & bit) !== 0 && !seenConflict.has(key)) {
        seenConflict.add(key);
        conflicts.push(lo, hi);
      }
      edgeDirs.set(key, prev | bit);
    }
  }
  return Uint32Array.from(conflicts);
}

/**
 * Finds closed connected components whose signed volume is negative (inward-
 * facing normals across the entire shell).
 */
export function detectInvertedClosedComponents(
  analysis: Analysis,
  positions: Float32Array,
  indices: Uint32Array,
): Uint32Array {
  const invertedTris: number[] = [];
  for (const comp of analysis.components) {
    if (!comp.closed || comp.triangles.length === 0) continue;
    let vol6 = 0;
    for (let i = 0; i < comp.triangles.length; i++) {
      const t = comp.triangles[i]!;
      const b = t * 3;
      const [ax, ay, az] = trianglePoint(positions, indices[b]!);
      const [bx, by, bz] = trianglePoint(positions, indices[b + 1]!);
      const [cx, cy, cz] = trianglePoint(positions, indices[b + 2]!);
      vol6 +=
        ax * (by * cz - bz * cy) -
        ay * (bx * cz - bz * cx) +
        az * (bx * cy - by * cx);
    }
    if (vol6 < -1e-12) {
      for (let i = 0; i < comp.triangles.length; i++) {
        invertedTris.push(comp.triangles[i]!);
      }
    }
  }
  return Uint32Array.from(invertedTris);
}

function orient2d(ax: number, ay: number, bx: number, by: number, cx: number, cy: number): number {
  return (bx - ax) * (cy - ay) - (by - ay) * (cx - ax);
}

function pointInTri2d(
  px: number,
  py: number,
  ax: number,
  ay: number,
  bx: number,
  by: number,
  cx: number,
  cy: number,
): boolean {
  const s1 = orient2d(ax, ay, bx, by, px, py);
  const s2 = orient2d(bx, by, cx, cy, px, py);
  const s3 = orient2d(cx, cy, ax, ay, px, py);
  const eps = 1e-10;
  return (s1 >= -eps && s2 >= -eps && s3 >= -eps) || (s1 <= eps && s2 <= eps && s3 <= eps);
}

function segsIntersect2d(
  ax: number,
  ay: number,
  bx: number,
  by: number,
  cx: number,
  cy: number,
  dx: number,
  dy: number,
): boolean {
  const d1 = orient2d(cx, cy, dx, dy, ax, ay);
  const d2 = orient2d(cx, cy, dx, dy, bx, by);
  const d3 = orient2d(ax, ay, bx, by, cx, cy);
  const d4 = orient2d(ax, ay, bx, by, dx, dy);
  return ((d1 > 0 && d2 < 0) || (d1 < 0 && d2 > 0)) && ((d3 > 0 && d4 < 0) || (d3 < 0 && d4 > 0));
}

function coplanarTriIntersect(
  p0: [number, number, number],
  p1: [number, number, number],
  p2: [number, number, number],
  q0: [number, number, number],
  q1: [number, number, number],
  q2: [number, number, number],
  nx: number,
  ny: number,
  nz: number,
): boolean {
  const ax = Math.abs(nx);
  const ay = Math.abs(ny);
  const az = Math.abs(nz);
  const u = ax >= ay && ax >= az ? 1 : 0;
  const v = ay >= ax && ay >= az ? 2 : u === 1 ? 2 : 1;
  const P: [number, number][] = [
    [p0[u], p0[v]],
    [p1[u], p1[v]],
    [p2[u], p2[v]],
  ];
  const Q: [number, number][] = [
    [q0[u], q0[v]],
    [q1[u], q1[v]],
    [q2[u], q2[v]],
  ];
  for (let i = 0; i < 3; i++) {
    const a = P[i]!;
    const b = P[(i + 1) % 3]!;
    for (let j = 0; j < 3; j++) {
      const c = Q[j]!;
      const d = Q[(j + 1) % 3]!;
      if (segsIntersect2d(a[0], a[1], b[0], b[1], c[0], c[1], d[0], d[1])) return true;
    }
  }
  return (
    pointInTri2d(P[0]![0], P[0]![1], Q[0]![0], Q[0]![1], Q[1]![0], Q[1]![1], Q[2]![0], Q[2]![1]) ||
    pointInTri2d(Q[0]![0], Q[0]![1], P[0]![0], P[0]![1], P[1]![0], P[1]![1], P[2]![0], P[2]![1])
  );
}

/** Exact Möller-style disjoint triangle-triangle intersection test. */
export function trianglesIntersect(
  p0: [number, number, number],
  p1: [number, number, number],
  p2: [number, number, number],
  q0: [number, number, number],
  q1: [number, number, number],
  q2: [number, number, number],
): boolean {
  // Plane of Q
  const q1x = q1[0] - q0[0];
  const q1y = q1[1] - q0[1];
  const q1z = q1[2] - q0[2];
  const q2x = q2[0] - q0[0];
  const q2y = q2[1] - q0[1];
  const q2z = q2[2] - q0[2];
  const n2x = q1y * q2z - q1z * q2y;
  const n2y = q1z * q2x - q1x * q2z;
  const n2z = q1x * q2y - q1y * q2x;
  const n2Len = Math.hypot(n2x, n2y, n2z);
  if (!(n2Len > 1e-18)) return false;

  const d2 = -(n2x * q0[0] + n2y * q0[1] + n2z * q0[2]);
  const dp0 = (n2x * p0[0] + n2y * p0[1] + n2z * p0[2] + d2) / n2Len;
  const dp1 = (n2x * p1[0] + n2y * p1[1] + n2z * p1[2] + d2) / n2Len;
  const dp2 = (n2x * p2[0] + n2y * p2[1] + n2z * p2[2] + d2) / n2Len;
  const eps = 1e-9;
  if (Math.abs(dp0) <= eps && Math.abs(dp1) <= eps && Math.abs(dp2) <= eps) {
    return coplanarTriIntersect(p0, p1, p2, q0, q1, q2, n2x, n2y, n2z);
  }
  if ((dp0 > eps && dp1 > eps && dp2 > eps) || (dp0 < -eps && dp1 < -eps && dp2 < -eps)) {
    return false;
  }

  // Plane of P
  const p1x = p1[0] - p0[0];
  const p1y = p1[1] - p0[1];
  const p1z = p1[2] - p0[2];
  const p2x = p2[0] - p0[0];
  const p2y = p2[1] - p0[1];
  const p2z = p2[2] - p0[2];
  const n1x = p1y * p2z - p1z * p2y;
  const n1y = p1z * p2x - p1x * p2z;
  const n1z = p1x * p2y - p1y * p2x;
  const n1Len = Math.hypot(n1x, n1y, n1z);
  if (!(n1Len > 1e-18)) return false;

  const d1 = -(n1x * p0[0] + n1y * p0[1] + n1z * p0[2]);
  const dq0 = (n1x * q0[0] + n1y * q0[1] + n1z * q0[2] + d1) / n1Len;
  const dq1 = (n1x * q1[0] + n1y * q1[1] + n1z * q1[2] + d1) / n1Len;
  const dq2 = (n1x * q2[0] + n1y * q2[1] + n1z * q2[2] + d1) / n1Len;
  if ((dq0 > eps && dq1 > eps && dq2 > eps) || (dq0 < -eps && dq1 < -eps && dq2 < -eps)) {
    return false;
  }

  // Intersection line direction L = N1 x N2
  const lx = n1y * n2z - n1z * n2y;
  const ly = n1z * n2x - n1x * n2z;
  const lz = n1x * n2y - n1y * n2x;
  const lLen = Math.hypot(lx, ly, lz);
  if (!(lLen > 1e-18)) {
    return false;
  }
  const dirX = lx / lLen;
  const dirY = ly / lLen;
  const dirZ = lz / lLen;

  const intervalOnLine = (
    a: [number, number, number],
    b: [number, number, number],
    c: [number, number, number],
    da: number,
    db: number,
    dc: number,
  ): [number, number] | null => {
    const pts: [number, number, number][] = [a, b, c];
    const ds = [da, db, dc];
    const ts: number[] = [];
    for (let i = 0; i < 3; i++) {
      const dA = ds[i]!;
      const pA = pts[i]!;
      if (Math.abs(dA) <= eps) {
        ts.push(pA[0] * dirX + pA[1] * dirY + pA[2] * dirZ);
      }
      const j = (i + 1) % 3;
      const dB = ds[j]!;
      const pB = pts[j]!;
      if ((dA > eps && dB < -eps) || (dA < -eps && dB > eps)) {
        const s = dA / (dA - dB);
        const ix = pA[0] + s * (pB[0] - pA[0]);
        const iy = pA[1] + s * (pB[1] - pA[1]);
        const iz = pA[2] + s * (pB[2] - pA[2]);
        ts.push(ix * dirX + iy * dirY + iz * dirZ);
      }
    }
    if (ts.length === 0) return null;
    return [Math.min(...ts), Math.max(...ts)];
  };

  const ip = intervalOnLine(p0, p1, p2, dp0, dp1, dp2);
  const iq = intervalOnLine(q0, q1, q2, dq0, dq1, dq2);
  if (ip === null || iq === null) return false;
  return ip[0] <= iq[1] + eps && iq[0] <= ip[1] + eps;
}

/**
 * Spatial-grid search for vertex-disjoint self-intersecting triangle pairs.
 * Bounded for large meshes so viewport interactions stay responsive.
 */
export function detectSelfIntersectingTriangles(
  positions: Float32Array,
  indices: Uint32Array,
  maxTrianglesOut = MAX_SI_HOTSPOT_SAMPLES,
): Uint32Array {
  const triCount = Math.floor(indices.length / 3);
  if (triCount < 2 || triCount > MAX_DETECT_TRIS) return new Uint32Array(0);

  let minX = Infinity;
  let minY = Infinity;
  let minZ = Infinity;
  let maxX = -Infinity;
  let maxY = -Infinity;
  let maxZ = -Infinity;
  for (let i = 0; i < positions.length; i += 3) {
    const x = positions[i]!;
    const y = positions[i + 1]!;
    const z = positions[i + 2]!;
    if (x < minX) minX = x;
    if (y < minY) minY = y;
    if (z < minZ) minZ = z;
    if (x > maxX) maxX = x;
    if (y > maxY) maxY = y;
    if (z > maxZ) maxZ = z;
  }
  const diag = Math.hypot(maxX - minX, maxY - minY, maxZ - minZ);
  const cellSize = Math.max(diag / Math.max(8, Math.cbrt(triCount) * 1.5), 1e-4);
  const invCell = 1 / cellSize;

  const tMinX = new Float32Array(triCount);
  const tMinY = new Float32Array(triCount);
  const tMinZ = new Float32Array(triCount);
  const tMaxX = new Float32Array(triCount);
  const tMaxY = new Float32Array(triCount);
  const tMaxZ = new Float32Array(triCount);
  const grid = new Map<string, number[]>();

  for (let t = 0; t < triCount; t++) {
    const b = t * 3;
    const [ax, ay, az] = trianglePoint(positions, indices[b]!);
    const [bx, by, bz] = trianglePoint(positions, indices[b + 1]!);
    const [cx, cy, cz] = trianglePoint(positions, indices[b + 2]!);
    const lx = Math.min(ax, bx, cx);
    const ly = Math.min(ay, by, cy);
    const lz = Math.min(az, bz, cz);
    const hx = Math.max(ax, bx, cx);
    const hy = Math.max(ay, by, cy);
    const hz = Math.max(az, bz, cz);
    tMinX[t] = lx;
    tMinY[t] = ly;
    tMinZ[t] = lz;
    tMaxX[t] = hx;
    tMaxY[t] = hy;
    tMaxZ[t] = hz;

    const gx0 = Math.floor(lx * invCell);
    const gy0 = Math.floor(ly * invCell);
    const gz0 = Math.floor(lz * invCell);
    const gx1 = Math.min(gx0 + 3, Math.floor(hx * invCell));
    const gy1 = Math.min(gy0 + 3, Math.floor(hy * invCell));
    const gz1 = Math.min(gz0 + 3, Math.floor(hz * invCell));
    for (let x = gx0; x <= gx1; x++) {
      for (let y = gy0; y <= gy1; y++) {
        for (let z = gz0; z <= gz1; z++) {
          const key = `${x},${y},${z}`;
          let bucket = grid.get(key);
          if (!bucket) {
            bucket = [];
            grid.set(key, bucket);
          }
          bucket.push(t);
        }
      }
    }
  }

  const hitTris = new Set<number>();
  const testedPairs = new Set<number>();
  const stride = triCount;

  for (const bucket of grid.values()) {
    if (bucket.length < 2) continue;
    const cap = Math.min(bucket.length, 64);
    for (let i = 0; i < cap; i++) {
      const t0 = bucket[i]!;
      const b0 = t0 * 3;
      const a0 = indices[b0]!;
      const a1 = indices[b0 + 1]!;
      const a2 = indices[b0 + 2]!;
      for (let j = i + 1; j < cap; j++) {
        const t1 = bucket[j]!;
        const lo = t0 < t1 ? t0 : t1;
        const hi = t0 < t1 ? t1 : t0;
        const pairKey = lo * stride + hi;
        if (testedPairs.has(pairKey)) continue;
        testedPairs.add(pairKey);

        if (
          tMaxX[t0]! < tMinX[t1]! ||
          tMinX[t0]! > tMaxX[t1]! ||
          tMaxY[t0]! < tMinY[t1]! ||
          tMinY[t0]! > tMaxY[t1]! ||
          tMaxZ[t0]! < tMinZ[t1]! ||
          tMinZ[t0]! > tMaxZ[t1]!
        ) {
          continue;
        }
        const b1 = t1 * 3;
        const c0 = indices[b1]!;
        const c1 = indices[b1 + 1]!;
        const c2 = indices[b1 + 2]!;
        if (a0 === c0 || a0 === c1 || a0 === c2 || a1 === c0 || a1 === c1 || a1 === c2 || a2 === c0 || a2 === c1 || a2 === c2) {
          continue;
        }
        if (
          trianglesIntersect(
            trianglePoint(positions, a0),
            trianglePoint(positions, a1),
            trianglePoint(positions, a2),
            trianglePoint(positions, c0),
            trianglePoint(positions, c1),
            trianglePoint(positions, c2),
          )
        ) {
          hitTris.add(t0);
          hitTris.add(t1);
          if (hitTris.size >= maxTrianglesOut) {
            return Uint32Array.from(hitTris);
          }
        }
      }
    }
  }
  return Uint32Array.from(hitTris);
}

function buildTriangleOverlayMesh(
  positions: Float32Array,
  indices: Uint32Array,
  tris: Uint32Array,
  color: number,
  name: string,
): THREE.Mesh | null {
  if (tris.length === 0) return null;
  const triPos = new Float32Array(tris.length * 9);
  for (let i = 0; i < tris.length; i++) {
    const base = tris[i]! * 3;
    const dst = i * 9;
    for (let k = 0; k < 3; k++) {
      const src = (indices[base + k] ?? 0) * 3;
      triPos[dst + k * 3] = positions[src] ?? 0;
      triPos[dst + k * 3 + 1] = positions[src + 1] ?? 0;
      triPos[dst + k * 3 + 2] = positions[src + 2] ?? 0;
    }
  }
  const geo = new THREE.BufferGeometry();
  geo.setAttribute("position", new THREE.BufferAttribute(triPos, 3));
  const mat = new THREE.MeshBasicMaterial({
    color,
    transparent: true,
    opacity: 0.55,
    side: THREE.DoubleSide,
    depthTest: true,
    depthWrite: false,
    polygonOffset: true,
    polygonOffsetFactor: -1,
    polygonOffsetUnits: -1,
  });
  const mesh = new THREE.Mesh(geo, mat);
  mesh.name = name;
  mesh.renderOrder = 2;
  mesh.frustumCulled = false;
  return mesh;
}

function buildEdgeSegments(
  positions: Float32Array,
  edgePairs: Uint32Array,
  color: number,
  name: string,
): THREE.LineSegments | null {
  if (edgePairs.length < 2) return null;
  const edgeCount = Math.floor(edgePairs.length / 2);
  const segPos = new Float32Array(edgeCount * 6);
  for (let e = 0; e < edgeCount; e++) {
    const a = edgePairs[e * 2]! * 3;
    const b = edgePairs[e * 2 + 1]! * 3;
    const dst = e * 6;
    segPos[dst] = positions[a] ?? 0;
    segPos[dst + 1] = positions[a + 1] ?? 0;
    segPos[dst + 2] = positions[a + 2] ?? 0;
    segPos[dst + 3] = positions[b] ?? 0;
    segPos[dst + 4] = positions[b + 1] ?? 0;
    segPos[dst + 5] = positions[b + 2] ?? 0;
  }
  const geo = new THREE.BufferGeometry();
  geo.setAttribute("position", new THREE.BufferAttribute(segPos, 3));
  const mat = new THREE.LineBasicMaterial({
    color,
    depthTest: false,
    transparent: true,
    opacity: 0.95,
  });
  const lines = new THREE.LineSegments(geo, mat);
  lines.name = name;
  lines.renderOrder = 3;
  lines.frustumCulled = false;
  return lines;
}

export function countHealerOverlays(
  analysis: Analysis,
  positions: Float32Array,
  indices: Uint32Array,
): HealerOverlayCounts {
  const slivers = detectSliverTriangles(positions, indices);
  const conflictEdges = detectOrientationConflictEdges(indices);
  const invertedTris = detectInvertedClosedComponents(analysis, positions, indices);
  const siTris =
    analysis.selfIntersects === true ? detectSelfIntersectingTriangles(positions, indices) : new Uint32Array(0);
  return {
    selfIntersections: siTris.length,
    inverted: Math.max(analysis.orientationConflicts, Math.floor(conflictEdges.length / 2)) + (invertedTris.length > 0 ? 1 : 0),
    slivers: slivers.length,
  };
}

/**
 * Combines standard `repair-overlays.ts` primitives with Mesh Healer overlays
 * (self-intersections, inverted winding / orientation conflicts, sliver faces).
 */
export function buildHealerOverlayGroup(
  analysis: Analysis,
  positions: Float32Array,
  indices: Uint32Array,
  repairKinds: Record<RepairOverlayKind, boolean>,
  healerKinds: Record<HealerOverlayKind, boolean>,
): THREE.Group | null {
  const base = buildRepairOverlayGroup(analysis, positions, indices, repairKinds);
  const extra: THREE.Object3D[] = [];

  if (healerKinds.slivers) {
    const tris = detectSliverTriangles(positions, indices);
    const mesh = buildTriangleOverlayMesh(positions, indices, tris, HEALER_OVERLAY_COLORS.slivers, "overlay-slivers");
    if (mesh !== null) extra.push(mesh);
  }
  if (healerKinds.inverted) {
    const conflicts = detectOrientationConflictEdges(indices);
    const lines = buildEdgeSegments(positions, conflicts, HEALER_OVERLAY_COLORS.inverted, "overlay-inverted-edges");
    if (lines !== null) extra.push(lines);
    const invTris = detectInvertedClosedComponents(analysis, positions, indices);
    const mesh = buildTriangleOverlayMesh(
      positions,
      indices,
      invTris,
      HEALER_OVERLAY_COLORS.inverted,
      "overlay-inverted-shells",
    );
    if (mesh !== null) extra.push(mesh);
  }
  if (healerKinds.selfIntersections && analysis.selfIntersects !== false) {
    const tris = detectSelfIntersectingTriangles(positions, indices);
    const mesh = buildTriangleOverlayMesh(
      positions,
      indices,
      tris,
      HEALER_OVERLAY_COLORS.selfIntersections,
      "overlay-selfIntersections",
    );
    if (mesh !== null) extra.push(mesh);
  }

  if (extra.length === 0) return base;
  const root = base ?? new THREE.Group();
  root.name = "repair-overlays";
  for (const child of extra) {
    root.add(child);
  }
  return root;
}

export function disposeHealerOverlayGroup(group: THREE.Group): void {
  for (const child of group.children) {
    if (child instanceof THREE.LineSegments || child instanceof THREE.Mesh) {
      child.geometry.dispose();
      const mat = child.material;
      if (Array.isArray(mat)) {
        for (const m of mat) m.dispose();
      } else {
        mat.dispose();
      }
    }
  }
}

/**
 * Extracts clickable defect hotspots ordered by severity for camera click-to-focus.
 */
export function extractDefectHotspots(
  analysis: Analysis,
  positions: Float32Array,
  indices: Uint32Array,
  maxItems = 12,
): DefectHotspot[] {
  const spots: DefectHotspot[] = [];

  // Closed boundary loops (holes), largest first.
  const closedLoops = analysis.loops.filter((l) => l.closed).sort((a, b) => b.vertices.length - a.vertices.length);
  for (let i = 0; i < Math.min(3, closedLoops.length); i++) {
    const loop = closedLoops[i]!;
    const { center, radius } = vertexListBounds(positions, loop.vertices);
    spots.push({
      id: `hole-${i}`,
      kind: "holes",
      label: `Hole #${i + 1} (${loop.id.len} edges)`,
      count: loop.id.len,
      center,
      radius,
    });
  }

  // Non-manifold edges.
  if (analysis.nonManifoldEdgeList.length > 0) {
    const verts: number[] = [];
    for (const [a, b] of analysis.nonManifoldEdgeList) {
      verts.push(a, b);
    }
    const { center, radius } = vertexListBounds(positions, verts);
    spots.push({
      id: "non-manifold",
      kind: "nonManifold",
      label: `Non-manifold edges (${analysis.nonManifoldEdges})`,
      count: analysis.nonManifoldEdges,
      center,
      radius,
    });
  }

  // Self-intersections when flagged.
  if (analysis.selfIntersects === true) {
    const siTris = detectSelfIntersectingTriangles(positions, indices);
    if (siTris.length > 0) {
      const { center, radius } = triangleBounds(positions, siTris, indices);
      spots.push({
        id: "self-intersections",
        kind: "selfIntersections",
        label: `Self-intersecting (${siTris.length} tris)`,
        count: siTris.length,
        center,
        radius,
      });
    }
  }

  // Inverted / orientation conflicts.
  const conflictEdges = detectOrientationConflictEdges(indices);
  if (conflictEdges.length > 0) {
    const { center, radius } = vertexListBounds(positions, conflictEdges);
    spots.push({
      id: "inverted-edges",
      kind: "inverted",
      label: `Orientation conflicts (${Math.floor(conflictEdges.length / 2)} edges)`,
      count: Math.floor(conflictEdges.length / 2),
      center,
      radius,
    });
  } else {
    const invTris = detectInvertedClosedComponents(analysis, positions, indices);
    if (invTris.length > 0) {
      const { center, radius } = triangleBounds(positions, invTris, indices);
      spots.push({
        id: "inverted-shell",
        kind: "inverted",
        label: `Inverted shell (${invTris.length} tris)`,
        count: invTris.length,
        center,
        radius,
      });
    }
  }

  // Sliver triangles.
  const slivers = detectSliverTriangles(positions, indices);
  if (slivers.length > 0) {
    const { center, radius } = triangleBounds(positions, slivers, indices);
    spots.push({
      id: "slivers",
      kind: "slivers",
      label: `Sliver faces (${slivers.length})`,
      count: slivers.length,
      center,
      radius,
    });
  }

  // Degenerate / duplicate triangles.
  const degenCount = analysis.degenerateList.length + analysis.duplicateList.length;
  if (degenCount > 0) {
    const tris = [...analysis.degenerateList, ...analysis.duplicateList];
    const { center, radius } = triangleBounds(positions, tris, indices);
    spots.push({
      id: "degenerate",
      kind: "degenerate",
      label: `Degenerate / duplicate (${degenCount})`,
      count: degenCount,
      center,
      radius,
    });
  }

  // Disconnected small components.
  if (analysis.components.length > 1) {
    const sorted = [...analysis.components].sort((a, b) => a.triangles.length - b.triangles.length);
    const smallest = sorted[0];
    if (smallest !== undefined && smallest.triangles.length > 0) {
      const { center, radius } = triangleBounds(positions, smallest.triangles, indices);
      spots.push({
        id: "small-component",
        kind: "components",
        label: `Smallest island (${smallest.triangles.length} tris)`,
        count: analysis.components.length - 1,
        center,
        radius,
      });
    }
  }

  return spots.slice(0, maxItems);
}
