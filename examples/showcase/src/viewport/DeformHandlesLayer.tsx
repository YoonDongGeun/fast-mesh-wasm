// Viewport 3D cage, anchor, handle, and Kelvinlet pull gizmos for #193 deformation.
import { useThree } from "@react-three/fiber";
import type {
  FfdBasis,
  FfdFrameData,
  FfdLatticeData,
  FreeformProtection,
  FreeformReport,
  KelvinletLoad,
  MeshHandle,
} from "fast-mesh-wasm";
import type { SceneStore } from "fast-mesh-wasm/store";
import { useEffect, useLayoutEffect, useMemo, useRef, useState, useSyncExternalStore } from "react";
import * as THREE from "three";
import { type AppState, type Item, useApp } from "../scene/app-state";
import type { Wave6DeformState } from "../scene/wave6-state";
import type { SculptMask } from "./sculpt";
import { isExtraFinger } from "./touch";
import { setDragging, view } from "./view";

export interface DeformHandlesLayerProps {
  app: AppState;
  store: SceneStore;
}

export type Vec3Tuple = [number, number, number];

export interface PinBoxProtection {
  min: Vec3Tuple;
  max: Vec3Tuple;
  positions: ArrayLike<number>;
  liveIds?: ArrayLike<number>;
}

export interface ProtectionBuildInput {
  mask?: SculptMask | null;
  activeMesh?: MeshHandle | null;
  protectMask?: boolean;
  frozenVertices?: readonly number[];
  weights?: readonly number[] | Float32Array;
  pinBox?: PinBoxProtection | null;
}

type DeformDomain = NonNullable<SceneStore["scene"]["deform"]>;

interface KelvinletPreview {
  start: Vec3Tuple;
  end: Vec3Tuple;
}

/** Returns the row-major flat index `(i * ny + j) * nz + k` for FFD control point `(i, j, k)`. */
export function ffdControlPointIndex(
  dims: readonly [number, number, number],
  i: number,
  j: number,
  k: number,
): number {
  const [nx, ny, nz] = dims;
  if (
    !Number.isInteger(nx) ||
    !Number.isInteger(ny) ||
    !Number.isInteger(nz) ||
    nx < 2 ||
    ny < 2 ||
    nz < 2
  ) {
    throw new Error(`Invalid FFD lattice dimensions [${nx}, ${ny}, ${nz}]; each axis must be an integer >= 2.`);
  }
  if (
    !Number.isInteger(i) ||
    !Number.isInteger(j) ||
    !Number.isInteger(k) ||
    i < 0 ||
    i >= nx ||
    j < 0 ||
    j >= ny ||
    k < 0 ||
    k >= nz
  ) {
    throw new Error(`FFD control point (${i}, ${j}, ${k}) out of bounds for dims [${nx}, ${ny}, ${nz}].`);
  }
  return (i * ny + j) * nz + k;
}

function bsplineKnot(n: number, p: number, idx: number): number {
  if (idx <= p) return 0;
  if (idx >= n) return 1;
  return (idx - p) / (n - p);
}

/** Normalized lattice parameter in `[0, 1]` matching Rust `axis_param` for Bernstein and Cubic B-Spline. */
export function axisParam(i: number, n: number, basis: FfdBasis): number {
  if (basis === "cubicBSpline") {
    const p = Math.min(n - 1, 3);
    let sum = 0;
    for (let offset = 1; offset <= p; offset++) {
      sum += bsplineKnot(n, p, i + offset);
    }
    return sum / p;
  }
  return i / (n - 1);
}

function paramToLocal(frame: FfdFrameData, uvw: readonly [number, number, number]): Vec3Tuple {
  let x = frame.center[0];
  let y = frame.center[1];
  let z = frame.center[2];
  for (let a = 0; a < 3; a++) {
    const s = (2 * uvw[a] - 1) * frame.halfExtents[a];
    const axis = frame.axes[a];
    x += s * axis[0];
    y += s * axis[1];
    z += s * axis[2];
  }
  return [x, y, z];
}

/** Computes displaced control point positions in mesh-local space in row-major order. */
export function computeLatticeControlPoints(lattice: FfdLatticeData): Vec3Tuple[] {
  const [nx, ny, nz] = lattice.dims;
  const total = nx * ny * nz;
  if (lattice.offsets.length < total * 3) {
    throw new Error(
      `FFD lattice offsets length ${lattice.offsets.length} is shorter than required ${total * 3}.`,
    );
  }
  const points: Vec3Tuple[] = new Array(total);
  for (let i = 0; i < nx; i++) {
    const u = axisParam(i, nx, lattice.basis);
    for (let j = 0; j < ny; j++) {
      const v = axisParam(j, ny, lattice.basis);
      for (let k = 0; k < nz; k++) {
        const w = axisParam(k, nz, lattice.basis);
        const idx = (i * ny + j) * nz + k;
        const rest = paramToLocal(lattice.frame, [u, v, w]);
        const base = idx * 3;
        points[idx] = [
          rest[0] + lattice.offsets[base],
          rest[1] + lattice.offsets[base + 1],
          rest[2] + lattice.offsets[base + 2],
        ];
      }
    }
  }
  return points;
}

/** Builds flat line-segment coordinates (`[x0,y0,z0, x1,y1,z1, ...]`) for the FFD cage wireframe. */
export function buildLatticeEdgeSegments(lattice: FfdLatticeData): Float32Array {
  const [nx, ny, nz] = lattice.dims;
  const points = computeLatticeControlPoints(lattice);
  const segCount = (nx - 1) * ny * nz + nx * (ny - 1) * nz + nx * ny * (nz - 1);
  const out = new Float32Array(segCount * 6);
  let cursor = 0;

  const pushSeg = (a: Vec3Tuple, b: Vec3Tuple) => {
    out[cursor++] = a[0];
    out[cursor++] = a[1];
    out[cursor++] = a[2];
    out[cursor++] = b[0];
    out[cursor++] = b[1];
    out[cursor++] = b[2];
  };

  for (let i = 0; i < nx; i++) {
    for (let j = 0; j < ny; j++) {
      for (let k = 0; k < nz; k++) {
        const idx = (i * ny + j) * nz + k;
        const p = points[idx];
        if (i + 1 < nx) {
          pushSeg(p, points[((i + 1) * ny + j) * nz + k]);
        }
        if (j + 1 < ny) {
          pushSeg(p, points[(i * ny + (j + 1)) * nz + k]);
        }
        if (k + 1 < nz) {
          pushSeg(p, points[(i * ny + j) * nz + (k + 1)]);
        }
      }
    }
  }
  return out;
}

/** Clones an `FfdLatticeData` object with a fresh `Float64Array` for offsets. */
export function cloneLattice(lattice: FfdLatticeData): FfdLatticeData {
  return {
    dims: [lattice.dims[0], lattice.dims[1], lattice.dims[2]],
    basis: lattice.basis,
    frame: {
      center: [lattice.frame.center[0], lattice.frame.center[1], lattice.frame.center[2]],
      axes: [
        [lattice.frame.axes[0][0], lattice.frame.axes[0][1], lattice.frame.axes[0][2]],
        [lattice.frame.axes[1][0], lattice.frame.axes[1][1], lattice.frame.axes[1][2]],
        [lattice.frame.axes[2][0], lattice.frame.axes[2][1], lattice.frame.axes[2][2]],
      ],
      halfExtents: [
        lattice.frame.halfExtents[0],
        lattice.frame.halfExtents[1],
        lattice.frame.halfExtents[2],
      ],
    },
    offsets: new Float64Array(lattice.offsets),
  };
}

/**
 * Pure immutable update of one control point offset by flat index or `(i, j, k)`.
 * `"delta"` adds `vector` to the existing offset; `"set"` replaces it.
 */
export function applyDisplacementToLattice(
  lattice: FfdLatticeData,
  indexOrIjk: number | readonly [number, number, number],
  vector: readonly [number, number, number],
  mode: "delta" | "set" = "delta",
): FfdLatticeData {
  if (!Number.isFinite(vector[0]) || !Number.isFinite(vector[1]) || !Number.isFinite(vector[2])) {
    throw new Error("FFD control point displacement vector must contain finite numbers.");
  }
  const [nx, ny, nz] = lattice.dims;
  const total = nx * ny * nz;
  const idx =
    typeof indexOrIjk === "number"
      ? indexOrIjk
      : ffdControlPointIndex(lattice.dims, indexOrIjk[0], indexOrIjk[1], indexOrIjk[2]);
  if (!Number.isInteger(idx) || idx < 0 || idx >= total) {
    throw new Error(`FFD control point flat index ${idx} out of bounds [0, ${total}).`);
  }
  const next = cloneLattice(lattice);
  const base = idx * 3;
  if (mode === "set") {
    next.offsets[base] = vector[0];
    next.offsets[base + 1] = vector[1];
    next.offsets[base + 2] = vector[2];
  } else {
    next.offsets[base] += vector[0];
    next.offsets[base + 1] += vector[1];
    next.offsets[base + 2] += vector[2];
  }
  return next;
}

/** Pure helper moving FFD control point `(i, j, k)` by delta or absolute offset. */
export function moveFfdControlPoint(
  lattice: FfdLatticeData,
  i: number,
  j: number,
  k: number,
  offsetOrDelta: readonly [number, number, number],
  mode: "delta" | "set" = "delta",
): FfdLatticeData {
  const idx = ffdControlPointIndex(lattice.dims, i, j, k);
  return applyDisplacementToLattice(lattice, idx, offsetOrDelta, mode);
}

/** Returns a clone of `lattice` with all control point offsets cleared to zero. */
export function resetLatticeOffsets(lattice: FfdLatticeData): FfdLatticeData {
  const next = cloneLattice(lattice);
  next.offsets.fill(0);
  return next;
}

/**
 * Converts a Showcase `SculptMask` on `activeMesh` into a `FreeformProtection` option.
 * Critical Wave-6 semantics: **`weight = 0` means FROZEN and `weight = 1` means FULL MOVE**.
 */
export function buildFreeformProtectionFromMask(
  mask: SculptMask | null | undefined,
  activeMesh?: MeshHandle | null,
): FreeformProtection | undefined {
  if (!mask || mask.vertices.length === 0) {
    return undefined;
  }
  if (activeMesh !== undefined && activeMesh !== null && mask.mesh !== activeMesh) {
    return undefined;
  }
  const vertices = new Uint32Array(mask.vertices);
  // 0 = completely frozen, 1 = full move.
  const weights = new Float32Array(vertices.length);
  return { vertices, weights };
}

/**
 * Builds a unified `FreeformProtection` object from sculpt mask, explicit weights, and/or pin box.
 * Critical Wave-6 semantics: **`weight = 0` means FROZEN and `weight = 1` means FULL MOVE**.
 */
export function buildProtectionOption(input: ProtectionBuildInput): FreeformProtection | undefined {
  const byVertex = new Map<number, number>();
  const setMinWeight = (v: number, w: number) => {
    const clamped = Math.max(0, Math.min(1, w));
    const prev = byVertex.get(v);
    byVertex.set(v, prev === undefined ? clamped : Math.min(prev, clamped));
  };

  if (input.protectMask !== false && input.mask) {
    const fromMask = buildFreeformProtectionFromMask(input.mask, input.activeMesh);
    if (fromMask) {
      for (let i = 0; i < fromMask.vertices.length; i++) {
        setMinWeight(fromMask.vertices[i], fromMask.weights ? fromMask.weights[i] : 0);
      }
    }
  }

  if (input.frozenVertices && input.frozenVertices.length > 0) {
    for (let i = 0; i < input.frozenVertices.length; i++) {
      const v = input.frozenVertices[i];
      if (!Number.isInteger(v) || v < 0) {
        throw new Error(`Invalid protected vertex index: ${v}`);
      }
      const w = input.weights ? Number(input.weights[i] ?? 0) : 0;
      if (!Number.isFinite(w)) {
        throw new Error(`Invalid protection weight at index ${i}: ${w}`);
      }
      setMinWeight(v, w);
    }
  }

  if (input.pinBox) {
    const { min, max, positions, liveIds } = input.pinBox;
    const minX = Math.min(min[0], max[0]);
    const maxX = Math.max(min[0], max[0]);
    const minY = Math.min(min[1], max[1]);
    const maxY = Math.max(min[1], max[1]);
    const minZ = Math.min(min[2], max[2]);
    const maxZ = Math.max(min[2], max[2]);
    const slotCount = Math.floor(positions.length / 3);
    const iterCount = liveIds ? liveIds.length : slotCount;
    for (let i = 0; i < iterCount; i++) {
      const v = liveIds ? liveIds[i] : i;
      const base = v * 3;
      if (base + 2 >= positions.length) continue;
      const x = positions[base];
      const y = positions[base + 1];
      const z = positions[base + 2];
      if (x >= minX && x <= maxX && y >= minY && y <= maxY && z >= minZ && z <= maxZ) {
        setMinWeight(v, 0);
      }
    }
  }

  if (byVertex.size === 0) {
    return undefined;
  }

  const sortedIds = Array.from(byVertex.keys()).sort((a, b) => a - b);
  const vertices = Uint32Array.from(sortedIds);
  const weights = new Float32Array(sortedIds.length);
  for (let i = 0; i < sortedIds.length; i++) {
    weights[i] = byVertex.get(sortedIds[i]) ?? 0;
  }
  return { vertices, weights };
}

/** Formats a concise status summary line for `wave6.deform.lastReport`. */
export function formatFreeformReport(label: string, report: FreeformReport): string {
  const base = `${label}: moved ${report.verticesMoved}/${report.verticesEvaluated} verts, max disp ${report.maxDisplacement.toFixed(4)}`;
  return report.selfCollisionContactsResolved > 0
    ? `${base}, ${report.selfCollisionContactsResolved} self-collisions resolved`
    : base;
}

/** Resolves the active mesh from SceneStore selection (falling back to the first loaded item). */
export function resolveActiveDeformMesh(store: SceneStore, items: readonly Item[]): MeshHandle | null {
  const selection = store.getSnapshot().selection;
  if (selection !== null) {
    return selection;
  }
  return items.length > 0 ? items[0].mesh : null;
}

/** Retrieves `store.scene.deform`, throwing if the scene was built without the deform domain. */
export function requireDeformDomain(store: SceneStore): DeformDomain {
  const deform = store.scene.deform;
  if (!deform) {
    throw new Error("Deform domain is unavailable on this scene.");
  }
  return deform;
}

function setNdcFromPointer(ev: PointerEvent, element: HTMLElement, target: THREE.Vector2): void {
  const rect = element.getBoundingClientRect();
  const w = rect.width || 1;
  const h = rect.height || 1;
  target.set(((ev.clientX - rect.left) / w) * 2 - 1, -((ev.clientY - rect.top) / h) * 2 + 1);
}

function buildKelvinletLoadFromDrag(kind: Wave6DeformState["kelvinletKind"], delta: Vec3Tuple): KelvinletLoad {
  switch (kind) {
    case "grab":
      return { kind: "grab", force: delta };
    case "twist":
      return { kind: "twist", torque: [delta[0] * 2, delta[1] * 2, delta[2] * 2] };
    case "scale": {
      const mag = Math.hypot(delta[0], delta[1], delta[2]) * (delta[1] + delta[0] >= 0 ? 1 : -1);
      return { kind: "scale", magnitude: mag };
    }
    case "pinch":
      return {
        kind: "pinch",
        forceDipole: [
          [delta[0], 0, 0],
          [0, delta[1], 0],
          [0, 0, delta[2]],
        ],
      };
  }
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

/** R3F viewport layer rendering FFD control cage, ARAP anchors/handles, and Kelvinlet pull gizmo. */
export function DeformHandlesLayer({ app, store }: DeformHandlesLayerProps) {
  const { camera, gl, raycaster, invalidate } = useThree();
  const deform = useApp(app, (s) => s.wave6.deform);
  const items = useApp(app, (s) => s.items);
  const sculptMask = useApp(app, (s) => s.sculptMask);

  const selection = useSyncExternalStore(
    store.subscribe,
    () => store.getSnapshot().selection,
    () => store.getSnapshot().selection,
  );
  const activeMesh = selection ?? items[0]?.mesh ?? null;

  const [sceneRev, setSceneRev] = useState(0);
  const [hoveredCp, setHoveredCp] = useState<number | null>(null);
  const [activeCp, setActiveCp] = useState<number | null>(null);
  const [kelvinletPreview, setKelvinletPreview] = useState<KelvinletPreview | null>(null);

  useEffect(() => {
    return store.subscribe(() => setSceneRev((n) => n + 1));
  }, [store]);

  const rootGroupRef = useRef<THREE.Group | null>(null);
  const cpMeshRefs = useRef<Array<THREE.Mesh | null>>([]);
  const deformRef = useRef(deform);
  deformRef.current = deform;
  const maskRef = useRef(sculptMask);
  maskRef.current = sculptMask;

  // Keep local-space handles synchronized with the active mesh's world matrix.
  useLayoutEffect(() => {
    const grp = rootGroupRef.current;
    if (!grp) return;
    if (activeMesh === null) {
      grp.matrix.identity();
      return;
    }
    try {
      const m = store.scene.matrix4(activeMesh);
      grp.matrix.fromArray(m);
    } catch {
      grp.matrix.identity();
    }
    invalidate();
  }, [activeMesh, store, sceneRev, deform.ffdLattice, deform.arapAnchors, deform.arapHandles, invalidate]);

  const lattice = deform.ffdLattice;
  const showFfd =
    activeMesh !== null &&
    lattice !== null &&
    (deform.algorithm === "ffd" || deform.activeTool === "ffdCage");

  const { cpPoints, edgePositions, sphereRadius } = useMemo(() => {
    if (!showFfd || !lattice) {
      return { cpPoints: [] as Vec3Tuple[], edgePositions: new Float32Array(0), sphereRadius: 0.03 };
    }
    const pts = computeLatticeControlPoints(lattice);
    const edges = buildLatticeEdgeSegments(lattice);
    const maxHalf = Math.max(
      lattice.frame.halfExtents[0],
      lattice.frame.halfExtents[1],
      lattice.frame.halfExtents[2],
      0.25,
    );
    return {
      cpPoints: pts,
      edgePositions: edges,
      sphereRadius: Math.max(0.015, maxHalf * 0.045),
    };
  }, [showFfd, lattice]);

  const wireGeometry = useMemo(() => new THREE.BufferGeometry(), []);
  useLayoutEffect(() => {
    cpMeshRefs.current.length = cpPoints.length;
    wireGeometry.setAttribute("position", new THREE.BufferAttribute(edgePositions, 3));
    wireGeometry.computeBoundingSphere();
    invalidate();
  }, [wireGeometry, edgePositions, cpPoints.length, invalidate]);

  const arapLineGeometry = useMemo(() => new THREE.BufferGeometry(), []);
  const kelvinletLineGeometry = useMemo(() => new THREE.BufferGeometry(), []);
  const sharedSphereGeo = useMemo(() => new THREE.SphereGeometry(1, 14, 12), []);
  const matWire = useMemo(() => new THREE.LineBasicMaterial({ color: 0x38bdf8, transparent: true, opacity: 0.85 }), []);
  const matCpRest = useMemo(() => new THREE.MeshBasicMaterial({ color: 0x38bdf8 }), []);
  const matCpMoved = useMemo(() => new THREE.MeshBasicMaterial({ color: 0xf59e0b }), []);
  const matCpActive = useMemo(() => new THREE.MeshBasicMaterial({ color: 0x34d399 }), []);
  const matAnchor = useMemo(() => new THREE.MeshBasicMaterial({ color: 0x3b82f6 }), []);
  const matHandle = useMemo(() => new THREE.MeshBasicMaterial({ color: 0xf59e0b }), []);
  const matTarget = useMemo(() => new THREE.MeshBasicMaterial({ color: 0x10b981 }), []);
  const matArapLine = useMemo(() => new THREE.LineBasicMaterial({ color: 0x10b981 }), []);
  const matKelvinlet = useMemo(() => new THREE.MeshBasicMaterial({ color: 0xec4899 }), []);
  const matKelvinletLine = useMemo(() => new THREE.LineBasicMaterial({ color: 0xec4899 }), []);

  useEffect(() => {
    return () => {
      wireGeometry.dispose();
      arapLineGeometry.dispose();
      kelvinletLineGeometry.dispose();
      sharedSphereGeo.dispose();
      matWire.dispose();
      matCpRest.dispose();
      matCpMoved.dispose();
      matCpActive.dispose();
      matAnchor.dispose();
      matHandle.dispose();
      matTarget.dispose();
      matArapLine.dispose();
      matKelvinlet.dispose();
      matKelvinletLine.dispose();
    };
  }, [
    wireGeometry,
    arapLineGeometry,
    kelvinletLineGeometry,
    sharedSphereGeo,
    matWire,
    matCpRest,
    matCpMoved,
    matCpActive,
    matAnchor,
    matHandle,
    matTarget,
    matArapLine,
    matKelvinlet,
    matKelvinletLine,
  ]);

  // Read live vertex positions for ARAP/XPBD anchor and handle markers.
  const showArapXpbd =
    activeMesh !== null &&
    (deform.algorithm === "arap" ||
      deform.algorithm === "xpbd" ||
      deform.activeTool === "arapPin" ||
      deform.activeTool === "arapHandle" ||
      deform.activeTool === "xpbdPin");

  const { anchorPoints, handlePairs, markerRadius } = useMemo(() => {
    if (!showArapXpbd || activeMesh === null) {
      return {
        anchorPoints: [] as Vec3Tuple[],
        handlePairs: [] as Array<{ from: Vec3Tuple; to: Vec3Tuple }>,
        markerRadius: 0.03,
      };
    }
    let positions: Float32Array;
    try {
      positions = store.scene.views(activeMesh).positions();
    } catch {
      return { anchorPoints: [], handlePairs: [], markerRadius: 0.03 };
    }
    const anchors: Vec3Tuple[] = [];
    for (const v of deform.arapAnchors) {
      const b = v * 3;
      if (b >= 0 && b + 2 < positions.length) {
        anchors.push([positions[b], positions[b + 1], positions[b + 2]]);
      }
    }
    const pairs: Array<{ from: Vec3Tuple; to: Vec3Tuple }> = [];
    for (const h of deform.arapHandles) {
      const b = h.vertex * 3;
      if (b >= 0 && b + 2 < positions.length) {
        pairs.push({
          from: [positions[b], positions[b + 1], positions[b + 2]],
          to: [h.target[0], h.target[1], h.target[2]],
        });
      }
    }
    let minX = Infinity;
    let maxX = -Infinity;
    for (let i = 0; i + 2 < positions.length; i += 3) {
      if (positions[i] < minX) minX = positions[i];
      if (positions[i] > maxX) maxX = positions[i];
    }
    const span = Number.isFinite(minX) && Number.isFinite(maxX) ? Math.max(0.2, maxX - minX) : 1;
    return { anchorPoints: anchors, handlePairs: pairs, markerRadius: Math.max(0.015, span * 0.025) };
  }, [showArapXpbd, activeMesh, store, sceneRev, deform.arapAnchors, deform.arapHandles]);

  useLayoutEffect(() => {
    const arr = new Float32Array(handlePairs.length * 6);
    for (let i = 0; i < handlePairs.length; i++) {
      const { from, to } = handlePairs[i];
      const base = i * 6;
      arr[base] = from[0];
      arr[base + 1] = from[1];
      arr[base + 2] = from[2];
      arr[base + 3] = to[0];
      arr[base + 4] = to[1];
      arr[base + 5] = to[2];
    }
    arapLineGeometry.setAttribute("position", new THREE.BufferAttribute(arr, 3));
    arapLineGeometry.computeBoundingSphere();
    invalidate();
  }, [arapLineGeometry, handlePairs, invalidate]);

  useLayoutEffect(() => {
    const arr = kelvinletPreview
      ? new Float32Array([
          kelvinletPreview.start[0],
          kelvinletPreview.start[1],
          kelvinletPreview.start[2],
          kelvinletPreview.end[0],
          kelvinletPreview.end[1],
          kelvinletPreview.end[2],
        ])
      : new Float32Array(0);
    kelvinletLineGeometry.setAttribute("position", new THREE.BufferAttribute(arr, 3));
    kelvinletLineGeometry.computeBoundingSphere();
    invalidate();
  }, [kelvinletLineGeometry, kelvinletPreview, invalidate]);

  // FFD Cage sphere pointer drag on camera-facing plane; preserves OrbitControls when inactive or clicking empty space.
  useEffect(() => {
    if (deform.activeTool !== "ffdCage" || activeMesh === null || lattice === null) {
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
      app.updateWave6((prev) => ({
        ...prev,
        deform: { ...prev.deform, ffdLattice: nextLattice },
      }));
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
          const protect = deformRef.current.xpbdProtectMask
            ? buildFreeformProtectionFromMask(maskRef.current, activeMesh)
            : undefined;
          const report = await domain.applyFfd(activeMesh, {
            lattice: stepLattice,
            outsideMode: "clamp",
            protect,
          });
          store.commitFrame();
          if (!disposed) {
            app.updateWave6((prev) => ({
              ...prev,
              deform: { ...prev.deform, lastReport: formatFreeformReport("FFD Drag", report) },
            }));
          }
        } catch (err) {
          if (!disposed) {
            const msg = err instanceof Error ? err.message : String(err);
            app.updateWave6((prev) => ({
              ...prev,
              deform: { ...prev.deform, lastReport: `FFD Drag failed: ${msg}` },
            }));
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
  }, [deform.activeTool, activeMesh, lattice !== null, app, store, gl, camera, raycaster, invalidate]);

  // ARAP Pin / Handle / XPBD Pin / Kelvinlet Drag pointer tools on the active mesh surface.
  useEffect(() => {
    const tool = deform.activeTool;
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
        app.updateWave6((prev) => {
          const exists = prev.deform.arapAnchors.includes(slot);
          const nextAnchors = exists
            ? prev.deform.arapAnchors.filter((v) => v !== slot)
            : [...prev.deform.arapAnchors, slot];
          return {
            ...prev,
            deform: { ...prev.deform, arapAnchors: nextAnchors },
          };
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

        app.updateWave6((prev) => {
          const rest = prev.deform.arapHandles.filter((h) => h.vertex !== slot);
          return {
            ...prev,
            deform: { ...prev.deform, arapHandles: [...rest, { vertex: slot, target: vPos }] },
          };
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
        app.updateWave6((prev) => ({
          ...prev,
          deform: {
            ...prev.deform,
            arapHandles: prev.deform.arapHandles.map((h) => (h.vertex === vertex ? { vertex, target } : h)),
          },
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
            const protect = cur.xpbdProtectMask
              ? buildFreeformProtectionFromMask(maskRef.current, activeMesh)
              : undefined;
            const report = await domain.applyArap(activeMesh, {
              anchors: cur.arapAnchors.map((vertex) => ({ vertex })),
              handles: cur.arapHandles,
              weights: cur.arapWeights,
              iterations: 15,
              protect,
            });
            store.commitFrame();
            if (!disposed) {
              app.updateWave6((prev) => ({
                ...prev,
                deform: { ...prev.deform, lastReport: formatFreeformReport("ARAP Drag", report) },
              }));
            }
          } catch (err) {
            if (!disposed) {
              const msg = err instanceof Error ? err.message : String(err);
              app.updateWave6((prev) => ({
                ...prev,
                deform: { ...prev.deform, lastReport: `ARAP Drag failed: ${msg}` },
              }));
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
          const protect = cur.xpbdProtectMask
            ? buildFreeformProtectionFromMask(maskRef.current, activeMesh)
            : undefined;
          const load = buildKelvinletLoadFromDrag(cur.kelvinletKind, delta);
          const report = await domain.applyKelvinlet(activeMesh, {
            center: drag.startLocal,
            load,
            epsilon: cur.kelvinletEpsilon,
            nu: cur.kelvinletNu,
            scale: cur.kelvinletScale,
            symmetry: { x: cur.kelvinletSymX, y: cur.kelvinletSymY, z: cur.kelvinletSymZ },
            protect,
          });
          store.commitFrame();
          if (!disposed) {
            app.updateWave6((prev) => ({
              ...prev,
              deform: { ...prev.deform, lastReport: formatFreeformReport("Kelvinlet Drag", report) },
            }));
          }
        } catch (err) {
          if (!disposed) {
            const msg = err instanceof Error ? err.message : String(err);
            app.updateWave6((prev) => ({
              ...prev,
              deform: { ...prev.deform, lastReport: `Kelvinlet Drag failed: ${msg}` },
            }));
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
  }, [deform.activeTool, activeMesh, app, store, gl, camera, raycaster, invalidate]);

  return (
    <group ref={rootGroupRef} matrixAutoUpdate={false}>
      {showFfd && lattice && (
        <group>
          <lineSegments geometry={wireGeometry} material={matWire} renderOrder={4} />
          {cpPoints.map((pos, idx) => {
            const b = idx * 3;
            const moved =
              Math.abs(lattice.offsets[b]) > 1e-6 ||
              Math.abs(lattice.offsets[b + 1]) > 1e-6 ||
              Math.abs(lattice.offsets[b + 2]) > 1e-6;
            const highlighted = activeCp === idx || hoveredCp === idx;
            const mat = highlighted ? matCpActive : moved ? matCpMoved : matCpRest;
            return (
              <mesh
                key={idx}
                ref={(el) => {
                  cpMeshRefs.current[idx] = el;
                }}
                position={pos}
                scale={highlighted ? sphereRadius * 1.25 : sphereRadius}
                geometry={sharedSphereGeo}
                material={mat}
                userData={{ cpIndex: idx }}
                renderOrder={5}
              />
            );
          })}
        </group>
      )}

      {showArapXpbd && (
        <group>
          <lineSegments geometry={arapLineGeometry} material={matArapLine} renderOrder={4} />
          {anchorPoints.map((pos, i) => (
            <mesh
              key={`anchor-${i}`}
              position={pos}
              scale={markerRadius}
              geometry={sharedSphereGeo}
              material={matAnchor}
              renderOrder={5}
            />
          ))}
          {handlePairs.map((pair, i) => (
            <group key={`handle-${i}`}>
              <mesh
                position={pair.from}
                scale={markerRadius}
                geometry={sharedSphereGeo}
                material={matHandle}
                renderOrder={5}
              />
              <mesh
                position={pair.to}
                scale={markerRadius * 0.9}
                geometry={sharedSphereGeo}
                material={matTarget}
                renderOrder={5}
              />
            </group>
          ))}
        </group>
      )}

      {kelvinletPreview && (
        <group>
          <lineSegments geometry={kelvinletLineGeometry} material={matKelvinletLine} renderOrder={5} />
          <mesh
            position={kelvinletPreview.start}
            scale={markerRadius}
            geometry={sharedSphereGeo}
            material={matKelvinlet}
            renderOrder={6}
          />
          <mesh
            position={kelvinletPreview.end}
            scale={markerRadius * 0.85}
            geometry={sharedSphereGeo}
            material={matCpActive}
            renderOrder={6}
          />
        </group>
      )}
    </group>
  );
}
