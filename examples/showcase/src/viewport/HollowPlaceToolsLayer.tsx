import type {
  BooleanMethod,
  DrainHoleOptions,
  DrainHoleReport,
  HollowOptions,
  HollowReport,
  MeshHandle,
  PlaceMeshOptions,
  PlaceMeshReport,
  PlaceUpAxis,
  Scene,
  SceneLike,
  Vec3,
} from "fast-mesh-wasm";
import type { SceneStore } from "fast-mesh-wasm/store";
import type { WorkerScene } from "fast-mesh-wasm/worker";
import { useThree } from "@react-three/fiber";
import { useEffect, useMemo, useRef } from "react";
import * as THREE from "three";
import { type AppState, useApp } from "../scene/app-state";
import type { Wave6PrintPrepState } from "../scene/wave6-state";
import { raycastVisibleSurface } from "./Viewport";
import { view } from "./view";

function tapSlop(pointerType: string, fallback = 4): number {
  return pointerType === "touch" ? 10 : fallback;
}

export interface HollowPlaceToolsLayerProps {
  app: AppState;
  store: SceneStore;
}

export interface SurfaceOrientation {
  axis: Vec3;
  quaternion: [number, number, number, number];
  matrix4: number[];
}

export type BuiltinStampKind = "peg" | "hexBoss" | "dome" | "cube";

const ALLOWED_UP_AXES: readonly PlaceUpAxis[] = ["negZ", "posZ", "posY", "negY"];

const alignToNormalByApp = new WeakMap<AppState, boolean>();

export function getPlaceAlignToNormal(app: AppState): boolean {
  return alignToNormalByApp.get(app) ?? true;
}

export function setPlaceAlignToNormal(app: AppState, align: boolean): void {
  alignToNormalByApp.set(app, align);
}

function isAllowedUpAxis(axis: string): axis is PlaceUpAxis {
  return (ALLOWED_UP_AXES as readonly string[]).includes(axis);
}

/**
 * Computes an orientation aligning local +Z with `normal`, optionally twisted around `normal`.
 */
export function orientationFromSurfaceNormal(normal: Readonly<Vec3>, twistRad = 0): SurfaceOrientation {
  const n = new THREE.Vector3(normal[0], normal[1], normal[2]);
  if (n.lengthSq() <= 1e-24 || !Number.isFinite(n.x) || !Number.isFinite(n.y) || !Number.isFinite(n.z)) {
    n.set(0, 0, 1);
  } else {
    n.normalize();
  }
  const q = new THREE.Quaternion().setFromUnitVectors(new THREE.Vector3(0, 0, 1), n);
  if (Number.isFinite(twistRad) && twistRad !== 0) {
    const twist = new THREE.Quaternion().setFromAxisAngle(n, twistRad);
    q.premultiply(twist);
  }
  q.normalize();
  const m = new THREE.Matrix4().makeRotationFromQuaternion(q);
  return {
    axis: [n.x, n.y, n.z],
    quaternion: [q.x, q.y, q.z, q.w],
    matrix4: Array.from(m.elements),
  };
}

/**
 * Computes world-space hit point and outward unit normal from a scene raycast hit.
 */
export function computeHitPointAndWorldNormal(
  scene: Pick<SceneLike, "views" | "matrix4">,
  hit: { mesh: MeshHandle; triangle: number; point: Readonly<Vec3> },
): { point: Vec3; normal: Vec3 } | null {
  const views = scene.views(hit.mesh);
  const indices = views.indices();
  const positions = views.positions();
  const t = hit.triangle;
  if (t < 0 || t * 3 + 2 >= indices.length) {
    return null;
  }
  const i0 = indices[t * 3] * 3;
  const i1 = indices[t * 3 + 1] * 3;
  const i2 = indices[t * 3 + 2] * 3;
  const ax = positions[i0];
  const ay = positions[i0 + 1];
  const az = positions[i0 + 2];
  const e1x = positions[i1] - ax;
  const e1y = positions[i1 + 1] - ay;
  const e1z = positions[i1 + 2] - az;
  const e2x = positions[i2] - ax;
  const e2y = positions[i2 + 1] - ay;
  const e2z = positions[i2 + 2] - az;
  const localNormal = new THREE.Vector3(
    e1y * e2z - e1z * e2y,
    e1z * e2x - e1x * e2z,
    e1x * e2y - e1y * e2x,
  );
  if (localNormal.lengthSq() <= 1e-24) {
    return null;
  }
  const modelMatrix = new THREE.Matrix4().fromArray(Array.from(scene.matrix4(hit.mesh)));
  const normalMatrix = new THREE.Matrix3().getNormalMatrix(modelMatrix);
  const worldNormal = localNormal.applyMatrix3(normalMatrix);
  if (worldNormal.lengthSq() <= 1e-24) {
    return null;
  }
  worldNormal.normalize();
  return {
    point: [hit.point[0], hit.point[1], hit.point[2]],
    normal: [worldNormal.x, worldNormal.y, worldNormal.z],
  };
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

function closedPrismGeometry(sides: number, radius: number, height: number): {
  positions: Float64Array;
  indices: Uint32Array;
} {
  const positions: number[] = [0, 0, 0, 0, 0, height];
  for (let i = 0; i < sides; i++) {
    const theta = (2 * Math.PI * i) / sides;
    const x = radius * Math.cos(theta);
    const y = radius * Math.sin(theta);
    positions.push(x, y, 0);
  }
  for (let i = 0; i < sides; i++) {
    const theta = (2 * Math.PI * i) / sides;
    const x = radius * Math.cos(theta);
    const y = radius * Math.sin(theta);
    positions.push(x, y, height);
  }
  const indices: number[] = [];
  const botStart = 2;
  const topStart = 2 + sides;
  for (let i = 0; i < sides; i++) {
    const next = (i + 1) % sides;
    const b0 = botStart + i;
    const b1 = botStart + next;
    const t0 = topStart + i;
    const t1 = topStart + next;
    indices.push(0, b1, b0);
    indices.push(1, t0, t1);
    indices.push(b0, b1, t1, b0, t1, t0);
  }
  return {
    positions: Float64Array.from(positions),
    indices: Uint32Array.from(indices),
  };
}

function closedDomeGeometry(radius: number, stacks: number, slices: number): {
  positions: Float64Array;
  indices: Uint32Array;
} {
  const positions: number[] = [0, 0, 0];
  for (let s = 0; s < stacks; s++) {
    const phi = (Math.PI * 0.5 * (stacks - s)) / stacks;
    const r = radius * Math.sin(phi);
    const z = radius * Math.cos(phi);
    for (let j = 0; j < slices; j++) {
      const theta = (2 * Math.PI * j) / slices;
      positions.push(r * Math.cos(theta), r * Math.sin(theta), z);
    }
  }
  positions.push(0, 0, radius);
  const pole = positions.length / 3 - 1;
  const ringVertex = (s: number, j: number) => 1 + s * slices + (j % slices);
  const indices: number[] = [];
  for (let j = 0; j < slices; j++) {
    indices.push(0, ringVertex(0, j + 1), ringVertex(0, j));
  }
  for (let s = 0; s < stacks - 1; s++) {
    for (let j = 0; j < slices; j++) {
      const a = ringVertex(s, j);
      const b = ringVertex(s, j + 1);
      const c = ringVertex(s + 1, j + 1);
      const d = ringVertex(s + 1, j);
      indices.push(a, b, c, a, c, d);
    }
  }
  const topRing = stacks - 1;
  for (let j = 0; j < slices; j++) {
    indices.push(ringVertex(topRing, j), ringVertex(topRing, j + 1), pole);
  }
  return {
    positions: Float64Array.from(positions),
    indices: Uint32Array.from(indices),
  };
}

/**
 * Generates a closed, outward-wound procedural stamp primitive oriented along +Z (base at z=0).
 */
export function createBuiltinStampMeshGeometry(kind: BuiltinStampKind): {
  positions: Float64Array;
  indices: Uint32Array;
} {
  switch (kind) {
    case "peg":
      return closedPrismGeometry(16, 1.5, 4.0);
    case "hexBoss":
      return closedPrismGeometry(6, 2.0, 3.0);
    case "dome":
      return closedDomeGeometry(2.0, 6, 16);
    case "cube": {
      const hx = 1.5;
      const hy = 1.5;
      const z0 = 0;
      const z1 = 3.0;
      const positions = Float64Array.from([
        -hx, -hy, z0,
        hx, -hy, z0,
        hx, hy, z0,
        -hx, hy, z0,
        -hx, -hy, z1,
        hx, -hy, z1,
        hx, hy, z1,
        -hx, hy, z1,
      ]);
      const indices = Uint32Array.from([
        0, 2, 1, 0, 3, 2,
        4, 5, 6, 4, 6, 7,
        0, 1, 5, 0, 5, 4,
        2, 3, 7, 2, 7, 6,
        1, 2, 6, 1, 6, 5,
        3, 0, 4, 3, 4, 7,
      ]);
      return { positions, indices };
    }
  }
}

function triangleCountOf(scene: Pick<SceneLike, "views">, mesh: MeshHandle): number {
  return Math.trunc(scene.views(mesh).indices().length / 3);
}

export async function runHollowMeshOp(args: {
  store: SceneStore;
  mesh: MeshHandle;
  options: HollowOptions;
}): Promise<{ beforeTriangles: number; afterTriangles: number; report: HollowReport }> {
  const { store, mesh, options } = args;
  const booleanDomain = store.scene.boolean;
  if (!booleanDomain?.hollowMesh) {
    throw new Error("hollowMesh is not available on this scene.");
  }
  const beforeTriangles = triangleCountOf(store.scene, mesh);
  const report = await booleanDomain.hollowMesh(mesh, options);
  await store.commitFrame();
  const afterTriangles = triangleCountOf(store.scene, mesh);
  return { beforeTriangles, afterTriangles, report };
}

export interface ExecuteDrainHoleArgs {
  store: SceneStore;
  app: AppState;
  target: MeshHandle;
  targetName?: string;
  hitPoint: Readonly<Vec3>;
  hitNormal: Readonly<Vec3>;
  drainParams: Wave6PrintPrepState["drainParams"];
  through?: boolean;
  booleanMethod?: BooleanMethod;
  plugClearance?: number;
}

export async function executeDrainHoleAtHit(args: ExecuteDrainHoleArgs): Promise<{
  beforeTriangles: number;
  afterTriangles: number;
  plugs: MeshHandle[];
  report: DrainHoleReport;
}> {
  const { store, app, target, hitPoint, hitNormal, drainParams } = args;
  const booleanDomain = store.scene.boolean;
  if (!booleanDomain?.punchDrainHole) {
    throw new Error("punchDrainHole is not available on this scene.");
  }
  const beforeTriangles = triangleCountOf(store.scene, target);
  const diameter = Math.max(0.1, drainParams.diameter);
  const outerChamfer = Math.max(0, Math.min(drainParams.outerChamfer, diameter * 0.45));
  const depth = args.through ? undefined : Math.max(drainParams.depth, outerChamfer + 0.05);
  const plugClearance = args.plugClearance ?? Math.min(0.15, diameter * 0.2);

  const opts: DrainHoleOptions = {
    surfacePoint: [hitPoint[0], hitPoint[1], hitPoint[2]],
    surfaceNormal: [hitNormal[0], hitNormal[1], hitNormal[2]],
    diameter,
    ...(depth !== undefined ? { depth } : {}),
    outerChamfer,
    taperAngleDeg: drainParams.taperDeg,
    generatePlug: drainParams.generatePlug,
    ...(drainParams.generatePlug ? { plugClearance } : {}),
    ...(args.booleanMethod !== undefined ? { booleanMethod: args.booleanMethod } : {}),
  };

  const report = await booleanDomain.punchDrainHole(target, opts);
  const baseName = args.targetName ?? "mesh";
  for (let i = 0; i < report.plugs.length; i++) {
    const plugName = report.plugs.length > 1 ? `${baseName} plug ${i + 1}` : `${baseName} plug`;
    app.adoptSceneMesh(store, report.plugs[i], plugName);
  }
  await store.commitFrame();
  const afterTriangles = triangleCountOf(store.scene, target);
  return { beforeTriangles, afterTriangles, plugs: report.plugs, report };
}

export interface ExecutePlaceStampArgs {
  store: SceneStore;
  app: AppState;
  target: MeshHandle;
  targetName?: string;
  hitPoint: Readonly<Vec3>;
  hitNormal: Readonly<Vec3>;
  placeParams: Wave6PrintPrepState["placeParams"];
  stampMesh?: MeshHandle | null;
  alignToNormal?: boolean;
  booleanMethod?: BooleanMethod;
}

export async function executePlaceStampAtHit(args: ExecutePlaceStampArgs): Promise<{
  beforeTriangles: number;
  afterTriangles: number;
  added: MeshHandle | null;
  report: PlaceMeshReport;
}> {
  const { store, app, target, hitPoint, hitNormal, placeParams } = args;
  const booleanDomain = store.scene.boolean;
  if (!booleanDomain?.placeMesh) {
    throw new Error("placeMesh is not available on this scene.");
  }
  if (!isAllowedUpAxis(placeParams.upAxis)) {
    throw new Error(`Unsupported placeMesh upAxis: ${String(placeParams.upAxis)}`);
  }

  let resolvedStamp: MeshHandle;
  let tempStamp: MeshHandle | null = null;
  let tempOwnedByStore = false;

  if (args.stampMesh !== undefined && args.stampMesh !== null) {
    resolvedStamp = args.stampMesh;
  } else if (placeParams.stampSource === "sceneMesh") {
    const items = app.getSnapshot().items;
    const stampItem =
      placeParams.stampMeshId === null
        ? null
        : (items.find((item) => item.id === placeParams.stampMeshId && item.mesh !== target) ?? null);
    if (stampItem === null) {
      throw new Error("Select a stamp source mesh other than the target.");
    }
    resolvedStamp = stampItem.mesh;
  } else {
    const geom = createBuiltinStampMeshGeometry(placeParams.stampSource);
    const rawScene = store.scene as Partial<Scene | WorkerScene>;
    if (typeof rawScene.addMesh === "function") {
      tempStamp = await rawScene.addMesh(geom.positions, geom.indices);
    } else {
      tempStamp = await store.addMesh(geom.positions, geom.indices);
      tempOwnedByStore = true;
    }
    resolvedStamp = tempStamp;
  }

  const beforeTriangles = triangleCountOf(store.scene, target);
  const effectiveNormal: Vec3 =
    args.alignToNormal === false ? [0, 0, 1] : [hitNormal[0], hitNormal[1], hitNormal[2]];

  const opts: PlaceMeshOptions = {
    hitPoint: [hitPoint[0], hitPoint[1], hitPoint[2]],
    hitNormal: effectiveNormal,
    upAxis: placeParams.upAxis,
    twistAngleRad: (placeParams.twistDeg * Math.PI) / 180,
    uniformScale: Math.max(0.01, placeParams.uniformScale),
    embedDepth: placeParams.embedDepth,
    mode: placeParams.mode,
    conformToSurface: placeParams.conformToSurface,
    ...(args.booleanMethod !== undefined ? { booleanMethod: args.booleanMethod } : {}),
  };

  try {
    const report = await booleanDomain.placeMesh(target, resolvedStamp, opts);
    if (report.added !== null) {
      app.adoptSceneMesh(store, report.added, `${args.targetName ?? "target"} stamp`);
    }
    await store.commitFrame();
    const afterTriangles = triangleCountOf(store.scene, target);
    return { beforeTriangles, afterTriangles, added: report.added, report };
  } finally {
    if (tempStamp !== null) {
      const rawScene = store.scene as Partial<Scene | WorkerScene>;
      if (!tempOwnedByStore && typeof rawScene.removeMesh === "function") {
        await rawScene.removeMesh(tempStamp);
      } else {
        await store.removeMesh(tempStamp);
      }
    }
  }
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

export function HollowPlaceToolsLayer({ app, store }: HollowPlaceToolsLayerProps) {
  const printPrep = useApp(app, (s) => s.wave6.printPrep);
  const items = useApp(app, (s) => s.items);
  const { camera, gl, raycaster } = useThree();
  const activeTool = printPrep.activeTool;
  const isToolActive = activeTool === "drainHole" || activeTool === "placeStamp";

  const busyRef = useRef(false);
  const downRef = useRef<{ x: number; y: number; pointerId: number } | null>(null);
  const printPrepRef = useRef(printPrep);
  printPrepRef.current = printPrep;
  const itemsRef = useRef(items);
  itemsRef.current = items;

  const preview = useMemo(() => {
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

    return { root, ring, ringMat, bore, boreMat, normalLine, normalMat, xAxis, yAxis, ringGeo, boreGeo, normalGeo, xAxisGeo, yAxisGeo, xAxisMat, yAxisMat };
  }, []);

  useEffect(() => {
    return () => {
      preview.ringGeo.dispose();
      preview.ringMat.dispose();
      preview.boreGeo.dispose();
      preview.boreMat.dispose();
      preview.normalGeo.dispose();
      preview.normalMat.dispose();
      preview.xAxisGeo.dispose();
      preview.xAxisMat.dispose();
      preview.yAxisGeo.dispose();
      preview.yAxisMat.dispose();
    };
  }, [preview]);

  useEffect(() => {
    if (!isToolActive) {
      preview.root.visible = false;
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
      const hit = await raycastVisibleSurface(
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

    const updatePreviewPose = (point: Vec3, normal: Vec3) => {
      const prep = printPrepRef.current;
      const tool = prep.activeTool;
      const effectiveNormal: Vec3 =
        tool === "placeStamp" && !getPlaceAlignToNormal(app) ? [0, 0, 1] : normal;
      const twistRad = tool === "placeStamp" ? (prep.placeParams.twistDeg * Math.PI) / 180 : 0;
      const orient = orientationFromSurfaceNormal(effectiveNormal, twistRad);

      preview.root.position.set(point[0], point[1], point[2]);
      preview.root.quaternion.set(
        orient.quaternion[0],
        orient.quaternion[1],
        orient.quaternion[2],
        orient.quaternion[3],
      );

      if (tool === "drainHole") {
        const radius = Math.max(0.25, prep.drainParams.diameter * 0.5);
        const depth = Math.max(1, prep.drainParams.depth);
        preview.ring.scale.set(radius, radius, 1);
        preview.bore.visible = true;
        preview.bore.scale.set(radius, radius, depth);
        preview.normalLine.scale.set(1, 1, Math.max(2, radius * 1.5));
        preview.xAxis.visible = false;
        preview.yAxis.visible = false;
        preview.ringMat.color.set("#38bdf8");
        preview.normalMat.color.set("#38bdf8");
      } else {
        const scale = Math.max(0.25, prep.placeParams.uniformScale);
        const ringR = 2 * scale;
        preview.ring.scale.set(ringR, ringR, 1);
        preview.bore.visible = false;
        preview.normalLine.scale.set(1, 1, Math.max(2.5, ringR * 1.25));
        preview.xAxis.visible = true;
        preview.xAxis.scale.set(ringR * 0.85, 1, 1);
        preview.yAxis.visible = true;
        preview.yAxis.scale.set(1, ringR * 0.85, 1);
        preview.ringMat.color.set("#f59e0b");
        preview.normalMat.color.set("#f59e0b");
      }
      preview.root.visible = true;
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
        preview.root.visible = false;
        return;
      }
      void raycastAtPointer(e.clientX, e.clientY).then((res) => {
        if (res === null) {
          preview.root.visible = false;
          return;
        }
        updatePreviewPose(res.point, res.normal);
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
      preview.root.visible = false;
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
      preview.root.visible = false;
    };
  }, [app, camera, gl, isToolActive, preview, raycaster, store]);

  return (
    <primitive
      object={preview.root}
      visible={isToolActive && printPrep.activeTool !== null && preview.root.visible}
    />
  );
}
