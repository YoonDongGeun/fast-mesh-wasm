// Interactive Face Selection viewport layer & 2D SVG marquee overlay (#233).
//
// Inside <Canvas>, <FaceSelectionLayer app={app} store={store} /> renders the
// selected-triangle overlay over visible meshes and handles box, lasso, brush,
// crease-angle, and connected-shell selection gestures via `store.scene.repair`.
// Outside <Canvas>, <FaceSelectionMarquee /> renders active box/lasso feedback.
import { useFrame, useThree } from "@react-three/fiber";
import type {
  MeshHandle,
  RayHit,
  ScreenSelectOptions,
  ScreenSelectionShape,
  Vec3,
} from "fast-mesh-wasm";
import { meshState, type SceneStore } from "fast-mesh-wasm/store";
import {
  useEffect,
  useLayoutEffect,
  useMemo,
  useRef,
  useState,
  useSyncExternalStore,
} from "react";
import * as THREE from "three";
import { type AppState, type Item, useApp } from "../scene/app-state";
import type { Wave6SelectState } from "../scene/wave6-state";
import { view } from "./view";

export interface FaceSelectionLayerProps {
  app: AppState;
  store: SceneStore;
}

export type FaceSelectionMarqueeState =
  | {
      kind: "box";
      x0: number;
      y0: number;
      x1: number;
      y1: number;
      width: number;
      height: number;
    }
  | {
      kind: "lasso";
      points: readonly (readonly [number, number])[];
      width: number;
      height: number;
    }
  | null;

const SELECTION_HIGHLIGHT_COLOR = "#ff8a00";
const SELECTION_HIGHLIGHT_OPACITY = 0.42;
const CLICK_DRAG_THRESHOLD_PX = 3;
const LASSO_MIN_STEP_PX = 3;
const RAYCAST_HIDDEN_SKIP_LIMIT = 8;

const EMPTY_U32 = new Uint32Array(0);
const NOOP_RAYCAST = () => {};

let currentMarquee: FaceSelectionMarqueeState = null;
const marqueeListeners = new Set<() => void>();

/** Returns the active 2D marquee overlay state used by {@link FaceSelectionMarquee}. */
export function getFaceSelectionMarquee(): FaceSelectionMarqueeState {
  return currentMarquee;
}

/** Publishes a new 2D marquee overlay state and notifies subscribers. */
export function setFaceSelectionMarquee(next: FaceSelectionMarqueeState): void {
  currentMarquee = next;
  for (const listener of marqueeListeners) {
    listener();
  }
}

/** Subscribes to changes of the 2D marquee overlay state. */
export function subscribeFaceSelectionMarquee(listener: () => void): () => void {
  marqueeListeners.add(listener);
  return () => {
    marqueeListeners.delete(listener);
  };
}

/** True when a triangle index slot holds a non-dead `[0, 0, 0]` triangle. */
export function isLiveTriangleSlot(indices: ArrayLike<number>, tri: number): boolean {
  const base = tri * 3;
  if (tri < 0 || base + 2 >= indices.length) {
    return false;
  }
  const a = indices[base];
  const b = indices[base + 1];
  const c = indices[base + 2];
  return !(a === 0 && b === 0 && c === 0);
}

/** Counts non-dead triangles in an index buffer. */
export function countLiveTriangles(indices: ArrayLike<number>): number {
  const total = Math.floor(indices.length / 3);
  let live = 0;
  for (let t = 0; t < total; t++) {
    if (isLiveTriangleSlot(indices, t)) {
      live++;
    }
  }
  return live;
}

/** Returns sorted, deduplicated non-negative integer triangle indices. */
function toSortedUniqueUint32(values: ArrayLike<number>): Uint32Array {
  if (values.length === 0) {
    return EMPTY_U32;
  }
  const clean: number[] = [];
  for (let i = 0; i < values.length; i++) {
    const v = values[i];
    if (Number.isInteger(v) && v >= 0) {
      clean.push(v);
    }
  }
  if (clean.length === 0) {
    return EMPTY_U32;
  }
  clean.sort((a, b) => a - b);
  let unique = 1;
  for (let i = 1; i < clean.length; i++) {
    if (clean[i] !== clean[unique - 1]) {
      clean[unique++] = clean[i];
    }
  }
  const out = new Uint32Array(unique);
  for (let i = 0; i < unique; i++) {
    out[i] = clean[i];
  }
  return out;
}

/** Combines `current` and `incoming` triangle selections according to `mode`. */
export function combineFaceSelection(
  current: Uint32Array | ArrayLike<number>,
  incoming: Uint32Array | ArrayLike<number>,
  mode: Wave6SelectState["combineMode"],
): Uint32Array {
  const b = toSortedUniqueUint32(incoming);
  if (mode === "replace") {
    return b;
  }
  const a = toSortedUniqueUint32(current);
  if (mode === "add") {
    const out: number[] = [];
    let i = 0;
    let j = 0;
    while (i < a.length && j < b.length) {
      if (a[i] < b[j]) {
        out.push(a[i++]);
      } else if (b[j] < a[i]) {
        out.push(b[j++]);
      } else {
        out.push(a[i]);
        i++;
        j++;
      }
    }
    while (i < a.length) out.push(a[i++]);
    while (j < b.length) out.push(b[j++]);
    return Uint32Array.from(out);
  }
  if (mode === "subtract") {
    if (a.length === 0 || b.length === 0) {
      return a;
    }
    const out: number[] = [];
    let i = 0;
    let j = 0;
    while (i < a.length && j < b.length) {
      if (a[i] < b[j]) {
        out.push(a[i++]);
      } else if (b[j] < a[i]) {
        j++;
      } else {
        i++;
        j++;
      }
    }
    while (i < a.length) out.push(a[i++]);
    return Uint32Array.from(out);
  }
  // mode === "intersect"
  if (a.length === 0 || b.length === 0) {
    return EMPTY_U32;
  }
  const out: number[] = [];
  let i = 0;
  let j = 0;
  while (i < a.length && j < b.length) {
    if (a[i] < b[j]) {
      i++;
    } else if (b[j] < a[i]) {
      j++;
    } else {
      out.push(a[i]);
      i++;
      j++;
    }
  }
  return Uint32Array.from(out);
}

/** Filters `selected` to unique live triangle slots within `indices`. */
export function sanitizeSelectedTriangles(
  indices: ArrayLike<number>,
  selected: Uint32Array | ArrayLike<number>,
): Uint32Array {
  const sorted = toSortedUniqueUint32(selected);
  if (sorted.length === 0) {
    return EMPTY_U32;
  }
  const out: number[] = [];
  for (let i = 0; i < sorted.length; i++) {
    const tri = sorted[i];
    if (isLiveTriangleSlot(indices, tri)) {
      out.push(tri);
    }
  }
  return out.length === 0 ? EMPTY_U32 : Uint32Array.from(out);
}

/** Inverts `selected` over all live triangle slots in `indices`. */
export function invertFaceSelection(
  indices: ArrayLike<number>,
  selected: Uint32Array | ArrayLike<number>,
): Uint32Array {
  const triCount = Math.floor(indices.length / 3);
  if (triCount <= 0) {
    return EMPTY_U32;
  }
  const excluded = new Set<number>();
  for (let i = 0; i < selected.length; i++) {
    excluded.add(selected[i]);
  }
  const out: number[] = [];
  for (let t = 0; t < triCount; t++) {
    if (!excluded.has(t) && isLiveTriangleSlot(indices, t)) {
      out.push(t);
    }
  }
  return Uint32Array.from(out);
}

/**
 * Builds a non-indexed `THREE.BufferGeometry` containing only the valid,
 * non-dead triangles listed in `selectedTris` for crisp face highlighting.
 */
export function buildSelectedFacesGeometry(
  positions: ArrayLike<number>,
  indices: ArrayLike<number>,
  selectedTris: ArrayLike<number>,
): THREE.BufferGeometry {
  const geometry = new THREE.BufferGeometry();
  const validTris: number[] = [];
  const seen = new Set<number>();
  const vertCount = Math.floor(positions.length / 3);

  for (let i = 0; i < selectedTris.length; i++) {
    const tri = selectedTris[i];
    if (!Number.isInteger(tri) || seen.has(tri) || !isLiveTriangleSlot(indices, tri)) {
      continue;
    }
    const base = tri * 3;
    const a = indices[base];
    const b = indices[base + 1];
    const c = indices[base + 2];
    if (a >= vertCount || b >= vertCount || c >= vertCount) {
      continue;
    }
    seen.add(tri);
    validTris.push(tri);
  }

  const outPositions = new Float32Array(validTris.length * 9);
  for (let i = 0; i < validTris.length; i++) {
    const base = validTris[i] * 3;
    const dst = i * 9;
    const va = indices[base] * 3;
    const vb = indices[base + 1] * 3;
    const vc = indices[base + 2] * 3;
    outPositions[dst] = positions[va];
    outPositions[dst + 1] = positions[va + 1];
    outPositions[dst + 2] = positions[va + 2];
    outPositions[dst + 3] = positions[vb];
    outPositions[dst + 4] = positions[vb + 1];
    outPositions[dst + 5] = positions[vb + 2];
    outPositions[dst + 6] = positions[vc];
    outPositions[dst + 7] = positions[vc + 1];
    outPositions[dst + 8] = positions[vc + 2];
  }

  geometry.setAttribute("position", new THREE.BufferAttribute(outPositions, 3));
  if (validTris.length > 0) {
    geometry.computeVertexNormals();
    geometry.computeBoundingSphere();
  }
  return geometry;
}

/** Converts client pixel coordinates inside `rect` to normalized device coordinates `[-1, 1]`. */
export function clientPointToNdc(
  clientX: number,
  clientY: number,
  rect: Pick<DOMRect, "left" | "top" | "width" | "height">,
): [number, number] {
  const width = Math.max(1, rect.width);
  const height = Math.max(1, rect.height);
  const ndcX = ((clientX - rect.left) / width) * 2 - 1;
  const ndcY = 1 - ((clientY - rect.top) / height) * 2;
  return [ndcX, ndcY];
}

function localPointToNdc(x: number, y: number, width: number, height: number): [number, number] {
  const w = Math.max(1, width);
  const h = Math.max(1, height);
  return [(x / w) * 2 - 1, 1 - (y / h) * 2];
}

/** Extracts `ScreenSelectOptions` from a Three.js camera for `repair.selectFacesScreen`. */
export function buildScreenSelectOptions(
  camera: THREE.Camera,
  visibleOnly: boolean,
  rejectBackfaces: boolean,
  includeTouching = true,
): ScreenSelectOptions {
  camera.updateMatrixWorld();
  camera.matrixWorldInverse.copy(camera.matrixWorld).invert();
  const viewProjMatrix = new THREE.Matrix4().multiplyMatrices(
    camera.projectionMatrix,
    camera.matrixWorldInverse,
  );
  const origin = new THREE.Vector3();
  const forward = new THREE.Vector3();
  camera.getWorldPosition(origin);
  camera.getWorldDirection(forward);
  if (forward.lengthSq() === 0) {
    forward.set(0, 0, -1);
  }
  return {
    viewProj: viewProjMatrix.toArray(),
    cameraOrigin: [origin.x, origin.y, origin.z],
    cameraForward: [forward.x, forward.y, forward.z],
    isOrthographic: Boolean((camera as THREE.OrthographicCamera).isOrthographicCamera),
    visibleOnly,
    rejectBackfaces,
    includeTouching,
  };
}

function isItemVisible(store: SceneStore, mesh: MeshHandle): boolean {
  const st = meshState(store.getSnapshot(), mesh);
  return st ? st.desired.visible : true;
}

/** Resolves the active item (primary selection when present, else first visible item). */
export function resolveActiveSelectItem(app: AppState, store: SceneStore): Item | null {
  const items = app.getSnapshot().items;
  const primary = store.getSnapshot().selection;
  if (primary !== null) {
    const matched = items.find((it) => it.mesh === primary);
    if (matched) {
      return matched;
    }
  }
  return items.find((it) => isItemVisible(store, it.mesh)) ?? items[0] ?? null;
}

/** Commits a new triangle selection array for `itemId` into `wave6.select.selectedByMesh`. */
export function writeMeshFaceSelection(app: AppState, itemId: number, selected: Uint32Array): void {
  app.updateWave6((prev) => ({
    ...prev,
    select: {
      ...prev.select,
      selectedByMesh: {
        ...prev.select.selectedByMesh,
        [itemId]: selected,
      },
    },
  }));
}

/** Raycasts visible scene meshes through `store.scene.inspect.raycast`, skipping hidden meshes. */
export async function raycastVisibleItem(
  store: SceneStore,
  items: readonly Item[],
  origin: Readonly<Vec3>,
  direction: Readonly<Vec3>,
  backfaceCull: boolean,
): Promise<{ item: Item; hit: RayHit } | null> {
  const inspect = store.scene.inspect;
  if (!inspect) {
    return null;
  }
  const byMesh = new Map<MeshHandle, Item>();
  for (const item of items) {
    if (isItemVisible(store, item.mesh)) {
      byMesh.set(item.mesh, item);
    }
  }
  if (byMesh.size === 0) {
    return null;
  }

  let minDistance = 0;
  for (let step = 0; step < RAYCAST_HIDDEN_SKIP_LIMIT; step++) {
    const hit = await Promise.resolve(
      inspect.raycast(origin, direction, {
        minDistance,
        cull: backfaceCull ? "back" : "none",
      }),
    );
    if (!hit) {
      return null;
    }
    const item = byMesh.get(hit.mesh);
    if (item) {
      return { item, hit };
    }
    minDistance = hit.distance + 1e-4;
  }
  return null;
}

/** Runs `repair.selectFacesScreen`, combines with `mode`, and stores into `app.wave6.select`. */
export async function applyScreenFaceSelection(params: {
  app: AppState;
  store: SceneStore;
  item: Item;
  shape: ScreenSelectionShape;
  options: ScreenSelectOptions;
  combineMode: Wave6SelectState["combineMode"];
}): Promise<Uint32Array> {
  const repair = params.store.scene.repair;
  if (!repair) {
    return EMPTY_U32;
  }
  const incoming = await Promise.resolve(
    repair.selectFacesScreen(params.item.mesh, params.shape, params.options),
  );
  const current = params.app.getSnapshot().wave6.select.selectedByMesh[params.item.id] ?? EMPTY_U32;
  const combined = combineFaceSelection(current, incoming, params.combineMode);
  writeMeshFaceSelection(params.app, params.item.id, combined);
  return combined;
}

function SelectionHighlightMesh(props: {
  item: Item;
  selected: Uint32Array;
  store: SceneStore;
  revision: number;
}) {
  const { item, selected, store, revision } = props;
  const meshRef = useRef<THREE.Mesh>(null);
  const matRef = useRef<THREE.MeshBasicMaterial>(null);

  const geometry = useMemo(() => {
    void revision;
    const meshView = store.scene.views(item.mesh);
    const positions = meshView.positions();
    const indices = meshView.indices();
    if (!positions || !indices || selected.length === 0) {
      return new THREE.BufferGeometry();
    }
    return buildSelectedFacesGeometry(positions, indices, selected);
  }, [item.mesh, selected, store, revision]);

  useEffect(() => () => geometry.dispose(), [geometry]);

  const syncTransform = () => {
    const overlay = meshRef.current;
    if (!overlay) {
      return;
    }
    overlay.raycast = NOOP_RAYCAST;
    const bound = view.bindings.get(item.mesh)?.object;
    if (bound) {
      overlay.matrix.copy(bound.matrixWorld);
      overlay.matrixWorld.copy(bound.matrixWorld);
      const hostMat = (bound as THREE.Mesh).material;
      if (
        matRef.current &&
        hostMat &&
        !Array.isArray(hostMat) &&
        matRef.current.clippingPlanes !== hostMat.clippingPlanes
      ) {
        matRef.current.clippingPlanes = hostMat.clippingPlanes;
      }
      return;
    }
    const raw = store.scene.matrix4(item.mesh);
    overlay.matrix.fromArray(raw);
    overlay.matrixWorld.copy(overlay.matrix);
  };

  useLayoutEffect(syncTransform);
  useFrame(syncTransform);

  return (
    <mesh
      ref={meshRef}
      geometry={geometry}
      matrixAutoUpdate={false}
      frustumCulled={false}
      renderOrder={2}
      raycast={NOOP_RAYCAST}
    >
      <meshBasicMaterial
        ref={matRef}
        color={SELECTION_HIGHLIGHT_COLOR}
        transparent
        opacity={SELECTION_HIGHLIGHT_OPACITY}
        side={THREE.DoubleSide}
        depthWrite={false}
        depthTest
        polygonOffset
        polygonOffsetFactor={-2}
        polygonOffsetUnits={-2}
        toneMapped={false}
      />
    </mesh>
  );
}

/** R3F viewport layer highlighting selected triangles and handling interactive face-selection tools. */
export function FaceSelectionLayer({ app, store }: FaceSelectionLayerProps) {
  const { camera, gl } = useThree();
  const selectState = useApp(app, (s) => s.wave6.select);
  const items = useApp(app, (s) => s.items);
  const storeSnap = useSyncExternalStore(store.subscribe, () => store.getSnapshot());
  const [revision, setRevision] = useState(0);
  const orbitLockRef = useRef(false);

  useEffect(() => {
    const unsub = store.scene.onChanges((changes) => {
      if (changes.length > 0) {
        setRevision((r) => r + 1);
      }
    });
    return unsub;
  }, [store]);

  useEffect(() => {
    const dom = gl.domElement;
    const activeTool = selectState.activeTool;

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

    const localCoords = (e: PointerEvent, rect: DOMRect): [number, number] => [
      Math.max(0, Math.min(rect.width, e.clientX - rect.left)),
      Math.max(0, Math.min(rect.height, e.clientY - rect.top)),
    ];

    const rayFromClient = (clientX: number, clientY: number, rect: DOMRect): { origin: Vec3; direction: Vec3 } => {
      const [nx, ny] = clientPointToNdc(clientX, clientY, rect);
      const raycaster = new THREE.Raycaster();
      raycaster.setFromCamera(new THREE.Vector2(nx, ny), camera);
      return {
        origin: [raycaster.ray.origin.x, raycaster.ray.origin.y, raycaster.ray.origin.z],
        direction: [raycaster.ray.direction.x, raycaster.ray.direction.y, raycaster.ray.direction.z],
      };
    };

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
        const ray = rayFromClient(clientX, clientY, rect);
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
      const ray = rayFromClient(clientX, clientY, rect);
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

    const rectShapeFromPixels = (
      x0: number,
      y0: number,
      x1: number,
      y1: number,
      rect: DOMRect,
      padPx: number,
    ): ScreenSelectionShape => {
      const a = localPointToNdc(Math.min(x0, x1) - padPx, Math.min(y0, y1) - padPx, rect.width, rect.height);
      const b = localPointToNdc(Math.max(x0, x1) + padPx, Math.max(y0, y1) + padPx, rect.width, rect.height);
      return {
        kind: "rect",
        minNdc: [Math.min(a[0], b[0]), Math.min(a[1], b[1])],
        maxNdc: [Math.max(a[0], b[0]), Math.max(a[1], b[1])],
      };
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
        const pad =
          Math.hypot(completed.x1 - completed.x0, completed.y1 - completed.y0) < CLICK_DRAG_THRESHOLD_PX
            ? 4
            : 0;
        void applyScreenFaceSelection({
          app,
          store,
          item: completed.item,
          shape: rectShapeFromPixels(
            completed.x0,
            completed.y0,
            completed.x1,
            completed.y1,
            completed.rect,
            pad,
          ),
          options,
          combineMode: cfg.combineMode,
        });
        return;
      }

      if (completed.kind === "lasso") {
        const shape: ScreenSelectionShape =
          completed.points.length >= 3
            ? {
                kind: "lasso",
                polygonNdc: completed.points.map(([px, py]) =>
                  localPointToNdc(px, py, completed.rect.width, completed.rect.height),
                ),
              }
            : rectShapeFromPixels(
                completed.points[0]?.[0] ?? 0,
                completed.points[0]?.[1] ?? 0,
                completed.points[0]?.[0] ?? 0,
                completed.points[0]?.[1] ?? 0,
                completed.rect,
                4,
              );
        void applyScreenFaceSelection({
          app,
          store,
          item: completed.item,
          shape,
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
  }, [app, store, camera, gl, selectState.activeTool]);

  return (
    <group name="wave6-face-selection-layer">
      {items.map((item) => {
        const selected = selectState.selectedByMesh[item.id];
        if (!selected || selected.length === 0) {
          return null;
        }
        const st = meshState(storeSnap, item.mesh);
        if (st && !st.desired.visible) {
          return null;
        }
        return (
          <SelectionHighlightMesh
            key={item.id}
            item={item}
            selected={selected}
            store={store}
            revision={revision}
          />
        );
      })}
    </group>
  );
}

/**
 * 2D DOM `<svg>` marquee overlay intended to render as a sibling outside `<Canvas>`
 * inside the viewport container with zero prop plumbing.
 */
export function FaceSelectionMarquee() {
  const marquee = useSyncExternalStore(
    subscribeFaceSelectionMarquee,
    getFaceSelectionMarquee,
    getFaceSelectionMarquee,
  );
  if (!marquee) {
    return null;
  }

  const w = Math.max(1, marquee.width);
  const h = Math.max(1, marquee.height);

  return (
    <svg
      aria-hidden="true"
      data-select-marquee-overlay="true"
      viewBox={`0 0 ${w} ${h}`}
      preserveAspectRatio="none"
      style={{
        position: "absolute",
        inset: 0,
        width: "100%",
        height: "100%",
        pointerEvents: "none",
        zIndex: 4,
      }}
    >
      {marquee.kind === "box" ? (
        <rect
          data-select-marquee="box"
          x={Math.min(marquee.x0, marquee.x1)}
          y={Math.min(marquee.y0, marquee.y1)}
          width={Math.abs(marquee.x1 - marquee.x0)}
          height={Math.abs(marquee.y1 - marquee.y0)}
          fill="rgba(255, 138, 0, 0.14)"
          stroke={SELECTION_HIGHLIGHT_COLOR}
          strokeWidth={1.5}
          strokeDasharray="5 3"
        />
      ) : (
        <polygon
          data-select-marquee="lasso"
          points={marquee.points.map(([px, py]) => `${px},${py}`).join(" ")}
          fill="rgba(255, 138, 0, 0.14)"
          stroke={SELECTION_HIGHLIGHT_COLOR}
          strokeWidth={1.5}
          strokeDasharray="5 3"
        />
      )}
    </svg>
  );
}
