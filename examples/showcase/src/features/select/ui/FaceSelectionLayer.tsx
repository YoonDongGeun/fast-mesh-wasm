// Interactive Face Selection viewport layer & 2D SVG marquee overlay (#233).
//
// Inside <Canvas>, <FaceSelectionLayer app={app} store={store} /> renders the
// selected-triangle overlay over visible meshes and handles box, lasso, brush,
// crease-angle, and connected-shell selection gestures via `store.scene.repair`.
// Outside <Canvas>, <FaceSelectionMarquee /> renders active box/lasso feedback.
import { useFrame, useThree } from "@react-three/fiber";
import { meshState, type SceneStore } from "fast-mesh-wasm/store";
import {
  useEffect,
  useLayoutEffect,
  useMemo,
  useRef,
  useSyncExternalStore,
} from "react";
import * as THREE from "three";
import { type AppState, type Item, useApp } from "../../../app/state/app-state";
import { view } from "../../../viewport/ux/view";
import { getFaceSelectionMarquee, subscribeFaceSelectionMarquee } from "../ux/face-selection-marquee";
import { isLiveTriangleSlot } from "../ux/face-sets";
import { useFaceSelectionTools } from "../ux/use-face-selection-tools";
import { useSceneRevision } from "../ux/use-scene-revision";

export interface FaceSelectionLayerProps {
  app: AppState;
  store: SceneStore;
}

const SELECTION_HIGHLIGHT_COLOR = "#ff8a00";
const SELECTION_HIGHLIGHT_OPACITY = 0.42;

const NOOP_RAYCAST = () => {};

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
  const [revision] = useSceneRevision(store);
  useFaceSelectionTools(app, store, camera, gl, selectState.activeTool);

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
