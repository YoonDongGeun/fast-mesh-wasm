// Viewport 3D cage, anchor, handle, and Kelvinlet pull gizmos for #193 deformation.
import { useThree } from "@react-three/fiber";
import type { SceneStore } from "fast-mesh-wasm/store";
import { useEffect, useLayoutEffect, useMemo, useRef } from "react";
import * as THREE from "three";
import type { AppState } from "../../../app/state/app-state";
import { deformMarkers, latticeCage } from "../ux/deform-gizmos";
import { useFfdCageDrag, useSurfaceDeformDrag } from "../ux/use-deform-drags";
import { useDeformState, useSceneRevision } from "../ux/use-deform-state";

export interface DeformHandlesLayerProps {
  app: AppState;
  store: SceneStore;
}

/** R3F viewport layer rendering FFD control cage, ARAP anchors/handles, and Kelvinlet pull gizmo. */
export function DeformHandlesLayer({ app, store }: DeformHandlesLayerProps) {
  const { camera, gl, raycaster, invalidate } = useThree();
  const { deform, sculptMask, activeMesh } = useDeformState(app, store);
  const sceneRev = useSceneRevision(store);

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

  const { cpPoints, edgePositions, sphereRadius } = useMemo(
    () => latticeCage(showFfd ? lattice : null),
    [showFfd, lattice],
  );

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
    let positions: Float32Array | null = null;
    if (showArapXpbd && activeMesh !== null) {
      try {
        positions = store.scene.views(activeMesh).positions();
      } catch {
        positions = null;
      }
    }
    return deformMarkers(positions, deform.arapAnchors, deform.arapHandles);
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

  const three = { camera, gl, raycaster, invalidate };
  const { hoveredCp, activeCp } = useFfdCageDrag({
    app,
    store,
    activeTool: deform.activeTool,
    activeMesh,
    hasLattice: lattice !== null,
    cpMeshRefs,
    deformRef,
    maskRef,
    three,
  });
  const kelvinletPreview = useSurfaceDeformDrag({
    app,
    store,
    activeTool: deform.activeTool,
    activeMesh,
    deformRef,
    maskRef,
    three,
  });

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
