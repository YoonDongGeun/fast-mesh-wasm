// The canvas: `<FastMesh>` per mesh (click to select), an orbit camera and a
// TransformControls gizmo on the selected mesh (moves through
// `attachTransformControls`, one undo step per drag). `<FastMesh>` drives
// the store's frames. Colourings are drawn with fragment shading, so the
// heatmap's boundaries are crisp, softened by the collision panel's slider.
// With `hoverable` (Group mode) the vertex under the pointer is tracked in
// `view.hover`, for the hover tip.
import { Canvas, type ThreeEvent, useFrame, useThree } from "@react-three/fiber";
import type { MeshHandle } from "fast-mesh-wasm";
import { FastMesh } from "fast-mesh-wasm/r3f";
import { useScene, useSelection } from "fast-mesh-wasm/react";
import { attachTransformControls, type MeshBinding } from "fast-mesh-wasm/three";
import { useCallback, useEffect, useLayoutEffect, useRef, useState } from "react";
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { TransformControls } from "three/addons/controls/TransformControls.js";
import type { FocusRequest } from "../App";
import type { Item } from "../scene/app-state";
import { setDragging, setHover, view } from "./view";

export type GizmoMode = "translate" | "rotate";

export function Viewport(props: {
  items: readonly Item[];
  mode: GizmoMode;
  focus: FocusRequest | null;
  /** Fragment shading's edge softness, in distance units. */
  edgeSoftness: number;
  /** Track the vertex under the pointer (`view.hover`). */
  hoverable: boolean;
}) {
  const { items, mode, focus, edgeSoftness, hoverable } = props;
  const [selection, select] = useSelection();
  // Each mesh's binding, as `<FastMesh onBind>` delivers it: for the gizmo.
  const [bindings, setBindings] = useState<ReadonlyMap<MeshHandle, MeshBinding>>(() => new Map());
  const onBind = useCallback((mesh: MeshHandle, bound: MeshBinding | null) => {
    setBindings((prev) => {
      const next = new Map(prev);
      if (bound === null) {
        next.delete(mesh);
      } else {
        next.set(mesh, bound);
      }
      return next;
    });
  }, []);
  useEffect(() => {
    view.bindings = bindings;
  }, [bindings]);
  useEffect(() => {
    if (!hoverable) {
      setHover(null);
    }
  }, [hoverable]);
  const selected = selection === null ? null : (bindings.get(selection) ?? null);
  const radiusOf = useCallback((mesh: MeshHandle) => items.find((i) => i.mesh === mesh)?.radius ?? 1, [items]);

  return (
    <Canvas camera={{ position: [-0.6, 3.4, 8.2], fov: 45, near: 0.01, far: 5000 }}>
      <color attach="background" args={["#1b1e23"]} />
      <hemisphereLight args={["#e8eef8", "#30343c", 1.1]} />
      <directionalLight position={[3, 6, 4]} intensity={2} />
      <directionalLight position={[-4, -2, -3]} intensity={0.5} />
      <gridHelper args={[20, 20, "#3a404a", "#2a2f36"]} position={[0, -0.4, 0]} />
      {items.map((item) => (
        <MeshView
          key={item.id}
          item={item}
          selected={item.mesh === selection}
          edgeSoftness={edgeSoftness}
          hoverable={hoverable}
          onPick={select}
          onBind={onBind}
        />
      ))}
      <Controls bound={selected} mode={mode} focus={focus} radiusOf={radiusOf} />
    </Canvas>
  );
}

/** A mesh drawn with `<FastMesh>`: click to select it. */
function MeshView(props: {
  item: Item;
  selected: boolean;
  edgeSoftness: number;
  hoverable: boolean;
  onPick: (mesh: MeshHandle) => void;
  onBind: (mesh: MeshHandle, bound: MeshBinding | null) => void;
}) {
  const { item, selected, edgeSoftness, hoverable, onPick, onBind } = props;
  useLayoutEffect(() => {
    item.material.emissive.set(selected ? 0x3a2a08 : 0x000000);
  }, [item.material, selected]);
  const onClick = useCallback(
    (e: ThreeEvent<MouseEvent>) => {
      e.stopPropagation();
      // The end of an orbit drag that started on the mesh is not a click.
      if (e.delta > 2) {
        return;
      }
      onPick(item.mesh);
    },
    [item.mesh, onPick],
  );
  // The hit triangle's corner nearest to the hit point: the vertex hovered.
  const onPointerMove = useCallback(
    (e: ThreeEvent<PointerEvent>) => {
      if (e.face == null) {
        return;
      }
      e.stopPropagation();
      const position = (e.object as THREE.Mesh).geometry.getAttribute("position");
      const local = e.object.worldToLocal(e.point.clone());
      const corner = new THREE.Vector3();
      let slot = e.face.a;
      let best = Infinity;
      for (const v of [e.face.a, e.face.b, e.face.c]) {
        const d = corner.fromBufferAttribute(position, v).distanceToSquared(local);
        if (d < best) {
          best = d;
          slot = v;
        }
      }
      setHover({ mesh: item.mesh, slot, x: e.nativeEvent.clientX, y: e.nativeEvent.clientY });
    },
    [item.mesh],
  );
  const onPointerOut = useCallback(() => {
    if (view.hover?.mesh === item.mesh) {
      setHover(null);
    }
  }, [item.mesh]);
  return (
    <FastMesh
      mesh={item.mesh}
      material={item.material}
      shading="fragment"
      edgeSoftness={edgeSoftness}
      onClick={onClick}
      onPointerMove={hoverable ? onPointerMove : undefined}
      onPointerOut={hoverable ? onPointerOut : undefined}
      onBind={(b) => onBind(item.mesh, b)}
    />
  );
}

/**
 * Orbit camera; a TransformControls gizmo on `bound`; and the camera pointed
 * at a mesh on each new `focus` request.
 */
function Controls(props: {
  bound: MeshBinding | null;
  mode: GizmoMode;
  focus: FocusRequest | null;
  radiusOf: (mesh: MeshHandle) => number;
}) {
  const { bound, mode, focus, radiusOf } = props;
  const store = useScene();
  const camera = useThree((s) => s.camera);
  const gl = useThree((s) => s.gl);
  const scene = useThree((s) => s.scene);
  const [orbit, setOrbit] = useState<OrbitControls | null>(null);
  const [gizmo, setGizmo] = useState<TransformControls | null>(null);
  // Read when a focus request arrives: a new list of items is not a new request.
  const radiusRef = useRef(radiusOf);
  radiusRef.current = radiusOf;

  useEffect(() => {
    view.camera = camera;
    view.canvas = gl.domElement;
    const o = new OrbitControls(camera, gl.domElement);
    o.target.set(-0.6, 0.9, 0);
    o.update();
    setOrbit(o);
    return () => {
      o.dispose();
      setOrbit((c) => (c === o ? null : c));
      if (view.camera === camera) {
        view.camera = null;
        view.canvas = null;
      }
    };
  }, [camera, gl]);

  useFrame(() => orbit?.update());

  useEffect(() => {
    if (orbit === null || bound === null) {
      return;
    }
    const controls = new TransformControls(camera, gl.domElement);
    const helper = controls.getHelper();
    scene.add(helper);
    controls.attach(bound.object);
    const detach = attachTransformControls(controls, store.scene, bound);
    const onDragging = (e: { value: unknown }) => {
      orbit.enabled = e.value !== true;
      setDragging(e.value === true);
    };
    controls.addEventListener("dragging-changed", onDragging);
    view.gizmo = controls;
    setGizmo(controls);
    return () => {
      controls.removeEventListener("dragging-changed", onDragging);
      detach();
      controls.detach();
      scene.remove(helper);
      controls.dispose();
      orbit.enabled = true;
      setDragging(false);
      if (view.gizmo === controls) {
        view.gizmo = null;
      }
      setGizmo((g) => (g === controls ? null : g));
    };
  }, [store, camera, gl, scene, orbit, bound]);

  useEffect(() => {
    gizmo?.setMode(mode);
  }, [gizmo, mode]);

  // Focus: the orbit target to the mesh's centre (its origin, where the
  // gizmo sits), the camera back along its current direction.
  useEffect(() => {
    if (orbit === null || focus === null) {
      return;
    }
    let m: Float64Array;
    try {
      m = store.scene.matrix4(focus.mesh);
    } catch {
      return; // removed meanwhile
    }
    const center = new THREE.Vector3(m[12], m[13], m[14]);
    const radius = Math.max(radiusRef.current(focus.mesh), 1e-3);
    const dir = camera.position.clone().sub(orbit.target);
    if (dir.lengthSq() === 0) {
      dir.set(0, 0.4, 1);
    }
    const fov = camera instanceof THREE.PerspectiveCamera ? camera.fov : 45;
    const distance = (radius * 1.15) / Math.sin(THREE.MathUtils.degToRad(fov) / 2);
    orbit.target.copy(center);
    camera.position.copy(center).addScaledVector(dir.normalize(), distance);
    if (camera instanceof THREE.PerspectiveCamera) {
      camera.near = Math.max(distance / 1000, 1e-4);
      camera.far = Math.max(distance * 100, 100);
      camera.updateProjectionMatrix();
    }
    orbit.update();
  }, [orbit, focus, store, camera]);

  return null;
}
