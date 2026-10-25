// The canvas: `<FastMesh>` per mesh (click to select), an orbit camera and a
// TransformControls gizmo on the selected mesh (moves through
// `attachTransformControls`, one undo step per drag). `<FastMesh>` drives
// the store's frames. Colourings are drawn with fragment shading, so the
// heatmap's boundaries are crisp, softened by the collision panel's slider.
// With `hoverable` (Group mode) the vertex under the pointer is tracked in
// `view.hover`, for the hover tip: through the engine's BVH raycast
// (`scene.raycast`, async on the Worker), one at a time and at most every
// HOVER_PERIOD_MS, never three's per-triangle `Mesh.raycast`.
import { Canvas, type ThreeEvent, useFrame, useThree } from "@react-three/fiber";
import type { MeshHandle, Scene } from "fast-mesh-wasm";
import { FastMesh } from "fast-mesh-wasm/r3f";
import { useScene, useSelection } from "fast-mesh-wasm/react";
import { attachTransformControls, type MeshBinding } from "fast-mesh-wasm/three";
import { useCallback, useEffect, useLayoutEffect, useRef, useState } from "react";
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { TransformControls } from "three/addons/controls/TransformControls.js";
import type { WorkerScene } from "fast-mesh-wasm/worker";
import type { FocusRequest } from "../App";
import type { Item } from "../scene/app-state";
import { setDragging, setHover, view } from "./view";

export type GizmoMode = "translate" | "rotate";

/** The hover raycast runs at most this often while the pointer moves. */
const HOVER_PERIOD_MS = 50;

export function Viewport(props: {
  items: readonly Item[];
  mode: GizmoMode;
  focus: FocusRequest | null;
  /** Fragment shading's edge softness, in distance units. */
  edgeSoftness: number;
  /** Track the vertex under the pointer (`view.hover`). */
  hoverable: boolean;
  /** The engine's scene, for the hover raycast. */
  scene: Scene | WorkerScene;
  /** The mesh whose brush center is being picked, if the pick is armed. */
  picking: MeshHandle | null;
  /** A landed brush-center pick, in the mesh's local coordinates. */
  onPickPoint: (mesh: MeshHandle, point: readonly [number, number, number]) => void;
}) {
  const { items, mode, focus, edgeSoftness, hoverable, scene, picking, onPickPoint } = props;
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
          onPick={select}
          onBind={onBind}
          picking={picking === item.mesh}
          onPickPoint={onPickPoint}
          scene={scene}
        />
      ))}
      <Controls bound={selected} mode={mode} focus={focus} radiusOf={radiusOf} />
      {hoverable ? <HoverRay scene={scene} /> : null}
      <PickCursor picking={picking !== null} />
    </Canvas>
  );
}

/** A crosshair while a brush-center pick is armed. */
function PickCursor(props: { picking: boolean }) {
  const gl = useThree((s) => s.gl);
  useEffect(() => {
    gl.domElement.style.cursor = props.picking ? "crosshair" : "";
    return () => {
      gl.domElement.style.cursor = "";
    };
  }, [gl, props.picking]);
  return null;
}

/** A mesh drawn with `<FastMesh>`: click to select it, or to land an armed brush-center pick. */
function MeshView(props: {
  item: Item;
  selected: boolean;
  edgeSoftness: number;
  onPick: (mesh: MeshHandle) => void;
  onBind: (mesh: MeshHandle, bound: MeshBinding | null) => void;
  picking: boolean;
  onPickPoint: (mesh: MeshHandle, point: readonly [number, number, number]) => void;
  scene: Scene | WorkerScene;
}) {
  const { item, selected, edgeSoftness, onPick, onBind, picking, onPickPoint, scene } = props;
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
      if (!picking) {
        onPick(item.mesh);
        return;
      }
      // A pick click never selects: raycast and interpolate the hit in the
      // mesh's local positions, the frame the brush center is in. A hit on
      // another mesh (or nothing) leaves the pick armed.
      const { origin, direction } = e.ray;
      void Promise.resolve()
        .then(() => scene.raycast([origin.x, origin.y, origin.z], [direction.x, direction.y, direction.z]))
        .then((hit) => {
          if (hit === null || hit.mesh !== item.mesh) {
            return;
          }
          const corners = scene.views(hit.mesh).indices().subarray(3 * hit.triangle, 3 * hit.triangle + 3);
          const positions = scene.views(hit.mesh).positions();
          const [u, v, w] = hit.barycentric;
          const at = (s: number): readonly [number, number, number] => [
            positions[3 * s],
            positions[3 * s + 1],
            positions[3 * s + 2],
          ];
          const [a, b, c] = [at(corners[0]), at(corners[1]), at(corners[2])];
          onPickPoint(item.mesh, [
            u * a[0] + v * b[0] + w * c[0],
            u * a[1] + v * b[1] + w * c[1],
            u * a[2] + v * b[2] + w * c[2],
          ]);
        })
        .catch(() => {
          // Removed meanwhile: the pick stays armed.
        });
    },
    [item.mesh, onPick, picking, onPickPoint, scene],
  );
  return (
    <FastMesh
      mesh={item.mesh}
      material={item.material}
      shading="fragment"
      edgeSoftness={edgeSoftness}
      onClick={onClick}
      onBind={(b) => onBind(item.mesh, b)}
    />
  );
}

/**
 * The vertex under the pointer, into `view.hover`: the engine's raycast
 * (its BVH) from the camera through the pointer, the hit triangle's corner
 * with the largest barycentric weight. One raycast at a time, at most every
 * HOVER_PERIOD_MS, for the latest pointer position; none while a button is
 * down (orbiting, or the gizmo dragging). Cleared when the pointer leaves
 * the canvas and on unmount.
 */
function HoverRay(props: { scene: Scene | WorkerScene }) {
  const { scene } = props;
  const camera = useThree((s) => s.camera);
  const gl = useThree((s) => s.gl);

  useEffect(() => {
    const canvas = gl.domElement;
    const caster = new THREE.Raycaster();
    let alive = true;
    let pending: { x: number; y: number } | null = null;
    let running = false;
    let last = -Infinity;
    let timer: ReturnType<typeof setTimeout> | undefined;
    const run = () => {
      if (!alive || running || pending === null) {
        return;
      }
      const wait = last + HOVER_PERIOD_MS - performance.now();
      if (wait > 0) {
        timer ??= setTimeout(() => {
          timer = undefined;
          run();
        }, wait);
        return;
      }
      const at = pending;
      pending = null;
      running = true;
      last = performance.now();
      const r = canvas.getBoundingClientRect();
      const ndc = new THREE.Vector2(((at.x - r.left) / r.width) * 2 - 1, 1 - ((at.y - r.top) / r.height) * 2);
      caster.setFromCamera(ndc, camera);
      const { origin, direction } = caster.ray;
      Promise.resolve()
        .then(() => scene.raycast([origin.x, origin.y, origin.z], [direction.x, direction.y, direction.z]))
        .then(
          (hit) => {
            if (!alive) {
              return;
            }
            if (hit === null) {
              setHover(null);
              return;
            }
            const corners = scene.views(hit.mesh).indices().subarray(3 * hit.triangle, 3 * hit.triangle + 3);
            const [u, v, w] = hit.barycentric;
            const k = u >= v && u >= w ? 0 : v >= w ? 1 : 2;
            setHover({ mesh: hit.mesh, slot: corners[k], x: at.x, y: at.y });
          },
          () => {
            // A mesh removed (or the scene restarted) meanwhile: the next move asks again.
          },
        )
        .finally(() => {
          running = false;
          run();
        });
    };
    const onMove = (e: PointerEvent) => {
      if (e.buttons !== 0 || view.dragging) {
        return;
      }
      pending = { x: e.clientX, y: e.clientY };
      run();
    };
    const onLeave = () => {
      pending = null;
      setHover(null);
    };
    canvas.addEventListener("pointermove", onMove);
    canvas.addEventListener("pointerleave", onLeave);
    return () => {
      alive = false;
      clearTimeout(timer);
      canvas.removeEventListener("pointermove", onMove);
      canvas.removeEventListener("pointerleave", onLeave);
      setHover(null);
    };
  }, [scene, camera, gl]);

  return null;
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
