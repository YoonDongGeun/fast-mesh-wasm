// The canvas: `<FastMesh>` per mesh, the section's contours, an orbit camera
// and a TransformControls gizmo on the selected mesh. Everything goes
// through the store from the nearest `<SceneProvider>`; `<FastMesh>` also
// drives its frames (`store.commitFrame()` once per rendered frame).
import { Canvas, type ThreeEvent, useFrame, useThree } from "@react-three/fiber";
import type { MeshHandle, SectionHandle } from "fast-mesh-wasm";
import { FastMesh } from "fast-mesh-wasm/r3f";
import { useHistory, useScene, useSection, useSelection } from "fast-mesh-wasm/react";
import { meshState, sectionState } from "fast-mesh-wasm/store";
import { attachTransformControls, type BoundMesh, bindSection } from "fast-mesh-wasm/three";
import { useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState } from "react";
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { TransformControls } from "three/addons/controls/TransformControls.js";
import { type Item, makeItem } from "./items";
import { fit, type MeshData, parseBinaryStl } from "./meshes";
import { Panel } from "./Panel";

/** A mesh the user clicked, with the object `<FastMesh>` draws it with: the gizmo's target. */
interface Picked {
  mesh: MeshHandle;
  object: THREE.Mesh;
}

export function App(props: { initialItems: Item[]; section: SectionHandle }) {
  const { section } = props;
  const store = useScene();
  const [items, setItems] = useState(props.initialItems);
  const [selection, select] = useSelection();
  const [picked, setPicked] = useState<Picked | null>(null);
  const [gizmoMode, setGizmoMode] = useState<"translate" | "rotate">("translate");
  const [note, setNote] = useState("");
  const { undo, redo } = useHistory();

  // ---- Keyboard: undo/redo, deselect, gizmo mode ----
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const key = e.key.toLowerCase();
      if ((e.ctrlKey || e.metaKey) && key === "z") {
        e.preventDefault();
        if (e.shiftKey) {
          redo();
        } else {
          undo();
        }
      } else if ((e.ctrlKey || e.metaKey) && key === "y") {
        e.preventDefault();
        redo();
      } else if (key === "escape") {
        select(null);
      } else if (key === "t") {
        setGizmoMode("translate");
      } else if (key === "r") {
        setGizmoMode("rotate");
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [undo, redo, select]);

  // ---- Drop a binary STL: it replaces the selected mesh (or the first) ----
  const itemsRef = useRef(items);
  itemsRef.current = items;

  const replace = useCallback(
    (old: Item, data: MeshData, name: string) => {
      const color = old.material.color.getHex();
      const added = makeItem(store, name, fit(data, old.center, old.radius), color);
      const snap = store.getSnapshot();
      // Colourings measured to the old mesh now measure to the new one.
      for (const other of itemsRef.current) {
        const coloring = meshState(snap, other.mesh).desired.coloring;
        if (other !== old && coloring?.distanceTo === old.mesh) {
          store.setDisplay(other.mesh, { coloring: { ...coloring, distanceTo: added.mesh } });
        }
      }
      // A section cutting the old mesh cuts the new one.
      const cut = sectionState(snap, section).desired.meshes;
      if (cut.includes(old.mesh)) {
        store.setSection(section, { meshes: cut.map((m) => (m === old.mesh ? added.mesh : m)) });
      }
      store.removeMesh(old.mesh);
      old.material.dispose();
      setItems(itemsRef.current.map((i) => (i === old ? added : i)));
    },
    [store, section],
  );

  useEffect(() => {
    const onDragOver = (e: DragEvent) => e.preventDefault();
    const onDrop = (e: DragEvent) => {
      e.preventDefault();
      const file = e.dataTransfer?.files[0];
      if (file === undefined) {
        return;
      }
      void file.arrayBuffer().then((buffer) => {
        const data = parseBinaryStl(buffer);
        if (data === null) {
          setNote(`${file.name} is not a binary STL`);
          return;
        }
        const current = itemsRef.current;
        const selected = store.getSnapshot().selection;
        const old = current.find((i) => i.mesh === selected) ?? current[0];
        replace(old, data, file.name);
        setNote("");
      });
    };
    window.addEventListener("dragover", onDragOver);
    window.addEventListener("drop", onDrop);
    return () => {
      window.removeEventListener("dragover", onDragOver);
      window.removeEventListener("drop", onDrop);
    };
  }, [store, replace]);

  const pick = useCallback(
    (mesh: MeshHandle, object: THREE.Mesh) => {
      setPicked((p) => (p?.mesh === mesh && p.object === object ? p : { mesh, object }));
      select(mesh);
    },
    [select],
  );

  return (
    <>
      <Canvas
        camera={{ position: [0, 3.5, 7], fov: 45 }}
        onCreated={({ gl }) => {
          // The section clips the materials it cuts (`bindSection`).
          gl.localClippingEnabled = true;
        }}
      >
        <color attach="background" args={["#1c1f24"]} />
        <ambientLight intensity={0.6} />
        <directionalLight position={[3, 6, 4]} intensity={2.2} />
        <directionalLight position={[-4, -2, -3]} intensity={0.6} />
        {items.map((item) => (
          <MeshView key={item.id} item={item} selected={item.mesh === selection} onPick={pick} />
        ))}
        <SectionView section={section} items={items} />
        <Controls picked={picked !== null && picked.mesh === selection ? picked : null} mode={gizmoMode} />
      </Canvas>
      <Panel items={items} section={section} note={note} />
    </>
  );
}

/** A mesh drawn with `<FastMesh>`: click to select it. */
function MeshView(props: { item: Item; selected: boolean; onPick: (mesh: MeshHandle, object: THREE.Mesh) => void }) {
  const { item, selected, onPick } = props;
  useLayoutEffect(() => {
    item.material.emissive.set(selected ? 0x3a2600 : 0x000000);
  }, [item.material, selected]);
  const onClick = useCallback(
    (e: ThreeEvent<MouseEvent>) => {
      e.stopPropagation();
      onPick(item.mesh, e.eventObject as THREE.Mesh);
    },
    [item.mesh, onPick],
  );
  return <FastMesh mesh={item.mesh} material={item.material} onClick={onClick} />;
}

/**
 * The section's contours, and the clipping of the meshes it cuts, with
 * `bindSection`. Rebound when the meshes it cuts (as applied) change.
 */
function SectionView(props: { section: SectionHandle; items: Item[] }) {
  const { section, items } = props;
  const store = useScene();
  const [state] = useSection(section);
  const cut = useMemo(() => items.filter((i) => state.meshes.includes(i.mesh)), [items, state.meshes]);
  const [group, setGroup] = useState<THREE.Group | null>(null);

  useLayoutEffect(() => {
    if (cut.length === 0) {
      return;
    }
    // `bindSection` reads a bound mesh's handle and its object's material
    // (to clip it). `<FastMesh>` keeps its binding to itself, so each cut
    // mesh is described by its handle and the material the app gave it.
    const bound: BoundMesh[] = cut.map((i) => ({
      object: new THREE.Mesh(undefined, i.material),
      mesh: i.mesh,
      setOpacity() {},
      dispose() {},
    }));
    const b = bindSection(store.scene, section, bound, { lineColor: 0xffd400 });
    setGroup(b.group);
    return () => {
      b.dispose();
      setGroup((g) => (g === b.group ? null : g));
    };
  }, [store, section, cut]);

  return group === null ? null : <primitive object={group} dispose={null} />;
}

/** Orbit camera, plus a TransformControls gizmo on `picked` (dragging through `attachTransformControls`). */
function Controls(props: { picked: Picked | null; mode: "translate" | "rotate" }) {
  const { picked, mode } = props;
  const store = useScene();
  const camera = useThree((s) => s.camera);
  const gl = useThree((s) => s.gl);
  const scene = useThree((s) => s.scene);
  const [orbit, setOrbit] = useState<OrbitControls | null>(null);
  const [gizmo, setGizmo] = useState<TransformControls | null>(null);

  useEffect(() => {
    const o = new OrbitControls(camera, gl.domElement);
    o.target.set(0, 0.5, 0);
    o.update();
    setOrbit(o);
    return () => {
      o.dispose();
      setOrbit((c) => (c === o ? null : c));
    };
  }, [camera, gl]);

  useFrame(() => orbit?.update());

  useEffect(() => {
    if (orbit === null || picked === null) {
      return;
    }
    const controls = new TransformControls(camera, gl.domElement);
    const helper = controls.getHelper();
    scene.add(helper);
    controls.attach(picked.object);
    // `attachTransformControls` reads a bound mesh's object and handle: the
    // object `<FastMesh>` draws (from the click) and the mesh it draws.
    const detach = attachTransformControls(controls, store.scene, {
      object: picked.object,
      mesh: picked.mesh,
      setOpacity() {},
      dispose() {},
    });
    const onDragging = (e: { value: unknown }) => {
      orbit.enabled = e.value !== true;
    };
    controls.addEventListener("dragging-changed", onDragging);
    setGizmo(controls);
    return () => {
      controls.removeEventListener("dragging-changed", onDragging);
      detach();
      controls.detach();
      scene.remove(helper);
      controls.dispose();
      orbit.enabled = true;
      setGizmo((g) => (g === controls ? null : g));
    };
  }, [store, camera, gl, scene, orbit, picked]);

  useEffect(() => {
    gizmo?.setMode(mode);
  }, [gizmo, mode]);

  return null;
}
