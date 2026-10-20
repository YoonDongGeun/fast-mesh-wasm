// The canvas: `<FastMesh>` per mesh, the section's contours, an orbit camera
// and a TransformControls gizmo on the selected mesh. Everything goes
// through the store from the nearest `<SceneProvider>`; `<FastMesh>` also
// drives its frames (`store.commitFrame()` once per rendered frame).
import { Canvas, type ThreeEvent, useFrame, useThree } from "@react-three/fiber";
import type { MeshHandle, SectionHandle } from "fast-mesh-wasm";
import { FastMesh } from "fast-mesh-wasm/r3f";
import { useHistory, useScene, useSection, useSelection } from "fast-mesh-wasm/react";
import { meshState, sectionState } from "fast-mesh-wasm/store";
import { attachTransformControls, bindSection, type MeshBinding } from "fast-mesh-wasm/three";
import { useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState } from "react";
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { TransformControls } from "three/addons/controls/TransformControls.js";
import { type Item, makeItem } from "./items";
import { fit, type MeshData, parseBinaryStl } from "./meshes";
import { Panel } from "./Panel";

export function App(props: { initialItems: Item[]; section: SectionHandle }) {
  const { section } = props;
  const store = useScene();
  const [items, setItems] = useState(props.initialItems);
  const [selection, select] = useSelection();
  // Each mesh's binding, as `<FastMesh onBind>` delivers it: for the section and the gizmo.
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
  const [gizmoMode, setGizmoMode] = useState<"translate" | "rotate">("translate");
  const [note, setNote] = useState("");
  const { undo, redo } = useHistory();

  // ---- Keyboard: undo/redo, deselect, gizmo mode ----
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      // Typing in a field (the opacity slider, say) is not a shortcut.
      const target = e.target as HTMLElement | null;
      if (target !== null && (target.isContentEditable || /^(INPUT|SELECT|TEXTAREA)$/.test(target.tagName))) {
        return;
      }
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
      } else if (e.ctrlKey || e.metaKey || e.altKey) {
        // Other modified keys (Ctrl+R, say) are the browser's.
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

  const selected = selection === null ? null : (bindings.get(selection) ?? null);

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
          <MeshView key={item.id} item={item} selected={item.mesh === selection} onPick={select} onBind={onBind} />
        ))}
        <SectionView section={section} bindings={bindings} />
        <Controls bound={selected} mode={gizmoMode} />
      </Canvas>
      <Panel items={items} section={section} note={note} />
    </>
  );
}

/** A mesh drawn with `<FastMesh>`: click to select it. */
function MeshView(props: {
  item: Item;
  selected: boolean;
  onPick: (mesh: MeshHandle) => void;
  onBind: (mesh: MeshHandle, bound: MeshBinding | null) => void;
}) {
  const { item, selected, onPick, onBind } = props;
  useLayoutEffect(() => {
    item.material.emissive.set(selected ? 0x3a2600 : 0x000000);
  }, [item.material, selected]);
  const onClick = useCallback(
    (e: ThreeEvent<MouseEvent>) => {
      e.stopPropagation();
      onPick(item.mesh);
    },
    [item.mesh, onPick],
  );
  return (
    <FastMesh mesh={item.mesh} material={item.material} onClick={onClick} onBind={(b) => onBind(item.mesh, b)} />
  );
}

/**
 * The section's contours, and the clipping of the meshes it cuts, with
 * `bindSection` over the `<FastMesh>` bindings. Rebound when the meshes it
 * cuts (as applied) or their bindings change: `bindSection` clips the
 * materials it sees when it is called.
 */
function SectionView(props: { section: SectionHandle; bindings: ReadonlyMap<MeshHandle, MeshBinding> }) {
  const { section, bindings } = props;
  const store = useScene();
  const [state] = useSection(section);
  const cut = useMemo(() => state.meshes.flatMap((m) => bindings.get(m) ?? []), [bindings, state.meshes]);
  const [group, setGroup] = useState<THREE.Group | null>(null);

  useLayoutEffect(() => {
    if (cut.length === 0) {
      return;
    }
    const b = bindSection(store.scene, section, cut, { lineColor: 0xffd400 });
    setGroup(b.group);
    return () => {
      b.dispose();
      setGroup((g) => (g === b.group ? null : g));
    };
  }, [store, section, cut]);

  return group === null ? null : <primitive object={group} dispose={null} />;
}

/** Orbit camera, plus a TransformControls gizmo on `bound` (dragging through `attachTransformControls`). */
function Controls(props: { bound: MeshBinding | null; mode: "translate" | "rotate" }) {
  const { bound, mode } = props;
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
  }, [store, camera, gl, scene, orbit, bound]);

  useEffect(() => {
    gizmo?.setMode(mode);
  }, [gizmo, mode]);

  return null;
}
