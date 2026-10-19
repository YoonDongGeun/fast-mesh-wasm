// `fast-mesh-wasm/r3f`: React Three Fiber bindings over a `SceneStore`.
// `<FastMesh>` draws a scene mesh (through `bindMesh` from
// `fast-mesh-wasm/three`) and follows its display; `useFrameDriver` calls
// `store.commitFrame()` once per rendered frame. `@react-three/fiber`,
// `three` and `react` are optional peers. Written without JSX, so the
// library needs no JSX transform.
import { type EventHandlers, useFrame } from "@react-three/fiber";
import {
  createElement,
  type JSX,
  type ReactElement,
  useCallback,
  useContext,
  useLayoutEffect,
  useState,
} from "react";
import * as THREE from "three";
import { FastMeshError } from "../core.js";
import { StoreContext } from "../react/context.js";
import { SceneProvider, useMeshDisplay, useScene, useStoreSelector } from "../react/index.js";
import type { MeshHandle } from "../scene.js";
import type { SceneSnapshot } from "../store/snapshot.js";
import type { SceneStore } from "../store/store.js";
import { type BindMeshOptions, type BoundMesh, bindMesh } from "../three/mesh.js";

/**
 * The frame drivers mounted per store, in mount order. Only the first one
 * commits, so a store is committed once per rendered frame however many
 * drivers (`<FastMesh>` instances) are mounted; when it unmounts, the next
 * one takes over. Election rather than a frame stamp: R3F calls every
 * `useFrame` subscriber once per frame, but the root state it passes (clock,
 * `gl.info`) is not a reliable per-frame identity (`frameloop="never"`,
 * `advance()` with a fixed timestamp, test renderers).
 */
const drivers = new WeakMap<SceneStore, object[]>();

/**
 * Calls `store.commitFrame()` once per rendered R3F frame, however many
 * components call this hook for the same store. Use it inside a `<Canvas>`;
 * `<FastMesh>` calls it, so an app that draws through `<FastMesh>` needs no
 * other driver. Do not also call `store.commitFrame()` yourself.
 *
 * @param store - The store to drive; defaults to the nearest `SceneProvider`'s.
 * @throws `INVALID_ARGUMENT` without a `store` outside a provider.
 */
export function useFrameDriver(store?: SceneStore): void {
  const fromContext = useContext(StoreContext);
  const target = store ?? fromContext;
  if (target === null) {
    throw new FastMeshError("INVALID_ARGUMENT", "useFrameDriver() needs a store or a <SceneProvider> above it");
  }
  const [token] = useState(() => ({}));
  useLayoutEffect(() => {
    let list = drivers.get(target);
    if (list === undefined) {
      list = [];
      drivers.set(target, list);
    }
    list.push(token);
    return () => {
      const at = list.indexOf(token);
      if (at !== -1) {
        list.splice(at, 1);
      }
    };
  }, [target, token]);
  useFrame(() => {
    if (drivers.get(target)?.[0] === token) {
      target.commitFrame();
    }
  });
}

/** Props of {@link FastMesh}: R3F's pointer events are passed through to the mesh. */
export interface FastMeshProps extends EventHandlers {
  /** The scene mesh to draw. */
  mesh: MeshHandle;
  /** The store; defaults to the nearest `SceneProvider`'s. */
  store?: SceneStore;
  /**
   * The material (see `BindMeshOptions.material`): give each `<FastMesh>` its
   * own, and dispose it yourself. Changing it rebinds the mesh.
   */
  material?: BindMeshOptions["material"];
  /** The colour while the mesh has no colouring (see `BindMeshOptions.color`). */
  color?: THREE.ColorRepresentation;
}

/**
 * Draws a scene mesh: binds it with `bindMesh` on mount and disposes the
 * binding on unmount (or when the mesh is removed, after which it renders
 * nothing). The mesh's applied `opacity` and `visible` (`useMeshDisplay`)
 * follow the store; colouring follows the scene. Drives the store's frames
 * with {@link useFrameDriver}. Use it inside a `<Canvas>`.
 *
 * @throws `INVALID_ARGUMENT` without a `store` outside a `SceneProvider`.
 */
export function FastMesh(props: FastMeshProps): JSX.Element {
  const body = createElement(FastMeshBody, props);
  return props.store === undefined ? body : createElement(SceneProvider, { store: props.store }, body);
}

function FastMeshBody(props: FastMeshProps): ReactElement | null {
  const { mesh, store: _store, material, color, ...events } = props;
  const store = useScene();
  useFrameDriver(store);
  const present = useStoreSelector(useCallback((s: SceneSnapshot) => s.meshes.has(mesh), [mesh]));
  const [display] = useMeshDisplay(mesh);
  // A colour as a value, so an inline `color={[r, g, b]}`-style prop does not rebind every render.
  const colorKey = color === undefined ? undefined : new THREE.Color(color).getHex();
  const [bound, setBound] = useState<BoundMesh | null>(null);

  useLayoutEffect(() => {
    if (!present) {
      return;
    }
    const b = bindMesh(store.scene, mesh, colorKey === undefined ? { material } : { material, color: colorKey });
    setBound(b);
    return () => {
      b.dispose();
      setBound((current) => (current === b ? null : current));
    };
  }, [store, mesh, present, material, colorKey]);

  useLayoutEffect(() => {
    if (bound !== null) {
      bound.setOpacity(display.opacity);
      bound.object.visible = display.visible;
    }
  }, [bound, display.opacity, display.visible]);

  if (bound === null) {
    return null;
  }
  // `dispose: null`: the binding, not R3F, owns the geometry and material.
  return createElement("primitive", { object: bound.object, dispose: null, ...events });
}
