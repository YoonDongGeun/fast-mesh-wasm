// `fast-mesh-wasm/r3f`: React Three Fiber bindings over a `SceneStore`.
// `<FastMesh>` draws a scene mesh (through `bindMesh` from
// `fast-mesh-wasm/three`) and follows its display; `useFrameDriver` calls
// `store.commitFrame()` once per rendered frame and invalidates the canvas
// on change. `@react-three/fiber`, `three` and `react` are optional peers.
// Written without JSX, so the library needs no JSX transform.
import { type EventHandlers, useFrame, useThree } from "@react-three/fiber";
import {
  createElement,
  type JSX,
  type ReactElement,
  useCallback,
  useContext,
  useLayoutEffect,
  useRef,
  useState,
} from "react";
import * as THREE from "three";
import { FastMeshError } from "../core.js";
import { StoreContext } from "../react/context.js";
import { SceneProvider, useScene, useStoreSelector } from "../react/index.js";
import type { MeshHandle } from "../scene.js";
import { type DisplayState, meshState, type SceneSnapshot } from "../store/snapshot.js";
import { type SceneStore, setFrameLoop } from "../store/store.js";
import { type BindMeshOptions, type BoundMesh, bindMesh } from "../three/mesh.js";

/** The frame drivers of one store. */
interface Drivers {
  /**
   * Driver tokens in mount order. Only the first one commits, so a store is
   * committed once per rendered frame however many drivers (`<FastMesh>`
   * instances) are mounted; when it unmounts, the next one takes over.
   * Election rather than a frame stamp: R3F calls every `useFrame`
   * subscriber once per frame, but the root state it passes (clock,
   * `gl.info`) is not a reliable per-frame identity (`frameloop="never"`,
   * `advance()` with a fixed timestamp, test renderers).
   */
  tokens: object[];
  /**
   * One change subscription per canvas (keyed by its `invalidate`), shared by
   * that canvas's drivers: however many meshes it draws, a change
   * invalidates it once.
   */
  canvases: Map<() => void, { count: number; off: () => void }>;
}

const drivers = new WeakMap<SceneStore, Drivers>();

/**
 * Calls `store.commitFrame()` once per rendered R3F frame, however many
 * components call this hook for the same store, and invalidates the canvas
 * when the store or the scene changes, so `frameloop="demand"` renders the
 * change. Use it inside a `<Canvas>`; `<FastMesh>` calls it, so an app that
 * draws through `<FastMesh>` needs no other driver. Do not also call
 * `store.commitFrame()` yourself.
 *
 * The first frame switches the store to frame flushing (declarative changes
 * wait for the next frame); when the store's last driver unmounts, the store
 * goes back to flushing them on a microtask.
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
  const invalidate = useThree((s) => s.invalidate);
  const [token] = useState(() => ({}));
  useLayoutEffect(() => {
    let found = drivers.get(target);
    if (found === undefined) {
      found = { tokens: [], canvases: new Map() };
      drivers.set(target, found);
    }
    const d = found;
    d.tokens.push(token);
    let subscribed = d.canvases.get(invalidate);
    if (subscribed === undefined) {
      // `invalidate` takes an optional frame count: always called bare.
      const offStore = target.subscribe(() => invalidate());
      const offScene = target.scene.onChanges((changes) => {
        // The scene's listeners run on every `commitFrame()`; only a change needs a frame.
        if (changes.length > 0) {
          invalidate();
        }
      });
      subscribed = {
        count: 0,
        off: () => {
          offStore();
          offScene();
        },
      };
      d.canvases.set(invalidate, subscribed);
    }
    const canvas = subscribed;
    canvas.count++;
    return () => {
      const at = d.tokens.indexOf(token);
      if (at !== -1) {
        d.tokens.splice(at, 1);
      }
      if (--canvas.count === 0) {
        d.canvases.delete(invalidate);
        canvas.off();
      }
      if (d.tokens.length === 0) {
        // No frame loop left: declarative changes flush on a microtask again.
        setFrameLoop(target, false);
      }
    };
  }, [target, token, invalidate]);
  useFrame(() => {
    if (drivers.get(target)?.tokens[0] === token) {
      target.commitFrame();
    }
  });
}

/** Props of {@link FastMesh}: R3F's pointer events are passed through to the mesh. */
export interface FastMeshProps extends EventHandlers {
  /** The scene mesh to draw; added through the store (`store.addMesh`). */
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
  /**
   * How a colouring is drawn (see `BindMeshOptions.shading`): `"vertex"` (the
   * default) or `"fragment"`. Changing it rebinds the mesh.
   */
  shading?: BindMeshOptions["shading"];
  /**
   * Fragment shading's edge softness in distance units (see
   * `BindMeshOptions.edgeSoftness`); 0 (the default) is a hard edge. Changing
   * it never rebinds. NaN, an infinite or a negative width draws as 0, with a
   * console warning (`bindMesh` would throw).
   */
  edgeSoftness?: number;
  /**
   * Called with the binding once the mesh is bound, and with `null` when that
   * binding is disposed (unmount, removal, or a rebind, which then delivers
   * the new one). Pass it to `bindSection` or `attachTransformControls`.
   * `bindSection` clips the materials it sees when it is called, so bind the
   * section again when a new binding arrives. A new callback never rebinds.
   */
  onBind?: (bound: BoundMesh | null) => void;
}

/**
 * Draws a scene mesh: binds it with `bindMesh` on mount and disposes the
 * binding on unmount (or when the mesh is removed, after which it renders
 * nothing). The mesh's applied `opacity` and `visible` follow the store in
 * the frame that applies them; colouring follows the scene. Drives the
 * store's frames with {@link useFrameDriver}. Use it inside a `<Canvas>`.
 *
 * The mesh must be added through the store (`store.addMesh`): a mesh added
 * straight to `store.scene` is not in the store, and renders nothing.
 *
 * @throws `INVALID_ARGUMENT` without a `store` outside a `SceneProvider`.
 */
/**
 * `width` when it is a finite number `>= 0` (what `setEdgeSoftness` takes),
 * else 0 with a warning: a bad prop (a slider's NaN, say) draws hard edges
 * rather than throwing inside the canvas.
 */
function usableSoftness(width: number): number {
  if (typeof width === "number" && Number.isFinite(width) && width >= 0) {
    return width;
  }
  console.warn(`fast-mesh-wasm: <FastMesh edgeSoftness={${String(width)}}> is not a finite number >= 0; using 0.`);
  return 0;
}

export function FastMesh(props: FastMeshProps): JSX.Element {
  const body = createElement(FastMeshBody, props);
  return props.store === undefined ? body : createElement(SceneProvider, { store: props.store }, body);
}

function FastMeshBody(props: FastMeshProps): ReactElement | null {
  const { mesh, store: _store, material, color, shading, edgeSoftness = 0, onBind, ...events } = props;
  const store = useScene();
  useFrameDriver(store);
  const present = useStoreSelector(useCallback((s: SceneSnapshot) => s.meshes.has(mesh), [mesh]));
  // A colour as a value, so an inline `color={[r, g, b]}`-style prop does not rebind every render.
  const colorKey = color === undefined ? undefined : new THREE.Color(color).getHex();
  const [bound, setBound] = useState<BoundMesh | null>(null);
  // The latest callback, read when binding: a new one never rebinds.
  const onBindRef = useRef(onBind);
  onBindRef.current = onBind;
  // The latest softness, read when binding: a new one is set, never rebinds.
  const softness = usableSoftness(edgeSoftness);
  const softnessRef = useRef(softness);
  softnessRef.current = softness;

  useLayoutEffect(() => {
    if (!present) {
      return;
    }
    const options = { material, shading, edgeSoftness: softnessRef.current };
    const b = bindMesh(store.scene, mesh, colorKey === undefined ? options : { ...options, color: colorKey });
    // Opacity and visibility from a store listener, which runs inside
    // `commitFrame()` (before the render), so they show in the same frame as
    // a colouring flushed with them; and once now.
    let last: DisplayState | undefined;
    const apply = () => {
      const d = meshState(store.getSnapshot(), mesh).display;
      if (d !== last) {
        last = d;
        b.setOpacity(d.opacity);
        b.object.visible = d.visible;
      }
    };
    apply();
    const off = store.subscribe(apply);
    setBound(b);
    onBindRef.current?.(b);
    return () => {
      off();
      b.dispose();
      setBound((current) => (current === b ? null : current));
      onBindRef.current?.(null);
    };
  }, [store, mesh, present, material, colorKey, shading]);

  useLayoutEffect(() => {
    bound?.setEdgeSoftness(softness);
  }, [bound, softness]);

  if (bound === null) {
    return null;
  }
  // `dispose: null`: the binding, not R3F, owns the geometry and material.
  return createElement("primitive", { object: bound.object, dispose: null, ...events });
}
