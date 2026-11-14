// The canvas: `<FastMesh>` per mesh (click to select), a trackball camera and a
// TransformControls gizmo on the selected mesh (moves through
// `attachTransformControls`, one undo step per drag). `<FastMesh>` drives
// the store's frames. Colourings are drawn with fragment shading, so the
// heatmap's boundaries are crisp, softened by the collision panel's slider.
// With `hoverable` (Group mode) the vertex under the pointer is tracked in
// `view.hover`, for the hover tip: through the engine's BVH raycast
// (`scene.raycast`, async on the Worker), one at a time and at most every
// HOVER_PERIOD_MS, never three's per-triangle `Mesh.raycast`.
import { Canvas, type ThreeEvent, useFrame, useThree } from "@react-three/fiber";
import type { MeshHandle, RepairOp, Scene, StretchReport } from "fast-mesh-wasm";
import { FastMesh } from "fast-mesh-wasm/r3f";
import { useScene, useSelection } from "fast-mesh-wasm/react";
import { attachTransformControls, type MeshBinding } from "fast-mesh-wasm/three";
import { useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState } from "react";
import * as THREE from "three";
import { TrackballControls } from "three/addons/controls/TrackballControls.js";
import { TransformControls } from "three/addons/controls/TransformControls.js";
import type { WorkerScene } from "fast-mesh-wasm/worker";
import type { FocusRequest } from "../App";
import type { Loader } from "../load/loader";
import { inspect } from "../panels/inspect";
import { type AppState, type Item, useApp } from "../scene/app-state";
import { selectMesh } from "../scene/selection";
import { InspectLayers, pickVertex } from "./InspectLayers";
import { setSliceClip } from "./clip-planes";
import { SceneSlices } from "./SceneSlices";
import { SceneVolume } from "./SceneVolume";
import { matchVolume } from "./volume-box";
import {
  dabOperation,
  faceNormal,
  grabLocalDelta,
  interpolateHit,
  liveMaxSplits,
  liveRemeshDue,
  meanEdgeLength,
  reflectCenters,
  remeshTooBig,
  type SculptBrush,
  type SculptMask,
  SculptStroke,
  type StrokeSteps,
  toSymmetry,
  unionMaskIds,
  widestGrabReach,
  worldPerPixel,
} from "./sculpt";
import { easeNormal, makeBrushCursor, NORMAL_MARKER, placeBrushCursor, setBrushDirection } from "./brush-cursor";
import { isExtraFinger, tapSlop, watchFingers } from "./touch";
import { setDragging, setHover, view } from "./view";
import { MaskOverlay } from "./mask-overlay";
import {
  inheritedSplits,
  MaskLiveTracker,
  maskMeetsRemesh,
  protectMaskedDab,
} from "./mask-protect";
import { setWireOverlay } from "./wire";
import {
  buildRepairOverlayGroup,
  countRepairOverlays,
  disposeRepairOverlayGroup,
} from "./repair-overlays";

export type GizmoMode = "translate" | "rotate";

/** The hover raycast runs at most this often while the pointer moves. */
const HOVER_PERIOD_MS = 50;
/** The sculpt raycast runs at most this often while a stroke paints. */
const STROKE_PERIOD_MS = 16;
/** Hold-still buildup: a forced dab this often while the button stays down without moving. */
const BUILDUP_MS = 16;
/** Holding this long builds up one full dab's strength. */
const BUILDUP_FULL_MS = 90;

export function Viewport(props: {
  app: AppState;
  items: readonly Item[];
  mode: GizmoMode;
  focus: FocusRequest | null;
  /** Bumped to restore the default camera pose. */
  resetView: number;
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
  /** The sculpt brush armed on a mesh, if any. */
  sculpt: SculptBrush | null;
  /** A stroke's end remeshes its region. */
  sculptRemesh: boolean;
  /** A stroke is remeshed live while it paints. */
  sculptLiveRemesh: boolean;
  /** The mask paint is armed: press-and-hold paints the protected region. */
  maskPaint: boolean;
  /** Lay the wireframe overlay over every shaded mesh. */
  wireframe: boolean;
  /** The loader, for the scene slice overlay's volume fetch. */
  loader: Loader;
}) {
  const {
    app,
    items,
    mode,
    focus,
    resetView,
    edgeSoftness,
    hoverable,
    scene,
    picking,
    onPickPoint,
    sculpt,
    sculptRemesh,
    sculptLiveRemesh,
    maskPaint,
    wireframe,
    loader,
  } = props;
  const [selection, select] = useSelection();
  const multi = useApp(app, (s) => s.multi);
  const downAt = useRef<{ x: number; y: number } | null>(null);
  const lastDragEnd = useRef(0);
  useEffect(() => {
    const onDrag = (dragging: boolean) => {
      if (!dragging) {
        lastDragEnd.current = Date.now();
      }
    };
    view.dragListeners.add(onDrag);
    return () => {
      view.dragListeners.delete(onDrag);
    };
  }, []);
  // The paint works on the selection: with none left (the mesh was removed)
  // it has nothing to work on, and would keep the next mesh's gizmo down.
  useEffect(() => {
    if (maskPaint && selection === null) {
      app.setMaskPaint(false);
    }
  }, [app, maskPaint, selection]);

  // A tap on empty space clears the selection; sculpt and mask paint own the
  // pointer, and gizmo drags and orbits are not taps. While an inspect
  // tool is armed the selection stays put (the ruler and query click
  // meshes without selecting them).
  const inspectTool = useApp(app, (s) => s.inspectTool);
  const onMissed = useCallback(
    (e: MouseEvent) => {
      if (sculpt !== null || maskPaint || picking !== null || view.dragging || view.multiTouch || view.pressTaken || selection === null) {
        return;
      }
      if (inspectTool !== null) {
        return;
      }
      if (view.gizmo !== null && view.gizmo.axis !== null) {
        return;
      }
      if (Date.now() - lastDragEnd.current < 350) {
        return;
      }
      const down = downAt.current;
      if (down === null || Math.hypot(e.clientX - down.x, e.clientY - down.y) > tapSlop(view.pointerType, 6)) {
        return;
      }
      // Through clickMulti: a shift-click on nothing is ignored, like on a mesh.
      selectMesh(app, select, selection, null, e.shiftKey);
    },
    [app, select, selection, sculpt, maskPaint, picking, inspectTool],
  );
  const mpr = useApp(app, (s) => s.volumeMpr);
  // The mesh shown as the scene volume instead of its surface, if any.
  // Sculpting shows the surface: the armed mesh drops out of volume mode.
  const volumeFound = matchVolume(items, selection, mpr);
  const sculptMesh = sculpt?.mesh ?? null;
  const volumeMesh =
    volumeFound !== null && volumeFound.mpr.inScene && volumeFound.item.mesh !== sculptMesh
      ? volumeFound.item.mesh
      : null;
  const onPick = useCallback(
    (mesh: MeshHandle, additive: boolean) => selectMesh(app, select, selection, mesh, additive),
    [app, select, selection],
  );
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
  // Shared live-mask cache between MaskLayer and SculptLayer so switching from
  // mask paint into sculpting preserves O(1) verticesInSphere trust on dab 1.
  const maskTracker = useMemo(() => new MaskLiveTracker(scene), [scene]);
  useEffect(() => () => maskTracker.dispose(), [maskTracker]);

  return (
    <Canvas camera={{ position: [-0.6, 3.4, 8.2], fov: 45, near: 0.01, far: 5000 }} onPointerMissed={onMissed}>
      <color attach="background" args={["#1b1e23"]} />
      <hemisphereLight args={["#e8eef8", "#30343c", 1.1]} />
      <directionalLight position={[3, 6, 4]} intensity={2} />
      <directionalLight position={[-4, -2, -3]} intensity={0.5} />
      <gridHelper args={[20, 20, "#3a404a", "#2a2f36"]} position={[0, -0.4, 0]} />
      {items.map((item) => (
        <MeshView
          key={item.id}
          app={app}
          item={item}
          selected={item.mesh === selection}
          volumeShown={item.mesh === volumeMesh}
          secondary={item.mesh !== selection && multi.includes(item.mesh)}
          edgeSoftness={edgeSoftness}
          onPick={onPick}
          onBind={onBind}
          picking={picking === item.mesh}
          onPickPoint={onPickPoint}
          scene={scene}
          sculpted={sculpt?.mesh === item.mesh}
          suppressClick={sculpt?.mesh === item.mesh || (maskPaint && sculpt === null)}
          wireframe={wireframe}
          bound={bindings.get(item.mesh) ?? null}
        />
      ))}
      {/* Sculpt and mask paint own the pointer while armed: the gizmo stands down, so a press on its handles still paints. */}
      <Controls bound={sculpt !== null || maskPaint ? null : selected} mode={mode} focus={focus} resetView={resetView} radiusOf={radiusOf} />
      {hoverable ? <HoverRay scene={scene} /> : null}
      {sculpt !== null ? (
        <SculptLayer
          app={app}
          scene={scene}
          sculpt={sculpt}
          remesh={sculptRemesh}
          liveRemesh={sculptLiveRemesh}
          maskTracker={maskTracker}
        />
      ) : null}
      {/* Mask paint owns the pointer while armed (like sculpt, it disarms the gizmo's claim); sculpt wins when both arm. */}
      {maskPaint && sculpt === null ? <MaskLayer app={app} scene={scene} maskTracker={maskTracker} /> : null}
      <PickCursor picking={picking !== null || sculpt !== null} />
      <MissTracker track={downAt} />
      <SceneSlices app={app} loader={loader} scene={scene} />
      <SceneVolume app={app} loader={loader} scene={scene} />
      <InspectLayers app={app} scene={scene} items={items} />
    </Canvas>
  );
}

/** Records canvas pointerdowns, so a missed click can tell a drag from a tap and a finger from a pinch. */
function MissTracker(props: { track: { current: { x: number; y: number } | null } }) {
  const gl = useThree((s) => s.gl);
  useEffect(() => {
    const el = gl.domElement;
    const onDown = (e: PointerEvent) => {
      view.pointerType = e.pointerType;
      view.multiTouch = isExtraFinger(e);
      if (!view.multiTouch) {
        view.pressTaken = false;
        props.track.current = e.button === 0 ? { x: e.clientX, y: e.clientY } : null;
      }
    };
    // Capture: before the tools' listeners, which may take this press.
    el.addEventListener("pointerdown", onDown, true);
    return () => el.removeEventListener("pointerdown", onDown, true);
  }, [gl, props.track]);
  return null;
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
  app: AppState;
  item: Item;
  selected: boolean;
  /** Shown as the scene volume instead of its surface (a volume mesh with the overlay on). */
  volumeShown: boolean;
  /** Shift-clicked after the primary: dimmer, and a Join source. */
  secondary: boolean;
  edgeSoftness: number;
  onPick: (mesh: MeshHandle, additive: boolean) => void;
  onBind: (mesh: MeshHandle, bound: MeshBinding | null) => void;
  picking: boolean;
  onPickPoint: (mesh: MeshHandle, point: readonly [number, number, number]) => void;
  scene: Scene | WorkerScene;
  /** The sculpt brush is armed on this mesh: clicks paint, they never select or pick. */
  sculpted: boolean;
  /** True while sculpting or mask painting owns pointer clicks on this mesh. */
  suppressClick: boolean;
  /** Lay the wireframe overlay over the shaded solid. */
  wireframe: boolean;
  /** The mesh's binding, once `<FastMesh onBind>` delivers it. */
  bound: MeshBinding | null;
}) {
  const {
    app,
    item,
    selected,
    volumeShown,
    secondary,
    edgeSoftness,
    onPick,
    onBind,
    picking,
    onPickPoint,
    scene,
    sculpted,
    suppressClick,
    wireframe,
    bound,
  } = props;
  // The armed inspect tool, if any: ruler and query clicks land here instead of selecting.
  const tool = useApp(app, (s) => s.inspectTool);
  useLayoutEffect(() => {
    item.material.emissive.set(selected ? 0x3a2a08 : secondary ? 0x1c1a10 : 0x000000);
  }, [item.material, selected, secondary]);
  useLayoutEffect(() => {
    if (bound !== null) {
      setWireOverlay(bound.object, wireframe);
    }
    return () => {
      if (bound !== null) {
        setWireOverlay(bound.object, false);
      }
    };
  }, [bound, wireframe]);
  const onClick = useCallback(
    (e: ThreeEvent<MouseEvent>) => {
      e.stopPropagation();
      // The end of an orbit drag that started on the mesh is not a click,
      // nor is a pinch's lift.
      if (e.delta > tapSlop(view.pointerType, 2) || view.multiTouch) {
        return;
      }
      // A stroke's release on the mesh is not a click either, nor on a mesh
      // behind it, which the sculpted mesh's missing handler lets the click reach.
      if (sculpted || suppressClick || view.pressTaken) {
        return;
      }
      // The inspect tools own the click while armed, and never select: the
      // ruler appends a world point (a third click restarts it), the query
      // records the snapped vertex. A hit on another mesh still lands.
      if (tool === "ruler" || tool === "query") {
        const { origin, direction } = e.ray;
        const at = app.getSnapshot();
        void Promise.resolve()
          .then(() =>
            pickVertex(scene, [origin.x, origin.y, origin.z], [direction.x, direction.y, direction.z]),
          )
          .then((picked) => {
            if (picked === null) {
              return;
            }
            if (tool === "ruler") {
              const ruler = at.ruler;
              if (ruler === null || ruler.b !== null) {
                app.setRuler({ a: picked.point, b: null });
              } else {
                app.setRuler({ a: ruler.a, b: picked.point });
              }
            } else {
              app.setInspectQuery({ mesh: picked.mesh, slot: picked.slot, point: picked.point });
            }
          })
          .catch(() => {
            // Removed meanwhile: the click is dropped.
          });
        return;
      }
      if (!picking) {
        onPick(item.mesh, e.nativeEvent.shiftKey);
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
    [item.mesh, onPick, picking, onPickPoint, scene, sculpted, suppressClick, tool, app],
  );
  return (
    <>
      <FastMesh
        mesh={item.mesh}
        material={item.material}
        shading="fragment"
        edgeSoftness={edgeSoftness}
        visible={!volumeShown}
        // While sculpting or mask painting a click means nothing here, and listening
        // for one costs a per-triangle three raycast on every press and release.
        onClick={sculpted || suppressClick ? undefined : onClick}
        onBind={(b) => onBind(item.mesh, b)}
      />
      <RepairOverlays app={app} scene={scene} item={item} bound={bound} />
      <SculptMaskOverlay app={app} item={item} bound={bound} />
    </>
  );
}

/**
 * The sculpt mask's tint on its mesh (see `viewport/mask-overlay.ts`), while
 * a sculpt tool or the mask paint is armed and it is not hidden. A child of
 * the bound mesh, so it hides with it; the frame loop follows the mask and
 * the mesh's triangles (a remesh, an undo), at most one build per frame.
 */
function SculptMaskOverlay(props: { app: AppState; item: Item; bound: MeshBinding | null }) {
  const { app, item, bound } = props;
  const masked = useApp(app, (s) => (s.sculptMask?.mesh === item.mesh ? s.sculptMask.vertices : null));
  const armed = useApp(app, (s) => s.sculpt !== null || s.maskPaint);
  const shown = useApp(app, (s) => s.sculptMaskShown);
  const on = armed && shown && masked !== null;
  const object = bound?.object ?? null;
  const overlayRef = useRef<MaskOverlay | null>(null);
  useLayoutEffect(() => {
    if (!on || object === null) {
      return;
    }
    const created = new MaskOverlay(object);
    overlayRef.current = created;
    return () => {
      overlayRef.current = null;
      created.dispose();
      inspect.sculpt.maskOverlay.triangles = 0;
    };
  }, [on, object]);
  useFrame(() => {
    const overlay = overlayRef.current;
    if (overlay === null || masked === null) {
      return;
    }
    const start = performance.now();
    if (overlay.sync(masked)) {
      const stats = inspect.sculpt.maskOverlay;
      stats.lastMs = performance.now() - start;
      stats.maxMs = Math.max(stats.maxMs, stats.lastMs);
      stats.builds++;
      stats.triangles = overlay.triangles;
    }
  });
  return null;
}

/**
 * The repair-issue overlays on their analysed mesh: hole boundaries,
 * non-manifold edges, degenerate triangles and small components from the
 * repair panel's latest analysis, as children of the bound mesh's object
 * (the wire overlay's path). A new analysis object (selection, repair, undo,
 * redo) rebuilds them, like the panel's numbers; toggles come from the
 * panel. Reads the local positions/indices once per build: later scene
 * calls may move the buffers.
 */
function RepairOverlays(props: {
  app: AppState;
  scene: Scene | WorkerScene;
  item: Item;
  /** The mesh's binding, once `<FastMesh onBind>` delivers it. */
  bound: MeshBinding | null;
}) {
  const { app, scene, item, bound } = props;
  const analysis = useApp(app, (s) => s.repairAnalysis);
  const kinds = useApp(app, (s) => s.repairOverlays);
  useLayoutEffect(() => {
    const object = bound?.object ?? null;
    if (object === null) {
      return;
    }
    inspect.repair.overlays.kinds = { ...kinds };
    const none = { holes: 0, nonManifold: 0, degenerate: 0, components: 0 };
    if (analysis === null || analysis.mesh !== item.mesh) {
      inspect.repair.overlays.counts = none;
      return;
    }
    let positions: Float32Array;
    let indices: Uint32Array;
    try {
      const views = scene.views(analysis.mesh);
      positions = views.positions().slice();
      indices = views.indices().slice();
    } catch {
      // Removed meanwhile: nothing to draw.
      inspect.repair.overlays.counts = none;
      return;
    }
    const group = buildRepairOverlayGroup(analysis.analysis, positions, indices, kinds);
    const all = countRepairOverlays(analysis.analysis);
    inspect.repair.overlays.counts = {
      holes: kinds.holes ? all.holes : 0,
      nonManifold: kinds.nonManifold ? all.nonManifold : 0,
      degenerate: kinds.degenerate ? all.degenerate : 0,
      components: kinds.components ? all.components : 0,
    };
    if (group !== null) {
      object.add(group);
    }
    return () => {
      if (group !== null) {
        object.remove(group);
        disposeRepairOverlayGroup(group);
      }
    };
  }, [scene, item.mesh, bound, analysis, kinds]);
  return null;
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
    // Touch has no hover: a tap asks for the point under it instead, and the
    // tip stays until the next touch (which clears it).
    let fingerAt: { x: number; y: number } | null = null;
    let fingerDown = false;
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
            if (!alive || fingerDown) {
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
    const onLeave = (e: PointerEvent) => {
      // A lifted finger leaves the canvas: that is when a tap's tip appears.
      if (e.pointerType === "touch") {
        return;
      }
      pending = null;
      setHover(null);
    };
    const onDown = (e: PointerEvent) => {
      if (e.pointerType !== "touch") {
        return;
      }
      fingerDown = true;
      fingerAt = isExtraFinger(e) ? null : { x: e.clientX, y: e.clientY };
      pending = null;
      setHover(null);
    };
    const onUp = (e: PointerEvent) => {
      if (e.pointerType !== "touch") {
        return;
      }
      fingerDown = false;
      const at = fingerAt;
      fingerAt = null;
      if (e.type === "pointerup" && at !== null && Math.hypot(e.clientX - at.x, e.clientY - at.y) <= tapSlop("touch", 0)) {
        pending = at;
        run();
      }
    };
    canvas.addEventListener("pointermove", onMove);
    canvas.addEventListener("pointerleave", onLeave);
    canvas.addEventListener("pointerdown", onDown);
    canvas.addEventListener("pointerup", onUp);
    canvas.addEventListener("pointercancel", onUp);
    return () => {
      alive = false;
      clearTimeout(timer);
      canvas.removeEventListener("pointermove", onMove);
      canvas.removeEventListener("pointerleave", onLeave);
      canvas.removeEventListener("pointerdown", onDown);
      canvas.removeEventListener("pointerup", onUp);
      canvas.removeEventListener("pointercancel", onUp);
      setHover(null);
    };
  }, [scene, camera, gl]);

  return null;
}

/**
 * The sculpt brush on its mesh: hover shows the brush cursor (a translucent
 * sphere of the brush radius around the surface point), press-and-hold
 * paints dabs along the pointer (and builds up where it holds still;
 * grab drags the surface along instead). A press on empty space is the
 * orbit's, as usual; a stroke holds a single undo step on the sync scene
 * (the Worker has no step nesting: one step per batch of dabs there), and its end
 * remeshes the stroked region when asked. Dabs hold the painted mask's
 * vertices in place and move the rest (`protectedDab`). One raycast at a
 * time, through the engine's BVH, never three's per-triangle `Mesh.raycast`.
 */
function SculptLayer(props: {
  app: AppState;
  scene: Scene | WorkerScene;
  sculpt: SculptBrush;
  remesh: boolean;
  liveRemesh: boolean;
  maskTracker: MaskLiveTracker;
}) {
  const { app, scene, sculpt, remesh, liveRemesh, maskTracker } = props;
  const store = useScene();
  const camera = useThree((s) => s.camera);
  const gl = useThree((s) => s.gl);
  // The brush inputs drive the armed brush live; the stroke survives them.
  const brushRef = useRef(sculpt);
  brushRef.current = sculpt;
  const remeshRef = useRef(remesh);
  remeshRef.current = remesh;
  const liveRef = useRef(liveRemesh);
  liveRef.current = liveRemesh;
  // The mask reads live (paint can accumulate between strokes); the layer
  // never re-renders for it.
  const mask = useApp(app, (s) => s.sculptMask);
  const maskRef = useRef<SculptMask | null>(mask);
  maskRef.current = mask;
  // The grab drag vector (local frame), refreshed per raycast while a grab
  // stroke paints; outside React, so moves never re-render.
  const dirRef = useRef<[number, number, number]>([0, 0, 0]);

  const ball = useMemo(() => makeBrushCursor(0xffd166), []);
  useEffect(() => {
    return () => {
      ball.removeFromParent();
      for (const child of ball.children) {
        const part = child as THREE.Mesh | THREE.Line;
        part.geometry.dispose();
        (part.material as THREE.Material).dispose();
      }
    };
  }, [ball]);

  useEffect(() => {
    const canvas = gl.domElement;
    const caster = new THREE.Raycaster();
    let alive = true;
    let running = false;
    let last = -Infinity;
    let timer: ReturnType<typeof setTimeout> | undefined;
    let pending: { x: number; y: number; period: number } | null = null;
    let hoverLocal: [number, number, number] | null = null;
    let stroke: SculptStroke | null = null;
    /** Dabs applied so far (a stroke-end remesh needs at least one). */
    let landed = 0;
    /** Dabs already counted live (the rest settle at the stroke's end). */
    let counted = 0;
    const syncDabs = () => {
      if (stroke !== null) {
        inspect.sculpt.dabs += stroke.dabs - counted;
        counted = stroke.dabs;
      }
    };
    let buildup: ReturnType<typeof setInterval> | undefined;
    /** The stroking pointer, or -1 when no stroke paints. */
    let active = -1;
    /** The pointer whose press is waiting on its raycast: a release meanwhile calls the stroke off. */
    let pressing = -1;
    /** The screen point the grab drag is measured from (the press). */
    let grabFrom: { x: number; y: number } | null = null;
    /** The press point in local coordinates, or null outside a grab stroke. */
    let grabPress: [number, number, number] | null = null;
    /** Peak O(1) grab influence reach seen during the stroke, or 0 outside a grab stroke. */
    let grabPeakReach = 0;
    /** The surface's way at the cursor, eased over the triangles it crosses. */
    let cursorNormal: [number, number, number] | null = null;
    /**
     * The armed mesh's mean edge length, taken once per armed session: the
     * remesh target every stroke converges back to (the engine's live mean
     * would ratchet ever finer as splits accumulate).
     */
    let densityAnchor: { mesh: MeshHandle; length: number } | null = null;
    /** The last live remesh: when, and how many dabs the stroke had. */
    let liveAt = 0;
    let liveDabs = 0;
    /** A live remesh in flight, if any (never two at once). */
    let liveRunning: Promise<unknown> | null = null;

    /** Live triangle slots, or null when the mesh is gone. */
    const triSlots = (mesh: MeshHandle): number | null => {
      try {
        return scene.views(mesh).indices().length / 3;
      } catch {
        return null;
      }
    };

    /** Splits between masked vertices join the mask, in order with the stroke: before the next dab. */
    const maskLive = (mesh: MeshHandle) => maskTracker.liveSnapshot(mesh, maskRef.current);

    /**
     * Fails the mask loudly: clears it with its notice (the toolbox shows
     * it) and counts it for the test hook, so a topology edit that took a
     * masked vertex never silently re-points the brush.
     */
    const maskWentStale = () => {
      maskTracker.invalidate(maskRef.current?.mesh);
      app.setSculptMask(null);
      app.setSculptMaskNotice("The mask went stale after a topology edit and was cleared — repaint it.");
      inspect.sculpt.maskStale++;
    };

    const adoptSplits = (mesh: MeshHandle, splits: ArrayLike<number> | undefined, slots?: number): void => {
      const m = maskRef.current;
      if (m === null || m.mesh !== mesh || splits === undefined || splits.length === 0) {
        return;
      }
      const painted = new Set(m.vertices);
      const fresh = inheritedSplits((id) => painted.has(id), splits);
      if (fresh.length === 0) {
        return;
      }
      const next: SculptMask = {
        ...m,
        vertices: unionMaskIds(m.vertices, Uint32Array.from(fresh).sort()),
        inherited: new Set([...(m.inherited ?? []), ...fresh]),
      };
      if (slots !== undefined) {
        maskTracker.trustSplits(mesh, m.vertices, next.vertices, slots);
      }
      maskRef.current = next;
      app.setSculptMask(next);
    };

    /** The one place a dab meets the mask: `op` with engine-localized soft-band protection in O(|masked|). */
    const protectedDab = (mesh: MeshHandle, op: RepairOp): RepairOp => {
      const gated = maskLive(mesh);
      if (gated === null || gated === "stale" || !("radius" in op)) {
        return op;
      }
      return protectMaskedDab(op, gated.vertices, gated.positions, gated.live, maskRef.current?.inherited);
    };

    /**
     * Whether a stroke-end remesh of this region would touch the mask. A full
     * pass can collapse what it covers, so it stands down there: the engine's
     * remesh has no pinned vertices, and the mask promises they stay. The live
     * pass needs no such check: it only splits, which keeps every id and position.
     */
    const remeshMeetsMask = (mesh: MeshHandle, b: { center: [number, number, number]; radius: number }): boolean => {
      const gated = maskLive(mesh);
      return gated !== null && gated !== "stale" && maskMeetsRemesh(gated.vertices, gated.positions, gated.indices, b.center, b.radius);
    };

    /** A stale mask fails loudly once (cleared, never re-pointed), and the dab paints on. */
    const checkMaskLive = (mesh: MeshHandle): void => {
      if (maskLive(mesh) === "stale") {
        maskWentStale();
      }
    };

    /** The remesh density anchor, taken lazily: only remeshes on <=100k meshes need it. */
    const ensureDensity = (mesh: MeshHandle): void => {
      if (densityAnchor !== null && densityAnchor.mesh === mesh) {
        return;
      }
      densityAnchor = null;
      try {
        const v = scene.views(mesh);
        const indices = v.indices();
        if (remeshTooBig(indices.length / 3)) {
          return;
        }
        const mean = meanEdgeLength(v.positions(), indices);
        if (mean !== null && mean > 0 && Number.isFinite(mean)) {
          densityAnchor = { mesh, length: mean };
        }
      } catch {
        // Removed meanwhile: the remeshes fall back to the engine mean.
      }
    };

    /**
     * Before the first dab: rest sizes unless the mesh holds valid ones. The
     * engine drops them when an edit outdates them, so strokes share one rest
     * shape for as long as it is true, and nothing here has to track that.
     */
    const ensureRest = (mesh: MeshHandle): void => {
      if (!liveRef.current) {
        return;
      }
      store.repair(mesh, { op: "captureRestSizes", onlyIfMissing: true }).catch(() => {
        // Removed meanwhile: the stroke paints on, and its passes fail alike.
      });
    };

    /** Splits what the stroke stretched inside `b`; the stroke paints on if it fails. */
    const splitStretched = (mesh: MeshHandle, b: { center: [number, number, number]; radius: number }) => {
      maskTracker.beginSplit(mesh);
      return store
        .repair(mesh, { op: "splitStretched", center: b.center, radius: b.radius, maxSplits: liveMaxSplits(triSlots(mesh)) })
        .then((report) => {
          const sr = report as StretchReport;
          const changed = sr.splits > 0 || sr.vertices.length > 0;
          let slots: number | undefined;
          if (changed) {
            try {
              slots = scene.views(mesh).positions().length / 3;
            } catch {
              slots = undefined;
            }
          }
          maskTracker.endSplit(mesh, changed, slots, sr.vertices);
          adoptSplits(mesh, sr.vertices, slots);
          return sr;
        })
        .catch(() => {
          maskTracker.endSplit(mesh, false);
          // A degenerate region, or a mesh removed meanwhile.
          return undefined;
        });
    };

    /**
     * Splits the triangles the stroke stretched, live (see `liveRemeshDue`
     * for the per-frame budget). Same undo step as the stroke (it just
     * repairs while the stroke's step is open, never opening one).
     */
    const maybeLiveRemesh = () => {
      if (!liveRef.current || stroke === null) {
        return;
      }
      if (!liveRemeshDue(stroke.dabs - liveDabs, performance.now() - liveAt, liveRunning !== null)) {
        return;
      }
      const b = strokeBounds(stroke, grabPeakReach, grabPress);
      if (b === null) {
        return;
      }
      const mesh = brushRef.current.mesh;
      liveAt = performance.now();
      liveDabs = stroke.dabs;
      inspect.sculpt.liveRemeshes++;
      // In order with the stroke's edits, so that the stroke's end waits for it
      // and the engine's StrokeSnapshot interpolates any new split midpoints.
      liveRunning = stroke.pass(() =>
        splitStretched(mesh, b).then(() => {
          liveRunning = null;
          liveAt = performance.now();
        }),
      );
    };

    const cast = (x: number, y: number) => {
      const r = canvas.getBoundingClientRect();
      caster.setFromCamera(
        new THREE.Vector2(((x - r.left) / r.width) * 2 - 1, 1 - ((y - r.top) / r.height) * 2),
        camera,
      );
      const { origin, direction } = caster.ray;
      return scene.raycast([origin.x, origin.y, origin.z], [direction.x, direction.y, direction.z]);
    };

    /** The hit over the mesh's local views: its point, and the surface's way there for the cursor. */
    const localHit = (hit: { triangle: number; barycentric: readonly [number, number, number] }) => {
      const v = scene.views(brushRef.current.mesh);
      const indices = v.indices();
      const positions = v.positions();
      const corners = [indices[3 * hit.triangle], indices[3 * hit.triangle + 1], indices[3 * hit.triangle + 2]];
      cursorNormal = easeNormal(cursorNormal, faceNormal(positions, corners[0], corners[1], corners[2]));
      return interpolateHit(positions, corners, hit.barycentric);
    };

    const hideBall = () => {
      hoverLocal = null;
      cursorNormal = null;
      ball.visible = false;
      inspect.sculpt.hovering = false;
    };

    /** The cursor on the surface point: a child of the mesh, in its local frame. */
    const showBall = (point: readonly [number, number, number]) => {
      const binding = view.bindings.get(brushRef.current.mesh);
      if (binding === undefined) {
        hideBall();
        return;
      }
      if (ball.parent !== binding.object) {
        binding.object.add(ball);
      }
      const brush = brushRef.current;
      inspect.sculpt.cursorShows++;
      placeBrushCursor(ball, point, cursorNormal ?? [0, 0, 1], brush.radius);
      // Which way the surface will go: along the normal, out or in. Smooth has no way, and a grab shows its drag.
      const [nx, ny, nz] = cursorNormal ?? [0, 0, 1];
      const out =
        brush.op === "inflate"
          ? Math.sign(brush.strength) * NORMAL_MARKER * brush.radius
          : brush.op === "crease" && brush.tool === "crease"
            ? -NORMAL_MARKER * brush.radius
            : 0;
      setBrushDirection(ball, [out * nx, out * ny, out * nz]);
    };

    /** The cursor during a grab: where the press was and how far it reaches now, with the drag marked. */
    const showGrab = (press: readonly [number, number, number], drag: readonly [number, number, number], reach: number) => {
      showBall(press);
      placeBrushCursor(ball, press, cursorNormal ?? [0, 0, 1], reach);
      setBrushDirection(ball, drag);
    };

    /** Where a pass over `s` has to look: a grab's widened reach goes past the dabs' bounds. */
    const strokeBounds = (s: SculptStroke, peakReach: number, press: readonly [number, number, number] | null) => {
      const b = s.bounds();
      if (b === null || peakReach <= 0 || press === null) {
        return b;
      }
      const off = Math.hypot(press[0] - b.center[0], press[1] - b.center[1], press[2] - b.center[2]);
      return { center: b.center, radius: Math.max(b.radius, off + peakReach * 1.5) };
    };

    const endStroke = () => {
      view.stroking = false;
      clearInterval(buildup);
      buildup = undefined;
      active = -1;
      grabFrom = null;
      const lastBounds = stroke === null ? null : strokeBounds(stroke, grabPeakReach, grabPress);
      grabPress = null;
      grabPeakReach = 0;
      dirRef.current = [0, 0, 0];
      if (stroke !== null) {
        const s = stroke;
        syncDabs();
        stroke = null;
        inspect.sculpt.strokes++;
        // Undo and redo wait for all of this to land.
        const landing: Promise<void> = s.done.then(() => {
          if (view.strokeLanding === landing) {
            view.strokeLanding = null;
          }
        });
        view.strokeLanding = landing;
        // The last dabs since the live pass get their splits before the step
        // closes: one undo step for the stroke and its splits (on the sync
        // scene). `pass` keeps them in order, and the stroke open until they land.
        if (liveRef.current) {
          const b = lastBounds;
          if (b !== null) {
            const mesh = brushRef.current.mesh;
            inspect.sculpt.liveRemeshes++;
            void s.pass(() => splitStretched(mesh, b));
          }
        }
        // The optional evening-out, off by default: it rewrites its whole
        // region, and it stands down on huge meshes, where it would stall the
        // worker.
        if (remeshRef.current && landed > 0) {
          const b = s.bounds();
          const mesh = brushRef.current.mesh;
          const slots = triSlots(mesh);
          if (b !== null && slots !== null && !remeshTooBig(slots) && !remeshMeetsMask(mesh, b)) {
            ensureDensity(mesh);
            maskTracker.invalidate(mesh);
            inspect.sculpt.remeshes++;
            // In order with the stroke: after what is due and any live pass.
            void s.pass(() =>
              store
                .repair(mesh, {
                  op: "remesh",
                  center: b.center,
                  radius: b.radius,
                  maxRounds: 4,
                  targetLength: densityAnchor?.mesh === mesh ? densityAnchor.length : undefined,
                })
                .catch(() => {
                  // A degenerate region (or a mesh removed meanwhile): the
                  // stroke still ends.
                }),
            );
          }
        }
        s.end();
      }
      if (view.orbit !== null) {
        view.orbit.enabled = true;
      }
    };

    /** Counts a stroke's work as landed; a failure means the mesh went away, and ends the stroke. */
    const landing = (work: Promise<unknown>) =>
      work
        .then((r) => {
          landed++;
          return r;
        })
        .catch((err: unknown) => {
          // Let the stroke count the failed dab.
          endStroke();
          throw err;
        });

    /** Refreshes the grab's total drag from the pointer's screen offset since the press. */
    const trackGrab = (x: number, y: number, point: readonly [number, number, number]) => {
      const from = grabFrom;
      const binding = view.bindings.get(brushRef.current.mesh);
      if (from === null || binding === undefined) {
        return;
      }
      const world = binding.object.localToWorld(new THREE.Vector3(point[0], point[1], point[2]));
      const distance = camera.position.distanceTo(world);
      const height = canvas.clientHeight;
      if (!(distance > 0) || !(height > 0)) {
        return;
      }
      const fov = camera instanceof THREE.PerspectiveCamera ? camera.fov : 45;
      const e = camera.matrixWorld.elements;
      const q = binding.object.getWorldQuaternion(new THREE.Quaternion()).invert();
      dirRef.current = grabLocalDelta(
        x - from.x,
        y - from.y,
        worldPerPixel(distance, fov, height),
        [e[0], e[1], e[2]],
        [e[4], e[5], e[6]],
        [q.x, q.y, q.z, q.w],
      );
    };

    /**
     * Drives the engine's localized `grab` op from the stroke's pre-stroke
     * rest snapshot (`fromStroke: true, adaptiveReach: true`) in O(brush),
     * with O(1) cursor reach and O(|masked|) protection (#197).
     */
    const paintGrab = (x: number, y: number) => {
      const s = stroke;
      const press = grabPress;
      if (s === null || press === null) {
        return;
      }
      trackGrab(x, y, press);
      const g: [number, number, number] = [dirRef.current[0], dirRef.current[1], dirRef.current[2]];
      const brush = brushRef.current;
      const k = brush.strength;
      const reach = widestGrabReach(brush.radius, brush.falloff, brush.methods?.grab ?? "sphere", brush.strength, g);
      grabPeakReach = Math.max(grabPeakReach, reach);
      const at: [number, number, number] = [press[0] + k * g[0], press[1] + k * g[1], press[2] + k * g[2]];
      const mesh = brush.mesh;
      s.move(
        () => {
          const cost = inspect.sculpt.dabCost;
          const start = performance.now();
          const rawOp = dabOperation({ ...brushRef.current, tool: brushRef.current.tool ?? "grab", direction: g }, press);
          const heldOp = protectedDab(mesh, rawOp);
          const sent = performance.now();
          if (cost.maskMs.length < 4096) {
            cost.maskMs.push(sent - start);
          }
          return landing(store.repair(mesh, heldOp)).then((r) => {
            if (cost.landMs.length < 4096) {
              cost.landMs.push(performance.now() - sent);
            }
            if ((r as { moved?: number }).moved === 0 && "protect" in heldOp) {
              inspect.sculpt.maskedDabs++;
              liveDabs++;
            }
            return r;
          });
        },
        at,
      );
      inspect.sculpt.grabTotal = Math.sqrt(g[0] * g[0] + g[1] * g[1] + g[2] * g[2]);
      inspect.sculpt.lastDab = at;
      showGrab(press, [k * g[0], k * g[1], k * g[2]], reach);
      syncDabs();
      maybeLiveRemesh();
    };

    const run = () => {
      if (!alive || running || pending === null) {
        return;
      }
      const wait = last + pending.period - performance.now();
      if (wait > 0) {
        timer ??= setTimeout(() => {
          timer = undefined;
          run();
        }, wait);
        return;
      }
      const at = pending;
      pending = null;
      if (stroke !== null && brushRef.current.op === "grab") {
        paintGrab(at.x, at.y);
        // From when it is done: where a move takes longer than the pointer's events
        // (a wide grab of a huge mesh on the sync scene), the moves between are skipped
        // instead of piling up, and the shape only depends on the last one.
        last = performance.now();
        return;
      }
      running = true;
      last = performance.now();
      Promise.resolve()
        .then(() => cast(at.x, at.y))
        .then(
          (hit) => {
            running = false;
            if (!alive) {
              return;
            }
            if (hit === null || hit.mesh !== brushRef.current.mesh) {
              hideBall();
              stroke?.lift();
            } else {
              let point: [number, number, number];
              try {
                point = localHit(hit);
              } catch {
                // Removed meanwhile: the next move asks again.
                run();
                return;
              }
              hoverLocal = point;
              inspect.sculpt.hovering = true;
              showBall(point);
              if (stroke !== null) {
                stroke.dab(point);
                syncDabs();
                maybeLiveRemesh();
              }
            }
            run();
          },
          () => {
            running = false;
            if (alive) {
              run();
            }
          },
        );
    };

    const onMove = (e: PointerEvent) => {
      if (stroke !== null && e.pointerId !== active) {
        return;
      }
      pending = { x: e.clientX, y: e.clientY, period: stroke !== null ? STROKE_PERIOD_MS : HOVER_PERIOD_MS };
      run();
    };
    const onDown = (e: PointerEvent) => {
      if (e.button !== 0 || isExtraFinger(e) || active !== -1 || view.dragging) {
        return;
      }
      pressing = e.pointerId;
      // Immediate: the stroke starts where the button lands, not a throttle later.
      Promise.resolve()
        .then(() => cast(e.clientX, e.clientY))
        .then(
          (hit) => {
            // The gizmo's own pointerdown runs first (its listeners attached
            // first): a grab there owns the gesture, not a stroke. A second
            // finger landed meanwhile: the camera's pinch owns it.
            if (!alive || active !== -1 || view.dragging || view.multiTouch) {
              return;
            }
            const brush = brushRef.current;
            if (hit === null || hit.mesh !== brush.mesh) {
              return; // empty space (or another mesh): the orbit keeps the gesture.
            }
            let point: [number, number, number];
            try {
              point = localHit(hit);
            } catch {
              return;
            }
            hoverLocal = point;
            inspect.sculpt.hovering = true;
            showBall(point);
            // Step nesting is the sync scene's; the Worker steps per dab.
            const maybe = scene as unknown as {
              beginStep?: () => void;
              endStep?: () => void;
              cancelStep?: () => void;
            };
            const steps: StrokeSteps = {
              beginStep: typeof maybe.beginStep === "function" ? () => maybe.beginStep?.() : undefined,
              endStep: typeof maybe.endStep === "function" ? () => maybe.endStep?.() : undefined,
              cancelStep: typeof maybe.cancelStep === "function" ? () => maybe.cancelStep?.() : undefined,
            };
            const mesh = brush.mesh;
            ensureRest(mesh);
            landed = 0;
            liveDabs = 0;
            grabFrom = { x: e.clientX, y: e.clientY };
            grabPress = brush.op === "grab" ? point : null;
            grabPeakReach = 0;
            dirRef.current = [0, 0, 0];
            // Where the scene nests steps, each dab is its own edit (its cost
            // follows the brush). The Worker takes them as a list: it runs a
            // list one op at a time too, as one undo step.
            const nests = steps.beginStep !== undefined;
            const s = new SculptStroke(
              () => ({ ...brushRef.current, direction: dirRef.current }),
              steps,
              (ops) => {
                const cost = inspect.sculpt.dabCost;
                const start = performance.now();
                const held = ops.map((op) => protectedDab(mesh, op));
                const sent = performance.now();
                if (cost.maskMs.length < 4096) {
                  cost.maskMs.push(sent - start);
                }
                return landing(
                  nests || held.length === 1
                    ? Promise.all(held.map((op) => store.repair(mesh, op)))
                    : store.repair(mesh, held),
                ).then((reports) => {
                  if (cost.landMs.length < 4096) {
                    cost.landMs.push(performance.now() - sent);
                  }
                  // A dab with only masked vertices under it moved nothing (counted, like a skip).
                  (reports as readonly { moved?: number }[]).forEach((r, k) => {
                    if (r.moved === 0 && "protect" in held[k]) {
                      inspect.sculpt.maskedDabs++;
                      // Nothing moved, so nothing stretched: no live pass for this one.
                      liveDabs++;
                    }
                  });
                  return reports;
                });
              },
              () => {
                checkMaskLive(mesh);
                return false;
              },
            );
            active = e.pointerId;
            stroke = s;
            view.stroking = true;
            // Its release is not a click on whatever is behind the mesh.
            view.pressTaken = true;
            counted = 0;
            if (view.orbit !== null) {
              view.orbit.enabled = false;
            }
            const needsStrokeState =
              brush.op === "grab" || (brush.op === "inflate" && brush.methods?.inflate === "strokeNormal");
            if (needsStrokeState) {
              void s.pass(() => store.repair(mesh, { op: "captureStrokeState", onlyIfMissing: false }).catch(() => undefined));
            }
            // Grab is move-driven only: the first move paints from the engine's
            // O(1) rest snapshot (a click without a drag leaves no step, no remesh).
            if (brush.op !== "grab") {
              s.dab(point);
            } else {
              checkMaskLive(mesh);
              grabPeakReach = brush.radius;
            }
            syncDabs();
            // Released meanwhile (a long frame runs the down and the up together): the
            // press keeps its dab, but its release has gone by, so nothing else ends the stroke.
            if (pressing !== e.pointerId) {
              endStroke();
              return;
            }
            clearInterval(buildup);
            let builtAt = performance.now();
            buildup = setInterval(() => {
              // Grab is move-driven only: holding still must not drift.
              if (brushRef.current.op === "grab") {
                return;
              }
              const at = hoverLocal;
              if (at !== null) {
                // By the time gone, so a busy scene builds up as fast as an idle one.
                const now = performance.now();
                if (stroke?.dab(at, true, Math.min(1, (now - builtAt) / BUILDUP_FULL_MS)) === "stamped") {
                  builtAt = now;
                }
                syncDabs();
                maybeLiveRemesh();
              }
            }, BUILDUP_MS);
          },
          () => {
            // Removed meanwhile: the next press asks again.
          },
        );
    };
    const onUp = (e: PointerEvent) => {
      pressing = -1;
      if (e.pointerId === active) {
        endStroke();
      }
    };
    const onLeave = (e: PointerEvent) => {
      pressing = -1;
      if (stroke !== null) {
        if (e.pointerId === active) {
          endStroke();
        }
        return;
      }
      pending = null;
      hideBall();
    };
    canvas.addEventListener("pointermove", onMove);
    canvas.addEventListener("pointerdown", onDown);
    canvas.addEventListener("pointerup", onUp);
    canvas.addEventListener("pointercancel", onUp);
    canvas.addEventListener("pointerleave", onLeave);
    return () => {
      alive = false;
      clearTimeout(timer);
      canvas.removeEventListener("pointermove", onMove);
      canvas.removeEventListener("pointerdown", onDown);
      canvas.removeEventListener("pointerup", onUp);
      canvas.removeEventListener("pointercancel", onUp);
      canvas.removeEventListener("pointerleave", onLeave);
      endStroke();
      hideBall();
    };
  }, [scene, camera, gl, store, ball, app, maskTracker]);

  return null;
}

/**
 * The mask paint on the selected mesh: hover shows the paint ball (blue,
 * against sculpt's amber), press-and-hold accumulates sphere-brush picks
 * into the mask. Painting never opens an undo step (the mask is not a mesh
 * edit); a new mesh replaces the mask, and a stale one fails loudly (see
 * `viewport/sculpt.ts`'s mask contract). One raycast at a time, through the
 * engine's BVH.
 */
function MaskLayer(props: { app: AppState; scene: Scene | WorkerScene; maskTracker: MaskLiveTracker }) {
  const { app, scene, maskTracker } = props;
  const store = useScene();
  const [selection] = useSelection();
  const camera = useThree((s) => s.camera);
  const gl = useThree((s) => s.gl);
  // The selection and radius drive the paint live, outside React.
  const selRef = useRef(selection);
  selRef.current = selection;
  const radius = useApp(app, (s) => s.maskRadius);
  const radiusRef = useRef(radius);
  radiusRef.current = radius;

  const ball = useMemo(() => makeBrushCursor(0x6ea8d9), []);
  useEffect(() => {
    return () => {
      ball.removeFromParent();
      for (const child of ball.children) {
        const part = child as THREE.Mesh | THREE.Line;
        part.geometry.dispose();
        (part.material as THREE.Material).dispose();
      }
    };
  }, [ball]);

  useEffect(() => {
    const canvas = gl.domElement;
    let cursorNormal: [number, number, number] | null = null;
    /** The pointer whose press is waiting on its raycast. */
    let pressing = -1;
    const caster = new THREE.Raycaster();
    let alive = true;
    let running = false;
    let last = -Infinity;
    let timer: ReturnType<typeof setTimeout> | undefined;
    let pending: { x: number; y: number; period: number } | null = null;
    /** The painting pointer, or -1 when no paint accumulates. */
    let active = -1;
    /** The mesh the paint accumulates on (the selection at press). */
    let paintMesh: MeshHandle | null = null;
    /** Serializes async `verticesInSphere` picks so rapid moves never race `sculptMask`. */
    let paintTail: Promise<void> = Promise.resolve();

    const cast = (x: number, y: number) => {
      const r = canvas.getBoundingClientRect();
      caster.setFromCamera(
        new THREE.Vector2(((x - r.left) / r.width) * 2 - 1, 1 - ((y - r.top) / r.height) * 2),
        camera,
      );
      const { origin, direction } = caster.ray;
      return scene.raycast([origin.x, origin.y, origin.z], [direction.x, direction.y, direction.z]);
    };

    /** The hit over the mesh's local views: its point, and the surface's way there for the cursor. */
    const localHit = (mesh: MeshHandle, hit: { triangle: number; barycentric: readonly [number, number, number] }) => {
      const v = scene.views(mesh);
      const indices = v.indices();
      const positions = v.positions();
      const corners = [indices[3 * hit.triangle], indices[3 * hit.triangle + 1], indices[3 * hit.triangle + 2]];
      cursorNormal = easeNormal(cursorNormal, faceNormal(positions, corners[0], corners[1], corners[2]));
      return interpolateHit(positions, corners, hit.barycentric);
    };

    const hideBall = () => {
      cursorNormal = null;
      ball.visible = false;
      inspect.sculpt.hovering = false;
    };

    /** The cursor on the surface point: a child of the mesh, in its local frame. */
    const showBall = (mesh: MeshHandle, point: readonly [number, number, number]) => {
      const binding = view.bindings.get(mesh);
      if (binding === undefined) {
        hideBall();
        return;
      }
      if (ball.parent !== binding.object) {
        binding.object.add(ball);
      }
      inspect.sculpt.cursorShows++;
      placeBrushCursor(ball, point, cursorNormal ?? [0, 0, 1], radiusRef.current);
    };

    /** Accumulates one BVH-localized sphere pick (plus active symmetry mirrors) into the mask. */
    const paintAt = async (mesh: MeshHandle, point: readonly [number, number, number]): Promise<void> => {
      const radius = radiusRef.current;
      const sym = toSymmetry(app.getSnapshot().sculptSymmetry);
      const centers = reflectCenters(point, sym);
      const topoBefore = maskTracker.topologyVersion(mesh);
      let picked: Uint32Array[];
      try {
        picked = await Promise.all(centers.map((c) => store.verticesInSphere(mesh, c, radius)));
      } catch {
        return; // Removed meanwhile or stale BVH: the next move asks again.
      }
      if (!alive) {
        return;
      }
      let slots: number;
      try {
        slots = scene.views(mesh).positions().length / 3;
      } catch {
        return; // Removed meanwhile: the next move asks again.
      }
      let fresh = picked[0] ?? new Uint32Array(0);
      for (let k = 1; k < picked.length; k++) {
        fresh = unionMaskIds(fresh, picked[k]);
      }
      const topoSame = topoBefore === maskTracker.topologyVersion(mesh);
      const prev = app.getSnapshot().sculptMask;
      if (prev !== null && prev.mesh === mesh) {
        const check = maskTracker.checkMask(mesh, prev);
        if (check === null) {
          return;
        }
        if (!check.ok) {
          // Loud, never a silent reinterpretation: drop it, count it, paint fresh.
          maskTracker.invalidate(mesh);
          app.setSculptMask(null);
          app.setSculptMaskNotice("The mask went stale after a topology edit and was cleared — repaint it.");
          inspect.sculpt.maskStale++;
          if (topoSame) {
            maskTracker.trust(mesh, fresh, slots);
          }
          app.setSculptMask({ mesh, vertices: fresh, slots });
          return;
        }
        const nextVertices = unionMaskIds(prev.vertices, fresh);
        if (topoSame) {
          maskTracker.trust(mesh, nextVertices, slots, prev.inherited === undefined || prev.inherited.size === 0);
        }
        app.setSculptMask({ mesh, vertices: nextVertices, slots, inherited: prev.inherited });
      } else {
        // A new mesh replaces the mask, like the sculpt brush re-arms.
        if (topoSame) {
          maskTracker.trust(mesh, fresh, slots);
        }
        app.setSculptMask({ mesh, vertices: fresh, slots });
      }
      // A successful paint resolves the loud notice.
      if (app.getSnapshot().sculptMaskNotice !== null) {
        app.setSculptMaskNotice(null);
      }
    };

    const enqueuePaint = (mesh: MeshHandle, point: readonly [number, number, number]): Promise<void> => {
      paintTail = paintTail.then(
        () => paintAt(mesh, point),
        () => paintAt(mesh, point),
      );
      return paintTail;
    };

    /** Ends the paint: the orbit goes back to the gesture. */
    const endPaint = () => {
      active = -1;
      paintMesh = null;
      if (view.orbit !== null) {
        view.orbit.enabled = true;
      }
    };

    const run = () => {
      if (!alive || running || pending === null) {
        return;
      }
      const wait = last + pending.period - performance.now();
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
      const mesh = paintMesh ?? selRef.current;
      Promise.resolve()
        .then(() => (mesh === null ? null : cast(at.x, at.y)))
        .then(async (hit) => {
          if (!alive || mesh === null) {
            return;
          }
          if (hit === null || hit.mesh !== mesh) {
            if (active === -1) {
              hideBall();
            }
            return;
          }
          let point: [number, number, number];
          try {
            point = localHit(mesh, hit);
          } catch {
            // Removed meanwhile: the next move asks again.
            return;
          }
          inspect.sculpt.hovering = true;
          showBall(mesh, point);
          if (active !== -1 && paintMesh !== null) {
            await enqueuePaint(paintMesh, point);
          }
        })
        .catch(() => {
          // Removed meanwhile: the next move asks again.
        })
        .finally(() => {
          running = false;
          if (alive) {
            run();
          }
        });
    };

    const onMove = (e: PointerEvent) => {
      if (active !== -1 && e.pointerId !== active) {
        return;
      }
      pending = { x: e.clientX, y: e.clientY, period: active !== -1 ? STROKE_PERIOD_MS : HOVER_PERIOD_MS };
      run();
    };
    const onDown = (e: PointerEvent) => {
      if (e.button !== 0 || isExtraFinger(e) || active !== -1 || view.dragging) {
        return;
      }
      const mesh = selRef.current;
      if (mesh === null) {
        return;
      }
      pressing = e.pointerId;
      // Immediate: the paint starts where the button lands, not a throttle later.
      Promise.resolve()
        .then(() => cast(e.clientX, e.clientY))
        .then(
          (hit) => {
            if (!alive || active !== -1 || view.dragging || view.multiTouch) {
              return;
            }
            if (hit === null || hit.mesh !== mesh) {
              return; // empty space (or another mesh): the orbit keeps the gesture.
            }
            let point: [number, number, number];
            try {
              point = localHit(mesh, hit);
            } catch {
              return;
            }
            inspect.sculpt.hovering = true;
            showBall(mesh, point);
            // Released meanwhile (a long frame runs the down and the up together): the
            // press still paints where it landed, but nothing would end a paint left open.
            if (pressing !== e.pointerId) {
              void enqueuePaint(mesh, point);
              return;
            }
            active = e.pointerId;
            // Its release is not a click on whatever is behind the mesh.
            view.pressTaken = true;
            paintMesh = mesh;
            if (view.orbit !== null) {
              view.orbit.enabled = false;
            }
            void enqueuePaint(mesh, point);
          },
          () => {
            // Removed meanwhile: the next press asks again.
          },
        );
    };
    const onUp = (e: PointerEvent) => {
      pressing = -1;
      if (e.pointerId === active) {
        endPaint();
      }
    };
    const onLeave = (e: PointerEvent) => {
      pressing = -1;
      if (active !== -1) {
        if (e.pointerId === active) {
          endPaint();
        }
        return;
      }
      pending = null;
      hideBall();
    };
    canvas.addEventListener("pointermove", onMove);
    canvas.addEventListener("pointerdown", onDown);
    canvas.addEventListener("pointerup", onUp);
    canvas.addEventListener("pointercancel", onUp);
    canvas.addEventListener("pointerleave", onLeave);
    return () => {
      alive = false;
      clearTimeout(timer);
      canvas.removeEventListener("pointermove", onMove);
      canvas.removeEventListener("pointerdown", onDown);
      canvas.removeEventListener("pointerup", onUp);
      canvas.removeEventListener("pointercancel", onUp);
      canvas.removeEventListener("pointerleave", onLeave);
      endPaint();
      hideBall();
    };
  }, [scene, store, camera, gl, ball, app, maskTracker]);

  return null;
}

/**
 * Trackball camera (free flipping in every direction, no poles); a
 * TransformControls gizmo on `bound`; and the camera pointed at a mesh on
 * each new `focus` request.
 */
function Controls(props: {
  bound: MeshBinding | null;
  mode: GizmoMode;
  focus: FocusRequest | null;
  resetView: number;
  radiusOf: (mesh: MeshHandle) => number;
}) {
  const { bound, mode, focus, resetView, radiusOf } = props;
  const store = useScene();
  const camera = useThree((s) => s.camera);
  const gl = useThree((s) => s.gl);
  const scene = useThree((s) => s.scene);
  const size = useThree((s) => s.size);
  const [orbit, setOrbit] = useState<TrackballControls | null>(null);
  const [gizmo, setGizmo] = useState<TransformControls | null>(null);
  // Read when a focus request arrives: a new list of items is not a new request.
  const radiusRef = useRef(radiusOf);
  radiusRef.current = radiusOf;

  useEffect(() => {
    view.camera = camera;
    view.canvas = gl.domElement;
    const unwatch = watchFingers(gl.domElement); // before the trackball: see touch.ts
    const o = new TrackballControls(camera, gl.domElement);
    o.target.set(-0.6, 0.9, 0);
    // No inertia: the camera stops where the pointer lifts (deterministic).
    o.staticMoving = true;
    o.update();
    view.orbit = o;
    setOrbit(o);
    return () => {
      o.dispose();
      unwatch();
      setOrbit((c) => (c === o ? null : c));
      if (view.camera === camera) {
        view.camera = null;
        view.canvas = null;
      }
      if (view.orbit === o) {
        view.orbit = null;
      }
    };
  }, [camera, gl]);

  // The trackball caches the canvas's box; a rotated phone or a resized panel changes it.
  useEffect(() => {
    orbit?.handleResize();
  }, [orbit, size]);

  useFrame(() => orbit?.update());

  useEffect(() => {
    if (orbit === null || bound === null) {
      return;
    }
    // The gizmo takes every pointer on the canvas, so a second finger would
    // grab a handle or move the one in hand, and its lift would end the drag.
    // It stands down for that finger's events: this runs before its listeners.
    const canvas = gl.domElement;
    const gate = (e: PointerEvent) => {
      controls.enabled = !isExtraFinger(e);
    };
    // It has no pointercancel of its own: a drag the system took over ends here.
    const onCancel = (e: PointerEvent) => {
      if (!isExtraFinger(e) && controls.dragging) {
        controls.pointerUp(null);
      }
    };
    const POINTER_EVENTS = ["pointerdown", "pointermove", "pointerup"] as const;
    for (const type of POINTER_EVENTS) {
      canvas.addEventListener(type, gate);
    }
    const controls = new TransformControls(camera, canvas);
    canvas.addEventListener("pointercancel", onCancel);
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
      for (const type of POINTER_EVENTS) {
        canvas.removeEventListener(type, gate);
      }
      canvas.removeEventListener("pointercancel", onCancel);
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

  // Reset view: the default pose (and an upright camera: the trackball
  // may have rolled it).
  useEffect(() => {
    if (orbit === null || resetView === 0) {
      return;
    }
    camera.position.set(-0.6, 3.4, 8.2);
    camera.up.set(0, 1, 0);
    if (camera instanceof THREE.PerspectiveCamera) {
      camera.near = 0.01;
      camera.far = 5000;
      camera.updateProjectionMatrix();
    }
    orbit.target.set(-0.6, 0.9, 0);
    orbit.update();
  }, [orbit, resetView, camera]);

  // Focus: the orbit target to the mesh's centre (its origin, where the
  // gizmo sits), the camera back along the requested (or current)
  // direction. A focus box frames that world box instead of the mesh.
  useEffect(() => {
    if (orbit === null || focus === null) {
      return;
    }
    let center: THREE.Vector3;
    let radius: number;
    if (focus.box !== undefined) {
      center = new THREE.Vector3(focus.box.center[0], focus.box.center[1], focus.box.center[2]);
      radius = Math.max(focus.box.radius, 1e-3);
    } else {
      let m: Float64Array;
      try {
        m = store.scene.matrix4(focus.mesh);
      } catch {
        return; // removed meanwhile
      }
      center = new THREE.Vector3(m[12], m[13], m[14]);
      radius = Math.max(radiusRef.current(focus.mesh), 1e-3);
    }
    const dir =
      focus.dir === undefined
        ? camera.position.clone().sub(orbit.target)
        : new THREE.Vector3(focus.dir[0], focus.dir[1], focus.dir[2]);
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
