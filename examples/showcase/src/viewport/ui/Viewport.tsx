// The canvas: `<FastMesh>` per mesh (click to select), a trackball camera and a
// TransformControls gizmo on the selected mesh (moves through
// `attachTransformControls`, one undo step per drag). `<FastMesh>` drives
// the store's frames. Colourings are drawn with fragment shading, so the
// heatmap's boundaries are crisp, softened by the collision panel's slider.
// With `hoverable` (Group mode) the vertex under the pointer is tracked in
// `view.hover`, for the hover tip: through the engine's BVH raycast
// (`scene.raycast`, async on the Worker), one at a time and at most every
// HOVER_PERIOD_MS, never three's per-triangle `Mesh.raycast`. This file
// composes the scene; how it behaves (clicks, the brushes' pointer loops, the
// gizmo, the camera) is in `viewport/ux/`.
import { Canvas, useFrame, useThree } from "@react-three/fiber";
import type { MeshHandle, Scene } from "fast-mesh-wasm";
import { FastMesh } from "fast-mesh-wasm/r3f";
import { useScene, useSelection, useStoreSelector } from "fast-mesh-wasm/react";
import { meshState } from "fast-mesh-wasm/store";
import type { MeshBinding } from "fast-mesh-wasm/three";
import { useCallback, useEffect, useLayoutEffect, useMemo, useRef } from "react";
import * as THREE from "three";
import type { WorkerScene } from "fast-mesh-wasm/worker";
import type { FocusRequest } from "../../app/App";
import type { Loader } from "../../io/loader";
import { inspect } from "../../app/state/readouts";
import { type AppState, type Item, useApp } from "../../app/state/app-state";
import { evaluateOverlayColor } from "../../engine/workbench-ops";
import { InspectLayers } from "../../features/inspect/ui/InspectLayers";
import { CreatePreviewLayer } from "../../features/home/ui/CreatePreviewLayer";
import { TextPreviewLayer } from "../../features/mesh-edit/ui/TextPreviewLayer";
import { SceneSlices } from "../../features/ct/ui/SceneSlices";
import { SceneVolume } from "../../features/ct/ui/SceneVolume";
import { matchVolume } from "../../features/ct/ux/volume-box";
import type { SculptBrush } from "../../features/sculpt/ux/sculpt";
import { makeBrushCursor } from "../../features/sculpt/ui/brush-cursor";
import { MaskOverlay } from "../../features/sculpt/ui/mask-overlay";
import type { MaskLiveTracker } from "../../features/sculpt/ux/mask-protect";
import {
  type AppliedShadingState,
  clearMeshDiagnosticShading,
  syncMeshDiagnosticShading,
} from "../../features/view/ui/shading";
import {
  countRepairOverlays,
} from "../../features/mesh-repair/ux/repair-overlay-data";
import { buildHealerOverlayGroup, disposeHealerOverlayGroup } from "../../features/mesh-repair/ui/healer-overlays";
import { DeformHandlesLayer } from "../../features/sculpt/ui/DeformHandlesLayer";
import { FaceSelectionLayer } from "../../features/select/ui/FaceSelectionLayer";
import { HollowPlaceToolsLayer } from "../../features/mesh-edit/ui/HollowPlaceToolsLayer";
import { MetrologyOverlaysLayer } from "../../features/points/ui/MetrologyOverlaysLayer";
import { PlacementPicksLayer } from "../../features/transform/ui/PlacementPicksLayer";
import { AlignToolsLayer } from "../../features/transform/ui/AlignToolsLayer";
import { SurfacePathLayer } from "../../features/inspect/ui/SurfacePathLayer";
import { useCameraRequests, useOrbitCamera } from "../ux/use-camera-controller";
import { useGizmoDrag, useMeshBindings } from "../ux/use-gizmo";
import { useHoverRay } from "../ux/use-hover-ray";
import { useMaskPaint, useMaskPaintDisarm, useMaskTracker } from "../ux/use-mask-paint";
import { useSculptStroke } from "../ux/use-sculpt-stroke";
import { useMeshClick, usePressTracker, useSelectionClicks } from "../ux/use-selection-clicks";

export type GizmoMode = "translate" | "rotate" | "scale";

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
  const store = useScene();
  const [selection, select] = useSelection();
  const multi = useApp(app, (s) => s.multi);
  const visMask = useStoreSelector((s) =>
    items.map((i) => (!i.locked && meshState(s, i.mesh).desired.visible ? "1" : "0")).join(""),
  );
  const hasEligibleMesh = visMask.includes("1");
  // Sculpt, mask paint, and Wave 6 viewport tools own the pointer; while an
  // inspect tool is armed the selection stays put (see `useSelectionClicks`).
  const inspectTool = useApp(app, (s) => s.inspectTool);
  const wave6SelectTool = useApp(app, (s) => s.wave6.select.activeTool);
  const wave6DeformTool = useApp(app, (s) => s.wave6.deform.activeTool);
  const wave6PrintTool = useApp(app, (s) => s.wave6.printPrep.activeTool);
  // An armed placement pick (#256) or landmark pick (#253) owns the pointer like the Wave 6 tools.
  const placementPicking = useApp(app, (s) => s.placementTool.pick !== null || s.alignTool.landmarks?.picking === true);
  // So does an armed geodesic or path pick (#252).
  const surfacePicking = useApp(app, (s) => s.surfacePath.armed !== null);
  const pointerOwnedByWave6 =
    wave6SelectTool !== null ||
    wave6DeformTool !== null ||
    wave6PrintTool !== null ||
    placementPicking ||
    surfacePicking;
  const { downAt, onMissed, onPick } = useSelectionClicks({
    app,
    select,
    selection,
    sculpt,
    maskPaint,
    pointerOwnedByWave6,
    picking,
    inspectTool,
  });
  useMaskPaintDisarm({ app, maskPaint, selection, items, hasEligibleMesh });
  const mpr = useApp(app, (s) => s.volumeMpr);
  // The mesh shown as the scene volume instead of its surface, if any.
  // Sculpting shows the surface: the armed mesh drops out of volume mode.
  const volumeFound = matchVolume(items, selection, mpr);
  const sculptMesh = sculpt?.mesh ?? null;
  const volumeMesh =
    volumeFound !== null && volumeFound.mpr.inScene && volumeFound.item.mesh !== sculptMesh
      ? volumeFound.item.mesh
      : null;
  const { bindings, onBind } = useMeshBindings();
  const selected = selection === null ? null : (bindings.get(selection) ?? null);
  const selectedItem = selection === null ? null : (items.find((i) => i.mesh === selection) ?? null);
  const pointerOwnedByPaint = sculpt !== null || maskPaint;
  const radiusOf = useCallback((mesh: MeshHandle) => items.find((i) => i.mesh === mesh)?.radius ?? 1, [items]);
  // Shared live-mask cache between MaskLayer and SculptLayer.
  const maskTracker = useMaskTracker(scene, store);

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
          suppressClick={pointerOwnedByPaint || pointerOwnedByWave6}
          wireframe={wireframe}
          bound={bindings.get(item.mesh) ?? null}
        />
      ))}
      {/* Sculpt, mask paint, and Wave 6 pointer tools (or a locked item) stand the gizmo down cleanly (#229, #235). */}
      <Controls
        app={app}
        items={items}
        bound={pointerOwnedByPaint || pointerOwnedByWave6 || selectedItem?.locked ? null : selected}
        mode={mode}
        focus={focus}
        resetView={resetView}
        radiusOf={radiusOf}
      />
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
      <PickCursor
        picking={
          picking !== null ||
          sculpt !== null ||
          wave6SelectTool !== null ||
          wave6PrintTool !== null ||
          placementPicking ||
          surfacePicking
        }
      />
      <MissTracker track={downAt} />
      <SceneSlices app={app} loader={loader} scene={scene} />
      <SceneVolume app={app} loader={loader} scene={scene} />
      <InspectLayers app={app} scene={scene} items={items} />
      <FaceSelectionLayer app={app} store={store} />
      <DeformHandlesLayer app={app} store={store} />
      <MetrologyOverlaysLayer app={app} store={store} />
      <HollowPlaceToolsLayer app={app} store={store} />
      <PlacementPicksLayer app={app} store={store} />
      <AlignToolsLayer app={app} store={store} />
      <SurfacePathLayer app={app} store={store} />
      <CreatePreviewLayer app={app} />
      <TextPreviewLayer app={app} />
    </Canvas>
  );
}

/** The canvas's pointerdowns, recorded for the missed click (`usePressTracker`). */
function MissTracker(props: { track: { current: { x: number; y: number } | null } }) {
  usePressTracker(props.track);
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
  const globalShading = useApp(app, (s) => s.shadingMode);
  const creaseAngleDeg = useApp(app, (s) => s.creaseAngleDeg);
  const effectiveShading = item.shadingOverride ?? globalShading;
  const shadingStateRef = useRef<AppliedShadingState | null>(null);
  useLayoutEffect(() => {
    item.material.emissive.set(selected ? 0x3a2a08 : secondary ? 0x1c1a10 : 0x000000);
  }, [item.material, selected, secondary]);
  useLayoutEffect(() => {
    if (bound !== null) {
      shadingStateRef.current = syncMeshDiagnosticShading({
        bound,
        material: item.material,
        mode: effectiveShading,
        globalWireframe: wireframe,
        creaseAngleDeg,
        prev: shadingStateRef.current,
      });
    }
    return () => {
      if (bound !== null) {
        clearMeshDiagnosticShading(bound, item.material);
        shadingStateRef.current = null;
      }
    };
  }, [bound, item.material, effectiveShading, wireframe, creaseAngleDeg]);
  useFrame(() => {
    if (bound !== null && effectiveShading === "autoCrease") {
      const idxCount = bound.object.geometry.getIndex()?.count ?? 0;
      const triCount = Math.floor(idxCount / 3);
      if (shadingStateRef.current?.triCount !== triCount) {
        shadingStateRef.current = syncMeshDiagnosticShading({
          bound,
          material: item.material,
          mode: effectiveShading,
          globalWireframe: wireframe,
          creaseAngleDeg,
          prev: shadingStateRef.current,
        });
      }
    }
  });
  const onClick = useMeshClick({
    app,
    mesh: item.mesh,
    scene,
    tool,
    picking,
    sculpted,
    suppressClick,
    onPick,
    onPickPoint,
  });
  return (
    <>
      <FastMesh
        mesh={item.mesh}
        material={item.material}
        shading="fragment"
        edgeSoftness={edgeSoftness}
        visible={!volumeShown}
        // While sculpting this mesh or mask-painting the selected mesh a click means nothing
        // here, and listening for one costs a per-triangle three raycast on every press and release.
        onClick={sculpted || suppressClick ? undefined : onClick}
        onBind={(b) => onBind(item.mesh, b)}
      />
      <RepairOverlays app={app} scene={scene} item={item} bound={bound} />
      <SculptMaskOverlay app={app} item={item} bound={bound} />
      <SurfaceFieldOverlay app={app} item={item} bound={bound} />
    </>
  );
}

/**
 * Vertex-colored overlay for client-evaluated surface fields (Gaussian curvature
 * or WorkerScene Thickness/Mean curvature) rendered over `bound.object` without
 * fighting `<FastMesh>`'s native shader uniforms.
 */
function SurfaceFieldOverlay(props: { app: AppState; item: Item; bound: MeshBinding | null }) {
  const { app, item, bound } = props;
  const field = useApp(app, (s) => (s.surfaceFieldValues?.mesh === item.mesh ? s.surfaceFieldValues : null));
  const overlayMesh = useMemo(() => {
    const g = new THREE.BufferGeometry();
    const m = new THREE.MeshBasicMaterial({
      vertexColors: true,
      side: THREE.DoubleSide,
      polygonOffset: true,
      polygonOffsetFactor: -1,
      polygonOffsetUnits: -1,
    });
    const mesh = new THREE.Mesh(g, m);
    mesh.matrixAutoUpdate = false;
    mesh.renderOrder = 1;
    return mesh;
  }, []);

  useEffect(() => {
    const parent = bound?.object ?? null;
    if (parent === null || field === null) {
      overlayMesh.visible = false;
      return;
    }
    const srcGeom = parent.geometry;
    const posAttr = srcGeom.getAttribute("position");
    const idxAttr = srcGeom.getIndex();
    if (!posAttr || !idxAttr) {
      overlayMesh.visible = false;
      return;
    }
    const vertCount = posAttr.count;
    const colors = new Float32Array(vertCount * 3);
    for (let i = 0; i < vertCount; i++) {
      const val = i < field.values.length ? field.values[i] : Number.NaN;
      const [r, g, b] = evaluateOverlayColor(field, val);
      colors[3 * i] = r;
      colors[3 * i + 1] = g;
      colors[3 * i + 2] = b;
    }
    overlayMesh.geometry.setAttribute("position", posAttr);
    overlayMesh.geometry.setIndex(idxAttr);
    overlayMesh.geometry.setAttribute("color", new THREE.BufferAttribute(colors, 3));
    overlayMesh.geometry.computeBoundingSphere();
    overlayMesh.visible = true;
    parent.add(overlayMesh);
    return () => {
      parent.remove(overlayMesh);
    };
  }, [bound, field, overlayMesh]);

  useEffect(() => {
    return () => {
      overlayMesh.geometry.dispose();
      (overlayMesh.material as THREE.Material).dispose();
    };
  }, [overlayMesh]);

  return null;
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
  const healerKinds = useApp(app, (s) => s.healerOverlays);
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
    const group = buildHealerOverlayGroup(analysis.analysis, positions, indices, kinds, healerKinds);
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
        disposeHealerOverlayGroup(group);
      }
    };
  }, [scene, item.mesh, bound, analysis, kinds, healerKinds]);
  return null;
}

/** The vertex under the pointer, into `view.hover` (`useHoverRay`), while Group mode hovers. */
function HoverRay(props: { scene: Scene | WorkerScene }) {
  useHoverRay(props.scene);
  return null;
}

/**
 * The sculpt brush on its mesh: the brush cursor (amber), which the stroke
 * driver (`useSculptStroke`) lays on the surface under the pointer while it
 * paints the stroke's dabs.
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
  useSculptStroke({ app, scene, sculpt, remesh, liveRemesh, maskTracker, ball });
  return null;
}

/**
 * The mask paint on the selected mesh: the paint ball (blue, against sculpt's
 * amber), which `useMaskPaint` lays on the surface under the pointer while it
 * paints the mask.
 */
function MaskLayer(props: { app: AppState; scene: Scene | WorkerScene; maskTracker: MaskLiveTracker }) {
  const { app, scene, maskTracker } = props;
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
  useMaskPaint({ app, scene, maskTracker, ball });
  return null;
}

/**
 * Trackball camera (free flipping in every direction, no poles); a
 * TransformControls gizmo on `bound`; and the camera pointed at a mesh on
 * each new `focus` request.
 */
function Controls(props: {
  app: AppState;
  items: readonly Item[];
  bound: MeshBinding | null;
  mode: GizmoMode;
  focus: FocusRequest | null;
  resetView: number;
  radiusOf: (mesh: MeshHandle) => number;
}) {
  const { app, items, bound, mode, focus, resetView, radiusOf } = props;
  const cameraProjection = useApp(app, (s) => s.cameraProjection);
  const { orbit, orbitTargetRef } = useOrbitCamera(cameraProjection);
  useGizmoDrag({ orbit, bound, mode });
  useCameraRequests({ app, items, orbit, orbitTargetRef, cameraProjection, focus, resetView, radiusOf });
  return null;
}
