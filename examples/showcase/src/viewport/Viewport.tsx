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
import { useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState } from "react";
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { TransformControls } from "three/addons/controls/TransformControls.js";
import type { WorkerScene } from "fast-mesh-wasm/worker";
import type { FocusRequest } from "../App";
import { inspect } from "../panels/inspect";
import type { Item } from "../scene/app-state";
import {
  type SculptBrush,
  SculptStroke,
  type StrokeSteps,
  GrabAnchor,
  grabLocalDelta,
  interpolateHit,
  worldPerPixel,
} from "./sculpt";
import { setDragging, setHover, view } from "./view";
import { setWireOverlay } from "./wire";

export type GizmoMode = "translate" | "rotate";

/** The hover raycast runs at most this often while the pointer moves. */
const HOVER_PERIOD_MS = 50;
/** The sculpt raycast runs at most this often while a stroke paints. */
const STROKE_PERIOD_MS = 16;
/** Hold-still buildup: a forced dab this often while the button stays down without moving. */
const BUILDUP_MS = 90;

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
  /** The sculpt brush armed on a mesh, if any. */
  sculpt: SculptBrush | null;
  /** A stroke's end remeshes its region. */
  sculptRemesh: boolean;
  /** A stroke is remeshed live while it paints. */
  sculptLiveRemesh: boolean;
  /** Lay the wireframe overlay over every shaded mesh. */
  wireframe: boolean;
}) {
  const {
    items,
    mode,
    focus,
    edgeSoftness,
    hoverable,
    scene,
    picking,
    onPickPoint,
    sculpt,
    sculptRemesh,
    sculptLiveRemesh,
    wireframe,
  } = props;
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
          sculpted={sculpt?.mesh === item.mesh}
          wireframe={wireframe}
          bound={bindings.get(item.mesh) ?? null}
        />
      ))}
      {/* Sculpt owns the pointer while armed: the gizmo stands down, so a press on its handles still paints. */}
      <Controls bound={sculpt !== null ? null : selected} mode={mode} focus={focus} radiusOf={radiusOf} />
      {hoverable ? <HoverRay scene={scene} /> : null}
      {sculpt !== null ? (
        <SculptLayer scene={scene} sculpt={sculpt} remesh={sculptRemesh} liveRemesh={sculptLiveRemesh} />
      ) : null}
      <PickCursor picking={picking !== null || sculpt !== null} />
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
  /** The sculpt brush is armed on this mesh: clicks paint, they never select or pick. */
  sculpted: boolean;
  /** Lay the wireframe overlay over the shaded solid. */
  wireframe: boolean;
  /** The mesh's binding, once `<FastMesh onBind>` delivers it. */
  bound: MeshBinding | null;
}) {
  const { item, selected, edgeSoftness, onPick, onBind, picking, onPickPoint, scene, sculpted, wireframe, bound } =
    props;
  useLayoutEffect(() => {
    item.material.emissive.set(selected ? 0x3a2a08 : 0x000000);
  }, [item.material, selected]);
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
      // The end of an orbit drag that started on the mesh is not a click.
      if (e.delta > 2) {
        return;
      }
      // A stroke's release on the mesh is not a click either.
      if (sculpted) {
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
    [item.mesh, onPick, picking, onPickPoint, scene, sculpted],
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
 * The sculpt brush on its mesh: hover shows the brush ball (a translucent
 * sphere of the brush radius around the surface point), press-and-hold
 * paints dabs along the pointer (and builds up where it holds still;
 * grab drags the surface along instead). A press on empty space is the
 * orbit's, as usual; a stroke holds a single undo step on the sync scene
 * (the Worker has no step nesting: one step per dab there), and its end
 * remeshes the stroked region when asked. One raycast at a time, through
 * the engine's BVH, never three's per-triangle `Mesh.raycast`.
 */
function SculptLayer(props: { scene: Scene | WorkerScene; sculpt: SculptBrush; remesh: boolean; liveRemesh: boolean }) {
  const { scene, sculpt, remesh, liveRemesh } = props;
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
  // The grab drag vector (local frame), refreshed per raycast while a grab
  // stroke paints; outside React, so moves never re-render.
  const dirRef = useRef<[number, number, number]>([0, 0, 0]);

  const ball = useMemo(() => {
    const group = new THREE.Group();
    group.add(
      new THREE.Mesh(
        new THREE.SphereGeometry(1, 40, 28),
        new THREE.MeshBasicMaterial({ color: 0xffd166, transparent: true, opacity: 0.18, depthWrite: false }),
      ),
    );
    group.add(
      new THREE.Mesh(new THREE.SphereGeometry(0.045, 16, 12), new THREE.MeshBasicMaterial({ color: 0xffd166 })),
    );
    group.visible = false;
    return group;
  }, []);
  useEffect(() => {
    return () => {
      ball.removeFromParent();
      for (const child of ball.children) {
        const mesh = child as THREE.Mesh;
        mesh.geometry.dispose();
        (mesh.material as THREE.Material).dispose();
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
    /** The screen point the last grab delta started from. */
    let grabFrom: { x: number; y: number } | null = null;
    /** The grab stroke's locked anchor (the press point), or null outside one. */
    let grabAnchor: GrabAnchor | null = null;
    /** The last live remesh: when, and how many dabs the stroke had. */
    let liveAt = 0;
    let liveDabs = 0;
    /** A live remesh in flight, if any (never two at once). */
    let liveRunning: Promise<unknown> | null = null;

    /**
     * Remeshes the stroked region live (at most every 600 ms and 4 dabs,
     * never overlapping): split-only, so the triangle count only grows and
     * the surface never pops down mid-stroke. The stroke-end pass evens the
     * density out fully.
     */
    const maybeLiveRemesh = () => {
      if (!liveRef.current || stroke === null || liveRunning !== null) {
        return;
      }
      if (stroke.dabs - liveDabs < 4) {
        return;
      }
      const now = performance.now();
      if (now - liveAt < 600) {
        return;
      }
      const b = stroke.bounds();
      if (b === null) {
        return;
      }
      liveAt = now;
      liveDabs = stroke.dabs;
      inspect.sculpt.liveRemeshes++;
      liveRunning = store
        .repair(brushRef.current.mesh, {
          op: "remesh",
          center: b.center,
          radius: b.radius,
          maxRounds: 2,
          splitOnly: true,
        })
        .catch(() => {
          // A degenerate region (or a mesh removed meanwhile): the stroke
          // paints on.
        })
        .finally(() => {
          liveRunning = null;
        });
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

    /** The hit over the mesh's local views: its point. */
    const localHit = (hit: { triangle: number; barycentric: readonly [number, number, number] }) => {
      const v = scene.views(brushRef.current.mesh);
      const indices = v.indices();
      const positions = v.positions();
      const corners = [indices[3 * hit.triangle], indices[3 * hit.triangle + 1], indices[3 * hit.triangle + 2]];
      return interpolateHit(positions, corners, hit.barycentric);
    };

    const hideBall = () => {
      hoverLocal = null;
      ball.visible = false;
      inspect.sculpt.hovering = false;
    };

    /** The ball around the surface point: a child of the mesh, in its local frame. */
    const showBall = (point: readonly [number, number, number]) => {
      const binding = view.bindings.get(brushRef.current.mesh);
      if (binding === undefined) {
        hideBall();
        return;
      }
      if (ball.parent !== binding.object) {
        binding.object.add(ball);
      }
      ball.visible = true;
      ball.position.set(point[0], point[1], point[2]);
      ball.scale.setScalar(Math.max(brushRef.current.radius, 1e-6));
    };

    const endStroke = () => {
      clearInterval(buildup);
      buildup = undefined;
      active = -1;
      grabFrom = null;
      grabAnchor = null;
      dirRef.current = [0, 0, 0];
      if (stroke !== null) {
        const s = stroke;
        syncDabs();
        stroke = null;
        inspect.sculpt.strokes++;
        // The stroke-end remesh lands before the step closes: one undo step
        // for the stroke and its evening-out (on the sync scene).
        if (remeshRef.current && landed > 0) {
          const b = s.bounds();
          if (b !== null) {
            inspect.sculpt.remeshes++;
            void store
              .repair(brushRef.current.mesh, { op: "remesh", center: b.center, radius: b.radius, maxRounds: 4 })
              .catch(() => {
                // A degenerate region (or a mesh removed meanwhile): the
                // stroke still ends.
              });
          }
        }
        s.end();
      }
      if (view.orbit !== null) {
        view.orbit.enabled = true;
      }
    };

    /** Refreshes the grab drag vector from the pointer's screen delta since the last dab. */
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
     * Paints one grab dab at the locked anchor (no raycast: the pointer's
     * hits may swim off the lump or onto another mesh, but the drag never
     * re-grabs). Only a stamped dab consumes its delta and advances the
     * anchor; a busy one retries it whole.
     */
    const paintGrab = (x: number, y: number) => {
      if (stroke === null || grabAnchor === null) {
        return;
      }
      const at = grabAnchor.center;
      trackGrab(x, y, at);
      if (stroke.dab(at, true) === "stamped") {
        const g = dirRef.current;
        inspect.sculpt.grabTotal += Math.sqrt(g[0] * g[0] + g[1] * g[1] + g[2] * g[2]);
        grabAnchor.advance(g);
        grabFrom = { x, y };
        // Past the advance (where the ball shows): consecutive centers then
        // differ by exactly the consumed delta the growth accounts.
        inspect.sculpt.lastDab = grabAnchor.center;
        showBall(grabAnchor.center);
      }
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
        last = performance.now();
        paintGrab(at.x, at.y);
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
      if (e.button !== 0 || active !== -1 || view.dragging) {
        return;
      }
      // Immediate: the stroke starts where the button lands, not a throttle later.
      Promise.resolve()
        .then(() => cast(e.clientX, e.clientY))
        .then(
          (hit) => {
            // The gizmo's own pointerdown runs first (its listeners attached
            // first): a grab there owns the gesture, not a stroke.
            if (!alive || active !== -1 || view.dragging) {
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
            landed = 0;
            liveDabs = 0;
            grabFrom = { x: e.clientX, y: e.clientY };
            grabAnchor = brush.op === "grab" ? new GrabAnchor(point) : null;
            dirRef.current = [0, 0, 0];
            const s = new SculptStroke(
              () => ({ ...brushRef.current, direction: dirRef.current }),
              steps,
              (op) =>
                store
                  .repair(mesh, op)
                  .then((r) => {
                    landed++;
                    return r;
                  })
                  .catch((err: unknown) => {
                    // The mesh went away mid-stroke: end it, and let the stroke count the failed dab.
                    endStroke();
                    throw err;
                  }),
            );
            active = e.pointerId;
            stroke = s;
            counted = 0;
            if (view.orbit !== null) {
              view.orbit.enabled = false;
            }
            // Grab is move-driven only: the press locks the anchor, the first
            // move paints (a click without a drag leaves no step, no remesh).
            if (brush.op !== "grab") {
              s.dab(point);
            }
            syncDabs();
            clearInterval(buildup);
            buildup = setInterval(() => {
              // Grab is move-driven only: holding still must not drift.
              if (brushRef.current.op === "grab") {
                return;
              }
              const at = hoverLocal;
              if (at !== null) {
                stroke?.dab(at, true);
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
      if (e.pointerId === active) {
        endStroke();
      }
    };
    const onLeave = (e: PointerEvent) => {
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
  }, [scene, camera, gl, store, ball]);

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
    view.orbit = o;
    setOrbit(o);
    return () => {
      o.dispose();
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
