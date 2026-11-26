// The sculpt stroke driver: the pointer loop that hovers the brush cursor over
// the surface, presses a stroke onto the hit mesh and paints its dabs, grabs,
// live splits and stroke-end remesh through the engine.
import { useThree } from "@react-three/fiber";
import type {
  DyntopoDabReport,
  MeshHandle,
  RepairOp,
  Scene,
  SculptBrushOp,
  StretchReport,
  VertexProtection,
} from "fast-mesh-wasm";
import { useScene, useSelection } from "fast-mesh-wasm/react";
import { meshState } from "fast-mesh-wasm/store";
import type { WorkerScene } from "fast-mesh-wasm/worker";
import { useEffect, useRef } from "react";
import * as THREE from "three";
import { inspect } from "../../app/state/readouts";
import { type AppState, useApp } from "../../app/state/app-state";
import {
  buildDyntopoOptions,
  dabOperation,
  dyntopoCursorRatio,
  faceNormal,
  grabLocalDelta,
  interpolateHit,
  liveMaxSplits,
  liveRemeshDue,
  meanEdgeLength,
  remeshTooBig,
  type SculptBrush,
  type SculptMask,
  SculptStroke,
  strokeRegionSpheres,
  type StrokeSteps,
  symmetricRemeshOps,
  unionMaskIds,
  widestGrabReach,
  worldPerPixel,
} from "../../features/sculpt/ux/sculpt";
import {
  easeNormal,
  NORMAL_MARKER,
  placeBrushCursor,
  setBrushDirection,
  setBrushDyntopoIndicator,
} from "../../features/sculpt/ui/brush-cursor";
import {
  inheritedSplits,
  type MaskLiveTracker,
  maskMeetsRemesh,
  protectMaskedDab,
} from "../../features/sculpt/ux/mask-protect";
import { applyAltPinClick, scaleBrushRadiusForTarget, shouldAllowStrokeHit } from "./brush-target";
import { castVisibleSurface, HOVER_PERIOD_MS, STROKE_PERIOD_MS } from "./surface-raycast";
import { isExtraFinger } from "./touch";
import { view } from "./view";

/** Hold-still buildup: a forced dab this often while the button stays down without moving. */
const BUILDUP_MS = 16;
/** Holding this long builds up one full dab's strength. */
const BUILDUP_FULL_MS = 90;

/**
 * The sculpt brush on its mesh: hover shows the brush cursor (a translucent
 * sphere of the brush radius around the surface point) over any visible,
 * unlocked mesh (or only `selection` when pinned), press-and-hold locks onto
 * that hit mesh until pointerup and paints dabs along the pointer (and builds
 * up where it holds still; grab drags the surface along instead). A press on
 * empty space is the orbit's, as usual; a stroke holds a single undo step on
 * the sync scene (the Worker has no step nesting: one step per batch of dabs
 * there), and its end remeshes the stroked region when asked. Dabs hold the
 * painted mask's vertices in place and move the rest (`protectedDab`). One
 * raycast at a time, through the engine's BVH, never three's per-triangle
 * `Mesh.raycast`. `ball` is the brush cursor it places, owned by the caller.
 */
export function useSculptStroke(params: {
  app: AppState;
  scene: Scene | WorkerScene;
  sculpt: SculptBrush;
  remesh: boolean;
  liveRemesh: boolean;
  maskTracker: MaskLiveTracker;
  ball: THREE.Group;
}): void {
  const { app, scene, sculpt, remesh, liveRemesh, maskTracker, ball } = params;
  const store = useScene();
  const [selection, select] = useSelection();
  const camera = useThree((s) => s.camera);
  const gl = useThree((s) => s.gl);
  const items = useApp(app, (s) => s.items);
  const pinToActive = useApp(app, (s) => s.sculptPinToActive);
  // The brush inputs drive the armed brush live; the stroke survives them.
  const brushRef = useRef(sculpt);
  brushRef.current = sculpt;
  const remeshRef = useRef(remesh);
  remeshRef.current = remesh;
  const liveRef = useRef(liveRemesh);
  liveRef.current = liveRemesh;
  const dyntopo = useApp(app, (s) => s.sculptDyntopo);
  const dyntopoMode = useApp(app, (s) => s.sculptDyntopoMode);
  const dyntopoDetail = useApp(app, (s) => s.sculptDyntopoDetail);
  const dyntopoRef = useRef(dyntopo);
  dyntopoRef.current = dyntopo;
  const dyntopoModeRef = useRef(dyntopoMode);
  dyntopoModeRef.current = dyntopoMode;
  const dyntopoDetailRef = useRef(dyntopoDetail);
  dyntopoDetailRef.current = dyntopoDetail;
  const itemsRef = useRef(items);
  itemsRef.current = items;
  const meshRadiusOf = (mesh: MeshHandle): number =>
    itemsRef.current.find((i) => i.mesh === mesh)?.radius ?? 1;
  const pinRef = useRef(pinToActive);
  pinRef.current = pinToActive;
  const selectionRef = useRef(selection);
  selectionRef.current = selection;
  const selectRef = useRef(select);
  selectRef.current = select;
  // The mask reads live (paint can accumulate between strokes); the layer
  // never re-renders for it.
  const mask = useApp(app, (s) => s.sculptMask);
  const maskRef = useRef<SculptMask | null>(mask);
  maskRef.current = mask;
  // The grab drag vector (local frame), refreshed per raycast while a grab
  // stroke paints; outside React, so moves never re-render.
  const dirRef = useRef<[number, number, number]>([0, 0, 0]);

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
    /** Locked target mesh captured at pointerdown for the duration of the active stroke (#229). */
    let strokeTargetMesh: MeshHandle | null = null;
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
      const staleMesh = maskRef.current?.mesh;
      maskRef.current = null;
      maskTracker.invalidate(staleMesh);
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
      return protectMaskedDab(op, gated.vertices, gated.positions, gated.live);
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
      if (dyntopoRef.current || !liveRef.current || stroke === null) {
        return;
      }
      if (!liveRemeshDue(stroke.dabs - liveDabs, performance.now() - liveAt, liveRunning !== null)) {
        return;
      }
      const b = strokeBounds(stroke, grabPeakReach, grabPress);
      if (b === null) {
        return;
      }
      const mesh = strokeTargetMesh ?? brushRef.current.mesh;
      const spheres = strokeRegionSpheres(b, brushRef.current.symmetry);
      liveAt = performance.now();
      liveDabs = stroke.dabs;
      inspect.sculpt.liveRemeshes++;
      // In order with the stroke's edits, so that the stroke's end waits for it
      // and the engine's StrokeSnapshot interpolates any new split midpoints.
      // Dispatch each mirrored sphere sequentially so every repair stays a single
      // `splitStretched` op recognized by `MaskLiveTracker`.
      liveRunning = stroke.pass(async () => {
        try {
          for (const sphere of spheres) {
            await splitStretched(mesh, sphere);
          }
        } finally {
          liveRunning = null;
          liveAt = performance.now();
        }
      });
    };

    const isVisibleMesh = (mesh: MeshHandle): boolean => {
      if (!itemsRef.current.some((i) => i.mesh === mesh)) {
        return false;
      }
      return meshState(store.getSnapshot(), mesh).desired.visible;
    };

    const isEligibleMesh = (mesh: MeshHandle): boolean => {
      const item = itemsRef.current.find((i) => i.mesh === mesh);
      return item !== undefined && !item.locked && meshState(store.getSnapshot(), mesh).desired.visible;
    };

    const canTargetHit = (mesh: MeshHandle): boolean =>
      shouldAllowStrokeHit({
        hitMesh: mesh,
        strokeTargetMesh,
        pinToActive: pinRef.current,
        selection: selectionRef.current,
        fallbackMesh: brushRef.current.mesh,
        items: itemsRef.current,
        storeSnapshot: store.getSnapshot(),
      });

    /** Hover-retargets the sculpt brush to `nextMesh` without changing `selection` (#229). */
    const retargetHoverBrush = (nextMesh: MeshHandle): void => {
      if (brushRef.current.mesh === nextMesh) {
        return;
      }
      const fromItem = itemsRef.current.find((i) => i.mesh === brushRef.current.mesh);
      const toItem = itemsRef.current.find((i) => i.mesh === nextMesh);
      const nextRadius = scaleBrushRadiusForTarget(brushRef.current.radius, fromItem, toItem);
      const updated = { ...brushRef.current, mesh: nextMesh, radius: nextRadius };
      brushRef.current = updated;
      hoverLocal = null;
      cursorNormal = null;
      app.setSculpt(updated);
    };

    const cast = (x: number, y: number) => castVisibleSurface(scene, caster, canvas, camera, x, y, isVisibleMesh);

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
      hoverLocal = null;
      cursorNormal = null;
      ball.visible = false;
      inspect.sculpt.hovering = false;
    };

    /** The cursor on the surface point: a child of the mesh, in its local frame. */
    const showBall = (point: readonly [number, number, number], mesh: MeshHandle = strokeTargetMesh ?? brushRef.current.mesh) => {
      const binding = view.bindings.get(mesh);
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
      setBrushDyntopoIndicator(
        ball,
        dyntopoRef.current
          ? dyntopoCursorRatio(
              dyntopoModeRef.current,
              dyntopoDetailRef.current,
              meshRadiusOf(brush.mesh),
              brush.radius,
            )
          : null,
      );
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
      const mesh = strokeTargetMesh ?? brushRef.current.mesh;
      strokeTargetMesh = null;
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
        if (!dyntopoRef.current && liveRef.current && lastBounds !== null) {
          const spheres = strokeRegionSpheres(lastBounds, brushRef.current.symmetry);
          inspect.sculpt.liveRemeshes++;
          void s.pass(async () => {
            for (const sphere of spheres) {
              await splitStretched(mesh, sphere);
            }
          });
        }
        // The optional evening-out, off by default: it rewrites its whole
        // region, and it stands down on huge meshes, where it would stall the
        // worker.
        if (!dyntopoRef.current && remeshRef.current && landed > 0) {
          const b = lastBounds;
          const slots = triSlots(mesh);
          if (b !== null && slots !== null && !remeshTooBig(slots)) {
            ensureDensity(mesh);
            const targetLength = densityAnchor?.mesh === mesh ? densityAnchor.length : undefined;
            const ops = symmetricRemeshOps(
              b,
              brushRef.current.symmetry,
              (sp) => remeshMeetsMask(mesh, sp),
              4,
              targetLength,
            );
            if (ops.length > 0) {
              inspect.sculpt.remeshes++;
              // In order with the stroke: after what is due and any live pass, so
              // invalidation happens once the remesh pass executes (never pre-empted by
              // an earlier live split pass registering #splitPending).
              void s.pass(() => {
                maskTracker.noteTopologyMutation(mesh);
                return store
                  .repair(mesh, ops.length === 1 ? ops[0] : ops)
                  .then((r) => {
                    maskTracker.noteTopologyMutation(mesh);
                    return r;
                  })
                  .catch(() => {
                    maskTracker.noteTopologyMutation(mesh);
                    // A degenerate region (or a mesh removed meanwhile): the
                    // stroke still ends.
                  });
              });
            }
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
      const activeMesh = strokeTargetMesh ?? brushRef.current.mesh;
      const binding = view.bindings.get(activeMesh);
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
      const scale = binding.object.getWorldScale(new THREE.Vector3());
      dirRef.current = grabLocalDelta(
        x - from.x,
        y - from.y,
        worldPerPixel(distance, fov, height),
        [e[0], e[1], e[2]],
        [e[4], e[5], e[6]],
        [q.x, q.y, q.z, q.w],
        [scale.x, scale.y, scale.z],
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
      const mesh = strokeTargetMesh ?? brush.mesh;
      s.move(
        () => {
          const cost = inspect.sculpt.dabCost;
          const start = performance.now();
          const rawOp = dabOperation({ ...brushRef.current, mesh, tool: "grab", direction: g }, press);
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
            if (hit === null || !canTargetHit(hit.mesh)) {
              hideBall();
              stroke?.lift();
            } else {
              if (stroke === null && hit.mesh !== brushRef.current.mesh) {
                retargetHoverBrush(hit.mesh);
              }
              const hitMesh = strokeTargetMesh ?? hit.mesh;
              let point: [number, number, number];
              try {
                point = localHit(hitMesh, hit);
              } catch {
                // Removed meanwhile: the next move asks again.
                run();
                return;
              }
              hoverLocal = point;
              inspect.sculpt.hovering = true;
              showBall(point, hitMesh);
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
      const altPressed = e.altKey;
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
            if (hit === null) {
              return; // empty space: the orbit keeps the gesture.
            }
            if (altPressed) {
              if (!isEligibleMesh(hit.mesh)) {
                return;
              }
              view.pressTaken = true;
              const nowPinned = applyAltPinClick({
                app,
                select: selectRef.current,
                selection: selectionRef.current,
                hitMesh: hit.mesh,
                pinToActive: pinRef.current,
              });
              if (nowPinned) {
                retargetHoverBrush(hit.mesh);
              }
              return;
            }
            if (!canTargetHit(hit.mesh)) {
              return;
            }
            if (hit.mesh !== brushRef.current.mesh) {
              retargetHoverBrush(hit.mesh);
            }
            strokeTargetMesh = hit.mesh;
            const brush = brushRef.current;
            let point: [number, number, number];
            try {
              point = localHit(hit.mesh, hit);
            } catch {
              strokeTargetMesh = null;
              return;
            }
            hoverLocal = point;
            inspect.sculpt.hovering = true;
            showBall(point, hit.mesh);
            // Step nesting is the sync scene's (or any Dyntopo stroke); non-dyntopo Worker steps per dab.
            const useSteps = !("status" in scene) || dyntopoRef.current;
            const steps: StrokeSteps = {
              beginStep: useSteps ? () => store.beginStep() : undefined,
              endStep: useSteps ? () => store.endStep() : undefined,
              cancelStep: useSteps ? () => store.cancelStep() : undefined,
            };
            const mesh = strokeTargetMesh;
            ensureRest(mesh);
            landed = 0;
            liveDabs = 0;
            grabFrom = { x: e.clientX, y: e.clientY };
            grabPress = brush.op === "grab" ? point : null;
            grabPeakReach = 0;
            dirRef.current = [0, 0, 0];
            // Sync `Scene` sends one dab per edit under nested steps; `WorkerScene`
            // (which exposes `.status`) always batches multiple dabs in a single RPC.
            const s = new SculptStroke(
              () => ({ ...brushRef.current, mesh, direction: dirRef.current }),
              steps,
              (ops) => {
                const cost = inspect.sculpt.dabCost;
                const start = performance.now();
                const held = ops.map((op) => protectedDab(mesh, op));
                const sent = performance.now();
                if (cost.maskMs.length < 4096) {
                  cost.maskMs.push(sent - start);
                }
                const dispatch = async (): Promise<
                  readonly { moved?: number; removedVertices?: Uint32Array }[]
                > => {
                  if (dyntopoRef.current) {
                    const opts = buildDyntopoOptions(
                      dyntopoModeRef.current,
                      dyntopoDetailRef.current,
                      meshRadiusOf(mesh),
                    );
                    const reports: DyntopoDabReport[] = [];
                    for (const op of held) {
                      const protect = "protect" in op ? (op as { protect?: VertexProtection }).protect : undefined;
                      reports.push(await scene.dyntopoDab(mesh, op as SculptBrushOp, opts, protect));
                    }
                    return reports;
                  }
                  return (!("status" in scene) || held.length === 1
                    ? Promise.all(held.map((op) => store.repair(mesh, op)))
                    : store.repair(mesh, held)) as Promise<readonly { moved?: number }[]>;
                };
                return landing(dispatch()).then((reports) => {
                  if (cost.landMs.length < 4096) {
                    cost.landMs.push(performance.now() - sent);
                  }
                  let anyRemoved = false;
                  (reports as readonly { moved?: number; removedVertices?: Uint32Array }[]).forEach((r, k) => {
                    if ((r.removedVertices?.length ?? 0) > 0) {
                      anyRemoved = true;
                    }
                    if (r.moved === 0 && "protect" in held[k]) {
                      inspect.sculpt.maskedDabs++;
                      // Nothing moved, so nothing stretched: no live pass for this one.
                      liveDabs++;
                    }
                  });
                  if (anyRemoved) {
                    checkMaskLive(mesh);
                  }
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
}
