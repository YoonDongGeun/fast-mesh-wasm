// The inspect tools' 3D side (#110): the ruler line, the query marker,
// the draggable section plane and the clip box helper. Points are stored
// in world coordinates, so the drawings sit unparented in the scene and
// survive placement undo; the matrix only maps the picked local vertex
// into the world (scale-aware, #103).
import type { MeshHandle, Scene } from "fast-mesh-wasm";
import { useScene, useSelection } from "fast-mesh-wasm/react";
import type { WorkerScene } from "fast-mesh-wasm/worker";
import { useThree } from "@react-three/fiber";
import { useEffect, useLayoutEffect, useMemo, useRef } from "react";
import * as THREE from "three";
import type { AppState, Item } from "../scene/app-state";
import { type AppSnapshot, useApp } from "../scene/app-state";
import { formatDistance } from "../panels/collision-math";
import { boxFaces, localVertex, normalTravel, rulerDistance, snapSlot, worldFromMatrix } from "./inspect-tools";
import { setBoxClip, setSectionClip } from "./clip-planes";
import { dragPlaneNormal, planeHit } from "./slice-drag";
import { isExtraFinger } from "./touch";
import { setDragging, view } from "./view";

/** The engine hit's snapped vertex in world coordinates. */
export interface VertexPick {
  mesh: MeshHandle;
  /** The vertex slot, for the coloring value. */
  slot: number;
  /** The vertex in world coordinates. */
  point: [number, number, number];
}

/**
 * Raycasts the engine (its BVH, never three's per-triangle raycast) and
 * snaps to the hit triangle's nearest vertex, mapped into the world
 * through the mesh's placement.
 */
export async function pickVertex(
  scene: Scene | WorkerScene,
  origin: readonly [number, number, number],
  direction: readonly [number, number, number],
): Promise<VertexPick | null> {
  const hit = await scene.raycast(origin, direction);
  if (hit === null) {
    return null;
  }
  try {
    const views = scene.views(hit.mesh);
    const k = snapSlot(hit.barycentric);
    const slot = views.indices()[3 * hit.triangle + k];
    const point = worldFromMatrix(scene.matrix4(hit.mesh), localVertex(views.positions(), slot));
    return { mesh: hit.mesh, slot, point };
  } catch {
    return null; // removed meanwhile
  }
}

export function InspectLayers(props: { app: AppState; scene: Scene | WorkerScene; items: readonly Item[] }) {
  const { app, scene, items } = props;
  return (
    <>
      <RulerLayer app={app} items={items} />
      <QueryMarker app={app} items={items} />
      <SectionLayer app={app} scene={scene} items={items} />
      <ClipBoxLayer app={app} items={items} />
    </>
  );
}

function dotSize(items: readonly Item[]): number {
  const r = Math.max(0, ...items.map((i) => i.radius));
  return r > 0 ? r * 0.012 : 0.05;
}

/** The ruler: a line through its world endpoints, balls on them, and the distance on a sprite. */
function RulerLayer(props: { app: AppState; items: readonly Item[] }) {
  const { app, items } = props;
  const ruler = useAppOf(app, (s) => s.ruler);
  const group = useMemo(() => new THREE.Group(), []);
  const line = useMemo(() => {
    const g = new THREE.BufferGeometry();
    g.setAttribute("position", new THREE.BufferAttribute(new Float32Array(6), 3));
    return new THREE.Line(g, new THREE.LineBasicMaterial({ color: 0xffd166 }));
  }, []);
  const balls = useMemo(
    () => [
      new THREE.Mesh(new THREE.SphereGeometry(1, 16, 12), new THREE.MeshBasicMaterial({ color: 0xffd166 })),
      new THREE.Mesh(new THREE.SphereGeometry(1, 16, 12), new THREE.MeshBasicMaterial({ color: 0xffd166 })),
    ],
    [],
  );
  const label = useMemo(() => makeLabel(), []);
  useEffect(() => {
    group.add(line, ...balls, label.sprite);
    return () => {
      group.remove(line, ...balls, label.sprite);
    };
  }, [group, line, balls, label]);
  useEffect(
    () => () => {
      line.geometry.dispose();
      (line.material as THREE.Material).dispose();
      for (const b of balls) {
        b.geometry.dispose();
        (b.material as THREE.Material).dispose();
      }
      label.dispose();
    },
    [line, balls, label],
  );

  useLayoutEffect(() => {
    const size = dotSize(items);
    const pos = line.geometry.getAttribute("position") as THREE.BufferAttribute;
    if (ruler === null) {
      group.visible = false;
      return;
    }
    group.visible = true;
    const b = ruler.b ?? ruler.a;
    pos.setXYZ(0, ruler.a[0], ruler.a[1], ruler.a[2]);
    pos.setXYZ(1, b[0], b[1], b[2]);
    pos.needsUpdate = true;
    line.computeLineDistances?.();
    balls[0].position.set(ruler.a[0], ruler.a[1], ruler.a[2]);
    balls[0].scale.setScalar(size);
    balls[1].position.set(b[0], b[1], b[2]);
    balls[1].scale.setScalar(size);
    balls[1].visible = ruler.b !== null;
    if (ruler.b !== null) {
      const d = rulerDistance(ruler.a, ruler.b);
      label.set(
        formatDistance(d),
        [(ruler.a[0] + b[0]) / 2, (ruler.a[1] + b[1]) / 2, (ruler.a[2] + b[2]) / 2],
        Math.max(d * 0.12, size * 4),
      );
      label.sprite.visible = true;
    } else {
      label.sprite.visible = false;
    }
  }, [group, line, balls, label, ruler, items]);

  return <primitive object={group} />;
}

/** The query hit: a ball on the picked world point. The panel reads the values out. */
function QueryMarker(props: { app: AppState; items: readonly Item[] }) {
  const { app, items } = props;
  const query = useAppOf(app, (s) => s.inspectQuery);
  const ball = useMemo(
    () => new THREE.Mesh(new THREE.SphereGeometry(1, 20, 14), new THREE.MeshBasicMaterial({ color: 0x6ea8d9 })),
    [],
  );
  useEffect(
    () => () => {
      ball.removeFromParent();
      ball.geometry.dispose();
      (ball.material as THREE.Material).dispose();
    },
    [ball],
  );
  useLayoutEffect(() => {
    if (query === null) {
      ball.removeFromParent();
      return;
    }
    ball.position.set(query.point[0], query.point[1], query.point[2]);
    ball.scale.setScalar(dotSize(items) * 1.4);
  }, [ball, query, items]);
  if (query === null) {
    return null;
  }
  return <primitive object={ball} />;
}

/**
 * The section plane: a clipping plane on every mesh plus a draggable
 * gizmo (a translucent quad with a handle ball). Drags move the plane
 * along its normal and commit through the store (`setSectionPlane`);
 * engine commits are throttled, the drawing follows every move.
 */
function SectionLayer(props: { app: AppState; scene: Scene | WorkerScene; items: readonly Item[] }) {
  const { app, scene, items } = props;
  const store = useScene();
  const [selection] = useSelection();
  const section = useAppOf(app, (s) => s.inspectSection);
  const camera = useThree((s) => s.camera);
  const gl = useThree((s) => s.gl);
  const plane = useMemo(() => new THREE.Plane(), []);
  const gizmo = useMemo(() => buildSectionGizmo(), []);
  const drag = useRef<{
    pointerId: number;
    grab: [number, number, number];
    planePoint: [number, number, number];
    planeNormal: [number, number, number];
    origin0: [number, number, number];
    normal: [number, number, number];
    lastCommit: number;
  } | null>(null);

  // The three.js clipping plane follows the engine plane; every mesh keeps
  // its slice and box parts (see clip-planes).
  useLayoutEffect(() => {
    gl.localClippingEnabled = true;
    if (section !== null) {
      plane.setFromNormalAndCoplanarPoint(
        new THREE.Vector3(section.plane.normal[0], section.plane.normal[1], section.plane.normal[2]),
        new THREE.Vector3(section.plane.origin[0], section.plane.origin[1], section.plane.origin[2]),
      );
    }
    for (const item of items) {
      setSectionClip(item.material, section === null ? null : plane);
    }
    return () => {
      for (const item of items) {
        try {
          setSectionClip(item.material, null);
        } catch {
          // Removed meanwhile.
        }
      }
    };
  }, [gl, plane, section, items]);

  // The gizmo rides the plane: sized by the selection.
  useLayoutEffect(() => {
    const selected = items.find((i) => i.mesh === selection) ?? items[0] ?? null;
    const size = Math.max((selected?.radius ?? 1) * 2.4, 1e-3);
    gizmo.fit(size);
    if (section === null) {
      gizmo.group.visible = false;
      return;
    }
    gizmo.group.visible = true;
    gizmo.group.position.set(section.plane.origin[0], section.plane.origin[1], section.plane.origin[2]);
    gizmo.group.quaternion.setFromUnitVectors(
      new THREE.Vector3(0, 0, 1),
      new THREE.Vector3(section.plane.normal[0], section.plane.normal[1], section.plane.normal[2]).normalize(),
    );
  }, [gizmo, section, items, selection]);

  useEffect(() => {
    view.sectionHandle = gizmo.handle;
    return () => {
      view.sectionHandle = null;
    };
  }, [gizmo]);

  // Dragging the handle ball moves the plane along its normal.
  useEffect(() => {
    void scene;
    const canvas = gl.domElement;
    const caster = new THREE.Raycaster();
    const rayOf = (x: number, y: number): THREE.Ray => {
      const r = canvas.getBoundingClientRect();
      caster.setFromCamera(
        new THREE.Vector2(((x - r.left) / r.width) * 2 - 1, 1 - ((y - r.top) / r.height) * 2),
        camera,
      );
      return caster.ray;
    };
    const commit = (origin: [number, number, number], normal: [number, number, number], force: boolean) => {
      const snap = app.getSnapshot();
      const cur = snap.inspectSection;
      if (cur === null) {
        return;
      }
      const d = drag.current;
      const now = performance.now();
      if (!force && d !== null && now - d.lastCommit < 80) {
        return;
      }
      if (d !== null) {
        d.lastCommit = now;
      }
      try {
        store.setSection(cur.handle, { plane: { origin: [...origin], normal: [...normal] } });
      } catch {
        // Removed meanwhile: the gizmo cleans up.
      }
      app.setInspectSection({ handle: cur.handle, plane: { origin: [...origin], normal: [...normal] } });
    };
    const onDown = (e: PointerEvent) => {
      const snap = app.getSnapshot();
      const cur = snap.inspectSection;
      if (e.button !== 0 || isExtraFinger(e) || drag.current !== null || view.dragging || cur === null) {
        return;
      }
      if (snap.sculpt !== null || snap.brushPick !== null) {
        return;
      }
      const ray = rayOf(e.clientX, e.clientY);
      if (caster.intersectObject(gizmo.handle, false).length === 0) {
        return;
      }
      const normal = new THREE.Vector3(cur.plane.normal[0], cur.plane.normal[1], cur.plane.normal[2]);
      const seen = camera.getWorldDirection(new THREE.Vector3());
      const dp = dragPlaneNormal([normal.x, normal.y, normal.z], [seen.x, seen.y, seen.z]);
      if (dp === null) {
        return;
      }
      const center = gizmo.handle.getWorldPosition(new THREE.Vector3());
      const grab = planeHit(
        [ray.origin.x, ray.origin.y, ray.origin.z],
        [ray.direction.x, ray.direction.y, ray.direction.z],
        [center.x, center.y, center.z],
        dp,
      );
      if (grab === null) {
        return;
      }
      drag.current = {
        pointerId: e.pointerId,
        grab,
        planePoint: [center.x, center.y, center.z],
        planeNormal: dp,
        origin0: [...cur.plane.origin],
        normal: [cur.plane.normal[0], cur.plane.normal[1], cur.plane.normal[2]],
        lastCommit: 0,
      };
      try {
        canvas.setPointerCapture(e.pointerId);
      } catch {
        // Already released: the drag still tracks until the button lifts.
      }
      if (view.orbit !== null) {
        view.orbit.enabled = false;
      }
      setDragging(true);
      canvas.style.cursor = "grabbing";
    };
    const onMove = (e: PointerEvent) => {
      const d = drag.current;
      if (d === null) {
        if (e.buttons !== 0 || view.dragging) {
          return;
        }
        const cur = app.getSnapshot().inspectSection;
        if (cur === null || !gizmo.group.visible) {
          return;
        }
        rayOf(e.clientX, e.clientY);
        canvas.style.cursor = caster.intersectObject(gizmo.handle, false).length === 0 ? "" : "grab";
        return;
      }
      if (e.pointerId !== d.pointerId) {
        return;
      }
      const ray = rayOf(e.clientX, e.clientY);
      const now = planeHit(
        [ray.origin.x, ray.origin.y, ray.origin.z],
        [ray.direction.x, ray.direction.y, ray.direction.z],
        d.planePoint,
        d.planeNormal,
      );
      if (now === null) {
        return;
      }
      const t = normalTravel(d.grab, now, d.normal);
      commit(
        [d.origin0[0] + d.normal[0] * t, d.origin0[1] + d.normal[1] * t, d.origin0[2] + d.normal[2] * t],
        d.normal,
        false,
      );
    };
    const endDrag = (e: PointerEvent, commitLast: boolean) => {
      const d = drag.current;
      if (d === null || e.pointerId !== d.pointerId) {
        return;
      }
      if (commitLast) {
        const cur = app.getSnapshot().inspectSection;
        if (cur !== null) {
          commit(
            [...cur.plane.origin] as [number, number, number],
            [...cur.plane.normal] as [number, number, number],
            true,
          );
        }
      }
      drag.current = null;
      canvas.style.cursor = "";
      if (view.orbit !== null) {
        view.orbit.enabled = true;
      }
      setDragging(false);
    };
    const onUp = (e: PointerEvent) => endDrag(e, true);
    const onCancel = (e: PointerEvent) => endDrag(e, false);
    const onLeave = () => {
      if (drag.current === null) {
        canvas.style.cursor = "";
      }
    };
    canvas.addEventListener("pointerdown", onDown);
    canvas.addEventListener("pointermove", onMove);
    canvas.addEventListener("pointerup", onUp);
    canvas.addEventListener("pointercancel", onCancel);
    canvas.addEventListener("pointerleave", onLeave);
    return () => {
      canvas.removeEventListener("pointerdown", onDown);
      canvas.removeEventListener("pointermove", onMove);
      canvas.removeEventListener("pointerup", onUp);
      canvas.removeEventListener("pointercancel", onCancel);
      canvas.removeEventListener("pointerleave", onLeave);
      drag.current = null;
    };
  }, [app, store, camera, gl, gizmo]);

  useEffect(
    () => () => {
      gizmo.dispose();
    },
    [gizmo],
  );

  if (section === null) {
    return null;
  }
  return <primitive object={gizmo.group} />;
}

/** The gizmo's meshes: a translucent quad on the plane and the drag handle out along its normal. */
function buildSectionGizmo(): {
  group: THREE.Group;
  handle: THREE.Mesh;
  fit: (size: number) => void;
  dispose: () => void;
} {
  const group = new THREE.Group();
  const quad = new THREE.Mesh(
    new THREE.PlaneGeometry(1, 1),
    new THREE.MeshBasicMaterial({
      color: 0xffd166,
      transparent: true,
      opacity: 0.12,
      side: THREE.DoubleSide,
      depthWrite: false,
    }),
  );
  const border = new THREE.LineSegments(
    new THREE.EdgesGeometry(new THREE.PlaneGeometry(1, 1)),
    new THREE.LineBasicMaterial({ color: 0xffd166 }),
  );
  const handle = new THREE.Mesh(
    new THREE.SphereGeometry(1, 16, 12),
    new THREE.MeshBasicMaterial({ color: 0xffd166, depthTest: false, transparent: true }),
  );
  handle.renderOrder = 999;
  group.add(quad, border, handle);
  group.visible = false;
  return {
    group,
    handle,
    fit: (size: number) => {
      quad.scale.setScalar(size);
      border.scale.setScalar(size);
      handle.position.set(0, 0, size * 0.35);
      handle.scale.setScalar(Math.max(size * 0.045, 1e-6));
    },
    dispose: () => {
      group.removeFromParent();
      quad.geometry.dispose();
      (quad.material as THREE.Material).dispose();
      border.geometry.dispose();
      (border.material as THREE.Material).dispose();
      handle.geometry.dispose();
      (handle.material as THREE.Material).dispose();
    },
  };
}

/** The clip box: its faces clip every mesh (keeping slice and section parts) and a helper draws it. */
function ClipBoxLayer(props: { app: AppState; items: readonly Item[] }) {
  const { app, items } = props;
  const clip = useAppOf(app, (s) => s.clipBox);
  const gl = useThree((s) => s.gl);
  const box = useMemo(
    () =>
      clip === null
        ? null
        : new THREE.Box3(
            new THREE.Vector3(clip.min[0], clip.min[1], clip.min[2]),
            new THREE.Vector3(clip.max[0], clip.max[1], clip.max[2]),
          ),
    [clip],
  );
  const planes = useMemo(
    () =>
      clip === null
        ? null
        : boxFaces(clip.min, clip.max).map((f) => new THREE.Plane(new THREE.Vector3(...f.normal), f.constant)),
    [clip],
  );
  const helper = useMemo(
    () => (box === null ? null : new THREE.Box3Helper(box, 0xffd166)),
    [box],
  );
  useEffect(
    () => () => {
      helper?.dispose();
    },
    [helper],
  );

  useLayoutEffect(() => {
    gl.localClippingEnabled = true;
    for (const item of items) {
      setBoxClip(item.material, planes);
    }
    return () => {
      for (const item of items) {
        try {
          setBoxClip(item.material, null);
        } catch {
          // Removed meanwhile.
        }
      }
    };
  }, [gl, planes, items]);

  if (helper === null) {
    return null;
  }
  return <primitive object={helper} />;
}

// `useApp` without a selector loop: the layers read one slice each.
function useAppOf<T>(app: AppState, select: (s: AppSnapshot) => T): T {
  return useApp(app, select);
}

/** A text sprite for the ruler's distance. */
function makeLabel(): {
  sprite: THREE.Sprite;
  set: (text: string, at: readonly [number, number, number], height: number) => void;
  dispose: () => void;
} {
  const canvas = document.createElement("canvas");
  const texture = new THREE.CanvasTexture(canvas);
  const sprite = new THREE.Sprite(
    new THREE.SpriteMaterial({ map: texture, depthTest: false, transparent: true }),
  );
  sprite.renderOrder = 999;
  return {
    sprite,
    set: (text: string, at: readonly [number, number, number], height: number) => {
      const ctx = canvas.getContext("2d");
      if (ctx === null) {
        return;
      }
      const font = "600 44px system-ui, sans-serif";
      ctx.font = font;
      const w = Math.ceil(ctx.measureText(text).width) + 36;
      canvas.width = Math.max(w, 2);
      canvas.height = 72;
      ctx.font = font;
      ctx.textBaseline = "middle";
      ctx.fillStyle = "rgba(20, 22, 26, 0.85)";
      ctx.beginPath();
      ctx.roundRect(0, 0, canvas.width, canvas.height, 14);
      ctx.fill();
      ctx.fillStyle = "#ffd166";
      ctx.fillText(text, 18, canvas.height / 2 + 2);
      texture.needsUpdate = true;
      sprite.position.set(at[0], at[1], at[2]);
      sprite.scale.set((height * canvas.width) / canvas.height, height, 1);
    },
    dispose: () => {
      sprite.removeFromParent();
      texture.dispose();
      (sprite.material as THREE.Material).dispose();
    },
  };
}
