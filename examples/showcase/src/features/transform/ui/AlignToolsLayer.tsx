// The align tools' viewport layer (#253): a numbered marker on each landmark,
// in one colour on the moving mesh and another on the fixed one, and the
// symmetry plane as a translucent quad over its mesh's bounds. Both follow
// their meshes as they move (gizmo drags included). The landmark taps are
// `useLandmarkPickPointer`'s.
import { useFrame } from "@react-three/fiber";
import type { MeshHandle, Vec3 } from "fast-mesh-wasm";
import type { SceneStore } from "fast-mesh-wasm/store";
import { useEffect, useLayoutEffect, useMemo } from "react";
import * as THREE from "three";
import { type AppState, type Item, useApp } from "../../../app/state/app-state";
import { view } from "../../../viewport/ux/view";
import { computeLocalExtents } from "../ux/transform-math";
import { type LandmarkSet, planeQuad, quadCorners, type SymmetryPlaneShown } from "../ux/align-tools";
import { useLandmarkPickPointer } from "../ux/use-landmark-picks";

const MOVING_COLOR = 0xf59e0b;
const FIXED_COLOR = 0x38bdf8;
const PLANE_COLOR = 0x22d3ee;
const NOOP_RAYCAST = () => {};

/** Copies the mesh's current world matrix (the drawn one while it is bound) into `out`; false when it is hidden or gone. */
function meshMatrix(store: SceneStore, mesh: MeshHandle, out: THREE.Matrix4): boolean {
  if (store.getSnapshot().meshes.get(mesh)?.desired.visible === false) {
    return false;
  }
  const bound = view.bindings.get(mesh)?.object;
  if (bound !== undefined) {
    out.copy(bound.matrixWorld);
    return true;
  }
  try {
    out.fromArray(store.scene.matrix4(mesh));
    return true;
  } catch {
    return false;
  }
}

/** A disc with the number `n` on it, for a marker's label; `null` without a 2D canvas. */
function labelTexture(n: number, color: number): THREE.CanvasTexture | null {
  const canvas = document.createElement("canvas");
  canvas.width = 64;
  canvas.height = 64;
  const ctx = canvas.getContext("2d");
  if (ctx === null) {
    return null;
  }
  ctx.fillStyle = `#${color.toString(16).padStart(6, "0")}`;
  ctx.beginPath();
  ctx.arc(32, 32, 30, 0, Math.PI * 2);
  ctx.fill();
  ctx.fillStyle = "#0b0d10";
  ctx.font = "bold 34px system-ui, sans-serif";
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  ctx.fillText(String(n), 32, 34);
  const texture = new THREE.CanvasTexture(canvas);
  texture.colorSpace = THREE.SRGBColorSpace;
  return texture;
}

export function AlignToolsLayer(props: { app: AppState; store: SceneStore }) {
  const { app, store } = props;
  useLandmarkPickPointer(app, store);
  const landmarks = useApp(app, (s) => s.alignTool.landmarks);
  const symmetry = useApp(app, (s) => s.alignTool.symmetry);
  const items = useApp(app, (s) => s.items);
  return (
    <>
      {landmarks !== null ? <LandmarkMarkers set={landmarks} items={items} store={store} /> : null}
      {symmetry !== null ? <SymmetryPlaneQuad plane={symmetry} store={store} /> : null}
    </>
  );
}

/** One side's markers: each point's local position and the group of its ball and label. */
interface SideMarkers {
  mesh: MeshHandle;
  points: readonly Vec3[];
  objects: THREE.Object3D[];
}

function LandmarkMarkers(props: { set: LandmarkSet; items: readonly Item[]; store: SceneStore }) {
  const { set, items, store } = props;
  const group = useMemo(() => new THREE.Group(), []);
  const ball = useMemo(() => new THREE.SphereGeometry(1, 16, 12), []);
  const balls = useMemo(
    () =>
      [MOVING_COLOR, FIXED_COLOR].map((color) => new THREE.MeshBasicMaterial({ color, depthTest: false, depthWrite: false })),
    [],
  );
  // Label materials by side and number, made as they are first needed.
  const labels = useMemo(() => new Map<string, THREE.SpriteMaterial | null>(), []);
  useEffect(
    () => () => {
      ball.dispose();
      for (const m of balls) m.dispose();
      for (const m of labels.values()) {
        m?.map?.dispose();
        m?.dispose();
      }
    },
    [ball, balls, labels],
  );

  const sides = useMemo<SideMarkers[]>(() => {
    const label = (side: 0 | 1, n: number): THREE.SpriteMaterial | null => {
      const key = `${side}:${n}`;
      if (!labels.has(key)) {
        const map = labelTexture(n, side === 0 ? MOVING_COLOR : FIXED_COLOR);
        labels.set(key, map === null ? null : new THREE.SpriteMaterial({ map, depthTest: false, depthWrite: false }));
      }
      return labels.get(key) ?? null;
    };
    const radius = Math.max(
      items.find((i) => i.mesh === set.moving)?.radius ?? 1,
      items.find((i) => i.mesh === set.fixed)?.radius ?? 1,
    );
    const size = radius > 0 ? radius * 0.015 : 0.05;
    const make = (side: 0 | 1, points: readonly Vec3[]): THREE.Object3D[] =>
      points.map((_, i) => {
        const marker = new THREE.Group();
        const dot = new THREE.Mesh(ball, balls[side]);
        dot.scale.setScalar(size);
        dot.renderOrder = 12;
        dot.raycast = NOOP_RAYCAST;
        marker.add(dot);
        const material = label(side, i + 1);
        if (material !== null) {
          const sprite = new THREE.Sprite(material);
          // Up and to the right of the point, so the point itself stays in view.
          sprite.center.set(-0.15, -0.15);
          sprite.scale.setScalar(size * 6);
          sprite.renderOrder = 13;
          sprite.raycast = NOOP_RAYCAST;
          marker.add(sprite);
        }
        return marker;
      });
    return [
      { mesh: set.moving, points: set.movingPoints, objects: make(0, set.movingPoints) },
      { mesh: set.fixed, points: set.fixedPoints, objects: make(1, set.fixedPoints) },
    ];
  }, [set, items, ball, balls, labels]);

  const matrix = useMemo(() => new THREE.Matrix4(), []);
  const sync = () => {
    for (const side of sides) {
      const placed = meshMatrix(store, side.mesh, matrix);
      side.objects.forEach((object, i) => {
        const p = side.points[i];
        object.visible = placed;
        object.position.set(p[0], p[1], p[2]).applyMatrix4(matrix);
      });
    }
  };

  useLayoutEffect(() => {
    group.clear();
    for (const side of sides) {
      for (const object of side.objects) group.add(object);
    }
    sync();
  });
  useFrame(sync);

  return <primitive object={group} />;
}

function SymmetryPlaneQuad(props: { plane: SymmetryPlaneShown; store: SceneStore }) {
  const { plane, store } = props;
  const group = useMemo(() => {
    const g = new THREE.Group();
    g.matrixAutoUpdate = false;
    return g;
  }, []);
  const fillMaterial = useMemo(
    () =>
      new THREE.MeshBasicMaterial({
        color: PLANE_COLOR,
        transparent: true,
        opacity: 0.22,
        side: THREE.DoubleSide,
        depthWrite: false,
      }),
    [],
  );
  const edgeMaterial = useMemo(
    () => new THREE.LineBasicMaterial({ color: PLANE_COLOR, transparent: true, opacity: 0.9 }),
    [],
  );

  // The quad in the mesh's local frame, over its local bounds.
  const { fill, edge } = useMemo(() => {
    let corners: Vec3[] = [];
    try {
      const views = store.scene.views(plane.mesh);
      const { min, max } = computeLocalExtents(views.positions(), views.indices());
      corners = quadCorners(planeQuad(plane.point, plane.normal, min, max));
    } catch {
      // Removed meanwhile: nothing to draw.
    }
    const flat = new Float32Array(corners.flat());
    const fill = new THREE.BufferGeometry();
    fill.setAttribute("position", new THREE.BufferAttribute(flat, 3));
    if (corners.length === 4) {
      fill.setIndex([0, 1, 2, 0, 2, 3]);
    }
    const edge = new THREE.BufferGeometry();
    edge.setAttribute("position", new THREE.BufferAttribute(flat.slice(), 3));
    return { fill, edge };
  }, [plane, store]);

  useEffect(() => () => {
    fill.dispose();
    edge.dispose();
  }, [fill, edge]);
  useEffect(
    () => () => {
      fillMaterial.dispose();
      edgeMaterial.dispose();
    },
    [fillMaterial, edgeMaterial],
  );

  useLayoutEffect(() => {
    group.clear();
    const quad = new THREE.Mesh(fill, fillMaterial);
    quad.renderOrder = 11;
    quad.raycast = NOOP_RAYCAST;
    const outline = new THREE.LineLoop(edge, edgeMaterial);
    outline.renderOrder = 11;
    outline.raycast = NOOP_RAYCAST;
    group.add(quad, outline);
  }, [group, fill, edge, fillMaterial, edgeMaterial]);

  const sync = () => {
    group.visible = meshMatrix(store, plane.mesh, group.matrix);
    group.matrixWorldNeedsUpdate = true;
  };
  useLayoutEffect(sync);
  useFrame(sync);

  return <primitive object={group} />;
}
