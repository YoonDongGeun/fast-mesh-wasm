// The geodesic and path picks' viewport layer (#252): a ball on each source
// and each path point (the first one in its own colour), the path drawn
// just off the surface on both sides (and faintly through the mesh, so its
// hidden parts still show), its length on a label beside its end, and the
// geodesic map's iso-lines. Everything is in world coordinates, so it stays
// put while the camera moves. The taps are `useSurfacePickPointer`'s.
import type { SceneStore } from "fast-mesh-wasm/store";
import { useEffect, useLayoutEffect, useMemo } from "react";
import * as THREE from "three";
import { type AppState, useApp } from "../../../app/state/app-state";
import { formatLength } from "../../../shared/ux/units";
import { pathLength, pathSegments, SURFACE_LINE_OFFSET } from "../ux/surface-paths";
import { useSurfacePickPointer } from "../ux/use-surface-pick";

const SOURCE_COLOR = 0xf472b6;
const PATH_COLOR = 0x5eead4;
const FIRST_POINT_COLOR = 0xfacc15;
const ISO_COLOR = 0x0b0d10;

export function SurfacePathLayer(props: { app: AppState; store: SceneStore }) {
  const { app, store } = props;
  useSurfacePickPointer(app, store);
  const tool = useApp(app, (s) => s.surfacePath);
  const items = useApp(app, (s) => s.items);
  const units = useApp(app, (s) => s.units);
  const parts = useMemo(() => buildParts(), []);
  useEffect(() => () => parts.dispose(), [parts]);

  useLayoutEffect(() => {
    const radius = items.find((i) => i.mesh === tool.mesh)?.radius ?? 1;
    const r = radius > 0 ? radius : 1;
    parts.markers.clear();
    const ball = (p: readonly number[], material: THREE.Material) => {
      const m = new THREE.Mesh(parts.ball, material);
      m.position.set(p[0], p[1], p[2]);
      m.scale.setScalar(r * 0.015);
      m.renderOrder = 12;
      parts.markers.add(m);
    };
    for (const s of tool.sources) {
      ball(s.point, parts.sourceMaterial);
    }
    const path = tool.path;
    path?.points.forEach((p, i) => ball(p.point, i === 0 ? parts.firstMaterial : parts.pathPointMaterial));

    const offset = r * SURFACE_LINE_OFFSET;
    const above = pathSegments(path, offset);
    const below = pathSegments(path, -offset);
    const both = new Float32Array(above.length + below.length);
    both.set(above);
    both.set(below, above.length);
    setPositions(parts.pathLine.geometry, both);
    setPositions(parts.pathThrough.geometry, pathSegments(path, 0));
    setPositions(parts.iso.geometry, tool.isoLines ?? new Float32Array(0));

    if (path !== null && path.legs.length > 0) {
      const at = path.closed ? path.points[0].point : path.points[path.points.length - 1].point;
      parts.label.set(formatLength(pathLength(path), units), at, r * 0.07);
      parts.label.sprite.visible = true;
    } else {
      parts.label.sprite.visible = false;
    }
  }, [parts, tool.mesh, tool.sources, tool.path, tool.isoLines, items, units]);

  return <primitive object={parts.group} />;
}

/** Replaces a line geometry's positions (segment endpoint pairs). */
function setPositions(geometry: THREE.BufferGeometry, positions: Float32Array): void {
  geometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));
  geometry.setDrawRange(0, positions.length / 3);
  geometry.computeBoundingSphere();
}

/** The layer's three.js objects, made once and disposed with it. */
function buildParts() {
  const group = new THREE.Group();
  const markers = new THREE.Group();
  const ball = new THREE.SphereGeometry(1, 16, 12);
  const marker = (color: number) =>
    new THREE.MeshBasicMaterial({ color, depthTest: false, depthWrite: false, transparent: true });
  const sourceMaterial = marker(SOURCE_COLOR);
  const pathPointMaterial = marker(PATH_COLOR);
  const firstMaterial = marker(FIRST_POINT_COLOR);
  const pathLine = new THREE.LineSegments(new THREE.BufferGeometry(), new THREE.LineBasicMaterial({ color: PATH_COLOR }));
  pathLine.renderOrder = 10;
  pathLine.frustumCulled = false;
  const pathThrough = new THREE.LineSegments(
    new THREE.BufferGeometry(),
    new THREE.LineBasicMaterial({ color: PATH_COLOR, transparent: true, opacity: 0.3, depthTest: false, depthWrite: false }),
  );
  pathThrough.renderOrder = 11;
  pathThrough.frustumCulled = false;
  const iso = new THREE.LineSegments(
    new THREE.BufferGeometry(),
    new THREE.LineBasicMaterial({ color: ISO_COLOR, transparent: true, opacity: 0.85 }),
  );
  iso.renderOrder = 9;
  iso.frustumCulled = false;
  const label = makeLabel();
  group.add(iso, pathLine, pathThrough, markers, label.sprite);
  return {
    group,
    markers,
    ball,
    sourceMaterial,
    pathPointMaterial,
    firstMaterial,
    pathLine,
    pathThrough,
    iso,
    label,
    dispose: () => {
      group.removeFromParent();
      markers.clear();
      ball.dispose();
      for (const m of [sourceMaterial, pathPointMaterial, firstMaterial]) {
        m.dispose();
      }
      for (const line of [pathLine, pathThrough, iso]) {
        line.geometry.dispose();
        (line.material as THREE.Material).dispose();
      }
      label.dispose();
    },
  };
}

/** A text sprite for the path's length. */
function makeLabel(): {
  sprite: THREE.Sprite;
  set: (text: string, at: readonly [number, number, number], height: number) => void;
  dispose: () => void;
} {
  const canvas = document.createElement("canvas");
  const texture = new THREE.CanvasTexture(canvas);
  const sprite = new THREE.Sprite(new THREE.SpriteMaterial({ map: texture, depthTest: false, transparent: true }));
  sprite.renderOrder = 13;
  sprite.visible = false;
  // Anchored by its lower left corner: beside the point, whichever way is up.
  sprite.center.set(-0.15, -0.3);
  return {
    sprite,
    set: (text, at, height) => {
      const ctx = canvas.getContext("2d");
      if (ctx === null) {
        return;
      }
      const font = "600 44px system-ui, sans-serif";
      ctx.font = font;
      canvas.width = Math.max(Math.ceil(ctx.measureText(text).width) + 36, 2);
      canvas.height = 72;
      ctx.font = font;
      ctx.textBaseline = "middle";
      ctx.fillStyle = "rgba(20, 22, 26, 0.85)";
      ctx.beginPath();
      ctx.roundRect(0, 0, canvas.width, canvas.height, 14);
      ctx.fill();
      ctx.fillStyle = "#5eead4";
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
