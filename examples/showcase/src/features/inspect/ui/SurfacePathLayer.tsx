// The geodesic and path picks' viewport layer (#252): a ball on each source
// and each path point (the first one in its own colour), the path as a white
// line with a dark edge, just off the surface on both sides (and faintly
// through the mesh, so its hidden parts still show), its length on a label
// beside its end, and the geodesic map's iso-lines. Everything is in world
// coordinates, so it stays put while the camera moves. The taps are
// `useSurfacePickPointer`'s.
import type { SceneStore } from "fast-mesh-wasm/store";
import { useEffect, useLayoutEffect, useMemo } from "react";
import * as THREE from "three";
import { LineMaterial } from "three/addons/lines/LineMaterial.js";
import { LineSegments2 } from "three/addons/lines/LineSegments2.js";
import { LineSegmentsGeometry } from "three/addons/lines/LineSegmentsGeometry.js";
import { type AppState, useApp } from "../../../app/state/app-state";
import { drawOrigin } from "../../../shared/ux/draw-origin";
import { formatLength } from "../../../shared/ux/units";
import { pathLength, pathSegments, SURFACE_LINE_OFFSET } from "../ux/surface-paths";
import { useSurfacePickPointer } from "../ux/use-surface-pick";

const SOURCE_COLOR = 0xf472b6;
const PATH_COLOR = 0xffffff;
const FIRST_POINT_COLOR = 0xfacc15;
/** The dark edge of the path and its balls, and the iso-lines: they read on any colour of the map. */
const DARK = 0x0b0d10;
/** A ball's radius as a share of the mesh's, and the path's line and its dark edge in pixels. */
const BALL = 0.02;
const LINE_PX = 2.5;
const EDGE_PX = 5;

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
    const ball = (p: readonly number[], color: THREE.Material) => {
      for (const [material, scale, order] of [
        [parts.edgeMaterial, 1.45, 12],
        [color, 1, 13],
      ] as const) {
        const m = new THREE.Mesh(parts.ball, material);
        m.position.set(p[0], p[1], p[2]);
        m.scale.setScalar(r * BALL * scale);
        m.renderOrder = order;
        parts.markers.add(m);
      }
    };
    for (const s of tool.sources) {
      ball(s.point, parts.sourceMaterial);
    }
    const path = tool.path;
    path?.points.forEach((p, i) => ball(p.point, i === 0 ? parts.firstMaterial : parts.pathPointMaterial));

    const offset = r * SURFACE_LINE_OFFSET;
    const around = drawOrigin(path?.points[0]?.point ?? [0, 0, 0]);
    for (const line of [parts.pathEdge, parts.pathLine, parts.pathThrough]) {
      line.position.set(...around);
    }
    const above = pathSegments(path, offset, around);
    const below = pathSegments(path, -offset, around);
    const both = new Float32Array(above.length + below.length);
    both.set(above);
    both.set(below, above.length);
    setFat(parts.pathEdge, both);
    setFat(parts.pathLine, both);
    setFat(parts.pathThrough, pathSegments(path, 0, around));
    setLines(parts.iso, tool.isoLines ?? new Float32Array(0));

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

/** Replaces a line's segments (endpoint pairs), and shows it only while it has any. */
function setFat(line: LineSegments2, segments: Float32Array): void {
  line.visible = segments.length > 0;
  line.geometry.dispose(); // frees the old segments' buffers: the new ones replace them
  if (line.visible) {
    line.geometry.setPositions(segments);
  } else {
    line.geometry.instanceCount = 0;
  }
}

/** The same for plain one-pixel lines. */
function setLines(line: THREE.LineSegments, segments: Float32Array): void {
  line.visible = segments.length > 0;
  line.geometry.dispose();
  line.geometry.setAttribute("position", new THREE.BufferAttribute(segments, 3));
  line.geometry.setDrawRange(0, segments.length / 3);
}

/** The layer's three.js objects, made once and disposed with it. */
function buildParts() {
  const group = new THREE.Group();
  const markers = new THREE.Group();
  const ball = new THREE.SphereGeometry(1, 16, 12);
  const solid = (color: number) =>
    new THREE.MeshBasicMaterial({ color, depthTest: false, depthWrite: false, transparent: true, toneMapped: false });
  const sourceMaterial = solid(SOURCE_COLOR);
  const pathPointMaterial = solid(PATH_COLOR);
  const firstMaterial = solid(FIRST_POINT_COLOR);
  const edgeMaterial = solid(DARK);
  const fat = (color: number, px: number, order: number, through = false) => {
    const line = new LineSegments2(
      new LineSegmentsGeometry(),
      new LineMaterial({
        color,
        linewidth: px,
        toneMapped: false,
        depthWrite: false, // a segment's dark end cap must not hide its neighbours' white line
        ...(through ? { transparent: true, opacity: 0.35, depthTest: false } : {}),
      }),
    );
    line.renderOrder = order;
    line.frustumCulled = false;
    line.visible = false;
    return line;
  };
  const pathEdge = fat(DARK, EDGE_PX, 10);
  const pathLine = fat(PATH_COLOR, LINE_PX, 11);
  const pathThrough = fat(PATH_COLOR, LINE_PX, 11, true);
  const iso = new THREE.LineSegments(
    new THREE.BufferGeometry(),
    new THREE.LineBasicMaterial({ color: DARK, transparent: true, opacity: 0.85 }),
  );
  iso.renderOrder = 9;
  iso.frustumCulled = false;
  iso.visible = false;
  const label = makeLabel();
  group.add(iso, pathEdge, pathLine, pathThrough, markers, label.sprite);
  return {
    group,
    markers,
    ball,
    sourceMaterial,
    pathPointMaterial,
    firstMaterial,
    edgeMaterial,
    pathEdge,
    pathLine,
    pathThrough,
    iso,
    label,
    dispose: () => {
      group.removeFromParent();
      markers.clear();
      ball.dispose();
      for (const m of [sourceMaterial, pathPointMaterial, firstMaterial, edgeMaterial]) {
        m.dispose();
      }
      for (const line of [pathEdge, pathLine, pathThrough, iso]) {
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
  const sprite = new THREE.Sprite(
    new THREE.SpriteMaterial({ map: texture, depthTest: false, transparent: true, toneMapped: false }),
  );
  sprite.renderOrder = 14;
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
      ctx.fillStyle = "#ffffff";
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
