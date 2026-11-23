// The object info overlays in the viewport (#257): the angle measure (its
// arms, arc, points and label, in world coordinates like the ruler), and on
// the selected mesh the bounding box (world- or local-aligned, with its
// dimensions) and the normals as short lines. The local box and the normals
// are children of the mesh's bound object, built in its local coordinates
// once per geometry change, so they follow gizmo moves and placement undo
// with no work per frame. The world box keeps its box through the matrix's
// linear part (`WorldBoxTracker`): a translate drag shifts it, only a
// rotation walks the vertices again.
import type { MeshHandle, Scene } from "fast-mesh-wasm";
import { useScene, useSelection } from "fast-mesh-wasm/react";
import type { WorkerScene } from "fast-mesh-wasm/worker";
import { useFrame } from "@react-three/fiber";
import { useEffect, useLayoutEffect, useMemo, useState } from "react";
import * as THREE from "three";
import { type AppState, type Item, useApp } from "../../../app/state/app-state";
import { formatLength, type LengthUnit } from "../../../shared/ux/units";
import { view } from "../../../viewport/ux/view";
import { angleArc, angleDegrees, angleLabelAnchor, arcRadius, formatAngle } from "../ux/angle-measure";
import { type Box, boxSize, linearBox, offsetBox, type InfoSpace, WorldBoxTracker } from "../ux/object-info";
import type { NormalsMode } from "../ux/object-info-state";
import { boxEdges, boxLabelAnchors, faceNormalLines, usedVertices, vertexNormalLines } from "../ux/overlay-geometry";
import { objectInfoProbe, watchMesh } from "../ux/use-object-info";
import { makeLabelSprite } from "./label-sprite";

const ANGLE_COLOR = 0xf78fb3;
const BOX_COLOR = 0x7fd4ff;
const FACE_NORMAL_COLOR = 0x9be564;
const VERTEX_NORMAL_COLOR = 0xffa94d;

export function ObjectInfoLayers(props: { app: AppState; scene: Scene | WorkerScene; items: readonly Item[] }) {
  const { app, scene, items } = props;
  const [selection] = useSelection();
  const item = items.find((i) => i.mesh === selection) ?? null;
  const showBox = useApp(app, (s) => s.objectInfo.showBox);
  const space = useApp(app, (s) => s.objectInfo.space);
  const normals = useApp(app, (s) => s.objectInfo.normals);
  const normalsScale = useApp(app, (s) => s.objectInfo.normalsScale);
  const units = useApp(app, (s) => s.units);
  return (
    <>
      <AngleLayer app={app} items={items} />
      {item !== null && showBox ? <BoxOverlay key={`box-${item.id}`} scene={scene} item={item} space={space} unit={units} /> : null}
      {item !== null && normals !== "off" ? (
        <NormalsOverlay key={`normals-${item.id}`} scene={scene} item={item} mode={normals} length={normalsScale * item.radius} />
      ) : null}
    </>
  );
}

/** Bumps once per settled geometry change of `mesh` (edits, undo, redo): what the overlays rebuild on. */
function useGeometryVersion(mesh: MeshHandle): number {
  const store = useScene();
  const [version, setVersion] = useState(0);
  // In a layout effect: no change slips between the render that read the buffers and the watch.
  useLayoutEffect(
    () =>
      watchMesh(store.scene, mesh, {
        onSettled: (kind) => {
          if (kind !== "placement") {
            setVersion((v) => v + 1);
          }
        },
      }),
    [store, mesh],
  );
  return version;
}

/** Keeps `child` under the mesh's bound object (so it follows the gizmo and hides with the mesh), checked each frame. */
function useAttachToMesh(child: THREE.Object3D, mesh: MeshHandle): void {
  useFrame(() => {
    const parent = view.bindings.get(mesh)?.object ?? null;
    if (child.parent !== parent) {
      child.removeFromParent();
      parent?.add(child);
    }
  });
  useEffect(
    () => () => {
      child.removeFromParent();
    },
    [child],
  );
}

/**
 * A line segments object drawn in its parent's coordinates. It is never a
 * click target: the events raycast a mesh's children too, and a line catches
 * rays a whole unit away.
 */
function makeLines(color: number): THREE.LineSegments {
  const lines = new THREE.LineSegments(new THREE.BufferGeometry(), new THREE.LineBasicMaterial({ color }));
  lines.matrixAutoUpdate = false;
  lines.frustumCulled = false;
  lines.raycast = () => {};
  return lines;
}

function setPositions(lines: THREE.LineSegments | THREE.Line, data: Float32Array): void {
  const geometry = lines.geometry;
  const attr = geometry.getAttribute("position") as THREE.BufferAttribute | undefined;
  if (attr !== undefined && attr.array.length === data.length) {
    (attr.array as Float32Array).set(data);
    attr.needsUpdate = true;
  } else {
    geometry.setAttribute("position", new THREE.BufferAttribute(data, 3));
  }
  geometry.setDrawRange(0, data.length / 3);
}

/** Frees the GPU side only: StrictMode runs cleanups once and keeps the objects, which three uploads again. */
function disposeObject(o: THREE.Mesh | THREE.Line | THREE.LineSegments): void {
  o.geometry.dispose();
  (o.material as THREE.Material).dispose();
}

/** The angle measure: the arms through the vertex, the arc, the picked points, and the angle on a label. */
function AngleLayer(props: { app: AppState; items: readonly Item[] }) {
  const { app, items } = props;
  const points = useApp(app, (s) => s.objectInfo.angle);
  const parts = useMemo(() => {
    const group = new THREE.Group();
    const arms = new THREE.Line(new THREE.BufferGeometry(), new THREE.LineBasicMaterial({ color: ANGLE_COLOR }));
    const arc = new THREE.Line(new THREE.BufferGeometry(), new THREE.LineBasicMaterial({ color: ANGLE_COLOR }));
    arms.frustumCulled = false;
    arc.frustumCulled = false;
    const balls = [0, 1, 2].map(
      () => new THREE.Mesh(new THREE.SphereGeometry(1, 16, 12), new THREE.MeshBasicMaterial({ color: ANGLE_COLOR })),
    );
    const label = makeLabelSprite("#f78fb3");
    group.add(arms, arc, ...balls, label.sprite);
    return { group, arms, arc, balls, label };
  }, []);
  useEffect(
    () => () => {
      disposeObject(parts.arms);
      disposeObject(parts.arc);
      parts.balls.forEach(disposeObject);
      parts.label.dispose();
    },
    [parts],
  );

  useLayoutEffect(() => {
    const { group, arms, arc, balls, label } = parts;
    const degrees = points.length === 3 ? angleDegrees(points[0], points[1], points[2]) : null;
    objectInfoProbe.angle = { points: points.map((p) => [p[0], p[1], p[2]]), degrees };
    if (points.length === 0) {
      group.visible = false;
      return;
    }
    group.visible = true;
    const r = Math.max(0, ...items.map((i) => i.radius));
    const size = r > 0 ? r * 0.012 : 0.05;
    balls.forEach((b, i) => {
      b.visible = i < points.length;
      if (b.visible) {
        b.position.set(points[i][0], points[i][1], points[i][2]);
        b.scale.setScalar(i === 1 ? size * 1.3 : size);
      }
    });
    setPositions(arms, Float32Array.from(points.flat()));
    arms.visible = points.length >= 2;
    if (points.length === 3 && degrees !== null) {
      const radius = arcRadius(points[0], points[1], points[2]);
      setPositions(arc, angleArc(points[0], points[1], points[2], radius));
      arc.visible = true;
      label.set(formatAngle(degrees), angleLabelAnchor(points[0], points[1], points[2], radius), Math.max(radius * 0.35, size * 4));
      label.sprite.visible = true;
    } else {
      arc.visible = false;
      label.sprite.visible = false;
    }
  }, [parts, points, items]);

  return <primitive object={parts.group} />;
}

/** The box's twelve edges and its three dimension labels, under `parent` (local box) or the scene (world box). */
function makeBoxParts() {
  const group = new THREE.Group();
  group.matrixAutoUpdate = false;
  const lines = makeLines(BOX_COLOR);
  const labels = [0, 1, 2].map(() => makeLabelSprite("#7fd4ff"));
  group.add(lines, ...labels.map((l) => l.sprite));
  return {
    group,
    /** Draws `box` (in the group's frame), labelled with `sizes` (the box's own, or the frame's). */
    show: (box: Box, unit: LengthUnit, height: number) => {
      setPositions(lines, boxEdges(box));
      const size = boxSize(box);
      boxLabelAnchors(box).forEach((at, i) => {
        labels[i].set(`${"XYZ"[i]} ${formatLength(size[i], unit)}`, at, height);
      });
    },
    dispose: () => {
      disposeObject(lines);
      labels.forEach((l) => l.dispose());
    },
  };
}

/**
 * The bounding box of the selected mesh: local (under its bound object) or
 * world-aligned (in the scene). Each frame has its own objects, so switching
 * rebuilds rather than reparents.
 */
function BoxOverlay(props: { scene: Scene | WorkerScene; item: Item; space: InfoSpace; unit: LengthUnit }) {
  return props.space === "local" ? <LocalBox {...props} /> : <WorldBox {...props} />;
}

/** The buffers a box reads, copied once per settled geometry change (views do not outlive the next scene call). */
function useBoxSource(scene: Scene | WorkerScene, mesh: MeshHandle) {
  const version = useGeometryVersion(mesh);
  return useMemo(() => {
    void version; // A new one reads the buffers again.
    try {
      const views = scene.views(mesh);
      const positions = Float32Array.from(views.positions());
      const used = usedVertices(views.indices(), positions.length / 3);
      return { positions, used, origin: scene.origin(mesh) };
    } catch {
      return null; // Removed meanwhile.
    }
  }, [scene, mesh, version]);
}

/** The box of the mesh's local coordinates, a child of its bound object: rebuilt per geometry change only. */
function LocalBox(props: { scene: Scene | WorkerScene; item: Item; unit: LengthUnit }) {
  const { scene, item, unit } = props;
  const source = useBoxSource(scene, item.mesh);
  const parts = useMemo(() => makeBoxParts(), []);
  useEffect(() => () => parts.dispose(), [parts]);
  useAttachToMesh(parts.group, item.mesh);
  const height = Math.max(item.radius * 0.06, 1e-3);
  useLayoutEffect(() => {
    const box = source === null ? null : linearBox(source.positions, source.used, null);
    parts.group.visible = box !== null;
    if (box !== null && source !== null) {
      parts.show(box, unit, height);
      objectInfoProbe.box = { space: "local", box: offsetBox(box, source.origin) };
    }
    return () => {
      objectInfoProbe.box = null;
    };
  }, [parts, source, unit, height]);
  return null;
}

/**
 * The world-aligned box of the mesh's vertices under its object's current
 * matrix (the gizmo's live pose included), checked each frame: the tracker
 * walks the vertices again only when the linear part changed.
 */
function WorldBox(props: { scene: Scene | WorkerScene; item: Item; unit: LengthUnit }) {
  const { scene, item, unit } = props;
  const source = useBoxSource(scene, item.mesh);
  const parts = useMemo(() => makeBoxParts(), []);
  useEffect(() => () => parts.dispose(), [parts]);
  const height = Math.max(item.radius * 0.06, 1e-3);
  const tracker = useMemo(() => (source === null ? null : new WorldBoxTracker(source.positions, source.used)), [source]);
  // The matrix last drawn; a new tracker or unit draws again.
  const last = useMemo(() => ({ m: null as number[] | null, tracker, unit, drawn: false }), [tracker, unit]);
  useLayoutEffect(
    () => () => {
      last.m = null;
      objectInfoProbe.box = null;
    },
    [last],
  );
  useFrame(() => {
    const object = view.bindings.get(item.mesh)?.object ?? null;
    if (object === null || last.tracker === null) {
      parts.group.visible = false;
      return;
    }
    object.updateWorldMatrix(true, false);
    const e = object.matrixWorld.elements;
    const prev = last.m;
    if (prev === null || e.some((x, i) => x !== prev[i])) {
      last.m = e.slice();
      const box = last.tracker.box(e);
      if (box !== null) {
        parts.show(box, last.unit, height);
        objectInfoProbe.box = { space: "world", box };
      }
      last.drawn = box !== null;
    }
    parts.group.visible = last.drawn && object.visible;
  });
  return <primitive object={parts.group} />;
}

/**
 * Face or vertex normals of the selected mesh as short lines under its
 * bound object, capped at `MAX_NORMAL_LINES`, rebuilt only when the geometry
 * (settled), the mode or the length change.
 */
function NormalsOverlay(props: { scene: Scene | WorkerScene; item: Item; mode: Exclude<NormalsMode, "off">; length: number }) {
  const { scene, item, mode, length } = props;
  const version = useGeometryVersion(item.mesh);
  const lines = useMemo(() => makeLines(mode === "face" ? FACE_NORMAL_COLOR : VERTEX_NORMAL_COLOR), [mode]);
  useEffect(() => () => disposeObject(lines), [lines]);
  useAttachToMesh(lines, item.mesh);

  useLayoutEffect(() => {
    let data: Float32Array = new Float32Array(0);
    try {
      const views = scene.views(item.mesh);
      const positions = views.positions();
      const indices = views.indices();
      data =
        mode === "face"
          ? faceNormalLines(positions, indices, length)
          : vertexNormalLines(positions, indices, usedVertices(indices, positions.length / 3), views.normals(), length);
    } catch {
      // Removed meanwhile: nothing to draw.
    }
    setPositions(lines, data);
    objectInfoProbe.normalLines = data.length / 6;
    objectInfoProbe.normalBuilds++;
    return () => {
      objectInfoProbe.normalLines = 0;
    };
  }, [scene, item.mesh, lines, mode, length, version]);

  return null;
}
