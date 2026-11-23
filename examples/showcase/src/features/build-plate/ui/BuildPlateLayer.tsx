// The build plate in the viewport (#266): a grid rectangle on the floor with
// a faint fill, while it is shown. It takes no clicks and no hover: pointer
// rays go through it to what lies behind.
import { useThree } from "@react-three/fiber";
import { useEffect, useLayoutEffect, useMemo } from "react";
import * as THREE from "three";
import { type AppState, useApp } from "../../../app/state/app-state";
import { type BuildPlate, plateHasSize } from "../ux/plate";
import { plateGrid } from "../ux/plate-grid";

/** The plate's scene objects, and how to free them. */
export function buildPlateObject(plate: Pick<BuildPlate, "width" | "depth" | "origin" | "up">): {
  root: THREE.Group;
  dispose(): void;
} {
  const { lines, corners } = plateGrid(plate);
  const through = (points: Float32Array) =>
    new THREE.BufferGeometry().setAttribute("position", new THREE.BufferAttribute(points, 3));
  const grid = through(lines);
  const edge = through(corners);
  const face = through(corners).setIndex([0, 1, 2, 0, 2, 3]);
  const gridMaterial = new THREE.LineBasicMaterial({
    color: 0x5b8fb0,
    transparent: true,
    opacity: 0.45,
    depthWrite: false,
  });
  const edgeMaterial = new THREE.LineBasicMaterial({ color: 0x7cc4ff });
  const faceMaterial = new THREE.MeshBasicMaterial({
    color: 0x7cc4ff,
    transparent: true,
    opacity: 0.06,
    side: THREE.DoubleSide,
    depthWrite: false,
    // Behind the meshes that rest on it.
    polygonOffset: true,
    polygonOffsetFactor: 1,
    polygonOffsetUnits: 1,
  });
  const root = new THREE.Group();
  root.name = "build-plate";
  root.add(
    new THREE.Mesh(face, faceMaterial),
    new THREE.LineSegments(grid, gridMaterial),
    new THREE.LineLoop(edge, edgeMaterial),
  );
  root.traverse((o) => {
    // Never under the pointer: clicks go to the meshes.
    o.raycast = () => {};
  });
  return {
    root,
    dispose: () => {
      for (const d of [grid, edge, face, gridMaterial, edgeMaterial, faceMaterial]) {
        d.dispose();
      }
    },
  };
}

export function BuildPlateLayer(props: { app: AppState }) {
  const plate = useApp(props.app, (s) => s.buildPlate);
  const invalidate = useThree((s) => s.invalidate);
  const shown = plate.visible && plateHasSize(plate);
  const { width, depth, up } = plate;
  const [x, y] = plate.origin;
  const built = useMemo(
    () => (shown ? buildPlateObject({ width, depth, origin: [x, y], up }) : null),
    [shown, width, depth, x, y, up],
  );
  useEffect(() => () => built?.dispose(), [built]);
  useLayoutEffect(() => invalidate(), [built, invalidate]);
  return built === null ? null : <primitive object={built.root} />;
}
