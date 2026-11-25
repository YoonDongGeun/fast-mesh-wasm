// The placement picks' viewport layer (#256): a ball on each point picked
// so far (3-2-1's plane, line and origin points in their own colours),
// drawn over the mesh. The taps themselves are `usePlacementPickPointer`'s.
import type { SceneStore } from "fast-mesh-wasm/store";
import { useEffect, useLayoutEffect, useMemo } from "react";
import * as THREE from "three";
import { type AppState, useApp } from "../../../app/state/app-state";
import { pickMarkerGroup } from "../ux/placement-picks";
import { usePlacementPickPointer } from "../ux/use-placement-picks";

const MARKER_COLORS = [0x38bdf8, 0xf59e0b, 0xe879f9] as const;

export function PlacementPicksLayer(props: { app: AppState; store: SceneStore }) {
  const { app, store } = props;
  usePlacementPickPointer(app, store);
  const pick = useApp(app, (s) => s.placementTool.pick);
  const items = useApp(app, (s) => s.items);
  const group = useMemo(() => new THREE.Group(), []);
  const ball = useMemo(() => new THREE.SphereGeometry(1, 16, 12), []);
  const materials = useMemo(
    () => MARKER_COLORS.map((color) => new THREE.MeshBasicMaterial({ color, depthTest: false, depthWrite: false })),
    [],
  );
  useEffect(
    () => () => {
      ball.dispose();
      for (const m of materials) {
        m.dispose();
      }
    },
    [ball, materials],
  );

  useLayoutEffect(() => {
    group.clear();
    if (pick === null) {
      return;
    }
    const radius = items.find((i) => i.mesh === pick.mesh)?.radius ?? 1;
    const size = radius > 0 ? radius * 0.015 : 0.05;
    pick.points.forEach((p, i) => {
      const marker = new THREE.Mesh(ball, materials[pickMarkerGroup(pick.kind, i)]);
      marker.position.set(p[0], p[1], p[2]);
      marker.scale.setScalar(size);
      marker.renderOrder = 12;
      group.add(marker);
    });
  }, [group, ball, materials, pick, items]);

  return <primitive object={group} />;
}
