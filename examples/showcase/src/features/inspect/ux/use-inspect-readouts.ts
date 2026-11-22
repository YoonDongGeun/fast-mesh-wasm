// The inspect panel's readouts (#110): the query hit's coloring value and
// mesh scalars, refreshed per hit, and the test hook's view of the tools.
import type { MeshHandle, Scene } from "fast-mesh-wasm";
import type { SceneStore } from "fast-mesh-wasm/store";
import type { WorkerScene } from "fast-mesh-wasm/worker";
import { useEffect, useState } from "react";
import type { AppSnapshot, InspectTool, Item } from "../../../app/state/app-state";
import { inspect } from "../../../app/state/readouts";
import { readMeasure } from "./inspect-tools";

/**
 * The query's coloring value (whatever the mesh is colored by, including
 * client-side surface fields) and the mesh's scalars, refreshed per hit.
 * A mesh without a coloring has none.
 */
export function useQueryReadout(
  store: SceneStore,
  scene: Scene | WorkerScene,
  query: AppSnapshot["inspectQuery"],
  overlayValues: AppSnapshot["surfaceFieldValues"],
): { queryValue: number | null; queryMeasure: { area: number; volume: number | null } | null } {
  const [queryValue, setQueryValue] = useState<number | null>(null);
  const [queryMeasure, setQueryMeasure] = useState<{ area: number; volume: number | null } | null>(null);
  useEffect(() => {
    if (query === null) {
      setQueryValue(null);
      setQueryMeasure(null);
      return;
    }
    let alive = true;
    if (overlayValues !== null && overlayValues.mesh === query.mesh) {
      const val = overlayValues.values[query.slot];
      setQueryValue(val !== undefined && Number.isFinite(val) ? val : null);
    } else {
      store
        .fetchColoringValues(query.mesh)
        .then((values) => {
          if (alive) {
            setQueryValue(values === null ? null : (values[query.slot] ?? null));
          }
        })
        .catch(() => {
          if (alive) {
            setQueryValue(null);
          }
        });
    }
    void readMeasure(scene, query.mesh).then((m) => {
      if (alive) {
        setQueryMeasure(m);
      }
    });
    return () => {
      alive = false;
    };
  }, [store, scene, query, overlayValues]);
  return { queryValue, queryMeasure };
}

/**
 * Publishes the tools to the test hook after every render (names, not
 * handles): the armed tool, the ruler, the query, the section and the clip box.
 */
export function usePublishInspectTools(state: {
  items: readonly Item[];
  tool: InspectTool | null;
  ruler: AppSnapshot["ruler"];
  distance: number | null;
  query: AppSnapshot["inspectQuery"];
  queryValue: number | null;
  section: AppSnapshot["inspectSection"];
  clip: AppSnapshot["clipBox"];
}): void {
  const { items, tool, ruler, distance, query, queryValue, section, clip } = state;
  useEffect(() => {
    const nameOf = (m: MeshHandle) => items.find((i) => i.mesh === m)?.name ?? null;
    inspect.tools.tool = tool;
    inspect.tools.ruler =
      ruler === null
        ? { a: [0, 0, 0], b: null, distance: null }
        : { a: [...ruler.a], b: ruler.b === null ? null : [...ruler.b], distance };
    inspect.tools.query =
      query === null
        ? null
        : {
            mesh: nameOf(query.mesh) ?? "?",
            slot: query.slot,
            point: [...query.point],
            value: queryValue,
          };
    inspect.tools.section =
      section === null
        ? null
        : {
            plane: { origin: [...section.plane.origin], normal: [...section.plane.normal] },
            meshes: items.length,
          };
    inspect.tools.clip = clip === null ? null : { min: [...clip.min], max: [...clip.max] };
  });
}
