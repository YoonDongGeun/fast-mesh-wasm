// The collision panel's Pair mode: two picked meshes, each coloured by its
// signed distance to the other, with live readouts of the deepest
// penetration, the gap, whether they intersect and the triangles in contact,
// and the topology warnings of both.
import type { MeshHandle, Scene } from "fast-mesh-wasm";
import { useScene } from "fast-mesh-wasm/react";
import type { WorkerScene } from "fast-mesh-wasm/worker";
import { useEffect } from "react";
import type { CollisionPair, Item } from "../../../app/state/app-state";
import { inspect, type Readouts } from "../../../app/state/readouts";
import { deepestPenetration } from "./collision-math";
import { meshesKey, useLiveReadouts, useTopologyWarnings } from "./live";
import type { Heatmap } from "./use-heatmap-settings";

/**
 * Colours the pair with `heat` while two different meshes are picked, and
 * keeps their readouts (contacts within `tolerance`) and warnings live.
 * `active` is the pair once two different meshes are picked, else `null`.
 */
export function usePairContact(
  scene: Scene | WorkerScene,
  items: readonly Item[],
  pair: CollisionPair,
  heat: Heatmap,
  tolerance: number,
): { active: readonly [MeshHandle, MeshHandle] | null; readouts: Readouts | null; warnings: string[] } {
  const store = useScene();
  const { a, b } = pair;
  const ready = a !== null && b !== null && a !== b;

  // Colour both, each against the other. Picking another mesh (or removing
  // one: AppState.removeMesh resets its picker) stops the old colouring.
  useEffect(() => {
    if (a === null || b === null || a === b) {
      return;
    }
    store.setDisplay(a, { coloring: { distanceTo: b, ...heat } });
    store.setDisplay(b, { coloring: { distanceTo: a, ...heat } });
    return () => {
      for (const m of [a, b]) {
        if (store.getSnapshot().meshes.has(m)) {
          store.setDisplay(m, { coloring: null });
        }
      }
    };
  }, [store, a, b, heat]);

  const nameOf = (m: MeshHandle) => items.find((i) => i.mesh === m)?.name ?? "the mesh";
  // B's warning first: the side A's colours are measured to.
  const watched = ready ? [b, a] : [];
  const key = meshesKey(items, watched);
  const bound = heat.maxDistance;
  const { readouts, poke } = useLiveReadouts<Readouts>(
    store,
    ready ? watched : null,
    key,
    async () => {
      if (!ready) {
        throw new Error("no pair");
      }
      const [va, vb, distance, hit, contacts] = await Promise.all([
        // After a commit (this runs from change records): the colouring last
        // sent. A colouring replaced meanwhile rejects; read it as none.
        store.fetchColoringValues(a).catch(() => null),
        store.fetchColoringValues(b).catch(() => null),
        Promise.resolve(scene.meshDistance(a, b)),
        Promise.resolve(scene.intersects(a, b)),
        Promise.resolve(scene.findContacts({ margin: tolerance, pairs: [[a, b]] })),
      ]);
      const c = contacts[0];
      return {
        penetration: deepestPenetration([va, vb], bound, hit),
        bound,
        gap: distance === null ? null : distance.distance,
        intersects: hit,
        contacts:
          c === undefined
            ? { a: 0, b: 0 }
            : c.a === a
              ? { a: c.trianglesA.length, b: c.trianglesB.length }
              : { a: c.trianglesB.length, b: c.trianglesA.length },
      };
    },
    inspect.collision,
  );
  // A new tolerance changes the contact count, a new bound how the depth
  // reads (the colours follow through change records).
  useEffect(() => poke(), [tolerance, bound]);
  const warnings = useTopologyWarnings(scene, watched, key, nameOf);
  useEffect(() => {
    inspect.collision.warnings = warnings;
  });
  return { active: ready ? [a, b] : null, readouts, warnings };
}
