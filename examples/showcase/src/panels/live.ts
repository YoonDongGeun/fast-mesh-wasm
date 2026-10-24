// What the collision panel's modes share: readouts kept live as their
// meshes change (throttled, paused while the gizmo drags), and the warnings
// for meshes that are open or inconsistently oriented as distance targets.
import type { MeshHandle, Scene } from "fast-mesh-wasm";
import { useScene } from "fast-mesh-wasm/react";
import type { SceneStore } from "fast-mesh-wasm/store";
import type { WorkerScene } from "fast-mesh-wasm/worker";
import { useEffect, useRef, useState } from "react";
import type { Item } from "../scene/app-state";
import { view } from "../viewport/view";
import { topologyOf } from "./collision-math";
import type { LiveProbe } from "./inspect";

/**
 * Outside a drag the readouts refresh at most this often (about 4 Hz), and
 * once more after the last change. During a gizmo drag they pause: reading
 * the field values forces every due field up to date, which would keep the
 * Worker from dropping recomputes of stale poses (the colours stay live).
 */
const READOUT_PERIOD_MS = 250;
/** A target is re-analysed this long after its triangles last changed. */
const TOPOLOGY_DELAY_MS = 250;

/**
 * Readouts of `meshes` from `query`, refreshed when any of them moves,
 * changes or is recoloured (its change records), at most every
 * READOUT_PERIOD_MS and once more after the last change. Paused while the
 * gizmo drags, then refreshed once when the drag ends. One query at a time;
 * `null` without meshes. `query` is read when it runs (the latest one
 * given), so the settings it reads are current. `poke` asks for a refresh
 * (a setting changed): the previous readouts stay until it lands. A mesh
 * removed mid-query rejects: that refresh is dropped. `key` names the set of
 * meshes; a new key starts over. `probe` mirrors the state for the test hook.
 */
export function useLiveReadouts<T>(
  store: SceneStore,
  meshes: readonly MeshHandle[] | null,
  key: string,
  query: () => Promise<T>,
  probe: LiveProbe<T>,
): { readouts: T | null; poke: () => void } {
  const [readouts, setReadouts] = useState<T | null>(null);
  const live = useRef({ query, meshes });
  live.current = { query, meshes };
  const poke = useRef<() => void>(() => {});
  const active = meshes !== null;

  useEffect(() => {
    // New meshes: the old readouts no longer apply.
    setReadouts(null);
    probe.readouts = null;
    const watched = live.current.meshes;
    if (!active || watched === null) {
      return;
    }
    let alive = true;
    let dirty = true;
    let running = false;
    let last = -Infinity;
    let timer: ReturnType<typeof setTimeout> | undefined;

    const kick = () => {
      probe.settled = !dirty && !running;
      if (!alive || running || !dirty || view.dragging) {
        // A drag pauses the readouts; its end kicks them again.
        return;
      }
      const wait = last + READOUT_PERIOD_MS - performance.now();
      if (wait > 0) {
        timer ??= setTimeout(() => {
          timer = undefined;
          kick();
        }, wait);
        return;
      }
      dirty = false;
      running = true;
      probe.settled = false;
      last = performance.now();
      Promise.resolve()
        .then(() => {
          probe.starts++;
          if (view.dragging) {
            probe.startsWhileDragging++;
          }
          return live.current.query();
        })
        .then((r) => (probe.hold === null ? r : probe.hold.then(() => r)))
        .then(
          (r) => {
            if (!alive) {
              return;
            }
            if (view.dragging) {
              // Started before the drag: it describes the pre-drag pose.
              // Drop it; the drag's end refreshes.
              dirty = true;
              probe.dropped++;
              return;
            }
            setReadouts(r);
            probe.readouts = r;
            probe.updates++;
            probe.updatesAfterDrag++;
          },
          () => {
            // A mesh removed (or the scene restarted) mid-query: skip this refresh.
          },
        )
        .finally(() => {
          running = false;
          kick();
        });
    };

    const off = store.scene.onChanges((changes) => {
      for (const c of changes) {
        if (!watched.includes(c.mesh)) {
          continue;
        }
        const changed = (r: readonly unknown[] | "all") => r === "all" || r.length > 0;
        if (c.placementChanged || changed(c.colors) || changed(c.indices) || c.fieldsChanged.length > 0) {
          dirty = true;
        }
      }
      kick();
    });
    const onDrag = (dragging: boolean) => {
      if (!dragging) {
        probe.updatesAfterDrag = 0;
        kick();
      }
    };
    view.dragListeners.add(onDrag);
    poke.current = () => {
      dirty = true;
      kick();
    };
    kick();
    return () => {
      alive = false;
      off();
      view.dragListeners.delete(onDrag);
      clearTimeout(timer);
      poke.current = () => {};
      probe.settled = true;
    };
  }, [store, key, active, probe]);

  return { readouts: active ? readouts : null, poke: () => poke.current() };
}

/**
 * Warnings for `meshes` that are open or inconsistently oriented: a field
 * measured to one takes its penetration side from its surface normals. From
 * `analyze` of each mesh (read-only, counts only), run when the meshes
 * change and again, coalesced, TOPOLOGY_DELAY_MS after a mesh's triangles
 * change (a repair, its undo); the latest analysis of each mesh wins. In the
 * order of `meshes`; `key` names the set.
 */
export function useTopologyWarnings(
  scene: Scene | WorkerScene,
  meshes: readonly MeshHandle[],
  key: string,
  nameOf: (m: MeshHandle) => string,
): string[] {
  const store = useScene();
  const [topology, setTopology] = useState<ReadonlyMap<MeshHandle, ReturnType<typeof topologyOf>>>(() => new Map());
  const list = useRef(meshes);
  list.current = meshes;

  useEffect(() => {
    setTopology(new Map());
    const watched = list.current;
    if (watched.length === 0) {
      return;
    }
    let alive = true;
    const seq = new Map<MeshHandle, number>();
    const timers = new Map<MeshHandle, ReturnType<typeof setTimeout>>();
    const run = (m: MeshHandle) => {
      const n = (seq.get(m) ?? 0) + 1;
      seq.set(m, n);
      Promise.resolve()
        .then(() => scene.analyze(m, { maxListed: 1 }))
        .then(
          (an) => {
            if (alive && seq.get(m) === n) {
              setTopology((t) => new Map(t).set(m, topologyOf(an)));
            }
          },
          () => {
            // Removed meanwhile: the meshes follow.
          },
        );
    };
    for (const m of watched) {
      run(m);
    }
    const off = store.scene.onChanges((changes) => {
      for (const c of changes) {
        if (watched.includes(c.mesh) && !c.removed && (c.indices === "all" || c.indices.length > 0)) {
          const m = c.mesh;
          clearTimeout(timers.get(m));
          timers.set(
            m,
            setTimeout(() => run(m), TOPOLOGY_DELAY_MS),
          );
        }
      }
    });
    return () => {
      alive = false;
      off();
      for (const t of timers.values()) {
        clearTimeout(t);
      }
    };
  }, [store, scene, key]);

  const warnings: string[] = [];
  for (const m of meshes) {
    const t = topology.get(m);
    if (t === undefined) {
      continue;
    }
    if (t.open) {
      warnings.push(
        `${nameOf(m)} is open: penetration side follows its surface normals; repair → orient/fill holes for a closed result`,
      );
    } else if (t.inconsistent) {
      warnings.push(
        `${nameOf(m)} is inconsistently oriented: penetration side follows its surface normals; repair → orient for a closed result`,
      );
    }
  }
  return warnings;
}

/** A stable key for a list of meshes (their list ids), for effect dependencies. */
export function meshesKey(items: readonly Item[], meshes: readonly MeshHandle[]): string {
  return meshes.map((m) => items.find((i) => i.mesh === m)?.id ?? "?").join(",");
}
