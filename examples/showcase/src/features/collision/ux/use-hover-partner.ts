// The Group mode's hover probe: over a coloured member, which partner the
// vertex under the pointer is nearest to (its field's `nearest` buffer,
// read through `fetchFieldNearest` / `fieldNearest`) and how far it is.
// A member's values are read when it is first hovered and again, at most
// every REFRESH_MS, after it changes; never while the gizmo drags (reading
// field values would force stale poses current), so the tip may lag a move.
import type { MeshHandle, Scene } from "fast-mesh-wasm";
import { useScene } from "fast-mesh-wasm/react";
import type { WorkerScene } from "fast-mesh-wasm/worker";
import { useEffect, useReducer, useRef, useSyncExternalStore } from "react";
import { inspect } from "../../../app/state/readouts";
import { view } from "../../../viewport/ux/view";
import { type MemberValues, nearestPartner } from "./group-math";
import { readMember } from "./use-group-contact";

const REFRESH_MS = 250;

const subscribe = (l: () => void) => {
  view.hoverListeners.add(l);
  return () => view.hoverListeners.delete(l);
};

/**
 * The hover (`view.hover`), the hovered member of `members` (`null` over
 * anything else), its values as last read (`null` until read) and the
 * partner nearest to the hovered vertex (`null` without one). Mirrored to
 * the test hook's `inspect.hover`.
 */
export function useHoverPartner(
  scene: Scene | WorkerScene,
  members: readonly MeshHandle[],
): {
  hover: typeof view.hover;
  member: MeshHandle | null;
  data: MemberValues<MeshHandle> | null;
  hit: { partner: MeshHandle; distance: number } | null;
} {
  const store = useScene();
  const hover = useSyncExternalStore(subscribe, () => view.hover);
  const [, rerender] = useReducer((n: number) => n + 1, 0);
  // Per member: the values last read, and whether it changed since.
  const cache = useRef(new Map<MeshHandle, { data: MemberValues<MeshHandle> | null; stale: boolean }>());
  const member = hover !== null && members.includes(hover.mesh) ? hover.mesh : null;

  useEffect(() => {
    const c = cache.current;
    return store.scene.onChanges((changes) => {
      for (const ch of changes) {
        const e = c.get(ch.mesh);
        if (ch.removed) {
          c.delete(ch.mesh);
        } else if (e !== undefined) {
          e.stale = true;
        }
      }
    });
  }, [store]);

  // Read the hovered member's values when it has none or they are stale.
  useEffect(() => {
    if (member === null) {
      return;
    }
    let alive = true;
    let timer: ReturnType<typeof setTimeout> | undefined;
    let running = false;
    let last = -Infinity;
    const c = cache.current;
    const tick = () => {
      const e = c.get(member);
      if (!alive || running || view.dragging || (e !== undefined && !e.stale)) {
        timer = setTimeout(tick, REFRESH_MS);
        return;
      }
      const wait = last + REFRESH_MS - performance.now();
      if (wait > 0) {
        timer = setTimeout(tick, wait);
        return;
      }
      running = true;
      last = performance.now();
      c.set(member, { data: e?.data ?? null, stale: false });
      readMember(store, scene, member)
        .then(
          (data) => {
            if (c.has(member)) {
              c.set(member, { data, stale: c.get(member)?.stale ?? false });
            }
            if (alive) {
              rerender();
            }
          },
          () => {
            // Removed meanwhile.
          },
        )
        .finally(() => {
          running = false;
        });
      timer = setTimeout(tick, REFRESH_MS);
    };
    tick();
    return () => {
      alive = false;
      clearTimeout(timer);
    };
  }, [store, scene, member]);

  const data = member === null ? null : (cache.current.get(member)?.data ?? null);
  const hit =
    hover === null || data === null || data.values === null || data.nearest === null
      ? null
      : nearestPartner(data.values, data.nearest, data.partners, hover.slot);
  const probe = member === null ? null : { mesh: member, partner: hit?.partner ?? null, distance: hit?.distance ?? null };
  useEffect(() => {
    inspect.hover = probe;
  });
  useEffect(
    () => () => {
      inspect.hover = null;
    },
    [],
  );
  return { hover, member, data, hit };
}
