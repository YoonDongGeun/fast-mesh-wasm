// Named mesh groups in the scene: each coloured group as a contact group of
// its live members (`store.setContactGroup`, one id per group, released when
// no longer wanted), each group's collision response override applied to
// every pair of its members, and the group actions (show or hide all its
// members, ungroup, remove its meshes: each its own undo entry).
import type { CollisionMode, MeshHandle } from "fast-mesh-wasm";
import { useScene } from "fast-mesh-wasm/react";
import { contactHeatmap } from "fast-mesh-wasm/store";
import { useEffect, useRef, useState } from "react";
import { type AppState, type Item, useApp } from "../../../app/state/app-state";
import type { Wave6CollisionState } from "../../../app/state/wave6-state";
import { buildCollisionRule, clampTolerance, collisionRuleSignature, meshScale } from "./collision-math";
import { liveMembers, type NamedGroup } from "./groups";
import type { Heatmap } from "./use-heatmap-settings";

/** The store's contact-group id of a named group. */
function contactId(id: number): string {
  return `showcase-named-${id}`;
}

/** The heatmap of a named group: the collision defaults at its members' scale. */
export function groupHeat(items: readonly Item[], members: readonly MeshHandle[]): Heatmap {
  const scale = meshScale(items, members);
  const step = scale / 1000;
  const depth = scale / 2;
  const gap = scale / 2;
  return contactHeatmap({ depth, gap, tolerance: clampTolerance(scale / 50, depth, gap, step), exactDepth: true });
}

/**
 * The named groups' scene wiring. `groupRules` holds each group's response
 * override (none: the default rule); `setGroupCollisionMode` sets or clears
 * one, `ungroup` and `removeGroupMeshes` clear it first.
 */
export function useNamedGroups(app: AppState): {
  items: readonly Item[];
  groups: readonly NamedGroup<MeshHandle>[];
  collision: Wave6CollisionState;
  groupRules: Record<number, CollisionMode>;
  setGroupCollisionMode: (groupId: number, members: readonly MeshHandle[], mode: CollisionMode | "default") => void;
  setVisible: (members: readonly MeshHandle[], visible: boolean) => void;
  ungroup: (groupId: number, members: readonly MeshHandle[]) => void;
  removeGroupMeshes: (groupId: number, members: readonly MeshHandle[]) => void;
} {
  const store = useScene();
  const items = useApp(app, (s) => s.items);
  const groups = useApp(app, (s) => s.meshGroups);
  const collision = useApp(app, (s) => s.wave6.collision);
  const [groupRules, setGroupRules] = useState<Record<number, CollisionMode>>({});
  const lastGroupSigRef = useRef(new Map<number, string>());
  const lastSceneRef = useRef<object | null>(store.scene);

  if (lastSceneRef.current !== store.scene) {
    lastSceneRef.current = store.scene;
    lastGroupSigRef.current.clear();
  }

  function syncGroupPairRules(groupId: number, members: readonly MeshHandle[], mode: CollisionMode | "default") {
    if (mode === "default") {
      lastGroupSigRef.current.delete(groupId);
      for (let i = 0; i < members.length; i++) {
        for (let j = i + 1; j < members.length; j++) {
          void store.scene.clearCollisionRule?.(members[i], members[j]);
        }
      }
      return;
    }
    const rule = buildCollisionRule(collision, { mode });
    const sig = `${members.join(",")}|${collisionRuleSignature(rule)}`;
    if (lastGroupSigRef.current.get(groupId) === sig) {
      return;
    }
    lastGroupSigRef.current.set(groupId, sig);
    for (let i = 0; i < members.length; i++) {
      for (let j = i + 1; j < members.length; j++) {
        store.scene.setCollisionRule?.(members[i], members[j], rule);
      }
    }
  }

  function setGroupCollisionMode(groupId: number, members: readonly MeshHandle[], mode: CollisionMode | "default") {
    syncGroupPairRules(groupId, members, mode);
    setGroupRules((prev) => {
      if (mode === "default") {
        if (!(groupId in prev)) {
          return prev;
        }
        const next = { ...prev };
        delete next[groupId];
        return next;
      }
      return { ...prev, [groupId]: mode };
    });
  }

  useEffect(() => {
    const activeIds = new Set(groups.map((g) => g.id));
    for (const key of lastGroupSigRef.current.keys()) {
      if (!activeIds.has(key)) {
        lastGroupSigRef.current.delete(key);
      }
    }
    setGroupRules((prev) => {
      let changed = false;
      const next: Record<number, CollisionMode> = {};
      for (const [rawId, mode] of Object.entries(prev)) {
        const id = Number(rawId);
        if (activeIds.has(id)) {
          next[id] = mode;
        } else {
          changed = true;
        }
      }
      return changed ? next : prev;
    });
  }, [groups]);

  useEffect(() => {
    for (const g of groups) {
      const mode = groupRules[g.id];
      if (!mode) {
        continue;
      }
      const live = liveMembers(g.members, items);
      if (live.length >= 2) {
        syncGroupPairRules(g.id, live, mode);
      }
    }
  }, [collision, groups, groupRules, items, store]);
  // The contact-group ids currently applied: released when no longer wanted.
  const applied = useRef(new Set<number>());
  useEffect(() => {
    const want = new Map<number, MeshHandle[]>();
    for (const g of groups) {
      const live = liveMembers(g.members, items);
      if (g.colored && live.length >= 2) {
        want.set(g.id, live);
      }
    }
    for (const [id, members] of want) {
      store.setContactGroup(contactId(id), { meshes: members, pairs: "all", coloring: groupHeat(items, members) });
    }
    for (const id of applied.current) {
      if (!want.has(id)) {
        store.setContactGroup(contactId(id), null);
      }
    }
    applied.current = new Set(want.keys());
  }, [store, groups, items]);
  useEffect(() => {
    const at = applied.current;
    return () => {
      for (const id of at) {
        store.setContactGroup(contactId(id), null);
      }
    };
  }, [store]);

  const setVisible = (members: readonly MeshHandle[], visible: boolean) => {
    for (const m of members) {
      store.setDisplay(m, { visible });
    }
  };
  const removeMeshes = (members: readonly MeshHandle[]) => {
    // One undo step brings them all back.
    store.beginStep();
    try {
      for (const item of items.filter((i) => members.includes(i.mesh))) {
        app.removeMesh(store, item);
      }
    } finally {
      store.endStep();
    }
  };

  return {
    items,
    groups,
    collision,
    groupRules,
    setGroupCollisionMode,
    setVisible,
    ungroup: (groupId, members) => {
      setGroupCollisionMode(groupId, members, "default");
      app.removeGroup(groupId);
    },
    removeGroupMeshes: (groupId, members) => {
      setGroupCollisionMode(groupId, members, "default");
      removeMeshes(members);
    },
  };
}
