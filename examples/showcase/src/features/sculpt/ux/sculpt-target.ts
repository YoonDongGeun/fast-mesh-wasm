// Which mesh the sculpt and mask tools work on (#229): only visible, unlocked
// meshes are eligible, and "Pin to active mesh" restricts them to the
// selection.
import type { MeshHandle } from "fast-mesh-wasm";
import { meshState, type SceneSnapshot } from "fast-mesh-wasm/store";
import type { Item } from "../../../app/state/app-state";

/** Filters `items` down to visible, unlocked meshes that sculpt/mask tools can target (#229). */
export function eligibleSculptItems(items: readonly Item[], storeSnapshot: SceneSnapshot): Item[] {
  return items.filter((item) => !item.locked && meshState(storeSnapshot, item.mesh).desired.visible);
}

/**
 * Resolves which visible, unlocked mesh sculpt/mask UI controls operate on (#229).
 * When `pinToActive` is true, restricts to `selection` (or the current target if unset).
 * Otherwise keeps `currentTarget` if still eligible, falling back to `selection` then first eligible item.
 */
export function resolveSculptTargetItem(
  items: readonly Item[],
  storeSnapshot: SceneSnapshot,
  selection: MeshHandle | null,
  currentTarget: MeshHandle | null,
  pinToActive: boolean,
): Item | null {
  return resolveTargetFromEligible(eligibleSculptItems(items, storeSnapshot), selection, currentTarget, pinToActive);
}

/** Returns true when at least one eligible target can arm a sculpt brush or mask paint (#229). */
export function canArmSculpt(
  items: readonly Item[],
  storeSnapshot: SceneSnapshot,
  selection: MeshHandle | null,
  pinToActive: boolean,
): boolean {
  const eligible = eligibleSculptItems(items, storeSnapshot);
  if (eligible.length === 0) {
    return false;
  }
  return pinToActive ? (selection === null ? false : eligible.some((i) => i.mesh === selection)) : true;
}

/** `resolveSculptTargetItem` over items already filtered to the eligible ones. */
export function resolveTargetFromEligible(
  eligible: readonly Item[],
  selection: MeshHandle | null,
  currentTarget: MeshHandle | null,
  pinToActive: boolean,
): Item | null {
  if (eligible.length === 0) {
    return null;
  }
  if (pinToActive) {
    if (selection !== null) {
      return eligible.find((i) => i.mesh === selection) ?? null;
    }
    return null;
  }
  if (currentTarget !== null) {
    const existing = eligible.find((i) => i.mesh === currentTarget);
    if (existing !== undefined) {
      return existing;
    }
  }
  if (selection !== null) {
    const selected = eligible.find((i) => i.mesh === selection);
    if (selected !== undefined) {
      return selected;
    }
  }
  return eligible[0];
}
