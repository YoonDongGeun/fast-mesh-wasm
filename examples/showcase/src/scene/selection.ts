// Multi-selection transitions: a plain click resets (the primary doubles as
// the store's single selection), a shift-click toggles while the primary
// stays first (removing it promotes the next), and the Join source seeds
// from the secondaries in order. Pure over handles and ids, for tests.
import type { MeshHandle } from "fast-mesh-wasm";
import type { AppState } from "./app-state";

/**
 * The next multi-selection + primary for a click on `mesh` (`null` clears
 * on a plain click and is ignored on a shift-click). A list that no longer
 * starts at the primary rebases onto it first.
 */
export function clickMulti(
  multi: readonly MeshHandle[],
  primary: MeshHandle | null,
  mesh: MeshHandle | null,
  additive: boolean,
): { multi: readonly MeshHandle[]; primary: MeshHandle | null } {
  if (!additive) {
    return mesh === null ? { multi: [], primary: null } : { multi: [mesh], primary: mesh };
  }
  if (mesh === null) {
    return { multi, primary };
  }
  const base = primary !== null && multi[0] === primary ? multi : primary === null ? [] : [primary];
  if (base.includes(mesh)) {
    const next = base.filter((m) => m !== mesh);
    return { multi: next, primary: next[0] ?? null };
  }
  return { multi: [...base, mesh], primary: base[0] ?? mesh };
}

/**
 * The join source to seed from a multi-selection: a manual pick stands
 * while it is still a secondary, else the first secondary (`""` when fewer
 * than two meshes are selected).
 */
export function nextJoinSource(secondaries: readonly number[], current: number | ""): number | "" {
  if (current !== "" && secondaries.includes(current)) {
    return current;
  }
  return secondaries[0] ?? "";
}

/**
 * Clicks `mesh` through the store and the app at once (`null` clears on a
 * plain click, ignored on a shift-click). Every selection change goes
 * through here, so the store's single selection (the gizmo, the panels)
 * and the app's multi list (the Join sources) stay in step.
 */
export function selectMesh(
  app: AppState,
  select: (mesh: MeshHandle | null) => void,
  primary: MeshHandle | null,
  mesh: MeshHandle | null,
  additive: boolean,
): void {
  const next = clickMulti(app.getSnapshot().multi, primary, mesh, additive);
  app.setMulti(next.multi);
  select(next.primary);
}
