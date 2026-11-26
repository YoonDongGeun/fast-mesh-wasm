// What the app says when an action is permanent (#281): the engine reports
// each step larger than its whole undo memory, which it cannot keep, and the
// status bar shows it in words. Display state: not undoable, not persisted.
import type { MeshHandle } from "fast-mesh-wasm";
import type { PermanentStep } from "fast-mesh-wasm/store";
import { formatBytes } from "../../features/mesh-repair/ux/wrap";

/** How long a notice stays, in milliseconds. */
export const HISTORY_NOTICE_MS = 15_000;

/** The undo memory of a `?history=N` URL, N in MiB; `undefined` for the engine's default. */
export function historyBudgetParam(params: URLSearchParams): number | undefined {
  const text = params.get("history");
  const mib = text === null || text.trim() === "" ? NaN : Number(text);
  return mib >= 0 ? mib * 2 ** 20 : undefined;
}

/** `"A"`, `"A" and "B"`, or `"A", "B" and 3 more`. */
function quoted(names: readonly string[]): string {
  const q = names.map((n) => `"${n}"`);
  if (q.length <= 2) {
    return q.join(" and ");
  }
  return `${q.slice(0, 2).join(", ")} and ${q.length - 2} more`;
}

/** Why `step` was not kept and what that means, in words. `nameOf` is `undefined` for a mesh never listed. */
export function describePermanent(
  step: PermanentStep,
  nameOf: (mesh: MeshHandle) => string | undefined,
  budget: number,
): string {
  const names = (meshes: readonly MeshHandle[]) => quoted(meshes.flatMap((m) => nameOf(m) ?? []));
  const gone = names(step.gone);
  const changed = names(step.meshes.filter((m) => !step.gone.includes(m)));
  let what: string;
  if (step.lost === "redo") {
    what = gone !== "" ? `Redo cannot bring back ${gone}` : "The undone change cannot be redone";
  } else if (gone !== "") {
    what = changed === "" ? `Removing ${gone} cannot be undone` : `Changing ${changed} and removing ${gone} cannot be undone`;
  } else {
    what = changed === "" ? "The last change cannot be undone" : `The last change to ${changed} cannot be undone`;
  }
  return `${what}: it takes ${formatBytes(step.bytes)} of undo memory, over the limit of ${formatBytes(budget)}.`;
}
