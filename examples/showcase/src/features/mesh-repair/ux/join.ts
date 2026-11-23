// Join's engine call: the source's geometry copied into the target, a loop
// on each bridged and the source removed, all one undo step. The list follows
// the source out and, on undo, back.
import type { JoinOptions, JoinReport, MeshHandle, Scene } from "fast-mesh-wasm";
import type { SceneStore } from "fast-mesh-wasm/store";
import type { WorkerScene } from "fast-mesh-wasm/worker";
import type { AppState, Item } from "../../../app/state/app-state";

/** Joins `source` into `target`; a failed join changes neither mesh. */
export async function joinMeshes(
  store: SceneStore,
  app: AppState,
  target: MeshHandle,
  source: Item,
  options: JoinOptions,
): Promise<JoinReport> {
  // Like a removal: release the pair colouring first, in the same tick the
  // engine takes the source out.
  const { collision, collisionMode } = app.getSnapshot();
  if (collisionMode === "pair" && (collision.a === source.mesh || collision.b === source.mesh)) {
    const other = collision.a === source.mesh ? collision.b : collision.a;
    if (other !== null && other !== source.mesh) {
      store.setDisplay(other, { coloring: null });
    }
  }
  const scene = store.scene as Scene | WorkerScene;
  const report = await scene.join(target, source.mesh, options);
  // The store and the list follow the source out at once.
  store.commitFrame();
  app.syncMeshes(store);
  return report;
}
