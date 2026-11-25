// The select panel's criteria (#258): the criterion picked, its settings,
// and the two actions. "Select faces" runs the criterion on the active mesh
// and combines its faces with the selection by the combine mode; "Expand to
// components" grows the selection to the whole components it touches. Both
// go through the panel's one-action-at-a-time runner.
import type { SceneStore } from "fast-mesh-wasm/store";
import { useState } from "react";
import type { AppState, Item } from "../../../app/state/app-state";
import { view } from "../../../viewport/ux/view";
import { commitFaceSelection, isItemVisible } from "./face-selection-ops";
import { EMPTY_U32, sanitizeSelectedTriangles } from "./face-sets";
import { cameraPoseOf } from "./screen-select";
import {
  type CriteriaParams,
  criterionLabel,
  DEFAULT_CRITERIA_PARAMS,
  expandSelectionToComponents,
  type FaceCriterion,
  findFacesByCriterion,
} from "./select-criteria";

type RunAction = (label: string, fn: () => Promise<string> | string) => Promise<void>;

export function useSelectCriteria(
  app: AppState,
  store: SceneStore,
  actions: { activeItem: Item | null; runAction: RunAction },
) {
  const [criterion, setCriterion] = useState<FaceCriterion>("area");
  const [params, setParams] = useState<CriteriaParams>(DEFAULT_CRITERIA_PARAMS);
  const { activeItem, runAction } = actions;

  const patchParams = (patch: Partial<CriteriaParams>) => setParams((prev) => ({ ...prev, ...patch }));

  const onSelectByCriterion = () => {
    if (!activeItem) return;
    const label = criterionLabel(criterion);
    // The mode and settings as clicked; the selection it combines with as it lands.
    const { combineMode, adjacency } = app.getSnapshot().wave6.select;
    const others = app
      .getSnapshot()
      .items.filter((it) => it.id !== activeItem.id && isItemVisible(store, it.mesh))
      .map((it) => it.mesh);
    const camera = view.camera === null ? null : cameraPoseOf(view.camera);
    void runAction(label, async () => {
      const incoming = await findFacesByCriterion(criterion, params, {
        scene: store.scene,
        mesh: activeItem.mesh,
        others,
        camera,
        adjacency,
      });
      const combined = commitFaceSelection(app, store, activeItem, incoming, combineMode);
      return `${label}: ${incoming.length} faces matched (${combined.length} selected).`;
    });
  };

  const onExpandToComponents = () => {
    if (!activeItem) return;
    void runAction("Expand to components", async () => {
      const indices = store.scene.views(activeItem.mesh).indices();
      const current = sanitizeSelectedTriangles(
        indices,
        app.getSnapshot().wave6.select.selectedByMesh[activeItem.id] ?? EMPTY_U32,
      );
      if (current.length === 0) {
        return "Nothing selected to expand.";
      }
      const grown = await expandSelectionToComponents(store.scene, activeItem.mesh, current);
      const combined = commitFaceSelection(app, store, activeItem, grown, "replace");
      return `Expanded to components (${combined.length} faces).`;
    });
  };

  return { criterion, setCriterion, params, patchParams, onSelectByCriterion, onExpandToComponents };
}
