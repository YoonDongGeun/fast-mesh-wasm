// Undo and redo through the store's history, minding a sculpt stroke in
// flight (see `afterStroke`): a press while the button is down is dropped,
// one after it waits for the stroke to land.
import { useHistory } from "fast-mesh-wasm/react";
import { useCallback } from "react";
import { afterStroke } from "../../viewport/ux/view";

export function useUndoRedo() {
  const { canUndo, canRedo, undo: undoNow, redo: redoNow } = useHistory();
  const undo = useCallback(() => afterStroke(undoNow), [undoNow]);
  const redo = useCallback(() => afterStroke(redoNow), [redoNow]);
  return { canUndo, canRedo, undo, redo };
}
