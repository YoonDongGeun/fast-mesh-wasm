// A counter that bumps on every non-empty batch of scene changes, so face
// selection views re-read mesh indices (live triangle counts, highlight
// geometry) after edits, undo and redo.
import type { SceneStore } from "fast-mesh-wasm/store";
import { type Dispatch, type SetStateAction, useEffect, useState } from "react";

/** The scene's change revision and its setter (a local edit can bump it at once). */
export function useSceneRevision(store: SceneStore): [number, Dispatch<SetStateAction<number>>] {
  const [revision, setRevision] = useState(0);

  useEffect(() => {
    const unsub = store.scene.onChanges((changes) => {
      if (changes.length > 0) {
        setRevision((r) => r + 1);
      }
    });
    return unsub;
  }, [store]);

  return [revision, setRevision];
}
