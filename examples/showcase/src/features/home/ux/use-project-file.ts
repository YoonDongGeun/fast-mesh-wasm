// Saving and reopening the session as a project file (issue #126): saving
// writes the session (meshes + placements + coloring/section state) through
// the File System Access picker where available, else a download; opening
// reads one back and replaces the session. A failure is kept as the panel's
// message; cancelling stays silent.
import { useScene } from "fast-mesh-wasm/react";
import { useState } from "react";
import { type AppState } from "../../../app/state/app-state";
import {
  collectProject,
  openProjectFile,
  parseProjectDoc,
  projectFileName,
  restoreProject,
  saveProjectFile,
} from "../../../io/project";

/** The project actions and the last one's error (`null` after a success). */
export function useProjectFile(app: AppState): { message: string | null; save: () => void; open: () => void } {
  const store = useScene();
  const [message, setMessage] = useState<string | null>(null);

  const save = (): void => {
    void (async () => {
      try {
        const text = JSON.stringify(collectProject(store, app));
        await saveProjectFile(text, projectFileName());
        setMessage(null);
      } catch (error) {
        setMessage(`Could not save the project: ${error instanceof Error ? error.message : String(error)}`);
      }
    })();
  };

  const open = (): void => {
    void (async () => {
      try {
        const file = await openProjectFile();
        if (file === null) {
          return;
        }
        restoreProject(store, app, parseProjectDoc(await file.text()));
        setMessage(null);
      } catch (error) {
        setMessage(`Could not open the project: ${error instanceof Error ? error.message : String(error)}`);
      }
    })();
  };

  return { message, save, open };
}
