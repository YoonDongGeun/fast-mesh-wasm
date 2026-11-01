// The project save/reopen buttons (issue #126): saving writes the session
// (meshes + placements + coloring/section state) through the File System
// Access picker where available, else a download; opening reads one back and
// replaces the session. Errors show in the panel; cancelling stays silent.
import { useScene } from "fast-mesh-wasm/react";
import { useState } from "react";
import { type AppState } from "../scene/app-state";
import {
  collectProject,
  openProjectFile,
  parseProjectDoc,
  projectFileName,
  restoreProject,
  saveProjectFile,
} from "../load/project";

export function ProjectPanel(props: { app: AppState }) {
  const { app } = props;
  const store = useScene();
  const [message, setMessage] = useState<string | null>(null);

  const onSave = (): void => {
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

  const onOpen = (): void => {
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

  return (
    <section className="panel" aria-label="Project">
      <header className="panel-head">
        <h2>Project</h2>
      </header>
      <div className="row">
        <button type="button" data-action="project-save" onClick={onSave} title="Save the session to a project file">
          Save project…
        </button>
        <button type="button" data-action="project-open" onClick={onOpen} title="Reopen a saved project file">
          Open project…
        </button>
      </div>
      {message !== null ? (
        <p className="error" role="alert">
          {message}
        </p>
      ) : null}
    </section>
  );
}
