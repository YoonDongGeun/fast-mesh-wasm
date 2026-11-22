// The project save/reopen buttons (issue #126), over `useProjectFile`.
// Errors show in the panel; cancelling stays silent.
import { type AppState } from "../../../app/state/app-state";
import { useProjectFile } from "../ux/use-project-file";

export function ProjectPanel(props: { app: AppState }) {
  const { app } = props;
  const { message, save: onSave, open: onOpen } = useProjectFile(app);

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
