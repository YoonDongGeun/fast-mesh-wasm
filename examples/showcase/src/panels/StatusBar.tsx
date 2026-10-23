// The status bar: the scene's overall status, loads in progress, the meshes
// and triangles shown, the engine (Worker with its thread count, or local)
// and whether the page is cross-origin isolated; on an error its message
// and Restart (a new Worker that replays the scene).
import { useScene, useSceneStatus } from "fast-mesh-wasm/react";
import { useState } from "react";
import type { Engine } from "../App";
import { type AppState, useApp } from "../scene/app-state";

export function StatusBar(props: { engine: Engine; app: AppState }) {
  const { engine, app } = props;
  const store = useScene();
  const { status, error } = useSceneStatus();
  const items = useApp(app, (s) => s.items);
  const loading = useApp(app, (s) => s.jobs.filter((j) => j.phase !== "done" && j.phase !== "error").length);
  const [restarting, setRestarting] = useState(false);
  // Re-rendered after a restart: the new Worker's thread count.
  const [, setRestarts] = useState(0);

  const restart = () => {
    setRestarting(true);
    store
      .restart()
      .catch((e: unknown) => console.warn("restart failed", e))
      .finally(() => {
        setRestarting(false);
        setRestarts((n) => n + 1);
      });
  };

  const label = restarting ? "Restarting…" : status === "error" ? "Error" : status === "computing" ? "Computing…" : "Ready";
  const threads = engine.threads();

  return (
    <footer className="statusbar" role="status">
      <span className={`state state-${restarting ? "computing" : status}`} data-status={status}>
        <span className="dot" aria-hidden="true" />
        {label}
      </span>
      {loading > 0 ? <span className="busy">Loading {loading === 1 ? "1 file" : `${loading} files`}…</span> : null}
      <span>
        {items.length} {items.length === 1 ? "mesh" : "meshes"}
      </span>
      <span className="spacer" />
      {error !== undefined ? (
        <span className="error" title={error.code}>
          {error.code}: {error.message}
        </span>
      ) : null}
      {error !== undefined && engine.mode === "worker" ? (
        <button type="button" onClick={restart} disabled={restarting}>
          Restart
        </button>
      ) : null}
      {error !== undefined && engine.mode === "local" ? (
        <button type="button" onClick={() => location.reload()}>
          Reload
        </button>
      ) : null}
      <span data-threads={threads} title={engine.mode === "worker" ? "WorkerScene.threads" : "?local: the synchronous Scene"}>
        {engine.mode === "worker" ? `Worker · ${threads} ${threads === 1 ? "thread" : "threads"}` : "Local · main thread"}
      </span>
      <span
        className={engine.isolated ? "ok" : "warn"}
        title="Threads need a cross-origin isolated page (COOP/COEP headers)"
      >
        {engine.isolated ? "cross-origin isolated" : "not cross-origin isolated"}
      </span>
    </footer>
  );
}
