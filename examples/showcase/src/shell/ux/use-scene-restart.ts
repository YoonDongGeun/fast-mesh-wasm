// Restarting the engine after an error: the store starts a new Worker that
// replays the scene. `restarting` holds while it runs, and the caller
// re-renders once it settles, to read the new Worker's thread count.
import { useScene } from "fast-mesh-wasm/react";
import { useState } from "react";

export function useSceneRestart() {
  const store = useScene();
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

  return { restarting, restart };
}
