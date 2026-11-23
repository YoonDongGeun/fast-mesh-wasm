// The camera requests the shell sends the viewport: frame a mesh (from a
// direction, or a world box instead of the mesh) and reset the view. Each
// request is a new value, so asking twice for the same thing moves the
// camera twice.
import type { MeshHandle } from "fast-mesh-wasm";
import { useCallback, useState } from "react";
import type { FocusRequest } from "../App";

export function useCameraRequests() {
  const [focus, setFocus] = useState<FocusRequest | null>(null);
  const [resetView, setResetView] = useState(0);

  const focusOn = useCallback(
    (
      mesh: MeshHandle | null,
      dir?: readonly [number, number, number],
      box?: { center: readonly [number, number, number]; radius: number },
    ) => setFocus((f) => ({ mesh, n: (f?.n ?? 0) + 1, dir, box })),
    [],
  );
  const requestResetView = () => setResetView((n) => n + 1);

  return { focus, focusOn, resetView, requestResetView };
}
