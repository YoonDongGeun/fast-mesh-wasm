// The store context shared by `fast-mesh-wasm/react` and `fast-mesh-wasm/r3f`.
// Internal: one module, so both entries see the same context object.
import { createContext } from "react";
import type { SceneStore } from "../store/store.js";

/** The store a `SceneProvider` provides, or `null` outside one. */
export const StoreContext = createContext<SceneStore | null>(null);
