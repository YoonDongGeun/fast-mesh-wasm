// `fast-mesh-wasm/react`: React hooks over a `SceneStore`
// (`fast-mesh-wasm/store`). Every hook reads the store through
// `useSyncExternalStore` with a memoised selector, so components never tear
// and re-render only when their slice of the snapshot changes. The server
// (SSR) renders `EMPTY_SNAPSHOT`. Free of three.js and the WebAssembly glue;
// written without JSX, so the library needs no JSX transform.
import {
  createContext,
  createElement,
  type ReactElement,
  type ReactNode,
  useCallback,
  useContext,
  useMemo,
  useSyncExternalStore,
} from "react";
import { FastMeshError } from "../core.js";
import type { MeshHandle, SectionHandle, SectionPlane } from "../scene.js";
import {
  type DisplayState,
  EMPTY_SNAPSHOT,
  type MeshState,
  meshState,
  type SceneSnapshot,
  type SectionState,
} from "../store/snapshot.js";
import type { SceneStore } from "../store/store.js";

const StoreContext = createContext<SceneStore | null>(null);

/** Provides `store` to the hooks below it. */
export function SceneProvider(props: { store: SceneStore; children?: ReactNode }): ReactElement {
  return createElement(StoreContext.Provider, { value: props.store }, props.children);
}

/**
 * The store of the nearest {@link SceneProvider}.
 *
 * @throws `INVALID_ARGUMENT` outside a provider.
 */
export function useScene(): SceneStore {
  const store = useContext(StoreContext);
  if (store === null) {
    throw new FastMeshError("INVALID_ARGUMENT", "useScene() needs a <SceneProvider> above it");
  }
  return store;
}

/**
 * Selects a slice of the store's snapshot; the component re-renders only when
 * the slice changes (by `isEqual`, `Object.is` by default). The selection is
 * cached per snapshot, and an equal selection keeps the previous value, so a
 * selector may build new objects. The server renders `EMPTY_SNAPSHOT`.
 */
export function useStoreSelector<T>(
  selector: (s: SceneSnapshot) => T,
  isEqual: (a: T, b: T) => boolean = Object.is,
): T {
  const store = useScene();
  const [get, getServer] = useMemo(() => {
    const memo = (read: () => SceneSnapshot) => {
      let has = false;
      let lastSnap: SceneSnapshot | undefined;
      let last: T;
      return () => {
        const snap = read();
        if (has && snap === lastSnap) {
          return last;
        }
        const next = selector(snap);
        lastSnap = snap;
        if (!has || !isEqual(last, next)) {
          last = next;
          has = true;
        }
        return last;
      };
    };
    return [memo(store.getSnapshot), memo(() => EMPTY_SNAPSHOT)] as const;
  }, [store, selector, isEqual]);
  return useSyncExternalStore(store.subscribe, get, getServer);
}

/**
 * A mesh's state merged with its applied display: `opacity`, `visible` and
 * `coloring` are what is shown, `desired` what was last requested.
 */
export type MeshDisplay = MeshState & DisplayState;

// One merged object per (frozen) mesh state, so the selection is stable.
const merged = new WeakMap<MeshState, MeshDisplay>();

function mergeDisplay(state: MeshState): MeshDisplay {
  let out = merged.get(state);
  if (out === undefined) {
    out = Object.freeze({ ...state, ...state.display });
    merged.set(state, out);
  }
  return out;
}

/**
 * A mesh's display and its setter (`store.setDisplay`). The toggle-button
 * pattern is `pressed = state.desired.coloring !== null`, with
 * `busy = state.status === "computing"`. A mesh not in the store (removed,
 * or rendered on the server) reads as `REMOVED_MESH_STATE`: `"idle"`,
 * default display. The setter throws `UNKNOWN_MESH` for such a mesh.
 */
export function useMeshDisplay(mesh: MeshHandle): [MeshDisplay, (patch: Partial<DisplayState>) => void] {
  const store = useScene();
  const select = useCallback((s: SceneSnapshot) => mergeDisplay(meshState(s, mesh)), [mesh]);
  const state = useStoreSelector(select);
  const set = useCallback((patch: Partial<DisplayState>) => store.setDisplay(mesh, patch), [store, mesh]);
  return [state, set];
}

/** Whether undo and redo are available, and the actions. */
export interface HistoryControls {
  canUndo: boolean;
  canRedo: boolean;
  undo(): void;
  redo(): void;
}

const selectHistory = (s: SceneSnapshot) => s.history;

/** Undo and redo, with whether each is available. */
export function useHistory(): HistoryControls {
  const store = useScene();
  const history = useStoreSelector(selectHistory);
  return useMemo(
    () => ({
      canUndo: history.canUndo,
      canRedo: history.canRedo,
      undo: () => store.undo(),
      redo: () => store.redo(),
    }),
    [store, history],
  );
}

/** A change to a section: its plane, meshes or fill. */
export type SectionUpdate = Partial<{ plane: SectionPlane; meshes: readonly MeshHandle[]; fill: boolean }>;

const NO_PLANE: SectionPlane = Object.freeze({
  origin: Object.freeze([0, 0, 0]),
  normal: Object.freeze([0, 0, 1]),
}) as SectionPlane;
const NO_SECTION_PARTS = Object.freeze({ plane: NO_PLANE, meshes: Object.freeze([]), fill: false });

/** The state of a section not in a snapshot (removed, or on the server): `"idle"`, no meshes. */
const REMOVED_SECTION_STATE: SectionState = Object.freeze({
  ...NO_SECTION_PARTS,
  desired: NO_SECTION_PARTS,
  status: "idle",
  revision: 0,
});

/**
 * A section's state (the applied plane, meshes and fill, `desired` and
 * `status`) and its setter (`store.setSection`). A section not in the store
 * reads as `"idle"` with no meshes; the setter throws `UNKNOWN_SECTION` for it.
 */
export function useSection(section: SectionHandle): [SectionState, (patch: SectionUpdate) => void] {
  const store = useScene();
  const select = useCallback((s: SceneSnapshot) => s.sections.get(section) ?? REMOVED_SECTION_STATE, [section]);
  const state = useStoreSelector(select);
  const set = useCallback((patch: SectionUpdate) => store.setSection(section, patch), [store, section]);
  return [state, set];
}

/** The scene's overall status. */
export interface SceneStatus {
  status: "idle" | "computing" | "error";
  error?: FastMeshError;
}

const selectStatus = (s: SceneSnapshot): SceneStatus =>
  s.error === undefined ? { status: s.status } : { status: s.status, error: s.error };
const sameStatus = (a: SceneStatus, b: SceneStatus) => a.status === b.status && a.error === b.error;

/** The overall status: `"error"` (with the first error) when anything failed, else `"computing"` while anything is pending. */
export function useSceneStatus(): SceneStatus {
  return useStoreSelector(selectStatus, sameStatus);
}

const selectSelection = (s: SceneSnapshot) => s.selection;

/** The selected mesh (or `null`) and the function that selects one (`store.select`). */
export function useSelection(): [MeshHandle | null, (mesh: MeshHandle | null) => void] {
  const store = useScene();
  const selection = useStoreSelector(selectSelection);
  const select = useCallback((mesh: MeshHandle | null) => store.select(mesh), [store]);
  return [selection, select];
}
