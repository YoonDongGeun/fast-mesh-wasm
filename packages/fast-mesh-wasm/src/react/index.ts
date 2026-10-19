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
  useRef,
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
  sectionState,
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

/** A memoised selection: the last snapshot and selector seen, and the value kept. */
interface Memo<T> {
  snap?: SceneSnapshot;
  selector?: (s: SceneSnapshot) => unknown;
  value?: T;
  has: boolean;
}

/**
 * Selects a slice of the store's snapshot; the component re-renders only when
 * the slice changes (by `isEqual`, `Object.is` by default). The selection is
 * cached per snapshot and selector, and a new selection equal to the
 * previous one keeps the previous value, so a selector may build new objects
 * and may be an inline function. The server renders `EMPTY_SNAPSHOT`.
 */
export function useStoreSelector<T>(
  selector: (s: SceneSnapshot) => T,
  isEqual: (a: T, b: T) => boolean = Object.is,
): T {
  const store = useScene();
  // The latest selector and equality, read by the stable getters below (as
  // use-sync-external-store/with-selector does), so an inline selector does
  // not reset the cache on every render of its parent.
  const selectorRef = useRef(selector);
  const isEqualRef = useRef(isEqual);
  selectorRef.current = selector;
  isEqualRef.current = isEqual;
  const [get, getServer] = useMemo(() => {
    const memo = (read: () => SceneSnapshot) => {
      const m: Memo<T> = { has: false };
      return (): T => {
        const snap = read();
        const sel = selectorRef.current;
        if (m.has && snap === m.snap && sel === m.selector) {
          return m.value as T;
        }
        const next = sel(snap);
        m.snap = snap;
        m.selector = sel;
        if (!m.has || !isEqualRef.current(m.value as T, next)) {
          m.value = next;
          m.has = true;
        }
        return m.value as T;
      };
    };
    return [memo(store.getSnapshot), memo(() => EMPTY_SNAPSHOT)] as const;
  }, [store]);
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

/** Undo and redo, with whether each is available. The object and its actions keep their identity while history is unchanged; the actions are stable per store. */
export function useHistory(): HistoryControls {
  const store = useScene();
  const history = useStoreSelector(selectHistory);
  const undo = useCallback(() => store.undo(), [store]);
  const redo = useCallback(() => store.redo(), [store]);
  return useMemo(
    () => ({ canUndo: history.canUndo, canRedo: history.canRedo, undo, redo }),
    [history, undo, redo],
  );
}

/** A change to a section: its plane, meshes or fill. */
export type SectionUpdate = Partial<{ plane: SectionPlane; meshes: readonly MeshHandle[]; fill: boolean }>;

/**
 * A section's state (the applied plane, meshes and fill, `desired` and
 * `status`) and its setter (`store.setSection`). A section not in the store
 * reads as `REMOVED_SECTION_STATE` (`"idle"`, no meshes); the setter throws
 * `UNKNOWN_SECTION` for it.
 */
export function useSection(section: SectionHandle): [SectionState, (patch: SectionUpdate) => void] {
  const store = useScene();
  const select = useCallback((s: SceneSnapshot) => sectionState(s, section), [section]);
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
