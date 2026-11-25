// Components whose code is a chunk of its own (#285), used as if they were
// imported. One that is here renders in the same pass; until then a layer
// shows nothing, a panel its heading at about its size, and a dialog that was
// opened a small one that can be closed. A failed chunk shows Retry.
//
// Not React.lazy: nothing suspends (a part that arrives inside the canvas
// leaves the canvas and its other layers alone), no fallback shows for a frame
// once the chunk is here, and a failed import is not for good. Not a transition either:
// one that edits or a drag keep interrupting would not finish.
import { type ComponentType, useEffect, useLayoutEffect, useSyncExternalStore } from "react";
import { type ChunkState, type LazyChunk, waitForChunk } from "../ux/lazy-chunk";

/** The chunk's state, as the part shows it. */
function useChunk<M>(chunk: LazyChunk<M>): ChunkState<M> {
  const state = useSyncExternalStore(chunk.subscribe, chunk.getSnapshot, chunk.getSnapshot);
  const here = state.status === "ready";
  // A layout effect: a command run right after the mount already finds it waiting.
  useLayoutEffect(() => (here ? undefined : waitForChunk(chunk)), [chunk, here]);
  return state;
}

/** `pick`'s component of `chunk`; `Fallback` (nothing by default) until it is here. */
export function lazyPart<M, P extends object>(
  chunk: LazyChunk<M>,
  pick: (module: M) => ComponentType<P>,
  Fallback?: ComponentType<{ chunk: LazyChunk; failed: boolean; props: P }>,
): ComponentType<P> {
  return function LazyPart(props: P) {
    const state = useChunk(chunk);
    if (state.status === "ready") {
      const Part = pick(state.module);
      return <Part {...props} />;
    }
    return Fallback === undefined ? null : <Fallback chunk={chunk} failed={state.status === "failed"} props={props} />;
  };
}

/**
 * A panel of `chunk`: until it is here, its heading in a box of about its
 * height (in CSS pixels). `useShown` says whether the panel shows at all.
 */
export function lazyPanel<M, P extends object>(
  chunk: LazyChunk<M>,
  pick: (module: M) => ComponentType<P>,
  panel: { title: string; height: number; className?: string; useShown?: (props: P) => boolean },
): ComponentType<P> {
  const useShown = panel.useShown ?? (() => true);
  return lazyPart<M, P>(chunk, pick, function PanelFallback({ failed, props }) {
    if (!useShown(props)) {
      return null;
    }
    return (
      <section
        className={panel.className === undefined ? "panel lazy-panel" : `panel lazy-panel ${panel.className}`}
        aria-label={panel.title}
        aria-busy={!failed}
        data-lazy-panel={chunk.name}
        style={{ minHeight: panel.height }}
      >
        <h2>{panel.title}</h2>
        {failed ? <ChunkFailed chunk={chunk} what="This panel" /> : <p className="muted small lazy-note">Loading…</p>}
      </section>
    );
  });
}

/** A dialog of `chunk` that shows only while `useOpen` says so; opening it loads the chunk. */
export function lazyDialog<M, P extends object>(
  chunk: LazyChunk<M>,
  pick: (module: M) => ComponentType<P>,
  dialog: { title: string; useOpen: (props: P) => { open: boolean; close: () => void } },
): ComponentType<P> {
  return lazyPart<M, P>(chunk, pick, function DialogFallback({ failed, props }) {
    const { open, close } = dialog.useOpen(props);
    useEffect(() => {
      if (open) {
        void chunk.load();
      }
    }, [open]);
    if (!open) {
      return null;
    }
    return (
      <section
        className="create-dialog lazy-panel"
        role="dialog"
        aria-modal="false"
        aria-label={dialog.title}
        aria-busy={!failed}
        data-lazy-dialog={chunk.name}
        onKeyDown={(e) => {
          if (e.key === "Escape") {
            e.stopPropagation();
            close();
          }
        }}
      >
        <header className="panel-head">
          <h2>{dialog.title}</h2>
          <button type="button" className="icon" aria-label={`Close the ${dialog.title} dialog`} onClick={close}>
            ×
          </button>
        </header>
        {failed ? <ChunkFailed chunk={chunk} what="This dialog" /> : <p className="muted small lazy-note">Loading…</p>}
      </section>
    );
  });
}

/** What a chunk that did not load shows: Retry, and a reload for a page older than the server. */
function ChunkFailed(props: { chunk: LazyChunk; what: string }) {
  const { chunk, what } = props;
  return (
    <div role="alert">
      <p className="error small">{what} did not load. Check the connection and try again.</p>
      <div className="op-row">
        <button type="button" data-action="chunk-retry" onClick={() => void chunk.load()}>
          Retry
        </button>
        <button
          type="button"
          data-action="chunk-reload"
          title="Loads the page again, in case it was updated meanwhile: the open meshes go, so save the project first"
          onClick={() => location.reload()}
        >
          Reload page
        </button>
      </div>
    </div>
  );
}
