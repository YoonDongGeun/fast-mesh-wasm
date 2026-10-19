// The side panel: one block per mesh (colouring toggle, opacity, visibility),
// the section, undo/redo and the scene's status. Controlled inputs read
// `desired` (what was last requested); `status === "computing"` covers the
// window until the scene has applied it.
import type { SectionHandle } from "fast-mesh-wasm";
import {
  useHistory,
  useMeshDisplay,
  useScene,
  useSceneStatus,
  useSection,
  useSelection,
} from "fast-mesh-wasm/react";
import { coloringTo, type Item, SECTION_NORMAL } from "./items";

export function Panel(props: { items: Item[]; section: SectionHandle; note: string }) {
  const { items, section, note } = props;
  return (
    <div className="panel">
      <StatusBlock />
      {items.map((item) => (
        <MeshBlock key={item.id} item={item} other={items.find((o) => o !== item) ?? null} />
      ))}
      <SectionBlock section={section} items={items} />
      <HistoryBlock />
      {note !== "" && <p className="error">{note}</p>}
      <p className="hint">
        Click a mesh to select it and drag its gizmo (<kbd>T</kbd> move, <kbd>R</kbd> rotate, <kbd>Esc</kbd>{" "}
        deselect). Drag empty space to orbit. Drop a binary STL to replace the selected mesh (or the first).
      </p>
    </div>
  );
}

function StatusBlock() {
  const store = useScene();
  const { status, error } = useSceneStatus();
  return (
    <section>
      <h2>Status</h2>
      <div className="row">
        <span>
          {store.mode === "worker" ? "Worker scene" : "Local scene"}: <span className={status}>{status}</span>
        </span>
        {status === "error" && store.mode === "worker" && (
          <button onClick={() => void store.restart().catch((e: unknown) => console.error(e))}>Restart</button>
        )}
      </div>
      {error !== undefined && <div className="error">{error.message}</div>}
    </section>
  );
}

function MeshBlock(props: { item: Item; other: Item | null }) {
  const { item, other } = props;
  const [state, setDisplay] = useMeshDisplay(item.mesh);
  const [selection] = useSelection();
  // Pressed is what was requested; busy covers the wait until it is shown.
  const pressed = state.desired.coloring !== null;
  const busy = state.status === "computing";
  const toggle = () => setDisplay({ coloring: pressed || other === null ? null : coloringTo(other.mesh) });
  return (
    <section>
      <h2 className={selection === item.mesh ? "selected" : undefined}>{item.name}</h2>
      <div className="row">
        <button aria-pressed={pressed} aria-busy={busy} disabled={other === null && !pressed} onClick={toggle}>
          Colour by distance{other === null ? "" : ` to ${other.name}`}
        </button>
        {busy && (
          <span className="busy" role="status">
            computing…
          </span>
        )}
        {state.status === "error" && (
          <span className="error" title={state.error?.message}>
            error
          </span>
        )}
      </div>
      <label className="row">
        Opacity
        <input
          type="range"
          min={0}
          max={1}
          step={0.05}
          value={state.desired.opacity}
          onChange={(e) => setDisplay({ opacity: Number(e.target.value) })}
        />
        <output>{state.desired.opacity.toFixed(2)}</output>
      </label>
      <label className="row">
        <input
          type="checkbox"
          checked={state.desired.visible}
          onChange={(e) => setDisplay({ visible: e.target.checked })}
        />
        Visible
      </label>
    </section>
  );
}

function SectionBlock(props: { section: SectionHandle; items: Item[] }) {
  const { section, items } = props;
  const [state, setSection] = useSection(section);
  const pressed = state.desired.meshes.length > 0;
  const busy = state.status === "computing";
  const offset = state.desired.plane.origin[2];
  return (
    <section>
      <h2>Section</h2>
      <div className="row">
        <button
          aria-pressed={pressed}
          aria-busy={busy}
          onClick={() => setSection({ meshes: pressed ? [] : items.map((i) => i.mesh) })}
        >
          Cut at z
        </button>
        {busy && (
          <span className="busy" role="status">
            computing…
          </span>
        )}
        {state.status === "error" && (
          <span className="error" title={state.error?.message}>
            error
          </span>
        )}
      </div>
      <label className="row">
        Offset
        <input
          type="range"
          min={-2}
          max={2}
          step={0.05}
          value={offset}
          onChange={(e) => setSection({ plane: { origin: [0, 0, Number(e.target.value)], normal: SECTION_NORMAL } })}
        />
        <output>{offset.toFixed(2)}</output>
      </label>
    </section>
  );
}

function HistoryBlock() {
  const { canUndo, canRedo, undo, redo } = useHistory();
  return (
    <section>
      <h2>History</h2>
      <div className="row">
        <button disabled={!canUndo} onClick={undo}>
          Undo
        </button>
        <button disabled={!canRedo} onClick={redo}>
          Redo
        </button>
        <span className="hint">
          <kbd>Ctrl</kbd>+<kbd>Z</kbd> / <kbd>Ctrl</kbd>+<kbd>Shift</kbd>+<kbd>Z</kbd>
        </span>
      </div>
    </section>
  );
}
