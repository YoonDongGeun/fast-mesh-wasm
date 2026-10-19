// @vitest-environment jsdom
// React hooks (`fast-mesh-wasm/react`) over a local `SceneStore` (the
// synchronous `Scene`), rendered with react-dom in jsdom.
import { existsSync, readFileSync } from "node:fs";
import { URL as NodeURL } from "node:url";
import { act, cleanup, fireEvent, render, screen } from "@testing-library/react";
import { StrictMode, startTransition, useEffect, useLayoutEffect, useState } from "react";
import { renderToString } from "react-dom/server";
import { afterEach, beforeAll, describe, expect, it, vi } from "vitest";
import { init, type MeshHandle, Scene, type SectionPlane } from "../src/web.js";
import {
  SceneProvider,
  useHistory,
  useMeshDisplay,
  useScene,
  useSceneStatus,
  useSection,
  useSelection,
  useStoreSelector,
} from "../src/react/index.js";
import { createSceneStore, type DistanceColoring, type SceneStore } from "../src/store/index.js";

// jsdom's global `URL` resolves a relative URL against a Windows `file:`
// base to `http:`, which breaks the Node build's wasm load on import
// (`readFileSync(new URL(..., import.meta.url))`). So the hooks run over the
// same `Scene` from the web entry, initialised with the .wasm bytes read here.
beforeAll(() => init(readFileSync(new NodeURL("../src/wasm/fast_mesh_wasm_bg.wasm", import.meta.url))));

// React's act() environment flag, so state updates inside act() are not warned about.
(globalThis as { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

const IDENTITY = [1, 0, 0, 0, 1, 0, 0, 0, 1];

/** A closed tetrahedron offset by `dx` along x. */
function tetra(dx = 0) {
  const positions = Float64Array.from([dx, 0, 0, dx + 1, 0, 0, dx, 1, 0, dx, 0, 1]);
  const indices = Uint32Array.from([0, 2, 1, 0, 1, 3, 0, 3, 2, 1, 2, 3]);
  return { positions, indices };
}

function ruleTo(to: MeshHandle): DistanceColoring {
  return { distanceTo: to, maxDistance: 5, map: "turbo", range: [0, 2] };
}

const disposers: (() => void)[] = [];

afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
  for (const d of disposers.splice(0)) {
    d();
  }
});

/**
 * A local store with meshes `a` and `b`, already in frame-loop mode (one
 * `commitFrame()` done), so declarative changes wait for the next frame.
 */
function setup(): { store: SceneStore; a: MeshHandle; b: MeshHandle } {
  const scene = new Scene();
  const store = createSceneStore(scene);
  disposers.push(() => {
    store.dispose();
    scene.free();
  });
  const t = tetra();
  const a = store.addMesh(t.positions, t.indices);
  const u = tetra(3);
  const b = store.addMesh(u.positions, u.indices);
  store.commitFrame();
  return { store, a, b };
}

/** A colouring toggle: pressed while colouring is requested, busy while computing. */
function Toggle({ mesh, to }: { mesh: MeshHandle; to?: MeshHandle }) {
  const [state, set] = useMeshDisplay(mesh);
  return (
    <button
      type="button"
      aria-pressed={state.desired.coloring !== null}
      aria-busy={state.status === "computing"}
      data-applied={String(state.coloring !== null)}
      data-opacity={state.opacity}
      data-status={state.status}
      onClick={() => set({ coloring: state.desired.coloring === null && to !== undefined ? ruleTo(to) : null })}
    >
      colour
    </button>
  );
}

function HistoryButtons({ label = "" }: { label?: string }) {
  const h = useHistory();
  return (
    <>
      <button type="button" disabled={!h.canUndo} onClick={h.undo}>
        {`Undo${label}`}
      </button>
      <button type="button" disabled={!h.canRedo} onClick={h.redo}>
        {`Redo${label}`}
      </button>
    </>
  );
}

describe("fast-mesh-wasm/react", () => {
  it("toggle_button_pressed_and_busy", () => {
    const { store, a, b } = setup();
    render(
      <SceneProvider store={store}>
        <Toggle mesh={a} to={b} />
      </SceneProvider>,
    );
    const button = screen.getByRole("button");
    expect(button.getAttribute("aria-pressed")).toBe("false");
    fireEvent.click(button);
    // Pressed at once (desired), not yet applied: the documented pending state.
    expect(button.getAttribute("aria-pressed")).toBe("true");
    expect(button.dataset.applied).toBe("false");
    expect(button.getAttribute("aria-busy")).toBe("true");
    expect(button.dataset.status).toBe("computing");
    act(() => {
      store.commitFrame();
    });
    expect(button.getAttribute("aria-pressed")).toBe("true");
    expect(button.dataset.applied).toBe("true");
    expect(button.dataset.status).toBe("ready");
    expect(button.getAttribute("aria-busy")).toBe("false");
    expect(store.getSnapshot().meshes.get(a)?.display.coloring).toEqual(ruleTo(b));
  });

  it("undo_redo_enable", () => {
    const { store, a } = setup();
    render(
      <SceneProvider store={store}>
        <HistoryButtons />
      </SceneProvider>,
    );
    const undo = screen.getByText("Undo") as HTMLButtonElement;
    const redo = screen.getByText("Redo") as HTMLButtonElement;
    expect(undo.disabled).toBe(true);
    expect(redo.disabled).toBe(true);
    act(() => {
      store.setPlacement(a, { rotation: IDENTITY, translation: [1, 0, 0] });
      store.commitFrame();
    });
    expect(undo.disabled).toBe(false);
    expect(redo.disabled).toBe(true);
    fireEvent.click(undo);
    act(() => {
      store.commitFrame();
    });
    expect(redo.disabled).toBe(false);
    expect(undo.disabled).toBe(true);
  });

  it("opacity_slider", () => {
    const { store, a } = setup();
    function Slider() {
      const [state, set] = useMeshDisplay(a);
      return (
        <input
          type="range"
          min={0}
          max={1}
          step={0.1}
          value={state.opacity}
          onChange={(e) => set({ opacity: Number(e.currentTarget.value) })}
        />
      );
    }
    render(
      <SceneProvider store={store}>
        <Slider />
      </SceneProvider>,
    );
    const input = screen.getByRole("slider") as HTMLInputElement;
    expect(input.value).toBe("1");
    act(() => {
      store.setDisplay(a, { opacity: 0.3 });
      store.commitFrame();
    });
    expect(input.value).toBe("0.3");
  });

  it("external_change_updates_all", () => {
    const { store, a } = setup();
    render(
      <SceneProvider store={store}>
        <HistoryButtons label="1" />
        <HistoryButtons label="2" />
      </SceneProvider>,
    );
    act(() => {
      store.setPlacement(a, { rotation: IDENTITY, translation: [1, 0, 0] });
      store.commitFrame();
    });
    expect((screen.getByText("Redo1") as HTMLButtonElement).disabled).toBe(true);
    expect((screen.getByText("Redo2") as HTMLButtonElement).disabled).toBe(true);
    // A keyboard shortcut handler, outside React.
    act(() => {
      store.undo();
      store.commitFrame();
    });
    for (const label of ["1", "2"]) {
      expect((screen.getByText(`Undo${label}`) as HTMLButtonElement).disabled).toBe(true);
      expect((screen.getByText(`Redo${label}`) as HTMLButtonElement).disabled).toBe(false);
    }
  });

  it("rerender_only_on_slice_change", () => {
    const { store, a, b } = setup();
    let renders = 0;
    function Counted() {
      const [state] = useMeshDisplay(b);
      renders++;
      return <span data-testid="b">{state.opacity}</span>;
    }
    render(
      <SceneProvider store={store}>
        <Counted />
      </SceneProvider>,
    );
    const base = renders;
    act(() => {
      store.setDisplay(a, { opacity: 0.5 });
    });
    act(() => {
      store.commitFrame();
    });
    act(() => {
      store.setPlacement(a, { rotation: IDENTITY, translation: [1, 0, 0] });
      store.commitFrame();
    });
    expect(renders).toBe(base);
    act(() => {
      store.setDisplay(b, { opacity: 0.5 });
      store.commitFrame();
    });
    expect(renders).toBeGreaterThan(base);
    expect(screen.getByTestId("b").textContent).toBe("0.5");
  });

  it("no_tearing_with_transition", () => {
    const { store, a } = setup();
    const commits: number[][] = [];
    let current: number[] = [];
    let mutateDuringRender = false;
    function Reader({ id }: { id: number }) {
      const [state] = useMeshDisplay(a);
      current[id] = state.revision;
      if (id === 0 && mutateDuringRender) {
        // The store changes between the two readers' renders (as a frame
        // or a Worker result could while a transition yields).
        mutateDuringRender = false;
        store.setDisplay(a, { opacity: 0.25 });
      }
      return <span>{state.revision}</span>;
    }
    function Commit() {
      useLayoutEffect(() => {
        commits.push([...current]);
        current = [];
      });
      return null;
    }
    let bump: () => void = () => {};
    function App() {
      const [n, setN] = useState(0);
      bump = () => setN((v) => v + 1);
      return (
        <>
          <Reader id={0} key={`0-${n}`} />
          <Reader id={1} key={`1-${n}`} />
          <Commit />
        </>
      );
    }
    render(
      <SceneProvider store={store}>
        <App />
      </SceneProvider>,
    );
    const errors = vi.spyOn(console, "error").mockImplementation(() => {});
    act(() => {
      store.setDisplay(a, { opacity: 0.5 });
      mutateDuringRender = true;
      startTransition(() => bump());
    });
    const logged = errors.mock.calls.map((c) => String(c[0]));
    errors.mockRestore();
    // The mutation happened mid-render, and React reported nothing but
    // (at most) its render-phase update warning.
    expect(mutateDuringRender).toBe(false);
    expect(store.getSnapshot().meshes.get(a)?.desired.opacity).toBe(0.25);
    expect(logged.filter((m) => !/while rendering a different component/.test(m))).toEqual([]);
    expect(commits.length).toBeGreaterThan(1);
    for (const c of commits) {
      expect(c[0]).toBe(c[1]);
    }
    // The last commit shows the latest state.
    expect(commits.at(-1)?.[0]).toBe(store.getSnapshot().meshes.get(a)?.revision);
  });

  it("hook_survives_removed_mesh", () => {
    const { store, a, b } = setup();
    store.select(a);
    function Selected() {
      const [sel] = useSelection();
      return <span data-testid="sel">{sel === null ? "none" : sel === a ? "a" : "other"}</span>;
    }
    render(
      <SceneProvider store={store}>
        <Toggle mesh={a} to={b} />
        <Selected />
      </SceneProvider>,
    );
    expect(screen.getByTestId("sel").textContent).toBe("a");
    act(() => {
      store.setDisplay(a, { coloring: ruleTo(b), opacity: 0.5 });
      store.commitFrame();
    });
    const button = screen.getByRole("button");
    expect(button.dataset.applied).toBe("true");
    expect(button.dataset.opacity).toBe("0.5");
    act(() => {
      store.removeMesh(a);
      store.commitFrame();
    });
    expect(button.dataset.status).toBe("idle");
    expect(button.dataset.applied).toBe("false");
    expect(button.dataset.opacity).toBe("1");
    expect(button.getAttribute("aria-pressed")).toBe("false");
    expect(screen.getByTestId("sel").textContent).toBe("none");
  });

  it("ssr_does_not_crash", () => {
    const { store, a, b } = setup();
    store.setDisplay(a, { coloring: ruleTo(b) });
    store.commitFrame();
    const html = renderToString(
      <SceneProvider store={store}>
        <Toggle mesh={a} to={b} />
        <HistoryButtons />
      </SceneProvider>,
    );
    expect(typeof html).toBe("string");
    // The server renders the empty snapshot: nothing pressed, history off.
    expect(html).toMatch(/aria-pressed="false"/);
    expect(html).toMatch(/data-status="idle"/);
  });

  it("strict_mode_idempotent", () => {
    const { store, a, b } = setup();
    const add = vi.spyOn(Scene.prototype, "addDistanceField");
    let renders = 0;
    function Colour() {
      const [, set] = useMeshDisplay(a);
      renders++;
      useEffect(() => {
        // An equal, not identical, rule on each (doubled) effect run.
        set({ coloring: ruleTo(b) });
      }, [set]);
      return null;
    }
    render(
      <StrictMode>
        <SceneProvider store={store}>
          <Colour />
        </SceneProvider>
      </StrictMode>,
    );
    act(() => {
      store.commitFrame();
    });
    expect(renders).toBeGreaterThan(1);
    expect(add).toHaveBeenCalledTimes(1);
    expect(store.getSnapshot().meshes.get(a)?.display.coloring).toEqual(ruleTo(b));
  });

  it("selector_new_objects_stable", () => {
    const { store, a, b } = setup();
    let renders = 0;
    let seen: unknown;
    function Pairs() {
      // A new object on every call: cached per snapshot, kept while equal.
      const pair = useStoreSelector(
        (s) => ({ a: s.meshes.get(a)?.desired.opacity, n: s.meshes.size }),
        (x, y) => x.a === y.a && x.n === y.n,
      );
      const status = useSceneStatus();
      renders++;
      seen = pair;
      return <span data-testid="pair">{`${pair.a}/${pair.n}/${status.status}`}</span>;
    }
    const errors = vi.spyOn(console, "error");
    render(
      <SceneProvider store={store}>
        <Pairs />
      </SceneProvider>,
    );
    const first = seen;
    const base = renders;
    act(() => {
      store.setDisplay(b, { opacity: 0.5 });
      store.commitFrame();
    });
    expect(renders).toBe(base);
    expect(seen).toBe(first);
    act(() => {
      store.setDisplay(a, { opacity: 0.5 });
      store.commitFrame();
    });
    expect(screen.getByTestId("pair").textContent).toBe("0.5/2/idle");
    expect(errors).not.toHaveBeenCalled();
  });

  it("section_hook", () => {
    const { store, a, b } = setup();
    const P1: SectionPlane = { origin: [0, 0, 0.2], normal: [0, 0, 1] };
    const P2: SectionPlane = { origin: [0, 0, 0.3], normal: [0, 0, 1] };
    const section = store.addSection({ plane: P1, meshes: [a] });
    let setter: ((p: { plane?: SectionPlane; meshes?: readonly MeshHandle[]; fill?: boolean }) => void) | undefined;
    const setters = new Set<unknown>();
    function Section() {
      const [state, set] = useSection(section);
      setter = set;
      setters.add(set);
      return <span data-testid="sec">{`${state.plane.origin[2]}/${state.desired.plane.origin[2]}/${state.meshes.length}/${state.status}`}</span>;
    }
    render(
      <SceneProvider store={store}>
        <Section />
      </SceneProvider>,
    );
    act(() => {
      store.commitFrame();
    });
    expect(screen.getByTestId("sec").textContent).toBe("0.2/0.2/1/ready");
    act(() => setter?.({ plane: P2, meshes: [a, b] }));
    // The documented pending state until the flush.
    expect(screen.getByTestId("sec").textContent).toBe("0.2/0.3/1/computing");
    act(() => {
      store.commitFrame();
    });
    expect(screen.getByTestId("sec").textContent).toBe("0.3/0.3/2/ready");
    // Stable setter across renders.
    expect(setters.size).toBe(1);
    act(() => {
      store.removeSection(section);
    });
    expect(screen.getByTestId("sec").textContent).toBe("0/0/0/idle");
  });

  it("history_controls_stable", () => {
    const { store, a } = setup();
    const seen: ReturnType<typeof useHistory>[] = [];
    function Spy() {
      seen.push(useHistory());
      return null;
    }
    render(
      <SceneProvider store={store}>
        <Spy />
      </SceneProvider>,
    );
    act(() => {
      store.setPlacement(a, { rotation: IDENTITY, translation: [1, 0, 0] });
      store.commitFrame();
    });
    const first = seen[0];
    const last = seen.at(-1)!;
    expect(last.canUndo).toBe(true);
    expect(last).not.toBe(first);
    // The actions keep their identity across a history change.
    expect(last.undo).toBe(first.undo);
    expect(last.redo).toBe(first.redo);
  });

  it("inline_selector_keeps_identity", () => {
    const { store, a } = setup();
    const seen: unknown[] = [];
    let selections = 0;
    let rerender: () => void = () => {};
    function Parent() {
      const [, setN] = useState(0);
      rerender = () => setN((n) => n + 1);
      // Inline selector and equality: new functions on every render.
      const view = useStoreSelector(
        (s) => {
          selections++;
          return { opacity: s.meshes.get(a)?.display.opacity };
        },
        (x, y) => x.opacity === y.opacity,
      );
      seen.push(view);
      return null;
    }
    render(
      <SceneProvider store={store}>
        <Parent />
      </SceneProvider>,
    );
    act(() => rerender());
    act(() => rerender());
    expect(seen.length).toBeGreaterThanOrEqual(3);
    for (const v of seen) {
      expect(v).toBe(seen[0]);
    }
    expect(selections).toBeGreaterThan(0);
    act(() => {
      store.setDisplay(a, { opacity: 0.5 });
      store.commitFrame();
    });
    expect(seen.at(-1)).toEqual({ opacity: 0.5 });
  });

  it("use_scene_outside_provider_throws", () => {
    function Bare() {
      useScene();
      return null;
    }
    const errors = vi.spyOn(console, "error").mockImplementation(() => {});
    expect(() => render(<Bare />)).toThrow(expect.objectContaining({ code: "INVALID_ARGUMENT" }));
    errors.mockRestore();
  });

  it("package_react_export", () => {
    const dist = new NodeURL("../dist/", import.meta.url);
    expect(existsSync(new NodeURL("react.js", dist))).toBe(true);
    expect(existsSync(new NodeURL("react.d.ts", dist))).toBe(true);
    const pkg = JSON.parse(readFileSync(new NodeURL("../package.json", import.meta.url), "utf8"));
    expect(pkg.exports["./react"]).toEqual({ types: "./dist/react.d.ts", default: "./dist/react.js" });
    expect(pkg.peerDependencies.react).toBe(">=18");
    expect(pkg.peerDependenciesMeta.react).toEqual({ optional: true });

    /** Every dist file a file imports, transitively. */
    const graph = (file: string, seen = new Set<string>()): Set<string> => {
      if (seen.has(file)) {
        return seen;
      }
      seen.add(file);
      const src = readFileSync(new NodeURL(file, dist), "utf8");
      for (const m of src.matchAll(/(?:import|export)\s[^"']*?from\s*["']([^"']+)["']|import\s*["']([^"']+)["']/g)) {
        const spec = m[1] ?? m[2];
        if (spec.startsWith("./")) {
          graph(new NodeURL(spec, new NodeURL(file, dist)).href.slice(dist.href.length), seen);
        } else {
          seen.add(spec);
        }
      }
      return seen;
    };
    const files = [...graph("react.js")];
    expect(files).toContain("react");
    expect(files.filter((f) => /three|@react-three|fast_mesh_wasm\.js|^wasm\//.test(f))).toEqual([]);
    // It shares the store's code (one SceneStore, one EMPTY_SNAPSHOT), never a copy.
    const store = graph("store.js");
    expect(files.filter((f) => f !== "react.js" && f.endsWith(".js") && store.has(f)).length).toBeGreaterThan(0);
    for (const f of files.filter((f) => f.endsWith(".js"))) {
      expect(readFileSync(new NodeURL(f, dist), "utf8"), f).not.toMatch(/\bimport\(/);
      if (f !== "react.js") {
        expect(readFileSync(new NodeURL(f, dist), "utf8"), f).not.toMatch(/from\s*["']react["']/);
      }
    }
  });
});
