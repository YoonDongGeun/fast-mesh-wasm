// @vitest-environment jsdom
// Parts that come from a chunk of their own (#285): what shows until they are
// here, a chunk that does not load, and the commands and the tour that wait for
// a panel's controls.
import { act, createElement, type ReactNode, useEffect, useState, useSyncExternalStore } from "react";
import { createRoot } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { lazyDialog, lazyPanel, lazyPart } from "../src/shared/ui/LazyPart";
import { LazyChunk } from "../src/shared/ux/lazy-chunk";
import { TourCard } from "../src/shell/ui/Help";
import { focusElement, triggerClick, whenPresent } from "../src/shell/ux/panel-triggers";
import { TOUR_STEPS } from "../src/shell/ux/tour";

(globalThis as { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

const cleanups: (() => void)[] = [];

beforeEach(() => {
  // jsdom has no layout to scroll.
  Element.prototype.scrollIntoView = vi.fn();
});

afterEach(() => {
  for (const fn of cleanups.splice(0).reverse()) {
    fn();
  }
  vi.useRealTimers();
  vi.restoreAllMocks();
});

function mount(node: ReactNode): HTMLElement {
  const host = document.createElement("div");
  document.body.append(host);
  const root = createRoot(host);
  act(() => root.render(node));
  cleanups.push(() => {
    act(() => root.unmount());
    host.remove();
  });
  return host;
}

/** A chunk whose import the test settles, any number of times. */
function gated<M>(name = "test") {
  const settlers: { resolve: (m: M) => void; reject: (e: unknown) => void }[] = [];
  const importer = vi.fn(() => new Promise<M>((resolve, reject) => settlers.push({ resolve, reject })));
  const chunk = new LazyChunk<M>(name, importer);
  return {
    chunk,
    importer,
    arrive: async (m: M) => {
      await act(async () => {
        settlers.at(-1)!.resolve(m);
      });
    },
    fail: async () => {
      vi.spyOn(console, "warn").mockImplementation(() => {});
      await act(async () => {
        settlers.at(-1)!.reject(new Error("offline"));
      });
    },
  };
}

type Props = { label: string };
const Panel = (props: Props) => (
  <section className="panel" data-panel="real">
    <h2>{props.label}</h2>
    <button type="button" data-run>
      Run
    </button>
  </section>
);

describe("lazyPart", () => {
  it("draws nothing until its chunk is here, then the part, and leaves its siblings alone", async () => {
    const { chunk, arrive } = gated<{ Panel: typeof Panel }>();
    const Part = lazyPart(chunk, (m) => m.Panel);
    const mounts = vi.fn();
    const unmounts = vi.fn();
    const Sibling = () => {
      useEffect(() => {
        mounts();
        return unmounts;
      }, []);
      return <i data-sibling />;
    };
    const host = mount([<Sibling key="a" />, <Part key="b" label="one" />, <Sibling key="c" />]);
    const before = [...host.querySelectorAll("[data-sibling]")];
    expect(host.querySelector("[data-panel]")).toBeNull();

    act(() => void chunk.load());
    expect(host.querySelector("[data-panel]")).toBeNull();
    await arrive({ Panel });
    expect(host.querySelector("[data-panel] h2")?.textContent).toBe("one");
    expect([...host.querySelectorAll("[data-sibling]")]).toEqual(before);
    expect(mounts).toHaveBeenCalledTimes(2);
    expect(unmounts).not.toHaveBeenCalled();
  });

  it("renders in the same pass once the chunk is here", async () => {
    const { chunk, arrive } = gated<{ Panel: typeof Panel }>();
    const Part = lazyPart(chunk, (m) => m.Panel, () => <p data-fallback />);
    const loaded = chunk.load();
    await arrive({ Panel });
    await loaded;
    const shown: string[] = [];
    const Watch = () => {
      const [, set] = useState(0);
      useEffect(() => set(1), []);
      shown.push(document.querySelector("[data-fallback]") === null ? "part" : "fallback");
      return null;
    };
    const host = mount([<Part key="p" label="x" />, <Watch key="w" />]);
    expect(host.querySelector("[data-panel=real]")).not.toBeNull();
    expect(host.querySelector("[data-fallback]")).toBeNull();
    expect(shown).not.toContain("fallback");
  });

  it("keeps the part's state while its props change", async () => {
    const { chunk, arrive } = gated<{ Counter: (p: { by: number }) => ReactNode }>();
    const Counter = (p: { by: number }) => {
      const [n, set] = useState(0);
      return (
        <button type="button" onClick={() => set(n + p.by)}>
          {n}
        </button>
      );
    };
    const Part = lazyPart(chunk, (m) => m.Counter);
    const host = document.createElement("div");
    document.body.append(host);
    const root = createRoot(host);
    cleanups.push(() => {
      act(() => root.unmount());
      host.remove();
    });
    act(() => root.render(<Part by={1} />));
    act(() => void chunk.load());
    await arrive({ Counter });
    act(() => host.querySelector("button")!.click());
    act(() => root.render(<Part by={5} />));
    act(() => host.querySelector("button")!.click());
    expect(host.textContent).toBe("6");
  });
});

describe("lazyPanel", () => {
  it("holds the panel's place under its name, and gives way to it", async () => {
    const { chunk, arrive } = gated<{ Panel: typeof Panel }>("repair");
    const Lazy = lazyPanel(chunk, (m) => m.Panel, { title: "Repair", height: 320, className: "extra" });
    const host = mount(<Lazy label="Repair" />);
    const place = host.querySelector<HTMLElement>("[data-lazy-panel=repair]")!;
    expect(place.className).toBe("panel lazy-panel extra");
    expect(place.getAttribute("aria-label")).toBe("Repair");
    expect(place.getAttribute("aria-busy")).toBe("true");
    expect(place.style.minHeight).toBe("320px");
    expect(place.querySelector("h2")?.textContent).toBe("Repair");
    expect(place.querySelector("button")).toBeNull();

    act(() => void chunk.load());
    await arrive({ Panel });
    expect(host.querySelector("[data-lazy-panel]")).toBeNull();
    expect(host.querySelector("[data-panel=real]")).not.toBeNull();
  });

  it("shows nothing while the panel would not", () => {
    const { chunk } = gated<{ Panel: typeof Panel }>();
    const Lazy = lazyPanel(chunk, (m) => m.Panel, { title: "Volume", height: 10, useShown: ({ label }) => label === "volume" });
    const host = mount([<Lazy key="a" label="mesh" />, <Lazy key="b" label="volume" />]);
    expect(host.querySelectorAll("[data-lazy-panel]").length).toBe(1);
  });

  it("offers Retry when the chunk does not load, and shows the panel once it does", async () => {
    const { chunk, importer, arrive, fail } = gated<{ Panel: typeof Panel }>("edit");
    const Lazy = lazyPanel(chunk, (m) => m.Panel, { title: "Edit", height: 100 });
    const host = mount(<Lazy label="Edit" />);
    act(() => void chunk.load());
    await fail();
    const place = host.querySelector("[data-lazy-panel=edit]")!;
    expect(place.getAttribute("aria-busy")).toBe("false");
    expect(place.querySelector("[role=alert]")?.textContent).toMatch(/did not load/);
    expect(place.querySelector("[data-action=chunk-reload]")).not.toBeNull();

    await act(async () => place.querySelector<HTMLButtonElement>("[data-action=chunk-retry]")!.click());
    expect(importer).toHaveBeenCalledTimes(2);
    expect(host.querySelector("[role=alert]")).toBeNull();
    await arrive({ Panel });
    expect(host.querySelector("[data-panel=real]")).not.toBeNull();
  });
});

describe("lazyDialog", () => {
  function setup() {
    const gate = gated<{ Dialog: (p: { state: State }) => ReactNode }>("home");
    interface State {
      open: boolean;
      listeners: Set<() => void>;
    }
    const state: State = { open: false, listeners: new Set() };
    const set = (open: boolean) => {
      state.open = open;
      for (const l of [...state.listeners]) l();
    };
    const useOpenState = (s: State) =>
      useSyncExternalStore(
        (l) => {
          s.listeners.add(l);
          return () => s.listeners.delete(l);
        },
        () => s.open,
      );
    const Dialog = (p: { state: State }) => (useOpenState(p.state) ? <section data-panel="real-dialog" /> : null);
    const Lazy = lazyDialog(gate.chunk, (m) => m.Dialog, {
      title: "Create shape",
      useOpen: (p: { state: State }) => ({ open: useOpenState(p.state), close: () => set(false) }),
    });
    const host = mount(<Lazy state={state} />);
    return { ...gate, Dialog, host, set, state };
  }

  it("loads nothing while it is closed, and its chunk when it opens", async () => {
    const { importer, arrive, Dialog, host, set } = setup();
    expect(importer).not.toHaveBeenCalled();
    expect(host.childElementCount).toBe(0);
    act(() => set(true));
    expect(importer).toHaveBeenCalledTimes(1);
    const place = host.querySelector("[data-lazy-dialog=home]")!;
    expect(place.getAttribute("role")).toBe("dialog");
    expect(place.getAttribute("aria-label")).toBe("Create shape");
    await arrive({ Dialog });
    expect(host.querySelector("[data-lazy-dialog]")).toBeNull();
    expect(host.querySelector("[data-panel=real-dialog]")).not.toBeNull();
  });

  it("closes from its button and from Esc, which goes no further", () => {
    const { host, set, state } = setup();
    const outside = vi.fn();
    document.body.addEventListener("keydown", outside);
    cleanups.push(() => document.body.removeEventListener("keydown", outside));
    act(() => set(true));
    act(() => host.querySelector<HTMLButtonElement>("[data-lazy-dialog] .icon")!.click());
    expect(state.open).toBe(false);
    act(() => set(true));
    act(() => {
      host
        .querySelector("[data-lazy-dialog] .icon")!
        .dispatchEvent(new KeyboardEvent("keydown", { key: "Escape", bubbles: true }));
    });
    expect(state.open).toBe(false);
    expect(outside).not.toHaveBeenCalled();
  });

  it("offers Retry when its chunk does not load", async () => {
    const { arrive, Dialog, fail, host, importer, set } = setup();
    act(() => set(true));
    await fail();
    await act(async () => host.querySelector<HTMLButtonElement>("[data-action=chunk-retry]")!.click());
    expect(importer).toHaveBeenCalledTimes(2);
    await arrive({ Dialog });
    expect(host.querySelector("[data-panel=real-dialog]")).not.toBeNull();
  });
});

describe("commands that press a panel's control", () => {
  it("press it at once when it is there, whatever its state, as ever", () => {
    const host = mount(<button type="button" id="here" disabled />);
    const act1 = vi.fn();
    whenPresent("#here", act1);
    expect(act1).toHaveBeenCalledWith(host.querySelector("#here"));
  });

  it("do nothing when no panel is on its way", () => {
    vi.useFakeTimers();
    const act1 = vi.fn();
    whenPresent("#nowhere", act1);
    expect(vi.getTimerCount()).toBe(0);
    expect(act1).not.toHaveBeenCalled();
  });

  it("wait for the panel, load it, and press its control once", async () => {
    vi.useFakeTimers();
    const { chunk, importer, arrive } = gated<{ Panel: (p: Props) => ReactNode }>();
    const clicked = vi.fn();
    const Clickable = () => <button type="button" data-run onClick={clicked} />;
    const Lazy = lazyPanel(chunk, (m) => m.Panel, { title: "Align", height: 10 });
    mount(<Lazy label="Align" />);
    triggerClick("[data-run]");
    expect(importer).toHaveBeenCalledTimes(1);
    await act(async () => vi.advanceTimersByTime(8000)); // however long the network takes
    expect(clicked).not.toHaveBeenCalled();
    await arrive({ Panel: Clickable });
    await act(async () => vi.advanceTimersByTime(200));
    expect(clicked).toHaveBeenCalledTimes(1);
    await act(async () => vi.advanceTimersByTime(10_000));
    expect(clicked).toHaveBeenCalledTimes(1);
    expect(vi.getTimerCount()).toBe(0);
  });

  it("wait until the control is there and enabled, for a while after the panel is", async () => {
    vi.useFakeTimers();
    const { chunk, arrive } = gated<{ Panel: (p: Props) => ReactNode }>();
    const clicked = vi.fn();
    const Analysed = () => {
      const [ready, setReady] = useState(false);
      useEffect(() => {
        const t = setTimeout(() => setReady(true), 1000);
        return () => clearTimeout(t);
      }, []);
      return <button type="button" data-run disabled={!ready} onClick={clicked} />;
    };
    const Lazy = lazyPanel(chunk, (m) => m.Panel, { title: "Repair", height: 10 });
    mount(<Lazy label="Repair" />);
    triggerClick("[data-run]");
    await arrive({ Panel: Analysed });
    await act(async () => vi.advanceTimersByTime(900));
    expect(clicked).not.toHaveBeenCalled();
    await act(async () => vi.advanceTimersByTime(300));
    await act(async () => vi.advanceTimersByTime(100));
    expect(clicked).toHaveBeenCalledTimes(1);

    // A control that never comes is given up.
    triggerClick("[data-other]");
    await act(async () => vi.advanceTimersByTime(10_000));
    expect(vi.getTimerCount()).toBe(0);
  });

  it("give way to the next command, and give up when the panel does not load", async () => {
    vi.useFakeTimers();
    const { chunk, arrive } = gated<{ Panel: (p: Props) => ReactNode }>();
    const clicked = vi.fn();
    const Two = () => (
      <>
        <button type="button" data-one onClick={() => clicked("one")} />
        <button type="button" data-two onClick={() => clicked("two")} />
      </>
    );
    const Lazy = lazyPanel(chunk, (m) => m.Panel, { title: "Align", height: 10 });
    mount(<Lazy label="Align" />);
    triggerClick("[data-one]");
    triggerClick("[data-two]");
    await arrive({ Panel: Two });
    await act(async () => vi.advanceTimersByTime(500));
    expect(clicked.mock.calls).toEqual([["two"]]);

    const broken = gated<{ Panel: (p: Props) => ReactNode }>();
    const Broken = lazyPanel(broken.chunk, (m) => m.Panel, { title: "Edit", height: 10 });
    mount(<Broken label="Edit" />);
    triggerClick("[data-three]");
    await broken.fail();
    await act(async () => vi.advanceTimersByTime(5000));
    expect(vi.getTimerCount()).toBe(0);
    // Retried much later, the old command is gone.
    const late = vi.fn();
    act(() => void broken.chunk.load());
    await broken.arrive({ Panel: () => <button type="button" data-three onClick={late} /> });
    await act(async () => vi.advanceTimersByTime(5000));
    expect(late).not.toHaveBeenCalled();
  });

  it("focus the panel once it is here", async () => {
    vi.useFakeTimers();
    const { chunk, arrive } = gated<{ Panel: (p: Props) => ReactNode }>();
    const Lazy = lazyPanel(chunk, (m) => m.Panel, { title: "Edit", height: 10 });
    const host = mount(<Lazy label="Edit" />);
    focusElement("[data-run]");
    await arrive({ Panel });
    await act(async () => vi.advanceTimersByTime(200));
    expect(document.activeElement).toBe(host.querySelector("[data-run]"));
    expect(Element.prototype.scrollIntoView).toHaveBeenCalled();
  });
});

describe("the tour", () => {
  const step = (target: string) => TOUR_STEPS.findIndex((s) => s.target === target);

  /** The tour on a page whose repair panel is still to come. */
  function tour() {
    vi.useFakeTimers();
    const { chunk, arrive } = gated<{ Panel: (p: Props) => ReactNode }>();
    const Lazy = lazyPanel(chunk, (m) => m.Panel, { title: "Repair", height: 10 });
    const host = document.createElement("div");
    document.body.append(host);
    const root = createRoot(host);
    cleanups.push(() => {
      act(() => root.unmount());
      host.remove();
    });
    const card = (i: number) => createElement(TourCard, { index: i, onNext: () => {}, onBack: () => {}, onDone: () => {} });
    const render = (target: string) =>
      act(() =>
        root.render(
          <>
            <div data-tour="gizmo" />
            <Lazy label="Repair" />
            {card(step(target))}
          </>,
        ),
      );
    return { arrive, host, render };
  }

  it("rings nothing for a step that was left before its panel came", async () => {
    const { arrive, host, render } = tour();
    render("repair");
    render("gizmo");
    await arrive({ Panel: () => <section data-tour="repair" /> });
    await act(async () => vi.advanceTimersByTime(5000));
    expect(host.querySelector("[data-tour=repair]")!.classList.contains("tour-spot")).toBe(false);
    expect(vi.getTimerCount()).toBe(0);
  });

  it("rings a panel that comes after its step, and takes the ring off with the step", async () => {
    const { arrive, host, render } = tour();
    render("repair");
    await act(async () => vi.advanceTimersByTime(1000));
    await arrive({ Panel: () => <section data-tour="repair" /> });
    await act(async () => vi.advanceTimersByTime(200));
    const anchor = host.querySelector("[data-tour=repair]")!;
    expect(anchor.classList.contains("tour-spot")).toBe(true);
    render("gizmo");
    expect(anchor.classList.contains("tour-spot")).toBe(false);
    expect(host.querySelector("[data-tour=gizmo]")!.classList.contains("tour-spot")).toBe(true);
  });
});
