// @vitest-environment jsdom
// The command palette's keyboard launcher with the whole shell mounted: the
// app's shortcuts and the palette both listen on the window, and Ctrl+K must
// still open the palette once (two toggles on one keypress cancel out).
import { act, createElement } from "react";
import { createRoot } from "react-dom/client";
import { afterEach, describe, expect, it } from "vitest";
import { AppState } from "../src/app/state/app-state";
import { useKeyboardShortcuts } from "../src/app/ux/use-keyboard-shortcuts";
import { CommandPalette } from "../src/shell/ui/CommandPalette";

(globalThis as { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

const cleanups: (() => void)[] = [];

afterEach(() => {
  for (const fn of cleanups.splice(0).reverse()) {
    fn();
  }
});

function Shell(props: { app: AppState }) {
  const { app } = props;
  const noop = () => {};
  useKeyboardShortcuts({
    app,
    select: noop,
    selection: null,
    undo: noop,
    redo: noop,
    focusOn: noop,
    setMode: noop,
    helpOpen: false,
    setHelpOpen: noop,
    tourStep: null,
    endTour: noop,
  });
  return createElement(CommandPalette, {
    app,
    items: [],
    selection: null,
    canUndo: false,
    canRedo: false,
    ortho: false,
    units: "mm",
    onSelectItem: noop,
    onSetMode: noop,
    onResetView: noop,
    onUndo: noop,
    onRedo: noop,
    onOpenHelp: noop,
  });
}

function mount(): AppState {
  const app = new AppState();
  const host = document.createElement("div");
  document.body.append(host);
  const root = createRoot(host);
  act(() => root.render(createElement(Shell, { app })));
  cleanups.push(() => {
    act(() => root.unmount());
    host.remove();
  });
  return app;
}

function press(init: KeyboardEventInit) {
  act(() => {
    window.dispatchEvent(new KeyboardEvent("keydown", { bubbles: true, cancelable: true, ...init }));
  });
}

describe("command palette launcher", () => {
  it("opens on Ctrl+K and closes on the next Ctrl+K", () => {
    const app = mount();
    press({ key: "k", ctrlKey: true });
    expect(app.getSnapshot().commandPaletteOpen).toBe(true);
    press({ key: "k", ctrlKey: true });
    expect(app.getSnapshot().commandPaletteOpen).toBe(false);
  });

  it("opens on Cmd+K", () => {
    const app = mount();
    press({ key: "k", metaKey: true });
    expect(app.getSnapshot().commandPaletteOpen).toBe(true);
  });

  it("opens on /", () => {
    const app = mount();
    press({ key: "/" });
    expect(app.getSnapshot().commandPaletteOpen).toBe(true);
  });
});
