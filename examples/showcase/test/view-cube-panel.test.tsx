// @vitest-environment jsdom
// The navigation cube box's wiring (#255): its arrows ask for orbit and roll
// steps (Shift for the fine orbit step), its faces and chips for presets and
// directions, its toggles flip the scale indicator, the axis basis and the
// projection (persisted with the other display settings), and it folds.
import { act, createElement } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { AppState } from "../src/app/state/app-state";
import { useKeyboardShortcuts } from "../src/app/ux/use-keyboard-shortcuts";
import { ViewportNavigationHud } from "../src/features/view/ui/ViewCube";
import { cameraPresetForKey, nextScaleIndicator, scaleGridBackground } from "../src/features/view/ux/navigation-hud";
import { defaultSettings, parseSettings, settingsFromSnapshot } from "../src/shared/ux/settings";

(globalThis as { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

let root: Root | null = null;
let container: HTMLDivElement | null = null;

beforeEach(() => {
  // The HUD's frame loop follows the viewport camera; none here.
  vi.stubGlobal("requestAnimationFrame", () => 0);
  vi.stubGlobal("cancelAnimationFrame", () => {});
});

afterEach(async () => {
  if (root !== null) {
    const r = root;
    await act(async () => r.unmount());
  }
  container?.remove();
  root = null;
  container = null;
  vi.unstubAllGlobals();
});

async function mount(app: AppState): Promise<HTMLDivElement> {
  const c = document.createElement("div");
  document.body.appendChild(c);
  container = c;
  await act(async () => {
    root = createRoot(c);
    root.render(createElement(ViewportNavigationHud, { app }));
  });
  return c;
}

async function click(el: Element | null, init: MouseEventInit = {}): Promise<void> {
  expect(el).not.toBeNull();
  await act(async () => {
    el!.dispatchEvent(new MouseEvent("click", { bubbles: true, ...init }));
  });
}

describe("navigation cube box", () => {
  it("asks_for_orbit_and_roll_steps_from_its_arrows", async () => {
    const app = new AppState();
    const c = await mount(app);
    expect(c.querySelectorAll("[data-view-orbit]").length).toBe(4);
    expect(c.querySelectorAll("[data-view-roll]").length).toBe(2);

    await click(c.querySelector('[data-view-orbit="right"]'));
    expect(app.getSnapshot().cameraStepRequest).toEqual({ kind: "orbit", direction: "right", deg: 90, n: 1 });
    await click(c.querySelector('[data-view-orbit="up"]'), { shiftKey: true });
    expect(app.getSnapshot().cameraStepRequest).toEqual({ kind: "orbit", direction: "up", deg: 15, n: 2 });
    await click(c.querySelector('[data-view-roll="cw"]'));
    expect(app.getSnapshot().cameraStepRequest).toEqual({ kind: "roll", direction: "cw", deg: 90, n: 3 });
    await click(c.querySelector('[data-view-roll="ccw"]'), { shiftKey: true });
    expect(app.getSnapshot().cameraStepRequest).toEqual({ kind: "roll", direction: "ccw", deg: 90, n: 4 });
  });

  it("keeps_the_face_edge_and_corner_views_inside_the_box", async () => {
    const app = new AppState();
    const c = await mount(app);
    const panel = c.querySelector(".view-cube-panel")!;
    expect(panel.querySelectorAll("[data-viewcube-face]").length).toBe(6);
    expect(panel.querySelector("[data-viewcube-corner]")).not.toBeNull();
    expect(panel.querySelector("[data-viewcube-edge]")).not.toBeNull();
    expect(panel.querySelector('[data-action="camera-projection"]')).not.toBeNull();
    expect(c.querySelector('[data-camera-preset="top"]')?.getAttribute("title")).toContain("(7)");

    await click(c.querySelector('[data-viewcube-face="left"]'));
    expect(app.getSnapshot().cameraPresetRequest).toEqual({ preset: "left", n: 1 });
    await click(c.querySelector('[data-viewcube-corner="top-front-right"]'));
    expect(app.getSnapshot().cameraDirectionRequest?.n).toBe(1);
  });

  it("flips_the_three_toggles_and_shows_what_they_say", async () => {
    const app = new AppState();
    const c = await mount(app);
    const scale = () => c.querySelector<HTMLButtonElement>('[data-action="view-scale"]')!;
    expect(scale().dataset.scaleIndicator).toBe("ruler");
    expect(c.querySelector(".scale-bar-rule")).not.toBeNull();
    expect(c.querySelector("[data-scale-grid]")).toBeNull();

    await click(scale());
    expect(app.getSnapshot().scaleIndicator).toBe("grid");
    expect(c.querySelector("[data-scale-grid]")).not.toBeNull();
    expect(c.querySelector(".scale-bar-label")?.textContent).toMatch(/^Grid /);
    expect(c.querySelector(".scale-bar-rule")).toBeNull();

    await click(scale());
    expect(app.getSnapshot().scaleIndicator).toBe("off");
    expect(scale().getAttribute("aria-pressed")).toBe("false");
    expect(c.querySelector("[data-scale-bar]")).toBeNull();
    expect(c.querySelector("[data-scale-grid]")).toBeNull();
    await click(scale());
    expect(app.getSnapshot().scaleIndicator).toBe("ruler");

    expect(c.querySelector("[data-axis-basis]")).not.toBeNull();
    await click(c.querySelector('[data-action="view-axes"]'));
    expect(app.getSnapshot().showAxisBasis).toBe(false);
    expect(c.querySelector("[data-axis-basis]")).toBeNull();

    const proj = () => c.querySelector<HTMLButtonElement>('[data-action="camera-projection"]')!;
    expect(proj().textContent).toBe("Persp");
    await click(proj());
    expect(app.getSnapshot().cameraProjection).toBe("orthographic");
    expect(proj().textContent).toBe("Ortho");
    expect(proj().getAttribute("aria-pressed")).toBe("true");
  });

  it("folds_to_its_toggle_and_opens_again", async () => {
    const app = new AppState();
    const c = await mount(app);
    const panel = () => c.querySelector<HTMLElement>(".view-cube-panel")!;
    const toggle = () => c.querySelector<HTMLButtonElement>('[data-action="view-cube-toggle"]')!;
    expect(panel().dataset.open).toBe("true");
    expect(toggle().getAttribute("aria-expanded")).toBe("true");
    await click(toggle());
    expect(panel().dataset.open).toBe("false");
    expect(toggle().getAttribute("aria-expanded")).toBe("false");
    // Folded, its controls stay mounted (hidden by CSS), so their hooks stay queryable.
    expect(c.querySelector('[data-view-orbit="up"]')).not.toBeNull();
    await click(toggle());
    expect(panel().dataset.open).toBe("true");
  });
});

describe("navigation cube settings", () => {
  it("persist_the_toggles_with_the_display_settings", () => {
    const d = defaultSettings();
    expect([d.cameraProjection, d.scaleIndicator, d.showAxisBasis]).toEqual(["perspective", "ruler", true]);

    const app = new AppState();
    app.setScaleIndicator("grid");
    app.setShowAxisBasis(false);
    app.toggleCameraProjection();
    const stored = JSON.parse(JSON.stringify(settingsFromSnapshot(app.getSnapshot()))) as unknown;
    const back = new AppState(parseSettings(stored)).getSnapshot();
    expect([back.scaleIndicator, back.showAxisBasis, back.cameraProjection]).toEqual(["grid", false, "orthographic"]);

    // An older document (no such fields) or a bad value takes the defaults.
    expect(parseSettings({ version: 1, theme: "light" })).toMatchObject({
      cameraProjection: "perspective",
      scaleIndicator: "ruler",
      showAxisBasis: true,
    });
    expect(parseSettings({ version: 1, scaleIndicator: "dots", cameraProjection: "fisheye", showAxisBasis: 1 })).toMatchObject({
      cameraProjection: "perspective",
      scaleIndicator: "ruler",
      showAxisBasis: true,
    });

    app.resetDisplaySettings();
    const reset = app.getSnapshot();
    expect([reset.scaleIndicator, reset.showAxisBasis, reset.cameraProjection]).toEqual(["ruler", true, "perspective"]);
  });

  it("cycles_the_scale_indicator_and_centres_its_grid", () => {
    expect(nextScaleIndicator("ruler")).toBe("grid");
    expect(nextScaleIndicator("grid")).toBe("off");
    expect(nextScaleIndicator("off")).toBe("ruler");
    expect(scaleGridBackground(80)).toEqual({
      backgroundSize: "80px 80px",
      backgroundPosition: "calc(50% + 40px) calc(50% + 40px)",
    });
  });

  it("maps_the_face_preset_keys", () => {
    expect(cameraPresetForKey("Digit1", false)).toBe("front");
    expect(cameraPresetForKey("Numpad1", true)).toBe("back");
    expect(cameraPresetForKey("Digit3", false)).toBe("left");
    expect(cameraPresetForKey("Digit3", true)).toBe("right");
    expect(cameraPresetForKey("Numpad7", false)).toBe("top");
    expect(cameraPresetForKey("Digit7", true)).toBe("bottom");
    expect(cameraPresetForKey("Digit2", false)).toBeNull();
    expect(cameraPresetForKey("KeyT", false)).toBeNull();
  });
});

function KeysOnly(props: { app: AppState }) {
  const noop = () => {};
  useKeyboardShortcuts({
    app: props.app,
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
  return null;
}

describe("face preset keys", () => {
  it("turn_the_view_to_a_face_outside_text_fields_only", async () => {
    const app = new AppState();
    const c = document.createElement("div");
    document.body.appendChild(c);
    container = c;
    await act(async () => {
      root = createRoot(c);
      root.render(createElement(KeysOnly, { app }));
    });
    const press = async (init: KeyboardEventInit, target: EventTarget = window) => {
      await act(async () => {
        target.dispatchEvent(new KeyboardEvent("keydown", { bubbles: true, cancelable: true, ...init }));
      });
    };
    await press({ key: "7", code: "Digit7" });
    expect(app.getSnapshot().cameraPresetRequest).toEqual({ preset: "top", n: 1 });
    await press({ key: "!", code: "Digit1", shiftKey: true });
    expect(app.getSnapshot().cameraPresetRequest).toEqual({ preset: "back", n: 2 });
    await press({ key: "1", code: "Digit1", ctrlKey: true });
    const input = document.createElement("input");
    c.appendChild(input);
    await press({ key: "3", code: "Digit3" }, input);
    expect(app.getSnapshot().cameraPresetRequest?.n).toBe(2);
  });
});
