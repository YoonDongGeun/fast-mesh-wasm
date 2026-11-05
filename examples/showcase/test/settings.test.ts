// The persisted display settings (issue #144): the tolerant loader, the
// round trip through storage, and the tour flag's namespaced key.
import { afterEach, describe, expect, it, vi } from "vitest";
import {
  DEFAULT_SIDEBAR_WIDTH,
  defaultSettings,
  loadSettings,
  parseSettings,
  resetSettings,
  saveSettings,
  SETTINGS_KEY,
  settingsFromSnapshot,
} from "../src/panels/settings";
import { markTourSeen, tourSeen } from "../src/panels/tour";

const LEGACY_SEEN_KEY = "fast-mesh-showcase.seen-tour";

/** A stub `localStorage` over a map, returned for inspection. */
function stubStorage(initial: Record<string, string> = {}): Map<string, string> {
  const store = new Map<string, string>(Object.entries(initial));
  vi.stubGlobal("localStorage", {
    getItem: (key: string): string | null => store.get(key) ?? null,
    setItem: (key: string, value: string): void => {
      store.set(String(key), String(value));
    },
    removeItem: (key: string): void => {
      store.delete(String(key));
    },
  });
  return store;
}

afterEach(() => {
  vi.unstubAllGlobals();
});

describe("parseSettings", () => {
  it("defaults_when_foreign", () => {
    const d = defaultSettings();
    for (const doc of [undefined, null, 0, "x", [], { version: 999 }, { version: "1" }, { theme: "light" }]) {
      expect(parseSettings(doc)).toEqual(d);
    }
  });

  it("keeps_valid_fields_and_repairs_the_rest", () => {
    const s = parseSettings({
      version: 1,
      theme: "light",
      leftVisible: false,
      leftWidth: 400,
      rightWidth: "wide",
      wireframe: 1,
      edgeSoftness: 0.5,
      repairOverlays: { holes: false, degenerate: "yes" },
      collisionMode: "sideways",
      heatmap: { depth: 1, gap: 2, tolerance: 0.5 },
      heatmapFast: true,
      decimateOnLoad: false,
      sculptRemesh: "eventually",
      maskRadius: -2,
    });
    expect(s.theme).toBe("light");
    expect(s.leftVisible).toBe(false);
    expect(s.rightVisible).toBe(true);
    expect(s.leftWidth).toBe(400);
    expect(s.rightWidth).toBe(DEFAULT_SIDEBAR_WIDTH);
    expect(s.wireframe).toBe(false);
    expect(s.edgeSoftness).toBe(0.5);
    expect(s.repairOverlays).toEqual({ holes: false, nonManifold: true, degenerate: true, components: true });
    expect(s.collisionMode).toBe("pair");
    expect(s.heatmap).toEqual({ depth: 1, gap: 2, tolerance: 0.5 });
    expect(s.heatmapFast).toBe(true);
    expect(s.decimateOnLoad).toBe(false);
    expect(s.sculptRemesh).toBe(true);
    expect(s.maskRadius).toBe(0.25);
  });

  it("rejects_bad_heatmap", () => {
    expect(parseSettings({ version: 1, heatmap: { depth: 1, gap: 2 } }).heatmap).toBe(null);
    expect(parseSettings({ version: 1, heatmap: { depth: 1, gap: 2, tolerance: Number.NaN } }).heatmap).toBe(null);
    expect(parseSettings({ version: 1, heatmap: { depth: -1, gap: 0, tolerance: 0 } }).heatmap).toBe(null);
  });

  it("clamps_widths", () => {
    expect(parseSettings({ version: 1, leftWidth: 50 }).leftWidth).toBe(200);
    expect(parseSettings({ version: 1, rightWidth: 9999 }).rightWidth).toBe(560);
    expect(parseSettings({ version: 1, leftWidth: Number.POSITIVE_INFINITY }).leftWidth).toBe(DEFAULT_SIDEBAR_WIDTH);
  });

  it("defaults_are_fresh_objects", () => {
    const a = defaultSettings();
    const b = defaultSettings();
    expect(a).toEqual(b);
    expect(a.repairOverlays).not.toBe(b.repairOverlays);
  });
});

describe("settings storage", () => {
  it("round_trips", () => {
    const store = stubStorage();
    expect(loadSettings()).toEqual(defaultSettings());
    const saved = saveSettings({ theme: "light", leftWidth: 400, heatmap: { depth: 1, gap: 2, tolerance: 0.5 } });
    expect(saved.theme).toBe("light");
    expect(JSON.parse(store.get(SETTINGS_KEY) ?? "{}").theme).toBe("light");
    expect(loadSettings()).toEqual(saved);
  });

  it("merges_patches_over_stored", () => {
    stubStorage();
    saveSettings({ theme: "light", wireframe: true });
    const next = saveSettings({ wireframe: false });
    expect(next.theme).toBe("light");
    expect(next.wireframe).toBe(false);
    expect(loadSettings()).toEqual(next);
  });

  it("survives_corrupt_and_missing_storage", () => {
    stubStorage({ [SETTINGS_KEY]: "{oops" });
    expect(loadSettings()).toEqual(defaultSettings());
    // No localStorage at all (or a throwing one): defaults, no throw.
    vi.unstubAllGlobals();
    expect(loadSettings()).toEqual(defaultSettings());
    expect(saveSettings({ theme: "light" }).theme).toBe("light");
    expect(resetSettings()).toEqual(defaultSettings());
  });

  it("reset_forgets", () => {
    const store = stubStorage();
    saveSettings({ theme: "light" });
    expect(store.has(SETTINGS_KEY)).toBe(true);
    expect(resetSettings()).toEqual(defaultSettings());
    expect(store.has(SETTINGS_KEY)).toBe(false);
    expect(loadSettings()).toEqual(defaultSettings());
  });

  it("the default unit survives storage, and older documents start on mm", () => {
    stubStorage();
    expect(defaultSettings().defaultUnits).toBe("mm");
    saveSettings({ defaultUnits: "in" });
    expect(loadSettings().defaultUnits).toBe("in");
    // A document written before #145 has no such field; the rest stands.
    expect(parseSettings({ version: 1, theme: "light" })).toMatchObject({ theme: "light", defaultUnits: "mm" });
    expect(parseSettings({ version: 1, defaultUnits: "furlong" }).defaultUnits).toBe("mm");
  });

  it("collects_a_snapshot", () => {
    const s = settingsFromSnapshot({
      ...defaultSettings(),
      heatmapParams: { depth: 1, gap: 2, tolerance: 0.5 },
    });
    expect(s.version).toBe(1);
    expect(s.defaultUnits).toBe("mm");
    expect(s.heatmap).toEqual({ depth: 1, gap: 2, tolerance: 0.5 });
    expect(s.repairOverlays).toEqual({ holes: true, nonManifold: true, degenerate: true, components: true });
  });
});

describe("tour storage", () => {
  it("persists_the_seen_flag_namespaced", () => {
    const store = stubStorage();
    expect(tourSeen()).toBe(false);
    markTourSeen();
    expect(tourSeen()).toBe(true);
    expect(store.get("fm.showcase.seen-tour")).toBe("1");
  });

  it("adopts_the_legacy_key_once", () => {
    const store = stubStorage({ [LEGACY_SEEN_KEY]: "1" });
    expect(tourSeen()).toBe(true);
    expect(store.get("fm.showcase.seen-tour")).toBe("1");
    expect(store.has(LEGACY_SEEN_KEY)).toBe(false);
  });
});
