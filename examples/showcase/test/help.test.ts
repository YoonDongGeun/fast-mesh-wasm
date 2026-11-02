// The first-run tour's start rule: fresh profiles tour, headless and
// opted-out ones never do, `?tour=1` forces it on. Plus the overlay's
// content (every binding in App's keyboard handler), the card's step
// machine, and the localStorage "seen" round-trip.
import { afterEach, describe, expect, it, vi } from "vitest";
import { SHORTCUTS } from "../src/panels/Help";
import { TOUR_STEPS, markTourSeen, resolveTourStart, tourBack, tourNext, tourSeen } from "../src/panels/tour";

const DESKTOP =
  "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/140.0.0.0 Safari/537.36";
const HEADLESS =
  "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) HeadlessChrome/140.0.0.0 Safari/537.36";

describe("tour start", () => {
  it("tours_fresh_profiles", () => {
    expect(resolveTourStart("", DESKTOP, false, false)).toBe("auto");
  });

  it("skips_seen_profiles", () => {
    expect(resolveTourStart("", DESKTOP, false, true)).toBe("off");
  });

  it("stays_off_headless", () => {
    // showcase-check.mjs drives --headless=new: no card over the canvas.
    expect(resolveTourStart("", HEADLESS, false, false)).toBe("off");
    expect(resolveTourStart("", DESKTOP, true, false)).toBe("off");
  });

  it("query_flag_forces", () => {
    expect(resolveTourStart("?tour=1", HEADLESS, false, true)).toBe("forced");
    expect(resolveTourStart("?tour=0", DESKTOP, false, false)).toBe("off");
    expect(resolveTourStart("?notour", DESKTOP, false, false)).toBe("off");
  });

  it("covers_the_editor", () => {
    expect(TOUR_STEPS.map((s) => s.target)).toEqual(["mesh-list", "gizmo", "repair", "collision"]);
  });
});

describe("shortcut overlay", () => {
  it("names_every_binding", () => {
    // Against App's keyboard handler: T/R, undo/redo (both spellings), F,
    // W, Esc and ? itself.
    const keys = SHORTCUTS.map(([k]) => k).join("\n");
    for (const binding of ["T", "R", "Ctrl+Z", "Ctrl+Shift+Z", "Ctrl+Y", "F", "W", "Esc", "?"]) {
      expect(keys).toContain(binding);
    }
    // Every row explains its keys.
    for (const [, what] of SHORTCUTS) {
      expect(what.length).toBeGreaterThan(0);
    }
  });
});

describe("tour steps", () => {
  it("next_walks_then_ends", () => {
    let step: number | null = 0;
    for (let i = 1; i < TOUR_STEPS.length; i++) {
      step = tourNext(step as number);
      expect(step).toBe(i);
    }
    expect(tourNext(step as number)).toBeNull();
  });

  it("back_stays_on_first", () => {
    expect(tourBack(0)).toBe(0);
    expect(tourBack(2)).toBe(1);
  });
});

describe("tour seen", () => {
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("round_trips_local_storage", () => {
    const store = new Map<string, string>();
    vi.stubGlobal("localStorage", {
      getItem: (k: string) => (store.has(k) ? store.get(k)! : null),
      setItem: (k: string, v: string) => void store.set(k, v),
    });
    expect(tourSeen()).toBe(false);
    markTourSeen();
    expect(tourSeen()).toBe(true);
  });

  it("survives_missing_or_broken_storage", () => {
    vi.stubGlobal("localStorage", undefined);
    expect(tourSeen()).toBe(false);
    expect(() => markTourSeen()).not.toThrow();
    vi.stubGlobal("localStorage", {
      getItem: () => {
        throw new Error("denied");
      },
      setItem: () => {
        throw new Error("denied");
      },
    });
    expect(tourSeen()).toBe(false);
    expect(() => markTourSeen()).not.toThrow();
  });
});
