// The first-run tour's start rule: fresh profiles tour, headless and
// opted-out ones never do, `?tour=1` forces it on.
import { describe, expect, it } from "vitest";
import { TOUR_STEPS, resolveTourStart } from "../src/panels/tour";

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
