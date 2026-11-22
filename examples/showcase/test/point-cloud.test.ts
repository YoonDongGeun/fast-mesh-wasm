// `.pts` point cloud text normalized to the `.xyz` rows the SDK parser reads:
// the leading count line and comments drop, intensity columns drop.
import { describe, expect, it } from "vitest";
import { normalizePtsText } from "../src/features/points/ux/point-cloud";

describe("normalizePtsText", () => {
  it("drops_the_count_line_comments_and_blank_lines", () => {
    expect(normalizePtsText("# scan\n2\n\n1 2 3\n// note\n4 5 6\n")).toBe("1 2 3\n4 5 6");
  });

  it("drops_the_intensity_column", () => {
    expect(normalizePtsText("1 2 3 99\r\n1 2 3 99 10 20 30")).toBe("1 2 3\n1 2 3 10 20 30");
  });

  it("keeps_other_rows_and_a_first_row_that_is_not_a_count", () => {
    expect(normalizePtsText("1 2 3\n7  8\t9 1 2")).toBe("1 2 3\n7 8 9 1 2");
  });
});
