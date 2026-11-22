// The navigation HUD's frame loop keys: the triad re-renders only when its
// drawing moves by half a pixel or more, or its painter's order changes.
import * as THREE from "three";
import { describe, expect, it } from "vitest";
import { projectBasisTriad } from "../src/features/view/ux/navigation-hud";
import { triadSummaryKey } from "../src/features/view/ux/use-navigation-hud";

describe("triadSummaryKey", () => {
  it("names_each_axis_at_half_pixel_steps", () => {
    expect(
      triadSummaryKey([
        { axis: "X", x: 10.2, y: -0.3, z: 0, color: "#f00" },
        { axis: "Y", x: 0, y: -19.76, z: 0, color: "#0f0" },
      ]),
    ).toBe("X:20,-1|Y:0,-40");
  });

  it("keeps_the_key_for_a_sub_half_pixel_turn", () => {
    const at = (yaw: number) => new THREE.Quaternion().setFromEuler(new THREE.Euler(0.3, 0.6 + yaw, 0));
    const a = projectBasisTriad(at(0), 20);
    expect(triadSummaryKey(projectBasisTriad(at(1e-4), 20))).toBe(triadSummaryKey(a));
    expect(triadSummaryKey(projectBasisTriad(at(0.5), 20))).not.toBe(triadSummaryKey(a));
  });
});
