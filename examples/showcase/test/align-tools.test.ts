// The align panel's landmark, coarse + ICP and symmetry plane logic (#253):
// pair picking order, the world pairs and their residuals, the plane between
// world and local frames, the quad drawn over the mesh, and the reports.
import type { MeshHandle, Vec3 } from "fast-mesh-wasm";
import { describe, expect, it } from "vitest";
import type { Item } from "../src/app/state/app-state";
import {
  ALIGN_COMMANDS,
  addLandmark,
  completePairs,
  formatNormal,
  invertAffine,
  landmarkPairs,
  landmarkPrompt,
  landmarkResiduals,
  landmarksProblem,
  multistageStages,
  multistageWarning,
  nextLandmarkSide,
  planeAxes,
  planeQuad,
  planeToLocal,
  planeToWorld,
  pointsOnALine,
  quadCorners,
  removeLastLandmark,
  startLandmarks,
} from "../src/features/transform/ux/align-tools";
import { placementMatrix, worldPoint } from "../src/features/transform/ux/placement-math";
import { resolveMesh } from "../src/features/transform/ux/use-align-tools";

const [moving, fixed, other] = [1, 2, 3].map((n) => n as unknown as MeshHandle);
const IDENTITY = placementMatrix({ rotation: [1, 0, 0, 0, 1, 0, 0, 0, 1], translation: [0, 0, 0] }, [0, 0, 0]);

const close = (a: readonly number[], b: readonly number[], digits = 9) => {
  expect(a.length).toBe(b.length);
  a.forEach((v, i) => expect(v).toBeCloseTo(b[i], digits));
};

describe("landmark pairs", () => {
  it("alternates_moving_then_fixed_and_counts_complete_pairs", () => {
    let set = startLandmarks(moving, fixed);
    expect(set.picking).toBe(true);
    expect(nextLandmarkSide(set)).toBe("moving");
    set = addLandmark(set, [1, 0, 0]);
    expect(nextLandmarkSide(set)).toBe("fixed");
    expect(completePairs(set)).toBe(0);
    set = addLandmark(set, [2, 0, 0]);
    expect(nextLandmarkSide(set)).toBe("moving");
    expect(completePairs(set)).toBe(1);
    set = addLandmark(set, [3, 0, 0]);
    expect(set.movingPoints).toEqual([
      [1, 0, 0],
      [3, 0, 0],
    ]);
    expect(set.fixedPoints).toEqual([[2, 0, 0]]);
    expect(completePairs(set)).toBe(1);
  });

  it("drops_the_latest_point_first", () => {
    let set = startLandmarks(moving, fixed);
    expect(removeLastLandmark(set)).toBe(set);
    set = addLandmark(addLandmark(addLandmark(set, [1, 0, 0]), [2, 0, 0]), [3, 0, 0]);
    set = removeLastLandmark(set);
    expect(set.movingPoints).toEqual([[1, 0, 0]]);
    expect(set.fixedPoints).toEqual([[2, 0, 0]]);
    set = removeLastLandmark(set);
    expect(set.fixedPoints).toEqual([]);
    expect(set.movingPoints).toEqual([[1, 0, 0]]);
    set = removeLastLandmark(set);
    expect(set.movingPoints).toEqual([]);
  });

  it("prompts_for_the_side_and_pair_number", () => {
    let set = startLandmarks(moving, fixed);
    expect(landmarkPrompt(set, "A", "B")).toBe("Pair 1: click a point on the moving mesh (A).");
    set = addLandmark(set, [0, 0, 0]);
    expect(landmarkPrompt(set, "A", "B")).toBe("Pair 1: click the matching point on the fixed mesh (B).");
    set = addLandmark(set, [0, 0, 0]);
    expect(landmarkPrompt(set, "A", "B")).toBe("Pair 2: click a point on the moving mesh (A).");
  });

  it("puts_complete_pairs_in_the_world_and_measures_them", () => {
    let set = startLandmarks(moving, fixed);
    for (const [m, f] of [
      [
        [0, 0, 0],
        [1, 0, 0],
      ],
      [
        [1, 0, 0],
        [2, 0, 0],
      ],
      [
        [0, 1, 0],
        [1, 1, 0],
      ],
    ] as [Vec3, Vec3][]) {
      set = addLandmark(addLandmark(set, m), f);
    }
    set = addLandmark(set, [9, 9, 9]); // an unmatched point is left out
    // The moving mesh sits 1 along +X, its origin at (5, 0, 0): local p is world p + (6, 0, 0).
    const movingMatrix = placementMatrix({ rotation: [1, 0, 0, 0, 1, 0, 0, 0, 1], translation: [1, 0, 0] }, [5, 0, 0]);
    const pairs = landmarkPairs(set, movingMatrix, IDENTITY);
    expect(pairs).toHaveLength(3);
    expect(pairs[0]).toEqual({ source: [6, 0, 0], target: [1, 0, 0] });
    expect(landmarkResiduals(set, movingMatrix, IDENTITY)).toEqual([5, 5, 5]);
    expect(landmarkResiduals(set, IDENTITY, IDENTITY)).toEqual([1, 1, 1]);
  });

  it("refuses_points_on_a_line_on_either_side", () => {
    expect(pointsOnALine([[0, 0, 0], [1, 1, 1], [5, 5, 5]])).toBe(true);
    // The ends come first or last, whatever the order.
    expect(pointsOnALine([[1, 1, 1], [5, 5, 5], [0, 0, 0], [2, 2, 2.01]])).toBe(true);
    expect(pointsOnALine([[0, 0, 0], [1, 0, 0], [0, 1, 0]])).toBe(false);
    expect(pointsOnALine([[2, 2, 2], [2, 2, 2], [2, 2, 2]])).toBe(true);
    // Within 1% of the length is on the line; 5% is off it.
    expect(pointsOnALine([[0, 0, 0], [50, 0.9, 0], [100, 0, 0]])).toBe(true);
    expect(pointsOnALine([[0, 0, 0], [50, 5, 0], [100, 0, 0]])).toBe(false);

    const line: Vec3[] = [[0, 0, 0], [1, 0, 0], [2, 0, 0]];
    const bent: Vec3[] = [[0, 0, 0], [1, 0, 0], [0, 1, 0]];
    const build = (m: readonly Vec3[], f: readonly Vec3[]) => {
      let set = startLandmarks(moving, fixed);
      m.forEach((p, i) => {
        set = addLandmark(addLandmark(set, p), f[i]);
      });
      return set;
    };
    expect(landmarksProblem(build(bent, bent))).toBeNull();
    expect(landmarksProblem(build(line, bent))).toContain("moving mesh lie on a line");
    expect(landmarksProblem(build(bent, line))).toContain("fixed mesh lie on a line");
    // Too few pairs is said elsewhere, and an unmatched point does not count.
    expect(landmarksProblem(build(line.slice(0, 2), line.slice(0, 2)))).toBeNull();
    expect(landmarksProblem(addLandmark(build(line, line), [0, 1, 0]))).toContain("lie on a line");
    expect(landmarksProblem(addLandmark(addLandmark(build(line, line), [0, 1, 0]), [0, 1, 0]))).toBeNull();
  });
});

describe("symmetry plane frames", () => {
  // A turn about Z by 90 degrees, scale 2 on X, moved to (10, -4, 3), origin at (1, 2, 3).
  const placement = { rotation: [0, -1, 0, 1, 0, 0, 0, 0, 1], translation: [10, -4, 3] as Vec3, scale: [2, 1, 1] as Vec3 };
  const matrix = placementMatrix(placement, [1, 2, 3]);

  it("inverts_an_affine_matrix", () => {
    const inv = invertAffine(matrix);
    for (const p of [
      [0, 0, 0],
      [1, -2, 5],
      [-3, 0.5, 7],
    ] as Vec3[]) {
      const w = worldPoint(matrix, p[0], p[1], p[2]);
      close(worldPoint(inv, w[0], w[1], w[2]), p, 12);
    }
    expect(() => invertAffine(new Float64Array(16))).toThrow();
  });

  it("carries_a_plane_to_local_and_back", () => {
    const world = { point: [12, -3, 4] as Vec3, normal: [0, 0.6, 0.8] as Vec3 };
    const local = planeToLocal(matrix, world.point, world.normal);
    expect(Math.hypot(...local.normal)).toBeCloseTo(1, 12);
    const back = planeToWorld(matrix, local.point, local.normal);
    close(back.point, world.point, 12);
    close(back.normal, world.normal, 12);
    // Points on the local plane land on the world plane.
    const [u] = planeAxes(local.normal);
    const onLocal: Vec3 = [local.point[0] + 3 * u[0], local.point[1] + 3 * u[1], local.point[2] + 3 * u[2]];
    const w = worldPoint(matrix, onLocal[0], onLocal[1], onLocal[2]);
    const d = (w[0] - world.point[0]) * world.normal[0] + (w[1] - world.point[1]) * world.normal[1] + (w[2] - world.point[2]) * world.normal[2];
    expect(d).toBeCloseTo(0, 12);
  });

  it("spans_the_plane_with_unit_axes", () => {
    for (const n of [
      [1, 0, 0],
      [0, 0, -1],
      [0.6, 0, 0.8],
    ] as Vec3[]) {
      const [u, v] = planeAxes(n);
      expect(Math.hypot(...u)).toBeCloseTo(1, 12);
      expect(Math.hypot(...v)).toBeCloseTo(1, 12);
      expect(u[0] * n[0] + u[1] * n[1] + u[2] * n[2]).toBeCloseTo(0, 12);
      expect(v[0] * n[0] + v[1] * n[1] + v[2] * n[2]).toBeCloseTo(0, 12);
      expect(u[0] * v[0] + u[1] * v[1] + u[2] * v[2]).toBeCloseTo(0, 12);
    }
  });
});

describe("planeQuad", () => {
  it("covers_the_box_seen_along_the_normal", () => {
    // The plane X = 1 through a box 0..4 x -1..1 x 0..6.
    const q = planeQuad([1, 5, 5], [1, 0, 0], [0, -1, 0], [4, 1, 6], 0);
    close(q.center, [1, 0, 3]);
    expect(new Set([q.halfU, q.halfV])).toEqual(new Set([1, 3]));
    const corners = quadCorners(q);
    for (const c of corners) {
      expect(c[0]).toBeCloseTo(1, 12);
    }
    const ys = corners.map((c) => c[1]);
    const zs = corners.map((c) => c[2]);
    expect(Math.min(...ys)).toBeCloseTo(-1, 12);
    expect(Math.max(...ys)).toBeCloseTo(1, 12);
    expect(Math.min(...zs)).toBeCloseTo(0, 12);
    expect(Math.max(...zs)).toBeCloseTo(6, 12);
  });

  it("grows_by_the_margin_and_stays_on_a_tilted_plane", () => {
    const n: Vec3 = [Math.SQRT1_2, Math.SQRT1_2, 0];
    const tight = planeQuad([0, 0, 0], n, [-1, -1, -1], [1, 1, 1], 0);
    const loose = planeQuad([0, 0, 0], n, [-1, -1, -1], [1, 1, 1]);
    expect(loose.halfU).toBeCloseTo(tight.halfU * 1.08, 12);
    expect(loose.halfV).toBeCloseTo(tight.halfV * 1.08, 12);
    // The box's corners project inside the tight quad.
    for (const c of quadCorners(tight)) {
      expect(c[0] * n[0] + c[1] * n[1] + c[2] * n[2]).toBeCloseTo(0, 12);
    }
    expect(Math.max(tight.halfU, tight.halfV)).toBeCloseTo(Math.SQRT2, 12);
    expect(Math.min(tight.halfU, tight.halfV)).toBeCloseTo(1, 12);
  });
});

describe("reports and picks", () => {
  it("lists_the_coarse_stage_then_the_icp_stage", () => {
    const stages = multistageStages(
      "orientedBoundingBox",
      { finalRmse: 0.5 },
      { finalRmse: 0.01, iterations: 12, status: "converged" },
    );
    expect(stages).toEqual([
      { label: "Coarse (oriented bounding box)", iterations: 0, rmse: 0.5, status: null },
      { label: "ICP", iterations: 12, rmse: 0.01, status: "converged" },
    ]);
    expect(multistageStages("pcaCentroid", { finalRmse: 1 }, { finalRmse: 1, iterations: 0, status: "maxIterations" })[0].label).toBe(
      "Coarse (principal axes)",
    );
  });

  it("warns_when_the_icp_did_not_converge", () => {
    const fine = (status: "converged" | "maxIterations" | "underconstrained" | "tooFewCorrespondences" | "diverged", finalRmse = 0.1) => ({
      status,
      finalRmse,
    });
    expect(multistageWarning({ finalRmse: 0.2 }, fine("converged"))).toBeNull();
    expect(multistageWarning({ finalRmse: 0.2 }, fine("maxIterations"))).toContain("iteration limit");
    expect(multistageWarning({ finalRmse: 0.2 }, fine("underconstrained"))).toContain("slide or spin");
    expect(multistageWarning({ finalRmse: 0.2 }, fine("tooFewCorrespondences", 1e-9))).toContain("arbitrary");
    expect(multistageWarning({ finalRmse: 0.2 }, fine("diverged"))).toContain("diverged");
    expect(multistageWarning({ finalRmse: Infinity }, fine("tooFewCorrespondences", Infinity))).toContain("nothing moved");
    // The coarse stage moved it, so it is not "nothing moved".
    expect(multistageWarning({ finalRmse: 0.2 }, fine("diverged", Infinity))).not.toContain("nothing moved");
  });

  it("formats_a_normal_without_signed_zeros", () => {
    expect(formatNormal([-1e-9, 1, -0])).toBe("(0.000, 1.000, 0.000)");
    expect(formatNormal([0.6, -0.8, 0])).toBe("(0.600, -0.800, 0.000)");
  });

  it("resolves_the_pick_then_the_selection_then_the_first_mesh", () => {
    const items = [moving, fixed].map((mesh) => ({ mesh }) as unknown as Item);
    expect(resolveMesh(items, fixed, moving)).toBe(fixed);
    expect(resolveMesh(items, other, fixed)).toBe(fixed);
    expect(resolveMesh(items, null, other)).toBe(moving);
    expect(resolveMesh([], null, null)).toBeNull();
  });

  it("names_a_panel_action_per_command", () => {
    expect(ALIGN_COMMANDS.map((c) => c.action)).toEqual(["landmarks-pick", "multistage-run", "symmetry-detect"]);
  });
});
