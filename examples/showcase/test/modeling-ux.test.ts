// The Modeling panel's pure behaviour: the Boolean source pick, the
// simplify target and its fields, the cut slider range, when Offset may
// run, the hollow request and the print-prep report lines, the stamp
// fields, and the viewport preview's pose on a surface.
import type { MeshHandle } from "fast-mesh-wasm";
import { describe, expect, it } from "vitest";
import type { Item } from "../src/app/state/app-state";
import { createDefaultWave6State } from "../src/app/state/wave6-state";
import {
  describeDrainResult,
  describeHollowResult,
  hollowOptionsFrom,
} from "../src/features/mesh-edit/ux/print-prep-ops";
import {
  cutRangeAlong,
  isVoxelOffset,
  offsetBlocked,
  pickOtherMesh,
  simplifyPercentInput,
  simplifyTargetFor,
  triangleCountInput,
} from "../src/features/mesh-edit/ux/use-modeling-ops";
import { printPrepPreviewPose } from "../src/features/mesh-edit/ux/use-print-prep-pointer";
import { spinInput, stampScaleInput, taperInput } from "../src/features/mesh-edit/ux/use-print-prep-ops";

const mesh = (n: number) => ({ __mesh: n }) as unknown as MeshHandle;
const item = (n: number) => ({ id: n, mesh: mesh(n) }) as unknown as Item;

describe("modeling ops", () => {
  it("picks a Boolean source other than the target", () => {
    const items = [item(1), item(2), item(3)];
    expect(pickOtherMesh(items, items[0]!.mesh, null)).toBe(items[1]!.mesh);
    expect(pickOtherMesh(items, items[0]!.mesh, items[2]!.mesh)).toBe(items[2]!.mesh);
    // The current pick stops counting once it is the target, or gone.
    expect(pickOtherMesh(items, items[2]!.mesh, items[2]!.mesh)).toBe(items[0]!.mesh);
    expect(pickOtherMesh(items, items[0]!.mesh, mesh(9))).toBe(items[1]!.mesh);
    // A single mesh pairs with itself; none, with nothing.
    expect(pickOtherMesh([items[0]!], items[0]!.mesh, null)).toBe(items[0]!.mesh);
    expect(pickOtherMesh([], null, null)).toBeNull();
  });

  it("aims Simplify at a share of the live triangles, at least one", () => {
    expect(simplifyTargetFor(1000, 50)).toBe(500);
    expect(simplifyTargetFor(7, 30)).toBe(2);
    expect(simplifyTargetFor(0, 50)).toBe(1);
    expect(simplifyPercentInput("30")).toBe(30);
    expect(simplifyPercentInput("250")).toBe(100);
    expect(simplifyPercentInput("0")).toBe(50);
    expect(simplifyPercentInput("")).toBe(50);
    expect(simplifyPercentInput("-5")).toBe(1);
    expect(triangleCountInput("12.4")).toBe(12);
    expect(triangleCountInput("")).toBe(1);
    expect(triangleCountInput("-3")).toBe(1);
  });

  it("pads the cut slider range a quarter of the extent (at least 1) along the axis", () => {
    const ext = { min: [0, -2, 10], max: [8, 2, 10.5], extents: [8, 4, 0.5] } as never;
    expect(cutRangeAlong(ext, "x")).toEqual([-2, 10]);
    expect(cutRangeAlong(ext, "y")).toEqual([-3, 3]);
    expect(cutRangeAlong(ext, "z")).toEqual([9, 11.5]);
    // A custom normal uses the z range.
    expect(cutRangeAlong(ext, "custom")).toEqual([9, 11.5]);
  });

  it("blocks Offset at zero distance on the normal offset, or with a bad voxel size", () => {
    expect(isVoxelOffset("bvhSdf")).toBe(true);
    expect(isVoxelOffset("fhEdt")).toBe(true);
    expect(isVoxelOffset("auto")).toBe(false);
    expect(isVoxelOffset("normal")).toBe(false);
    expect(offsetBlocked("auto", 0, "")).toBe(true);
    expect(offsetBlocked("normal", 0.5, "")).toBe(false);
    // A voxel offset at zero distance is a voxel remesh.
    expect(offsetBlocked("bvhSdf", 0, "")).toBe(false);
    expect(offsetBlocked("fhEdt", 0.5, "0.4")).toBe(false);
    expect(offsetBlocked("fhEdt", 0.5, "0")).toBe(true);
    expect(offsetBlocked("bvhSdf", 0.5, "x")).toBe(true);
    // The normal offset ignores the voxel field.
    expect(offsetBlocked("normal", 0.5, "0")).toBe(false);
  });
});

describe("print prep", () => {
  it("builds the hollow request from the panel's fields", () => {
    const fields = { wallMm: 1.5, backend: "bvhSdf" as const, voxelSizeText: "", maxVoxels: 524288, capInnerShell: false };
    expect(hollowOptionsFrom(fields)).toEqual({ wallThickness: 1.5, backend: "bvhSdf", maxVoxels: 524288 });
    expect(hollowOptionsFrom({ ...fields, wallMm: 0, voxelSizeText: " 0.8 ", capInnerShell: true })).toEqual({
      wallThickness: 0.1,
      backend: "bvhSdf",
      voxelSize: 0.8,
      maxVoxels: 524288,
      maxCavityTriangles: 12000,
    });
    expect(hollowOptionsFrom({ ...fields, voxelSizeText: "-1" })).not.toHaveProperty("voxelSize");
    expect(hollowOptionsFrom({ ...fields, voxelSizeText: "auto" })).not.toHaveProperty("voxelSize");
  });

  it("reports hollow and drain hole runs on one line", () => {
    expect(
      describeHollowResult({
        beforeTriangles: 12,
        afterTriangles: 40,
        report: { cavitiesCreated: 1, cavityTriangles: 28, cavityVolume: 12.345 },
      } as never),
    ).toBe("12 -> 40 tris · 1 cavity (28 cavity tris, 12.3 mm3)");
    const drain = { beforeTriangles: 40, afterTriangles: 72, report: { holesPunched: 1 } };
    expect(describeDrainResult({ ...drain, plugs: [] } as never)).toBe("40 -> 72 tris (1 hole)");
    expect(describeDrainResult({ ...drain, plugs: [mesh(5)] } as never)).toBe("40 -> 72 tris (1 hole · 1 plug adopted)");
  });

  it("parses the taper, stamp scale and spin fields", () => {
    expect(taperInput("60")).toBe(45);
    expect(taperInput("-3")).toBe(0);
    expect(taperInput("")).toBe(0);
    expect(taperInput("12.5")).toBe(12.5);
    expect(stampScaleInput("0")).toBe(1);
    expect(stampScaleInput("0.01")).toBe(0.05);
    expect(stampScaleInput("2")).toBe(2);
    expect(spinInput("x")).toBe(0);
    expect(spinInput("-15")).toBe(-15);
  });

  it("stands the drain preview on the surface normal, sized by the drain", () => {
    const prep = { ...createDefaultWave6State().printPrep, activeTool: "drainHole" as const };
    const pose = printPrepPreviewPose(prep, [1, 2, 3], [1, 0, 0], false);
    expect(pose.drain).toBe(true);
    expect(pose.position).toEqual([1, 2, 3]);
    expect(pose.ringRadius).toBe(2);
    expect(pose.depth).toBe(12);
    expect(pose.normalLength).toBe(3);
    // Local +Z turns onto +X: a quarter turn about +Y.
    expect(pose.quaternion[1]).toBeCloseTo(Math.SQRT1_2, 6);
    expect(pose.quaternion[3]).toBeCloseTo(Math.SQRT1_2, 6);
    // A through hole still shows a bore of depth 1.
    const through = { ...prep, drainParams: { ...prep.drainParams, depth: 0, diameter: 0.1 } };
    const thin = printPrepPreviewPose(through, [0, 0, 0], [0, 0, 1], true);
    expect([thin.ringRadius, thin.depth, thin.normalLength]).toEqual([0.25, 1, 2]);
  });

  it("stands a stamp upright unless it aligns to the normal, spun by its twist", () => {
    const base = createDefaultWave6State().printPrep;
    const prep = {
      ...base,
      activeTool: "placeStamp" as const,
      placeParams: { ...base.placeParams, uniformScale: 2, twistDeg: 90 },
    };
    const upright = printPrepPreviewPose(prep, [0, 0, 0], [1, 0, 0], false);
    expect(upright.drain).toBe(false);
    expect(upright.ringRadius).toBe(4);
    expect(upright.normalLength).toBe(5);
    expect(upright.axisLength).toBeCloseTo(3.4, 12);
    // Upright with a quarter spin about +Z.
    expect(upright.quaternion[2]).toBeCloseTo(Math.SQRT1_2, 6);
    expect(upright.quaternion[3]).toBeCloseTo(Math.SQRT1_2, 6);
    const aligned = printPrepPreviewPose(prep, [0, 0, 0], [1, 0, 0], true);
    expect(aligned.quaternion).not.toEqual(upright.quaternion);
  });
});
