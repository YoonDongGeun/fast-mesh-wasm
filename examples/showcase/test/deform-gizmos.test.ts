// Where the deform gizmos go and how big they are (#193): the FFD cage's
// points, wireframe and sphere size, and the anchor and handle markers over
// the live vertex positions.
import type { FfdLatticeData } from "fast-mesh-wasm";
import { describe, expect, it } from "vitest";
import { deformMarkers, latticeCage } from "../src/features/sculpt/ux/deform-gizmos";
import { buildLatticeEdgeSegments } from "../src/features/sculpt/ux/ffd-lattice";

function lattice(halfExtents: [number, number, number]): FfdLatticeData {
  return {
    dims: [2, 3, 2],
    basis: "bernstein",
    frame: {
      center: [0, 0, 0],
      axes: [
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
      ],
      halfExtents,
    },
    offsets: new Float64Array(2 * 3 * 2 * 3),
  };
}

describe("lattice cage", () => {
  it("is_empty_without_a_lattice", () => {
    const cage = latticeCage(null);
    expect(cage.cpPoints).toEqual([]);
    expect(cage.edgePositions.length).toBe(0);
    expect(cage.sphereRadius).toBe(0.03);
  });

  it("draws_every_control_point_and_edge", () => {
    const l = lattice([1, 2, 0.5]);
    const cage = latticeCage(l);
    expect(cage.cpPoints.length).toBe(12);
    expect(cage.cpPoints[0]).toEqual([-1, -2, -0.5]);
    expect(Array.from(cage.edgePositions)).toEqual(Array.from(buildLatticeEdgeSegments(l)));
  });

  it("sizes_spheres_to_the_frame_with_a_floor", () => {
    expect(latticeCage(lattice([1, 2, 0.5])).sphereRadius).toBeCloseTo(0.09, 9);
    expect(latticeCage(lattice([0.1, 0.1, 0.1])).sphereRadius).toBe(0.015);
  });
});

describe("deform markers", () => {
  it("is_empty_without_positions", () => {
    expect(deformMarkers(null, [0], [{ vertex: 0, target: [0, 0, 0] }])).toEqual({
      anchorPoints: [],
      handlePairs: [],
      markerRadius: 0.03,
    });
  });

  it("marks_in_range_anchors_and_handles", () => {
    const positions = Float32Array.from([-1, 0, 0, 1, 0, 0, 0, 1, 0]);
    const markers = deformMarkers(positions, [0, 9], [
      { vertex: 2, target: [0, 2, 0] },
      { vertex: 4, target: [5, 5, 5] },
    ]);
    expect(markers.anchorPoints).toEqual([[-1, 0, 0]]);
    expect(markers.handlePairs).toEqual([{ from: [0, 1, 0], to: [0, 2, 0] }]);
    expect(markers.markerRadius).toBeCloseTo(0.05, 9);
  });

  it("sizes_markers_to_the_x_span_with_floors", () => {
    expect(deformMarkers(Float32Array.from([0, 0, 0, 0.1, 0, 0]), [], []).markerRadius).toBe(0.015);
    expect(deformMarkers(new Float32Array(0), [], []).markerRadius).toBe(0.025);
  });
});
