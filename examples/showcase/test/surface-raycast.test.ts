// The canvas's raycasts: the ray through a pointer, the raycast that looks
// past hidden meshes, and what a hit means in its mesh's local positions.
import type { MeshHandle, RayHit, Scene } from "fast-mesh-wasm";
import * as THREE from "three";
import { describe, expect, it } from "vitest";
import {
  castVisibleSurface,
  localHitPoint,
  nearestCorner,
  rayThroughPointer,
  raycastVisibleSurface,
} from "../src/viewport/ux/surface-raycast";

const hidden = 1 as unknown as MeshHandle;
const shown = 2 as unknown as MeshHandle;

/** A scene whose raycast meets `layers` front to back, honouring `minDistance`; it records each call. */
function layeredScene(layers: readonly { mesh: MeshHandle; distance: number }[]) {
  const calls: { origin: readonly number[]; direction: readonly number[]; minDistance: number | undefined }[] = [];
  const scene = {
    raycast: (
      origin: readonly [number, number, number],
      direction: readonly [number, number, number],
      options?: { minDistance?: number },
    ): Promise<RayHit | null> => {
      calls.push({ origin, direction, minDistance: options?.minDistance });
      const min = options?.minDistance ?? 0;
      const layer = layers.find((l) => l.distance >= min);
      return Promise.resolve(
        layer === undefined
          ? null
          : { mesh: layer.mesh, triangle: 0, point: [0, 0, 0], distance: layer.distance, barycentric: [1, 0, 0] },
      );
    },
  };
  return { scene: scene as unknown as Scene, calls };
}

/** One triangle's local views: corners 4, 5, 6 of a seven-vertex buffer. */
const triangleScene = {
  views: () => ({
    indices: () => new Uint32Array([0, 1, 2, 4, 5, 6]),
    positions: () =>
      new Float32Array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 2, 0, 0, 0, 4]),
  }),
} as unknown as Scene;

describe("raycastVisibleSurface", () => {
  it("returns_the_first_hit_when_it_is_visible", async () => {
    const { scene, calls } = layeredScene([{ mesh: shown, distance: 3 }]);
    const hit = await raycastVisibleSurface(scene, [0, 0, 0], [0, 0, -1], (m) => m === shown);
    expect(hit?.mesh).toBe(shown);
    expect(calls).toHaveLength(1);
    expect(calls[0].minDistance).toBe(undefined);
  });

  it("looks_past_a_hidden_mesh_in_front", async () => {
    const { scene, calls } = layeredScene([
      { mesh: hidden, distance: 1 },
      { mesh: shown, distance: 2 },
    ]);
    const hit = await raycastVisibleSurface(scene, [0, 0, 0], [0, 0, -1], (m) => m === shown);
    expect(hit?.mesh).toBe(shown);
    expect(calls.map((c) => c.minDistance)).toEqual([undefined, 1 + 1e-4]);
  });

  it("returns_null_when_nothing_visible_is_behind", async () => {
    const { scene } = layeredScene([{ mesh: hidden, distance: 1 }]);
    expect(await raycastVisibleSurface(scene, [0, 0, 0], [0, 0, -1], (m) => m === shown)).toBe(null);
  });

  it("gives_up_after_six_hidden_layers", async () => {
    const { scene, calls } = layeredScene(Array.from({ length: 8 }, (_, k) => ({ mesh: hidden, distance: k + 1 })));
    expect(await raycastVisibleSurface(scene, [0, 0, 0], [0, 0, -1], () => false)).toBe(null);
    expect(calls).toHaveLength(6);
  });
});

describe("rayThroughPointer", () => {
  const canvas = {
    getBoundingClientRect: () => ({ left: 10, top: 20, width: 200, height: 100 }),
  } as unknown as HTMLCanvasElement;
  const camera = new THREE.PerspectiveCamera(45, 2, 0.1, 100);
  camera.position.set(0, 0, 5);
  camera.lookAt(0, 0, 0);
  camera.updateMatrixWorld();

  it("aims_the_canvas_center_along_the_view", () => {
    const ray = rayThroughPointer(new THREE.Raycaster(), canvas, camera, 110, 70);
    expect(ray.origin.toArray()).toEqual([0, 0, 5]);
    expect(ray.direction.x).toBeCloseTo(0, 9);
    expect(ray.direction.y).toBeCloseTo(0, 9);
    expect(ray.direction.z).toBeCloseTo(-1, 9);
  });

  it("aims_the_top_left_corner_up_and_left", () => {
    const ray = rayThroughPointer(new THREE.Raycaster(), canvas, camera, 10, 20);
    expect(ray.direction.x).toBeLessThan(0);
    expect(ray.direction.y).toBeGreaterThan(0);
    expect(ray.direction.z).toBeLessThan(0);
  });

  it("casts_that_ray_past_hidden_meshes", async () => {
    const { scene, calls } = layeredScene([
      { mesh: hidden, distance: 1 },
      { mesh: shown, distance: 2 },
    ]);
    const hit = await castVisibleSurface(scene, new THREE.Raycaster(), canvas, camera, 110, 70, (m) => m === shown);
    expect(hit?.mesh).toBe(shown);
    expect(calls[0].origin).toEqual([0, 0, 5]);
    expect(calls[0].direction[2]).toBeCloseTo(-1, 9);
  });
});

describe("hit_in_local_positions", () => {
  const hit = { mesh: shown, triangle: 1, barycentric: [0.2, 0.5, 0.3] as [number, number, number] };

  it("snaps_to_the_corner_with_the_largest_weight", () => {
    expect(nearestCorner(triangleScene, hit)).toBe(5);
    expect(nearestCorner(triangleScene, { ...hit, barycentric: [0.6, 0.2, 0.2] })).toBe(4);
    expect(nearestCorner(triangleScene, { ...hit, barycentric: [0.1, 0.3, 0.6] })).toBe(6);
  });

  it("interpolates_the_corners_by_their_weights", () => {
    const [x, y, z] = localHitPoint(triangleScene, hit);
    expect(x).toBeCloseTo(0.2, 9);
    expect(y).toBeCloseTo(1, 9);
    expect(z).toBeCloseTo(1.2, 9);
  });
});
