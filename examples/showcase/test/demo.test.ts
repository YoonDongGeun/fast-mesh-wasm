// The generated demo meshes: A and B closed and consistently oriented (every
// edge used once each way), B penetrating A, C with holes and a floating piece.
import { describe, expect, it } from "vitest";
import { blobbySphere, demoMeshes, openShellWithPiece, roundedBox } from "../src/scene/demo";
import { bounds, type MeshData } from "../src/load/mesh-data";

/** Directed edges used more than once, and edges with no twin (boundary edges). */
function edges(m: MeshData): { repeated: number; boundary: number } {
  const directed = new Map<string, number>();
  for (let t = 0; t < m.indices.length; t += 3) {
    for (let k = 0; k < 3; k++) {
      const key = `${m.indices[t + k]},${m.indices[t + ((k + 1) % 3)]}`;
      directed.set(key, (directed.get(key) ?? 0) + 1);
    }
  }
  let repeated = 0;
  let boundary = 0;
  for (const [key, n] of directed) {
    if (n > 1) {
      repeated++;
    }
    const [a, b] = key.split(",");
    if (!directed.has(`${b},${a}`)) {
      boundary++;
    }
  }
  return { repeated, boundary };
}

/** Connected components by shared vertices. */
function components(m: MeshData): number {
  const parent = Array.from({ length: m.positions.length / 3 }, (_, i) => i);
  const find = (i: number): number => (parent[i] === i ? i : (parent[i] = find(parent[i])));
  for (let t = 0; t < m.indices.length; t += 3) {
    parent[find(m.indices[t + 1])] = find(m.indices[t]);
    parent[find(m.indices[t + 2])] = find(m.indices[t]);
  }
  const used = new Set<number>();
  for (const i of m.indices) {
    used.add(find(i));
  }
  return used.size;
}

/** The signed volume: positive for a closed mesh with outward (counter-clockwise) triangles. */
function volume(m: MeshData): number {
  let v = 0;
  const p = m.positions;
  for (let t = 0; t < m.indices.length; t += 3) {
    const [a, b, c] = [m.indices[t], m.indices[t + 1], m.indices[t + 2]].map((i) => 3 * i);
    v +=
      p[a] * (p[b + 1] * p[c + 2] - p[b + 2] * p[c + 1]) -
      p[a + 1] * (p[b] * p[c + 2] - p[b + 2] * p[c]) +
      p[a + 2] * (p[b] * p[c + 1] - p[b + 1] * p[c]);
  }
  return v / 6;
}

describe("demo meshes", () => {
  it("A, a blobby sphere, is closed and outward", () => {
    const m = blobbySphere([0, 0, 0], 1, 32, 16);
    expect(edges(m)).toEqual({ repeated: 0, boundary: 0 });
    expect(components(m)).toBe(1);
    expect(volume(m)).toBeGreaterThan(3);
  });

  it("B, a rounded box, is closed and outward", () => {
    const m = roundedBox([0, 0, 0], [1, 1, 1], 0.2, 8);
    expect(edges(m)).toEqual({ repeated: 0, boundary: 0 });
    expect(components(m)).toBe(1);
    // A 2 x 2 x 2 box with rounded edges: a bit under 8.
    expect(volume(m)).toBeGreaterThan(7);
    expect(volume(m)).toBeLessThan(8);
  });

  it("C has holes and a floating piece", () => {
    const m = openShellWithPiece([0, 0, 0], 1);
    const e = edges(m);
    expect(e.repeated).toBe(0);
    expect(e.boundary).toBeGreaterThan(0);
    expect(components(m)).toBe(2);
  });

  it("B overlaps A, and ?big makes both about a million triangles", () => {
    const [a, b, c] = demoMeshes(false);
    expect([a.name, b.name, c.name].map((n) => n[0])).toEqual(["A", "B", "C"]);
    const ba = bounds(a.data.positions);
    const bb = bounds(b.data.positions);
    expect(Math.hypot(ba.center[0] - bb.center[0], ba.center[1] - bb.center[1], ba.center[2] - bb.center[2])).toBeLessThan(
      ba.radius / 2 + bb.radius / 2,
    );
    const big = demoMeshes(true);
    for (const m of big.slice(0, 2)) {
      expect(m.data.indices.length / 3).toBeGreaterThan(900_000);
      expect(m.data.indices.length / 3).toBeLessThan(1_200_000);
    }
  });
});
