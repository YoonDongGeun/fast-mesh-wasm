// The built-in stamps Place Mesh puts on a surface: closed, outward-wound
// primitives along +Z with their base at z = 0 (a peg, a hex boss, a dome
// and a cube), as plain position and index buffers for the engine.

export type BuiltinStampKind = "peg" | "hexBoss" | "dome" | "cube";

function closedPrismGeometry(sides: number, radius: number, height: number): {
  positions: Float64Array;
  indices: Uint32Array;
} {
  const positions: number[] = [0, 0, 0, 0, 0, height];
  for (let i = 0; i < sides; i++) {
    const theta = (2 * Math.PI * i) / sides;
    const x = radius * Math.cos(theta);
    const y = radius * Math.sin(theta);
    positions.push(x, y, 0);
  }
  for (let i = 0; i < sides; i++) {
    const theta = (2 * Math.PI * i) / sides;
    const x = radius * Math.cos(theta);
    const y = radius * Math.sin(theta);
    positions.push(x, y, height);
  }
  const indices: number[] = [];
  const botStart = 2;
  const topStart = 2 + sides;
  for (let i = 0; i < sides; i++) {
    const next = (i + 1) % sides;
    const b0 = botStart + i;
    const b1 = botStart + next;
    const t0 = topStart + i;
    const t1 = topStart + next;
    indices.push(0, b1, b0);
    indices.push(1, t0, t1);
    indices.push(b0, b1, t1, b0, t1, t0);
  }
  return {
    positions: Float64Array.from(positions),
    indices: Uint32Array.from(indices),
  };
}

function closedDomeGeometry(radius: number, stacks: number, slices: number): {
  positions: Float64Array;
  indices: Uint32Array;
} {
  const positions: number[] = [0, 0, 0];
  for (let s = 0; s < stacks; s++) {
    const phi = (Math.PI * 0.5 * (stacks - s)) / stacks;
    const r = radius * Math.sin(phi);
    const z = radius * Math.cos(phi);
    for (let j = 0; j < slices; j++) {
      const theta = (2 * Math.PI * j) / slices;
      positions.push(r * Math.cos(theta), r * Math.sin(theta), z);
    }
  }
  positions.push(0, 0, radius);
  const pole = positions.length / 3 - 1;
  const ringVertex = (s: number, j: number) => 1 + s * slices + (j % slices);
  const indices: number[] = [];
  for (let j = 0; j < slices; j++) {
    indices.push(0, ringVertex(0, j + 1), ringVertex(0, j));
  }
  for (let s = 0; s < stacks - 1; s++) {
    for (let j = 0; j < slices; j++) {
      const a = ringVertex(s, j);
      const b = ringVertex(s, j + 1);
      const c = ringVertex(s + 1, j + 1);
      const d = ringVertex(s + 1, j);
      indices.push(a, b, c, a, c, d);
    }
  }
  const topRing = stacks - 1;
  for (let j = 0; j < slices; j++) {
    indices.push(ringVertex(topRing, j), ringVertex(topRing, j + 1), pole);
  }
  return {
    positions: Float64Array.from(positions),
    indices: Uint32Array.from(indices),
  };
}

/**
 * Generates a closed, outward-wound procedural stamp primitive oriented along +Z (base at z=0).
 */
export function createBuiltinStampMeshGeometry(kind: BuiltinStampKind): {
  positions: Float64Array;
  indices: Uint32Array;
} {
  switch (kind) {
    case "peg":
      return closedPrismGeometry(16, 1.5, 4.0);
    case "hexBoss":
      return closedPrismGeometry(6, 2.0, 3.0);
    case "dome":
      return closedDomeGeometry(2.0, 6, 16);
    case "cube": {
      const hx = 1.5;
      const hy = 1.5;
      const z0 = 0;
      const z1 = 3.0;
      const positions = Float64Array.from([
        -hx, -hy, z0,
        hx, -hy, z0,
        hx, hy, z0,
        -hx, hy, z0,
        -hx, -hy, z1,
        hx, -hy, z1,
        hx, hy, z1,
        -hx, hy, z1,
      ]);
      const indices = Uint32Array.from([
        0, 2, 1, 0, 3, 2,
        4, 5, 6, 4, 6, 7,
        0, 1, 5, 0, 5, 4,
        2, 3, 7, 2, 7, 6,
        1, 2, 6, 1, 6, 5,
        3, 0, 4, 3, 4, 7,
      ]);
      return { positions, indices };
    }
  }
}
