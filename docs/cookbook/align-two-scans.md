# Align two scans

`scene.register(moving, fixed, options)` rigidly moves one mesh onto another
by point-to-plane ICP. The closest points come from the fixed mesh's BVH. Use
it to stitch two scans of one object, or to compare two measurements of it.

ICP is local: **the initial guess is the current placements**, and it finds the
nearest fit. Put the meshes roughly in place first with `setPlacement`.

## Align and undo

```ts
import { init, Scene } from "fast-mesh-wasm";

await init();
const scene = new Scene();

// A height field over [x0, x0 + n] x [0, n]: the "scans" of one surface.
function scan(x0: number, n: number) {
  const positions: number[] = [];
  const indices: number[] = [];
  for (let j = 0; j <= n; j++) {
    for (let i = 0; i <= n; i++) {
      const [x, y] = [x0 + i, j];
      positions.push(x, y, 2 * Math.sin(0.4 * x + 0.1 * y) + 1.5 * Math.cos(0.3 * y - 0.01 * x * x));
    }
  }
  for (let j = 0; j < n; j++) {
    for (let i = 0; i < n; i++) {
      const a = j * (n + 1) + i;
      indices.push(a, a + 1, a + n + 2, a, a + n + 2, a + n + 1);
    }
  }
  return { positions, indices };
}

const whole = scan(0, 40);
const fixed = scene.addMesh(whole.positions, whole.indices);
const moving = scene.addMesh(whole.positions, whole.indices);

// The rough initial guess: off by half a cell and a couple of degrees.
const [s, c] = [Math.sin(0.03), Math.cos(0.03)];
const guess = {
  rotation: [c, -s, 0, s, c, 0, 0, 0, 1],
  translation: [0.5, -0.3, 0.2] as [number, number, number],
};
scene.setPlacement(moving, guess);

const r = scene.register(moving, fixed);
if (!r.converged || r.rmse > 1e-4) {
  throw new Error(`alignment failed: ${r.status}, rmse ${r.rmse}`);
}
console.log(r.placement, r.rmse, r.iterations);

// The result was applied as one undo step.
scene.undo();
if (scene.placement(moving).translation[0] !== guess.translation[0]) {
  throw new Error("undo should restore the guess");
}
scene.redo();
```

`converged` says the iteration settled and the surface determined every rigid
motion. It does not say the fit is the right one: read `rmse` for that, in world
units, over the points used.

## Partial overlap

Two scans usually overlap only in part. Set `overlap` a little below the
fraction of the moving mesh that lies over the fixed one. Each iteration then
uses that many points, closest first, so the part beyond the overlap and any
outliers do not pull the result. `maxDistance` drops far points outright.

```ts
// Covers x in [12, 52]: 70% of it lies over the fixed scan's [0, 40].
const part = scan(12, 40);
const second = scene.addMesh(part.positions, part.indices);
scene.setPlacement(second, guess);

const trimmed = scene.register(second, fixed, { overlap: 0.6, apply: false });
if (!trimmed.converged || trimmed.rmse > 1e-3) {
  throw new Error(`partial alignment failed: ${trimmed.status}, rmse ${trimmed.rmse}`);
}
console.log(`${trimmed.inliers} of ${trimmed.samples} points used`);
```

## Return only, and apply yourself

`apply: false`, as above, leaves the scene alone. A result that did not converge
is never applied, with or without it, but `placement` still holds the best
estimate:

```ts
if (!trimmed.applied) {
  scene.setPlacement(second, trimmed.placement);
}
```

## When it cannot converge

Some surfaces do not determine the alignment. A plane can slide anywhere along
itself, and a tube along its axis. `register` leaves such motions where the
placement put them and reports `status: "underconstrained"`, with
`converged: false`, rather than making up an answer:

```ts
const quad = [-50, -50, 0, 50, -50, 0, 50, 50, 0, -50, 50, 0];
const quadTris = [0, 1, 2, 0, 2, 3];
const ground = scene.addMesh(quad, quadTris);
const tile = scene.addMesh(quad.map((v) => v / 10), quadTris);
scene.setPlacement(tile, { rotation: [1, 0, 0, 0, 1, 0, 0, 0, 1], translation: [7, 3, 0.5] });

const sliding = scene.register(tile, ground);
if (sliding.converged || sliding.status !== "underconstrained") {
  throw new Error("a plane on a plane must not report convergence");
}
```

`conditioning` tells how well the worst-determined motion is fixed. A sphere's
spin is fixed only by its facets, so a sphere recovers its center reliably while
its rotation means little, and the number falls the finer it is tessellated.

## Scales

Only rotation and translation are solved. The moving mesh's scale is applied to
its points and kept in the result. The fixed mesh's scale, uniform or not, is
followed exactly on a baked copy, which costs a BVH build per call.

## On `WorkerScene`

`register` returns a promise there, and runs with the placements the Worker has
when the request reaches it. An applied result reaches `placement()` with its
batch, like a `setPlacement`.
