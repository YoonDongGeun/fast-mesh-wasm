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

// Applied only if it converged and its rmse is within `maxRmse` (by default 1%
// of the moving mesh's extent). Tighten it for your data: it is checked before
// anything moves, so a failure leaves the mesh where it was.
const r = scene.register(moving, fixed, { maxRmse: 1e-3 });
if (!r.applied) {
  throw new Error(`alignment failed: ${r.status}, ${r.notApplied}, rmse ${r.rmse}`);
}
console.log(r.placement, r.rmse, r.iterations);

// The result was applied as one undo step.
scene.undo();
if (scene.placement(moving).translation[0] !== guess.translation[0]) {
  throw new Error("undo should restore the guess");
}
scene.redo();
```

`converged` says the iteration settled and the data determine every rigid motion
locally. It does not say the fit is the right one: a far guess can settle in a
wrong minimum, which only `rmse` shows (world units, over the points used). That
is what `maxRmse` is for, and `notApplied` says when it stopped a result.

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

`apply: false`, as above, leaves the scene alone, so you can check the result
with your own rule first. A result that is not good is never applied, with or
without it, but `placement` still holds the estimate:

```ts
if (trimmed.notApplied === "notRequested") {
  scene.setPlacement(second, trimmed.placement);
}
```

## When it cannot converge

Some surfaces do not determine the alignment. A plane can slide anywhere along
itself, a tube along its axis, and a sphere or a cylinder can spin. Facets or
noise can still give the iteration something to settle on, with a tiny `rmse`, so
`register` checks: it displaces the result along each motion and iterates again.
If a restart stays away with a fit as good, the data do not choose. It reports
`status: "underconstrained"` and `converged: false`, moves nothing, and lists the
motions in `undetermined`:

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
// Three motions: the two slides and the spin about the plane's normal.
for (const m of sliding.undetermined) {
  console.log(m.translation, m.rotation, m.about);
}
```

The check is local. Regular meshes with exact fits close by count as undetermined
(a 16-sided sphere fits every 22.5 degrees), while a cube's 90-degree symmetry does
not. `conditioning` is a geometric warning only: a high value does not mean a
motion is determined.

## Scales

Only rotation and translation are solved. The moving mesh's scale is applied to
its points and kept in the result. The fixed mesh's scale, uniform or not, is
followed exactly on a baked copy, which costs a BVH build per call.

## On `WorkerScene`

`register` returns a promise there, and runs with the placements the Worker has
when the request reaches it. An applied result reaches `placement()` with its
batch, like a `setPlacement`.
