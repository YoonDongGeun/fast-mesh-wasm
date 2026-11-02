# Cookbook

Task-oriented recipes for `fast-mesh-wasm`. Each recipe is runnable
against the built package and shows the minimal calls for one job.

| Recipe | What it does |
| --- | --- |
| [Load → repair → color → export](load-repair-color-export.md) | The whole session end to end: glTF in, repair, paint, STL out |
| [Measure → ruler / query](measure-ruler.md) | `measure`, `meshDistance`, `closestPoint`, `raycast` |
| [Headless Node pipeline](headless-node.md) | Sync `Scene` in Node, plus the `createWorker` adapter for `WorkerScene` |
| [IO: export / import](io-export-import.md) | `exportSTL` / `OBJ` / `PLY` / `glTF`, `importGLTF`, Draco |
| [Selection + mask](selection-mask.md) | Brush / lasso / component picks, grow / shrink / invert, `isolate` |
| [Auto-repair](auto-repair.md) | `analyze` → `repair` pipeline |
| [Contact heatmap of two meshes](contact-heatmap.md) | Distance field + `setColoring` between a pair |
| [Custom colormap + fragment shading](custom-colormap.md) | Custom stops, `center`, fragment shading via `bindMesh` |

Conventions used in every recipe:

```ts
import { Scene } from "fast-mesh-wasm"; // everywhere: browsers, Node.js, Bun (await init() first; a no-op in Node.js)
```

The blocks of one recipe share a scope and build top to bottom: later
blocks reuse the scene and meshes the first ones set up. Every block
runs in CI (`packages/fast-mesh-wasm/test/cookbook.test.ts`
concatenates each file's fences and imports them against the built
package), except fences tagged `no-run`, which need reader-supplied
data and are listed in that test.

Recipes run on the sync `Scene`. On `WorkerScene` the queries and the
edits are async instead (`analyze`, `repair`, `raycast`,
`closestPoint`, `meshDistance`, `findContacts`, ...); `measure` and
`select` exist only on `Scene`. The headless-Node recipe covers the
difference.
