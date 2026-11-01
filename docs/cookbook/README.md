# Cookbook

Task-oriented recipes for `fast-mesh-wasm`. Each recipe is runnable
against the built package and shows the minimal calls for one job.

| Recipe | What it does |
| --- | --- |
| [Measure → ruler / query](measure-ruler.md) | `measure`, `meshDistance`, `closestPoint`, `raycast` |
| [Headless Node pipeline](headless-node.md) | Sync `Scene` in Node, plus the `createWorker` adapter for `WorkerScene` |
| [IO: export / import](io-export-import.md) | `exportSTL` / `OBJ` / `PLY` / `glTF`, `importGLTF`, Draco |
| [Selection + mask](selection-mask.md) | Brush / lasso / component picks, grow / shrink / invert, `isolate` |
| [Auto-repair](auto-repair.md) | `analyze` → `repair` pipeline |
| [Contact heatmap of two meshes](contact-heatmap.md) | Distance field + `setColoring` between a pair |
| [Custom colormap + fragment shading](custom-colormap.md) | Custom stops, `center`, fragment shading via `bindMesh` |

Conventions used in every recipe:

```ts
import { Scene } from "fast-mesh-wasm"; // browser / bundler (await init() first)
import { Scene } from "fast-mesh-wasm/node"; // Node.js / Bun (wasm loads on import)
```

`Scene` and `WorkerScene` share the `SceneLike` surface the recipes
use (`addMesh`, `measure`, `select`, fields, `setColoring`, `views`,
`matrix4`). `WorkerScene` versions of the state-changing calls are
async where noted; the headless-Node recipe covers the difference.
