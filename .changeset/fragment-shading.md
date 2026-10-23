---
"fast-mesh-wasm": minor
---

Add fragment shading for colourings, so the edge between coloured and uncoloured areas is crisp instead of blurred across a triangle. `bindMesh` (`fast-mesh-wasm/three`) takes `shading: "vertex" | "fragment"` (default `"vertex"`, unchanged) and `edgeSoftness` (distance units, 0 = a hard edge; `BoundMesh.setEdgeSoftness` changes it without a recompile). In `"fragment"` mode each pixel is coloured from the interpolated field value through the colouring's rule (a lookup texture and a shader chunk added with `onBeforeCompile`), so range edges and a contact heatmap's tolerance band are exact iso-lines and gradients are smooth per pixel; at a vertex the colour equals the vertex colour, and far (`Infinity`) and NaN values never tint their neighbours. `<FastMesh>` passes `shading` and `edgeSoftness` through.

Add `views(mesh).values()`: the values a mesh's colours are made from (one `f32` per vertex slot, `null` without colouring), changing only at the colours' dirty ranges and mirrored by `WorkerScene`; and `coloring(mesh)` on `Scene` and `WorkerScene`, the mesh's colour rule. On `SceneLike` and `MeshViews` both are optional. Colours of a reused vertex slot are now reported (and recoloured from its reset value).
