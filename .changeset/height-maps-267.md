---
"fast-mesh-wasm": minor
---

Add height maps and reliefs (#267): `Scene.heightMapFrame(meshes, options)` fits an orthographic image of square pixels to one mesh or several, looking along any direction, and `Scene.heightMap(meshes, frame, rows)` samples the height of each pixel, whole or in bands of rows (the bands hold exactly what the whole would). Rays run by tiles that share their walk of each BVH, in parallel in the threaded build, and the heights are the same bit for bit either way. `Scene.addHeightfield(field, options)` adds a grid of heights as a closed, outward-wound relief with a flat base as one undo step, and `heightMapPlacement(frame)` puts a relief of a height map where the map saw its surface. `generateHeightfield(field)` returns a relief's buffers without a scene. All are on `WorkerScene` too. In Rust: `fast_mesh_query::RayGrid`, `fast_mesh_ops::heightfield`, `Scene::height_map_frame`, `Scene::height_map` and `Scene::add_heightfield`.
