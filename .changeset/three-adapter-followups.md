---
"fast-mesh-wasm": patch
---

three.js adapter follow-ups ([#43](https://github.com/YoonDongGeun/fast-mesh-wasm/issues/43)). `bindMesh` takes its bounding box and sphere from the scene's live-vertex bounds instead of walking every slot twice per edit frame, so sculpt-style drags no longer pay a JS-side walk per frame; on a host without `SceneLike.bounds` (now optional) it walks the mirrors as before. Bound objects and section groups must sit under an untransformed parent (their matrices are driven as world matrices): a transformed ancestor is now detected and warned about once instead of silently offsetting them. CI also runs the adapter tests against `three@0.160`, the peer floor.
