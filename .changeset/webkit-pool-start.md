---
"fast-mesh-wasm": patch
---

Multithreaded Worker scene: on WebKit 26 (as in Playwright 1.56's build; Safari 26 may be affected), the scene now starts single-threaded in about 50 ms instead of after the 10 s thread-pool timeout. WebKit 26 traps `memory.fill` and `memory.copy` on a shared memory that another thread grew, so its thread pool could never start. A probe of about 10 ms, run while the threaded glue is imported, now detects this before the glue is instantiated. The verdict is kept for the page, so restarts and later scenes skip the probe. Other browsers are unchanged and keep the threaded engine. See "Older WebKit" in the README.
