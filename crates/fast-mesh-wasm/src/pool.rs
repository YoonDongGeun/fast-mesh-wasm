//! Crash reports from rayon's pool threads (threaded artifact only).
//!
//! Each pool thread runs in a nested Web Worker. A panic there (under
//! `panic = "abort"`) or a trap (`unreachable`, an out-of-bounds access, a
//! failed allocation's abort, the JavaScript call stack running out) ends
//! only that Worker's call into rayon's loop; the thread that waits for its
//! job, the engine Worker, stays blocked in `memory.atomic.wait` and can
//! never report it. So the pool's threads report it themselves, on a
//! `BroadcastChannel` the scene on the page listens to: the panic hook posts
//! the panic message before the abort, and an `error` / `unhandledrejection`
//! listener on each pool Worker posts any trap (the abort that follows a
//! panic too; the scene takes the first report).

use std::sync::OnceLock;

use wasm_bindgen::prelude::*;

#[wasm_bindgen(inline_js = r#"
const describe = (e) => (e instanceof Error ? `${e.name}: ${e.message}` : String(e));
export function post_crash(channel, detail) {
  try {
    const c = new BroadcastChannel(channel);
    c.postMessage({ kind: "poolCrash", detail });
    c.close();
  } catch {
    // Nothing to report on: the scene cannot hear this thread.
  }
}
export function watch_worker(channel) {
  // The thread's loop runs in an async message handler: a trap rejects it.
  addEventListener("unhandledrejection", (e) => post_crash(channel, describe(e.reason)));
  addEventListener("error", (e) => post_crash(channel, e.error === undefined ? String(e.message) : describe(e.error)));
}
"#)]
extern "C" {
    fn post_crash(channel: &str, detail: &str);
    fn watch_worker(channel: &str);
}

/// The channel crashes are reported on, set once by [`watch_pool_crashes`].
static CHANNEL: OnceLock<String> = OnceLock::new();

/// `watchPoolCrashes(channel)`: from now on, a panic or a trap on a thread of
/// rayon's pool posts `{ kind: "poolCrash", detail }` on the
/// `BroadcastChannel` named `channel`. Call once, after `initThreadPool`, on
/// the thread that started the pool; later calls do nothing. Costs nothing
/// until a thread crashes.
#[wasm_bindgen(js_name = watchPoolCrashes)]
pub fn watch_pool_crashes(channel: String) {
    if CHANNEL.set(channel).is_err() {
        return;
    }
    let previous = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        // Only pool threads: a panic on the engine's own thread traps there,
        // and the engine reports it (`INSTANCE_POISONED`).
        if rayon::current_thread_index().is_some() {
            if let Some(channel) = CHANNEL.get() {
                post_crash(channel, &info.to_string());
            }
        }
        previous(info);
    }));
    let channel = CHANNEL.get().map(String::as_str).unwrap_or_default();
    // Every pool thread installs its Worker's listeners (the threads are idle).
    rayon::broadcast(|_| watch_worker(channel));
}
