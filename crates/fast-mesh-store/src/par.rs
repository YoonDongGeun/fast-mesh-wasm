//! Deterministic data-parallel helpers shared by the engine crates.
//!
//! With the `parallel` feature, work at or above a caller-chosen threshold runs
//! on rayon's pool. Without it, below the threshold, or while a test fault hook
//! forces it off, the same work runs in a loop on the calling thread. Either way
//! the work is cut into the same fixed chunks and the results come back in chunk
//! order, so a caller that combines them in that order gets bit-identical
//! results with the feature on or off and for any thread count.
//!
//! The signatures are the same with and without the feature. Not part of the
//! public API: other engine crates use it, applications should not.

use std::cell::Cell;
use std::ops::Range;

thread_local! {
    /// How many [`set_force_sequential`]`(true)` calls on this thread are not
    /// yet matched by a `false`: one per armed test fault hook.
    static FORCE_SEQUENTIAL: Cell<usize> = const { Cell::new(0) };
}

#[cfg(any(test, feature = "test-counters"))]
thread_local! {
    /// Helper calls from this thread that took the parallel path.
    static PARALLEL_ENTRIES: Cell<usize> = const { Cell::new(0) };
}

/// True when the crate was built with the `parallel` feature.
pub fn enabled() -> bool {
    cfg!(feature = "parallel")
}

/// The threads the helpers called from this thread would run work on: the
/// current pool's size with the `parallel` feature (unless forced off), and
/// one otherwise. For sizing work per step, never for cutting work into
/// chunks (results must not depend on it).
pub fn threads() -> usize {
    #[cfg(feature = "parallel")]
    if !force_sequential() {
        return rayon::current_num_threads().max(1);
    }
    1
}

/// Makes every helper called from this thread run sequentially (`true`), or
/// undoes one earlier `true` (`false`). The engine crates' out-of-memory
/// fault hooks are thread-local, so their test helpers set this while a hook
/// is armed: work handed to other threads would not see the hook.
///
/// Counted, not a flag: helpers run sequentially until every `true` has been
/// matched by a `false`, so one hook (or helper) turning itself off cannot
/// turn off another that is still armed. A `false` with no `true` left does
/// nothing. Thread-local for the same reason as the hooks, so tests running
/// side by side cannot turn each other's off.
pub fn set_force_sequential(on: bool) {
    FORCE_SEQUENTIAL.with(|c| {
        c.set(if on {
            c.get() + 1
        } else {
            c.get().saturating_sub(1)
        })
    });
}

/// Whether some [`set_force_sequential`]`(true)` on this thread is not yet
/// undone.
pub fn force_sequential() -> bool {
    FORCE_SEQUENTIAL.with(|c| c.get() > 0)
}

/// How many helper calls from this thread have taken the parallel path, for
/// tests that check small inputs stay sequential.
#[cfg(any(test, feature = "test-counters"))]
pub fn parallel_entries() -> usize {
    PARALLEL_ENTRIES.with(Cell::get)
}

/// Whether work of this size goes to the pool; counts it when it does.
#[cfg(feature = "parallel")]
fn go_parallel(large_enough: bool) -> bool {
    let parallel = large_enough && !force_sequential();
    #[cfg(any(test, feature = "test-counters"))]
    if parallel {
        PARALLEL_ENTRIES.with(|c| c.set(c.get() + 1));
    }
    parallel
}

/// Calls `f` on `0..len` cut into consecutive ranges of `chunk` (the last
/// may be shorter) and returns the results in range order. Runs in parallel
/// when `len >= threshold` and there is more than one range.
///
/// When several ranges fail, the error of the first one is returned, however
/// the work was scheduled. Ranges after a known failure may be skipped. Fails
/// with `oom()` if the vectors it allocates (one or two, `len / chunk` long)
/// cannot be allocated. Those allocations do not pass through the crates'
/// fault-injection hooks; callers that need that coverage check their hook
/// before calling.
///
/// # Panics
/// If `chunk` is 0.
pub fn map_range<R: Send, E: Send>(
    len: usize,
    chunk: usize,
    threshold: usize,
    oom: impl Fn() -> E,
    f: impl Fn(Range<usize>) -> Result<R, E> + Sync,
) -> Result<Vec<R>, E> {
    assert!(chunk > 0, "chunk must be positive");
    let count = len.div_ceil(chunk);
    let range = |i: usize| i * chunk..((i + 1) * chunk).min(len);
    let mut out = Vec::new();
    out.try_reserve_exact(count).map_err(|_| oom())?;
    #[cfg(feature = "parallel")]
    if go_parallel(len >= threshold && count > 1) {
        use rayon::prelude::*;
        use std::sync::atomic::{AtomicUsize, Ordering};

        let mut slots: Vec<Option<Result<R, E>>> = Vec::new();
        slots.try_reserve_exact(count).map_err(|_| oom())?;
        slots.resize_with(count, || None);
        let first_error = AtomicUsize::new(usize::MAX);
        slots.par_iter_mut().enumerate().for_each(|(i, slot)| {
            if i > first_error.load(Ordering::Relaxed) {
                return;
            }
            let result = f(range(i));
            if result.is_err() {
                first_error.fetch_min(i, Ordering::Relaxed);
            }
            *slot = Some(result);
        });
        for slot in slots {
            match slot {
                Some(Ok(r)) => out.push(r),
                Some(Err(e)) => return Err(e),
                // A range is skipped only after an earlier one failed, and
                // that failure is returned first.
                None => unreachable!("skipped range before any failure"),
            }
        }
        return Ok(out);
    }
    #[cfg(not(feature = "parallel"))]
    let _ = threshold;
    for i in 0..count {
        out.push(f(range(i))?);
    }
    Ok(out)
}

/// [`map_range`] over `items`: `f` gets each chunk's index and its items.
pub fn map_chunks<T: Sync, R: Send, E: Send>(
    items: &[T],
    chunk: usize,
    threshold: usize,
    oom: impl Fn() -> E,
    f: impl Fn(usize, &[T]) -> Result<R, E> + Sync,
) -> Result<Vec<R>, E> {
    map_range(items.len(), chunk, threshold, oom, |r| {
        f(r.start / chunk, &items[r])
    })
}

/// Calls `f` on `items` cut into consecutive chunks of `chunk` (the last may
/// be shorter), with each chunk's index, in parallel when
/// `items.len() >= threshold` and there is more than one chunk. Allocates
/// nothing, so it cannot fail; for work that writes each item in place.
///
/// # Panics
/// If `chunk` is 0.
pub fn for_each_chunk_mut<T: Send>(
    items: &mut [T],
    chunk: usize,
    threshold: usize,
    f: impl Fn(usize, &mut [T]) + Sync,
) {
    assert!(chunk > 0, "chunk must be positive");
    #[cfg(feature = "parallel")]
    if go_parallel(items.len() >= threshold && items.len() > chunk) {
        use rayon::prelude::*;
        items
            .par_chunks_mut(chunk)
            .enumerate()
            .for_each(|(i, c)| f(i, c));
        return;
    }
    #[cfg(not(feature = "parallel"))]
    let _ = threshold;
    for (i, c) in items.chunks_mut(chunk).enumerate() {
        f(i, c);
    }
}

/// Runs `a` and `b`, in parallel when `parallel` is true (and the feature is
/// on and not forced off), otherwise `a` first. Both always run.
pub fn join<A: Send, B: Send>(
    parallel: bool,
    a: impl FnOnce() -> A + Send,
    b: impl FnOnce() -> B + Send,
) -> (A, B) {
    #[cfg(feature = "parallel")]
    if go_parallel(parallel) {
        return rayon::join(a, b);
    }
    #[cfg(not(feature = "parallel"))]
    let _ = parallel;
    (a(), b())
}

/// Runs `f` on a fresh pool of `threads` threads (with the feature), or just
/// runs it. For tests and benches that compare thread counts.
///
/// # Panics
/// If the pool cannot be created.
#[cfg(feature = "parallel")]
pub fn with_threads<R: Send>(threads: usize, f: impl FnOnce() -> R + Send) -> R {
    rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build()
        .expect("thread pool")
        .install(f)
}

/// Runs `f` on a fresh pool of `threads` threads (with the feature), or just
/// runs it. For tests and benches that compare thread counts.
#[cfg(not(feature = "parallel"))]
pub fn with_threads<R: Send>(_threads: usize, f: impl FnOnce() -> R + Send) -> R {
    f()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::StoreError;

    /// Thread counts every pool-dependent test runs under.
    const POOLS: [usize; 3] = [1, 2, 8];

    const OOM: fn() -> StoreError = || StoreError::OutOfMemory;

    #[test]
    fn map_chunks_order_and_results() {
        let items: Vec<u64> = (0..1000).collect();
        let expected: Vec<(usize, u64)> = items
            .chunks(64)
            .enumerate()
            .map(|(i, c)| (i, c.iter().sum()))
            .collect();
        for k in POOLS {
            let got = with_threads(k, || {
                map_chunks(&items, 64, 0, OOM, |i, c| {
                    Ok::<_, StoreError>((i, c.iter().sum::<u64>()))
                })
            });
            assert_eq!(got.unwrap(), expected, "{k} threads");
            let ranges = with_threads(k, || map_range(1000, 64, 0, OOM, Ok));
            let want: Vec<Range<usize>> =
                (0..16).map(|i| i * 64..(i * 64 + 64).min(1000)).collect();
            assert_eq!(ranges.unwrap(), want, "{k} threads");
        }
        assert!(map_range(0, 8, 0, OOM, Ok).unwrap().is_empty());
    }

    #[test]
    fn lowest_index_error_wins() {
        let items = vec![0u8; 20 * 16];
        for k in POOLS {
            let got = with_threads(k, || {
                map_chunks(&items, 16, 0, OOM, |i, _| {
                    if i == 7 {
                        // Give chunk 12 every chance to fail first.
                        std::thread::sleep(std::time::Duration::from_millis(20));
                    }
                    if i == 7 || i == 12 {
                        Err(StoreError::InvalidVertex(i as u32))
                    } else {
                        Ok(i)
                    }
                })
            });
            assert_eq!(got, Err(StoreError::InvalidVertex(7)), "{k} threads");
        }
    }

    #[test]
    fn below_threshold_is_sequential() {
        let items = vec![1u32; 4096];
        let sum = |_: usize, c: &[u32]| Ok::<_, StoreError>(c.iter().sum::<u32>());
        let before = parallel_entries();
        assert_eq!(map_chunks(&items, 256, 4097, OOM, sum).unwrap().len(), 16);
        assert_eq!(map_chunks(&items, 4096, 0, OOM, sum).unwrap(), vec![4096]);
        assert_eq!(join(false, || 1, || 2), (1, 2));
        assert_eq!(parallel_entries(), before);

        set_force_sequential(true);
        assert!(force_sequential());
        map_chunks(&items, 256, 0, OOM, sum).unwrap();
        join(true, || 1, || 2);
        set_force_sequential(false);
        assert_eq!(parallel_entries(), before);

        // At the threshold the parallel path is taken, with the feature only.
        assert_eq!(
            map_chunks(&items, 256, 4096, OOM, sum).unwrap(),
            vec![256; 16]
        );
        assert_eq!(join(true, || 1, || 2), (1, 2));
        let entered = if enabled() { 2 } else { 0 };
        assert_eq!(parallel_entries(), before + entered);
    }

    #[test]
    fn for_each_chunk_mut_writes_every_chunk() {
        let expected: Vec<(usize, usize)> = (0..1000).map(|j| (j / 64, j)).collect();
        for k in POOLS {
            let mut items = vec![(usize::MAX, usize::MAX); 1000];
            with_threads(k, || {
                for_each_chunk_mut(&mut items, 64, 0, |i, c| {
                    for (j, x) in c.iter_mut().enumerate() {
                        *x = (i, i * 64 + j);
                    }
                })
            });
            assert_eq!(items, expected, "{k} threads");
        }
        let before = parallel_entries();
        let mut items = vec![0u8; 4096];
        for_each_chunk_mut(&mut items, 256, 4097, |_, c| c.fill(1));
        for_each_chunk_mut(&mut items, 4096, 0, |_, c| c.fill(2));
        set_force_sequential(true);
        for_each_chunk_mut(&mut items, 256, 0, |_, c| c.fill(3));
        set_force_sequential(false);
        assert_eq!(parallel_entries(), before);
        assert!(items.iter().all(|&x| x == 3));
        for_each_chunk_mut(&mut items, 256, 4096, |_, c| c.fill(4));
        assert!(items.iter().all(|&x| x == 4));
        let entered = if enabled() { 1 } else { 0 };
        assert_eq!(parallel_entries(), before + entered);
    }

    #[test]
    fn force_sequential_is_counted() {
        assert!(!force_sequential());
        // Two hooks armed; the first one turned off leaves the second on.
        set_force_sequential(true);
        set_force_sequential(true);
        set_force_sequential(false);
        assert!(force_sequential());
        let before = parallel_entries();
        assert_eq!(join(true, || 1, || 2), (1, 2));
        assert_eq!(parallel_entries(), before);
        set_force_sequential(false);
        assert!(!force_sequential());
        // An unmatched `false` does not go below zero.
        set_force_sequential(false);
        set_force_sequential(true);
        assert!(force_sequential());
        set_force_sequential(false);
        assert!(!force_sequential());
    }

    #[test]
    fn enabled_follows_the_feature() {
        assert_eq!(enabled(), cfg!(feature = "parallel"));
    }
}
