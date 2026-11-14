//! Reusable, fallible, thread-local scratch buffer pool.
//!
//! Hot geometry kernels (BVH traversal, distance transforms, Marching Cubes,
//! batch queries) often need temporary `Vec` storage whose lifetime spans a
//! single call. Retaining those buffers in thread-local storage avoids repeated
//! heap allocations on hot paths, while keeping an explicit element cap so a
//! single oversized transient operation does not pin memory forever.

use std::cell::RefCell;
use std::ops::{Deref, DerefMut};
use std::thread::LocalKey;

use crate::column::try_reserve as store_try_reserve;
use crate::{Result, StoreError};

/// Default per-buffer element cap above which a scratch buffer is dropped
/// instead of returned to the thread-local pool.
pub const DEFAULT_MAX_RETAINED_ELEMS: usize = 1 << 20;

/// Maximum number of free buffers retained per pool on a single thread.
const MAX_POOLED_BUFFERS: usize = 8;

/// Descriptor for a thread-local pool of reusable `Vec<T>` scratch buffers.
pub struct ScratchPool<T: 'static> {
    key: &'static LocalKey<RefCell<Vec<Vec<T>>>>,
    max_retained_elems: usize,
}

impl<T: 'static> ScratchPool<T> {
    /// Creates a pool descriptor using [`DEFAULT_MAX_RETAINED_ELEMS`].
    pub const fn new(key: &'static LocalKey<RefCell<Vec<Vec<T>>>>) -> Self {
        Self {
            key,
            max_retained_elems: DEFAULT_MAX_RETAINED_ELEMS,
        }
    }

    /// Creates a pool descriptor with an explicit `max_retained_elems` cap.
    pub const fn with_max_retained(
        key: &'static LocalKey<RefCell<Vec<Vec<T>>>>,
        max_retained_elems: usize,
    ) -> Self {
        Self {
            key,
            max_retained_elems,
        }
    }

    /// Returns the maximum capacity a buffer may have and still be retained.
    pub const fn max_retained_elems(&self) -> usize {
        self.max_retained_elems
    }

    /// Acquires a cleared scratch buffer with room for at least `min_capacity`
    /// elements, consulting `fast-mesh-store`'s fault-injection hook when growth
    /// is required.
    pub fn try_acquire(&'static self, min_capacity: usize) -> Result<ScratchVec<'static, T>> {
        self.try_acquire_with(min_capacity, store_alloc_allowed, || {
            StoreError::OutOfMemory
        })
    }

    /// Acquires a cleared scratch buffer with room for at least `min_capacity`
    /// elements, consulting `check_alloc` whenever heap growth is needed and
    /// mapping any OOM error through `map_oom`.
    ///
    /// If the current thread-local pool is already borrowed (re-entrancy) or
    /// unavailable (thread teardown), falls back to a fresh fallible `Vec<T>`.
    pub fn try_acquire_with<E>(
        &'static self,
        min_capacity: usize,
        mut check_alloc: impl FnMut() -> std::result::Result<(), E>,
        mut map_oom: impl FnMut() -> E,
    ) -> std::result::Result<ScratchVec<'static, T>, E> {
        let mut buf = self
            .key
            .try_with(|cell| {
                if let Ok(mut pool) = cell.try_borrow_mut() {
                    take_matching(&mut pool, min_capacity)
                } else {
                    None
                }
            })
            .ok()
            .flatten()
            .unwrap_or_default();

        buf.clear();
        if buf.capacity() < min_capacity {
            check_alloc()?;
            let additional = min_capacity.saturating_sub(buf.len());
            buf.try_reserve(additional).map_err(|_| map_oom())?;
        }

        Ok(ScratchVec {
            buf,
            pool: Some(self),
        })
    }

    /// Acquires a scratch buffer of exact length `len` initialized to `value`.
    pub fn try_acquire_filled_with<E>(
        &'static self,
        len: usize,
        value: T,
        check_alloc: impl FnMut() -> std::result::Result<(), E>,
        map_oom: impl FnMut() -> E,
    ) -> std::result::Result<ScratchVec<'static, T>, E>
    where
        T: Clone,
    {
        let mut guard = self.try_acquire_with(len, check_alloc, map_oom)?;
        guard.buf.resize(len, value);
        Ok(guard)
    }

    /// Releases all retained buffers on the calling thread.
    pub fn clear(&'static self) {
        let _ = self.key.try_with(|cell| {
            if let Ok(mut pool) = cell.try_borrow_mut() {
                pool.clear();
            }
        });
    }

    #[cfg(test)]
    fn retained_count(&'static self) -> usize {
        self.key
            .try_with(|cell| cell.try_borrow().map(|p| p.len()).unwrap_or(0))
            .unwrap_or(0)
    }

    fn recycle(&self, mut buf: Vec<T>) {
        buf.clear();
        if buf.capacity() == 0 || buf.capacity() > self.max_retained_elems {
            return;
        }
        let _ = self.key.try_with(|cell| {
            if let Ok(mut pool) = cell.try_borrow_mut()
                && pool.len() < MAX_POOLED_BUFFERS
            {
                pool.push(buf);
            }
        });
    }
}

/// Pops the tightest-fitting cached buffer (`cap >= min_capacity`), or else the
/// largest cached buffer so an existing allocation is grown rather than left
/// idle.
fn take_matching<T>(pool: &mut Vec<Vec<T>>, min_capacity: usize) -> Option<Vec<T>> {
    if pool.is_empty() {
        return None;
    }
    let mut best_fit: Option<(usize, usize)> = None;
    let mut largest: (usize, usize) = (0, pool[0].capacity());
    for (idx, buf) in pool.iter().enumerate() {
        let cap = buf.capacity();
        if cap >= min_capacity && best_fit.is_none_or(|(_, best_cap)| cap < best_cap) {
            best_fit = Some((idx, cap));
        }
        if cap > largest.1 {
            largest = (idx, cap);
        }
    }
    let chosen = best_fit.map_or(largest.0, |(idx, _)| idx);
    Some(pool.swap_remove(chosen))
}

/// RAII guard for a scratch buffer checked out from a [`ScratchPool`].
///
/// Clears and returns the buffer to the thread-local pool on drop unless its
/// capacity exceeds `max_retained_elems` or the pool is unavailable.
pub struct ScratchVec<'a, T: 'static> {
    buf: Vec<T>,
    pool: Option<&'a ScratchPool<T>>,
}

impl<'a, T: 'static> ScratchVec<'a, T> {
    /// Ensures room for at least `additional` more elements, consulting
    /// `check_alloc` only when reallocation is actually required.
    pub fn try_reserve_with<E>(
        &mut self,
        additional: usize,
        mut check_alloc: impl FnMut() -> std::result::Result<(), E>,
        mut map_oom: impl FnMut() -> E,
    ) -> std::result::Result<(), E> {
        let needed = self.buf.len().checked_add(additional);
        match needed {
            Some(total) if total <= self.buf.capacity() => Ok(()),
            Some(_) => {
                check_alloc()?;
                self.buf.try_reserve(additional).map_err(|_| map_oom())
            }
            None => Err(map_oom()),
        }
    }

    /// Pushes `value` with fallible capacity growth.
    pub fn try_push_with<E>(
        &mut self,
        value: T,
        check_alloc: impl FnMut() -> std::result::Result<(), E>,
        map_oom: impl FnMut() -> E,
    ) -> std::result::Result<(), E> {
        self.try_reserve_with(1, check_alloc, map_oom)?;
        self.buf.push(value);
        Ok(())
    }

    /// Resizes to `new_len` elements, consulting `check_alloc` on growth.
    pub fn try_resize_with<E>(
        &mut self,
        new_len: usize,
        value: T,
        check_alloc: impl FnMut() -> std::result::Result<(), E>,
        map_oom: impl FnMut() -> E,
    ) -> std::result::Result<(), E>
    where
        T: Clone,
    {
        if let Some(additional) = new_len.checked_sub(self.buf.len()) {
            self.try_reserve_with(additional, check_alloc, map_oom)?;
        }
        self.buf.resize(new_len, value);
        Ok(())
    }

    /// Detaches the underlying `Vec<T>` from the pool (it will not be recycled).
    pub fn into_vec(mut self) -> Vec<T> {
        self.pool = None;
        std::mem::take(&mut self.buf)
    }
}

impl<T: 'static> Deref for ScratchVec<'_, T> {
    type Target = Vec<T>;

    fn deref(&self) -> &Self::Target {
        &self.buf
    }
}

impl<T: 'static> DerefMut for ScratchVec<'_, T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.buf
    }
}

impl<T: 'static> AsRef<[T]> for ScratchVec<'_, T> {
    fn as_ref(&self) -> &[T] {
        &self.buf
    }
}

impl<T: 'static> AsMut<[T]> for ScratchVec<'_, T> {
    fn as_mut(&mut self) -> &mut [T] {
        &mut self.buf
    }
}

impl<T: 'static> Drop for ScratchVec<'_, T> {
    fn drop(&mut self) {
        if let Some(pool) = self.pool.take() {
            let buf = std::mem::take(&mut self.buf);
            pool.recycle(buf);
        }
    }
}

thread_local! {
    static POOL_U8: RefCell<Vec<Vec<u8>>> = const { RefCell::new(Vec::new()) };
    static POOL_U32: RefCell<Vec<Vec<u32>>> = const { RefCell::new(Vec::new()) };
    static POOL_U64: RefCell<Vec<Vec<u64>>> = const { RefCell::new(Vec::new()) };
    static POOL_USIZE: RefCell<Vec<Vec<usize>>> = const { RefCell::new(Vec::new()) };
    static POOL_F32: RefCell<Vec<Vec<f32>>> = const { RefCell::new(Vec::new()) };
    static POOL_F64: RefCell<Vec<Vec<f64>>> = const { RefCell::new(Vec::new()) };
}

/// Thread-local scratch pool for `u8` / byte buffers.
pub static SCRATCH_U8: ScratchPool<u8> = ScratchPool::new(&POOL_U8);
/// Thread-local scratch pool for `u32` index buffers.
pub static SCRATCH_U32: ScratchPool<u32> = ScratchPool::new(&POOL_U32);
/// Thread-local scratch pool for `u64` key/edge buffers.
pub static SCRATCH_U64: ScratchPool<u64> = ScratchPool::new(&POOL_U64);
/// Thread-local scratch pool for `usize` offset/index buffers.
pub static SCRATCH_USIZE: ScratchPool<usize> = ScratchPool::new(&POOL_USIZE);
/// Thread-local scratch pool for `f32` scalar buffers.
pub static SCRATCH_F32: ScratchPool<f32> = ScratchPool::new(&POOL_F32);
/// Thread-local scratch pool for `f64` scalar buffers.
pub static SCRATCH_F64: ScratchPool<f64> = ScratchPool::new(&POOL_F64);

/// Checked-out thread-local `u8` scratch buffer.
pub type ScratchU8 = ScratchVec<'static, u8>;
/// Checked-out thread-local `u32` scratch buffer.
pub type ScratchU32 = ScratchVec<'static, u32>;
/// Checked-out thread-local `u64` scratch buffer.
pub type ScratchU64 = ScratchVec<'static, u64>;
/// Checked-out thread-local `usize` scratch buffer.
pub type ScratchUsize = ScratchVec<'static, usize>;
/// Checked-out thread-local `f32` scratch buffer.
pub type ScratchF32 = ScratchVec<'static, f32>;
/// Checked-out thread-local `f64` scratch buffer.
pub type ScratchF64 = ScratchVec<'static, f64>;

macro_rules! impl_convenience {
    ($ty:ty, $pool:ident) => {
        impl ScratchVec<'static, $ty> {
            /// Acquires a cleared buffer from the thread-local pool with
            /// capacity for at least `min_capacity` elements.
            pub fn try_acquire(min_capacity: usize) -> Result<Self> {
                $pool.try_acquire(min_capacity)
            }

            /// Acquires a cleared buffer using a custom allocation gate.
            pub fn try_acquire_with<E>(
                min_capacity: usize,
                check_alloc: impl FnMut() -> std::result::Result<(), E>,
                map_oom: impl FnMut() -> E,
            ) -> std::result::Result<Self, E> {
                $pool.try_acquire_with(min_capacity, check_alloc, map_oom)
            }
        }
    };
}

impl_convenience!(u8, SCRATCH_U8);
impl_convenience!(u32, SCRATCH_U32);
impl_convenience!(u64, SCRATCH_U64);
impl_convenience!(usize, SCRATCH_USIZE);
impl_convenience!(f32, SCRATCH_F32);
impl_convenience!(f64, SCRATCH_F64);

/// Invokes `column::try_reserve` with a zero-length probe to honor
/// `fast-mesh-store`'s OOM fault-injection counter cleanly.
fn store_alloc_allowed() -> Result<()> {
    let mut probe = Vec::<u8>::new();
    store_try_reserve(&mut probe, 0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::column::fail_allocations_after;

    #[test]
    fn reuses_buffer_without_reallocating() {
        SCRATCH_U32.clear();
        let ptr = {
            let mut g = ScratchU32::try_acquire(256).unwrap();
            g.extend_from_slice(&[1, 2, 3, 4]);
            g.as_ptr()
        };
        assert_eq!(SCRATCH_U32.retained_count(), 1);

        // Even with fault-injection set to 0 remaining allocations, acquiring
        // within cached capacity succeeds with zero heap allocations.
        fail_allocations_after(Some(0));
        let g2 = ScratchU32::try_acquire(128);
        fail_allocations_after(None);
        let g2 = g2.unwrap();
        assert!(g2.is_empty());
        assert_eq!(g2.as_ptr(), ptr);
    }

    #[test]
    fn drops_oversized_buffers_above_cap() {
        thread_local! {
            static SMALL_POOL_CELL: RefCell<Vec<Vec<u32>>> = const { RefCell::new(Vec::new()) };
        }
        static SMALL_POOL: ScratchPool<u32> = ScratchPool::with_max_retained(&SMALL_POOL_CELL, 64);

        {
            let g = SMALL_POOL.try_acquire(32).unwrap();
            assert!(g.capacity() >= 32);
        }
        assert_eq!(SMALL_POOL.retained_count(), 1);

        {
            let g = SMALL_POOL.try_acquire(128).unwrap();
            assert!(g.capacity() >= 128);
        }
        // Oversized buffer was dropped rather than returned to the pool.
        assert_eq!(SMALL_POOL.retained_count(), 0);
    }

    #[test]
    fn reentrancy_falls_back_to_fresh_fallible_vec() {
        thread_local! {
            static REENTRANT_CELL: RefCell<Vec<Vec<f32>>> = const { RefCell::new(Vec::new()) };
        }
        static REENTRANT_POOL: ScratchPool<f32> = ScratchPool::new(&REENTRANT_CELL);

        {
            let mut seeded = REENTRANT_POOL.try_acquire(16).unwrap();
            seeded.push(1.0);
        }
        assert_eq!(REENTRANT_POOL.retained_count(), 1);

        // Hold an active mutable borrow on the TLS cell to simulate re-entrancy.
        REENTRANT_CELL.with(|cell| {
            let _borrow = cell.borrow_mut();
            let mut fallback = REENTRANT_POOL.try_acquire(8).unwrap();
            fallback.push(42.0);
            assert_eq!(fallback.as_slice(), &[42.0]);

            // With fault injection armed on a fallback buffer that must grow:
            fail_allocations_after(Some(0));
            let err = REENTRANT_POOL.try_acquire(8).err();
            fail_allocations_after(None);
            assert_eq!(err, Some(StoreError::OutOfMemory));
        });

        // Pool stays intact and uncorrupted afterwards.
        assert_eq!(REENTRANT_POOL.retained_count(), 1);
    }

    #[test]
    fn try_reserve_and_fill_respect_oom_gate() {
        SCRATCH_F64.clear();
        fail_allocations_after(Some(0));
        let err = SCRATCH_F64.try_acquire(64).err();
        fail_allocations_after(None);
        assert_eq!(err, Some(StoreError::OutOfMemory));

        let mut g = SCRATCH_F64
            .try_acquire_filled_with(4, 1.5, || Ok(()), || StoreError::OutOfMemory)
            .unwrap();
        assert_eq!(g.as_slice(), &[1.5; 4]);
        let needed = g.capacity() + 32;
        fail_allocations_after(Some(0));
        let err = g.try_resize_with(needed, 2.0, store_alloc_allowed, || StoreError::OutOfMemory);
        fail_allocations_after(None);
        assert_eq!(err, Err(StoreError::OutOfMemory));
    }

    #[test]
    fn multiple_simultaneous_borrows_reuse_independently() {
        SCRATCH_U32.clear();
        {
            let a = ScratchU32::try_acquire(100).unwrap();
            let b = ScratchU32::try_acquire(200).unwrap();
            assert!(a.capacity() >= 100);
            assert!(b.capacity() >= 200);
        }
        assert_eq!(SCRATCH_U32.retained_count(), 2);

        fail_allocations_after(Some(0));
        let a = ScratchU32::try_acquire(100).unwrap();
        let b = ScratchU32::try_acquire(200).unwrap();
        fail_allocations_after(None);
        assert!(a.capacity() >= 100);
        assert!(b.capacity() >= 200);
    }
}
