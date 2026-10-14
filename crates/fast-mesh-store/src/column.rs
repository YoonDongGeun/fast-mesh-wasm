//! A growable column of values with an optional write journal.
//!
//! While a journal is active, the first write to each pre-existing index records the
//! old value. `prepare_finish()` and `seal()` turn the journal into a [`ColumnPatch`] that
//! can re-apply or revert the changes; `rollback()` restores the column to its state
//! at `begin()`.

use crate::{Result, StoreError};

pub(crate) struct Column<T> {
    data: Vec<T>,
    /// Bitset of indices already journaled in the current transaction. Kept between
    /// transactions and cleared bit by bit, so beginning a transaction costs nothing.
    touched: Vec<u64>,
    journal: Option<Journal<T>>,
}

struct Journal<T> {
    start_len: usize,
    ids: Vec<u32>,
    old: Vec<T>,
}

/// The recorded effect of one transaction on one column.
#[derive(Debug, Clone)]
pub(crate) enum ColumnPatch<T> {
    /// Per-index old and new values, plus values appended past the old length.
    Sparse {
        ids: Vec<u32>,
        old: Vec<T>,
        new: Vec<T>,
        old_len: usize,
        appended: Vec<T>,
    },
    /// Full copies, used when they are smaller than the sparse form.
    Snapshot { old: Vec<T>, new: Vec<T> },
}

/// The vectors of a [`ColumnPatch`] built by [`Column::prepare_finish`] while the
/// journal is still intact. [`Column::seal`] turns them into the patch infallibly.
pub(crate) enum Prepared<T> {
    Sparse { new: Vec<T>, appended: Vec<T> },
    Snapshot { old: Vec<T>, new: Vec<T> },
}

#[cfg(test)]
thread_local! {
    /// Test-only fault injection: number of fallible reservations that still succeed.
    static ALLOCATIONS_LEFT: std::cell::Cell<Option<usize>> = const { std::cell::Cell::new(None) };
}

/// Makes the `n`-th next fallible reservation on this thread (0-based) and every one
/// after it fail with `OutOfMemory`. `None` turns failures off.
#[cfg(test)]
pub(crate) fn fail_allocations_after(n: Option<usize>) {
    ALLOCATIONS_LEFT.with(|c| c.set(n));
}

/// Every fallible reservation passes through here, so tests can make it fail.
fn allocation_allowed() -> Result<()> {
    #[cfg(test)]
    {
        if let Some(n) = ALLOCATIONS_LEFT.with(|c| c.get()) {
            if n == 0 {
                return Err(StoreError::OutOfMemory);
            }
            ALLOCATIONS_LEFT.with(|c| c.set(Some(n - 1)));
        }
    }
    Ok(())
}

/// `Vec::try_reserve` reporting `OutOfMemory`.
pub(crate) fn try_reserve<T>(v: &mut Vec<T>, additional: usize) -> Result<()> {
    allocation_allowed()?;
    v.try_reserve(additional)
        .map_err(|_| StoreError::OutOfMemory)
}

/// `Vec::try_reserve_exact` reporting `OutOfMemory`.
pub(crate) fn try_reserve_exact<T>(v: &mut Vec<T>, additional: usize) -> Result<()> {
    allocation_allowed()?;
    v.try_reserve_exact(additional)
        .map_err(|_| StoreError::OutOfMemory)
}

pub(crate) fn try_vec<T: Clone>(len: usize, value: T) -> Result<Vec<T>> {
    let mut v = Vec::new();
    try_reserve_exact(&mut v, len)?;
    v.resize(len, value);
    Ok(v)
}

/// A copy of `values` allocated fallibly.
pub(crate) fn try_copy<T: Copy>(values: &[T]) -> Result<Vec<T>> {
    let mut v = Vec::new();
    try_reserve_exact(&mut v, values.len())?;
    v.extend_from_slice(values);
    Ok(v)
}

impl<T: Copy + Default> Column<T> {
    pub(crate) fn from_vec(data: Vec<T>) -> Self {
        Self {
            data,
            touched: Vec::new(),
            journal: None,
        }
    }

    pub(crate) fn len(&self) -> usize {
        self.data.len()
    }

    pub(crate) fn as_slice(&self) -> &[T] {
        &self.data
    }

    /// Mutable access that bypasses the journal. Only for data that is not undoable.
    pub(crate) fn as_mut_slice(&mut self) -> &mut [T] {
        debug_assert!(
            self.journal.is_none(),
            "unjournaled write during a transaction"
        );
        &mut self.data
    }

    pub(crate) fn get(&self, i: usize) -> T {
        self.data[i]
    }

    pub(crate) fn begin(&mut self) {
        debug_assert!(self.journal.is_none(), "begin() called twice");
        let words = self.data.len().div_ceil(64);
        if self.touched.len() < words {
            self.touched.resize(words, 0);
        }
        self.journal = Some(Journal {
            start_len: self.data.len(),
            ids: Vec::new(),
            old: Vec::new(),
        });
    }

    pub(crate) fn set(&mut self, i: usize, value: T) {
        if let Some(j) = &mut self.journal
            && i < j.start_len
        {
            let (word, bit) = (i / 64, 1u64 << (i % 64));
            if self.touched[word] & bit == 0 {
                self.touched[word] |= bit;
                j.ids.push(i as u32);
                j.old.push(self.data[i]);
            }
        }
        self.data[i] = value;
    }

    /// Appends a value and returns its index. Appended values are not journaled:
    /// rollback truncates them and patches store them as `appended`.
    pub(crate) fn push(&mut self, value: T) -> Result<usize> {
        try_reserve(&mut self.data, 1)?;
        self.data.push(value);
        Ok(self.data.len() - 1)
    }

    /// Grows the column to `len` with `value`. Never shrinks.
    pub(crate) fn grow_to(&mut self, len: usize, value: T) -> Result<()> {
        self.try_reserve_to(len)?;
        self.resize_reserved(len, value);
        Ok(())
    }

    /// Reserves capacity for `len` values without changing the length.
    pub(crate) fn try_reserve_to(&mut self, len: usize) -> Result<()> {
        let extra = len.saturating_sub(self.data.len());
        try_reserve(&mut self.data, extra)
    }

    /// Grows the column to `len` with `value` within capacity reserved by
    /// [`Column::try_reserve_to`], so it does not allocate. Never shrinks.
    pub(crate) fn resize_reserved(&mut self, len: usize, value: T) {
        if self.data.len() < len {
            debug_assert!(self.data.capacity() >= len, "capacity was not reserved");
            self.data.resize(len, value);
        }
    }

    /// Indices journaled so far in the current transaction, in first-write order and
    /// free of duplicates. Only indices that existed at `begin()` are journaled.
    pub(crate) fn journal_ids(&self) -> &[u32] {
        self.journal.as_ref().map_or(&[], |j| &j.ids)
    }

    /// Builds every vector of the patch without touching the journal, so a failure
    /// leaves the transaction intact for rollback.
    pub(crate) fn prepare_finish(&self) -> Result<Prepared<T>> {
        let j = self.journal.as_ref().expect("finish() without begin()");
        let appended_len = self.data.len() - j.start_len;
        let size = std::mem::size_of::<T>();
        let sparse_bytes = j.ids.len() * (4 + 2 * size) + appended_len * size;
        let snapshot_bytes = (j.start_len + self.data.len()) * size;
        if !j.ids.is_empty() && snapshot_bytes < sparse_bytes {
            let mut old = try_copy(&self.data[..j.start_len])?;
            for (k, &i) in j.ids.iter().enumerate() {
                old[i as usize] = j.old[k];
            }
            let new = try_copy(&self.data)?;
            Ok(Prepared::Snapshot { old, new })
        } else {
            let appended = try_copy(&self.data[j.start_len..])?;
            let mut new = Vec::new();
            try_reserve_exact(&mut new, j.ids.len())?;
            new.extend(j.ids.iter().map(|&i| self.data[i as usize]));
            Ok(Prepared::Sparse { new, appended })
        }
    }

    /// Ends the journal and turns prepared vectors into the patch. Cannot fail.
    pub(crate) fn seal(&mut self, prepared: Prepared<T>) -> ColumnPatch<T> {
        let j = self.journal.take().expect("finish() without begin()");
        self.clear_touched(&j.ids);
        match prepared {
            Prepared::Snapshot { old, new } => ColumnPatch::Snapshot { old, new },
            Prepared::Sparse { new, appended } => ColumnPatch::Sparse {
                ids: j.ids,
                old: j.old,
                new,
                old_len: j.start_len,
                appended,
            },
        }
    }

    /// Ends the journal, keeping the changes but recording no patch (for data that
    /// is not undoable, such as derived channels).
    pub(crate) fn end_journal(&mut self) {
        let j = self.journal.take().expect("end_journal() without begin()");
        self.clear_touched(&j.ids);
    }

    /// [`Column::prepare_finish`] then [`Column::seal`]. On failure the journal is
    /// still active.
    #[cfg(test)]
    pub(crate) fn finish(&mut self) -> Result<ColumnPatch<T>> {
        let prepared = self.prepare_finish()?;
        Ok(self.seal(prepared))
    }

    /// Test helper: shrinks the column, e.g. to recreate a pre-commit layout.
    #[cfg(test)]
    pub(crate) fn truncate(&mut self, len: usize) {
        self.data.truncate(len);
    }

    pub(crate) fn rollback(&mut self) {
        let j = self.journal.take().expect("rollback() without begin()");
        self.clear_touched(&j.ids);
        for (k, &i) in j.ids.iter().enumerate() {
            self.data[i as usize] = j.old[k];
        }
        self.data.truncate(j.start_len);
    }

    fn clear_touched(&mut self, ids: &[u32]) {
        for &i in ids {
            self.touched[i as usize / 64] &= !(1u64 << (i % 64));
        }
    }
}

impl<T: Copy + Default> ColumnPatch<T> {
    /// Length the column must have before [`ColumnPatch::apply`].
    pub(crate) fn required_len(&self) -> usize {
        match self {
            Self::Sparse {
                old_len, appended, ..
            } => old_len + appended.len(),
            Self::Snapshot { new, .. } => new.len(),
        }
    }

    /// Re-applies the changes. The column must already be `required_len()` long: grow
    /// it first, which is the only step that can fail.
    pub(crate) fn apply(&self, column: &mut Column<T>) {
        debug_assert!(column.len() >= self.required_len());
        match self {
            Self::Sparse {
                ids,
                new,
                old_len,
                appended,
                ..
            } => {
                for (k, &i) in ids.iter().enumerate() {
                    column.data[i as usize] = new[k];
                }
                column.data[*old_len..old_len + appended.len()].copy_from_slice(appended);
            }
            Self::Snapshot { new, .. } => {
                column.data[..new.len()].copy_from_slice(new);
            }
        }
    }

    /// Restores the old values. The column keeps its length: slots appended by the
    /// transaction stay allocated so their ids are never handed out twice, and are
    /// reset to `T::default()` (dead liveness bits, `[0, 0, 0]` triangles).
    pub(crate) fn revert(&self, column: &mut Column<T>) {
        let (old_len, new_len) = match self {
            Self::Sparse {
                ids,
                old,
                old_len,
                appended,
                ..
            } => {
                for (k, &i) in ids.iter().enumerate() {
                    column.data[i as usize] = old[k];
                }
                (*old_len, old_len + appended.len())
            }
            Self::Snapshot { old, new } => {
                column.data[..old.len()].copy_from_slice(old);
                (old.len(), new.len())
            }
        };
        column.data[old_len..new_len].fill(T::default());
    }

    pub(crate) fn is_empty(&self) -> bool {
        match self {
            Self::Sparse { ids, appended, .. } => ids.is_empty() && appended.is_empty(),
            Self::Snapshot { .. } => false,
        }
    }

    pub(crate) fn byte_size(&self) -> usize {
        let size = std::mem::size_of::<T>();
        match self {
            Self::Sparse {
                ids,
                old,
                new,
                appended,
                ..
            } => ids.len() * 4 + (old.len() + new.len() + appended.len()) * size,
            Self::Snapshot { old, new } => (old.len() + new.len()) * size,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn column(values: &[u32]) -> Column<u32> {
        Column::from_vec(values.to_vec())
    }

    #[test]
    fn journals_only_the_first_write_per_index() {
        let mut c = column(&[1, 2, 3]);
        c.begin();
        c.set(1, 20);
        c.set(1, 200);
        match c.finish().unwrap() {
            ColumnPatch::Sparse { ids, old, new, .. } => {
                assert_eq!(ids, vec![1]);
                assert_eq!(old, vec![2]);
                assert_eq!(new, vec![200]);
            }
            ColumnPatch::Snapshot { .. } => panic!("expected sparse patch"),
        }
    }

    #[test]
    fn revert_and_apply_round_trip() {
        let mut c = column(&[1, 2, 3]);
        c.begin();
        c.set(0, 10);
        c.push(4).unwrap();
        let patch = c.finish().unwrap();
        assert_eq!(c.as_slice(), &[10, 2, 3, 4]);
        patch.revert(&mut c);
        assert_eq!(&c.as_slice()[..3], &[1, 2, 3], "old values restored");
        assert_eq!(c.len(), 4, "appended slot stays allocated");
        assert_eq!(c.get(3), 0, "appended slot is reset to the default");
        patch.apply(&mut c);
        assert_eq!(c.as_slice(), &[10, 2, 3, 4]);
    }

    #[test]
    fn apply_after_growth_writes_appended_values() {
        let mut c = column(&[1, 2, 3]);
        c.begin();
        c.set(0, 10);
        c.push(4).unwrap();
        let patch = c.finish().unwrap();
        let mut fresh = column(&[1, 2, 3]);
        fresh.grow_to(patch.required_len(), 0).unwrap();
        patch.apply(&mut fresh);
        assert_eq!(fresh.as_slice(), &[10, 2, 3, 4]);
    }

    #[test]
    fn rollback_restores_values_and_length() {
        let mut c = column(&[1, 2, 3]);
        c.begin();
        c.set(2, 30);
        c.push(4).unwrap();
        c.rollback();
        assert_eq!(c.as_slice(), &[1, 2, 3]);
        // The touched bits were cleared: a new transaction journals index 2 again.
        c.begin();
        c.set(2, 31);
        match c.finish().unwrap() {
            ColumnPatch::Sparse { ids, old, .. } => {
                assert_eq!(ids, vec![2]);
                assert_eq!(old, vec![3]);
            }
            ColumnPatch::Snapshot { .. } => panic!("expected sparse patch"),
        }
    }

    #[test]
    fn rewriting_most_values_switches_to_a_snapshot() {
        let mut c = column(&[0; 100]);
        c.begin();
        for i in 0..100 {
            c.set(i, 1);
        }
        let patch = c.finish().unwrap();
        assert!(matches!(patch, ColumnPatch::Snapshot { .. }));
        patch.revert(&mut c);
        assert_eq!(c.as_slice(), &[0; 100]);
        patch.apply(&mut c);
        assert_eq!(c.as_slice(), &[1; 100]);
    }

    #[test]
    fn a_failed_finish_keeps_the_journal_for_rollback() {
        for snapshot in [false, true] {
            let mut c = column(&[1, 2, 3, 4]);
            c.begin();
            if snapshot {
                for i in 0..4 {
                    c.set(i, 9);
                }
            } else {
                c.set(1, 9);
            }
            c.push(5).unwrap();
            fail_allocations_after(Some(0));
            assert_eq!(c.finish().err(), Some(StoreError::OutOfMemory));
            fail_allocations_after(None);
            c.rollback();
            assert_eq!(c.as_slice(), &[1, 2, 3, 4]);
        }
    }

    #[test]
    fn untouched_column_produces_an_empty_patch() {
        let mut c = column(&[1, 2]);
        c.begin();
        let patch = c.finish().unwrap();
        assert!(patch.is_empty());
        assert_eq!(patch.byte_size(), 0);
    }
}
