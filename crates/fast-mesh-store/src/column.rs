//! A growable column of values with an optional write journal.
//!
//! While a journal is active, the first write to each pre-existing index records the
//! old value. `finish()` turns the journal into a [`ColumnPatch`] that can re-apply or
//! revert the changes; `rollback()` restores the column to its state at `begin()`.

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

pub(crate) fn try_vec<T: Clone>(len: usize, value: T) -> Result<Vec<T>> {
    let mut v = Vec::new();
    v.try_reserve_exact(len)
        .map_err(|_| StoreError::OutOfMemory)?;
    v.resize(len, value);
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
        self.data
            .try_reserve(1)
            .map_err(|_| StoreError::OutOfMemory)?;
        self.data.push(value);
        Ok(self.data.len() - 1)
    }

    /// Grows the column to `len` with `value`. Never shrinks.
    pub(crate) fn grow_to(&mut self, len: usize, value: T) -> Result<()> {
        if self.data.len() < len {
            let extra = len - self.data.len();
            self.data
                .try_reserve(extra)
                .map_err(|_| StoreError::OutOfMemory)?;
            self.data.resize(len, value);
        }
        Ok(())
    }

    pub(crate) fn finish(&mut self) -> ColumnPatch<T> {
        let j = self.journal.take().expect("finish() without begin()");
        self.clear_touched(&j.ids);
        let appended = self.data[j.start_len..].to_vec();
        let size = std::mem::size_of::<T>();
        let sparse_bytes = j.ids.len() * (4 + 2 * size) + appended.len() * size;
        let snapshot_bytes = (j.start_len + self.data.len()) * size;
        if !j.ids.is_empty() && snapshot_bytes < sparse_bytes {
            let mut old = self.data[..j.start_len].to_vec();
            for (k, &i) in j.ids.iter().enumerate() {
                old[i as usize] = j.old[k];
            }
            ColumnPatch::Snapshot {
                old,
                new: self.data.clone(),
            }
        } else {
            let new = j.ids.iter().map(|&i| self.data[i as usize]).collect();
            ColumnPatch::Sparse {
                ids: j.ids,
                old: j.old,
                new,
                old_len: j.start_len,
                appended,
            }
        }
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
    pub(crate) fn apply(&self, column: &mut Column<T>) -> Result<()> {
        match self {
            Self::Sparse {
                ids,
                new,
                old_len,
                appended,
                ..
            } => {
                let end = old_len + appended.len();
                column.grow_to(end, T::default())?;
                for (k, &i) in ids.iter().enumerate() {
                    column.data[i as usize] = new[k];
                }
                column.data[*old_len..end].copy_from_slice(appended);
            }
            Self::Snapshot { new, .. } => {
                column.grow_to(new.len(), T::default())?;
                column.data[..new.len()].copy_from_slice(new);
            }
        }
        Ok(())
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
        match c.finish() {
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
        let patch = c.finish();
        assert_eq!(c.as_slice(), &[10, 2, 3, 4]);
        patch.revert(&mut c);
        assert_eq!(&c.as_slice()[..3], &[1, 2, 3], "old values restored");
        assert_eq!(c.len(), 4, "appended slot stays allocated");
        assert_eq!(c.get(3), 0, "appended slot is reset to the default");
        patch.apply(&mut c).unwrap();
        assert_eq!(c.as_slice(), &[10, 2, 3, 4]);
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
        match c.finish() {
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
        let patch = c.finish();
        assert!(matches!(patch, ColumnPatch::Snapshot { .. }));
        patch.revert(&mut c);
        assert_eq!(c.as_slice(), &[0; 100]);
        patch.apply(&mut c).unwrap();
        assert_eq!(c.as_slice(), &[1; 100]);
    }

    #[test]
    fn untouched_column_produces_an_empty_patch() {
        let mut c = column(&[1, 2]);
        c.begin();
        let patch = c.finish();
        assert!(patch.is_empty());
        assert_eq!(patch.byte_size(), 0);
    }
}
