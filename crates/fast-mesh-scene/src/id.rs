//! Generational ids and the slot map behind them.

use fast_mesh_store::StoreError;

use crate::{Result, SceneError};

/// A handle to a mesh in a [`Scene`](crate::Scene).
///
/// Ids are generational: once a mesh is removed its id never matches again,
/// even when the slot is reused. Ids order by `(index, generation)`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct MeshId {
    index: u32,
    generation: u32,
}

struct Slot<T> {
    generation: u32,
    value: Option<T>,
}

/// Values addressed by [`MeshId`], iterated in index order.
pub(crate) struct SlotMap<T> {
    slots: Vec<Slot<T>>,
    /// Indices of empty slots that can be reused.
    free: Vec<u32>,
}

fn oom() -> SceneError {
    SceneError::Store(StoreError::OutOfMemory)
}

impl<T> SlotMap<T> {
    pub(crate) fn new() -> Self {
        SlotMap {
            slots: Vec::new(),
            free: Vec::new(),
        }
    }

    /// Stores `value` and returns its id. Leaves the map unchanged on error.
    pub(crate) fn insert(&mut self, value: T) -> Result<MeshId> {
        if let Some(index) = self.free.pop() {
            let slot = &mut self.slots[index as usize];
            slot.value = Some(value);
            return Ok(MeshId {
                index,
                generation: slot.generation,
            });
        }
        let index = u32::try_from(self.slots.len()).map_err(|_| oom())?;
        self.slots.try_reserve(1).map_err(|_| oom())?;
        self.slots.push(Slot {
            generation: 0,
            value: Some(value),
        });
        Ok(MeshId {
            index,
            generation: 0,
        })
    }

    /// Removes and returns the value; the id stops matching.
    pub(crate) fn remove(&mut self, id: MeshId) -> Option<T> {
        let slot = self.slots.get_mut(id.index as usize)?;
        if slot.generation != id.generation || slot.value.is_none() {
            return None;
        }
        let value = slot.value.take();
        // A slot whose generation space is used up is retired, not reused. So
        // is one whose free-list entry cannot be reserved (best effort).
        if let Some(next) = slot.generation.checked_add(1) {
            slot.generation = next;
            if self.free.try_reserve(1).is_ok() {
                self.free.push(id.index);
            }
        }
        value
    }

    pub(crate) fn get(&self, id: MeshId) -> Option<&T> {
        let slot = self.slots.get(id.index as usize)?;
        if slot.generation == id.generation {
            slot.value.as_ref()
        } else {
            None
        }
    }

    pub(crate) fn get_mut(&mut self, id: MeshId) -> Option<&mut T> {
        let slot = self.slots.get_mut(id.index as usize)?;
        if slot.generation == id.generation {
            slot.value.as_mut()
        } else {
            None
        }
    }

    /// Live `(id, value)` pairs in index order.
    pub(crate) fn iter(&self) -> impl Iterator<Item = (MeshId, &T)> {
        self.slots.iter().enumerate().filter_map(|(i, s)| {
            s.value.as_ref().map(|v| {
                (
                    MeshId {
                        index: i as u32,
                        generation: s.generation,
                    },
                    v,
                )
            })
        })
    }

    /// Live `(id, value)` pairs in index order, mutably.
    pub(crate) fn iter_mut(&mut self) -> impl Iterator<Item = (MeshId, &mut T)> {
        self.slots.iter_mut().enumerate().filter_map(|(i, s)| {
            let generation = s.generation;
            s.value.as_mut().map(|v| {
                (
                    MeshId {
                        index: i as u32,
                        generation,
                    },
                    v,
                )
            })
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reuse_bumps_generation() {
        let mut m = SlotMap::new();
        let a = m.insert("a").unwrap();
        assert_eq!(m.remove(a), Some("a"));
        assert_eq!(m.remove(a), None);
        let b = m.insert("b").unwrap();
        assert_eq!(a.index, b.index);
        assert_ne!(a, b);
        assert!(m.get(a).is_none());
        assert_eq!(m.get(b), Some(&"b"));
    }
}
