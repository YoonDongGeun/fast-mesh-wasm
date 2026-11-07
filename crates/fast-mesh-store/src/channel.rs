use crate::column::Column;

/// Name of the UV channel: a source channel of 2 values per vertex slot, in
/// the file's own convention (glTF's has `v` pointing down), interpolated for
/// new vertices. Read it with [`crate::Mesh::uvs`].
///
/// A mesh without it pays nothing. The name lives in the channel namespace;
/// the plane `(u, v)` coordinates of a section are plain result fields and do
/// not meet it.
pub const UV_CHANNEL: &str = "uv";

/// Whether a channel's values are undoable input or recomputable output.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChannelKind {
    /// Values provided by the user (e.g. a painted mask). Journaled and undoable.
    Source,
    /// Values computed from other data (e.g. distances, colors). Not journaled;
    /// recomputed after edits, undo and redo.
    Derived,
}

/// How a source channel gets values for vertices created by an edit.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TopologyPolicy {
    /// Weighted average of the creation parents' values.
    Interpolate,
    /// The channel's default value.
    Default,
    /// The value of the creation parent with the largest weight.
    Nearest,
}

/// Describes a per-vertex attribute channel.
#[derive(Debug, Clone, PartialEq)]
pub struct ChannelDesc {
    pub name: String,
    /// Values per vertex, 1 to 4.
    pub width: usize,
    pub kind: ChannelKind,
    pub policy: TopologyPolicy,
    /// Value for new vertices without usable parents, and for derived channels.
    pub default: f32,
    /// The values describe the mesh as it was when they were written, so they
    /// stop being true when it changes in ways they cannot follow. A commit that
    /// changes topology drops the channel, as
    /// [`Transaction::drop_channel`](crate::Transaction::drop_channel) would (its
    /// undo brings it back), unless the writes ran under
    /// [`Transaction::keeping_bound_channels`](crate::Transaction::keeping_bound_channels).
    /// Undoing or redoing a patch recorded before the channel was added removes
    /// the channel for good: that takes the mesh to a state it never described.
    pub topology_bound: bool,
}

impl ChannelDesc {
    /// A source channel that interpolates values for new vertices and defaults to 0.
    pub fn source(name: impl Into<String>, width: usize) -> Self {
        Self {
            name: name.into(),
            width,
            kind: ChannelKind::Source,
            policy: TopologyPolicy::Interpolate,
            default: 0.0,
            topology_bound: false,
        }
    }

    /// A derived channel whose new values are NaN until computed.
    pub fn derived(name: impl Into<String>, width: usize) -> Self {
        Self {
            name: name.into(),
            width,
            kind: ChannelKind::Derived,
            policy: TopologyPolicy::Default,
            default: f32::NAN,
            topology_bound: false,
        }
    }

    /// The UV channel: [`UV_CHANNEL`], a source channel of 2 interpolated values.
    pub fn uv() -> Self {
        Self::source(UV_CHANNEL, 2)
    }

    /// Makes the channel [`topology_bound`](Self::topology_bound).
    pub fn bound_to_topology(mut self) -> Self {
        self.topology_bound = true;
        self
    }

    pub fn with_policy(mut self, policy: TopologyPolicy) -> Self {
        self.policy = policy;
        self
    }

    pub fn with_default(mut self, default: f32) -> Self {
        self.default = default;
        self
    }
}

pub(crate) struct Channel {
    /// Unique within its mesh, never reused. Patches refer to channels by this id, so
    /// a channel removed and re-added under the same name is a different channel.
    pub(crate) id: u64,
    pub(crate) desc: ChannelDesc,
    /// `width` values per vertex slot.
    pub(crate) data: Column<f32>,
    /// Dropped by a committed edit whose patch can still bring it back: name
    /// lookups skip it, while its column keeps following the slots so that
    /// undo restores it exactly.
    pub(crate) hidden: bool,
}

impl Channel {
    /// Value for a new vertex created from `parents` (vertex id, weight) for component `k`.
    pub(crate) fn initial_value(&self, parents: &[(u32, f32)], k: usize) -> f32 {
        let d = &self.desc;
        if d.kind == ChannelKind::Derived || parents.is_empty() {
            return d.default;
        }
        let at = |v: u32| self.data.get(v as usize * d.width + k);
        match d.policy {
            TopologyPolicy::Default => d.default,
            TopologyPolicy::Nearest => {
                let (v, _) = parents
                    .iter()
                    .copied()
                    .max_by(|a, b| a.1.total_cmp(&b.1))
                    .expect("parents is not empty");
                at(v)
            }
            TopologyPolicy::Interpolate => {
                let total: f32 = parents.iter().map(|p| p.1).sum();
                if total == 0.0 {
                    d.default
                } else {
                    parents.iter().map(|&(v, w)| at(v) * w).sum::<f32>() / total
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn channel(desc: ChannelDesc, values: &[f32]) -> Channel {
        Channel {
            id: 0,
            desc,
            data: Column::from_vec(values.to_vec()),
            hidden: false,
        }
    }

    #[test]
    fn interpolate_uses_normalized_weights() {
        let c = channel(ChannelDesc::source("m", 1), &[10.0, 20.0]);
        assert_eq!(c.initial_value(&[(0, 1.0), (1, 3.0)], 0), 17.5);
    }

    #[test]
    fn nearest_takes_the_heaviest_parent() {
        let c = channel(
            ChannelDesc::source("m", 1).with_policy(TopologyPolicy::Nearest),
            &[10.0, 20.0],
        );
        assert_eq!(c.initial_value(&[(0, 0.9), (1, 0.1)], 0), 10.0);
    }

    #[test]
    fn no_parents_or_zero_weights_use_the_default() {
        let c = channel(ChannelDesc::source("m", 1).with_default(-1.0), &[10.0]);
        assert_eq!(c.initial_value(&[], 0), -1.0);
        assert_eq!(c.initial_value(&[(0, 0.0)], 0), -1.0);
    }

    #[test]
    fn derived_channels_always_start_at_the_default() {
        let c = channel(ChannelDesc::derived("d", 1), &[10.0]);
        assert!(c.initial_value(&[(0, 1.0)], 0).is_nan());
    }

    #[test]
    fn components_are_read_per_width() {
        let c = channel(
            ChannelDesc::source("rgb", 3),
            &[1.0, 2.0, 3.0, 5.0, 6.0, 7.0],
        );
        let parents = [(0, 1.0), (1, 1.0)];
        assert_eq!(
            [0, 1, 2].map(|k| c.initial_value(&parents, k)),
            [3.0, 4.0, 5.0]
        );
    }
}
