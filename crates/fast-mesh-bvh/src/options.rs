use crate::{BvhError, Result};

/// How the tree is split when it is built.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Quality {
    /// Binned surface-area heuristic: slower to build, faster to query.
    Sah,
    /// Morton-code (LBVH) build: faster to build, somewhat slower to query.
    Fast,
}

/// Options for building (and rebuilding parts of) a BVH.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BuildOptions {
    pub quality: Quality,
    /// Most triangles per leaf, 1 to 16.
    pub max_leaf_size: u32,
    /// Bins per axis for the SAH build, 4 to 64.
    pub sah_bins: u32,
    /// Most triangles in a treelet, the unit of partial rebuilds. At least `max_leaf_size`.
    pub treelet_size: u32,
}

impl Default for BuildOptions {
    fn default() -> Self {
        Self {
            quality: Quality::Sah,
            max_leaf_size: 8,
            sah_bins: 16,
            treelet_size: 4096,
        }
    }
}

impl BuildOptions {
    pub(crate) fn validate(&self) -> Result<()> {
        if !(1..=16).contains(&self.max_leaf_size) {
            return Err(BvhError::InvalidOptions("max_leaf_size must be 1 to 16"));
        }
        if !(4..=64).contains(&self.sah_bins) {
            return Err(BvhError::InvalidOptions("sah_bins must be 4 to 64"));
        }
        if self.treelet_size < self.max_leaf_size {
            return Err(BvhError::InvalidOptions(
                "treelet_size must be at least max_leaf_size",
            ));
        }
        Ok(())
    }
}

/// When `update` falls back to heavier strategies.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct UpdatePolicy {
    /// Rebuild everything when more than this fraction of alive triangles is
    /// dirty (added, removed, rewritten or moved), 0 to 1.
    pub full_rebuild_fraction: f32,
    /// Rebuild a refit treelet when its SAH cost exceeds this factor times its
    /// cost when it was built. At least 1.
    pub degradation_factor: f32,
}

impl Default for UpdatePolicy {
    fn default() -> Self {
        Self {
            full_rebuild_fraction: 0.30,
            degradation_factor: 1.5,
        }
    }
}

impl UpdatePolicy {
    pub(crate) fn validate(&self) -> Result<()> {
        if !(0.0..=1.0).contains(&self.full_rebuild_fraction) {
            return Err(BvhError::InvalidOptions(
                "full_rebuild_fraction must be 0 to 1",
            ));
        }
        if self.degradation_factor.is_nan() || self.degradation_factor < 1.0 {
            return Err(BvhError::InvalidOptions(
                "degradation_factor must be at least 1",
            ));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_are_valid() {
        assert!(BuildOptions::default().validate().is_ok());
        assert!(UpdatePolicy::default().validate().is_ok());
    }

    #[test]
    fn out_of_range_options_are_rejected() {
        let bad_leaf = BuildOptions {
            max_leaf_size: 0,
            ..Default::default()
        };
        assert!(matches!(
            bad_leaf.validate(),
            Err(BvhError::InvalidOptions(_))
        ));
        let bad_bins = BuildOptions {
            sah_bins: 65,
            ..Default::default()
        };
        assert!(bad_bins.validate().is_err());
        let bad_treelet = BuildOptions {
            treelet_size: 4,
            ..Default::default()
        };
        assert!(bad_treelet.validate().is_err());
        let nan = UpdatePolicy {
            degradation_factor: f32::NAN,
            ..Default::default()
        };
        assert!(nan.validate().is_err());
        let fraction = UpdatePolicy {
            full_rebuild_fraction: 1.5,
            ..Default::default()
        };
        assert!(fraction.validate().is_err());
    }
}
