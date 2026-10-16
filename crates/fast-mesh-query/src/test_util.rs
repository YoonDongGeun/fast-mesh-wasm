//! Fixtures shared by the query tests.

use fast_mesh_bvh::{BuildOptions, Bvh, Quality};
use fast_mesh_store::Mesh;

/// A `(n + 1) x (n + 1)` grid of vertices with `z = (x + y) % 3`, two triangles per cell.
pub(crate) fn wavy(n: u32) -> Mesh {
    let mut positions = Vec::new();
    for y in 0..=n {
        for x in 0..=n {
            positions.extend([x as f32, y as f32, ((x + y) % 3) as f32]);
        }
    }
    let mut indices = Vec::new();
    for y in 0..n {
        for x in 0..n {
            let a = y * (n + 1) + x;
            indices.extend([a, a + 1, a + n + 2, a, a + n + 2, a + n + 1]);
        }
    }
    Mesh::from_buffers(&positions, &indices).unwrap()
}

/// A BVH with small leaves, so that the tests exercise deep trees.
pub(crate) fn build(mesh: &Mesh, quality: Quality) -> Bvh {
    let opts = BuildOptions {
        quality,
        max_leaf_size: 2,
        ..Default::default()
    };
    Bvh::build(mesh, &opts).unwrap()
}

/// A deterministic linear congruential generator.
pub(crate) struct Lcg(pub(crate) u64);

impl Lcg {
    /// Uniform in `[0, 1)`.
    pub(crate) fn next(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.0 >> 40) as f32 / (1u64 << 24) as f32
    }
}
