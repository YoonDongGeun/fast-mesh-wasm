//! Shared builders for the fuzz targets: bounded arbitrary meshes and inputs.
//!
//! Every target fuzzes the no-panic contract of a geometry kernel over small
//! meshes built two ways, mirroring the `test_util` fixtures of the engine
//! crates: random triangle soup (ragged boundary, degenerate triangles,
//! duplicated vertices) and grids with random cells punched out (clean square
//! holes, the `fill` target's bread and butter). Coordinates are finite and
//! quantized so that coincident vertices occur naturally.
//!
//! Each target compiles this module separately and uses only part of it.
#![allow(dead_code)]

use arbitrary::{Arbitrary, Unstructured};
use fast_mesh_store::Mesh;

/// Caps keep every target fast enough for short CI runs: at most this many
/// vertices and triangles per fuzzed mesh.
pub const MAX_VERTS: usize = 48;
pub const MAX_TRIS: usize = 96;

/// A finite coordinate in `[-8, 8]`, quantized to `0.05`.
fn coord(u: &mut Unstructured<'_>) -> arbitrary::Result<f32> {
    Ok(f32::from(u16::arbitrary(u)? % 321) * 0.05 - 8.0)
}

/// Flat positions and indices that always build: indices are already reduced
/// into range, so [`Mesh::from_buffers`] only fails on a library bug.
#[derive(Debug)]
pub struct FuzzBuffers {
    pub positions: Vec<f32>,
    pub indices: Vec<u32>,
}

impl<'a> Arbitrary<'a> for FuzzBuffers {
    fn arbitrary(u: &mut Unstructured<'a>) -> arbitrary::Result<Self> {
        // Random soup, the occasional triangle degenerate.
        let nv = usize::from(u8::arbitrary(u)? % (MAX_VERTS as u8 + 1));
        let mut positions = Vec::with_capacity(nv * 3);
        for _ in 0..nv {
            positions.extend([coord(u)?, coord(u)?, coord(u)?]);
        }
        let nt = usize::from(u8::arbitrary(u)? % (MAX_TRIS as u8 + 1));
        let mut indices = Vec::with_capacity(nt * 3);
        for _ in 0..nt {
            if nv == 0 {
                break;
            }
            let mut tri = [
                u32::from(u8::arbitrary(u)?) % nv as u32,
                u32::from(u8::arbitrary(u)?) % nv as u32,
                u32::from(u8::arbitrary(u)?) % nv as u32,
            ];
            if u8::arbitrary(u)? % 8 == 0 {
                tri[2] = tri[0];
            }
            indices.extend(tri);
        }
        Ok(FuzzBuffers { positions, indices })
    }
}

impl FuzzBuffers {
    /// The buffers as a mesh, through the plain constructor.
    pub fn mesh(&self) -> Mesh {
        Mesh::from_buffers(&self.positions, &self.indices).expect("valid by construction")
    }

    /// The buffers as a mesh, through the `f64` constructor (exercises origin
    /// selection and offset validation).
    pub fn mesh_f64(&self) -> Mesh {
        let positions: Vec<f64> = self.positions.iter().map(|&c| f64::from(c)).collect();
        Mesh::from_buffers_f64(&positions, &self.indices).expect("valid by construction")
    }
}

/// A `(n + 1) x (n + 1)` grid at `z = 0` with a random subset of cells punched
/// out, after `test_util::grid_without` in `fast-mesh-ops`.
#[derive(Debug)]
pub struct FuzzGrid {
    pub buffers: FuzzBuffers,
}

impl<'a> Arbitrary<'a> for FuzzGrid {
    fn arbitrary(u: &mut Unstructured<'a>) -> arbitrary::Result<Self> {
        let n = u32::from(u8::arbitrary(u)? % 6 + 1);
        let mut positions = Vec::new();
        for y in 0..=n {
            for x in 0..=n {
                positions.extend([x as f32, y as f32, 0.0]);
            }
        }
        let mut indices = Vec::new();
        for y in 0..n {
            for x in 0..n {
                // Punch out about a third of the cells.
                if u8::arbitrary(u)? % 3 == 0 {
                    continue;
                }
                let a = y * (n + 1) + x;
                indices.extend([a, a + 1, a + n + 2, a, a + n + 2, a + n + 1]);
            }
        }
        Ok(FuzzGrid {
            buffers: FuzzBuffers { positions, indices },
        })
    }
}

/// The box `[lo, hi]` as 12 triangles, after `test_util::box_mesh`: closed and
/// manifold when `lo < hi` on every axis, degenerate otherwise (both are
/// useful inputs for the `boolean` target).
#[derive(Debug)]
pub struct FuzzBox {
    pub buffers: FuzzBuffers,
}

impl<'a> Arbitrary<'a> for FuzzBox {
    fn arbitrary(u: &mut Unstructured<'a>) -> arbitrary::Result<Self> {
        let corner = |u: &mut Unstructured<'a>| -> arbitrary::Result<[f32; 3]> {
            Ok([coord(u)?, coord(u)?, coord(u)?])
        };
        let (lo, hi) = (corner(u)?, corner(u)?);
        let mut positions = Vec::new();
        for i in 0..8 {
            positions.extend((0..3).map(|k| if i >> k & 1 == 1 { hi[k] } else { lo[k] }));
        }
        let indices = vec![
            0, 4, 6, 0, 6, 2, // -x
            1, 3, 7, 1, 7, 5, // +x
            0, 1, 5, 0, 5, 4, // -y
            2, 6, 7, 2, 7, 3, // +y
            0, 2, 3, 0, 3, 1, // -z
            4, 5, 7, 4, 7, 6, // +z
        ];
        Ok(FuzzBox {
            buffers: FuzzBuffers { positions, indices },
        })
    }
}
