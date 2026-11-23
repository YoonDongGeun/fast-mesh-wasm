//! Height fields as solids: [`heightfield`] turns a grid of heights (an
//! image's pixels, say) into a relief with a flat base.
//!
//! The samples are the vertices of the top surface: one vertex per sample,
//! at exactly its height. In the mesh's own frame `x` runs along a row (an
//! image's right), `y` against the rows (the first row is at the top, as in
//! an image) and `z` is the height, so that the relief reads like the image
//! from `+z`. It is centred on the origin in `x` and `y`; height 0 is the
//! plane `z = 0` and the base is the plane `z = -base`. It keeps the rules of
//! [`primitives`](crate::primitives): welded, closed, and wound
//! counter-clockwise seen from outside.
//!
//! - Each cell of four samples is split along the diagonal whose ends differ
//!   less in height (the one from its top left sample on a tie), so that
//!   ridges and valleys follow the samples in both diagonal directions.
//! - The sides go straight down from the rim of the top to a ring of as many
//!   vertices on the base.
//! - The base uses the ring's vertices only, joined across by a zipper, so it
//!   adds about `2 (w + h)` triangles rather than a second grid, and few
//!   triangles meet at any of its vertices.
//!
//! A `w` by `h` grid has `w h + r` vertices and `2 (w - 1) (h - 1) + 3 r - 2`
//! triangles, with `r = 2 (w + h) - 4` the length of the rim.
//!
//! Buffers are reserved fallibly and filled in parallel with the `parallel`
//! feature; the output depends on the input only, bit for bit.

use fast_mesh_store::{StoreError, par};

use crate::append::AppendSource;
use crate::primitives::{ids, length};
use crate::{OpsError, Result, filled};

/// Options for [`heightfield`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HeightfieldOptions {
    /// Distance between samples along a row, and between rows. Default
    /// `[1, 1]`.
    pub pixel_size: [f64; 2],
    /// Thickness of the base below height 0. Every height must be above
    /// `-base`. Default 1.
    pub base: f64,
}

impl Default for HeightfieldOptions {
    fn default() -> Self {
        HeightfieldOptions {
            pixel_size: [1.0; 2],
            base: 1.0,
        }
    }
}

/// Most samples along a side: past `2^24` neighbouring columns would no
/// longer have distinct `f32` positions, and this leaves room for rounding.
const MAX_SIDE: u32 = 1 << 22;

/// Samples from which the buffers are filled in parallel.
const PAR_MIN_SAMPLES: usize = 1 << 16;

/// Samples per parallel task, rounded to whole rows.
const CHUNK_SAMPLES: usize = 1 << 14;

/// Vertices and triangles of the relief of a `width` by `height` grid,
/// without building it.
///
/// # Errors
/// [`OpsError::InvalidInput`] as for [`heightfield`]: fewer than two samples
/// on a side, too many, or a relief too large for `u32` ids.
pub fn heightfield_size(width: u32, height: u32) -> Result<(u32, u32)> {
    if width < 2 || height < 2 {
        return Err(OpsError::InvalidInput(
            "a height field needs at least 2 samples on each side",
        ));
    }
    if width > MAX_SIDE || height > MAX_SIDE {
        return Err(OpsError::InvalidInput(
            "a height field has at most 4194304 samples on each side",
        ));
    }
    let (w, h) = (u128::from(width), u128::from(height));
    let rim = 2 * (w + h) - 4;
    Ok((ids(w * h + rim)?, ids(2 * (w - 1) * (h - 1) + 3 * rim - 2)?))
}

/// The height a sample stands at: NaN (no surface there) is 0.
fn sample(h: f32) -> f32 {
    if h.is_nan() { 0.0 } else { h }
}

/// The relief of `heights`, a `width` by `height` grid in rows from the top
/// left sample; see the [module docs](self). A NaN height, as a height map
/// has where it saw no surface, stands at 0.
///
/// # Errors
/// [`OpsError::InvalidInput`] for a grid with fewer than two samples or more
/// than `2^22` on a side, a `heights` length that is not `width * height`, a
/// relief too large for `u32` ids, a pixel size, extent or base that is not
/// finite and within `[1e-30, 1e30]`, or a height that is infinite or not
/// above `-base`, before anything is allocated; [`OpsError::Store`] when the
/// output cannot be allocated.
pub fn heightfield(
    heights: &[f32],
    width: u32,
    height: u32,
    options: &HeightfieldOptions,
) -> Result<AppendSource> {
    let (vertices, triangles) = heightfield_size(width, height)?;
    let (w, h) = (width as usize, height as usize);
    if heights.len() != w * h {
        return Err(OpsError::InvalidInput(
            "heights must hold width times height values",
        ));
    }
    let what =
        "the pixel size and the height field's extent must be finite and within [1e-30, 1e30]";
    let pitch = [
        length(options.pixel_size[0], what)?,
        length(options.pixel_size[1], what)?,
    ];
    length(pitch[0] * f64::from(width - 1), what)?;
    length(pitch[1] * f64::from(height - 1), what)?;
    let base = length(options.base, "base must be finite and within [1e-30, 1e30]")?;
    let floor = -(base as f32);
    par::map_chunks(
        heights,
        CHUNK_SAMPLES,
        PAR_MIN_SAMPLES,
        || OpsError::Store(StoreError::OutOfMemory),
        |_, chunk| {
            let fits = |&z: &f32| z.is_nan() || (z > floor && z < f32::INFINITY);
            if chunk.iter().all(fits) {
                Ok(())
            } else {
                Err(OpsError::InvalidInput(
                    "heights must be finite and above the base plane (-base)",
                ))
            }
        },
    )?;

    let mut positions: Vec<[f32; 3]> = filled(vertices as usize, [0.0; 3])?;
    let mut triangles: Vec<[u32; 3]> = filled(triangles as usize, [0; 3])?;
    let rows = (CHUNK_SAMPLES / w).max(1);

    // The top: a vertex per sample, two triangles per cell.
    let (x0, y0) = (0.5 * f64::from(width - 1), 0.5 * f64::from(height - 1));
    let x = |col: usize| (pitch[0] * (col as f64 - x0)) as f32;
    let y = |row: usize| (pitch[1] * (y0 - row as f64)) as f32;
    let (top, ring) = positions.split_at_mut(w * h);
    par::for_each_chunk_mut(top, rows * w, PAR_MIN_SAMPLES, |i, chunk| {
        for (row, out) in (i * rows..).zip(chunk.chunks_exact_mut(w)) {
            let y = y(row);
            let line = &heights[row * w..(row + 1) * w];
            for (col, (p, &z)) in out.iter_mut().zip(line).enumerate() {
                *p = [x(col), y, sample(z)];
            }
        }
    });
    let per_row = 2 * (w - 1);
    let (top, rest) = triangles.split_at_mut(per_row * (h - 1));
    par::for_each_chunk_mut(top, rows * per_row, PAR_MIN_SAMPLES, |i, chunk| {
        for (row, out) in (i * rows..).zip(chunk.chunks_exact_mut(per_row)) {
            let upper = &heights[row * w..(row + 1) * w];
            let lower = &heights[(row + 1) * w..(row + 2) * w];
            for (col, pair) in out.as_chunks_mut::<2>().0.iter_mut().enumerate() {
                // The cell's corners: a top left, b top right, c bottom
                // right, d bottom left.
                let a = (row * w + col) as u32;
                let (b, d) = (a + 1, a + width);
                let c = d + 1;
                let ac = (sample(upper[col]) - sample(lower[col + 1])).abs();
                let bd = (sample(upper[col + 1]) - sample(lower[col])).abs();
                *pair = if ac <= bd {
                    [[a, d, c], [a, c, b]]
                } else {
                    [[a, d, b], [d, c, b]]
                };
            }
        }
    });

    // The rim's samples, counter-clockwise seen from above: down the left
    // side, along the bottom row, up the right side, back along the top row.
    let rim = ring.len();
    let (down, along) = (h - 1, w - 1);
    let rim_sample = |k: usize| -> usize {
        if k <= down {
            k * w
        } else if k <= down + along {
            down * w + (k - down)
        } else if k <= 2 * down + along {
            (2 * down + along - k) * w + along
        } else {
            rim - k
        }
    };
    for (k, p) in ring.iter_mut().enumerate() {
        let s = rim_sample(k);
        *p = [x(s % w), y(s / w), floor];
    }
    let bottom = |k: usize| (w * h + k) as u32;
    let (sides, base) = rest.split_at_mut(2 * rim);
    for (k, pair) in sides.as_chunks_mut::<2>().0.iter_mut().enumerate() {
        let next = (k + 1) % rim;
        let (t0, t1) = (rim_sample(k) as u32, rim_sample(next) as u32);
        pair[0] = [t1, t0, bottom(k)];
        pair[1] = [t1, bottom(k), bottom(next)];
    }
    // The base: the ring split at two opposite corners into two chains of
    // `half` edges each, zipped together by rungs between them. No rung lies
    // along a side, so no triangle is degenerate.
    let half = rim / 2;
    let left = |i: usize| bottom(i);
    let right = |i: usize| bottom((rim - i) % rim);
    base[0] = [left(0), right(1), left(1)];
    for (i, pair) in (1..half - 1).zip(base[1..].as_chunks_mut::<2>().0) {
        pair[0] = [left(i), right(i), right(i + 1)];
        pair[1] = [left(i), right(i + 1), left(i + 1)];
    }
    base[2 * half - 3] = [left(half - 1), right(half - 1), left(half)];
    Ok(AppendSource {
        positions,
        triangles,
    })
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use fast_mesh_store::Mesh;

    use super::*;
    use crate::{fail_allocations_after, measure};

    /// A deterministic bumpy grid with heights in `[0, 4]`.
    fn bumps(w: usize, h: usize) -> Vec<f32> {
        (0..w * h)
            .map(|i| {
                let (x, y) = ((i % w) as f32, (i / w) as f32);
                2.0 + (x * 0.7).sin() + (y * 0.45 + x * 0.1).cos()
            })
            .collect()
    }

    fn edge_key(a: u32, b: u32) -> (u32, u32) {
        (a.min(b), a.max(b))
    }

    /// Checks that `src` is welded, uses every vertex, has no degenerate
    /// triangle, and is closed with consistent winding (each directed edge
    /// once, with its opposite). Returns the Euler characteristic and the
    /// most triangles at one vertex.
    fn closed(src: &AppendSource) -> (i64, usize) {
        let v = src.positions.len();
        let mut bits: Vec<[u32; 3]> = src.positions.iter().map(|p| p.map(f32::to_bits)).collect();
        bits.sort_unstable();
        bits.dedup();
        assert_eq!(bits.len(), v, "two vertices share a position");
        let mut valence = vec![0usize; v];
        let mut directed: HashMap<(u32, u32), usize> = HashMap::new();
        for t in &src.triangles {
            let p = t.map(|i| src.positions[i as usize].map(f64::from));
            let e = |k: usize| -> [f64; 3] { std::array::from_fn(|i| p[k][i] - p[0][i]) };
            let (a, b) = (e(1), e(2));
            let n = [
                a[1] * b[2] - a[2] * b[1],
                a[2] * b[0] - a[0] * b[2],
                a[0] * b[1] - a[1] * b[0],
            ];
            assert!(n.iter().any(|&c| c != 0.0), "degenerate triangle {t:?}");
            for k in 0..3 {
                valence[t[k] as usize] += 1;
                *directed.entry((t[k], t[(k + 1) % 3])).or_default() += 1;
            }
        }
        assert!(valence.iter().all(|&n| n > 0), "unused vertex");
        for (&(a, b), &n) in &directed {
            assert_eq!(n, 1, "edge {a} {b} crossed twice");
            assert!(directed.contains_key(&(b, a)), "boundary edge {a} {b}");
        }
        let edges = directed.len() as i64 / 2;
        (
            v as i64 - edges + src.triangles.len() as i64,
            valence.into_iter().max().unwrap(),
        )
    }

    fn mesh(src: &AppendSource) -> Mesh {
        Mesh::from_buffers(src.positions.as_flattened(), src.triangles.as_flattened()).unwrap()
    }

    #[test]
    fn reliefs_are_closed_and_their_heights_match_the_samples() {
        for (w, h) in [
            (2, 2),
            (2, 3),
            (3, 2),
            (2, 9),
            (9, 2),
            (3, 3),
            (17, 5),
            (5, 17),
            (33, 40),
        ] {
            let heights = bumps(w, h);
            let o = HeightfieldOptions {
                pixel_size: [0.5, 0.25],
                base: 1.5,
            };
            let src = heightfield(&heights, w as u32, h as u32, &o).unwrap();
            let size = heightfield_size(w as u32, h as u32).unwrap();
            assert_eq!(
                (src.positions.len(), src.triangles.len()),
                (size.0 as usize, size.1 as usize)
            );
            let (chi, valence) = closed(&src);
            assert_eq!(chi, 2, "{w} x {h}");
            assert!(valence <= 8, "{valence} triangles at one vertex");
            for (i, (p, &z)) in src.positions.iter().zip(&heights).enumerate() {
                assert_eq!(p[2].to_bits(), z.to_bits());
                let (col, row) = ((i % w) as f32, (i / w) as f32);
                assert_eq!(p[0], 0.5 * (col - 0.5 * (w - 1) as f32));
                assert_eq!(p[1], 0.25 * (0.5 * (h - 1) as f32 - row));
            }
            let floor: Vec<_> = src.positions[w * h..].iter().map(|p| p[2]).collect();
            assert!(floor.iter().all(|&z| z == -1.5));
            let m = measure(&mesh(&src)).unwrap();
            assert!(m.closed && m.boundary_loops == 0 && m.non_manifold_edges == 0);
            assert_eq!(m.orientation_conflicts, 0);
            // Under each triangle of the top: the mean of its heights, plus
            // the base, over half a cell.
            let top = &src.triangles[..2 * (w - 1) * (h - 1)];
            let want: f64 = top
                .iter()
                .map(|t| {
                    let sum: f64 = t.iter().map(|&i| f64::from(heights[i as usize])).sum();
                    (sum / 3.0 + 1.5) * 0.5 * 0.5 * 0.25
                })
                .sum();
            let volume = m.volume.unwrap();
            assert!((volume - want).abs() <= 1e-5 * want, "{volume} vs {want}");
        }
    }

    #[test]
    fn a_flat_relief_is_a_box() {
        let o = HeightfieldOptions {
            pixel_size: [2.0, 3.0],
            base: 0.5,
        };
        let src = heightfield(&[1.5; 20], 5, 4, &o).unwrap();
        let volume = measure(&mesh(&src)).unwrap().volume.unwrap();
        assert!((volume - 8.0 * 9.0 * 2.0).abs() < 1e-4, "{volume}");
        let (lo, hi) = src
            .positions
            .iter()
            .fold(([f32::MAX; 3], [f32::MIN; 3]), |(lo, hi), p| {
                (
                    std::array::from_fn(|k| p[k].min(lo[k])),
                    std::array::from_fn(|k| p[k].max(hi[k])),
                )
            });
        assert_eq!((lo, hi), ([-4.0, -4.5, -0.5], [4.0, 4.5, 1.5]));
    }

    #[test]
    fn cells_split_along_their_flatter_diagonal() {
        // A ridge from the top left to the bottom right, and one the other
        // way: the diagonal edge lies along each.
        let o = HeightfieldOptions::default();
        let has = |src: &AppendSource, a: u32, b: u32| {
            src.triangles[..2]
                .iter()
                .all(|t| (0..3).any(|k| edge_key(t[k], t[(k + 1) % 3]) == edge_key(a, b)))
        };
        assert!(has(
            &heightfield(&[1., 0., 0.5, 1.], 2, 2, &o).unwrap(),
            0,
            3
        ));
        assert!(has(
            &heightfield(&[0., 1., 1., 0.5], 2, 2, &o).unwrap(),
            1,
            2
        ));
        assert!(has(&heightfield(&[0.; 4], 2, 2, &o).unwrap(), 0, 3));
    }

    #[test]
    fn missing_samples_stand_at_zero() {
        let mut heights = bumps(6, 5);
        heights[7] = f32::NAN;
        heights[29] = f32::NAN;
        let src = heightfield(&heights, 6, 5, &HeightfieldOptions::default()).unwrap();
        assert_eq!((src.positions[7][2], src.positions[29][2]), (0.0, 0.0));
        assert_eq!(closed(&src).0, 2);
        // Negative heights are fine above the base.
        let src = heightfield(&[-0.5; 4], 2, 2, &HeightfieldOptions::default()).unwrap();
        assert!(measure(&mesh(&src)).unwrap().volume.unwrap() > 0.0);
    }

    #[test]
    fn bad_input_is_refused_before_allocating() {
        let o = HeightfieldOptions::default();
        let bad = |heights: &[f32], w: u32, h: u32, o: &HeightfieldOptions| {
            fail_allocations_after(Some(0));
            let r = heightfield(heights, w, h, o);
            fail_allocations_after(None);
            assert!(matches!(r, Err(OpsError::InvalidInput(_))), "{r:?}");
        };
        bad(&[0.0; 2], 1, 2, &o);
        bad(&[0.0; 2], 2, 1, &o);
        bad(&[0.0; 5], 2, 2, &o);
        bad(&[], MAX_SIDE + 1, 2, &o);
        bad(&[], 1 << 20, 1 << 20, &o);
        for z in [-1.0, -2.0, f32::INFINITY, f32::NEG_INFINITY] {
            bad(&[0.0, 0.0, z, 0.0], 2, 2, &o);
        }
        for size in [0.0, -1.0, f64::NAN, f64::INFINITY, 1e31, 1e-31] {
            let pixel_size = [1.0, size];
            bad(&[0.0; 4], 2, 2, &HeightfieldOptions { pixel_size, ..o });
            bad(&[0.0; 4], 2, 2, &HeightfieldOptions { base: size, ..o });
        }
        let pixel_size = [9e29, 1.0];
        bad(&[0.0; 6], 3, 2, &HeightfieldOptions { pixel_size, ..o });
    }

    #[test]
    fn out_of_memory_is_reported() {
        for n in 0..2 {
            fail_allocations_after(Some(n));
            let r = heightfield(&[0.0; 4], 2, 2, &HeightfieldOptions::default());
            fail_allocations_after(None);
            assert_eq!(r, Err(OpsError::Store(StoreError::OutOfMemory)));
        }
    }

    #[test]
    fn parallel_fills_match_the_sequential_ones() {
        for (w, h) in [(513, 300), (70_000, 2), (2, 70_000)] {
            let heights = bumps(w, h);
            let o = HeightfieldOptions::default();
            let parallel = heightfield(&heights, w as u32, h as u32, &o).unwrap();
            par::set_force_sequential(true);
            let sequential = heightfield(&heights, w as u32, h as u32, &o).unwrap();
            par::set_force_sequential(false);
            let bits = |s: &AppendSource| -> Vec<u32> {
                s.positions.iter().flatten().map(|c| c.to_bits()).collect()
            };
            assert_eq!(bits(&parallel), bits(&sequential));
            assert_eq!(parallel.triangles, sequential.triangles);
        }
        // A bad height in a late chunk is still found.
        let mut heights = bumps(513, 300);
        heights[513 * 299 + 100] = -9.0;
        let r = heightfield(&heights, 513, 300, &HeightfieldOptions::default());
        assert!(matches!(r, Err(OpsError::InvalidInput(_))));
    }
}
