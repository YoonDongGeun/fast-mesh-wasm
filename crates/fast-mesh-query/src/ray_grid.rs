//! The nearest hits of a grid of parallel rays, for depth images.
//!
//! A [`RayGrid`] casts its rays a tile at a time ([`RAY_TILE`] squared), so
//! that rays next to each other share one walk of the BVH: the walk tests each
//! node against the beam the tile sweeps, and a leaf's triangles only against
//! the rays that cross its box. The ray-triangle test is [`raycast`]'s
//! watertight one, so a ray gets the same nearest `t` as its own
//! [`raycast`] would find.
//!
//! [`raycast`]: crate::raycast()

use fast_mesh_bvh::{Aabb, Aabb4, Bvh, NodeKind};
use fast_mesh_store::Mesh;

use crate::error::{QueryError, Result};
use crate::ray::points;
use crate::simd::{Tri4, ray_triangles4};
use crate::tri::{RayPre, ray_triangle};
use crate::{Cull, check};

/// Rays along each side of a tile.
pub const RAY_TILE: usize = 8;

/// A tile's `t` values, row by row (always [`RAY_TILE`] to a row).
pub type TileT = [f32; RAY_TILE * RAY_TILE];

/// Pixels by which a leaf's footprint is grown, against rounding: far more
/// than any `f64` error in computing it, and far less than it costs to test
/// one more ray.
const PIXEL_SLACK: f64 = 1e-3;

/// `f32` epsilons of the coordinates' size by which beams and footprints are
/// grown, for the rounding of ray origins and inside the triangle test.
const F32_SLACK: f64 = 8.0 * f32::EPSILON as f64;

/// A grid of parallel rays to cast at one mesh, in the mesh's local space:
/// ray `(col, row)` starts at `origin + col * step_col + row * step_row` and
/// runs along `dir`, `t` in units of `dir`. Computed in `f64`, each ray's
/// origin and the direction are then rounded to `f32`.
///
/// It borrows the mesh and its BVH, checked to belong together and to be
/// current when it is made. It can be shared between threads, and a tile's
/// result depends on nothing but the tile.
#[derive(Debug)]
pub struct RayGrid<'a> {
    mesh: &'a Mesh,
    bvh: &'a Bvh,
    origin: [f64; 3],
    step_col: [f64; 3],
    step_row: [f64; 3],
    /// Precomputed for the direction; its origin is set per ray.
    pre: RayPre,
    /// Map an offset from a ray's origin, along any `t`, to columns and rows.
    dual_col: [f64; 3],
    dual_row: [f64; 3],
    /// The largest absolute coordinate of the mesh's box per axis.
    reach: [f64; 3],
}

fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

/// `Σ |a[k]| b[k]`: how far a point in a box with half extents `b` gets
/// along `a`.
fn spread(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0].abs() * b[0] + a[1].abs() * b[1] + a[2].abs() * b[2]
}

impl<'a> RayGrid<'a> {
    /// The grid of rays `origin + col * step_col + row * step_row + t * dir`
    /// at `mesh`, whose BVH is `bvh`.
    ///
    /// # Errors
    /// [`QueryError::InvalidInput`] for a value that is not finite, a `dir`
    /// that is zero or infinite in `f32`, or steps that do not span a plane
    /// across `dir`; then [`QueryError::MeshMismatch`] or
    /// [`QueryError::StaleBvh`] when `bvh` does not match the current `mesh`.
    pub fn new(
        mesh: &'a Mesh,
        bvh: &'a Bvh,
        origin: [f64; 3],
        dir: [f64; 3],
        step_col: [f64; 3],
        step_row: [f64; 3],
    ) -> Result<Self> {
        let finite = |v: [f64; 3]| v.iter().all(|c| c.is_finite());
        if ![origin, dir, step_col, step_row].into_iter().all(finite) {
            return Err(QueryError::InvalidInput("a ray grid must be finite"));
        }
        let dir32 = dir.map(|c| c as f32);
        if !dir32.iter().all(|c| c.is_finite()) || dir32 == [0.0; 3] {
            return Err(QueryError::InvalidInput(
                "ray grid direction must not be zero and must fit f32",
            ));
        }
        // The rounded direction, so that the dual basis fits the rays cast.
        let d = dir32.map(f64::from);
        let across = cross(step_row, d);
        let det = dot(step_col, across);
        let dual_col = across.map(|c| c / det);
        let dual_row = cross(d, step_col).map(|c| c / det);
        if !finite(dual_col) || !finite(dual_row) {
            return Err(QueryError::InvalidInput(
                "ray grid steps must span a plane across its direction",
            ));
        }
        check(mesh, bvh)?;
        let root = bvh.root_bounds().filter(|b| !b.is_empty());
        let reach = root.map_or([0.0; 3], |b| {
            std::array::from_fn(|k| f64::from(b.min[k].abs().max(b.max[k].abs())))
        });
        Ok(RayGrid {
            mesh,
            bvh,
            origin,
            step_col,
            step_row,
            pre: RayPre::new([0.0; 3], dir32),
            dual_col,
            dual_row,
            reach,
        })
    }

    /// Casts the rays of the tile with first ray `(col, row)`, `cols` by
    /// `rows` of them (at most [`RAY_TILE`] each). Ray `(col + c, row + r)`
    /// is `t[r * RAY_TILE + c]`: in, the largest `t` that counts as a hit for
    /// it; out, the `t` of its nearest hit from `min_t` to there, or what it
    /// was when there is none. Returns the rays that were hit, as a bit mask
    /// over `t`'s indices. Both faces of a triangle count.
    ///
    /// A ray exactly through an edge or vertex hits the triangles meeting
    /// there (the test is watertight), and gets the nearest of those hits. A
    /// triangle seen edge on is not hit.
    ///
    /// # Panics
    /// If `cols` or `rows` is above [`RAY_TILE`].
    pub fn nearest_in_tile(
        &self,
        col: u32,
        row: u32,
        cols: usize,
        rows: usize,
        min_t: f32,
        t: &mut TileT,
    ) -> u64 {
        assert!(cols <= RAY_TILE && rows <= RAY_TILE, "tile too large");
        if cols == 0 || rows == 0 {
            return 0;
        }
        let at = |c: f64, r: f64| -> [f64; 3] {
            std::array::from_fn(|k| self.origin[k] + (c * self.step_col[k] + r * self.step_row[k]))
        };
        let (c0, r0) = (f64::from(col), f64::from(row));
        let first = at(c0, r0);
        let mut origins = [[0.0f32; 3]; RAY_TILE * RAY_TILE];
        let mut lo = [f32::INFINITY; 3];
        let mut hi = [f32::NEG_INFINITY; 3];
        let mut worst = f32::NEG_INFINITY;
        for r in 0..rows {
            for c in 0..cols {
                let i = r * RAY_TILE + c;
                let o = at(c0 + c as f64, r0 + r as f64).map(|x| x as f32);
                origins[i] = o;
                for k in 0..3 {
                    lo[k] = lo[k].min(o[k]);
                    hi[k] = hi[k].max(o[k]);
                }
                worst = worst.max(t[i]);
            }
        }
        // A ray that starts out of f32 range hits nothing; neither does the
        // tile around it, which the beam could not be built for.
        if !(lo.iter().chain(&hi).all(|c| c.is_finite()) && worst >= min_t) {
            return 0;
        }

        // The beam as one ray and boxes grown by the rest of it: the rays'
        // origins are within `half` of `centre`.
        let centre: [f32; 3] = std::array::from_fn(|k| 0.5 * lo[k] + 0.5 * hi[k]);
        let half: [f32; 3] = std::array::from_fn(|k| {
            let h = (hi[k] - centre[k]).max(centre[k] - lo[k]);
            let slack = F32_SLACK * (f64::from(centre[k].abs()) + self.reach[k]);
            ((f64::from(h) * (1.0 + F32_SLACK) + slack) as f32).next_up()
        });
        let grow = |b: &Aabb| Aabb {
            min: std::array::from_fn(|k| b.min[k] - half[k]),
            max: std::array::from_fn(|k| b.max[k] + half[k]),
        };
        // How far rounding can move a ray across the grid, in pixels.
        let size: [f64; 3] =
            std::array::from_fn(|k| f64::from(lo[k].abs().max(hi[k].abs())) + self.reach[k]);
        let rounding = size.map(|s| F32_SLACK * (s + size[self.pre.kz]));
        let slack_col = PIXEL_SLACK + spread(self.dual_col, rounding);
        let slack_row = PIXEL_SLACK + spread(self.dual_row, rounding);

        let start = worst;
        let mut pre = self.pre;
        let mut hit = 0u64;
        self.bvh.traverse_by_pair::<f32>(
            |a, b| {
                let boxes = Aabb4::from_two(&grow(a), &grow(b));
                let (mask, enter) = boxes.ray_slab(centre, self.pre.inv_dir, min_t, start);
                (
                    ((mask & 1) != 0).then_some(enter[0]),
                    ((mask & 2) != 0).then_some(enter[1]),
                )
            },
            |_, node, entry| {
                if entry > worst {
                    return false;
                }
                let NodeKind::Leaf { triangles } = node.kind else {
                    return true;
                };
                // The rays that cross the leaf's box.
                let b = &node.bounds;
                let (min, max) = (b.min.map(f64::from), b.max.map(f64::from));
                let mid: [f64; 3] = std::array::from_fn(|k| 0.5 * min[k] + 0.5 * max[k] - first[k]);
                let ext: [f64; 3] = std::array::from_fn(|k| 0.5 * max[k] - 0.5 * min[k]);
                let range = |dual: [f64; 3], slack: f64, n: usize| {
                    let (m, e) = (dot(mid, dual), spread(dual, ext) + slack);
                    let from = (m - e).ceil().max(0.0);
                    let to = (m + e).floor().min((n - 1) as f64);
                    (from <= to).then_some((from as usize, to as usize))
                };
                let Some((left, right)) = range(self.dual_col, slack_col, cols) else {
                    return true;
                };
                let Some((top, bottom)) = range(self.dual_row, slack_row, rows) else {
                    return true;
                };
                let mut lowered = false;
                let (chunks, rest) = triangles.as_chunks::<4>();
                for &chunk in chunks {
                    let tri4 = Tri4::from_mesh(self.mesh, chunk);
                    for r in top..=bottom {
                        for i in r * RAY_TILE + left..=r * RAY_TILE + right {
                            pre.origin = origins[i];
                            let hits = ray_triangles4(&pre, &tri4, Cull::None, min_t, t[i]);
                            for h in hits.into_iter().flatten() {
                                lowered |= h.t < t[i];
                                t[i] = t[i].min(h.t);
                                hit |= 1 << i;
                            }
                        }
                    }
                }
                for &id in rest {
                    let p = points(self.mesh, id);
                    for r in top..=bottom {
                        for i in r * RAY_TILE + left..=r * RAY_TILE + right {
                            pre.origin = origins[i];
                            if let Some(h) = ray_triangle(&pre, p, Cull::None, min_t, t[i]) {
                                lowered |= h.t < t[i];
                                t[i] = h.t;
                                hit |= 1 << i;
                            }
                        }
                    }
                }
                if lowered {
                    worst = f32::NEG_INFINITY;
                    for line in t.as_chunks::<RAY_TILE>().0.iter().take(rows) {
                        worst = line[..cols].iter().fold(worst, |w, &x| w.max(x));
                    }
                }
                true
            },
        );
        hit
    }
}

#[cfg(test)]
mod tests {
    use fast_mesh_bvh::{BuildOptions, Quality};

    use super::*;
    use crate::test_util::{Lcg, build, random_normal, random_rotation, uv_sphere, wavy};
    use crate::{Ray, RayOptions, raycast};

    /// Casts a whole `w` by `h` grid, tile by tile, from `max_t`.
    fn cast(grid: &RayGrid, w: usize, h: usize, min_t: f32, max_t: f32) -> Vec<Option<f32>> {
        let mut out = vec![None; w * h];
        for row in (0..h).step_by(RAY_TILE) {
            for col in (0..w).step_by(RAY_TILE) {
                let (cols, rows) = (RAY_TILE.min(w - col), RAY_TILE.min(h - row));
                let mut t = [max_t; RAY_TILE * RAY_TILE];
                let hit = grid.nearest_in_tile(col as u32, row as u32, cols, rows, min_t, &mut t);
                for r in 0..RAY_TILE {
                    for c in 0..RAY_TILE {
                        let i = r * RAY_TILE + c;
                        if hit & (1 << i) == 0 {
                            assert_eq!(t[i].to_bits(), max_t.to_bits());
                            continue;
                        }
                        assert!(c < cols && r < rows && t[i] >= min_t && t[i] <= max_t);
                        out[(row + r) * w + col + c] = Some(t[i]);
                    }
                }
            }
        }
        out
    }

    /// The same through one [`raycast`] per ray.
    fn cast_each(grid: &RayGrid, w: usize, h: usize, min_t: f32, max_t: f32) -> Vec<Option<f32>> {
        let opts = RayOptions {
            min_t,
            max_t,
            cull: Cull::None,
        };
        (0..w * h)
            .map(|i| {
                let (c, r) = ((i % w) as f64, (i / w) as f64);
                let origin = std::array::from_fn(|k| {
                    (grid.origin[k] + (c * grid.step_col[k] + r * grid.step_row[k])) as f32
                });
                let ray = Ray {
                    origin,
                    dir: grid.pre.dir,
                };
                raycast(grid.mesh, grid.bvh, &ray, &opts)
                    .unwrap()
                    .map(|h| h.t)
            })
            .collect()
    }

    fn bits(v: &[Option<f32>]) -> Vec<Option<u32>> {
        v.iter().map(|t| t.map(f32::to_bits)).collect()
    }

    /// A grid of `w` by `h` rays along `dir` that covers the box of half
    /// size `r` about the origin, starting `back` before it.
    fn cover(dir: [f64; 3], r: f64, back: f64, w: usize, h: usize) -> [[f64; 3]; 4] {
        let hint = if dir[0].abs() < 0.9 {
            [1.0, 0.0, 0.0]
        } else {
            [0.0, 1.0, 0.0]
        };
        let unit = |v: [f64; 3]| v.map(|c| c / dot(v, v).sqrt());
        let d = unit(dir);
        let u = unit(cross(d, hint));
        let v = cross(d, u);
        let (pu, pv) = (2.0 * r / w as f64, 2.0 * r / h as f64);
        let (su, sv) = (u.map(|c| c * pu), v.map(|c| c * pv));
        let origin = std::array::from_fn(|k| {
            -(back * d[k]) - 0.5 * ((w - 1) as f64 * su[k] + (h - 1) as f64 * sv[k])
        });
        [origin, dir, su, sv]
    }

    #[test]
    fn tiles_match_a_raycast_per_ray() {
        let mut rng = Lcg(7);
        let meshes = [uv_sphere(24, 32, 1.0), wavy(12), uv_sphere(3, 3, 1.0)];
        for mesh in &meshes {
            for quality in [Quality::Sah, Quality::Fast] {
                let bvh = build(mesh, quality);
                let coarse = Bvh::build(mesh, &BuildOptions::default()).unwrap();
                for k in 0..12 {
                    let dir = match k {
                        0 => [0.0, 0.0, -1.0],
                        1 => [0.0, 1.0, 0.0],
                        2 => [-3.0, 0.0, 0.0],
                        _ => random_normal(&mut rng),
                    };
                    let (w, h) = [(40, 27), (8, 8), (9, 1), (70, 64)][k % 4];
                    let r = [1.1, 14.0, 0.3][k % 3];
                    let [o, d, su, sv] = cover(dir, r, [30.0, 0.0][k % 2], w, h);
                    for bvh in [&bvh, &coarse] {
                        let grid = RayGrid::new(mesh, bvh, o, d, su, sv).unwrap();
                        for (min_t, max_t) in [(0.0, f32::INFINITY), (0.0, 12.0), (3.0, 9.0)] {
                            let got = cast(&grid, w, h, min_t, max_t);
                            let want = cast_each(&grid, w, h, min_t, max_t);
                            assert_eq!(bits(&got), bits(&want), "case {k}");
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn sheared_grids_match_too() {
        // As a non-uniformly scaled placement makes them: steps that are not
        // perpendicular to each other or to the direction.
        let mut rng = Lcg(11);
        let mesh = uv_sphere(20, 24, 1.0);
        let bvh = build(&mesh, Quality::Sah);
        for _ in 0..20 {
            let m = random_rotation(&mut rng);
            let scale: [f64; 3] = std::array::from_fn(|_| 0.2 + 2.0 * f64::from(rng.next()));
            let map = |v: [f64; 3]| -> [f64; 3] {
                std::array::from_fn(|k| {
                    (m[k][0] * v[0] + m[k][1] * v[1] + m[k][2] * v[2]) / scale[k]
                })
            };
            let [o, d, su, sv] = cover(random_normal(&mut rng), 2.0, 5.0, 33, 30);
            let grid = RayGrid::new(&mesh, &bvh, map(o), map(d), map(su), map(sv)).unwrap();
            let got = cast(&grid, 33, 30, 0.0, f32::INFINITY);
            assert!(got.iter().any(Option::is_some));
            let want = cast_each(&grid, 33, 30, 0.0, f32::INFINITY);
            assert_eq!(bits(&got), bits(&want));
        }
    }

    /// A closed box of `n` cells per side from 0 to `n`, so that grid lines
    /// and cell diagonals lie on whole coordinates.
    fn lattice_box(n: u32) -> Mesh {
        let mut positions = Vec::new();
        let mut index = std::collections::HashMap::new();
        let mut triangles: Vec<u32> = Vec::new();
        let mut vertex = |p: [u32; 3]| -> u32 {
            *index.entry(p).or_insert_with(|| {
                positions.extend(p.map(|c| c as f32));
                (positions.len() / 3 - 1) as u32
            })
        };
        for axis in 0..3 {
            for side in [0, n] {
                for j in 0..n {
                    for i in 0..n {
                        let at = |di: u32, dj: u32| {
                            let mut p = [0; 3];
                            p[axis] = side;
                            p[(axis + 1) % 3] = i + di;
                            p[(axis + 2) % 3] = j + dj;
                            p
                        };
                        let q = [at(0, 0), at(1, 0), at(1, 1), at(0, 1)].map(&mut vertex);
                        let [a, b, c, d] = q;
                        if side == n {
                            triangles.extend([a, b, c, a, c, d]);
                        } else {
                            triangles.extend([a, c, b, a, d, c]);
                        }
                    }
                }
            }
        }
        Mesh::from_buffers(&positions, &triangles).unwrap()
    }

    #[test]
    fn rays_through_edges_and_vertices_take_the_nearest_hit() {
        // Rays on every half unit of a box's top: through vertices, along
        // grid lines and cell diagonals, and between them. Each must see the
        // top once, and none the bottom through a crack; on the silhouette,
        // where the top meets the sides (edge on, never hit), too.
        let n = 10;
        let mesh = lattice_box(n);
        for quality in [Quality::Sah, Quality::Fast] {
            let bvh = build(&mesh, quality);
            for (axis, sign) in [(2, -1.0), (2, 1.0), (0, 1.0), (1, -1.0)] {
                let mut origin = [0.0; 3];
                origin[axis] = if sign < 0.0 { 25.0 } else { -15.0 };
                let mut dir = [0.0; 3];
                dir[axis] = sign;
                let (mut su, mut sv) = ([0.0; 3], [0.0; 3]);
                su[(axis + 1) % 3] = 0.5;
                sv[(axis + 2) % 3] = 0.5;
                let side = 2 * n as usize + 1;
                let grid = RayGrid::new(&mesh, &bvh, origin, dir, su, sv).unwrap();
                let got = cast(&grid, side, side, 0.0, f32::INFINITY);
                assert_eq!(got, vec![Some(15.0); side * side]);
                // One step past the silhouette there is nothing.
                let past: [f64; 3] = std::array::from_fn(|k| origin[k] - su[k] - sv[k]);
                let grid = RayGrid::new(&mesh, &bvh, past, dir, su, sv).unwrap();
                let got = cast(&grid, side + 2, side + 2, 0.0, f32::INFINITY);
                for (i, t) in got.iter().enumerate() {
                    let (c, r) = (i % (side + 2), i / (side + 2));
                    let inside = (1..=side).contains(&c) && (1..=side).contains(&r);
                    assert_eq!(t.is_some(), inside, "ray {c} {r}");
                }
            }
        }
    }

    #[test]
    fn bounds_are_inclusive_and_hits_lower_them() {
        let mesh = lattice_box(2);
        let bvh = build(&mesh, Quality::Sah);
        let (x, y) = ([1.0, 0.0, 0.0], [0.0, 1.0, 0.0]);
        let start = [0.5, 0.5, 6.0];
        let grid = RayGrid::new(&mesh, &bvh, start, [0.0, 0.0, -2.0], x, y).unwrap();
        // The top is at t = 2, the bottom at t = 3.
        let mut t = [0.0; RAY_TILE * RAY_TILE];
        (t[0], t[1]) = (2.0, 1.9);
        (t[RAY_TILE], t[RAY_TILE + 1]) = (f32::INFINITY, 2.5);
        let hit = grid.nearest_in_tile(0, 0, 2, 2, 0.0, &mut t);
        assert_eq!(hit, 1 | 1 << RAY_TILE | 1 << (RAY_TILE + 1));
        assert_eq!((t[0], t[1]), (2.0, 1.9));
        assert_eq!((t[RAY_TILE], t[RAY_TILE + 1]), (2.0, 2.0));
        let mut t = [2.0; RAY_TILE * RAY_TILE];
        assert_eq!(grid.nearest_in_tile(0, 0, 2, 2, 2.0, &mut t), hit | 2);
        let mut t = [2.5; RAY_TILE * RAY_TILE];
        assert_eq!(grid.nearest_in_tile(0, 0, 2, 2, 2.1, &mut t), 0);
        assert_eq!(grid.nearest_in_tile(0, 0, 0, 2, 0.0, &mut t), 0);
        assert_eq!(grid.nearest_in_tile(50, 0, 2, 2, 0.0, &mut t), 0);
        // A tile beyond f32 range hits nothing rather than failing.
        let far = [1e38, 0.0, 0.0];
        let grid = RayGrid::new(&mesh, &bvh, start, [0.0, 0.0, -1.0], far, y).unwrap();
        let mut t = [f32::INFINITY; RAY_TILE * RAY_TILE];
        assert_eq!(grid.nearest_in_tile(4000, 0, 8, 2, 0.0, &mut t), 0);
        assert_eq!(grid.nearest_in_tile(0, 0, 1, 1, 0.0, &mut t), 1);
    }

    #[test]
    fn bad_grids_are_refused() {
        let mesh = lattice_box(1);
        let bvh = build(&mesh, Quality::Sah);
        let new = |o, d, u, v| RayGrid::new(&mesh, &bvh, o, d, u, v).map(|_| ());
        let (x, y, z) = ([1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]);
        assert_eq!(new([0.0; 3], z, x, y), Ok(()));
        for bad in [
            new([f64::NAN, 0.0, 0.0], z, x, y),
            new([0.0; 3], [0.0; 3], x, y),
            new([0.0; 3], [1e-60, 0.0, 0.0], x, y),
            new([0.0; 3], [0.0, 1e300, 0.0], x, z),
            new([0.0; 3], z, x, x),
            new([0.0; 3], z, x, z),
            new([0.0; 3], z, [0.0; 3], y),
            new([0.0; 3], z, x, [f64::INFINITY, 0.0, 0.0]),
        ] {
            assert!(matches!(bad, Err(QueryError::InvalidInput(_))), "{bad:?}");
        }
        let other = lattice_box(1);
        let e = RayGrid::new(&other, &bvh, [0.0; 3], z, x, y).map(|_| ());
        assert_eq!(e, Err(QueryError::MeshMismatch));
        let empty = Mesh::new();
        let none = Bvh::build(&empty, &BuildOptions::default()).unwrap();
        let grid = RayGrid::new(&empty, &none, [0.0; 3], z, x, y).unwrap();
        let mut t = [f32::INFINITY; RAY_TILE * RAY_TILE];
        assert_eq!(grid.nearest_in_tile(0, 0, 8, 8, 0.0, &mut t), 0);
    }
}
