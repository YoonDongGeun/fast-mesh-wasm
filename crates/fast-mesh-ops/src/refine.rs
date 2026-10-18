//! Refining a hole patch to the density of the mesh around it.
//!
//! [`refine`] follows Liepa (2003): every loop vertex gets a target edge
//! length, the mean length of its mesh edges, and every new vertex the mean
//! of its parents'. Rounds of centroid splits refine the triangles that are
//! coarse for those lengths, each round followed by edge flips that relax
//! the patch towards Delaunay, until a round splits nothing. Only patch
//! edges are flipped; the loop edges stay.

use std::f64::consts::{PI, SQRT_2};

use fast_mesh_store::{Mesh, StoreError, VertexId};

use crate::topology::{EdgeMap, VertexFans, cross, dot, point, sub};
use crate::{OpsError, Result, push, reserve};

/// Rounds of splits at most; each round refines every triangle that is
/// still coarse, so this is never reached at sane densities.
const MAX_ROUNDS: usize = 64;

/// Flip passes after a round of splits at most.
const MAX_FLIP_PASSES: usize = 32;

/// A triangle counts as degenerate when its doubled area is at most this
/// times its longest edge squared.
const DEGENERATE: f64 = 1e-12;

/// A flip must improve the opposite-angle sum by more than this (radians).
const ANGLE_EPSILON: f64 = 1e-9;

/// A loop's patch after refinement, in local vertex indices: local `i` is the
/// mesh vertex `old[i]` for `i < old.len()`, and otherwise new vertex
/// `i - old.len()`.
pub(crate) struct RefinedPatch {
    /// The distinct mesh vertices of the loop, ascending.
    pub(crate) old: Vec<VertexId>,
    /// The position of every local vertex: the old ones, then the new ones
    /// in creation order.
    pub(crate) points: Vec<[f64; 3]>,
    /// The creation parents of each new vertex, as local indices with their
    /// weights: the corners of the triangle it split, 1/3 each. A parent is
    /// an old vertex or a new one created before it.
    pub(crate) parents: Vec<[(u32, f32); 3]>,
    /// The patch triangles, wound like the triangulation.
    pub(crate) triangles: Vec<[u32; 3]>,
}

impl RefinedPatch {
    /// The number of new vertices.
    pub(crate) fn new_vertices(&self) -> usize {
        self.points.len() - self.old.len()
    }
}

/// The triangulation `patch` of the loop `loop_vertices` refined to the
/// density of the mesh around it.
///
/// - Each loop vertex's target length σ is the mean length of its edges in
///   `mesh` (its neighbours in `fans`); a loop vertex without a usable one
///   takes the mean of the others. A new vertex takes the mean σ of its
///   parents.
/// - A triangle is split at its centroid `c` when, for each corner `m`,
///   `√2 · |c − m|` exceeds both σ(c) and σ(m) (Liepa's density rule);
///   degenerate triangles are not split.
/// - After each round of splits, interior patch edges are flipped while the
///   two angles opposite an edge sum to more than π (the circumcircle
///   criterion), the flip creates no edge the patch or `edges` already has
///   and folds neither new triangle against the old pair.
///
/// The winding of `patch` is kept. With no usable σ at all, `patch` comes
/// back unrefined.
///
/// # Errors
/// [`OpsError::Store`] when memory runs out.
pub(crate) fn refine(
    mesh: &Mesh,
    fans: &VertexFans,
    edges: &EdgeMap,
    loop_vertices: &[VertexId],
    patch: &[[VertexId; 3]],
) -> Result<RefinedPatch> {
    let mut old = Vec::new();
    reserve(&mut old, loop_vertices.len())?;
    old.extend_from_slice(loop_vertices);
    old.sort_unstable();
    old.dedup();
    let mut points = Vec::new();
    reserve(&mut points, old.len())?;
    points.extend(old.iter().map(|&v| point(mesh, v)));
    let mut sigma = Vec::new();
    reserve(&mut sigma, old.len())?;
    let mut scratch = Vec::new();
    for &v in &old {
        sigma.push(mean_edge(mesh, fans, v, &mut scratch)?);
    }
    let usable = |s: &f64| s.is_finite() && *s > 0.0;
    let count = sigma.iter().filter(|s| usable(s)).count();
    let mean = sigma.iter().filter(|s| usable(s)).sum::<f64>() / count as f64;
    let mut triangles = Vec::new();
    reserve(&mut triangles, patch.len())?;
    // Every patch corner is a loop vertex.
    triangles.extend(
        patch
            .iter()
            .map(|t| t.map(|v| old.binary_search(&v).map_or(0, |i| i as u32))),
    );
    let mut refined = RefinedPatch {
        old,
        points,
        parents: Vec::new(),
        triangles,
    };
    if count == 0 {
        return Ok(refined);
    }
    for s in &mut sigma {
        if !usable(s) {
            *s = mean;
        }
    }
    let mut relax = Relax::default();
    for _ in 0..MAX_ROUNDS {
        let split = split_round(&mut refined, &mut sigma)?;
        relax.run(&mut refined, edges)?;
        if !split {
            break;
        }
    }
    Ok(refined)
}

/// The mean length of the mesh edges at `v`, NaN when it has none.
fn mean_edge(
    mesh: &Mesh,
    fans: &VertexFans,
    v: VertexId,
    scratch: &mut Vec<VertexId>,
) -> Result<f64> {
    scratch.clear();
    let fan = fans.fan(v);
    reserve(scratch, 2 * fan.len())?;
    for &t in fan {
        scratch.extend(mesh.triangles()[t as usize].iter().filter(|&&u| u != v));
    }
    scratch.sort_unstable();
    scratch.dedup();
    let p = point(mesh, v);
    let total: f64 = scratch
        .iter()
        .map(|&u| {
            let d = sub(point(mesh, u), p);
            dot(d, d).sqrt()
        })
        .sum();
    Ok(total / scratch.len() as f64)
}

fn distance(a: [f64; 3], b: [f64; 3]) -> f64 {
    let d = sub(a, b);
    dot(d, d).sqrt()
}

/// The normal of `[a, b, c]` (twice the area), or `None` when degenerate.
fn normal(a: [f64; 3], b: [f64; 3], c: [f64; 3]) -> Option<[f64; 3]> {
    let n = cross(sub(b, a), sub(c, a));
    let longest = [distance(a, b), distance(b, c), distance(c, a)]
        .into_iter()
        .fold(0.0, f64::max);
    let len = dot(n, n).sqrt();
    (len.is_finite() && len > DEGENERATE * longest * longest).then_some(n)
}

/// Splits every triangle that is coarse for the target lengths `sigma` at
/// its centroid, once; whether any was split.
fn split_round(patch: &mut RefinedPatch, sigma: &mut Vec<f64>) -> Result<bool> {
    let third = 1.0f32 / 3.0;
    let count = patch.triangles.len();
    let mut split = false;
    for t in 0..count {
        let [a, b, c] = patch.triangles[t];
        let p = [a, b, c].map(|v| patch.points[v as usize]);
        if normal(p[0], p[1], p[2]).is_none() {
            continue;
        }
        let centre: [f64; 3] = std::array::from_fn(|k| (p[0][k] + p[1][k] + p[2][k]) / 3.0);
        let s = (sigma[a as usize] + sigma[b as usize] + sigma[c as usize]) / 3.0;
        let coarse = [a, b, c].iter().zip(p).all(|(&m, q)| {
            let d = SQRT_2 * distance(centre, q);
            d > s && d > sigma[m as usize]
        });
        if !coarse {
            continue;
        }
        let m = u32::try_from(patch.points.len())
            .ok()
            .filter(|&m| m < u32::MAX)
            .ok_or(OpsError::Store(StoreError::OutOfMemory))?;
        push(&mut patch.points, centre)?;
        push(sigma, s)?;
        push(&mut patch.parents, [(a, third), (b, third), (c, third)])?;
        reserve(&mut patch.triangles, 2)?;
        patch.triangles[t] = [a, b, m];
        patch.triangles.extend([[b, c, m], [c, a, m]]);
        split = true;
    }
    Ok(split)
}

/// The undirected key of the local edge `a`-`b`.
fn key(a: u32, b: u32) -> (u32, u32) {
    if a < b { (a, b) } else { (b, a) }
}

/// The angle at `at` between the directions to `a` and `b`.
fn angle(at: [f64; 3], a: [f64; 3], b: [f64; 3]) -> f64 {
    let (u, v) = (sub(a, at), sub(b, at));
    let c = cross(u, v);
    dot(c, c).sqrt().atan2(dot(u, v))
}

/// `t` rotated so that it starts with the directed edge `a -> b`, if it has it.
fn starting_with(t: [u32; 3], a: u32, b: u32) -> Option<[u32; 3]> {
    (0..3)
        .map(|r| [t[r], t[(r + 1) % 3], t[(r + 2) % 3]])
        .find(|r| r[0] == a && r[1] == b)
}

/// Scratch space for the flip passes.
#[derive(Default)]
struct Relax {
    /// Every patch edge with a triangle using it, sorted.
    pairs: Vec<((u32, u32), u32)>,
    /// The triangles changed in this pass; their edges in `pairs` are stale.
    touched: Vec<bool>,
    /// The edges created in this pass.
    added: Vec<(u32, u32)>,
}

impl Relax {
    /// Flips interior patch edges, pass by pass, until a pass flips none.
    fn run(&mut self, patch: &mut RefinedPatch, edges: &EdgeMap) -> Result<()> {
        for _ in 0..MAX_FLIP_PASSES {
            self.pairs.clear();
            reserve(&mut self.pairs, 3 * patch.triangles.len())?;
            for (t, &[a, b, c]) in patch.triangles.iter().enumerate() {
                for (u, v) in [(a, b), (b, c), (c, a)] {
                    self.pairs.push((key(u, v), t as u32));
                }
            }
            self.pairs.sort_unstable();
            self.touched.clear();
            reserve(&mut self.touched, patch.triangles.len())?;
            self.touched.resize(patch.triangles.len(), false);
            self.added.clear();
            let mut flipped = false;
            let mut i = 0;
            while i < self.pairs.len() {
                let run = self.pairs[i..]
                    .iter()
                    .take_while(|p| p.0 == self.pairs[i].0)
                    .count();
                if run == 2 {
                    let (e, t1) = self.pairs[i];
                    let t2 = self.pairs[i + 1].1;
                    if !self.touched[t1 as usize]
                        && !self.touched[t2 as usize]
                        && self.flip(patch, edges, e, t1 as usize, t2 as usize)?
                    {
                        self.touched[t1 as usize] = true;
                        self.touched[t2 as usize] = true;
                        flipped = true;
                    }
                }
                i += run;
            }
            if !flipped {
                break;
            }
        }
        Ok(())
    }

    /// Whether the edge `local` would be an edge of the patch or the mesh
    /// already.
    fn exists(&self, patch: &RefinedPatch, edges: &EdgeMap, local: (u32, u32)) -> bool {
        let at = self.pairs.partition_point(|p| p.0 < local);
        let old = patch.old.len() as u32;
        self.pairs.get(at).is_some_and(|p| p.0 == local)
            || self.added.contains(&local)
            || (local.1 < old
                && edges
                    .kind(patch.old[local.0 as usize], patch.old[local.1 as usize])
                    .is_some())
    }

    /// Flips the edge `e` between triangles `t1` and `t2` when that improves
    /// the circumcircle criterion and is valid; whether it did.
    fn flip(
        &mut self,
        patch: &mut RefinedPatch,
        edges: &EdgeMap,
        e: (u32, u32),
        t1: usize,
        t2: usize,
    ) -> Result<bool> {
        // `t1` is `i -> j -> k`; `t2` must run `j -> i -> l`.
        let Some([i, j, k]) = starting_with(patch.triangles[t1], e.0, e.1)
            .or_else(|| starting_with(patch.triangles[t1], e.1, e.0))
        else {
            return Ok(false);
        };
        let Some([_, _, l]) = starting_with(patch.triangles[t2], j, i) else {
            return Ok(false);
        };
        let p = |v: u32| patch.points[v as usize];
        if k == l || angle(p(k), p(i), p(j)) + angle(p(l), p(j), p(i)) <= PI + ANGLE_EPSILON {
            return Ok(false);
        }
        if self.exists(patch, edges, key(k, l)) {
            return Ok(false);
        }
        let (Some(n1), Some(n2), Some(m1), Some(m2)) = (
            normal(p(i), p(j), p(k)),
            normal(p(j), p(i), p(l)),
            normal(p(i), p(l), p(k)),
            normal(p(l), p(j), p(k)),
        ) else {
            return Ok(false);
        };
        let unit = |n: [f64; 3]| n.map(|x| x / dot(n, n).sqrt());
        let (n1, n2) = (unit(n1), unit(n2));
        let before = [n1[0] + n2[0], n1[1] + n2[1], n1[2] + n2[2]];
        if dot(m1, m2) <= 0.0 || dot(m1, before) <= 0.0 || dot(m2, before) <= 0.0 {
            return Ok(false);
        }
        push(&mut self.added, key(k, l))?;
        patch.triangles[t1] = [i, l, k];
        patch.triangles[t2] = [l, j, k];
        Ok(true)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::boundary_loops;
    use crate::fill_dp::triangulate_loop;
    use crate::test_util::*;

    /// The refined patch of the shortest loop of `mesh` (the hole, not a
    /// grid's border), with that loop.
    fn refined(mesh: &Mesh) -> (Vec<VertexId>, RefinedPatch) {
        let loops = boundary_loops(mesh).unwrap();
        let l = loops.iter().min_by_key(|l| l.vertices.len()).unwrap();
        let edges = EdgeMap::build(mesh).unwrap();
        let fans = VertexFans::build(mesh).unwrap();
        let tris = triangulate_loop(mesh, &l.vertices, &edges, 200)
            .unwrap()
            .unwrap();
        let patch = refine(mesh, &fans, &edges, &l.vertices, &tris).unwrap();
        (l.vertices.clone(), patch)
    }

    #[test]
    fn parents_valid() {
        let mesh = punch_holes(&uv_sphere(32, 64, 1.0), &[[0.6, 0.0, 0.8]], 0.4);
        let loops = boundary_loops(&mesh).unwrap();
        assert_eq!(loops.len(), 1);
        let (l, patch) = refined(&mesh);
        let (old, new) = (patch.old.len(), patch.new_vertices());
        assert!(new > 0);
        assert_eq!(patch.parents.len(), new);
        assert_eq!(patch.triangles.len(), l.len() - 2 + 2 * new);
        for (k, parents) in patch.parents.iter().enumerate() {
            let total: f32 = parents.iter().map(|p| p.1).sum();
            assert!((total - 1.0).abs() < 1e-6, "{total}");
            // Old, or new and created earlier.
            assert!(parents.iter().all(|p| (p.0 as usize) < old + k));
            // A centroid split: the new vertex starts at its parents' mean.
            let mean: [f64; 3] = std::array::from_fn(|d| {
                parents
                    .iter()
                    .map(|p| patch.points[p.0 as usize][d])
                    .sum::<f64>()
                    / 3.0
            });
            assert_eq!(patch.points[old + k], mean);
        }
        assert!(patch.old.windows(2).all(|w| w[0] < w[1]));
        assert!(patch.old.iter().all(|&v| mesh.is_vertex_alive(v)));
        let len = patch.points.len() as u32;
        assert!(patch.triangles.iter().flatten().all(|&v| v < len));
    }

    #[test]
    fn flips_keep_the_patch_manifold_and_unfolded() {
        // A flat L-shaped hole: 6 x 3 cells plus 3 x 3 on top of one end.
        let cells: Vec<(u32, u32)> = (2..11)
            .flat_map(|y| (2..8).map(move |x| (x, y)))
            .filter(|&(x, y)| y < 5 || x < 5)
            .collect();
        let mesh = grid_without(12, &cells);
        let (l, patch) = refined(&mesh);
        assert!(patch.new_vertices() > 0);
        // Every interior patch edge is in two triangles, every loop edge in one.
        let mut keys: Vec<(u32, u32)> = patch
            .triangles
            .iter()
            .flat_map(|t| [(t[0], t[1]), (t[1], t[2]), (t[2], t[0])])
            .map(|(a, b)| key(a, b))
            .collect();
        keys.sort_unstable();
        let runs: Vec<usize> = keys.chunk_by(|a, b| a == b).map(<[_]>::len).collect();
        assert_eq!(runs.iter().filter(|&&r| r == 1).count(), l.len());
        assert!(runs.iter().all(|&r| r <= 2));
        // Every triangle faces +z like the grid.
        for t in &patch.triangles {
            let [a, b, c] = t.map(|v| patch.points[v as usize]);
            assert!(cross(sub(b, a), sub(c, a))[2] > 0.0);
        }
    }
}
