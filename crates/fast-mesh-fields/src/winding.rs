//! Fast generalized winding numbers (Barill, Dickson, Schmidt, Levin and
//! Jacobson, "Fast winding numbers for soups and clouds", 2018), order 1.
//!
//! Every BVH node stores the area-weighted normal sum `N = Σ ½·(e1 × e2)` of
//! its triangles, their area-weighted centroid `c` and a radius `r` around
//! `c` that encloses them. Seen from a point `p` with `|c − p| > beta·r`, the
//! node's solid angle is approximated by the dipole term
//! `N · (c − p) / |c − p|³`; nearer leaves sum their triangles' exact solid
//! angles (Van Oosterom and Strackee, 1983). The winding number is the total
//! over 4π.
//!
//! **Accuracy.** The dipole term over-estimates a near-flat patch seen
//! face-on by about `¾·(a/d)²` relatively (a disk of radius `a` at distance
//! `d`), an error the order-2 term cannot correct because it vanishes on flat
//! patches. Seen from inside a closed mesh, these errors share a sign and add
//! up over the whole surface, so the total error falls only like `1/beta²`.
//! Measured on the test meshes (points beyond 1% of the bounding-box diagonal
//! from the surface), the worst error is 4.5e-2 at `beta = 2`, 2.1e-3 at 6
//! and 5.8e-4 at 8 (all on the torus), hence [`BETA`] = 8.

use std::f64::consts::PI;

use fast_mesh_bvh::{Aabb, Bvh, NodeId, NodeKind};
use fast_mesh_store::{Mesh, TriangleId};

use crate::geom::{add, cross, dot, norm, scale, sub, widen};
use crate::sign::collect_span;
use crate::{FieldsError, Result};

/// The accuracy parameter: a node is far when its centroid is more than
/// `BETA` times its radius away from the query point. Chosen so that the
/// error stays below 1e-3 away from the surface (see the module docs).
const BETA: f64 = 8.0;

/// The accuracy parameter for sign-only queries ([`Moments::sign_winding`],
/// as used by [`SignData::is_inside`](crate::SignData::is_inside)): a node
/// is far when its centroid is more than `SIGN_BETA` times its radius away.
/// Lower than [`BETA`], so fewer triangles are summed exactly; the error
/// stays well below the 0.5 inside/outside threshold away from the surface
/// on the closed test meshes.
const SIGN_BETA: f64 = 4.0;

/// The order-1 moments of one BVH node.
#[derive(Debug, Clone, Copy, Default)]
struct Moment {
    /// Area-weighted normal sum: `Σ ½·(b − a) × (c − a)`.
    normal: [f64; 3],
    /// Total area.
    area: f64,
    /// Area-weighted centroid (the box center when the area is zero).
    centroid: [f64; 3],
    /// Distance from `centroid` that encloses every triangle of the node.
    radius: f64,
}

/// Per-node order-1 dipole moments of a BVH, indexed by `NodeId::index`.
/// Empty when not built: then every node counts as near.
#[derive(Debug, Clone, Default)]
pub(crate) struct Moments(Vec<Moment>);

fn corners(mesh: &Mesh, t: TriangleId) -> [[f64; 3]; 3] {
    mesh.triangles()[t as usize].map(|v| widen(mesh.positions()[v as usize]))
}

fn center(bounds: &Aabb) -> [f64; 3] {
    std::array::from_fn(|k| (f64::from(bounds.min[k]) + f64::from(bounds.max[k])) / 2.0)
}

/// The distance from `c` to the farthest corner of `bounds`.
fn farthest_corner(c: [f64; 3], bounds: &Aabb) -> f64 {
    let d: [f64; 3] = std::array::from_fn(|k| {
        (f64::from(bounds.min[k]) - c[k])
            .abs()
            .max((f64::from(bounds.max[k]) - c[k]).abs())
    });
    norm(d)
}

fn leaf_moment(mesh: &Mesh, triangles: &[TriangleId], bounds: &Aabb) -> Moment {
    let (mut normal, mut area, mut weighted) = ([0.0; 3], 0.0, [0.0; 3]);
    for &t in triangles {
        let [a, b, c] = corners(mesh, t);
        let n = scale(cross(sub(b, a), sub(c, a)), 0.5);
        let w = norm(n);
        normal = add(normal, n);
        area += w;
        weighted = add(weighted, scale(add(a, add(b, c)), w / 3.0));
    }
    let centroid = if area > 0.0 {
        scale(weighted, 1.0 / area)
    } else {
        center(bounds)
    };
    let radius = triangles
        .iter()
        .flat_map(|&t| corners(mesh, t))
        .map(|x| norm(sub(x, centroid)))
        .fold(0.0, f64::max);
    Moment {
        normal,
        area,
        centroid,
        radius,
    }
}

fn inner_moment(l: &Moment, r: &Moment, bounds: &Aabb) -> Moment {
    let area = l.area + r.area;
    let centroid = if area > 0.0 {
        scale(
            add(scale(l.centroid, l.area), scale(r.centroid, r.area)),
            1.0 / area,
        )
    } else {
        center(bounds)
    };
    let from_children = (norm(sub(l.centroid, centroid)) + l.radius)
        .max(norm(sub(r.centroid, centroid)) + r.radius);
    Moment {
        normal: add(l.normal, r.normal),
        area,
        centroid,
        radius: from_children.min(farthest_corner(centroid, bounds)),
    }
}

/// The solid angle of triangle `[a, b, c]` seen from the origin
/// (Van Oosterom–Strackee): positive when the triangle faces away.
fn solid_angle([a, b, c]: [[f64; 3]; 3]) -> f64 {
    let (la, lb, lc) = (norm(a), norm(b), norm(c));
    let num = dot(a, cross(b, c));
    let den = la * lb * lc + dot(a, b) * lc + dot(b, c) * la + dot(c, a) * lb;
    2.0 * num.atan2(den)
}

impl Moments {
    /// Every number of every node, as bits, for comparing builds.
    #[cfg(test)]
    pub(crate) fn bits(&self) -> impl Iterator<Item = u64> + '_ {
        self.0.iter().flat_map(|m| {
            m.normal
                .into_iter()
                .chain([m.area])
                .chain(m.centroid)
                .chain([m.radius])
                .map(f64::to_bits)
        })
    }

    /// The winding number of `mesh` at `p`, and how many triangles were
    /// summed exactly. `mesh` and `bvh` must be those the moments were built
    /// for. Nothing is allocated.
    pub(crate) fn winding(&self, mesh: &Mesh, bvh: &Bvh, p: [f32; 3]) -> (f64, usize) {
        self.winding_with_beta(mesh, bvh, p, BETA)
    }

    /// As [`Moments::winding`], with the sign-only [`SIGN_BETA`]: fewer
    /// triangles summed, same inside/outside away from the surface.
    pub(crate) fn sign_winding(&self, mesh: &Mesh, bvh: &Bvh, p: [f32; 3]) -> (f64, usize) {
        self.winding_with_beta(mesh, bvh, p, SIGN_BETA)
    }

    fn winding_with_beta(&self, mesh: &Mesh, bvh: &Bvh, p: [f32; 3], beta: f64) -> (f64, usize) {
        let q = widen(p);
        let (mut total, mut summed) = (0.0, 0);
        bvh.traverse(|id, view| {
            if let Some(m) = self.0.get(id.index()) {
                let d = sub(m.centroid, q);
                let dist = norm(d);
                if dist > beta * m.radius {
                    total += dot(m.normal, d) / (dist * dist * dist);
                    return false;
                }
            }
            if let NodeKind::Leaf { triangles } = view.kind {
                for &t in triangles {
                    total += solid_angle(corners(mesh, t).map(|x| sub(x, q)));
                }
                summed += triangles.len();
            }
            true
        });
        (total / (4.0 * PI), summed)
    }
}

/// A build of the [`Moments`] of every node of a BVH, bottom-up. The leaves'
/// moments come first (in parallel chunks on large trees, each leaf on
/// its own). Then, visiting the leaves in pre-order, which finishes a left
/// subtree before it enters the right one: when a leaf that is a right
/// child is done, its parent can be combined, and so on up while the
/// finished node is a right child.
///
/// The build runs in steps: first the node slots are initialised, a slice
/// at a time; then the leaves are listed from a pre-order frontier and the
/// next leaves' moments computed, up to the budget per step, combining
/// every inner node they finish. Each leaf is computed on its own and the
/// combines run in the same order whatever the steps, so the moments are
/// the same, bit for bit.
#[derive(Debug, Clone)]
pub(crate) struct MomentsBuild {
    nodes: Vec<Moment>,
    /// Node slots initialised to the default moment; leaves are listed and
    /// computed only once every slot is.
    filled: usize,
    leaves: Vec<NodeId>,
    /// The pre-order frontier: the nodes whose listing is still pending
    /// (top last), left children on top of their right siblings.
    stack: Vec<NodeId>,
    /// The first leaf whose moment is not computed yet.
    next: usize,
}

impl MomentsBuild {
    /// Reserves the moments of `bvh`'s nodes; the slots are initialised,
    /// and the leaves listed, by the steps.
    pub(crate) fn new(bvh: &Bvh) -> Result<MomentsBuild> {
        let mut nodes = Vec::new();
        nodes
            .try_reserve_exact(bvh.node_slots())
            .map_err(|_| FieldsError::OutOfMemory)?;
        // The leaves are fewer than the node slots, so listing never
        // reallocates.
        let mut leaves = Vec::new();
        leaves
            .try_reserve_exact(bvh.node_slots())
            .map_err(|_| FieldsError::OutOfMemory)?;
        let mut stack = Vec::new();
        if let Some(root) = bvh.root() {
            stack.try_reserve(1).map_err(|_| FieldsError::OutOfMemory)?;
            stack.push(root);
        }
        Ok(MomentsBuild {
            nodes,
            filled: 0,
            leaves,
            stack,
            next: 0,
        })
    }

    /// Leaves computed so far.
    pub(crate) fn computed(&self) -> usize {
        self.next
    }

    /// Initialised node slots plus computed leaves. Every step advances it
    /// while the build is not done, so the sign build charges its budget
    /// against it.
    pub(crate) fn progress(&self) -> usize {
        self.filled + self.computed()
    }

    /// Initialises up to `budget` more node slots (at least one), or, once
    /// every slot is, lists up to `budget` more leaves and computes up to
    /// `budget` more leaves (at least one) of `mesh` and `bvh`, the BVH it
    /// was listed from, combining the inner nodes they finish. Each node is
    /// visited once across the whole build, so the listing costs no more
    /// than the moments. Returns whether every leaf is done.
    pub(crate) fn step(&mut self, mesh: &Mesh, bvh: &Bvh, budget: usize) -> Result<bool> {
        let want = budget.max(1);
        if self.filled < bvh.node_slots() {
            let end = self.filled.saturating_add(want).min(bvh.node_slots());
            // Within the reserved capacity: no allocation.
            self.nodes.resize(end, Moment::default());
            self.filled = end;
            return Ok(false);
        }
        while self.leaves.len() - self.next < want {
            let Some(id) = self.stack.pop() else {
                break;
            };
            match bvh.node(id).kind {
                NodeKind::Leaf { .. } => {
                    self.leaves
                        .try_reserve(1)
                        .map_err(|_| FieldsError::OutOfMemory)?;
                    self.leaves.push(id);
                }
                NodeKind::Inner { left, right } => {
                    self.stack
                        .try_reserve(2)
                        .map_err(|_| FieldsError::OutOfMemory)?;
                    self.stack.push(right);
                    self.stack.push(left);
                }
            }
        }
        let from = self.next;
        let to = self.leaves.len().min(from.saturating_add(want));
        let leaves = &self.leaves;
        let moments = collect_span(from..to, |k| {
            let view = bvh.node(leaves[k]);
            match view.kind {
                NodeKind::Leaf { triangles } => leaf_moment(mesh, triangles, &view.bounds),
                NodeKind::Inner { .. } => unreachable!("collected as a leaf"),
            }
        })?;
        let nodes = &mut self.nodes;
        for (&id, moment) in leaves[from..to].iter().zip(moments) {
            nodes[id.index()] = moment;
            let mut child = id;
            while let Some(parent) = bvh.parent(child) {
                let pv = bvh.node(parent);
                let NodeKind::Inner { left, right } = pv.kind else {
                    break;
                };
                if right != child {
                    break;
                }
                nodes[parent.index()] =
                    inner_moment(&nodes[left.index()], &nodes[right.index()], &pv.bounds);
                child = parent;
            }
        }
        self.next = to;
        // The listing ends when the frontier does; only then is every leaf
        // listed, and `to == leaves.len()` means every leaf is computed.
        Ok(self.stack.is_empty() && to == self.leaves.len())
    }

    /// The moments, once every leaf is done.
    pub(crate) fn finish(self) -> Moments {
        Moments(self.nodes)
    }
}

#[cfg(test)]
mod tests {
    use fast_mesh_store::Mesh;

    use super::MomentsBuild;
    use crate::test_util::*;
    use crate::{FieldsError, SignData, SignMode, signed_distances};

    fn open_cube() -> Mesh {
        let (p, mut t) = cube_buffers();
        t.drain(2..4);
        mesh(&p, &t)
    }

    #[test]
    fn moments_build_lists_leaves_incrementally() {
        let mesh = terrain(8);
        let bvh = build_bvh(&mesh);
        let mut build = MomentsBuild::new(&bvh).unwrap();
        // `new` only reserves: nothing is listed or computed yet.
        assert!(build.leaves.is_empty());
        assert_eq!(build.computed(), 0);
        // One leaf per step at budget 1, listed just ahead of computed.
        let mut steps = 0;
        while !build.step(&mesh, &bvh, 1).unwrap() {
            assert!(build.leaves.len() - build.computed() <= 1);
            steps += 1;
        }
        assert!(steps > 1);
        let stepped = build.finish();
        // The same moments as computing everything at once, bit for bit.
        let mut whole = MomentsBuild::new(&bvh).unwrap();
        while !whole.step(&mesh, &bvh, usize::MAX).unwrap() {}
        assert!(stepped.bits().eq(whole.finish().bits()));
        // An empty tree is done at once.
        let empty = Mesh::from_buffers(&[], &[]).unwrap();
        let empty_bvh = build_bvh(&empty);
        let mut build = MomentsBuild::new(&empty_bvh).unwrap();
        assert!(build.step(&empty, &empty_bvh, 1).unwrap());
    }

    #[test]
    fn sign_winding_agrees_and_sums_less() {
        // The sign-only beta sums fewer triangles than the accurate one,
        // while giving the same inside/outside on the closed test meshes
        // away from the surface.
        let mut rng = Lcg(37);
        for (name, mut mesh) in [
            ("cube", cube()),
            ("icosphere", icosphere()),
            ("torus", torus()),
            ("tetrahedron", tetrahedron()),
            ("star prism", star_prism()),
        ] {
            let bvh = build_bvh(&mesh);
            let sign = SignData::build(&mut mesh, &bvh, SignMode::Winding).unwrap();
            let diag = diagonal(&mesh);
            let near = diag as f32 / 10.0;
            let (mut full_tri, mut sign_tri, mut checked) = (0usize, 0usize, 0usize);
            for p in sample_points(&mesh, &mut rng, 500, near) {
                if brute_distance(&mesh, p) <= 0.01 * diag {
                    continue;
                }
                let (w_full, n_full) = sign.moments.winding(&mesh, &bvh, p);
                let (w_sign, n_sign) = sign.moments.sign_winding(&mesh, &bvh, p);
                assert_eq!(
                    w_sign >= 0.5,
                    w_full >= 0.5,
                    "{name} at {p:?}: sign {w_sign} vs full {w_full}"
                );
                full_tri += n_full;
                sign_tri += n_sign;
                checked += 1;
            }
            assert!(checked > 400, "{name}: only {checked} checked");
            assert!(
                sign_tri <= full_tri,
                "{name}: sign summed {sign_tri}, full {full_tri}"
            );
        }
    }

    #[test]
    fn fast_winding_matches_exact() {
        let mut rng = Lcg(17);
        let meshes = [
            ("cube", cube()),
            ("icosphere", icosphere()),
            ("torus", torus()),
            ("tetrahedron", tetrahedron()),
            ("star prism", star_prism()),
            ("open cube", open_cube()),
        ];
        for (name, mesh) in meshes {
            let (worst, _) = compare_with_exact(name, mesh, &mut rng, 1e-3);
            println!("{name}: worst error {worst:.2e}");
        }
    }

    /// Compares the fast winding number with the exact one at 2000 sample
    /// points farther than 1% of the diagonal from the surface, within `tol`.
    /// Returns the worst error and the average number of triangles summed
    /// exactly per point.
    fn compare_with_exact(name: &str, mut mesh: Mesh, rng: &mut Lcg, tol: f64) -> (f64, f64) {
        let bvh = build_bvh(&mesh);
        let sign = SignData::build(&mut mesh, &bvh, SignMode::Winding).unwrap();
        assert_eq!(sign.mode(), SignMode::Winding, "{name}");
        let diag = diagonal(&mesh);
        let near = diag as f32 / 10.0;
        let (mut checked, mut summed, mut worst) = (0, 0, 0.0f64);
        for p in sample_points(&mesh, rng, 2000, near) {
            if brute_distance(&mesh, p) <= 0.01 * diag {
                continue;
            }
            let fast = sign.winding_number(&mesh, &bvh, p);
            let exact = exact_winding(&mesh, p);
            let err = (fast - exact).abs();
            assert!(err <= tol, "{name} at {p:?}: fast {fast} vs exact {exact}");
            worst = worst.max(err);
            checked += 1;
            summed += sign.moments.winding(&mesh, &bvh, p).1;
        }
        assert!(checked > 1000, "{name}: only {checked} points checked");
        (worst, summed as f64 / checked as f64)
    }

    #[test]
    fn fast_winding_skips_far_triangles() {
        // On a larger mesh most triangles are covered by dipole terms. A wide
        // flat sheet is the worst case for the order-1 expansion (seen from
        // above, every patch's error has the same sign), so it gets a looser
        // bound than the 1e-3 of the closed meshes.
        let mesh = terrain(32);
        let count = mesh.triangle_count() as f64;
        let (worst, average) = compare_with_exact("terrain", mesh, &mut Lcg(29), 2e-3);
        println!("terrain: worst error {worst:.2e}, {average:.1} of {count} triangles per point");
        assert!(
            average < 0.25 * count,
            "{average} triangles summed per point"
        );
    }

    #[test]
    fn winding_sign_on_closed_meshes() {
        let cases = [
            ("cube", cube(), [0.1, 0.2, 0.3], [3.0, 0.0, 0.0]),
            ("icosphere", icosphere(), [0.1, -0.2, 0.3], [0.0, 2.0, 0.0]),
            ("torus", torus(), [1.0, 0.0, 0.1], [0.0, 0.0, 0.0]),
            (
                "tetrahedron",
                tetrahedron(),
                [0.1, 0.1, 0.0],
                [2.0, 2.0, 2.0],
            ),
            ("star prism", star_prism(), [0.5, 0.0, 0.2], [0.5, 0.5, 0.0]),
        ];
        let mut rng = Lcg(19);
        for (name, mut mesh, inside, outside) in cases {
            let bvh = build_bvh(&mesh);
            let sign = SignData::build(&mut mesh, &bvh, SignMode::Winding).unwrap();
            let status = sign.status();
            assert_eq!(status.mode, SignMode::Winding, "{name}");
            assert!(status.closed && status.consistent_orientation, "{name}");
            let w_in = sign.winding_number(&mesh, &bvh, inside);
            let w_out = sign.winding_number(&mesh, &bvh, outside);
            assert!((w_in - 1.0).abs() < 1e-3, "{name}: inside {w_in}");
            assert!(w_out.abs() < 1e-3, "{name}: outside {w_out}");

            // Signed distances take their sign from the winding number.
            let near = diagonal(&mesh) as f32 / 20.0;
            let mut points = sample_points(&mesh, &mut rng, 1000, near);
            points.extend([inside, outside]);
            let mut out = vec![f32::NAN; points.len()];
            signed_distances(&mesh, &bvh, &sign, &points, f32::INFINITY, &mut out).unwrap();
            let checked = check_against_exact(&mesh, &points, &out);
            assert!(checked > 900, "{name}: only {checked} signs checked");
            assert!(
                out[out.len() - 2] < 0.0 && out[out.len() - 1] > 0.0,
                "{name}"
            );
        }
    }

    #[test]
    fn auto_falls_back_on_inconsistent_orientation() {
        let (p, mut t) = cube_buffers();
        t[0].swap(1, 2);
        let mut mesh = mesh(&p, &t);
        let bvh = build_bvh(&mesh);
        let sign = SignData::build(&mut mesh, &bvh, SignMode::Auto).unwrap();
        let status = sign.status();
        assert_eq!(status.mode, SignMode::Winding);
        assert!(status.closed);
        assert!(!status.consistent_orientation);

        // Signs follow the exact winding number everywhere...
        let mut rng = Lcg(23);
        let points = sample_points(&mesh, &mut rng, 2000, 0.2);
        let mut out = vec![f32::NAN; points.len()];
        signed_distances(&mesh, &bvh, &sign, &points, f32::INFINITY, &mut out).unwrap();
        assert!(check_against_exact(&mesh, &points, &out) > 1900);

        // ...which is right about the box away from the flipped triangle,
        // where pseudonormals would be wrong.
        let probes = [
            ([0.0, 0.0, 0.0], true),
            ([-0.5, 0.5, -0.5], true),
            ([0.5, -0.5, 0.5], true),
            ([0.0, 0.0, -1.5], false),
            ([2.0, 1.0, 0.0], false),
        ];
        let points = probes.map(|(p, _)| p);
        let mut out = [f32::NAN; 5];
        signed_distances(&mesh, &bvh, &sign, &points, f32::INFINITY, &mut out).unwrap();
        for ((p, inside), value) in probes.iter().zip(out) {
            assert_eq!(value < 0.0, *inside, "at {p:?}: {value}");
        }
        let pseudo = SignData::build(&mut mesh, &bvh, SignMode::Pseudonormal).unwrap();
        signed_distances(&mesh, &bvh, &pseudo, &points, f32::INFINITY, &mut out).unwrap();
        assert!(
            out[1] > 0.0,
            "pseudonormals are wrong next to the flipped face"
        );
    }

    #[test]
    fn moments_rebuild_when_bvh_changes() {
        let mut mesh = icosphere();
        let mut bvh = build_bvh(&mesh);
        let sign = SignData::build(&mut mesh, &bvh, SignMode::Winding).unwrap();
        let p = [0.0, 0.0, 1.3];
        assert!(sign.winding_number(&mesh, &bvh, p).abs() < 1e-3);

        // Stretch the sphere along z so that p ends up inside.
        let moved: Vec<[f32; 3]> = mesh
            .positions()
            .iter()
            .map(|q| [q[0], q[1], q[2] * 2.0])
            .collect();
        let ids: Vec<u32> = (0..moved.len() as u32).collect();
        let mut tx = mesh.begin_edit();
        tx.set_positions(&ids, &moved).unwrap();
        let patch = tx.commit().unwrap();
        bvh.update(&mut mesh, patch.changes()).unwrap();
        assert!(!sign.is_current(&mesh, &bvh));
        assert!(sign.winding_number(&mesh, &bvh, p).is_nan());
        let mut out = [0.0f32];
        assert!(matches!(
            signed_distances(&mesh, &bvh, &sign, &[p], f32::INFINITY, &mut out),
            Err(FieldsError::InvalidInput(_))
        ));

        let sign = SignData::build(&mut mesh, &bvh, SignMode::Winding).unwrap();
        let w = sign.winding_number(&mesh, &bvh, p);
        assert!((w - exact_winding(&mesh, p)).abs() < 1e-3 && w > 0.5, "{w}");

        // A restructured BVH has new nodes: the moments follow.
        bvh.rebuild(&mesh).unwrap();
        assert!(sign.winding_number(&mesh, &bvh, p).is_nan());
        let sign = SignData::build(&mut mesh, &bvh, SignMode::Winding).unwrap();
        let w = sign.winding_number(&mesh, &bvh, p);
        assert!((w - exact_winding(&mesh, p)).abs() < 1e-3, "{w}");
        signed_distances(&mesh, &bvh, &sign, &[p], f32::INFINITY, &mut out).unwrap();
        assert!(out[0] < 0.0);

        // Pseudonormal data has no moments, but still answers exactly.
        let pseudo = SignData::build(&mut mesh, &bvh, SignMode::Pseudonormal).unwrap();
        let w = pseudo.winding_number(&mesh, &bvh, p);
        assert!((w - exact_winding(&mesh, p)).abs() < 1e-9, "{w}");
    }
}
