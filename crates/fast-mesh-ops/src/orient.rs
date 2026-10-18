//! Making triangle winding consistent.
//!
//! [`orient()`] reads its whole plan from `tx.mesh()`: the patches with each
//! triangle's parity, each patch's majority, then the outward pass over the
//! closed components. It writes every flip in one `set_triangles` call.

use fast_mesh_store::{Mesh, Transaction, TriangleId, VertexId};

use crate::topology::{
    EdgeMap, components, cross, dot, point, proper_triangle, sub, triangle_area,
};
use crate::{Result, filled, push, reserve};

const NONE: u32 = u32::MAX;

/// What [`orient()`] did.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct OrientReport {
    /// Triangles reversed.
    pub flipped_triangles: usize,
    /// Patches found: triangles connected through interior edges.
    pub patches: usize,
    /// Patches that admit no consistent winding, left unchanged.
    pub non_orientable_patches: usize,
    /// Closed components turned to face outward.
    pub flipped_outward_components: usize,
}

/// Triangles connected through interior edges, as found by the search.
struct Patch {
    /// The patch's triangles are `order[start..end]`, its smallest id first.
    start: usize,
    end: usize,
    /// `false` when some triangle would need to be both flipped and not.
    orientable: bool,
    /// The total area of the triangles with parity `false`, then `true`.
    area: [f64; 2],
}

/// Makes triangle winding consistent, reversing triangles by swapping their
/// last two vertices.
///
/// - Patches grow from the smallest unvisited triangle id across interior
///   edges only (two triangles); boundary and non-manifold edges stop them.
///   A neighbour that runs along the shared edge in the same direction gets
///   the opposite parity.
/// - A patch that reaches a triangle with both parities is non-orientable
///   (a Möbius-like strip): it is counted and none of its triangles change.
/// - Every orientable patch takes the winding of the larger area on its own,
///   so patches meeting only at non-manifold edges are not aligned with each
///   other. On a tie the winding of the patch's smallest triangle id wins.
/// - Then each closed component (no boundary and no non-manifold edge) whose
///   signed volume, with the planned flips, is negative is reversed whole,
///   so that it faces outward.
///
/// Triangles with a repeated vertex id are skipped: never flipped, never
/// part of a patch. A consistent mesh is left without a write.
///
/// # Errors
/// [`OpsError::Store`] when memory runs out; nothing is written then.
///
/// [`OpsError::Store`]: crate::OpsError::Store
pub fn orient(tx: &mut Transaction) -> Result<OrientReport> {
    let mesh = tx.mesh();
    let edges = EdgeMap::build(mesh)?;
    let slots = mesh.triangle_slots();
    let mut patch_of = filled(slots, NONE)?;
    let mut parity = filled(slots, false)?;
    // Every proper triangle in visiting order; each patch is a run of it.
    let mut order: Vec<TriangleId> = Vec::new();
    reserve(&mut order, mesh.triangle_count())?;
    let mut patches: Vec<Patch> = Vec::new();
    for seed in 0..slots as TriangleId {
        if patch_of[seed as usize] != NONE || proper_triangle(mesh, seed).is_none() {
            continue;
        }
        let index = patches.len() as u32;
        let start = order.len();
        let mut orientable = true;
        patch_of[seed as usize] = index;
        order.push(seed);
        let mut head = start;
        while head < order.len() {
            let u = order[head];
            head += 1;
            let [x, y, z] = mesh.triangles()[u as usize];
            for (a, b) in [(x, y), (y, z), (z, x)] {
                let &[s, t] = edges.triangles(a, b) else {
                    continue;
                };
                let w = if s == u { t } else { s };
                // `u` runs from `a` to `b`; so does `w` exactly when the two
                // disagree, and then `w` needs the opposite parity.
                let want = parity[u as usize] ^ runs_forward(mesh, w, a, b);
                if patch_of[w as usize] == NONE {
                    patch_of[w as usize] = index;
                    parity[w as usize] = want;
                    order.push(w);
                } else if parity[w as usize] != want {
                    orientable = false;
                }
            }
        }
        let mut area = [0.0; 2];
        for &t in &order[start..] {
            let tri = mesh.triangles()[t as usize];
            area[usize::from(parity[t as usize])] += triangle_area(mesh, tri);
        }
        push(
            &mut patches,
            Patch {
                start,
                end: order.len(),
                orientable,
                area,
            },
        )?;
    }

    let mut flip = filled(slots, false)?;
    for p in patches.iter().filter(|p| p.orientable) {
        // The seed has parity `false`, so `false` keeps the tie.
        let odd_wins = p.area[1] > p.area[0];
        for &t in &order[p.start..p.end] {
            flip[t as usize] = parity[t as usize] != odd_wins;
        }
    }

    let mut report = OrientReport {
        patches: patches.len(),
        non_orientable_patches: patches.iter().filter(|p| !p.orientable).count(),
        ..OrientReport::default()
    };
    for c in components(mesh, &edges)? {
        // A closed component is a single patch, but a non-orientable one
        // (a Klein bottle) has no outside.
        let orientable = c
            .triangles
            .iter()
            .all(|&t| patches[patch_of[t as usize] as usize].orientable);
        if c.closed && orientable && signed_volume(mesh, &c.triangles, &flip) < 0.0 {
            for &t in &c.triangles {
                flip[t as usize] = !flip[t as usize];
            }
            report.flipped_outward_components += 1;
        }
    }

    let count = flip.iter().filter(|&&f| f).count();
    let (mut ids, mut flipped) = (Vec::new(), Vec::new());
    reserve(&mut ids, count)?;
    reserve(&mut flipped, count)?;
    for t in 0..slots as TriangleId {
        if flip[t as usize] {
            let [a, b, c] = mesh.triangles()[t as usize];
            ids.push(t);
            flipped.push([a, c, b]);
        }
    }
    report.flipped_triangles = count;
    tx.set_triangles(&ids, &flipped)?;
    Ok(report)
}

/// Whether triangle `t` runs along its edge `a`-`b` from `a` to `b`.
fn runs_forward(mesh: &Mesh, t: TriangleId, a: VertexId, b: VertexId) -> bool {
    let [x, y, z] = mesh.triangles()[t as usize];
    (x, y) == (a, b) || (y, z) == (a, b) || (z, x) == (a, b)
}

/// Six times the signed volume enclosed by `triangles` once those marked in
/// `flip` are reversed, taken about the first triangle's first vertex so
/// that large coordinates do not swamp it.
fn signed_volume(mesh: &Mesh, triangles: &[TriangleId], flip: &[bool]) -> f64 {
    let Some(&first) = triangles.first() else {
        return 0.0;
    };
    let o = point(mesh, mesh.triangles()[first as usize][0]);
    let mut sum = 0.0;
    for &t in triangles {
        let [a, mut b, mut c] = mesh.triangles()[t as usize];
        if flip[t as usize] {
            std::mem::swap(&mut b, &mut c);
        }
        let [pa, pb, pc] = [a, b, c].map(|v| sub(point(mesh, v), o));
        sum += dot(pa, cross(pb, pc));
    }
    sum
}

#[cfg(test)]
mod tests {
    use fast_mesh_store::{Mesh, TriangleId};

    use super::*;
    use crate::analyze::{AnalyzeOptions, analyze};
    use crate::test_util::*;

    fn conflicts(mesh: &Mesh) -> usize {
        analyze(mesh, &AnalyzeOptions::default())
            .unwrap()
            .orientation_conflicts
    }

    /// The corners of triangle `t`'s vertices in `f64`.
    fn corners(mesh: &Mesh, t: TriangleId) -> [[f64; 3]; 3] {
        mesh.triangle(t)
            .unwrap()
            .map(|v| mesh.positions()[v as usize].map(f64::from))
    }

    /// The dot product of triangle `t`'s (unnormalised) normal and its
    /// centroid: positive when it faces away from the origin.
    fn facing(mesh: &Mesh, t: TriangleId) -> f64 {
        let [a, b, c] = corners(mesh, t);
        let (u, v) = (
            std::array::from_fn::<f64, 3, _>(|k| b[k] - a[k]),
            std::array::from_fn::<f64, 3, _>(|k| c[k] - a[k]),
        );
        let n = [
            u[1] * v[2] - u[2] * v[1],
            u[2] * v[0] - u[0] * v[2],
            u[0] * v[1] - u[1] * v[0],
        ];
        (0..3).map(|k| n[k] * (a[k] + b[k] + c[k])).sum()
    }

    /// The mesh's signed volume about its first vertex.
    fn volume(mesh: &Mesh) -> f64 {
        let o = mesh.positions()[0].map(f64::from);
        let mut sum = 0.0;
        for t in 0..mesh.triangle_slots() as TriangleId {
            if mesh.triangle(t).is_none() {
                continue;
            }
            let [a, b, c] =
                corners(mesh, t).map(|p| std::array::from_fn::<f64, 3, _>(|k| p[k] - o[k]));
            sum += a[0] * (b[1] * c[2] - b[2] * c[1]) - a[1] * (b[0] * c[2] - b[2] * c[0])
                + a[2] * (b[0] * c[1] - b[1] * c[0]);
        }
        sum / 6.0
    }

    /// `count` distinct ids below `n` from a fixed linear congruential sequence, ascending.
    fn scattered(n: u32, count: usize) -> Vec<TriangleId> {
        let mut out = Vec::new();
        let mut x: u64 = 12345;
        while out.len() < count {
            x = x
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let t = ((x >> 33) % u64::from(n)) as TriangleId;
            if !out.contains(&t) {
                out.push(t);
            }
        }
        out.sort_unstable();
        out
    }

    #[test]
    fn hemisphere_random_flips_fixed() {
        let clean = hemisphere(8, 24, 1.0);
        assert_eq!(clean.triangle_count(), 360);
        let random = scattered(360, 4);
        let patch: Vec<TriangleId> = (0..30).collect();
        for flips in [random, patch] {
            let mut mesh = flip(&clean, &flips);
            assert!(conflicts(&mesh) > 0);
            let (r, _) = edit(&mut mesh, orient);
            assert_eq!(
                r,
                OrientReport {
                    flipped_triangles: flips.len(),
                    patches: 1,
                    non_orientable_patches: 0,
                    flipped_outward_components: 0,
                }
            );
            assert_eq!(conflicts(&mesh), 0);
            for t in 0..360 {
                assert!(facing(&mesh, t) > 0.0, "triangle {t} faces inward");
            }
            assert_eq!(parts(&mesh), parts(&clean));
            let (_, empty) = edit(&mut mesh, orient);
            assert!(empty, "a second run writes nothing");
        }
    }

    #[test]
    fn inward_sphere_turned_outward() {
        let clean = uv_sphere(8, 12, 1.0);
        let n = clean.triangle_count();
        let all: Vec<TriangleId> = (0..n as TriangleId).collect();
        let mut mesh = flip(&clean, &all);
        assert!(volume(&mesh) < 0.0);
        let (r, _) = edit(&mut mesh, orient);
        assert_eq!(
            r,
            OrientReport {
                flipped_triangles: n,
                patches: 1,
                non_orientable_patches: 0,
                flipped_outward_components: 1,
            }
        );
        assert!(volume(&mesh) > 0.0);
        assert_eq!(parts(&mesh), parts(&clean));

        // Inward with a few outward strays: the majority turns the strays
        // inward, then the whole sphere outward, so the strays stay as they are.
        let strays = scattered(n as u32, 5);
        let inward: Vec<TriangleId> = all
            .iter()
            .copied()
            .filter(|t| !strays.contains(t))
            .collect();
        let mut mesh = flip(&clean, &inward);
        let (r, _) = edit(&mut mesh, orient);
        assert_eq!(
            (r.flipped_triangles, r.flipped_outward_components),
            (n - 5, 1)
        );
        assert_eq!(parts(&mesh), parts(&clean));

        // Far from the origin, where the volume is taken about a local point.
        let far = box_mesh([1e5; 3], [1e5 + 1.0; 3]);
        let mut mesh = flip(&far, &(0..12).collect::<Vec<_>>());
        let (r, _) = edit(&mut mesh, orient);
        assert_eq!((r.flipped_triangles, r.flipped_outward_components), (12, 1));
        assert_eq!(parts(&mesh), parts(&far));
    }

    #[test]
    fn mobius_reported_unchanged() {
        let mut mesh = mobius(12);
        let before = parts(&mesh);
        let (r, empty) = edit(&mut mesh, orient);
        assert_eq!(
            r,
            OrientReport {
                flipped_triangles: 0,
                patches: 1,
                non_orientable_patches: 1,
                flipped_outward_components: 0,
            }
        );
        assert!(empty);
        assert_eq!(parts(&mesh), before);

        // Beside it, a damaged hemisphere is still repaired.
        let clean = combine(&[&mobius(12), &hemisphere(8, 24, 1.0)]);
        let mut mesh = flip(&clean, &[30, 100, 101, 250]);
        let (r, _) = edit(&mut mesh, orient);
        assert_eq!(
            (r.flipped_triangles, r.patches, r.non_orientable_patches),
            (4, 2, 1)
        );
        assert_eq!(parts(&mesh), parts(&clean));
    }

    #[test]
    fn fin_patches_independent() {
        // Quad 0: triangle 1 reversed, triangle 0 (the smallest id) wins the
        // tie. Quad 1: triangle 2 reversed and wins the tie. Quad 2: both
        // reversed, consistent on its own, so left alone.
        let clean = fin();
        let mut mesh = flip(&clean, &[1, 2, 4, 5]);
        let (r, _) = edit(&mut mesh, orient);
        assert_eq!(
            r,
            OrientReport {
                flipped_triangles: 2,
                patches: 3,
                non_orientable_patches: 0,
                flipped_outward_components: 0,
            }
        );
        assert_eq!(conflicts(&mesh), 0);
        let expected = flip(&clean, &[2, 3, 4, 5]);
        assert_eq!(parts(&mesh), parts(&expected));
    }

    #[test]
    fn already_consistent_is_empty_patch() {
        for clean in [uv_sphere(8, 12, 1.0), hemisphere(4, 8, 1.0), grid(3), fin()] {
            let mut mesh = clean;
            let (r, empty) = edit(&mut mesh, orient);
            assert_eq!(r.flipped_triangles, 0);
            assert!(empty);
        }
    }

    #[test]
    fn damaged_sphere_flips_fixed_and_degenerates_skipped() {
        let mut mesh = damaged_sphere();
        let before = parts(&mesh);
        let (r, _) = edit(&mut mesh, orient);
        // Holes, slits and the copied triangle leave the sphere open: no
        // outward pass, only the five reversed triangles are turned back.
        assert_eq!(r.flipped_triangles, 5);
        assert_eq!(
            (r.non_orientable_patches, r.flipped_outward_components),
            (0, 0)
        );
        assert_eq!(conflicts(&mesh), 0);
        assert_eq!(mesh.triangle(166), before.1[166].into());
        assert_eq!(mesh.triangle(167), before.1[167].into());
    }

    #[test]
    fn deterministic() {
        let damaged = || flip(&hemisphere(8, 24, 1.0), &scattered(360, 40));
        let (mut a, mut b) = (damaged(), damaged());
        let (ra, _) = edit(&mut a, orient);
        let (rb, _) = edit(&mut b, orient);
        assert_eq!(ra, rb);
        assert_eq!(parts(&a), parts(&b));
    }

    #[test]
    fn out_of_memory_is_reported_and_writes_nothing() {
        assert!(sweep(damaged_sphere(), |tx| orient(tx).map(drop)) > 5);
        let clean = uv_sphere(8, 12, 1.0);
        let inward = flip(
            &clean,
            &(0..clean.triangle_count() as TriangleId).collect::<Vec<_>>(),
        );
        assert!(sweep(inward, |tx| orient(tx).map(drop)) > 5);
    }
}
