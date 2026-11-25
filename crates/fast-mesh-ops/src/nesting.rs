//! Which closed components of a mesh lie inside which others.
//!
//! A hollow solid is an outward component around inward ones, its cavities.
//! Taken one by one, a cavity looks like a small, inside-out loose part;
//! repairs that treat it so fill the void or turn it into overlapping
//! material. [`Nesting`] tells the two apart.

use fast_mesh_bvh::{BuildOptions, Bvh};
use fast_mesh_query::{Ray, RayHit, RayOptions, raycast_all};
use fast_mesh_store::{Mesh, TriangleId};

use crate::topology::{Component, EdgeMap, components, cross, dot, point, sub, triangle_area};
use crate::{OpsError, Result, filled, push, reserve};

const NONE: u32 = u32::MAX;

/// A component whose volume is below this fraction of `area^1.5` encloses
/// nothing: its sign is rounding noise.
const FLAT: f64 = 1e-9;

/// The directions tried to place a point, none of them along an axis, a
/// diagonal or a plane that models are commonly aligned to.
const DIRS: [[f32; 3]; 5] = [
    [0.619_37, 0.273_91, 0.735_83],
    [-0.341_19, 0.872_53, 0.349_67],
    [0.157_71, -0.431_83, -0.888_07],
    [-0.791_23, -0.511_37, 0.335_29],
    [0.472_61, 0.687_11, -0.551_93],
];

/// Six times the signed volume enclosed by `triangles` once those marked in
/// `flip` are reversed, taken about the first triangle's first vertex so
/// that large coordinates do not swamp it.
pub(crate) fn signed_volume(mesh: &Mesh, triangles: &[TriangleId], flip: Option<&[bool]>) -> f64 {
    let Some(&first) = triangles.first() else {
        return 0.0;
    };
    let o = point(mesh, mesh.triangles()[first as usize][0]);
    let mut sum = 0.0;
    for &t in triangles {
        let [a, mut b, mut c] = mesh.triangles()[t as usize];
        if flip.is_some_and(|f| f[t as usize]) {
            std::mem::swap(&mut b, &mut c);
        }
        let [pa, pb, pc] = [a, b, c].map(|v| sub(point(mesh, v), o));
        sum += dot(pa, cross(pb, pc));
    }
    sum
}

/// The windings of the closed components and what encloses the inward ones.
pub(crate) struct Nesting {
    /// Per component: `1` closed and outward, `-1` closed and inward, `0`
    /// open or without volume.
    signs: Vec<i8>,
    /// Per component, the closed components it lies inside, ascending; empty
    /// for those not asked for.
    inside: Vec<Vec<u32>>,
}

impl Nesting {
    /// Finds what encloses the inward components of `comps`. Costs one pass
    /// over the triangles when there is nothing to find.
    pub(crate) fn of_inward(mesh: &Mesh, comps: &[Component]) -> Result<Self> {
        let mut signs = filled(comps.len(), 0i8)?;
        let mut inside = filled(comps.len(), Vec::new())?;
        if comps.len() >= 2 {
            for (s, c) in signs.iter_mut().zip(comps) {
                if c.closed {
                    let v = signed_volume(mesh, &c.triangles, None);
                    if v.abs() > 6.0 * FLAT * c.area.powf(1.5) {
                        *s = if v < 0.0 { -1 } else { 1 };
                    }
                }
            }
            if signs.contains(&-1) && signs.contains(&1) {
                let closed = |i: usize| signs[i] != 0;
                let inward = |i: usize| signs[i] < 0;
                inside = containers(mesh, comps, closed, inward)?;
            }
        }
        Ok(Self { signs, inside })
    }

    /// How many layers of material component `i` lies in, counting only the
    /// enclosing components that `keep` accepts.
    pub(crate) fn layers(&self, i: usize, keep: impl Fn(usize) -> bool) -> i32 {
        let kept = self.inside[i].iter().filter(|&&j| keep(j as usize));
        kept.map(|&j| i32::from(self.signs[j as usize])).sum()
    }

    /// Whether component `i` is a cavity: inward, and in the material of the
    /// components around it.
    pub(crate) fn is_cavity(&self, i: usize) -> bool {
        self.signs[i] < 0 && self.layers(i, |_| true) >= 1
    }

    /// How many components enclose component `i`.
    pub(crate) fn depth(&self, i: usize) -> usize {
        self.inside[i].len()
    }
}

/// [`components`] with [`Component::cavity`] set on the cavities.
pub(crate) fn components_with_cavities(mesh: &Mesh, edges: &EdgeMap) -> Result<Vec<Component>> {
    let mut comps = components(mesh, edges)?;
    let nesting = Nesting::of_inward(mesh, &comps)?;
    for (i, c) in comps.iter_mut().enumerate() {
        c.cavity = nesting.is_cavity(i);
    }
    Ok(comps)
}

/// For each component `i` with `wanted(i)`, the components `j` with
/// `closed(j)` it lies inside, ascending; empty for the others. Windings do
/// not matter here: a point is inside a closed component when a ray from it
/// crosses the component an odd number of times.
///
/// A ray through an edge or a vertex can miscount, so two rays must agree;
/// otherwise the majority of five decides.
pub(crate) fn containers(
    mesh: &Mesh,
    comps: &[Component],
    closed: impl Fn(usize) -> bool,
    wanted: impl Fn(usize) -> bool,
) -> Result<Vec<Vec<u32>>> {
    let mut out = filled(comps.len(), Vec::new())?;
    let mut owner = filled(mesh.triangle_slots(), NONE)?;
    for (i, c) in comps.iter().enumerate() {
        if closed(i) {
            for &t in &c.triangles {
                owner[t as usize] = i as u32;
            }
        }
    }
    let bvh = Bvh::build(mesh, &BuildOptions::default()).map_err(OpsError::from)?;
    let mut hits = Vec::new();
    for (i, c) in comps.iter().enumerate() {
        if !wanted(i) {
            continue;
        }
        let origin = sample(mesh, &c.triangles);
        let mut cast = |dir| crossed(mesh, &bvh, &owner, i as u32, Ray { origin, dir }, &mut hits);
        let mut votes = cast(DIRS[0])?;
        let second = cast(DIRS[1])?;
        if votes == second {
            out[i] = votes;
            continue;
        }
        reserve(&mut votes, second.len())?;
        votes.extend(second);
        for &dir in &DIRS[2..] {
            let more = cast(dir)?;
            reserve(&mut votes, more.len())?;
            votes.extend(more);
        }
        votes.sort_unstable();
        for run in votes.chunk_by(|a, b| a == b) {
            if run.len() > DIRS.len() / 2 {
                push(&mut out[i], run[0])?;
            }
        }
    }
    Ok(out)
}

/// The middle of the largest of `triangles`, the first of equals: a point on
/// the component, away from its edges.
fn sample(mesh: &Mesh, triangles: &[TriangleId]) -> [f32; 3] {
    let mut best = (triangles[0], -1.0);
    for &t in triangles {
        let area = triangle_area(mesh, mesh.triangles()[t as usize]);
        if area > best.1 {
            best = (t, area);
        }
    }
    let corners = mesh.triangles()[best.0 as usize].map(|v| point(mesh, v));
    std::array::from_fn(|k| (corners.iter().map(|p| p[k]).sum::<f64>() / 3.0) as f32)
}

/// The components but `own` that `ray` crosses an odd number of times,
/// ascending.
fn crossed(
    mesh: &Mesh,
    bvh: &Bvh,
    owner: &[u32],
    own: u32,
    ray: Ray,
    hits: &mut Vec<RayHit>,
) -> Result<Vec<u32>> {
    hits.clear();
    raycast_all(mesh, bvh, &ray, &RayOptions::default(), hits)?;
    let mut owners: Vec<u32> = Vec::new();
    reserve(&mut owners, hits.len())?;
    owners.extend(hits.iter().map(|h| owner[h.triangle as usize]));
    owners.retain(|&c| c != NONE && c != own);
    owners.sort_unstable();
    let mut odd = Vec::new();
    for run in owners.chunk_by(|a, b| a == b) {
        if run.len() % 2 == 1 {
            push(&mut odd, run[0])?;
        }
    }
    Ok(odd)
}

#[cfg(test)]
mod tests {
    use fast_mesh_store::{Mesh, TriangleId};

    use crate::analyze::{AnalyzeOptions, analyze};
    use crate::test_util::*;
    use crate::{
        AutoRepairOptions, ComponentOptions, auto_repair, measure, orient, remove_small_components,
    };

    fn inward(m: &Mesh) -> Mesh {
        let all: Vec<TriangleId> = (0..m.triangle_count() as TriangleId).collect();
        flip(m, &all)
    }

    fn cube(center: [f32; 3], half: f32) -> Mesh {
        box_mesh(center.map(|c| c - half), center.map(|c| c + half))
    }

    /// A solid ball with a cubic void of volume 8.
    fn hollow_ball() -> Mesh {
        combine(&[&uv_sphere(40, 40, 10.0), &inward(&cube([2.0; 3], 1.0))])
    }

    fn ball_volume() -> f64 {
        volume(&uv_sphere(40, 40, 10.0))
    }

    fn volume(m: &Mesh) -> f64 {
        measure(m).unwrap().volume.unwrap()
    }

    fn cavities(m: &Mesh) -> Vec<bool> {
        let a = analyze(m, &AnalyzeOptions::default()).unwrap();
        a.components.iter().map(|c| c.cavity).collect()
    }

    fn below(n: usize) -> ComponentOptions {
        ComponentOptions {
            min_triangles: Some(n),
            ..ComponentOptions::default()
        }
    }

    fn close(a: f64, b: f64) -> bool {
        (a - b).abs() < 1e-2
    }

    #[test]
    fn analysis_marks_cavities_only() {
        assert_eq!(cavities(&hollow_ball()), [false, true]);
        let ball = uv_sphere(40, 40, 10.0);
        // Loose outside, overlapping inside, inside out but outside, inside
        // out and alone.
        let loose = cube([20.0; 3], 1.0);
        let nested = cube([2.0; 3], 1.0);
        assert_eq!(cavities(&combine(&[&ball, &loose])), [false, false]);
        assert_eq!(cavities(&combine(&[&ball, &nested])), [false, false]);
        assert_eq!(
            cavities(&combine(&[&ball, &inward(&loose)])),
            [false, false]
        );
        assert_eq!(cavities(&inward(&ball)), [false]);
        // In the corner of the ball's box, but outside the ball.
        let corner = cube([8.0; 3], 0.5);
        assert_eq!(
            cavities(&combine(&[&ball, &inward(&corner)])),
            [false, false]
        );
    }

    #[test]
    fn an_island_in_a_cavity_and_its_own_cavity() {
        let m = combine(&[
            &uv_sphere(40, 40, 10.0),
            &inward(&cube([0.0; 3], 4.0)),
            &cube([0.0; 3], 2.0),
            &inward(&cube([0.0; 3], 1.0)),
        ]);
        assert_eq!(cavities(&m), [false, true, false, true]);
        // The wrong way round in the cavity: no material there to be a void of.
        let m = combine(&[
            &uv_sphere(40, 40, 10.0),
            &inward(&cube([0.0; 3], 4.0)),
            &inward(&cube([0.0; 3], 2.0)),
        ]);
        assert_eq!(cavities(&m), [false, true, false]);
    }

    #[test]
    fn many_cavities_on_a_lattice_of_rays_through_edges() {
        // Cubes on a regular grid, so rays meet many edges and corners.
        let ball = uv_sphere(40, 40, 10.0);
        let mut parts = vec![ball];
        for x in -2..=2 {
            for y in -2..=2 {
                for z in -2..=2 {
                    parts.push(inward(&cube([x, y, z].map(|k| k as f32 * 2.0), 0.5)));
                }
            }
        }
        let m = combine(&parts.iter().collect::<Vec<_>>());
        let found = cavities(&m);
        assert_eq!(found.iter().filter(|&&c| c).count(), 125);
        assert!(!found[0]);
    }

    #[test]
    fn far_from_the_origin() {
        let far = |m: &Mesh| {
            let (p, t) = parts(m);
            mesh_of(&p.iter().map(|x| x + 65_536.0).collect::<Vec<_>>(), &t)
        };
        assert_eq!(cavities(&far(&hollow_ball())), [false, true]);
    }

    #[test]
    fn orient_keeps_cavities() {
        let mut m = hollow_ball();
        let before = parts(&m);
        let (r, unchanged) = edit(&mut m, orient);
        assert!(unchanged);
        assert_eq!(r.flipped_outward_components, 0);
        assert_eq!(parts(&m), before);

        // Inside out as a whole: turned as a whole.
        let mut m = inward(&hollow_ball());
        let (r, _) = edit(&mut m, orient);
        assert_eq!(r.flipped_outward_components, 2);
        assert!(close(volume(&m), ball_volume() - 8.0));

        // Nested parts that face the same way are parts, not cavities.
        let mut m = inward(&combine(&[&uv_sphere(40, 40, 10.0), &cube([2.0; 3], 1.0)]));
        let (r, _) = edit(&mut m, orient);
        assert_eq!(r.flipped_outward_components, 2);
        assert!(close(volume(&m), ball_volume() + 8.0));

        // An inside-out loose part still turns outward.
        let mut m = combine(&[&uv_sphere(40, 40, 10.0), &inward(&cube([20.0; 3], 1.0))]);
        let (r, _) = edit(&mut m, orient);
        assert_eq!(r.flipped_outward_components, 1);
        assert!(close(volume(&m), ball_volume() + 8.0));

        // Ball, cavity, island, cavity in the island; both ways round.
        let nest = combine(&[
            &uv_sphere(40, 40, 10.0),
            &inward(&cube([0.0; 3], 4.0)),
            &cube([0.0; 3], 2.0),
            &inward(&cube([0.0; 3], 1.0)),
        ]);
        let want = ball_volume() - 512.0 + 64.0 - 8.0;
        for mut m in [inward(&nest), nest] {
            edit(&mut m, orient);
            assert!(close(volume(&m), want));
        }
    }

    #[test]
    fn orient_repairs_a_cavity_with_random_flips() {
        let mut m = flip(&hollow_ball(), &[3, 700, 1500, 3121, 3125]);
        edit(&mut m, orient);
        assert!(close(volume(&m), ball_volume() - 8.0));
    }

    #[test]
    fn removal_keeps_cavities_and_drops_stray_pieces() {
        let mut m = combine(&[
            &hollow_ball(),
            &cube([20.0; 3], 1.0),
            &inward(&cube([30.0; 3], 1.0)),
        ]);
        let (r, _) = edit(&mut m, |tx| remove_small_components(tx, &below(500)));
        assert_eq!((r.removed_components, r.removed_triangles), (2, 24));
        assert!(close(volume(&m), ball_volume() - 8.0));

        // On request the cavity goes too, as it did before.
        let mut m = hollow_ball();
        let options = ComponentOptions::builder()
            .min_triangles(500)
            .keep_cavities(false)
            .build();
        let (r, _) = edit(&mut m, |tx| remove_small_components(tx, &options));
        assert_eq!(r.removed_components, 1);
        assert!(close(volume(&m), ball_volume()));
    }

    #[test]
    fn removal_drops_a_piece_floating_in_a_cavity() {
        let mut m = combine(&[
            &uv_sphere(40, 40, 10.0),
            &inward(&cube([0.0; 3], 4.0)),
            &cube([0.0; 3], 2.0),
            &inward(&cube([0.0; 3], 1.0)),
        ]);
        let (r, _) = edit(&mut m, |tx| remove_small_components(tx, &below(500)));
        // The island and its own cavity.
        assert_eq!(r.removed_components, 2);
        assert!(close(volume(&m), ball_volume() - 512.0));
    }

    #[test]
    fn a_small_hollow_piece_goes_with_its_cavity_whatever_its_size() {
        let mut m = combine(&[&uv_sphere(60, 60, 30.0), &cube([60.0; 3], 10.0), &{
            let (p, t) = parts(&inward(&uv_sphere(40, 40, 5.0)));
            mesh_of(&p.iter().map(|x| x + 60.0).collect::<Vec<_>>(), &t)
        }]);
        assert_eq!(cavities(&m), [false, false, true]);
        let (r, _) = edit(&mut m, |tx| remove_small_components(tx, &below(500)));
        assert_eq!(r.removed_components, 2);
        assert!(close(volume(&m), volume(&uv_sphere(60, 60, 30.0))));
    }

    #[test]
    fn auto_repair_keeps_the_void() {
        for min_triangles in [0, 500] {
            let mut m = hollow_ball();
            let options = AutoRepairOptions {
                min_triangles,
                ..AutoRepairOptions::default()
            };
            edit(&mut m, |tx| auto_repair(tx, &options));
            assert!(close(volume(&m), ball_volume() - 8.0), "{min_triangles}");
        }
    }

    #[test]
    fn out_of_memory_is_reported_and_writes_nothing() {
        let stray = cube([20.0; 3], 1.0);
        let m = combine(&[
            &uv_sphere(8, 8, 10.0),
            &inward(&cube([2.0; 3], 1.0)),
            &stray,
        ]);
        let removal = sweep(m, |tx| remove_small_components(tx, &below(20)).map(|_| ()));
        assert!(removal > 0);
        let hollow = combine(&[&uv_sphere(8, 8, 10.0), &inward(&cube([2.0; 3], 1.0))]);
        let orienting = sweep(inward(&hollow), |tx| orient(tx).map(|_| ()));
        assert!(orienting > 0);
    }

    #[test]
    fn same_result_across_threads() {
        assert_identical_across_threads("cavities", || {
            let a = analyze(&hollow_ball(), &AnalyzeOptions::default()).unwrap();
            a.components
        });
    }
}
