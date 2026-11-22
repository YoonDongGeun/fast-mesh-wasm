//! Convex hull properties, checked with exact predicates on every result:
//! every input point is inside or on the hull, the hull is convex (every
//! vertex on or behind every face plane), closed (every edge has exactly two
//! faces, used in opposite directions, and every vertex one fan) and oriented
//! outward, with no degenerate face.
//!
//! The generators lean on degeneracy: points on a small integer lattice (many
//! duplicates, coplanar and collinear points), boxes with grid points on their
//! faces, and points a single ulp off a plane or a line. A lattice set
//! without volume must be reported as the error that names its degeneracy.
//!
//! Seeds: `PROPTEST_RNG_SEED` is honoured; without it the seed is fixed, so
//! that CI runs are deterministic.

use std::collections::{BTreeMap, BTreeSet};

use fast_mesh_ops::inspect::PointCloud;
use fast_mesh_ops::{
    ConvexHull, OpsError, convex_hull, convex_hull_mesh, convex_hull_meshes,
    convex_hull_point_cloud,
};
use fast_mesh_store::{Mesh, par};
use proptest::prelude::*;
use proptest::test_runner::RngSeed;
use robust::{Coord, Coord3D, orient2d, orient3d};

/// The seed used when `PROPTEST_RNG_SEED` is not set.
const DEFAULT_SEED: u64 = 0x0c0e_e8a1;

/// `PROPTEST_CASES` (else `cases`) and `PROPTEST_RNG_SEED` (else a fixed seed).
fn config(cases: u32) -> ProptestConfig {
    let base = ProptestConfig::default();
    let cases = std::env::var("PROPTEST_CASES")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(cases);
    let rng_seed = match base.rng_seed {
        RngSeed::Random => RngSeed::Fixed(DEFAULT_SEED),
        fixed => fixed,
    };
    ProptestConfig {
        cases,
        rng_seed,
        ..base
    }
}

fn c3(p: &[f64; 3]) -> Coord3D<f64> {
    Coord3D {
        x: p[0],
        y: p[1],
        z: p[2],
    }
}

/// Exact: positive when `d` is behind the outward face `(a, b, c)`.
fn behind(a: &[f64; 3], b: &[f64; 3], c: &[f64; 3], d: &[f64; 3]) -> f64 {
    orient3d(c3(a), c3(b), c3(c), c3(d))
}

fn collinear(a: &[f64; 3], b: &[f64; 3], c: &[f64; 3]) -> bool {
    [(0, 1), (1, 2), (2, 0)].iter().all(|&(i, j)| {
        let p = |q: &[f64; 3]| Coord { x: q[i], y: q[j] };
        orient2d(p(a), p(b), p(c)) == 0.0
    })
}

/// Checks every hull property exactly, panicking with `what` otherwise.
fn check_hull(points: &[[f64; 3]], hull: &ConvexHull, what: &str) {
    assert!(
        hull.vertices.windows(2).all(|w| w[0] < w[1]),
        "{what}: vertices not ascending"
    );
    assert!(
        hull.vertices.iter().all(|&v| (v as usize) < points.len()),
        "{what}: vertex out of range"
    );
    let pos: Vec<[f64; 3]> = hull.vertices.iter().map(|&v| points[v as usize]).collect();
    let n = pos.len();

    // Closed and consistently wound: every directed edge once, its reverse once.
    let mut edges = BTreeMap::new();
    let mut fans: Vec<BTreeMap<u32, u32>> = vec![BTreeMap::new(); n];
    for t in &hull.triangles {
        assert!(
            t.iter().all(|&c| (c as usize) < n),
            "{what}: corner out of range"
        );
        assert!(
            t[0] != t[1] && t[1] != t[2] && t[2] != t[0],
            "{what}: repeated corner {t:?}"
        );
        for e in 0..3 {
            let (a, b, c) = (t[e], t[(e + 1) % 3], t[(e + 2) % 3]);
            *edges.entry((a, b)).or_insert(0u32) += 1;
            // The fan around `a`: `b -> c`.
            assert!(
                fans[a as usize].insert(b, c).is_none(),
                "{what}: vertex {a} fan repeats {b}"
            );
        }
    }
    for (&(a, b), &k) in &edges {
        assert_eq!(k, 1, "{what}: directed edge {a}->{b} used {k} times");
        assert_eq!(
            edges.get(&(b, a)),
            Some(&1),
            "{what}: edge {a}->{b} is open"
        );
    }
    // Every vertex is used, by one closed fan.
    for (v, fan) in fans.iter().enumerate() {
        assert!(!fan.is_empty(), "{what}: unused hull vertex {v}");
        let (&start, _) = fan.iter().next().unwrap();
        let mut cur = start;
        let mut steps = 0;
        loop {
            cur = fan[&cur];
            steps += 1;
            if cur == start {
                break;
            }
            assert!(steps <= fan.len(), "{what}: vertex {v} fan is open");
        }
        assert_eq!(steps, fan.len(), "{what}: vertex {v} has more than one fan");
    }
    // A sphere: V - E + F = 2.
    let (v, e, f) = (
        n as i64,
        edges.len() as i64 / 2,
        hull.triangles.len() as i64,
    );
    assert_eq!(v - e + f, 2, "{what}: Euler characteristic");

    for t in &hull.triangles {
        let [a, b, c] = t.map(|i| &pos[i as usize]);
        assert!(!collinear(a, b, c), "{what}: degenerate face {t:?}");
        // Convex, and outward: every vertex on or behind, some strictly.
        let mut strictly = false;
        for q in &pos {
            let o = behind(a, b, c, q);
            assert!(o >= 0.0, "{what}: a hull vertex is above face {t:?}");
            strictly |= o > 0.0;
        }
        assert!(strictly, "{what}: face {t:?} has every vertex on its plane");
        for q in points {
            assert!(
                behind(a, b, c, q) >= 0.0,
                "{what}: input point {q:?} is outside face {t:?}"
            );
        }
    }
}

/// What the hull of a set without volume must say.
fn expected_degeneracy(points: &[[f64; 3]]) -> Option<&'static str> {
    let a = &points[0];
    let Some(b) = points.iter().find(|p| *p != a) else {
        return Some("convex hull points all coincide");
    };
    let Some(c) = points.iter().find(|p| !collinear(a, b, p)) else {
        return Some("convex hull points are collinear");
    };
    if points.iter().all(|p| behind(a, b, c, p) == 0.0) {
        return Some("convex hull points are coplanar");
    }
    None
}

/// A hull of `points`, checked; or, for a set without volume, the right error.
fn hull_or_degenerate(points: &[[f64; 3]], what: &str) -> Option<ConvexHull> {
    let expected = expected_degeneracy(points);
    match convex_hull(points) {
        Ok(hull) => {
            assert_eq!(expected, None, "{what}: degenerate set got a hull");
            check_hull(points, &hull, what);
            Some(hull)
        }
        Err(OpsError::InvalidInput(msg)) => {
            assert_eq!(Some(msg), expected, "{what}: wrong error");
            None
        }
        Err(e) => panic!("{what}: {e}"),
    }
}

fn cube_corners() -> Vec<[f64; 3]> {
    (0..8)
        .map(|i| {
            [
                f64::from(i & 1),
                f64::from((i >> 1) & 1),
                f64::from((i >> 2) & 1),
            ]
        })
        .collect()
}

/// A deterministic linear congruential generator.
struct Lcg(u64);

impl Lcg {
    fn unit(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.0 >> 11) as f64 / (1u64 << 53) as f64
    }

    fn on_sphere(&mut self) -> [f64; 3] {
        loop {
            let p = [0, 1, 2].map(|_| 2.0 * self.unit() - 1.0);
            let r2 = p[0] * p[0] + p[1] * p[1] + p[2] * p[2];
            if r2 < 1.0 && r2 > 1e-4 {
                let l = r2.sqrt();
                return p.map(|c| c / l);
            }
        }
    }
}

#[test]
fn tetrahedron_and_cube() {
    let tet = [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ];
    let h = convex_hull(&tet).unwrap();
    assert_eq!(h.vertices, vec![0, 1, 2, 3]);
    assert_eq!(h.triangles.len(), 4);
    check_hull(&tet, &h, "tetrahedron");
    // The other winding of the same four points.
    let flipped = [tet[0], tet[2], tet[1], tet[3]];
    check_hull(
        &flipped,
        &convex_hull(&flipped).unwrap(),
        "flipped tetrahedron",
    );

    let cube = cube_corners();
    let h = convex_hull(&cube).unwrap();
    assert_eq!((h.vertices.len(), h.triangles.len()), (8, 12));
    check_hull(&cube, &h, "cube");
}

#[test]
fn duplicates_and_inside_points_are_left_out() {
    let mut pts = cube_corners();
    pts.extend(cube_corners());
    pts.extend([
        [0.5, 0.5, 0.5],
        [0.25, 0.75, 0.5],
        [0.5, 0.5, 1.0],
        [0.5, 0.0, 0.0],
    ]);
    let h = convex_hull(&pts).unwrap();
    assert!(h.vertices.iter().all(|&v| v < 8), "{:?}", h.vertices);
    assert_eq!((h.vertices.len(), h.triangles.len()), (8, 12));
    check_hull(&pts, &h, "doubled cube with face, edge and inside points");
}

#[test]
fn a_lattice_cube_keeps_only_its_corners() {
    let mut pts = Vec::new();
    for z in 0..6 {
        for y in 0..6 {
            for x in 0..6 {
                pts.push([f64::from(x), f64::from(y), f64::from(z)]);
            }
        }
    }
    let h = convex_hull(&pts).unwrap();
    check_hull(&pts, &h, "6^3 lattice");
    assert_eq!((h.vertices.len(), h.triangles.len()), (8, 12));
}

#[test]
fn points_one_ulp_off_a_plane_or_a_line() {
    let up = |x: f64| f64::from_bits(x.to_bits() + 1);
    let mut pts: Vec<[f64; 3]> = (0..16)
        .map(|i| {
            let (x, y) = (f64::from(i % 4), f64::from(i / 4));
            [x, y, x + y]
        })
        .collect();
    pts.push([1.0, 2.0, up(3.0)]);
    let h = convex_hull(&pts).unwrap();
    check_hull(&pts, &h, "one ulp off a plane");
    assert!(h.vertices.contains(&16));

    let mut line: Vec<[f64; 3]> = (0..8)
        .map(|i| [f64::from(i), 2.0 * f64::from(i), 1.0])
        .collect();
    line.push([3.0, up(6.0), 1.0]);
    line.push([5.0, 10.0, up(1.0)]);
    check_hull(&line, &convex_hull(&line).unwrap(), "one ulp off a line");
}

#[test]
fn points_in_general_position_give_the_same_hull_in_any_order() {
    let mut r = Lcg(11);
    let pts: Vec<[f64; 3]> = (0..2000)
        .map(|_| [r.unit(), r.unit(), r.unit()].map(|c| 2.0 * c - 1.0))
        .collect();
    let mut order: Vec<usize> = (0..pts.len()).collect();
    for i in (1..order.len()).rev() {
        order.swap(i, (r.unit() * (i + 1) as f64) as usize);
    }
    let shuffled: Vec<[f64; 3]> = order.iter().map(|&i| pts[i]).collect();
    // Faces as sets of coordinates, each rotated to start at its least corner.
    let faces = |pts: &[[f64; 3]], h: &ConvexHull| -> BTreeSet<[[u64; 3]; 3]> {
        h.triangles
            .iter()
            .map(|t| {
                let c = t.map(|i| pts[h.vertices[i as usize] as usize].map(f64::to_bits));
                let k = (0..3).min_by_key(|&k| c[k]).unwrap();
                [c[k], c[(k + 1) % 3], c[(k + 2) % 3]]
            })
            .collect()
    };
    let a = convex_hull(&pts).unwrap();
    let b = convex_hull(&shuffled).unwrap();
    check_hull(&pts, &a, "random");
    assert_eq!(faces(&pts, &a), faces(&shuffled, &b));
}

#[test]
fn parallel_and_sequential_hulls_are_bit_identical() {
    struct Sequential;
    impl Drop for Sequential {
        fn drop(&mut self) {
            par::set_force_sequential(false);
        }
    }
    let mut r = Lcg(5);
    let sphere: Vec<[f64; 3]> = (0..200_000).map(|_| r.on_sphere()).collect();
    // A lattice box: coplanar faces, so the triangulation depends on the
    // order points are classified in.
    let mut lattice = Vec::new();
    for i in 0..150_000u32 {
        let c = [i % 53, (i / 53) % 59, (i * 7) % 61].map(f64::from);
        lattice.push(c);
    }
    let mut cube: Vec<[f64; 3]> = (0..300_000)
        .map(|_| [r.unit(), r.unit(), r.unit()])
        .collect();
    cube.extend(cube_corners());
    for (name, pts) in [("sphere", &sphere), ("lattice", &lattice), ("cube", &cube)] {
        let threaded = par::with_threads(4, || convex_hull(pts)).unwrap();
        let sequential = par::with_threads(1, || {
            par::set_force_sequential(true);
            let _guard = Sequential;
            convex_hull(pts)
        })
        .unwrap();
        assert_eq!(threaded, sequential, "{name}");
    }
}

#[test]
fn mesh_and_point_cloud_hulls() {
    let cube: Vec<f32> = cube_corners().iter().flatten().map(|&c| c as f32).collect();
    // An open box (no top face) with an unused vertex at its centre.
    let mut positions = cube.clone();
    positions.extend([0.5, 0.5, 0.5]);
    let indices = [0, 2, 1, 1, 2, 3, 0, 1, 5, 0, 5, 4, 0, 4, 6, 0, 6, 2];
    let mesh = Mesh::from_buffers(&positions, &indices).unwrap();
    let hull = convex_hull_mesh(&mesh).unwrap();
    assert_eq!(hull.origin(), mesh.origin());
    // Corner 7 and the centre are unused: a cube with a corner cut off.
    assert_eq!((hull.vertex_count(), hull.triangle_count()), (7, 10));
    for p in hull.positions() {
        assert!(mesh.positions().contains(p), "{p:?} is not a mesh position");
    }

    // A far tetrahedron with another origin: the union's corners are the
    // box's x = 0 face and three of the tetrahedron's.
    let far = Mesh::from_buffers_f64(
        &[
            10.0, 0.0, 0.0, 11.0, 0.0, 0.0, 10.0, 1.0, 0.0, 10.0, 0.0, 1.0,
        ],
        &[0, 2, 1, 0, 1, 3, 0, 3, 2, 1, 2, 3],
    )
    .unwrap();
    assert_ne!(far.origin(), mesh.origin());
    let both = convex_hull_meshes(&[&mesh, &far]).unwrap();
    assert_eq!(both.vertex_count(), 7);
    let mut world: Vec<[f64; 3]> = Vec::new();
    for v in 0..both.vertex_slots() as u32 {
        world.push(both.world_position(v).unwrap());
    }
    assert!(world.contains(&[11.0, 0.0, 0.0]) && world.contains(&[0.0, 1.0, 1.0]));

    let cloud =
        PointCloud::from_positions(cube_corners().iter().map(|p| p.map(|c| c as f32)).collect());
    let hull = convex_hull_point_cloud(&cloud).unwrap();
    assert_eq!((hull.vertex_count(), hull.triangle_count()), (8, 12));
    assert_eq!(hull.origin(), [0.0; 3]);
}

/// Points on the integer lattice `-r..=r`: duplicates, coplanar and
/// collinear sets, and sets without volume.
fn lattice(r: i32, max: usize) -> impl Strategy<Value = Vec<[f64; 3]>> {
    prop::collection::vec(prop::array::uniform3(-r..=r), 4..max)
        .prop_map(|v| v.into_iter().map(|p| p.map(f64::from)).collect())
}

/// Random points in a cube with some repeated.
fn scattered() -> impl Strategy<Value = Vec<[f64; 3]>> {
    (
        prop::collection::vec(prop::array::uniform3(-1.0f64..1.0), 4..300),
        prop::collection::vec(any::<prop::sample::Index>(), 0..40),
    )
        .prop_map(|(mut pts, dupes)| {
            for d in dupes {
                let p = pts[d.index(pts.len())];
                pts.push(p);
            }
            pts
        })
}

/// A box's grid points on its faces, some random inside points, in a random
/// order: big coplanar sets on every side.
fn boxed() -> impl Strategy<Value = (Vec<[f64; 3]>, [u32; 3])> {
    (
        prop::array::uniform3(1u32..5),
        prop::collection::vec(prop::array::uniform3(0.01f64..0.99), 0..30),
        any::<u64>(),
    )
        .prop_map(|(dims, inside, seed)| {
            let mut pts = Vec::new();
            for z in 0..=dims[2] {
                for y in 0..=dims[1] {
                    for x in 0..=dims[0] {
                        let c = [x, y, z];
                        if (0..3).any(|k| c[k] == 0 || c[k] == dims[k]) {
                            pts.push(c.map(f64::from));
                        }
                    }
                }
            }
            pts.extend(
                inside
                    .iter()
                    .map(|p| std::array::from_fn(|k| p[k] * f64::from(dims[k]))),
            );
            let mut r = Lcg(seed);
            for i in (1..pts.len()).rev() {
                pts.swap(i, (r.unit() * (i + 1) as f64) as usize);
            }
            (pts, dims)
        })
}

proptest! {
    #![proptest_config(config(256))]

    #[test]
    fn lattice_points_give_a_valid_hull_or_the_right_error(pts in lattice(2, 60)) {
        hull_or_degenerate(&pts, "lattice r=2");
    }

    #[test]
    fn wide_lattice_points_give_a_valid_hull_or_the_right_error(pts in lattice(6, 200)) {
        hull_or_degenerate(&pts, "lattice r=6");
    }

    #[test]
    fn flat_lattice_sets_are_errors(
        pts in prop::collection::vec((-4i32..=4, -4i32..=4), 4..40),
        normal in prop::array::uniform3(-2i32..=2),
    ) {
        // Every point on the plane `n . p = 0`, through integer coordinates
        // where `n` has a non-zero z, else on the z = 0 plane.
        let pts: Vec<[f64; 3]> = pts
            .into_iter()
            .map(|(x, y)| {
                let z = if normal[2] != 0 { -(normal[0] * x + normal[1] * y) * normal[2] } else { 0 };
                let s = if normal[2] != 0 { normal[2] * normal[2] } else { 1 };
                [x * s, y * s, z].map(f64::from)
            })
            .collect();
        prop_assert!(expected_degeneracy(&pts).is_some());
        prop_assert!(hull_or_degenerate(&pts, "flat lattice").is_none());
    }

    #[test]
    fn scattered_points_give_a_valid_hull(pts in scattered()) {
        let hull = hull_or_degenerate(&pts, "scattered");
        prop_assert!(hull.is_some());
    }

    #[test]
    fn boxes_keep_their_corners(case in boxed()) {
        let (pts, dims) = case;
        let hull = hull_or_degenerate(&pts, "box").unwrap();
        let verts: BTreeSet<[u64; 3]> = hull
            .vertices
            .iter()
            .map(|&v| pts[v as usize].map(f64::to_bits))
            .collect();
        for i in 0..8u32 {
            let corner: [f64; 3] = std::array::from_fn(|k| f64::from(((i >> k) & 1) * dims[k]));
            prop_assert!(verts.contains(&corner.map(f64::to_bits)), "corner {:?}", corner);
        }
    }

    #[test]
    fn sphere_points_give_a_valid_hull(seed in any::<u64>(), n in 4usize..400) {
        let mut r = Lcg(seed);
        let pts: Vec<[f64; 3]> = (0..n)
            .map(|_| r.on_sphere().map(|c| f64::from(c as f32)))
            .collect();
        hull_or_degenerate(&pts, "sphere");
    }
}
