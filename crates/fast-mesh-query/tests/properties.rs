//! Property tests: after any sequence of edits, undos and redos, an updated BVH
//! must answer every query exactly like a brute-force scan over the alive
//! triangles, and rays fired from inside a closed mesh must never escape it.
//!
//! The brute-force oracles test each alive triangle on its own, as a
//! one-triangle mesh queried through the public API. That reuses the exact
//! per-triangle arithmetic and leaves the BVH traversal, the pruning and the
//! tie rules (smallest triangle id wins) as the thing under test.
//!
//! The mesh-vs-mesh oracles do the same for every pair of alive triangles: each
//! pair is two one-triangle meshes queried through the public API.

use std::collections::HashMap;

use fast_mesh_bvh::{Aabb, BuildOptions, Bvh, Quality};
use fast_mesh_query::{
    ClosestOptions, Cull, MeshDistance, Ray, RayOptions, Region, Rigid, closest_point,
    intersecting_pairs, intersects, mesh_distance, raycast, raycast_all, raycast_any,
    triangles_in, triangles_near, vertices_in,
};
use fast_mesh_store::{History, Mesh, Transaction};
use proptest::prelude::*;

/// The case count: `PROPTEST_CASES` when set, else `default`. `with_cases`
/// alone would ignore the environment variable.
fn cases(default: u32) -> u32 {
    std::env::var("PROPTEST_CASES")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(default)
}

/// A deterministic linear congruential generator.
struct Lcg(u64);

impl Lcg {
    /// Uniform in `[0, 1)`.
    fn next(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.0 >> 40) as f32 / (1u64 << 24) as f32
    }

    /// Uniform in `[lo, hi)`.
    fn range(&mut self, lo: f32, hi: f32) -> f32 {
        lo + self.next() * (hi - lo)
    }
}

#[derive(Debug, Clone)]
enum Op {
    Move(u32, [i8; 3]),
    AddVertex(u32, u32, u8),
    AddTriangle([u32; 3]),
    RemoveTriangle(u32),
    SetTriangle(u32, [u32; 3]),
}

#[derive(Debug, Clone)]
enum Step {
    Commit(Vec<Op>),
    Undo,
    Redo,
}

fn op() -> impl Strategy<Value = Op> {
    prop_oneof![
        4 => (any::<u32>(), any::<[i8; 3]>()).prop_map(|(v, d)| Op::Move(v, d)),
        1 => (any::<u32>(), any::<u32>(), any::<u8>()).prop_map(|(a, b, w)| Op::AddVertex(a, b, w)),
        2 => any::<[u32; 3]>().prop_map(Op::AddTriangle),
        2 => any::<u32>().prop_map(Op::RemoveTriangle),
        1 => (any::<u32>(), any::<[u32; 3]>()).prop_map(|(t, v)| Op::SetTriangle(t, v)),
    ]
}

fn step() -> impl Strategy<Value = Step> {
    prop_oneof![
        5 => prop::collection::vec(op(), 1..8).prop_map(Step::Commit),
        1 => Just(Step::Undo),
        1 => Just(Step::Redo),
    ]
}

fn options() -> impl Strategy<Value = BuildOptions> {
    (
        prop_oneof![Just(Quality::Sah), Just(Quality::Fast)],
        1u32..=8,
    )
        .prop_map(|(quality, max_leaf_size)| BuildOptions {
            quality,
            max_leaf_size,
            ..Default::default()
        })
}

/// A wavy n×n grid, so triangles are not all coplanar.
fn grid(n: u32) -> Mesh {
    let mut positions = Vec::new();
    for y in 0..=n {
        for x in 0..=n {
            positions.extend([x as f32, y as f32, ((x * 7 + y * 3) % 5) as f32 * 0.3]);
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

fn alive(m: &Mesh, vertices: bool) -> Vec<u32> {
    let n = if vertices {
        m.vertex_slots()
    } else {
        m.triangle_slots()
    } as u32;
    (0..n)
        .filter(|&i| {
            if vertices {
                m.is_vertex_alive(i)
            } else {
                m.is_triangle_alive(i)
            }
        })
        .collect()
}

fn pick(ids: &[u32], seed: u32) -> Option<u32> {
    (!ids.is_empty()).then(|| ids[seed as usize % ids.len()])
}

fn run(tx: &mut Transaction<'_>, op: &Op) {
    let vs = alive(tx.mesh(), true);
    let ts = alive(tx.mesh(), false);
    let v = |s: u32| pick(&vs, s);
    match *op {
        Op::Move(s, d) => {
            if let Some(v) = v(s) {
                let p = tx.mesh().position(v).unwrap();
                let q = [0, 1, 2].map(|k| p[k] + d[k] as f32 * 0.05);
                tx.set_positions(&[v], &[q]).unwrap();
            }
        }
        Op::AddVertex(a, b, w) => {
            if let (Some(a), Some(b)) = (v(a), v(b)) {
                let (pa, pb) = (
                    tx.mesh().position(a).unwrap(),
                    tx.mesh().position(b).unwrap(),
                );
                let t = w as f32 / 255.0;
                let p = [0, 1, 2].map(|k| pa[k] * (1.0 - t) + pb[k] * t);
                tx.add_vertex_with_parents(p, &[(a, 1.0 - t), (b, t)])
                    .unwrap();
            }
        }
        Op::AddTriangle(s) => {
            if let (Some(a), Some(b), Some(c)) = (v(s[0]), v(s[1]), v(s[2])) {
                tx.add_triangles(&[[a, b, c]]).unwrap();
            }
        }
        Op::RemoveTriangle(s) => {
            if let Some(t) = pick(&ts, s) {
                tx.remove_triangles(&[t]).unwrap();
            }
        }
        Op::SetTriangle(t, s) => {
            if let (Some(t), Some(a), Some(b), Some(c)) = (pick(&ts, t), v(s[0]), v(s[1]), v(s[2]))
            {
                tx.set_triangles(&[t], &[[a, b, c]]).unwrap();
            }
        }
    }
}

/// An alive triangle as a mesh of its own, with its BVH.
struct Single {
    id: u32,
    /// The corner positions, three coordinates each.
    flat: Vec<f32>,
    mesh: Mesh,
    bvh: Bvh,
}

fn singles(mesh: &Mesh) -> Vec<Single> {
    alive(mesh, false)
        .into_iter()
        .map(|id| {
            let flat: Vec<f32> = mesh
                .triangle(id)
                .unwrap()
                .iter()
                .flat_map(|&v| mesh.position(v).unwrap())
                .collect();
            let single = Mesh::from_buffers(&flat, &[0, 1, 2]).unwrap();
            let bvh = Bvh::build(&single, &BuildOptions::default()).unwrap();
            Single {
                id,
                flat,
                mesh: single,
                bvh,
            }
        })
        .collect()
}

/// Every hit, sorted by `(t, triangle)`.
fn brute_rays(singles: &[Single], ray: &Ray, opts: &RayOptions) -> Vec<(u32, f32)> {
    let mut hits: Vec<(u32, f32)> = singles
        .iter()
        .filter_map(|s| {
            raycast(&s.mesh, &s.bvh, ray, opts)
                .unwrap()
                .map(|h| (s.id, h.t))
        })
        .collect();
    hits.sort_by(|a, b| a.1.total_cmp(&b.1).then(a.0.cmp(&b.0)));
    hits
}

/// The nearest triangle and its distance, ties to the smaller id.
///
/// The query ranks by the squared distance, which the public API does not
/// report (the square roots of two different squared distances can be equal).
/// So the winner is found by a tournament over pairs: the alive triangles are
/// walked in ascending id, and each candidate plays the current best on a
/// two-triangle mesh, lower id first, in one leaf. That reproduces the exact
/// squared-distance comparison and the smaller-id rule.
fn brute_closest(singles: &[Single], p: [f32; 3], opts: &ClosestOptions) -> Option<(u32, f32)> {
    let mut best: Option<(&Single, f32)> = None;
    for s in singles {
        let Some(h) = closest_point(&s.mesh, &s.bvh, p, opts).unwrap() else {
            continue;
        };
        let Some((b, _)) = best else {
            best = Some((s, h.distance));
            continue;
        };
        let flat: Vec<f32> = b.flat.iter().chain(&s.flat).copied().collect();
        let pair = Mesh::from_buffers(&flat, &[0, 1, 2, 3, 4, 5]).unwrap();
        let bvh = Bvh::build(&pair, &BuildOptions::default()).unwrap();
        let w = closest_point(&pair, &bvh, p, opts).unwrap().unwrap();
        if w.triangle == 1 {
            best = Some((s, w.distance));
        } else {
            best = Some((b, w.distance));
        }
    }
    best.map(|(s, d)| (s.id, d))
}

fn brute_triangles(singles: &[Single], region: &Region) -> Vec<u32> {
    singles
        .iter()
        .filter(|s| {
            let mut out = Vec::new();
            triangles_in(&s.mesh, &s.bvh, region, &mut out).unwrap() == 1
        })
        .map(|s| s.id)
        .collect()
}

fn contains(region: &Region, q: [f32; 3]) -> bool {
    match region {
        Region::Box(b) => (0..3).all(|k| b.min[k] <= q[k] && q[k] <= b.max[k]),
        Region::Sphere { center, radius } => {
            let d: f32 = (0..3)
                .map(|k| (q[k] - center[k]) * (q[k] - center[k]))
                .sum();
            d <= radius * radius
        }
    }
}

fn brute_vertices(mesh: &Mesh, region: &Region) -> Vec<u32> {
    let mut v: Vec<u32> = alive(mesh, false)
        .into_iter()
        .flat_map(|t| mesh.triangle(t).unwrap())
        .filter(|&v| contains(region, mesh.position(v).unwrap()))
        .collect();
    v.sort_unstable();
    v.dedup();
    v
}

/// A ray from around the grid, either in a random direction or aimed at a
/// vertex, so that it grazes shared edges and corners.
fn random_ray(rng: &mut Lcg, mesh: &Mesh) -> Ray {
    let origin = [
        rng.range(-1.0, 7.0),
        rng.range(-1.0, 7.0),
        rng.range(-1.5, 3.0),
    ];
    let vs = alive(mesh, true);
    let mut dir = [
        rng.range(-1.0, 1.0),
        rng.range(-1.0, 1.0),
        rng.range(-1.0, 1.0),
    ];
    if rng.next() < 0.5
        && let Some(v) = pick(&vs, (rng.next() * 1e6) as u32)
    {
        let p = mesh.position(v).unwrap();
        dir = [0, 1, 2].map(|k| p[k] - origin[k]);
    }
    if dir == [0.0; 3] {
        dir = [0.0, 0.0, -1.0];
    }
    Ray { origin, dir }
}

fn random_ray_options(rng: &mut Lcg) -> RayOptions {
    match (rng.next() * 3.0) as u32 {
        0 => RayOptions::default(),
        1 => RayOptions {
            min_t: 0.5,
            max_t: 3.0,
            cull: Cull::None,
        },
        _ => RayOptions {
            cull: if rng.next() < 0.5 {
                Cull::Back
            } else {
                Cull::Front
            },
            ..RayOptions::default()
        },
    }
}

fn random_region(rng: &mut Lcg, sphere: bool) -> Region {
    let c = [
        rng.range(-1.0, 7.0),
        rng.range(-1.0, 7.0),
        rng.range(-1.0, 3.0),
    ];
    if sphere {
        Region::Sphere {
            center: c,
            radius: rng.range(0.0, 3.0),
        }
    } else {
        let h = [
            rng.range(0.0, 3.0),
            rng.range(0.0, 3.0),
            rng.range(0.0, 2.0),
        ];
        Region::Box(Aabb::new(
            std::array::from_fn(|k| c[k] - h[k]),
            std::array::from_fn(|k| c[k] + h[k]),
        ))
    }
}

fn check_queries(mesh: &Mesh, bvh: &Bvh, seed: u64) -> Result<(), TestCaseError> {
    prop_assert!(bvh.is_current(mesh));
    let mut rng = Lcg(seed);
    let singles = singles(mesh);
    for _ in 0..16 {
        let ray = random_ray(&mut rng, mesh);
        let opts = random_ray_options(&mut rng);
        let all = brute_rays(&singles, &ray, &opts);

        let got = raycast(mesh, bvh, &ray, &opts).unwrap();
        prop_assert_eq!(got.map(|h| (h.triangle, h.t)), all.first().copied());
        let any = raycast_any(mesh, bvh, &ray, &opts).unwrap();
        prop_assert_eq!(any.is_some(), !all.is_empty());
        if let Some(h) = any {
            prop_assert!(all.contains(&(h.triangle, h.t)));
        }
        let mut out = Vec::new();
        raycast_all(mesh, bvh, &ray, &opts, &mut out).unwrap();
        let got: Vec<_> = out.iter().map(|h| (h.triangle, h.t)).collect();
        prop_assert_eq!(got, all);

        let p = [
            rng.range(-2.0, 8.0),
            rng.range(-2.0, 8.0),
            rng.range(-3.0, 4.0),
        ];
        for max_distance in [f32::INFINITY, 1.0] {
            let opts = ClosestOptions { max_distance };
            let got = closest_point(mesh, bvh, p, &opts).unwrap();
            prop_assert_eq!(
                got.map(|h| (h.triangle, h.distance)),
                brute_closest(&singles, p, &opts)
            );
        }

        for sphere in [false, true] {
            let region = random_region(&mut rng, sphere);
            let mut tris = Vec::new();
            triangles_in(mesh, bvh, &region, &mut tris).unwrap();
            tris.sort_unstable();
            prop_assert_eq!(tris, brute_triangles(&singles, &region), "{:?}", region);
            let mut verts = Vec::new();
            vertices_in(mesh, bvh, &region, &mut verts).unwrap();
            prop_assert_eq!(verts, brute_vertices(mesh, &region), "{:?}", region);
        }
    }
    Ok(())
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(cases(64)))]

    #[test]
    fn queries_match_brute_force_after_edits(
        options in options(),
        seed in any::<u64>(),
        steps in prop::collection::vec(step(), 1..12),
    ) {
        let mut m = grid(6);
        let mut bvh = Bvh::build(&m, &options).unwrap();
        let mut h = History::new(usize::MAX);
        check_queries(&m, &bvh, seed)?;
        for (i, step) in steps.iter().enumerate() {
            apply_step(&mut m, &mut bvh, &mut h, step);
            check_queries(&m, &bvh, seed.wrapping_add(i as u64 + 1))?;
        }
    }
}

/// Applies one edit, undo or redo to the mesh, then updates its BVH.
fn apply_step(m: &mut Mesh, bvh: &mut Bvh, h: &mut History, step: &Step) {
    let changes = match step {
        Step::Commit(ops) => {
            let mut tx = m.begin_edit();
            for op in ops {
                run(&mut tx, op);
            }
            let patch = tx.commit().unwrap();
            let changes = patch.changes().clone();
            h.push(m, patch).unwrap();
            Some(changes)
        }
        Step::Undo => h.undo(m).unwrap(),
        Step::Redo => h.redo(m).unwrap(),
    };
    if let Some(changes) = changes {
        bvh.update(m, &changes).unwrap();
    }
}

/// A random rigid transform that puts B's middle near A's middle: a random unit
/// quaternion (or one of a few exact rotations) and a translation with jitter,
/// so that the meshes touch, overlap or sit slightly apart.
fn random_rigid(rng: &mut Lcg) -> Rigid {
    let rotation = if rng.next() < 0.25 {
        // Exact rotations: identity, a flip about x, a quarter turn about z.
        [
            Rigid::IDENTITY.rotation,
            [[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]],
            [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
        ][(rng.next() * 3.0) as usize % 3]
    } else {
        let mut q = [0; 4].map(|_| f64::from(rng.range(-1.0, 1.0)));
        let norm = q.iter().map(|v| v * v).sum::<f64>().sqrt();
        if norm < 1e-3 {
            q = [1.0, 0.0, 0.0, 0.0];
        } else {
            q = q.map(|v| v / norm);
        }
        let [w, x, y, z] = q;
        [
            [
                1.0 - 2.0 * (y * y + z * z),
                2.0 * (x * y - z * w),
                2.0 * (x * z + y * w),
            ],
            [
                2.0 * (x * y + z * w),
                1.0 - 2.0 * (x * x + z * z),
                2.0 * (y * z - x * w),
            ],
            [
                2.0 * (x * z - y * w),
                2.0 * (y * z + x * w),
                1.0 - 2.0 * (x * x + y * y),
            ],
        ]
    };
    // The middles of the two grids, before edits.
    let (ca, cb) = ([3.0, 3.0, 0.6], [2.5, 2.5, 0.6]);
    let jitter = [
        f64::from(rng.range(-6.0, 6.0)),
        f64::from(rng.range(-6.0, 6.0)),
        f64::from(rng.range(-5.0, 5.0)),
    ];
    let translation = std::array::from_fn(|i| {
        let rotated: f64 = (0..3).map(|k| rotation[i][k] * cb[k]).sum();
        ca[i] - rotated + jitter[i]
    });
    Rigid {
        rotation,
        translation,
    }
}

/// One pair of alive triangles, one from each mesh, as the brute force sees it.
struct PairInfo {
    a: u32,
    b: u32,
    /// Whether the triangles intersect or touch.
    hit: bool,
    /// The single-pair `mesh_distance`: the f32 distance and the points.
    dist: MeshDistance,
}

/// Every pair of alive triangles, in ascending `(a, b)`, tested as two
/// one-triangle meshes.
fn pair_table(sa: &[Single], sb: &[Single], xf: &Rigid) -> Vec<PairInfo> {
    let mut out = Vec::with_capacity(sa.len() * sb.len());
    for a in sa {
        for b in sb {
            let mut pairs = Vec::new();
            let n = intersecting_pairs(&a.mesh, &a.bvh, &b.mesh, &b.bvh, xf, &mut pairs).unwrap();
            let dist = mesh_distance(&a.mesh, &a.bvh, &b.mesh, &b.bvh, xf, f32::INFINITY)
                .unwrap()
                .unwrap();
            out.push(PairInfo {
                a: a.id,
                b: b.id,
                hit: n == 1,
                dist,
            });
        }
    }
    out
}

/// Whether a single pair is within `margin`: it touches, or its `f64` distance
/// is at most `margin` (which `mesh_distance` decides exactly as its
/// `max_distance`).
fn pair_is_near(sa: &Single, sb: &Single, xf: &Rigid, margin: f32) -> bool {
    mesh_distance(&sa.mesh, &sa.bvh, &sb.mesh, &sb.bvh, xf, margin)
        .unwrap()
        .is_some()
}

fn near(
    ma: &Mesh,
    ba: &Bvh,
    mb: &Mesh,
    bb: &Bvh,
    xf: &Rigid,
    margin: f32,
) -> (Vec<u32>, Vec<u32>) {
    let (mut a, mut b) = (Vec::new(), Vec::new());
    triangles_near(ma, ba, mb, bb, xf, margin, &mut a, &mut b).unwrap();
    (a, b)
}

/// All four pair queries against the brute force over every pair of alive
/// triangles, under `xf`.
fn check_pair_queries(
    (ma, ba): (&Mesh, &Bvh),
    (mb, bb): (&Mesh, &Bvh),
    xf: &Rigid,
) -> Result<(), TestCaseError> {
    let (sa, sb) = (singles(ma), singles(mb));
    let table = pair_table(&sa, &sb, xf);
    let single = |id: u32, s: &[Single]| s.iter().position(|x| x.id == id).unwrap();

    // intersects and intersecting_pairs.
    let want_pairs: Vec<(u32, u32)> = table
        .iter()
        .filter(|p| p.hit)
        .map(|p| (p.a, p.b))
        .collect();
    prop_assert_eq!(intersects(ma, ba, mb, bb, xf), Ok(!want_pairs.is_empty()));
    let mut out = vec![(999, 999)];
    let n = intersecting_pairs(ma, ba, mb, bb, xf, &mut out).unwrap();
    prop_assert_eq!(n, want_pairs.len());
    prop_assert_eq!(out[0], (999, 999));
    prop_assert_eq!(&out[1..], &want_pairs[..]);

    // mesh_distance: the f32 minimum is the minimum of the f32 pair distances
    // (rounding is monotone), and the pair it names is one of the pairs at that
    // distance, with the same points. Ties inside f32 rounding are not told
    // apart here; the unit tests cover the smallest-pair rule on f64.
    let min = table
        .iter()
        .map(|p| p.dist.distance)
        .min_by(f32::total_cmp);
    let got = mesh_distance(ma, ba, mb, bb, xf, f32::INFINITY).unwrap();
    prop_assert_eq!(got.map(|g| g.distance), min);
    if let (Some(g), Some(min)) = (got, min) {
        let info = &table[single(g.triangle_a, &sa) * sb.len() + single(g.triangle_b, &sb)];
        prop_assert_eq!((info.a, info.b), (g.triangle_a, g.triangle_b));
        prop_assert_eq!(info.dist.distance, min);
        if min > 0.0 {
            // A positive distance comes from the same arithmetic as the pair.
            prop_assert_eq!(info.dist.point_a, g.point_a);
            prop_assert_eq!(info.dist.point_b, g.point_b);
            // max_distance is inclusive.
            let above = mesh_distance(ma, ba, mb, bb, xf, min.next_up()).unwrap();
            prop_assert_eq!(above, got);
            prop_assert_eq!(mesh_distance(ma, ba, mb, bb, xf, min.next_down()), Ok(None));
            // Cross-query consistency.
            let (na, nb) = near(ma, ba, mb, bb, xf, min * (1.0 + 1e-6));
            prop_assert!(na.contains(&g.triangle_a), "{:?}", g);
            prop_assert!(nb.contains(&g.triangle_b), "{:?}", g);
        } else {
            // Zero means touching or closer than rounding: the pair intersects
            // or its computed distance is 0.
            prop_assert!(info.hit || info.dist.distance == 0.0);
            let zero = mesh_distance(ma, ba, mb, bb, xf, 0.0).unwrap();
            prop_assert_eq!(zero.map(|z| z.distance), Some(0.0));
        }
    } else {
        prop_assert_eq!(mesh_distance(ma, ba, mb, bb, xf, 1.0), Ok(None));
    }

    // triangles_near. The f32 distance is a monotone rounding of the f64 one, so
    // a pair farther than the margin in f32 is not near; the rest are decided by
    // the exact single-pair query.
    for margin in [0.0, 0.1, 0.6] {
        let near_pairs: Vec<&PairInfo> = table
            .iter()
            .filter(|p| {
                p.hit
                    || (p.dist.distance <= margin
                        && pair_is_near(
                            &sa[single(p.a, &sa)],
                            &sb[single(p.b, &sb)],
                            xf,
                            margin,
                        ))
            })
            .collect();
        let mut want_a: Vec<u32> = near_pairs.iter().map(|p| p.a).collect();
        let mut want_b: Vec<u32> = near_pairs.iter().map(|p| p.b).collect();
        want_a.sort_unstable();
        want_a.dedup();
        want_b.sort_unstable();
        want_b.dedup();
        let (mut a, mut b) = (vec![999], vec![998]);
        triangles_near(ma, ba, mb, bb, xf, margin, &mut a, &mut b).unwrap();
        prop_assert_eq!((a[0], b[0]), (999, 998));
        prop_assert_eq!(&a[1..], &want_a[..], "margin {}", margin);
        prop_assert_eq!(&b[1..], &want_b[..], "margin {}", margin);
    }
    Ok(())
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(cases(64)))]

    #[test]
    fn pair_queries_match_brute_force_after_edits(
        options_a in options(),
        options_b in options(),
        seed in any::<u64>(),
        steps_a in prop::collection::vec(step(), 1..6),
        steps_b in prop::collection::vec(step(), 1..6),
    ) {
        let (mut ma, mut mb) = (grid(6), grid(5));
        let mut ba = Bvh::build(&ma, &options_a).unwrap();
        let mut bb = Bvh::build(&mb, &options_b).unwrap();
        let (mut ha, mut hb) = (History::new(usize::MAX), History::new(usize::MAX));
        let mut rng = Lcg(seed);
        for _ in 0..2 {
            let xf = random_rigid(&mut rng);
            check_pair_queries((&ma, &ba), (&mb, &bb), &xf)?;
        }
        for i in 0..steps_a.len().max(steps_b.len()) {
            if let Some(s) = steps_a.get(i) {
                apply_step(&mut ma, &mut ba, &mut ha, s);
            }
            if let Some(s) = steps_b.get(i) {
                apply_step(&mut mb, &mut bb, &mut hb, s);
            }
            for _ in 0..2 {
                let xf = random_rigid(&mut rng);
                check_pair_queries((&ma, &ba), (&mb, &bb), &xf)?;
            }
        }
    }
}

/// A closed cube surface of side 2 centered at the origin. Each face is a 4×4
/// grid of quads; vertices on shared edges are welded through a map on integer
/// coordinates, so the mesh is watertight.
fn cube() -> Mesh {
    let mut ids: HashMap<[i32; 3], u32> = HashMap::new();
    let mut positions: Vec<f32> = Vec::new();
    let mut indices: Vec<u32> = Vec::new();
    let mut vertex = |c: [i32; 3]| -> u32 {
        *ids.entry(c).or_insert_with(|| {
            positions.extend(c.map(|v| v as f32 / 2.0));
            (positions.len() / 3 - 1) as u32
        })
    };
    for axis in 0..3 {
        for side in [-2, 2] {
            let (u, v) = ((axis + 1) % 3, (axis + 2) % 3);
            let at = |i: i32, j: i32| {
                let mut c = [0; 3];
                c[axis] = side;
                c[u] = i - 2;
                c[v] = j - 2;
                c
            };
            for i in 0..4 {
                for j in 0..4 {
                    let q = [
                        vertex(at(i, j)),
                        vertex(at(i + 1, j)),
                        vertex(at(i + 1, j + 1)),
                        vertex(at(i, j + 1)),
                    ];
                    indices.extend([q[0], q[1], q[2], q[0], q[2], q[3]]);
                }
            }
        }
    }
    Mesh::from_buffers(&positions, &indices).unwrap()
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(cases(256)))]

    #[test]
    fn rays_never_escape_a_closed_mesh(
        origin in prop::array::uniform3(-0.9f32..0.9),
        dir in prop::array::uniform3(-1.0f32..1.0),
        // A point of the surface in quarter units: even coordinates are
        // vertices, odd ones are edge midpoints and quad centers.
        target in (0usize..3, any::<bool>(), -4i32..=4, -4i32..=4),
        aimed in any::<bool>(),
    ) {
        let mesh = cube();
        prop_assert_eq!(mesh.vertex_count(), 98);
        prop_assert_eq!(mesh.triangle_count(), 192);
        let bvh = Bvh::build(&mesh, &BuildOptions::default()).unwrap();
        let dir = if aimed {
            let (axis, positive, a, b) = target;
            let mut c = [0; 3];
            c[axis] = if positive { 4 } else { -4 };
            c[(axis + 1) % 3] = a;
            c[(axis + 2) % 3] = b;
            [0, 1, 2].map(|k| c[k] as f32 / 4.0 - origin[k])
        } else {
            dir
        };
        prop_assume!(dir != [0.0; 3]);
        let ray = Ray { origin, dir };
        let hit = raycast(&mesh, &bvh, &ray, &RayOptions::default()).unwrap();
        prop_assert!(hit.is_some(), "{:?}", ray);
        prop_assert!(raycast_any(&mesh, &bvh, &ray, &RayOptions::default()).unwrap().is_some());
    }
}
