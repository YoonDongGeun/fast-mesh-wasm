//! The heat method on meshes large enough to need its bands and multigrid
//! levels, and geodesic queries run in steps.

use std::collections::HashMap;
use std::sync::Arc;

use fast_mesh_fields::{
    FieldsError, GeodesicHeatSolver, GeodesicMethod, GeodesicOptions, GeodesicPhase, GeodesicQuery,
    geodesic_distances,
};
use fast_mesh_store::{Mesh, par};

/// A flat grid of `nx` by `ny` square cells.
fn grid(nx: u32, ny: u32, size: f32) -> Mesh {
    let mut p = Vec::new();
    for y in 0..=ny {
        for x in 0..=nx {
            p.extend([x as f32 * size, y as f32 * size, 0.0]);
        }
    }
    let mut t = Vec::new();
    for y in 0..ny {
        for x in 0..nx {
            let a = y * (nx + 1) + x;
            t.extend([a, a + 1, a + nx + 2, a, a + nx + 2, a + nx + 1]);
        }
    }
    Mesh::from_buffers(&p, &t).unwrap()
}

/// The unit icosahedron with its faces split into four `subdivisions` times.
fn icosphere(subdivisions: u32) -> Mesh {
    let g = (1.0 + 5.0f64.sqrt()) / 2.0;
    let unit = |p: [f64; 3]| {
        let l = (p[0] * p[0] + p[1] * p[1] + p[2] * p[2]).sqrt();
        p.map(|c| c / l)
    };
    let mut p: Vec<[f64; 3]> = [
        [-1.0, g, 0.0],
        [1.0, g, 0.0],
        [-1.0, -g, 0.0],
        [1.0, -g, 0.0],
        [0.0, -1.0, g],
        [0.0, 1.0, g],
        [0.0, -1.0, -g],
        [0.0, 1.0, -g],
        [g, 0.0, -1.0],
        [g, 0.0, 1.0],
        [-g, 0.0, -1.0],
        [-g, 0.0, 1.0],
    ]
    .map(unit)
    .to_vec();
    #[rustfmt::skip]
    let mut t: Vec<[u32; 3]> = vec![
        [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
        [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
        [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
        [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1],
    ];
    for _ in 0..subdivisions {
        let mut middle: HashMap<(u32, u32), u32> = HashMap::new();
        let mut next = Vec::with_capacity(4 * t.len());
        for [a, b, c] in t {
            let mut mid = |u: u32, v: u32| {
                *middle.entry((u.min(v), u.max(v))).or_insert_with(|| {
                    let (x, y) = (p[u as usize], p[v as usize]);
                    p.push(unit([x[0] + y[0], x[1] + y[1], x[2] + y[2]]));
                    p.len() as u32 - 1
                })
            };
            let (ab, bc, ca) = (mid(a, b), mid(b, c), mid(c, a));
            next.extend([[a, ab, ca], [b, bc, ab], [c, ca, bc], [ab, bc, ca]]);
        }
        t = next;
    }
    let p: Vec<f32> = p.iter().flatten().map(|&c| c as f32).collect();
    Mesh::from_buffers(&p, t.as_flattened()).unwrap()
}

/// `m` with its vertices and triangles in a scrambled order.
fn scrambled(m: &Mesh) -> Mesh {
    let mut state = 7u64;
    let mut next = move || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        state >> 33
    };
    let n = m.vertex_slots();
    let mut to: Vec<u32> = (0..n as u32).collect();
    for i in (1..n).rev() {
        to.swap(i, next() as usize % (i + 1));
    }
    let mut p = vec![[0.0f32; 3]; n];
    for (v, &k) in to.iter().enumerate() {
        p[k as usize] = m.positions()[v];
    }
    let mut t: Vec<[u32; 3]> = m.triangles().to_vec();
    for tri in &mut t {
        *tri = tri.map(|v| to[v as usize]);
    }
    for i in (1..t.len()).rev() {
        t.swap(i, next() as usize % (i + 1));
    }
    Mesh::from_buffers(p.as_flattened(), t.as_flattened()).unwrap()
}

/// Mean and largest relative error of distances on a unit sphere from
/// vertex `source`, over the vertices at least a quarter radian away.
fn sphere_error(m: &Mesh, source: u32, distances: &[f32]) -> (f64, f64) {
    let at = |v: usize| m.positions()[v].map(f64::from);
    let a = at(source as usize);
    let (mut sum, mut worst, mut count) = (0.0, 0.0f64, 0);
    for (v, &d) in distances.iter().enumerate() {
        let b = at(v);
        let cross = [
            a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0],
        ];
        let sin = cross.iter().map(|c| c * c).sum::<f64>().sqrt();
        let exact = sin.atan2(a[0] * b[0] + a[1] * b[1] + a[2] * b[2]);
        if exact > 0.25 {
            let e = (f64::from(d) - exact).abs() / exact;
            sum += e;
            worst = worst.max(e);
            count += 1;
        }
    }
    (sum / f64::from(count), worst)
}

fn options(method: GeodesicMethod) -> GeodesicOptions {
    GeodesicOptions {
        method,
        ..Default::default()
    }
}

fn bits(d: &[f32]) -> Vec<u32> {
    d.iter().map(|x| x.to_bits()).collect()
}

/// Runs a query in steps of `work`; its distances and the phases it went through.
fn run(mut q: GeodesicQuery, m: &Mesh, work: usize) -> (Vec<u32>, Vec<GeodesicPhase>, usize) {
    let mut phases = vec![q.phase()];
    let mut steps = 0;
    loop {
        steps += 1;
        assert!(q.distances().is_none());
        let done = q.step(m, work).unwrap();
        if phases.last() != Some(&q.phase()) {
            phases.push(q.phase());
        }
        if done {
            break;
        }
    }
    // Done stays done.
    assert!(q.step(m, work).unwrap());
    let d = bits(q.distances().unwrap());
    assert_eq!(bits(&q.into_distances().unwrap()), d);
    (d, phases, steps)
}

#[test]
fn errors_shrink_as_the_mesh_is_refined() {
    let mut last = f64::INFINITY;
    for subdivisions in [4, 5, 6] {
        let m = icosphere(subdivisions);
        for source in [0, m.vertex_slots() as u32 / 3] {
            let mut d = Vec::new();
            geodesic_distances(&m, &[source], &GeodesicOptions::default(), &mut d).unwrap();
            let (mean, worst) = sphere_error(&m, source, &d);
            assert!(mean < 0.012, "{subdivisions}, {source}: mean error {mean}");
            assert!(worst < 0.06, "{subdivisions}, {source}: error {worst}");
            if source == 0 {
                assert!(mean < last, "{subdivisions}: mean {mean} after {last}");
                last = mean;
            }
        }
    }
    assert!(last < 0.006, "mean {last}");
}

#[test]
fn heat_reaches_past_the_range_of_f64() {
    // 4000 cells from the source: heat falls by far more than e^700.
    let m = grid(4000, 2, 0.5);
    let mut d = Vec::new();
    geodesic_distances(&m, &[0], &GeodesicOptions::default(), &mut d).unwrap();
    for (v, &got) in d.iter().enumerate() {
        let p = m.positions()[v];
        let exact = p[0].hypot(p[1]);
        assert!(
            (got - exact).abs() < 0.01 * exact + 1.0,
            "{p:?}: {got} instead of {exact}"
        );
    }
}

#[test]
fn how_a_query_is_cut_changes_nothing() {
    let m = scrambled(&icosphere(4));
    let sources = [5, 900];
    for method in [GeodesicMethod::Dijkstra, GeodesicMethod::HeatMethod] {
        let o = options(method);
        let mut whole = Vec::new();
        geodesic_distances(&m, &sources, &o, &mut whole).unwrap();
        for work in [1, 700, 50_000, usize::MAX] {
            let q = GeodesicQuery::new(&m, &sources, &o).unwrap();
            assert!(q.is_for(&m));
            let (d, phases, steps) = run(q, &m, work);
            assert_eq!(d, bits(&whole), "{method:?} in steps of {work}");
            match work {
                1 => assert!(steps > 20, "{method:?}: {steps} steps"),
                usize::MAX => assert_eq!(steps, 1),
                _ => {}
            }
            if work == 1 && method == GeodesicMethod::HeatMethod {
                use GeodesicPhase::*;
                assert_eq!(
                    phases,
                    [
                        Assemble, Factorize, Diffuse, Gradient, Poisson, Diffuse, Done
                    ]
                );
            }
        }
    }
}

#[test]
fn operators_are_reused() {
    let m = icosphere(4);
    let o = options(GeodesicMethod::HeatMethod);
    let mut first = GeodesicQuery::new(&m, &[0], &o).unwrap();
    assert!(first.solver().is_none());
    while !first.step(&m, 1).unwrap() {}
    let solver = Arc::clone(first.solver().unwrap());
    assert!(solver.is_for(&m));
    assert!(solver.byte_size() > 0);

    let near = GeodesicOptions {
        max_distance: Some(0.5),
        time_step_scale: 3.0,
        ..o
    };
    for (o, sources) in [(o, vec![77]), (near, vec![3, 400])] {
        let mut fresh = Vec::new();
        geodesic_distances(&m, &sources, &o, &mut fresh).unwrap();
        let q = GeodesicQuery::with_solver(&m, &sources, &o, Arc::clone(&solver)).unwrap();
        let (d, phases, _) = run(q, &m, 1);
        assert_eq!(d, bits(&fresh));
        assert!(!phases.contains(&GeodesicPhase::Assemble));
    }
    // The solver itself, with a time step of its own.
    let mut own = GeodesicHeatSolver::build(&m, 1.0).unwrap();
    own.set_time_step_scale(3.0).unwrap();
    assert_eq!(own.time_step_scale(), 3.0);
    assert!(own.set_time_step_scale(0.0).is_err());
    let (mut a, mut b) = (Vec::new(), Vec::new());
    own.solve(&[9], None, &mut a).unwrap();
    GeodesicHeatSolver::build(&m, 3.0)
        .unwrap()
        .solve(&[9], None, &mut b)
        .unwrap();
    assert_eq!(bits(&a), bits(&b));
}

#[test]
fn a_query_belongs_to_its_mesh_state() {
    let mut m = icosphere(3);
    let o = options(GeodesicMethod::HeatMethod);
    let mut started = GeodesicQuery::new(&m, &[0], &o).unwrap();
    started.step(&m, 1).unwrap();
    let solver = Arc::new(GeodesicHeatSolver::build(&m, 1.0).unwrap());

    let moved = m.positions()[7].map(|c| c * 1.1);
    let mut tx = m.begin_edit();
    tx.set_positions(&[7], &[moved]).unwrap();
    let _ = tx.commit().unwrap();
    assert!(!started.is_for(&m));
    assert!(matches!(
        started.step(&m, 1),
        Err(FieldsError::InvalidInput(_))
    ));
    // Operators of another state are built again.
    assert!(!solver.is_for(&m));
    let q = GeodesicQuery::with_solver(&m, &[0], &o, solver).unwrap();
    assert_eq!(q.phase(), GeodesicPhase::Assemble);
    let mut fresh = Vec::new();
    geodesic_distances(&m, &[0], &o, &mut fresh).unwrap();
    assert_eq!(run(q, &m, 1).0, bits(&fresh));
}

#[test]
fn every_part_is_solved_on_its_own() {
    // A large part between many small ones.
    let (large, small) = (grid(30, 30, 1.0), grid(2, 3, 1.0));
    let mut p: Vec<[f32; 3]> = Vec::new();
    let mut t: Vec<[u32; 3]> = Vec::new();
    let mut firsts = Vec::new();
    for k in 0..40u32 {
        let part = if k == 20 { &large } else { &small };
        let base = p.len() as u32;
        firsts.push(base);
        p.extend(
            part.positions()
                .iter()
                .map(|v| [v[0], v[1], v[2] + k as f32]),
        );
        t.extend(part.triangles().iter().map(|tri| tri.map(|v| v + base)));
    }
    let m = Mesh::from_buffers(p.as_flattened(), t.as_flattened()).unwrap();
    let mut d = Vec::new();
    let sources = [firsts[3], firsts[20], firsts[39]];
    geodesic_distances(&m, &sources, &GeodesicOptions::default(), &mut d).unwrap();
    for (k, &first) in firsts.iter().enumerate() {
        let part = if k == 20 { &large } else { &small };
        let got = &d[first as usize..][..part.vertex_slots()];
        if sources.contains(&first) {
            // Straight from its first corner.
            for (d, p) in got.iter().zip(part.positions()) {
                let want = p[0].hypot(p[1]);
                let near = 0.25 + 0.01 * want;
                assert!((d - want).abs() < near, "part {k}: {d} instead of {want}");
            }
        } else {
            assert!(got.iter().all(|x| x.is_infinite()), "part {k}");
        }
    }
}

#[test]
fn distances_do_not_depend_on_threads() {
    // Enough vertices for every parallel loop.
    let m = scrambled(&icosphere(6));
    let query = || {
        let mut d = Vec::new();
        geodesic_distances(&m, &[1, 20_000], &GeodesicOptions::default(), &mut d).unwrap();
        bits(&d)
    };
    par::set_force_sequential(true);
    let sequential = query();
    par::set_force_sequential(false);
    for threads in [1, 2, 7] {
        assert_eq!(par::with_threads(threads, query), sequential, "{threads}");
    }
}

#[test]
fn any_soup_of_triangles_gets_distances() {
    // Random corners: slivers, fans, folded and repeated triangles, edges
    // with many triangles, at any scale. (Cut into some thousands each.)
    let mut state = 1u64;
    let mut next = move || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (state >> 33) as u32
    };
    for round in 0..40 {
        let n = 4 + next() % 400;
        let scale = 10f32.powi(next() as i32 % 25 - 12);
        let flat = if round % 3 == 0 { 1e-4 } else { 1.0 };
        let p: Vec<[f32; 3]> = (0..n)
            .map(|_| {
                let c = |v: u32| (v % 20_001) as f32 / 1e4 - 1.0;
                [
                    c(next()) * scale,
                    c(next()) * scale * flat,
                    c(next()) * scale,
                ]
            })
            .collect();
        let near = 1 + next() % n;
        let t: Vec<[u32; 3]> = (0..1 + next() % 900)
            .map(|_| {
                let a = next() % n;
                [a, (a + next() % near) % n, (a + next() % near) % n]
            })
            .collect();
        let m = Mesh::from_buffers(p.as_flattened(), t.as_flattened()).unwrap();
        let sources = [next() % n, next() % n];
        for time_step_scale in [1.0, 1e-3, 1e4] {
            let o = GeodesicOptions {
                time_step_scale,
                ..Default::default()
            };
            let mut d = Vec::new();
            geodesic_distances(&m, &sources, &o, &mut d).unwrap();
            assert_eq!(d.len(), n as usize);
            for (v, x) in d.iter().enumerate() {
                assert!(*x >= 0.0, "round {round}, {time_step_scale}, {v}: {x}");
            }
            for s in sources {
                assert_eq!(d[s as usize], 0.0, "round {round}");
            }
        }
    }
}
