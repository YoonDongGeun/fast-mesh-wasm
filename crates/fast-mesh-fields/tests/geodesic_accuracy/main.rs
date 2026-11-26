//! How far off heat-method distances are, against exact ones, on meshes
//! whose triangles make it hard: stretched, obtuse, fans, or of all sizes.

mod exact;
mod shapes;

use std::f64::consts::{PI, TAU};

use exact::exact_distances;
use fast_mesh_fields::{
    GeodesicHeatSolver, GeodesicMethod, GeodesicOptions, GeodesicPhase, GeodesicQuery,
    geodesic_distances,
};
use fast_mesh_store::{Mesh, par};
use shapes::*;

/// The bound documented with `GeodesicMethod::HeatMethod`: the mean relative
/// error (of the distances above a tenth of the largest), ...
const MEAN: f64 = 0.03;
/// ... and the largest error, relative to the largest distance.
const LARGEST: f64 = 0.04;

fn mesh(s: &Shape) -> Mesh {
    let p: Vec<[f32; 3]> = s.positions.iter().map(|p| p.map(|c| c as f32)).collect();
    Mesh::from_buffers(p.as_flattened(), s.triangles.as_flattened()).unwrap()
}

fn heat(m: &Mesh, sources: &[u32]) -> Vec<f64> {
    let mut d = Vec::new();
    geodesic_distances(m, sources, &GeodesicOptions::default(), &mut d).unwrap();
    d.into_iter().map(f64::from).collect()
}

fn distance(a: [f64; 3], b: [f64; 3]) -> f64 {
    (0..3).map(|k| (a[k] - b[k]).powi(2)).sum::<f64>().sqrt()
}

/// The diffusion's length as it is documented: the mean edge length, or
/// `sqrt(area / 2000)` where that is smaller.
fn diffusion_length(s: &Shape) -> f64 {
    let mut edges: Vec<(u32, u32)> = Vec::new();
    let mut area = 0.0;
    for &[a, b, c] in &s.triangles {
        for (a, b) in [(a, b), (b, c), (c, a)] {
            edges.push((a.min(b), a.max(b)));
        }
        let [a, b, c] = [a, b, c].map(|v| s.positions[v as usize]);
        let (u, v) = (distance(a, b), distance(a, c));
        let dot: f64 = (0..3).map(|k| (b[k] - a[k]) * (c[k] - a[k])).sum();
        area += 0.5 * (u * u * v * v - dot * dot).max(0.0).sqrt();
    }
    edges.sort_unstable();
    edges.dedup();
    let length = |&(a, b): &(u32, u32)| distance(s.positions[a as usize], s.positions[b as usize]);
    let mean = edges.iter().map(length).sum::<f64>() / edges.len() as f64;
    mean.min((area / 2000.0).sqrt())
}

#[derive(Debug)]
struct Error {
    /// Mean and largest, relative, of the distances above a tenth of the
    /// largest.
    mean: f64,
    worst: f64,
    /// The largest one, relative to the largest distance.
    largest: f64,
    /// The largest one.
    absolute: f64,
}

fn error(exact: &[f64], got: &[f64]) -> Error {
    let far = exact.iter().filter(|d| d.is_finite());
    let far = far.fold(0.0f64, |a, &b| a.max(b));
    let (mut sum, mut count) = (0.0, 0);
    let (mut worst, mut absolute) = (0.0f64, 0.0f64);
    for (&want, &got) in exact.iter().zip(got) {
        if want == f64::INFINITY && got == want {
            continue;
        }
        assert!(got.is_finite() && got >= 0.0);
        let off = (got - want).abs();
        absolute = absolute.max(off);
        if want > 0.1 * far {
            sum += off / want;
            worst = worst.max(off / want);
            count += 1;
        }
    }
    Error {
        mean: sum / f64::from(count),
        worst,
        largest: absolute / far,
        absolute,
    }
}

/// The reference meshes, and where the distances are from.
fn reference() -> Vec<(&'static str, Shape, Vec<u32>)> {
    let mut set: Vec<(&'static str, Shape, Vec<u32>)> = Vec::new();
    let mut add = |name, s: Shape, sources: &[u32]| set.push((name, s.rounded(), sources.to_vec()));
    let sphere = |s: u32| {
        let i = icosphere(s);
        let sources = [0, i.furthest([0.0, 1.0, 0.0]), i.furthest([1.0, 0.3, 0.2])];
        (i, sources)
    };
    // Well shaped, for comparison.
    let (s, from) = sphere(4);
    add("sphere", s.clone(), &from);
    // Stretched.
    let by = |y: f64| move |p: [f64; 3]| [p[0], y * p[1], p[2]];
    add("sphere, 5x", s.clone().mapped(by(5.0)), &from);
    add("sphere, 20x", s.clone().mapped(by(20.0)), &from);
    add("sphere, 100x", s.clone().mapped(by(100.0)), &from);
    add("sphere, flat", s.clone().mapped(by(0.05)), &from);
    let skewed = |p: [f64; 3]| [p[0] + 8.0 * p[1], 8.0 * p[1] - p[2], p[2]];
    add("sphere, skewed", s.mapped(skewed), &from);
    let (s, from) = sphere(3);
    add("small sphere", s.clone(), &from);
    add("small sphere, 20x", s.mapped(by(20.0)), &from);
    // Capped cylinders.
    add("cylinder", fan_cylinder(128, 64, 2.0), &[0, 4096, 8321]);
    add(
        "cylinder, few rings",
        fan_cylinder(128, 16, 2.0),
        &[0, 1024, 2177],
    );
    add("cylinder, one ring", fan_cylinder(64, 1, 2.0), &[0, 129]);
    add(
        "cylinder, short",
        fan_cylinder(256, 4, 0.2),
        &[0, 512, 1281],
    );
    add("cylinder, coarse", fan_cylinder(12, 2, 3.0), &[0, 12, 37]);
    // Thin fans.
    add("disc", disc(64), &[0, 1]);
    add("disc, fine", disc(512), &[0, 1]);
    add("fan", fan_strip(64, 8.0, 1.0), &[0, 1, 2, 34]);
    add("fan, long", fan_strip(128, 1.0, 10.0), &[0, 1, 2, 66]);
    // Obtuse everywhere, with boundaries.
    let sheared = |by: f64| move |p: [f64; 3]| [p[0] + by * p[1], p[1], p[2]];
    let g = || grid(30, 30, 1.0, 1.0);
    add("sheared", g().mapped(sheared(6.0)), &[0, 30, 480]);
    add("sheared back", g().mapped(sheared(-6.0)), &[0, 30, 480]);
    add("notch", notched(24).mapped(sheared(3.0)), &[0, 24, 400]);
    add("cells", grid(20, 40, 20.0, 1.0), &[0, 10, 430]);
    add("tube", tube(24, 12, 30.0), &[0, 150]);
    // Of all sizes.
    let graded = graded_sphere(4, 3.0);
    let from = [[0.0, 0.0, 1.0], [0.0, 0.0, -1.0], [1.0, 0.0, 0.0]];
    let from = from.map(|d| graded.furthest(d));
    add("graded", graded, &from);
    add("jittered", jittered_sphere(4, 0.05), &[0, 77, 2000]);
    add("rings", uv_sphere(64, 32), &[0, 961, 257]);
    set
}

#[test]
fn distances_are_within_the_documented_bound() {
    let mut bad = 0;
    for (name, s, sources) in reference() {
        let m = mesh(&s);
        let solver = GeodesicHeatSolver::build(&m, 1.0).unwrap();
        let length = diffusion_length(&s);
        for source in sources {
            let exact = exact_distances(&s.positions, &s.triangles, &[source]);
            let mut d = Vec::new();
            solver.solve(&[source], None, &mut d).unwrap();
            let d: Vec<f64> = d.into_iter().map(f64::from).collect();
            let e = error(&exact, &d);
            let ok = e.mean < MEAN && e.largest < LARGEST;
            bad += usize::from(!ok);
            println!(
                "{name:20} {:5} triangles, from {source:4}: mean {:.2} %, worst {:4.1} %, \
                 largest {:.2} % or {:.2} lengths{}",
                s.triangles.len(),
                100.0 * e.mean,
                100.0 * e.worst,
                100.0 * e.largest,
                e.absolute / length,
                if ok { "" } else { " !" }
            );
            assert_eq!(d[source as usize], 0.0);
        }
    }
    assert_eq!(bad, 0);
}

#[test]
fn stretched_spheres_match_their_meridians() {
    // The way from the tip of a spheroid is along a meridian.
    let mut last = f64::INFINITY;
    for subdivisions in [3, 4, 5] {
        let s = icosphere(subdivisions).mapped(|p| [p[0], 20.0 * p[1], p[2]]);
        let from = s.furthest([0.0, 1.0, 0.0]);
        // The tip is a vertex: an edge's middle.
        assert!((s.positions[from as usize][1] - 20.0).abs() < 1e-9);
        let exact: Vec<f64> = s
            .positions
            .iter()
            .map(|p| {
                let to = p[0].hypot(p[2]).atan2(p[1] / 20.0);
                let speed = |t: f64| t.cos().hypot(20.0 * t.sin());
                let steps = 400;
                let h = to / f64::from(steps);
                (0..steps)
                    .map(|k| {
                        let t = h * f64::from(k);
                        h / 6.0 * (speed(t) + 4.0 * speed(t + 0.5 * h) + speed(t + h))
                    })
                    .sum()
            })
            .collect();
        let e = error(&exact, &heat(&mesh(&s), &[from]));
        println!("{subdivisions}: {e:?}");
        assert!(e.mean < 0.02 && e.largest < 0.02, "{subdivisions}: {e:?}");
        assert!(e.largest < last, "{subdivisions}: {e:?} after {last}");
        last = e.largest;
    }
    assert!(last < 0.01);
}

/// The distance of two points of the pages of a book with a spine of some
/// length from each other, unfolded.
fn over_the_spine(a: [f64; 3], b: [f64; 3], length: f64) -> f64 {
    let (ra, rb) = (a[0].hypot(a[1]), b[0].hypot(b[1]));
    let same = ra < 1e-9 || rb < 1e-9 || (a[0] * b[0] + a[1] * b[1]) / (ra * rb) > 0.999;
    if same {
        return distance(a, b);
    }
    // Where the straight line crosses, or the end of the spine nearest to that.
    let z = (a[2] * rb + b[2] * ra) / (ra + rb);
    let at = [0.0, 0.0, z.clamp(0.0, length)];
    distance(a, at) + distance(at, b)
}

#[test]
fn edges_with_many_triangles_are_no_obstacle() {
    for (pages, shear) in [(3, 0.0), (3, 4.0), (5, -4.0)] {
        let s = book(pages, 16, 16, 0.1, 1.0, shear);
        let m = mesh(&s);
        for from in [8, 17 + 17 * 5 + 8, 17 + 17 * 16 + 17 * 9 + 2] {
            let d = heat(&m, &[from]);
            let a = s.positions[from as usize];
            let exact = s.positions.iter().map(|&b| over_the_spine(a, b, 16.0));
            let exact: Vec<f64> = exact.collect();
            let e = error(&exact, &d);
            println!("{pages} pages sheared by {shear}, from {from}: {e:?}");
            assert!(
                e.mean < MEAN && e.largest < LARGEST,
                "{pages} pages sheared by {shear}, from {from}: {e:?}"
            );
        }
    }
}

#[test]
fn the_triangles_order_and_winding_hardly_matter() {
    let s = icosphere(4).mapped(|p| [p[0], 9.0 * p[1], p[2]]).rounded();
    let exact = exact_distances(&s.positions, &s.triangles, &[30]);
    let mut wound = s.clone();
    for t in wound.triangles.iter_mut().step_by(3) {
        t.swap(0, 1);
    }
    for (name, s) in [("shuffled", s.clone().shuffled()), ("wound", wound)] {
        let e = error(&exact, &heat(&mesh(&s), &[30]));
        println!("{name}: {e:?}");
        assert!(e.mean < MEAN && e.largest < LARGEST, "{name}: {e:?}");
    }
}

#[test]
fn several_sources_and_parts() {
    // A stretched sphere, a fan and a strip, far apart.
    let parts = [
        icosphere(3).mapped(|p| [p[0], 12.0 * p[1], p[2]]),
        disc(80),
        grid(12, 12, 1.0, 1.0).mapped(|p| [p[0] + 7.0 * p[1], p[1], 0.0]),
    ];
    let mut all = Shape {
        positions: Vec::new(),
        triangles: Vec::new(),
    };
    let mut firsts = Vec::new();
    for (k, part) in parts.iter().enumerate() {
        let base = all.positions.len() as u32;
        firsts.push(base);
        let x = 100.0 * k as f64;
        let moved = part.positions.iter().map(|p| [p[0] + x, p[1], p[2]]);
        all.positions.extend(moved);
        all.triangles
            .extend(part.triangles.iter().map(|t| t.map(|v| v + base)));
    }
    let all = all.rounded();
    let m = mesh(&all);
    let sources = [firsts[0] + 4, firsts[0] + 300, firsts[2] + 80];
    let exact = exact_distances(&all.positions, &all.triangles, &sources);
    let mut d = Vec::new();
    geodesic_distances(&m, &sources, &GeodesicOptions::default(), &mut d).unwrap();
    for s in sources {
        assert_eq!(d[s as usize], 0.0);
    }
    for (k, &first) in firsts.iter().enumerate() {
        let range = first as usize..first as usize + parts[k].positions.len();
        if k == 1 {
            assert!(d[range].iter().all(|d| d.is_infinite()));
            continue;
        }
        let got: Vec<f64> = d[range.clone()].iter().copied().map(f64::from).collect();
        let e = error(&exact[range], &got);
        println!("part {k}: {e:?}");
        assert!(e.mean < 0.05 && e.largest < 0.05, "part {k}: {e:?}");
    }
    // Nothing above a limit.
    let near = GeodesicOptions {
        max_distance: Some(0.7),
        ..Default::default()
    };
    let mut limited = Vec::new();
    geodesic_distances(&m, &sources, &near, &mut limited).unwrap();
    for (a, b) in d.iter().zip(&limited) {
        let want = if *a > 0.7 { f32::INFINITY } else { *a };
        assert_eq!(want.to_bits(), b.to_bits());
    }
}

#[test]
fn threads_and_steps_change_nothing_on_bad_triangles() {
    // Enough for every parallel loop, flips and new vertices.
    let sphere = icosphere(6).mapped(|p| [p[0], 4.0 * p[1], p[2]]);
    let sphere = mesh(&sphere.shuffled());
    let cylinder = mesh(&fan_cylinder(300, 60, 3.0).shuffled());
    for (m, sources) in [(&sphere, [1, 20_000]), (&cylinder, [7, 9_000])] {
        let query = || {
            let mut d = Vec::new();
            geodesic_distances(m, &sources, &GeodesicOptions::default(), &mut d).unwrap();
            d.into_iter().map(f32::to_bits).collect::<Vec<_>>()
        };
        par::set_force_sequential(true);
        let sequential = query();
        par::set_force_sequential(false);
        for threads in [1, 2, 7] {
            assert_eq!(par::with_threads(threads, query), sequential, "{threads}");
        }
        let mut q = GeodesicQuery::new(m, &sources, &GeodesicOptions::default()).unwrap();
        let (mut steps, mut assembling) = (0, 0);
        loop {
            assembling += usize::from(q.phase() == GeodesicPhase::Assemble);
            steps += 1;
            if q.step(m, 1).unwrap() {
                break;
            }
        }
        // The edges are flipped, and the vertices put in, in steps as well.
        assert!(steps > 30 && assembling > 8, "{steps}, {assembling}");
        let stepped = q.into_distances().unwrap();
        let stepped: Vec<u32> = stepped.into_iter().map(f32::to_bits).collect();
        assert_eq!(stepped, sequential);
    }
}

/// Meshes that are hardly surfaces.
fn oddities() -> Vec<(String, Shape)> {
    let mut state = 5u64;
    let mut random = move || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (state >> 33) as f64 / (1u64 << 31) as f64
    };
    let mut shapes: Vec<(String, Shape)> = Vec::new();
    for k in [1e-3, 1e-5, 1e-7, 1e-10] {
        let mut add = |name: &str, s: Shape| shapes.push((format!("{name} {k}"), s));
        let s = icosphere(3);
        add("flat", grid(12, 12, 1.0, k));
        add("thin", s.clone().mapped(|p| [p[0], p[1], k * p[2]]));
        add("hair", s.clone().mapped(|p| [k * p[0], p[1], k * p[2]]));
        let sheared = |p: [f64; 3]| [p[0] + 3e2 * p[1], p[1], 0.0];
        add("sheared", grid(9, 9, 1.0, k).mapped(sheared));
        let small = jittered_sphere(2, 1.5).mapped(|p| p.map(|c| k * c));
        add("small", small);
        let large = fan_cylinder(9, 2, 1.0).mapped(|p| p.map(|c| c / k));
        add("large", large);
        add("line", disc(30).mapped(|p| [p[0], k * p[1], 0.0]));
        // A few triangles much larger than the rest.
        let mut spike = s;
        spike.positions[200] = [0.0, 1.0 / k, 0.0];
        add("spike", spike);
    }
    let mut add = |name: &str, s: Shape| shapes.push((name.into(), s));
    // Triangles twice, inside out, and folded back onto each other.
    let mut twice = icosphere(2).mapped(|p| [p[0], 7.0 * p[1], p[2]]);
    let again = twice.triangles.iter().step_by(3);
    let again: Vec<_> = again.map(|&[a, b, c]| [c, b, a]).collect();
    twice.triangles.extend(again);
    add("twice", twice);
    let pillow = Shape {
        positions: vec![[0.0, 0.0, 0.0], [5.0, 0.0, 0.0], [2.0, 0.1, 0.0]],
        triangles: vec![[0, 1, 2], [1, 0, 2]],
    };
    add("pillow", pillow);
    let one = Shape {
        positions: vec![[0.0, 0.0, 0.0], [9.0, 0.0, 0.0], [4.0, 1e-4, 0.0]],
        triangles: vec![[0, 1, 2]],
    };
    add("one", one);
    // Around the tip there is next to no angle: edges lead back to their start.
    let cone = Shape {
        positions: vec![
            [0.0, 0.0, 1.0],
            [0.1, 0.0, 0.0],
            [0.0, 0.1, 0.0],
            [-0.1, -0.1, 0.0],
        ],
        triangles: vec![[0, 1, 2], [0, 2, 3], [0, 3, 1]],
    };
    add("cone", cone.clone());
    let closed = Shape {
        triangles: vec![[0, 1, 2], [0, 2, 3], [0, 3, 1], [3, 2, 1]],
        ..cone
    };
    add("needle", closed);
    // A band with a twist has no two sides.
    let mut band = grid(40, 2, 1.0, 1.0).mapped(|p| {
        let (a, r) = (TAU * p[0], p[1] - 0.5);
        let out = 2.0 + r * (a / 2.0).cos();
        [out * a.cos(), out * a.sin(), r * (a / 2.0).sin()]
    });
    for t in &mut band.triangles {
        *t = t.map(|v| if v % 41 == 40 { (2 - v / 41) * 41 } else { v });
    }
    add("twisted", band);
    // A thousand triangles at one edge.
    let mut many = Shape {
        positions: vec![[0.0, 0.0, 0.0], [0.0, 0.0, 10.0]],
        triangles: Vec::new(),
    };
    for k in 0..1000 {
        let a = TAU * random();
        many.positions
            .push([0.1 * a.cos(), 0.1 * a.sin(), 1.0 + 8.0 * random()]);
        many.triangles.push([0, 1, 2 + k]);
    }
    add("many", many);
    // Random triangles between random points of a thin box.
    for round in 0..8 {
        let n = 20 + (200.0 * random()) as u32;
        let positions = (0..n).map(|_| [random(), 1e-2 * random(), 30.0 * random()]);
        let positions = positions.collect();
        let mut corner = || ((random() * f64::from(n)) as u32).min(n - 1);
        let triangles = (0..2 * n).map(|_| [corner(), corner(), corner()]).collect();
        let s = Shape {
            positions,
            triangles,
        };
        add(&format!("random {round}"), s);
    }
    shapes
}

/// The furthest distance along the edges.
fn along_the_edges(m: &Mesh, source: u32) -> f32 {
    let o = GeodesicOptions {
        method: GeodesicMethod::Dijkstra,
        ..Default::default()
    };
    let mut d = Vec::new();
    geodesic_distances(m, &[source], &o, &mut d).unwrap();
    let near = d.into_iter().filter(|x| x.is_finite());
    near.fold(0.0, f32::max)
}

#[test]
fn next_to_flat_triangles_are_no_obstacle() {
    // The graph's way to the far end is straight, or nearly: its distance
    // is near the exact one.
    for (name, s) in oddities() {
        let flat = ["flat", "hair", "line", "spike"];
        let k = name.split_once(' ').map_or("", |(_, k)| k);
        if !flat.iter().any(|f| name.starts_with(f)) || !["0.001", "0.00001"].contains(&k) {
            continue;
        }
        let m = mesh(&s);
        let d = heat(&m, &[0]);
        let far = d.into_iter().fold(0.0, f64::max);
        let want = f64::from(along_the_edges(&m, 0));
        assert!(
            (far - want).abs() < 0.03 * want,
            "{name}: {far} instead of {want}"
        );
    }
}

#[test]
fn any_triangles_get_distances() {
    for (name, s) in oddities() {
        let m = mesh(&s);
        let n = s.positions.len() as u32;
        let mut solver = GeodesicHeatSolver::build(&m, 1.0).unwrap();
        for (scale, source) in [(1.0, 0), (1.0, n - 1), (1e-3, n / 2), (1e2, n / 2)] {
            solver.set_time_step_scale(scale).unwrap();
            let mut d = Vec::new();
            solver.solve(&[source], None, &mut d).unwrap();
            assert_eq!(d.len(), n as usize, "{name}");
            assert_eq!(d[source as usize], 0.0, "{name}");
            assert!(d.iter().all(|x| *x >= 0.0), "{name}");
        }
    }
}

#[test]
fn the_reference_is_exact() {
    let check = |name: &str, s: &Shape, from: u32, want: &dyn Fn([f64; 3]) -> f64, near: f64| {
        let d = exact_distances(&s.positions, &s.triangles, &[from]);
        for (&got, &p) in d.iter().zip(&s.positions) {
            let want = want(p);
            assert!(
                got == want || (got - want).abs() <= near,
                "{name} from {from}: {got} instead of {want} at {p:?}"
            );
        }
    };
    // Flat and convex: straight lines, however it is cut.
    let sheared = |p: [f64; 3]| [p[0] + 6.0 * p[1], p[1], p[2]];
    for (name, s) in [
        ("grid", grid(9, 7, 1.0, 2.0)),
        ("sheared", grid(11, 11, 1.0, 1.0).mapped(sheared)),
        ("disc", disc(40)),
        ("fan", fan_strip(40, 3.0, 1.0)),
    ] {
        for from in [0, 1, s.positions.len() as u32 - 1] {
            let a = s.positions[from as usize];
            check(name, &s, from, &|p| distance(a, p), 1e-9);
        }
    }
    // Around the end of a notch.
    let s = notched(16);
    let (u, v) = ([6.0 / 16.0, 0.5, 0.0], [10.0 / 16.0, 0.5, 0.0]);
    check(
        "notch",
        &s,
        0,
        &|p| {
            let a = [0.0; 3];
            // Straight, around the first corner, or around both.
            if p[0] <= u[0] || p[1] * u[0] >= u[1] * p[0] - 1e-12 {
                distance(a, p)
            } else if p[1] >= 0.5 {
                distance(a, u) + distance(u, p)
            } else if p[0] >= v[0] {
                distance(a, u) + distance(u, v) + distance(v, p)
            } else {
                // (Cut out.)
                f64::INFINITY
            }
        },
        1e-9,
    );
    // The side of a prism unrolls, either way around.
    let (segments, s) = (14, tube(14, 5, 3.0));
    let side = 2.0 * (PI / f64::from(segments)).sin();
    check(
        "tube",
        &s,
        0,
        &|p| {
            let turns = p[1].atan2(p[0]) / TAU * f64::from(segments);
            (turns.abs() * side).hypot(p[2])
        },
        1e-9,
    );
    // The far corners of a cube are √5 edges apart, two faces unfolded.
    let cube = Shape {
        positions: (0..8)
            .map(|i| [f64::from(i & 1), f64::from(i >> 1 & 1), f64::from(i >> 2)])
            .collect(),
        #[rustfmt::skip]
        triangles: vec![
            [0, 2, 3], [0, 3, 1], [4, 5, 7], [4, 7, 6], [0, 1, 5], [0, 5, 4],
            [2, 6, 7], [2, 7, 3], [0, 4, 6], [0, 6, 2], [1, 3, 7], [1, 7, 5],
        ],
    };
    let d = exact_distances(&cube.positions, &cube.triangles, &[0]);
    assert!((d[7] - 5.0f64.sqrt()).abs() < 1e-12, "{}", d[7]);
    assert!((d[3] - 2.0f64.sqrt()).abs() < 1e-12, "{}", d[3]);
    // Around a point with more than a full turn, ways that are more than half
    // a turn apart meet at it.
    let n = 12;
    let mut saddle = disc(n);
    for (k, p) in saddle.positions.iter_mut().enumerate().skip(1) {
        p[2] = if k % 2 == 0 { 0.8 } else { -0.8 };
    }
    let edge = distance(saddle.positions[1], saddle.positions[2]);
    let r = distance(saddle.positions[0], saddle.positions[1]);
    let angle = 2.0 * (0.5 * edge / r).asin();
    assert!(angle * f64::from(n) > 1.2 * TAU);
    let d = exact_distances(&saddle.positions, &saddle.triangles, &[1]);
    for k in 1..n {
        let turn = angle * f64::from(k.min(n - k));
        let want = if turn < PI {
            // Straight, or along the rim's corners where that bends outwards.
            2.0 * r * (0.5 * turn).sin()
        } else {
            2.0 * r
        };
        let got = d[1 + k as usize];
        assert!((got - want).abs() < 1e-9, "{k}: {got} instead of {want}");
    }
    // Near the great circles of a sphere.
    let s = icosphere(4);
    let a = s.positions[7];
    check(
        "sphere",
        &s,
        7,
        &|p| 2.0 * (0.5 * distance(a, p)).clamp(0.0, 1.0).asin(),
        3e-3,
    );
    // Several sources: the nearest.
    let s = grid(8, 8, 1.0, 1.0);
    let d = exact_distances(&s.positions, &s.triangles, &[0, 80]);
    for (&got, &p) in d.iter().zip(&s.positions) {
        let want = distance(p, [0.0; 3]).min(distance(p, [1.0, 1.0, 0.0]));
        assert!((got - want).abs() < 1e-9);
    }
}
