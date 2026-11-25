//! One world, many ways to store it (#289): the same world shapes are put
//! into the scene through every mix of a mesh's `f64` origin, its pose and its
//! scale, near the world origin and far from it, and every query and op must
//! give the same world answer, to the rounding of small `f32` coordinates.

use fast_mesh_scene::{MeshId, MeshOptions, Placement, Scene, SceneOptions};
use fast_mesh_store::Mesh;

mod collision;
mod fields;
mod measures;
mod pairs;
mod points;
mod registration;
mod sections;
mod selections;
mod solids;

pub type V3 = [f64; 3];

/// What every shape, point and plane of a test is moved by in the world.
pub const NEAR: V3 = [10.0, -4.0, 6.0];
pub const FAR: V3 = [1.0e5, -2.5e5, 7.3e4];

pub const SCALES: [V3; 4] = [
    [1.0, 1.0, 1.0],
    [2.0, 2.0, 2.0],
    [0.25, 0.25, 0.25],
    [2.0, 1.5, 0.5],
];

/// A rotation about z, then about x.
pub fn turn(z: f64, x: f64) -> [[f64; 3]; 3] {
    let ((sz, cz), (sx, cx)) = (z.sin_cos(), x.sin_cos());
    [
        [cz, -sz, 0.0],
        [cx * sz, cx * cz, -sx],
        [sx * sz, sx * cz, cx],
    ]
}

pub fn add(a: V3, b: V3) -> V3 {
    std::array::from_fn(|k| a[k] + b[k])
}

pub fn sub(a: V3, b: V3) -> V3 {
    std::array::from_fn(|k| a[k] - b[k])
}

pub fn dot(a: V3, b: V3) -> f64 {
    (0..3).map(|k| a[k] * b[k]).sum()
}

pub fn cross(a: V3, b: V3) -> V3 {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

pub fn norm(a: V3) -> f64 {
    dot(a, a).sqrt()
}

pub fn unit(a: V3) -> V3 {
    let n = norm(a);
    a.map(|c| c / n)
}

/// A world shape around zero.
#[derive(Debug, Clone)]
pub struct Shape {
    pub points: Vec<V3>,
    pub triangles: Vec<[u32; 3]>,
}

impl Shape {
    pub fn moved(mut self, by: V3) -> Shape {
        for p in &mut self.points {
            *p = add(*p, by);
        }
        self
    }

    pub fn centre(&self) -> V3 {
        let (mut lo, mut hi) = ([f64::INFINITY; 3], [f64::NEG_INFINITY; 3]);
        for p in &self.points {
            for k in 0..3 {
                lo[k] = lo[k].min(p[k]);
                hi[k] = hi[k].max(p[k]);
            }
        }
        std::array::from_fn(|k| 0.5 * lo[k] + 0.5 * hi[k])
    }
}

pub fn block(lo: V3, hi: V3) -> Shape {
    Shape {
        points: (0..8)
            .map(|i| std::array::from_fn(|k| if i >> k & 1 == 1 { hi[k] } else { lo[k] }))
            .collect(),
        #[rustfmt::skip]
        triangles: vec![
            [0, 4, 6], [0, 6, 2], [1, 3, 7], [1, 7, 5], [0, 1, 5], [0, 5, 4],
            [2, 6, 7], [2, 7, 3], [0, 2, 3], [0, 3, 1], [4, 5, 7], [4, 7, 6],
        ],
    }
}

/// A closed, outward-wound surface of revolution about z: one ring per
/// `(radius, z)` of `profile` (upwards), an apex at each end, every point
/// moved by `bend`.
pub fn revolved(profile: &[(f64, f64)], segments: u32, bend: impl Fn(V3) -> V3) -> Shape {
    let n = segments;
    let mut points = vec![[0.0, 0.0, profile[0].1]];
    for &(r, z) in &profile[1..profile.len() - 1] {
        for s in 0..n {
            let (sin, cos) = (f64::from(s) / f64::from(n) * std::f64::consts::TAU).sin_cos();
            points.push([r * cos, r * sin, z]);
        }
    }
    points.push([0.0, 0.0, profile[profile.len() - 1].1]);
    let rings = profile.len() as u32 - 2;
    let last = points.len() as u32 - 1;
    let at = |ring: u32, s: u32| 1 + ring * n + s % n;
    let mut triangles = Vec::new();
    for s in 0..n {
        triangles.push([0, at(0, s + 1), at(0, s)]);
        for r in 0..rings - 1 {
            let (a, b, c, d) = (at(r, s), at(r, s + 1), at(r + 1, s + 1), at(r + 1, s));
            triangles.push([a, b, c]);
            triangles.push([a, c, d]);
        }
        triangles.push([last, at(rings - 1, s), at(rings - 1, s + 1)]);
    }
    for p in &mut points {
        *p = bend(*p);
    }
    Shape { points, triangles }
}

fn half_circle(r: f64, steps: u32) -> Vec<(f64, f64)> {
    (0..=steps)
        .map(|i| {
            let (s, c) = (f64::from(i) / f64::from(steps) * std::f64::consts::PI).sin_cos();
            (r * s, -r * c)
        })
        .collect()
}

/// A closed lump of about 2.4 x 1.8 x 2, without symmetries.
pub fn lump() -> Shape {
    revolved(&half_circle(1.0, 10), 14, |p| {
        let f = 1.0 + 0.13 * (2.3 * p[0] + 0.4).sin() + 0.09 * (1.7 * p[1] - 1.1 * p[2]).cos();
        [1.15 * f * p[0], 0.85 * f * p[1], f * p[2] + 0.07 * p[0]]
    })
}

/// A ball of radius `r`, fine enough to be smooth.
pub fn ball(r: f64) -> Shape {
    revolved(&half_circle(r, 24), 48, |p| p)
}

/// An open sheet over `[-1.2, 1.2] x [-0.9, 0.9]`, gently bent, facing +z.
pub fn sheet(n: u32) -> Shape {
    let mut points = Vec::new();
    for j in 0..=n {
        for i in 0..=n {
            let (x, y) = (f64::from(i) / f64::from(n), f64::from(j) / f64::from(n));
            let (x, y) = (2.4 * x - 1.2, 1.8 * y - 0.9);
            let z = 0.15 * (1.9 * x).sin() + 0.1 * (2.3 * y + 0.3).cos();
            points.push([x, y, z]);
        }
    }
    let mut triangles = Vec::new();
    for j in 0..n {
        for i in 0..n {
            let a = j * (n + 1) + i;
            triangles.push([a, a + 1, a + n + 2]);
            triangles.push([a, a + n + 2, a + n + 1]);
        }
    }
    Shape { points, triangles }
}

/// Where a mesh's `f64` origin is.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Origin {
    /// Zero: the placement takes the mesh to its place.
    Zero,
    /// The mesh's own centre: it is where it is without a translation.
    Centre,
    /// Some of both.
    Both,
    /// Far off, which the placement takes back.
    Away,
}

/// How a world shape is put into the scene.
#[derive(Debug, Clone, Copy)]
pub struct Put {
    pub scale: V3,
    pub turned: bool,
    pub origin: Origin,
    /// What the shape is moved by in the world.
    pub at: V3,
}

/// As if there were no placements.
pub const PLAIN: Put = Put {
    scale: [1.0; 3],
    turned: false,
    origin: Origin::Zero,
    at: [0.0; 3],
};

impl Put {
    pub fn rotation(&self) -> [[f64; 3]; 3] {
        if self.turned {
            turn(0.4, -0.7)
        } else {
            Placement::IDENTITY.rotation
        }
    }

    /// A world direction in the mesh: the scale's inverse times the
    /// rotation's transpose.
    fn back(&self, d: V3) -> V3 {
        let r = self.rotation();
        std::array::from_fn(|k| (0..3).map(|i| r[i][k] * d[i]).sum::<f64>() / self.scale[k])
    }

    fn out(&self, d: V3) -> V3 {
        let (r, s) = (self.rotation(), self.scale);
        std::array::from_fn(|i| (0..3).map(|k| r[i][k] * s[k] * d[k]).sum())
    }

    /// The mesh of the world shape `shape`, moved by `at`, and its placement.
    pub fn mesh(&self, shape: &Shape) -> (Mesh, Placement) {
        let centre = add(self.at, shape.centre());
        let (origin, translation) = match self.origin {
            Origin::Zero => ([0.0; 3], self.at),
            Origin::Centre => (self.back(centre), [0.0; 3]),
            Origin::Both => {
                let t = [3.0, -2.0, 1.0];
                (self.back(sub(centre, t)), t)
            }
            Origin::Away => {
                let far = if self.at == FAR { 1.0 } else { 1e-4 };
                let o = [2.0e5, 4.0e4, -1.0e5].map(|c| c * far);
                (o, sub(centre, self.out(o)))
            }
        };
        let positions: Vec<f64> = shape
            .points
            .iter()
            .flat_map(|&p| self.back(sub(add(self.at, p), translation)))
            .collect();
        let indices = shape.triangles.as_flattened();
        let mesh = Mesh::from_buffers_f64_with_origin(&positions, indices, origin).unwrap();
        let placement = Placement {
            rotation: self.rotation(),
            translation,
            scale: self.scale,
        };
        (mesh, placement)
    }

    pub fn add(&self, scene: &mut Scene, shape: &Shape) -> MeshId {
        let (mesh, placement) = self.mesh(shape);
        let options = MeshOptions {
            placement,
            ..MeshOptions::default()
        };
        scene.add_mesh(mesh, options).unwrap()
    }

    /// A model point in the world.
    pub fn world(&self, p: V3) -> V3 {
        add(self.at, p)
    }

    /// A world point in the model.
    pub fn model(&self, w: V3) -> V3 {
        sub(w, self.at)
    }
}

pub fn puts_at(at: V3) -> Vec<Put> {
    let mut out = Vec::new();
    for scale in SCALES {
        for turned in [false, true] {
            for origin in [Origin::Zero, Origin::Centre, Origin::Both, Origin::Away] {
                out.push(Put {
                    scale,
                    turned,
                    origin,
                    at,
                });
            }
        }
    }
    out
}

/// Fewer, for what costs more.
pub fn some_puts_at(at: V3) -> Vec<Put> {
    let (one, quarter, uneven) = (SCALES[0], SCALES[2], SCALES[3]);
    [
        (one, true, Origin::Both),
        (quarter, false, Origin::Centre),
        (quarter, true, Origin::Away),
        (uneven, false, Origin::Both),
        (uneven, true, Origin::Zero),
        (uneven, true, Origin::Centre),
    ]
    .map(|(scale, turned, origin)| Put {
        scale,
        turned,
        origin,
        at,
    })
    .to_vec()
}

pub fn scene() -> Scene {
    Scene::new(SceneOptions::default())
}

/// How far the numbers of `got` are from those of `want` (infinite when there
/// are others, or some are finite in one only).
pub fn off(got: &[f64], want: &[f64]) -> f64 {
    if got.len() != want.len() {
        return f64::INFINITY;
    }
    let pairs = got.iter().zip(want);
    pairs.fold(0.0, |m: f64, (&g, &w)| {
        let same = g == w || (g.is_nan() && w.is_nan());
        let d = if same { 0.0 } else { (g - w).abs() };
        if d.is_nan() { f64::INFINITY } else { m.max(d) }
    })
}

#[track_caller]
fn report<T: std::fmt::Debug>(what: &str, of: usize, tol: f64, wrong: &[(f64, T)]) {
    if wrong.is_empty() {
        return;
    }
    let mut lines = format!("{what}: {} of {of} are off by more than {tol}", wrong.len());
    for (d, put) in wrong {
        lines += &format!("\n  {d:.3e} {put:?}");
    }
    panic!("{lines}");
}

/// Fails unless `run` gives the same numbers for every put as for [`PLAIN`].
#[track_caller]
pub fn agree_for(what: &str, puts: &[Put], tol: f64, run: impl Fn(Put) -> Vec<f64>) {
    let want = run(PLAIN);
    assert!(!want.is_empty(), "{what}: nothing to compare");
    let mut wrong = Vec::new();
    for &put in puts {
        let d = off(&run(put), &want);
        if d > tol {
            wrong.push((d, put));
        }
    }
    report(what, puts.len(), tol, &wrong);
}

/// [`agree_for`] every put, near and far.
#[track_caller]
pub fn agree(what: &str, tol: f64, run: impl Fn(Put) -> Vec<f64>) {
    let puts = [puts_at(NEAR), puts_at(FAR)].concat();
    agree_for(what, &puts, tol, run);
}

/// [`agree_for`] some puts, near and far.
#[track_caller]
pub fn agree_some(what: &str, tol: f64, run: impl Fn(Put) -> Vec<f64>) {
    let puts = [some_puts_at(NEAR), some_puts_at(FAR)].concat();
    agree_for(what, &puts, tol, run);
}

/// Fails unless `run` gives the same numbers for the pairs of puts `pick`
/// picks, near and far, as for [`PLAIN`] twice.
#[track_caller]
fn agree_for_pairs(
    what: &str,
    tol: f64,
    pick: impl Fn(usize, usize) -> bool,
    run: impl Fn(Put, Put) -> Vec<f64>,
) {
    let want = run(PLAIN, PLAIN);
    assert!(!want.is_empty(), "{what}: nothing to compare");
    let (mut wrong, mut of) = (Vec::new(), 0);
    for at in [NEAR, FAR] {
        let puts = some_puts_at(at);
        for (i, &a) in puts.iter().enumerate() {
            for (j, &b) in puts.iter().enumerate() {
                if !pick(i, j) {
                    continue;
                }
                of += 1;
                let d = off(&run(a, b), &want);
                if d > tol {
                    wrong.push((d, (a, b)));
                }
            }
        }
    }
    report(what, of, tol, &wrong);
}

/// [`agree_some`] for every pair of puts of two shapes.
#[track_caller]
pub fn agree_pairs(what: &str, tol: f64, run: impl Fn(Put, Put) -> Vec<f64>) {
    agree_for_pairs(what, tol, |_, _| true, run);
}

/// [`agree_pairs`] for a few of them, for what costs a lot.
#[track_caller]
pub fn agree_few_pairs(what: &str, tol: f64, run: impl Fn(Put, Put) -> Vec<f64>) {
    agree_for_pairs(what, tol, |i, j| j == (2 * i + 1) % 6, run);
}

/// The world corners of mesh `id`'s live triangles, in the model of `put`.
pub fn corners(scene: &Scene, id: MeshId, put: Put) -> Vec<[V3; 3]> {
    let mesh = scene.mesh(id).unwrap();
    (0..mesh.triangle_slots() as u32)
        .filter_map(|t| mesh.triangle(t))
        .map(|tri| {
            tri.map(|v| {
                let p = mesh.position(v).unwrap().map(f64::from);
                put.model(scene.local_to_world(id, p).unwrap())
            })
        })
        .collect()
}

/// The model positions of mesh `id`'s live vertices, in slot order.
pub fn positions(scene: &Scene, id: MeshId, put: Put) -> Vec<f64> {
    let mesh = scene.mesh(id).unwrap();
    (0..mesh.vertex_slots() as u32)
        .filter_map(|v| mesh.position(v))
        .flat_map(|p| put.model(scene.local_to_world(id, p.map(f64::from)).unwrap()))
        .collect()
}

/// Volume, area, bounds and first moments of mesh `id` in the model.
pub fn solid(scene: &Scene, id: MeshId, put: Put) -> Vec<f64> {
    let (mut volume, mut area, mut moment) = (0.0, 0.0, [0.0; 3]);
    let (mut lo, mut hi) = ([f64::INFINITY; 3], [f64::NEG_INFINITY; 3]);
    for [a, b, c] in corners(scene, id, put) {
        area += 0.5 * norm(cross(sub(b, a), sub(c, a)));
        let v = dot(a, cross(b, c)) / 6.0;
        volume += v;
        for k in 0..3 {
            moment[k] += v * (a[k] + b[k] + c[k]) / 4.0;
            for p in [a, b, c] {
                lo[k] = lo[k].min(p[k]);
                hi[k] = hi[k].max(p[k]);
            }
        }
    }
    let mut out = vec![volume, area];
    out.extend(lo.into_iter().chain(hi).chain(moment));
    out
}

pub fn f64s(values: &[f32]) -> Vec<f64> {
    values.iter().map(|&v| f64::from(v)).collect()
}
