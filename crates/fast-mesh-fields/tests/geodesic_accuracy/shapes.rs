//! Meshes whose triangles are stretched, obtuse, fans or of all sizes.

use std::collections::HashMap;
use std::f64::consts::{PI, TAU};

/// Vertices and triangles.
#[derive(Clone)]
pub struct Shape {
    pub positions: Vec<[f64; 3]>,
    pub triangles: Vec<[u32; 3]>,
}

impl Shape {
    pub fn mapped(mut self, f: impl Fn([f64; 3]) -> [f64; 3]) -> Shape {
        for p in &mut self.positions {
            *p = f(*p);
        }
        self
    }

    /// `self` with every coordinate rounded to what a mesh can hold.
    pub fn rounded(self) -> Shape {
        self.mapped(|p| p.map(|c| f64::from(c as f32)))
    }

    /// The vertex that lies furthest along `direction`.
    pub fn furthest(&self, direction: [f64; 3]) -> u32 {
        let along = |v: u32| {
            let p = self.positions[v as usize];
            (0..3).map(|k| p[k] * direction[k]).sum::<f64>()
        };
        let vertices = 0..self.positions.len() as u32;
        vertices
            .max_by(|&a, &b| along(a).total_cmp(&along(b)))
            .unwrap()
    }

    /// The triangles in two orders at once: every other one from the start,
    /// the rest backwards.
    pub fn shuffled(mut self) -> Shape {
        let (even, odd): (Vec<_>, Vec<_>) = self
            .triangles
            .into_iter()
            .enumerate()
            .partition(|(k, _)| k % 2 == 0);
        let t = even.into_iter().chain(odd.into_iter().rev());
        self.triangles = t.map(|(_, t)| t).collect();
        self
    }
}

fn unit(p: [f64; 3]) -> [f64; 3] {
    let l = (p[0] * p[0] + p[1] * p[1] + p[2] * p[2]).sqrt();
    p.map(|c| c / l)
}

/// The unit icosahedron with its faces split into four `subdivisions` times.
pub fn icosphere(subdivisions: u32) -> Shape {
    let g = (1.0 + 5.0f64.sqrt()) / 2.0;
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
    Shape {
        positions: p,
        triangles: t,
    }
}

/// An icosphere with every vertex moved about on the sphere.
pub fn jittered_sphere(subdivisions: u32, by: f64) -> Shape {
    let mut state = 99u64;
    let mut s = icosphere(subdivisions);
    for p in &mut s.positions {
        *p = unit(p.map(|c| {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            c + by * ((state >> 33) as f64 / (1u64 << 31) as f64 - 0.5)
        }));
    }
    s
}

/// An icosphere with triangles `by²` times as large at the top as without,
/// and as much smaller at the bottom: well shaped, of all sizes.
pub fn graded_sphere(subdivisions: u32, by: f64) -> Shape {
    icosphere(subdivisions).mapped(|p| {
        // To the plane, scaled there, and back.
        let (x, y) = (p[0] / (1.0 - p[2]), p[1] / (1.0 - p[2]));
        if !(x.is_finite() && y.is_finite()) {
            return p;
        }
        let (x, y) = (x * by, y * by);
        let d = 1.0 + x * x + y * y;
        [2.0 * x / d, 2.0 * y / d, (d - 2.0) / d]
    })
}

/// A sphere of rings around Z, with needles at the poles.
pub fn uv_sphere(segments: u32, rings: u32) -> Shape {
    let mut p = vec![[0.0, 0.0, 1.0]];
    for r in 1..rings {
        let a = f64::from(r) / f64::from(rings) * PI;
        for s in 0..segments {
            let b = f64::from(s) / f64::from(segments) * TAU;
            p.push([a.sin() * b.cos(), a.sin() * b.sin(), a.cos()]);
        }
    }
    let last = p.len() as u32;
    p.push([0.0, 0.0, -1.0]);
    let mut t = Vec::new();
    for s in 0..segments {
        let n = (s + 1) % segments;
        t.push([0, 1 + s, 1 + n]);
        for r in 0..rings - 2 {
            let (a, b) = (1 + r * segments + s, 1 + r * segments + n);
            let (c, d) = (a + segments, b + segments);
            t.push([a, c, d]);
            t.push([a, d, b]);
        }
        let base = 1 + (rings - 2) * segments;
        t.push([base + s, last, base + n]);
    }
    Shape {
        positions: p,
        triangles: t,
    }
}

/// The side of a cylinder of radius 1 along Z. Ring `k` starts at vertex
/// `k * segments`.
pub fn tube(segments: u32, rings: u32, height: f64) -> Shape {
    let mut p = Vec::new();
    let mut t = Vec::new();
    for k in 0..=rings {
        for s in 0..segments {
            let a = f64::from(s) / f64::from(segments) * TAU;
            p.push([a.cos(), a.sin(), height * f64::from(k) / f64::from(rings)]);
        }
    }
    for k in 0..rings {
        for s in 0..segments {
            let (a, b) = (k * segments + s, k * segments + (s + 1) % segments);
            let (c, d) = (a + segments, b + segments);
            t.push([a, b, d]);
            t.push([a, d, c]);
        }
    }
    Shape {
        positions: p,
        triangles: t,
    }
}

/// A cylinder whose caps are fans around one vertex each, the last two.
pub fn fan_cylinder(segments: u32, rings: u32, height: f64) -> Shape {
    let mut s = tube(segments, rings, height);
    let low = s.positions.len() as u32;
    s.positions.push([0.0, 0.0, 0.0]);
    s.positions.push([0.0, 0.0, height]);
    let top = rings * segments;
    for k in 0..segments {
        let next = (k + 1) % segments;
        s.triangles.push([low, next, k]);
        s.triangles.push([low + 1, top + k, top + next]);
    }
    s
}

/// A flat disc: one fan, around vertex 0.
pub fn disc(segments: u32) -> Shape {
    let mut p = vec![[0.0, 0.0, 0.0]];
    let mut t = Vec::new();
    for s in 0..segments {
        let a = f64::from(s) / f64::from(segments) * TAU;
        p.push([a.cos(), a.sin(), 0.0]);
        t.push([0, 1 + s, 1 + (s + 1) % segments]);
    }
    Shape {
        positions: p,
        triangles: t,
    }
}

/// A flat rectangle that is one thin fan: from its first corner to the `n`
/// parts of its far side, which start at vertex 2.
pub fn fan_strip(n: u32, width: f64, height: f64) -> Shape {
    let mut p = vec![[0.0, 0.0, 0.0], [width, 0.0, 0.0]];
    let mut t = Vec::new();
    for k in 0..=n {
        p.push([width * f64::from(k) / f64::from(n), height, 0.0]);
    }
    for k in 0..n {
        t.push([0, 2 + k + 1, 2 + k]);
    }
    t.push([0, 1, 2 + n]);
    Shape {
        positions: p,
        triangles: t,
    }
}

/// A flat rectangle of `nx` by `ny` cells.
pub fn grid(nx: u32, ny: u32, width: f64, height: f64) -> Shape {
    let mut p = Vec::new();
    for y in 0..=ny {
        for x in 0..=nx {
            p.push([
                width * f64::from(x) / f64::from(nx),
                height * f64::from(y) / f64::from(ny),
                0.0,
            ]);
        }
    }
    let mut t = Vec::new();
    for y in 0..ny {
        for x in 0..nx {
            let a = y * (nx + 1) + x;
            t.push([a, a + 1, a + nx + 2]);
            t.push([a, a + nx + 2, a + nx + 1]);
        }
    }
    Shape {
        positions: p,
        triangles: t,
    }
}

/// A grid of square cells with a notch cut into it from below, so that the
/// way from its one foot to the other bends around the notch's end.
pub fn notched(n: u32) -> Shape {
    let mut s = grid(n, n, 1.0, 1.0);
    let (lo, hi) = (n / 2 - n / 8, n / 2 + n / 8);
    let mut k = 0;
    s.triangles.retain(|_| {
        let cell = k / 2;
        k += 1;
        let (x, y) = (cell % n, cell / n);
        !((lo..hi).contains(&x) && y < n / 2)
    });
    s
}

/// `pages` rectangles, sheared along the edge they share, which lies on Z;
/// every page's vertices come in rows along it from the edge out, the edge's
/// own first. Cells are `dx` by `dz`.
pub fn book(pages: u32, nx: u32, nz: u32, dx: f64, dz: f64, shear: f64) -> Shape {
    let mut p: Vec<[f64; 3]> = (0..=nz).map(|k| [0.0, 0.0, dz * f64::from(k)]).collect();
    let mut t = Vec::new();
    for page in 0..pages {
        let a = f64::from(page) / f64::from(pages) * TAU;
        let first = p.len() as u32;
        for x in 1..=nx {
            for z in 0..=nz {
                let r = dx * f64::from(x);
                p.push([a.cos() * r, a.sin() * r, dz * f64::from(z) + shear * r]);
            }
        }
        let at = |x: u32, z: u32| match x {
            0 => z,
            _ => first + (nz + 1) * (x - 1) + z,
        };
        for x in 0..nx {
            for z in 0..nz {
                let (a, b, c, d) = (at(x, z), at(x + 1, z), at(x + 1, z + 1), at(x, z + 1));
                t.push([a, b, c]);
                t.push([a, c, d]);
            }
        }
    }
    Shape {
        positions: p,
        triangles: t,
    }
}
