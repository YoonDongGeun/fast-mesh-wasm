//! The implicit surfaces of [`LatticeKind`], and the distance to them.
//!
//! Every surface is the zero set of a trigonometric function of the phase
//! `u = 2 pi x / cell`, so one cell is one period along each axis. The
//! functions are the usual level-set approximations of the minimal surfaces:
//! their areas are within 0.4 % of the minimal ones.

use std::f64::consts::PI;

/// A triply periodic surface that a lattice's walls follow.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum LatticeKind {
    /// The gyroid: no straight lines and no mirror planes, about as stiff in
    /// every direction.
    #[default]
    Gyroid,
    /// The Schwarz P surface: a cubic grid of round chambers joined by necks.
    SchwarzP,
    /// The Schwarz D (diamond) surface: the most wall of the three for the
    /// same cell size and wall thickness.
    Diamond,
}

impl LatticeKind {
    /// Every kind, in declaration order.
    pub const ALL: [LatticeKind; 3] = [
        LatticeKind::Gyroid,
        LatticeKind::SchwarzP,
        LatticeKind::Diamond,
    ];

    /// The area of the minimal surface in one cell of edge length 1.
    pub fn area_per_cell(self) -> f64 {
        match self {
            LatticeKind::Gyroid => 3.0915,
            LatticeKind::SchwarzP => 2.3451,
            LatticeKind::Diamond => 3.8378,
        }
    }

    /// The Euler characteristic of the surface in one cell.
    fn euler_per_cell(self) -> f64 {
        match self {
            LatticeKind::Gyroid => -8.0,
            LatticeKind::SchwarzP => -4.0,
            LatticeKind::Diamond => -16.0,
        }
    }

    /// The share of space that walls of `thickness` fill in cells of
    /// `cell_size`, by the tube formula for the minimal surface: area times
    /// thickness, less what the saddles save. From 0 to 1.
    pub fn solid_fraction(self, cell_size: f64, thickness: f64) -> f64 {
        let x = thickness / cell_size;
        // Past its peak the formula turns down: the walls have met long ago.
        let peak = (-2.0 * self.area_per_cell() / (PI * self.euler_per_cell())).sqrt();
        if x >= peak {
            return 1.0;
        }
        let v = self.area_per_cell() * x + PI / 6.0 * self.euler_per_cell() * x * x * x;
        v.clamp(0.0, 1.0)
    }

    /// The area of both faces of walls of `thickness` per volume, in cells of
    /// `cell_size`.
    pub(super) fn wall_area(self, cell_size: f64, thickness: f64) -> f64 {
        let r = 0.5 * thickness / cell_size;
        let a = self.area_per_cell() + 2.0 * PI * self.euler_per_cell() * r * r;
        2.0 * a.max(0.0) / cell_size
    }

    /// The value and the phase gradient at a point whose coordinates have the
    /// given sines and cosines.
    fn eval(self, [x, y, z]: [Trig; 3]) -> (f64, [f64; 3]) {
        match self {
            LatticeKind::Gyroid => (
                x.s * y.c + y.s * z.c + z.s * x.c,
                [
                    x.c * y.c - z.s * x.s,
                    y.c * z.c - x.s * y.s,
                    z.c * x.c - y.s * z.s,
                ],
            ),
            LatticeKind::SchwarzP => (x.c + y.c + z.c, [-x.s, -y.s, -z.s]),
            LatticeKind::Diamond => (
                x.s * (y.s * z.s + y.c * z.c) + x.c * (y.s * z.c + y.c * z.s),
                [
                    x.c * (y.s * z.s + y.c * z.c) - x.s * (y.s * z.c + y.c * z.s),
                    y.c * (x.s * z.s + x.c * z.c) - y.s * (x.s * z.c + x.c * z.s),
                    z.c * (x.s * y.s + x.c * y.c) - z.s * (x.s * y.c + x.c * y.s),
                ],
            ),
        }
    }

    /// The second derivative along `g`, times `|g|²`.
    fn bend(self, [x, y, z]: [Trig; 3], f: f64, g: [f64; 3]) -> f64 {
        // The Hessian's diagonal, then its xy, xz and yz entries.
        let (d, [xy, xz, yz]) = match self {
            LatticeKind::Gyroid => (
                [
                    -x.s * y.c - z.s * x.c,
                    -y.s * z.c - x.s * y.c,
                    -z.s * x.c - y.s * z.c,
                ],
                [-x.c * y.s, -z.c * x.s, -y.c * z.s],
            ),
            LatticeKind::SchwarzP => ([-x.c, -y.c, -z.c], [0.0; 3]),
            LatticeKind::Diamond => (
                [-f; 3],
                [
                    z.s * (x.c * y.c + x.s * y.s) - z.c * (x.c * y.s + x.s * y.c),
                    y.s * (x.c * z.c + x.s * z.s) - y.c * (x.c * z.s + x.s * z.c),
                    x.s * (y.c * z.c + y.s * z.s) - x.c * (y.c * z.s + y.s * z.c),
                ],
            ),
        };
        d[0] * g[0] * g[0]
            + d[1] * g[1] * g[1]
            + d[2] * g[2] * g[2]
            + 2.0 * (xy * g[0] * g[1] + xz * g[0] * g[2] + yz * g[1] * g[2])
    }

    /// The distance, in phase, from the point `p` to the surface.
    ///
    /// Within `band` of `level` it is the distance to the closest point of
    /// the surface (by projection); elsewhere it is the distance along the
    /// gradient to second order, which is enough to tell the sides of
    /// `level` apart.
    pub(super) fn distance(self, p: [Trig; 3], level: f64, band: f64) -> f64 {
        let (f, g) = self.eval(p);
        let b = norm(g);
        if b < 1e-6 {
            // A critical point: the middle of a chamber or of a neck.
            return FAR;
        }
        // Solves `|f| - b x + q x² / 2 = 0` along the gradient, towards zero.
        let q = f.signum() * self.bend(p, f, g) / (b * b);
        let root = (b * b - 2.0 * q * f.abs()).max(0.0).sqrt();
        let x = 2.0 * f.abs() / (b + root);
        if (x - level).abs() > band || x > FAR {
            return x;
        }
        // Twice: go to the surface, and measure along the normal there.
        let mut n = g.map(|c| c / b);
        let mut d = x.copysign(f);
        for _ in 0..2 {
            let (f, g) = self.eval(std::array::from_fn(|k| p[k].back(d * n[k])));
            let b = norm(g);
            if b < 1e-3 {
                return d.abs();
            }
            let m = g.map(|c| c / b);
            d = d * (n[0] * m[0] + n[1] * m[1] + n[2] * m[2]) + f / b;
            n = m;
        }
        d.abs()
    }
}

/// A phase distance past every wall, and the longest step the projection of
/// [`LatticeKind::distance`] takes.
const FAR: f64 = 1.6;

fn norm(v: [f64; 3]) -> f64 {
    (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt()
}

/// The sine and cosine of one phase coordinate.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(super) struct Trig {
    s: f64,
    c: f64,
}

impl Trig {
    pub(super) fn new(u: f64) -> Trig {
        let (s, c) = u.sin_cos();
        Trig { s, c }
    }

    /// The coordinate `d` less than this one, for `|d|` up to 2: the sine and
    /// cosine of `d` are short series, accurate to 1e-6 (1e-8 up to 1.6),
    /// instead of library calls.
    #[inline]
    fn back(self, d: f64) -> Trig {
        let d = d.clamp(-2.0, 2.0);
        let x = d * d;
        let sine = [1.0 / 110.0, 1.0 / 72.0, 1.0 / 42.0, 1.0 / 20.0, 1.0 / 6.0]
            .into_iter()
            .fold(1.0 - x / 156.0, |t, k| 1.0 - x * k * t);
        let cosine = [1.0 / 90.0, 1.0 / 56.0, 1.0 / 30.0, 1.0 / 12.0, 0.5]
            .into_iter()
            .fold(1.0 - x / 132.0, |t, k| 1.0 - x * k * t);
        let (s, c) = (d * sine, cosine);
        Trig {
            s: self.s * c - self.c * s,
            c: self.c * c + self.s * s,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::marching_cubes_raw;
    use fast_mesh_bvh::{BuildOptions, Bvh};
    use fast_mesh_query::{ClosestOptions, closest_point};
    use fast_mesh_store::Mesh;

    const TAU: f64 = 2.0 * PI;

    fn trig(u: [f64; 3]) -> [Trig; 3] {
        u.map(Trig::new)
    }

    fn rng(seed: &mut u64) -> f64 {
        *seed ^= *seed << 13;
        *seed ^= *seed >> 7;
        *seed ^= *seed << 17;
        (*seed >> 11) as f64 / (1u64 << 53) as f64
    }

    #[test]
    fn gradients_and_bends_match_differences() {
        let mut seed = 0x9E37_79B9_7F4A_7C15;
        let e = 1e-4;
        for kind in LatticeKind::ALL {
            for _ in 0..200 {
                let u: [f64; 3] = std::array::from_fn(|_| TAU * rng(&mut seed));
                let at = |v: [f64; 3]| kind.eval(trig(v)).0;
                let (f, g) = kind.eval(trig(u));
                for (k, &g) in g.iter().enumerate() {
                    let step =
                        |s: f64| std::array::from_fn(|j| u[j] + if j == k { s } else { 0.0 });
                    let slope = (at(step(e)) - at(step(-e))) / (2.0 * e);
                    assert!((slope - g).abs() < 1e-6, "{kind:?} axis {k}");
                }
                let along = |s: f64| std::array::from_fn(|j| u[j] + s * g[j]);
                let second = (at(along(e)) - 2.0 * f + at(along(-e))) / (e * e);
                let bend = kind.bend(trig(u), f, g);
                assert!((second - bend).abs() < 1e-5, "{kind:?}: {second} {bend}");
            }
        }
    }

    #[test]
    fn short_series_turn_back_exactly_enough() {
        for i in -200..=200 {
            let d = f64::from(i) / 100.0;
            let got = Trig::new(0.7).back(d);
            let want = Trig::new(0.7 - d);
            let tol = if d.abs() <= FAR { 1e-8 } else { 1e-6 };
            assert!(
                (got.s - want.s).abs() < tol && (got.c - want.c).abs() < tol,
                "{d}"
            );
        }
    }

    /// The zero set over two cells, as a fine mesh in phase units.
    fn zero_set(kind: LatticeKind) -> Mesh {
        let n = 129;
        let h = 2.0 * TAU / 128.0;
        let mut values = Vec::with_capacity(n * n * n);
        for z in 0..n {
            for y in 0..n {
                for x in 0..n {
                    let u = [x, y, z].map(|i| i as f64 * h);
                    values.push(kind.eval(trig(u)).0 as f32);
                }
            }
        }
        let iso = marching_cubes_raw([0.0; 3], [n; 3], h as f32, &values, 0.0).unwrap();
        Mesh::from_buffers(iso.positions.as_flattened(), iso.triangles.as_flattened()).unwrap()
    }

    #[test]
    fn distances_match_the_closest_point_on_the_surface() {
        let mut seed = 7;
        for kind in LatticeKind::ALL {
            let surface = zero_set(kind);
            let bvh = Bvh::build(&surface, &BuildOptions::default()).unwrap();
            let (mut worst, mut count) = (0.0f64, 0);
            for _ in 0..4000 {
                // In the middle of the sampled cells, away from their rim.
                let u: [f64; 3] = std::array::from_fn(|_| TAU * (0.5 + rng(&mut seed)));
                let hit = closest_point(
                    &surface,
                    &bvh,
                    u.map(|c| c as f32),
                    &ClosestOptions::default(),
                )
                .unwrap()
                .unwrap();
                let exact = f64::from(hit.distance);
                // Walls up to a quarter of a cell thick: half of one, in phase.
                for level in [0.15, 0.4, 0.75] {
                    let d = kind.distance(trig(u), level, 0.4);
                    if (exact - level).abs() < 0.1 {
                        worst = worst.max((d - exact).abs());
                        count += 1;
                    }
                    // Never on the wrong side of the level.
                    if (exact - level).abs() > 0.03 {
                        assert_eq!(
                            d < level,
                            exact < level,
                            "{kind:?} at {u:?}: {d} vs {exact} for {level}"
                        );
                    }
                }
            }
            assert!(count > 500, "{kind:?}: {count}");
            assert!(worst < 0.03, "{kind:?}: off by {worst}");
        }
    }

    #[test]
    fn the_surfaces_have_the_area_of_the_minimal_ones() {
        for kind in LatticeKind::ALL {
            let surface = zero_set(kind);
            let area = crate::measure(&surface).unwrap().area / 8.0 / (TAU * TAU);
            let want = kind.area_per_cell();
            assert!((area / want - 1.0).abs() < 0.005, "{kind:?}: {area} {want}");
        }
    }

    #[test]
    fn solid_fractions_grow_to_full() {
        for kind in LatticeKind::ALL {
            let mut last = 0.0;
            for i in 1..=60 {
                let f = kind.solid_fraction(10.0, f64::from(i) / 10.0);
                assert!(f >= last && f <= 1.0, "{kind:?} {i}: {f}");
                last = f;
            }
            assert_eq!(kind.solid_fraction(10.0, 10.0), 1.0);
        }
        let g = LatticeKind::Gyroid.solid_fraction(1.0, 0.1);
        assert!((g - (0.30915 - 8.0 * PI / 6.0 * 1e-3)).abs() < 1e-12);
    }
}
