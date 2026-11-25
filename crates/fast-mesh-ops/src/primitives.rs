//! Primitive mesh generators: a box, cylinder, cone, UV sphere, icosphere,
//! torus, arrow, n-gon prism and plane, built from their parameters.
//!
//! Each generator returns its geometry as an [`AppendSource`], positions and
//! triangles indexed from zero, ready for [`append_mesh`](crate::append_mesh)
//! or for building a mesh from buffers. Every shape:
//!
//! - is centred on the origin: a shape of revolution turns around an axis
//!   through it, its extent along the axis centred there, and the box, the
//!   icosphere and the plane have their bounding box centre there;
//! - is welded: each point of the surface is one vertex, shared by every
//!   triangle around it, so no two vertices have the same position;
//! - is closed, except the plane: every edge is shared by exactly two
//!   triangles, so there is no boundary;
//! - is consistently wound counter-clockwise seen from outside, so its
//!   normals point outward and its signed volume is positive. The plane faces
//!   `+axis`.
//!
//! Shapes of revolution (cylinder, cone, UV sphere, torus, arrow, prism) turn
//! around an [`Axis`], `Y` by default, and so does the plane's normal. Their
//! cross-sections are regular polygons inscribed in the circle they
//! approximate, so their volume converges to the analytic one from below as
//! the resolution grows; the box and the prism are exact.
//!
//! Lengths must be finite and within `[1e-30, 1e30]`; resolutions are counts
//! with a lower bound per parameter. A shape with more vertices or triangles
//! than `u32` ids can name is refused, and output buffers are reserved
//! fallibly, so running out of memory is [`OpsError::Store`] with
//! [`StoreError::OutOfMemory`](fast_mesh_store::StoreError::OutOfMemory).
//! Positions are computed in `f64` and stored as `f32`. The output depends on
//! the parameters only.

use std::f64::consts::{PI, TAU};

use crate::append::AppendSource;
use crate::{OpsError, Result, filled, reserve};

/// The axis a shape of revolution turns around, or the plane's normal.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Axis {
    X,
    #[default]
    Y,
    Z,
}

impl Axis {
    /// Maps a point of the canonical frame, whose axis is `Z`, onto this
    /// axis by a cyclic permutation of the coordinates (a rotation, so the
    /// winding is kept).
    fn map(self, [x, y, z]: [f64; 3]) -> [f64; 3] {
        match self {
            Axis::X => [z, x, y],
            Axis::Y => [y, z, x],
            Axis::Z => [x, y, z],
        }
    }

    fn index(self) -> usize {
        match self {
            Axis::X => 0,
            Axis::Y => 1,
            Axis::Z => 2,
        }
    }
}

/// A box with a size per axis, its faces split into grids.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BoxOptions {
    /// Extent along X, Y and Z. Default `[1, 1, 1]`.
    pub size: [f64; 3],
    /// Grid cells along X, Y and Z, at least 1 each. Default `[1, 1, 1]`.
    pub segments: [u32; 3],
}

impl Default for BoxOptions {
    fn default() -> Self {
        BoxOptions {
            size: [1.0; 3],
            segments: [1; 3],
        }
    }
}

/// A capped cylinder.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CylinderOptions {
    /// Default 0.5.
    pub radius: f64,
    /// Extent along the axis. Default 1.
    pub height: f64,
    /// Sides around the axis, at least 3. Default 32.
    pub segments: u32,
    /// Bands along the axis, at least 1. Default 1.
    pub rings: u32,
    /// Concentric bands of each cap, at least 1. Default 1 (a fan).
    pub cap_rings: u32,
    /// Default [`Axis::Y`].
    pub axis: Axis,
}

impl Default for CylinderOptions {
    fn default() -> Self {
        CylinderOptions {
            radius: 0.5,
            height: 1.0,
            segments: 32,
            rings: 1,
            cap_rings: 1,
            axis: Axis::Y,
        }
    }
}

/// A cone, or a frustum when both radii are positive.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ConeOptions {
    /// Radius at `-axis`; 0 puts the apex there. Default 0.5.
    pub radius: f64,
    /// Radius at `+axis`; 0 (the default) puts the apex there.
    pub top_radius: f64,
    /// Extent along the axis. Default 1.
    pub height: f64,
    /// Sides around the axis, at least 3. Default 32.
    pub segments: u32,
    /// Bands along the axis, at least 1. Default 1.
    pub rings: u32,
    /// Concentric bands of each cap, at least 1. Default 1 (a fan).
    pub cap_rings: u32,
    /// Default [`Axis::Y`].
    pub axis: Axis,
}

impl Default for ConeOptions {
    fn default() -> Self {
        ConeOptions {
            radius: 0.5,
            top_radius: 0.0,
            height: 1.0,
            segments: 32,
            rings: 1,
            cap_rings: 1,
            axis: Axis::Y,
        }
    }
}

/// A latitude-longitude sphere with a vertex at each pole.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct UvSphereOptions {
    /// Default 0.5.
    pub radius: f64,
    /// Meridians, at least 3. Default 32.
    pub segments: u32,
    /// Latitude bands from pole to pole, at least 2. Default 16.
    pub rings: u32,
    /// The pole axis. Default [`Axis::Y`].
    pub axis: Axis,
}

impl Default for UvSphereOptions {
    fn default() -> Self {
        UvSphereOptions {
            radius: 0.5,
            segments: 32,
            rings: 16,
            axis: Axis::Y,
        }
    }
}

/// A geodesic sphere: an icosahedron whose faces are split into triangular
/// grids and pushed onto the sphere.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct IcosphereOptions {
    /// Default 0.5.
    pub radius: f64,
    /// Each level splits every triangle into four: `20 · 4^subdivisions`
    /// triangles. At most 16. Default 3.
    pub subdivisions: u32,
}

impl Default for IcosphereOptions {
    fn default() -> Self {
        IcosphereOptions {
            radius: 0.5,
            subdivisions: 3,
        }
    }
}

/// A ring torus.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TorusOptions {
    /// From the axis to the tube's centre. Default 0.4.
    pub major_radius: f64,
    /// The tube's radius, below `major_radius`. Default 0.1.
    pub minor_radius: f64,
    /// Sides around the axis, at least 3. Default 48.
    pub segments: u32,
    /// Sides around the tube, at least 3. Default 24.
    pub tube_segments: u32,
    /// Default [`Axis::Y`].
    pub axis: Axis,
}

impl Default for TorusOptions {
    fn default() -> Self {
        TorusOptions {
            major_radius: 0.4,
            minor_radius: 0.1,
            segments: 48,
            tube_segments: 24,
            axis: Axis::Y,
        }
    }
}

/// An arrow: a capped shaft and a cone head, pointing to `+axis`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ArrowOptions {
    /// From the tail to the tip. Default 1.
    pub length: f64,
    /// Default 0.05.
    pub shaft_radius: f64,
    /// The head's base radius, above `shaft_radius`. Default 0.1.
    pub head_radius: f64,
    /// The head's extent along the axis, below `length`. Default 0.25.
    pub head_length: f64,
    /// Sides around the axis, at least 3. Default 32.
    pub segments: u32,
    /// Default [`Axis::Y`].
    pub axis: Axis,
}

impl Default for ArrowOptions {
    fn default() -> Self {
        ArrowOptions {
            length: 1.0,
            shaft_radius: 0.05,
            head_radius: 0.1,
            head_length: 0.25,
            segments: 32,
            axis: Axis::Y,
        }
    }
}

/// A right prism over a regular polygon.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PrismOptions {
    /// Corners of the polygon, at least 3. Default 6.
    pub sides: u32,
    /// From the axis to each corner (the circumradius). Default 0.5.
    pub radius: f64,
    /// Extent along the axis. Default 1.
    pub height: f64,
    /// Default [`Axis::Y`].
    pub axis: Axis,
}

impl Default for PrismOptions {
    fn default() -> Self {
        PrismOptions {
            sides: 6,
            radius: 0.5,
            height: 1.0,
            axis: Axis::Y,
        }
    }
}

/// A flat rectangular grid, open, facing `+axis`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PlaneOptions {
    /// Extents along the other two axes, in `x, y, z` order (for the
    /// default `Y` normal: X, then Z). Default `[1, 1]`.
    pub size: [f64; 2],
    /// Grid cells along the same two axes, at least 1 each. Default `[1, 1]`.
    pub segments: [u32; 2],
    /// The normal. Default [`Axis::Y`].
    pub axis: Axis,
}

impl Default for PlaneOptions {
    fn default() -> Self {
        PlaneOptions {
            size: [1.0; 2],
            segments: [1; 2],
            axis: Axis::Y,
        }
    }
}

/// One shape and its parameters; see [`generate`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Primitive {
    Box(BoxOptions),
    Cylinder(CylinderOptions),
    Cone(ConeOptions),
    UvSphere(UvSphereOptions),
    Icosphere(IcosphereOptions),
    Torus(TorusOptions),
    Arrow(ArrowOptions),
    Prism(PrismOptions),
    Plane(PlaneOptions),
}

/// Generates `primitive`; see the module docs for what every shape
/// guarantees.
///
/// # Errors
/// [`OpsError::InvalidInput`] for a parameter out of range or a shape too
/// large for `u32` ids, before anything is allocated;
/// [`OpsError::Store`] when the output cannot be allocated.
pub fn generate(primitive: &Primitive) -> Result<AppendSource> {
    match primitive {
        Primitive::Box(o) => cuboid(o),
        Primitive::Cylinder(o) => cylinder(o),
        Primitive::Cone(o) => cone(o),
        Primitive::UvSphere(o) => uv_sphere(o),
        Primitive::Icosphere(o) => icosphere(o),
        Primitive::Torus(o) => torus(o),
        Primitive::Arrow(o) => arrow(o),
        Primitive::Prism(o) => prism(o),
        Primitive::Plane(o) => plane(o),
    }
}

const MIN_LENGTH: f64 = 1e-30;
const MAX_LENGTH: f64 = 1e30;
const TOO_LARGE: &str = "the primitive has more vertices or triangles than u32 ids can name";

/// A length in `[MIN_LENGTH, MAX_LENGTH]`.
pub(crate) fn length(v: f64, what: &'static str) -> Result<f64> {
    if v.is_finite() && (MIN_LENGTH..=MAX_LENGTH).contains(&v) {
        Ok(v)
    } else {
        Err(OpsError::InvalidInput(what))
    }
}

/// A count of at least `min`.
fn count(v: u32, min: u32, what: &'static str) -> Result<u32> {
    if v >= min {
        Ok(v)
    } else {
        Err(OpsError::InvalidInput(what))
    }
}

/// `n` as a `u32` id count, or the "too large" error. Counts are computed
/// in `u128`, where no product of two `u32` parameters overflows.
pub(crate) fn ids(n: u128) -> Result<u32> {
    u32::try_from(n).map_err(|_| OpsError::InvalidInput(TOO_LARGE))
}

/// Output buffers for `vertices` and `triangles`, reserved fallibly.
fn buffers(vertices: u128, triangles: u128) -> Result<AppendSource> {
    let (v, t) = (ids(vertices)?, ids(triangles)?);
    let mut out = AppendSource::default();
    reserve(&mut out.positions, v as usize)?;
    reserve(&mut out.triangles, t as usize)?;
    Ok(out)
}

fn to_f32([x, y, z]: [f64; 3]) -> [f32; 3] {
    [x as f32, y as f32, z as f32]
}

/// A box; see [`BoxOptions`]. Named for the shape, `box` being reserved.
///
/// # Errors
/// As for [`generate`].
pub fn cuboid(o: &BoxOptions) -> Result<AppendSource> {
    let what = "box size must be finite and within [1e-30, 1e30]";
    let size = [
        length(o.size[0], what)?,
        length(o.size[1], what)?,
        length(o.size[2], what)?,
    ];
    let seg = "box segments must be at least 1";
    let [nx, ny, nz] = [
        count(o.segments[0], 1, seg)?,
        count(o.segments[1], 1, seg)?,
        count(o.segments[2], 1, seg)?,
    ];
    let (x, y, z) = (u128::from(nx), u128::from(ny), u128::from(nz));
    // Two full layers (k = 0 and k = nz), and the perimeter ring of each
    // layer between them.
    let layer = (x + 1) * (y + 1);
    let ring = 2 * (x + y);
    let vertices = ids(2 * layer + (z - 1) * ring)?;
    let triangles = ids(4 * (x * y + y * z + z * x))?;
    // Both are at most the vertex count, which fits.
    let (layer, ring) = (layer as u32, ring as u32);
    let mut out = AppendSource {
        positions: filled(vertices as usize, [0.0; 3])?,
        triangles: Vec::new(),
    };
    reserve(&mut out.triangles, triangles as usize)?;

    // The vertex of lattice point (i, j, k) on the surface: layer k = 0 row
    // by row, then the rings of the layers between, then layer k = nz.
    let id = |i: u32, j: u32, k: u32| -> u32 {
        if k == 0 {
            i + j * (nx + 1)
        } else if k == nz {
            layer + (nz - 1) * ring + i + j * (nx + 1)
        } else {
            let along = if j == 0 {
                i
            } else if i == nx {
                nx + j
            } else if j == ny {
                nx + ny + (nx - i)
            } else {
                2 * nx + ny + (ny - j)
            };
            layer + (k - 1) * ring + along
        }
    };
    let at = |i: u32, j: u32, k: u32| -> [f32; 3] {
        to_f32([
            size[0] * (f64::from(i) / f64::from(nx) - 0.5),
            size[1] * (f64::from(j) / f64::from(ny) - 0.5),
            size[2] * (f64::from(k) / f64::from(nz) - 0.5),
        ])
    };
    // Only the surface points: whole layers at the ends, rings between.
    for k in 0..=nz {
        let mut put = |i, j| out.positions[id(i, j, k) as usize] = at(i, j, k);
        if k == 0 || k == nz {
            for j in 0..=ny {
                for i in 0..=nx {
                    put(i, j);
                }
            }
        } else {
            for i in 0..=nx {
                put(i, 0);
                put(i, ny);
            }
            for j in 1..ny {
                put(0, j);
                put(nx, j);
            }
        }
    }

    // Each face as a grid over (u, v) whose u × v is the outward normal;
    // `lattice` maps a grid point to its (i, j, k).
    let mut face = |nu: u32, nv: u32, lattice: &dyn Fn(u32, u32) -> (u32, u32, u32)| {
        let v_of = |u, v| {
            let (i, j, k) = lattice(u, v);
            id(i, j, k)
        };
        for v in 0..nv {
            for u in 0..nu {
                let (a, b) = (v_of(u, v), v_of(u + 1, v));
                let (c, d) = (v_of(u + 1, v + 1), v_of(u, v + 1));
                out.triangles.push([a, b, c]);
                out.triangles.push([a, c, d]);
            }
        }
    };
    face(ny, nx, &|u, v| (v, u, 0)); // -Z: y × x
    face(nx, ny, &|u, v| (u, v, nz)); // +Z: x × y
    face(nx, nz, &|u, v| (u, 0, v)); // -Y: x × z
    face(nz, nx, &|u, v| (v, ny, u)); // +Y: z × x
    face(nz, ny, &|u, v| (0, v, u)); // -X: z × y
    face(ny, nz, &|u, v| (nx, u, v)); // +X: y × z
    Ok(out)
}

/// A surface of revolution around the canonical `Z` axis, mapped onto
/// `axis`: each `profile` point `[r, h]` (distance from the axis, height
/// along it) becomes a ring of `segments` vertices, or one vertex where
/// `r` is 0. An open profile runs from a pole (`r = 0`) up the outside to a
/// pole, with every point between off the axis; a closed one (`closed`) is
/// a loop off the axis, counter-clockwise in the `(r, h)` plane. Either way
/// the outside is to the right of the direction of travel, so the triangles
/// come out wound outward.
fn lathe(profile: &[[f64; 2]], closed: bool, segments: u32, axis: Axis) -> Result<AppendSource> {
    let (vertices, triangles) = lathe_size(profile.len() as u128, segments, closed)?;
    let rings = if closed {
        profile.len()
    } else {
        profile.len() - 2
    };
    let mut out = buffers(u128::from(vertices), u128::from(triangles))?;
    let mut trig = Vec::new();
    reserve(&mut trig, segments as usize)?;
    trig.extend((0..segments).map(|j| {
        let phi = TAU * f64::from(j) / f64::from(segments);
        (phi.cos(), phi.sin())
    }));
    let put = |out: &mut AppendSource, [r, h]: [f64; 2], (c, s): (f64, f64)| {
        out.positions.push(to_f32(axis.map([r * c, r * s, h])));
    };
    let interior = if closed {
        profile
    } else {
        put(&mut out, profile[0], (1.0, 0.0));
        &profile[1..profile.len() - 1]
    };
    for &p in interior {
        for &cs in &trig {
            put(&mut out, p, cs);
        }
    }
    let base = u32::from(!closed);
    let ring = |i: u32, j: u32| base + i * segments + j % segments;
    // At most the vertex count, which fits.
    let rings = rings as u32;
    let bands = if closed { rings } else { rings - 1 };
    for i in 0..bands {
        let next = (i + 1) % rings;
        for j in 0..segments {
            let (a, b) = (ring(i, j), ring(i, j + 1));
            let (c, d) = (ring(next, j + 1), ring(next, j));
            out.triangles.push([a, b, c]);
            out.triangles.push([a, c, d]);
        }
    }
    if !closed {
        let top = base + rings * segments;
        put(&mut out, profile[profile.len() - 1], (1.0, 0.0));
        for j in 0..segments {
            out.triangles.push([0, ring(0, j + 1), ring(0, j)]);
            out.triangles
                .push([ring(rings - 1, j), ring(rings - 1, j + 1), top]);
        }
    }
    Ok(out)
}

/// The vertex and triangle counts of a lathe of `points` profile points:
/// two poles and the rings between, with a fan at each pole, or a ring per
/// point when `closed`. Checked before any profile is built, so that huge
/// parameters are refused without allocating.
fn lathe_size(points: u128, segments: u32, closed: bool) -> Result<(u32, u32)> {
    let s = u128::from(segments);
    if closed {
        Ok((ids(points * s)?, ids(2 * points * s)?))
    } else {
        let rings = points - 2;
        Ok((ids(2 + rings * s)?, ids(2 * rings * s)?))
    }
}

/// The open profile of a capped cone, frustum or cylinder: radii `r0` at
/// the bottom and `r1` at the top (one of them may be 0, an apex), over
/// `height`, with `rings` bands on the side and `cap_rings` on each cap.
/// Its size is checked against `segments` first (see [`lathe_size`]).
fn cone_profile(
    r0: f64,
    r1: f64,
    height: f64,
    rings: u32,
    cap_rings: u32,
    segments: u32,
) -> Result<Vec<[f64; 2]>> {
    let half = height / 2.0;
    let caps = |r: f64| if r > 0.0 { u128::from(cap_rings) } else { 0 };
    let points = 2 + caps(r0) + caps(r1) + u128::from(rings) - 1;
    lathe_size(points, segments, false)?;
    let mut p = Vec::new();
    reserve(&mut p, points as usize)?;
    p.push([0.0, -half]);
    if r0 > 0.0 {
        p.extend((1..=cap_rings).map(|c| [r0 * f64::from(c) / f64::from(cap_rings), -half]));
    }
    p.extend((1..rings).map(|k| {
        let t = f64::from(k) / f64::from(rings);
        [r0 + (r1 - r0) * t, height * (t - 0.5)]
    }));
    if r1 > 0.0 {
        p.extend(
            (1..=cap_rings)
                .rev()
                .map(|c| [r1 * f64::from(c) / f64::from(cap_rings), half]),
        );
    }
    p.push([0.0, half]);
    Ok(p)
}

fn radial(segments: u32) -> Result<u32> {
    count(segments, 3, "segments must be at least 3")
}

/// A cylinder; see [`CylinderOptions`].
///
/// # Errors
/// As for [`generate`].
pub fn cylinder(o: &CylinderOptions) -> Result<AppendSource> {
    let r = length(o.radius, "radius must be finite and within [1e-30, 1e30]")?;
    let h = length(o.height, "height must be finite and within [1e-30, 1e30]")?;
    let rings = count(o.rings, 1, "rings must be at least 1")?;
    let caps = count(o.cap_rings, 1, "cap rings must be at least 1")?;
    let segments = radial(o.segments)?;
    let profile = cone_profile(r, r, h, rings, caps, segments)?;
    lathe(&profile, false, segments, o.axis)
}

/// A cone or frustum; see [`ConeOptions`].
///
/// # Errors
/// As for [`generate`].
pub fn cone(o: &ConeOptions) -> Result<AppendSource> {
    let radius = |v: f64| {
        if v == 0.0 {
            Ok(0.0)
        } else {
            length(v, "cone radii must be 0 or finite and within [1e-30, 1e30]")
        }
    };
    let (r0, r1) = (radius(o.radius)?, radius(o.top_radius)?);
    if r0 == 0.0 && r1 == 0.0 {
        return Err(OpsError::InvalidInput("a cone needs a positive radius"));
    }
    let h = length(o.height, "height must be finite and within [1e-30, 1e30]")?;
    let rings = count(o.rings, 1, "rings must be at least 1")?;
    let caps = count(o.cap_rings, 1, "cap rings must be at least 1")?;
    let segments = radial(o.segments)?;
    let profile = cone_profile(r0, r1, h, rings, caps, segments)?;
    lathe(&profile, false, segments, o.axis)
}

/// A UV sphere; see [`UvSphereOptions`].
///
/// # Errors
/// As for [`generate`].
pub fn uv_sphere(o: &UvSphereOptions) -> Result<AppendSource> {
    let r = length(o.radius, "radius must be finite and within [1e-30, 1e30]")?;
    let rings = count(o.rings, 2, "sphere rings must be at least 2")?;
    let segments = radial(o.segments)?;
    lathe_size(u128::from(rings) + 1, segments, false)?;
    let mut profile = Vec::new();
    reserve(&mut profile, rings as usize + 1)?;
    profile.push([0.0, -r]);
    profile.extend((1..rings).map(|i| {
        let theta = PI * (f64::from(i) / f64::from(rings) - 0.5);
        [r * theta.cos(), r * theta.sin()]
    }));
    profile.push([0.0, r]);
    lathe(&profile, false, segments, o.axis)
}

/// A torus; see [`TorusOptions`].
///
/// # Errors
/// As for [`generate`].
pub fn torus(o: &TorusOptions) -> Result<AppendSource> {
    let big = length(
        o.major_radius,
        "major radius must be finite and within [1e-30, 1e30]",
    )?;
    let small = length(
        o.minor_radius,
        "minor radius must be finite and within [1e-30, 1e30]",
    )?;
    if small >= big {
        return Err(OpsError::InvalidInput(
            "a torus needs its minor radius below its major radius",
        ));
    }
    let m = count(o.tube_segments, 3, "tube segments must be at least 3")?;
    let segments = radial(o.segments)?;
    lathe_size(u128::from(m), segments, true)?;
    let mut profile = Vec::new();
    reserve(&mut profile, m as usize)?;
    profile.extend((0..m).map(|k| {
        let theta = TAU * f64::from(k) / f64::from(m);
        [big + small * theta.cos(), small * theta.sin()]
    }));
    lathe(&profile, true, segments, o.axis)
}

/// An arrow; see [`ArrowOptions`].
///
/// # Errors
/// As for [`generate`].
pub fn arrow(o: &ArrowOptions) -> Result<AppendSource> {
    let len = length(o.length, "length must be finite and within [1e-30, 1e30]")?;
    let shaft = length(
        o.shaft_radius,
        "shaft radius must be finite and within [1e-30, 1e30]",
    )?;
    let head = length(
        o.head_radius,
        "head radius must be finite and within [1e-30, 1e30]",
    )?;
    let head_len = length(
        o.head_length,
        "head length must be finite and within [1e-30, 1e30]",
    )?;
    if head <= shaft {
        return Err(OpsError::InvalidInput(
            "an arrow needs its head radius above its shaft radius",
        ));
    }
    if head_len >= len {
        return Err(OpsError::InvalidInput(
            "an arrow needs its head length below its length",
        ));
    }
    let (tail, tip) = (-len / 2.0, len / 2.0);
    let neck = tip - head_len;
    let profile = [
        [0.0, tail],
        [shaft, tail],
        [shaft, neck],
        [head, neck],
        [0.0, tip],
    ];
    lathe(&profile, false, radial(o.segments)?, o.axis)
}

/// A prism; see [`PrismOptions`]. One corner lies on `+Z` for the `Y` axis
/// (the default), on `+X` for `Z` and on `+Y` for `X`.
///
/// # Errors
/// As for [`generate`].
pub fn prism(o: &PrismOptions) -> Result<AppendSource> {
    let sides = count(o.sides, 3, "a prism needs at least 3 sides")?;
    let r = length(o.radius, "radius must be finite and within [1e-30, 1e30]")?;
    let h = length(o.height, "height must be finite and within [1e-30, 1e30]")?;
    lathe(&cone_profile(r, r, h, 1, 1, sides)?, false, sides, o.axis)
}

/// A plane; see [`PlaneOptions`].
///
/// # Errors
/// As for [`generate`].
pub fn plane(o: &PlaneOptions) -> Result<AppendSource> {
    let what = "plane size must be finite and within [1e-30, 1e30]";
    let size = [length(o.size[0], what)?, length(o.size[1], what)?];
    let seg = "plane segments must be at least 1";
    let [nu, nv] = [count(o.segments[0], 1, seg)?, count(o.segments[1], 1, seg)?];
    let (u, v) = (u128::from(nu), u128::from(nv));
    let mut out = buffers((u + 1) * (v + 1), 2 * u * v)?;
    // The in-plane axes in x, y, z order; their cross product is +normal
    // for X and Z, -normal for Y, which flips the winding.
    let n = o.axis.index();
    let (a0, a1) = match o.axis {
        Axis::X => (1, 2),
        Axis::Y => (0, 2),
        Axis::Z => (0, 1),
    };
    let flip = o.axis == Axis::Y;
    for j in 0..=nv {
        for i in 0..=nu {
            let mut p = [0.0; 3];
            p[a0] = size[0] * (f64::from(i) / f64::from(nu) - 0.5);
            p[a1] = size[1] * (f64::from(j) / f64::from(nv) - 0.5);
            p[n] = 0.0;
            out.positions.push(to_f32(p));
        }
    }
    let id = |i: u32, j: u32| i + j * (nu + 1);
    for j in 0..nv {
        for i in 0..nu {
            let (a, b, c, d) = (id(i, j), id(i + 1, j), id(i + 1, j + 1), id(i, j + 1));
            if flip {
                out.triangles.push([a, c, b]);
                out.triangles.push([a, d, c]);
            } else {
                out.triangles.push([a, b, c]);
                out.triangles.push([a, c, d]);
            }
        }
    }
    Ok(out)
}

/// The icosahedron's corners, `(0, ±1, ±φ)` and its cyclic permutations,
/// and its faces wound outward.
const ICOSA_FACES: [[usize; 3]; 20] = [
    [0, 11, 5],
    [0, 5, 1],
    [0, 1, 7],
    [0, 7, 10],
    [0, 10, 11],
    [1, 5, 9],
    [5, 11, 4],
    [11, 10, 2],
    [10, 7, 6],
    [7, 1, 8],
    [3, 9, 4],
    [3, 4, 2],
    [3, 2, 6],
    [3, 6, 8],
    [3, 8, 9],
    [4, 9, 5],
    [2, 4, 11],
    [6, 2, 10],
    [8, 6, 7],
    [9, 8, 1],
];

fn icosa_corners() -> [[f64; 3]; 12] {
    let t = (1.0 + 5f64.sqrt()) / 2.0;
    let raw = [
        [-1.0, t, 0.0],
        [1.0, t, 0.0],
        [-1.0, -t, 0.0],
        [1.0, -t, 0.0],
        [0.0, -1.0, t],
        [0.0, 1.0, t],
        [0.0, -1.0, -t],
        [0.0, 1.0, -t],
        [t, 0.0, -1.0],
        [t, 0.0, 1.0],
        [-t, 0.0, -1.0],
        [-t, 0.0, 1.0],
    ];
    raw.map(unit)
}

fn unit([x, y, z]: [f64; 3]) -> [f64; 3] {
    let n = (x * x + y * y + z * z).sqrt();
    [x / n, y / n, z / n]
}

/// An icosphere; see [`IcosphereOptions`]. Built as a class I geodesic
/// grid of frequency `2^subdivisions` on each icosahedron face, sharing the
/// points on the 30 edges and the 12 corners; every point is projected onto
/// the sphere once.
///
/// # Errors
/// As for [`generate`].
pub fn icosphere(o: &IcosphereOptions) -> Result<AppendSource> {
    let r = length(o.radius, "radius must be finite and within [1e-30, 1e30]")?;
    if o.subdivisions > 16 {
        return Err(OpsError::InvalidInput(
            "icosphere subdivisions must be at most 16",
        ));
    }
    let f = 1u32 << o.subdivisions;
    let ff = u128::from(f) * u128::from(f);
    let mut out = buffers(10 * ff + 2, 20 * ff)?;
    let corners = icosa_corners();
    // The 30 edges, numbered in order of first use.
    let mut edges: Vec<[usize; 2]> = Vec::new();
    reserve(&mut edges, 30)?;
    for face in &ICOSA_FACES {
        for e in 0..3 {
            let (a, b) = (face[e], face[(e + 1) % 3]);
            let key = [a.min(b), a.max(b)];
            if !edges.contains(&key) {
                edges.push(key);
            }
        }
    }
    let edge_of = |a: usize, b: usize| {
        let key = [a.min(b), a.max(b)];
        // Every face edge was numbered above.
        edges.iter().position(|e| *e == key).unwrap_or(0) as u32
    };
    let lerp = |a: [f64; 3], b: [f64; 3], t: f64| {
        [
            a[0] + (b[0] - a[0]) * t,
            a[1] + (b[1] - a[1]) * t,
            a[2] + (b[2] - a[2]) * t,
        ]
    };
    let on_sphere = |p: [f64; 3]| {
        let [x, y, z] = unit(p);
        to_f32([r * x, r * y, r * z])
    };
    for &c in &corners {
        out.positions.push(on_sphere(c));
    }
    // Edge e holds its points at k / f from its lower corner, k = 1..f-1.
    let inner = f - 1;
    for &[a, b] in &edges {
        for k in 1..f {
            let t = f64::from(k) / f64::from(f);
            out.positions
                .push(on_sphere(lerp(corners[a], corners[b], t)));
        }
    }
    let edge_base = 12u32;
    let face_base = edge_base + 30 * inner;
    // Interior points of a face: (f - 1)(f - 2) / 2, rows j = 1..f-2 of
    // f - 1 - j points each.
    let per_face = inner * inner.saturating_sub(1) / 2;
    for &[a, b, c] in &ICOSA_FACES {
        for j in 1..f.saturating_sub(1) {
            for i in 1..f - j {
                let (u, v) = (f64::from(i) / f64::from(f), f64::from(j) / f64::from(f));
                let w = 1.0 - u - v;
                let p = [
                    w * corners[a][0] + u * corners[b][0] + v * corners[c][0],
                    w * corners[a][1] + u * corners[b][1] + v * corners[c][1],
                    w * corners[a][2] + u * corners[b][2] + v * corners[c][2],
                ];
                out.positions.push(on_sphere(p));
            }
        }
    }
    for (n, &[a, b, c]) in ICOSA_FACES.iter().enumerate() {
        let interior = face_base + n as u32 * per_face;
        // The point at k / f along the edge from corner x to corner y.
        let edge = |x: usize, y: usize| (edge_base + edge_of(x, y) * inner, x < y);
        let (ab, ac, bc) = (edge(a, b), edge(a, c), edge(b, c));
        let along =
            |(start, forward): (u32, bool), k: u32| start + if forward { k } else { f - k } - 1;
        // Grid point (i, j): i steps towards b, j towards c.
        let id = |i: u32, j: u32| -> u32 {
            match (i, j) {
                (0, 0) => a as u32,
                (i, 0) if i == f => b as u32,
                (0, j) if j == f => c as u32,
                (i, 0) => along(ab, i),
                (0, j) => along(ac, j),
                (i, j) if i + j == f => along(bc, j),
                (i, j) => {
                    let row = (j - 1) * inner - (j - 1) * j / 2;
                    interior + row + (i - 1)
                }
            }
        };
        for j in 0..f {
            for i in 0..f - j {
                out.triangles.push([id(i, j), id(i + 1, j), id(i, j + 1)]);
                if i + j + 1 < f {
                    out.triangles
                        .push([id(i + 1, j), id(i + 1, j + 1), id(i, j + 1)]);
                }
            }
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use fast_mesh_store::{Mesh, StoreError};

    use super::*;
    use crate::{fail_allocations_after, measure};

    /// What the topology check counted.
    struct Topology {
        vertices: usize,
        edges: usize,
        faces: usize,
        boundary_edges: usize,
    }

    impl Topology {
        fn euler(&self) -> i64 {
            self.vertices as i64 - self.edges as i64 + self.faces as i64
        }
    }

    /// Checks that `src` is welded (no two vertices at the same position),
    /// uses every vertex, has no degenerate triangle, and crosses each edge
    /// at most once per direction, so that two triangles sharing an edge
    /// cross it in opposite directions (consistent winding, manifold edges).
    fn topology(src: &AppendSource) -> Topology {
        let v = src.positions.len();
        assert!(src.positions.iter().flatten().all(|c| c.is_finite()));
        let mut bits: Vec<[u32; 3]> = src.positions.iter().map(|p| p.map(f32::to_bits)).collect();
        bits.sort_unstable();
        bits.dedup();
        assert_eq!(bits.len(), v, "two vertices share a position");
        let mut used = vec![false; v];
        let mut directed: HashMap<(u32, u32), usize> = HashMap::new();
        for t in &src.triangles {
            assert!(
                t[0] != t[1] && t[1] != t[2] && t[0] != t[2],
                "repeated vertex"
            );
            for k in 0..3 {
                let (a, b) = (t[k], t[(k + 1) % 3]);
                assert!((a as usize) < v);
                used[a as usize] = true;
                *directed.entry((a, b)).or_default() += 1;
            }
            let [a, b, c] = t.map(|i| src.positions[i as usize].map(f64::from));
            let n = cross(sub(b, a), sub(c, a));
            assert!(dot(n, n) > 0.0, "degenerate triangle");
        }
        assert!(used.iter().all(|&u| u), "unused vertex");
        let (mut boundary, mut edges) = (0, 0);
        for (&(a, b), &n) in &directed {
            assert_eq!(n, 1, "an edge is crossed twice in the same direction");
            if !directed.contains_key(&(b, a)) {
                boundary += 1;
                edges += 1;
            } else if a < b {
                edges += 1;
            }
        }
        Topology {
            vertices: v,
            edges,
            faces: src.triangles.len(),
            boundary_edges: boundary,
        }
    }

    fn sub(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
        [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
    }

    fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
        [
            a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0],
        ]
    }

    fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
        a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
    }

    fn sinc(x: f64) -> f64 {
        x.sin() / x
    }

    fn mesh(src: &AppendSource) -> Mesh {
        let p: Vec<f32> = src.positions.iter().flatten().copied().collect();
        let t: Vec<u32> = src.triangles.iter().flatten().copied().collect();
        Mesh::from_buffers(&p, &t).unwrap()
    }

    /// The bounding box, `(min, max)`.
    fn bounds(src: &AppendSource) -> ([f64; 3], [f64; 3]) {
        let mut lo = [f64::INFINITY; 3];
        let mut hi = [f64::NEG_INFINITY; 3];
        for p in &src.positions {
            for k in 0..3 {
                lo[k] = lo[k].min(f64::from(p[k]));
                hi[k] = hi[k].max(f64::from(p[k]));
            }
        }
        (lo, hi)
    }

    /// The smallest distance from the origin to a triangle's plane,
    /// negative when a triangle faces the origin.
    fn nearest_plane(src: &AppendSource) -> f64 {
        src.triangles
            .iter()
            .map(|t| {
                let [a, b, c] = t.map(|i| src.positions[i as usize].map(f64::from));
                let n = cross(sub(b, a), sub(c, a));
                dot(n, a) / dot(n, n).sqrt()
            })
            .fold(f64::INFINITY, f64::min)
    }

    /// Rounding of `f32` positions, relative.
    const F32_REL: f64 = 1e-5;

    /// Which coordinates of the bounding box centre are 0: all three, or
    /// only the axis of a shape of revolution (an odd polygon is not
    /// centred across it).
    #[derive(Clone, Copy)]
    enum Centred {
        All,
        Along(Axis),
    }

    /// Checks a closed shape: no boundary edge, Euler characteristic `chi`,
    /// centred on the origin, and an enclosed volume (as [`measure`] reports
    /// it) that is positive (outward winding), within `F32_REL` of
    /// `polyhedral` (the exact volume of the polyhedron the parameters
    /// describe) and within `tolerance` of `analytic` (the smooth shape's).
    /// Returns the volume.
    fn closed(src: &AppendSource, centred: Centred, chi: i64, volumes: [f64; 3]) -> f64 {
        let [polyhedral, analytic, tolerance] = volumes;
        let t = topology(src);
        assert_eq!(t.boundary_edges, 0, "boundary edges on a closed shape");
        assert_eq!(t.euler(), chi);
        let (lo, hi) = bounds(src);
        let scale = (0..3).map(|k| hi[k] - lo[k]).fold(0.0, f64::max);
        let axes = match centred {
            Centred::All => vec![0, 1, 2],
            Centred::Along(a) => vec![a.index()],
        };
        for k in axes {
            assert!(
                (lo[k] + hi[k]).abs() <= 1e-6 * scale,
                "not centred: {lo:?} {hi:?}"
            );
        }
        let m = measure(&mesh(src)).unwrap();
        assert!(m.closed && m.boundary_loops == 0 && m.non_manifold_edges == 0);
        assert_eq!(m.orientation_conflicts, 0);
        let volume = m.volume.unwrap();
        assert!(volume > 0.0, "inward winding");
        let exact = (volume - polyhedral).abs() / polyhedral;
        assert!(
            exact <= F32_REL,
            "volume {volume} vs polyhedral {polyhedral}"
        );
        let err = (volume - analytic).abs() / analytic;
        assert!(
            err <= tolerance + F32_REL,
            "volume {volume} vs {analytic}: {err} > {tolerance}"
        );
        volume
    }

    /// A lathe of `n` sides keeps `sinc(2π/n)` of the solid of revolution:
    /// each wedge between two half-planes `d` apart holds `sin(d)` times the
    /// profile's first moment about the axis, against `d` times it for the
    /// smooth sweep.
    fn lathe_ratio(n: u32) -> f64 {
        sinc(TAU / f64::from(n))
    }

    #[test]
    fn boxes_are_closed_exact_and_split_per_axis() {
        for segments in [[1, 1, 1], [2, 3, 4], [1, 5, 1], [7, 1, 2], [16, 16, 16]] {
            let o = BoxOptions {
                size: [2.0, 3.0, 0.5],
                segments,
            };
            let src = cuboid(&o).unwrap();
            let [x, y, z] = segments.map(|n| n as usize);
            assert_eq!(src.triangles.len(), 4 * (x * y + y * z + z * x));
            closed(&src, Centred::All, 2, [3.0, 3.0, 0.0]);
            assert_eq!(bounds(&src), ([-1.0, -1.5, -0.25], [1.0, 1.5, 0.25]));
        }
    }

    #[test]
    fn cylinders_converge_to_the_analytic_volume() {
        let mut last = f64::INFINITY;
        for (segments, rings, cap_rings) in
            [(3, 1, 1), (8, 2, 3), (32, 1, 1), (128, 4, 2), (1024, 1, 1)]
        {
            let o = CylinderOptions {
                radius: 2.0,
                height: 3.0,
                segments,
                rings,
                cap_rings,
                axis: Axis::Z,
            };
            let src = cylinder(&o).unwrap();
            let analytic = PI * 4.0 * 3.0;
            let ratio = lathe_ratio(segments);
            let v = closed(
                &src,
                Centred::Along(Axis::Z),
                2,
                [analytic * ratio, analytic, 1.0 - ratio],
            );
            let err = analytic - v;
            assert!(err > 0.0 && err < last);
            last = err;
            let (lo, hi) = bounds(&src);
            assert_eq!((lo[2], hi[2]), (-1.5, 1.5));
        }
    }

    #[test]
    fn cones_and_frustums_match_their_volume() {
        for (r0, r1) in [(1.0, 0.0), (0.0, 1.5), (1.0, 0.5), (0.25, 2.0), (1.0, 1.0)] {
            for (segments, rings, cap_rings) in [(3, 1, 1), (24, 3, 2), (256, 1, 1)] {
                let o = ConeOptions {
                    radius: r0,
                    top_radius: r1,
                    height: 2.0,
                    segments,
                    rings,
                    cap_rings,
                    axis: Axis::Y,
                };
                let src = cone(&o).unwrap();
                let analytic = PI * 2.0 / 3.0 * (r0 * r0 + r0 * r1 + r1 * r1);
                let ratio = lathe_ratio(segments);
                closed(
                    &src,
                    Centred::Along(Axis::Y),
                    2,
                    [analytic * ratio, analytic, 1.0 - ratio],
                );
            }
        }
    }

    #[test]
    fn uv_spheres_converge_to_the_ball() {
        let mut last = f64::INFINITY;
        for (segments, rings) in [(3, 2), (8, 4), (32, 16), (128, 64), (512, 256)] {
            let o = UvSphereOptions {
                radius: 1.5,
                segments,
                rings,
                axis: Axis::X,
            };
            let src = uv_sphere(&o).unwrap();
            assert_eq!(src.positions.len(), 2 + (segments * (rings - 1)) as usize);
            // The profile, a polygon of `rings` chords inscribed in the
            // half disc, keeps `cos²(π / 2·rings)` of its first moment.
            let analytic = 4.0 / 3.0 * PI * 1.5f64.powi(3);
            let ratio = lathe_ratio(segments) * (PI / (2.0 * f64::from(rings))).cos().powi(2);
            let v = closed(
                &src,
                Centred::Along(Axis::X),
                2,
                [analytic * ratio, analytic, 1.0 - ratio],
            );
            let err = analytic - v;
            assert!(err > 0.0 && err < last);
            last = err;
            assert!(nearest_plane(&src) > 0.0);
        }
    }

    #[test]
    fn icospheres_converge_to_the_ball() {
        let mut last = f64::INFINITY;
        for subdivisions in 0..=7 {
            let o = IcosphereOptions {
                radius: 0.75,
                subdivisions,
            };
            let src = icosphere(&o).unwrap();
            let f = 1usize << subdivisions;
            assert_eq!(src.triangles.len(), 20 * f * f);
            assert_eq!(src.positions.len(), 10 * f * f + 2);
            for p in &src.positions {
                let d = p.iter().map(|&c| f64::from(c).powi(2)).sum::<f64>().sqrt();
                assert!((d - 0.75).abs() < 1e-6);
            }
            // Inscribed and convex: it holds the ball as far as its nearest
            // face plane, and is held by the sphere.
            let near = nearest_plane(&src);
            assert!(near > 0.0, "a face turned inward");
            let analytic = 4.0 / 3.0 * PI * 0.75f64.powi(3);
            let tolerance = 1.0 - (near / 0.75).powi(3);
            let m = measure(&mesh(&src)).unwrap();
            let v = closed(
                &src,
                Centred::All,
                2,
                [m.volume.unwrap(), analytic, tolerance],
            );
            // The deficit shrinks about fourfold per level.
            let err = analytic - v;
            assert!(err > 0.0 && err < last);
            if subdivisions > 2 {
                assert!(err < last / 3.5);
            }
            last = err;
        }
    }

    #[test]
    fn tori_have_genus_one_and_converge() {
        let mut last = f64::INFINITY;
        for (segments, tube) in [(3, 3), (12, 8), (48, 24), (256, 128), (512, 256)] {
            let o = TorusOptions {
                major_radius: 2.0,
                minor_radius: 0.5,
                segments,
                tube_segments: tube,
                axis: Axis::Z,
            };
            let src = torus(&o).unwrap();
            let analytic = 2.0 * PI * PI * 2.0 * 0.25;
            // The tube polygon keeps `sinc(2π/m)` of the disc's area, with
            // its centroid on the circle of radius `major_radius`.
            let ratio = lathe_ratio(segments) * sinc(TAU / f64::from(tube));
            let v = closed(
                &src,
                Centred::Along(Axis::Z),
                0,
                [analytic * ratio, analytic, 1.0 - ratio],
            );
            let err = analytic - v;
            assert!(err > 0.0 && err < last);
            last = err;
        }
    }

    #[test]
    fn arrows_are_one_closed_shell() {
        for segments in [3, 16, 64, 512] {
            let o = ArrowOptions {
                length: 2.0,
                shaft_radius: 0.1,
                head_radius: 0.3,
                head_length: 0.5,
                segments,
                axis: Axis::Y,
            };
            let src = arrow(&o).unwrap();
            let analytic = PI * 0.01 * 1.5 + PI * 0.09 * 0.5 / 3.0;
            let ratio = lathe_ratio(segments);
            closed(
                &src,
                Centred::Along(Axis::Y),
                2,
                [analytic * ratio, analytic, 1.0 - ratio],
            );
            let (lo, hi) = bounds(&src);
            assert_eq!((lo[1], hi[1]), (-1.0, 1.0));
        }
    }

    #[test]
    fn prisms_are_exact() {
        for sides in [3, 4, 5, 6, 12] {
            let o = PrismOptions {
                sides,
                radius: 1.0,
                height: 2.0,
                axis: Axis::Z,
            };
            let src = prism(&o).unwrap();
            assert_eq!(src.positions.len(), 2 + 2 * sides as usize);
            let n = f64::from(sides);
            let exact = n / 2.0 * (TAU / n).sin() * 2.0;
            closed(&src, Centred::Along(Axis::Z), 2, [exact, exact, 0.0]);
        }
    }

    #[test]
    fn planes_are_open_discs_facing_their_axis() {
        for axis in [Axis::X, Axis::Y, Axis::Z] {
            for segments in [[1, 1], [3, 2], [10, 1]] {
                let o = PlaneOptions {
                    size: [2.0, 4.0],
                    segments,
                    axis,
                };
                let src = plane(&o).unwrap();
                let t = topology(&src);
                let [u, v] = segments.map(|n| n as usize);
                assert_eq!(t.boundary_edges, 2 * (u + v));
                assert_eq!(t.euler(), 1);
                let k = axis.index();
                for tri in &src.triangles {
                    let [a, b, c] = tri.map(|i| src.positions[i as usize].map(f64::from));
                    let n = cross(sub(b, a), sub(c, a));
                    assert!(n[k] > 0.0 && n[(k + 1) % 3] == 0.0 && n[(k + 2) % 3] == 0.0);
                }
                let (lo, hi) = bounds(&src);
                let (a0, a1) = [(1, 2), (0, 2), (0, 1)][k];
                assert_eq!((lo[a0], hi[a0], lo[a1], hi[a1]), (-1.0, 1.0, -2.0, 2.0));
                assert_eq!((lo[k], hi[k]), (0.0, 0.0));
                let m = measure(&mesh(&src)).unwrap();
                assert!((m.area - 8.0).abs() < 1e-9);
                assert!(!m.closed && m.boundary_loops == 1);
            }
        }
    }

    #[test]
    fn the_axis_turns_the_shape_without_changing_it() {
        let shapes = |axis| {
            [
                Primitive::Cylinder(CylinderOptions {
                    axis,
                    height: 3.0,
                    ..Default::default()
                }),
                Primitive::Cone(ConeOptions {
                    axis,
                    height: 3.0,
                    ..Default::default()
                }),
                Primitive::UvSphere(UvSphereOptions {
                    axis,
                    ..Default::default()
                }),
                Primitive::Torus(TorusOptions {
                    axis,
                    ..Default::default()
                }),
                Primitive::Arrow(ArrowOptions {
                    axis,
                    ..Default::default()
                }),
                Primitive::Prism(PrismOptions {
                    axis,
                    height: 3.0,
                    ..Default::default()
                }),
            ]
        };
        for (k, axis) in [Axis::X, Axis::Y, Axis::Z].into_iter().enumerate() {
            for (s, z) in shapes(axis).iter().zip(shapes(Axis::Z)) {
                let (a, b) = (generate(s).unwrap(), generate(&z).unwrap());
                assert_eq!(a.triangles, b.triangles);
                for (p, q) in a.positions.iter().zip(&b.positions) {
                    // The canonical Z component lands on the chosen axis.
                    assert_eq!(p[k], q[2]);
                }
                let va = measure(&mesh(&a)).unwrap().volume.unwrap();
                let vb = measure(&mesh(&b)).unwrap().volume.unwrap();
                assert!((va - vb).abs() <= 1e-12 * vb);
            }
        }
    }

    #[test]
    fn defaults_generate_every_shape() {
        let all = [
            Primitive::Box(BoxOptions::default()),
            Primitive::Cylinder(CylinderOptions::default()),
            Primitive::Cone(ConeOptions::default()),
            Primitive::UvSphere(UvSphereOptions::default()),
            Primitive::Icosphere(IcosphereOptions::default()),
            Primitive::Torus(TorusOptions::default()),
            Primitive::Arrow(ArrowOptions::default()),
            Primitive::Prism(PrismOptions::default()),
            Primitive::Plane(PlaneOptions::default()),
        ];
        for p in &all {
            let src = generate(p).unwrap();
            let t = topology(&src);
            let (open, torus) = (
                matches!(p, Primitive::Plane(_)),
                matches!(p, Primitive::Torus(_)),
            );
            assert_eq!(t.boundary_edges == 0, !open, "{p:?}");
            let chi = if open {
                1
            } else if torus {
                0
            } else {
                2
            };
            assert_eq!(t.euler(), chi, "{p:?}");
            let (lo, hi) = bounds(&src);
            // Every default fits the unit cube around the origin.
            for k in 0..3 {
                assert!(lo[k] >= -0.5 - 1e-6 && hi[k] <= 0.5 + 1e-6, "{p:?}");
            }
        }
    }

    #[test]
    fn bad_parameters_are_refused() {
        let invalid = |r: Result<AppendSource>| matches!(r, Err(OpsError::InvalidInput(_)));
        let b = BoxOptions::default();
        let sized = |size| BoxOptions { size, ..b };
        assert!(invalid(cuboid(&sized([1.0, 0.0, 1.0]))));
        assert!(invalid(cuboid(&sized([f64::NAN, 1.0, 1.0]))));
        assert!(invalid(cuboid(&sized([1e31, 1.0, 1.0]))));
        assert!(invalid(cuboid(&BoxOptions {
            segments: [1, 0, 1],
            ..b
        })));
        let c = CylinderOptions::default();
        assert!(invalid(cylinder(&CylinderOptions { segments: 2, ..c })));
        assert!(invalid(cylinder(&CylinderOptions { rings: 0, ..c })));
        assert!(invalid(cylinder(&CylinderOptions { cap_rings: 0, ..c })));
        assert!(invalid(cylinder(&CylinderOptions { radius: -1.0, ..c })));
        assert!(invalid(cylinder(&CylinderOptions {
            height: f64::INFINITY,
            ..c
        })));
        let k = ConeOptions::default();
        assert!(invalid(cone(&ConeOptions {
            radius: 0.0,
            top_radius: 0.0,
            ..k
        })));
        assert!(invalid(cone(&ConeOptions {
            top_radius: -0.5,
            ..k
        })));
        let s = UvSphereOptions::default();
        assert!(invalid(uv_sphere(&UvSphereOptions { rings: 1, ..s })));
        assert!(invalid(uv_sphere(&UvSphereOptions { segments: 2, ..s })));
        let ico = IcosphereOptions::default();
        assert!(invalid(icosphere(&IcosphereOptions {
            subdivisions: 17,
            ..ico
        })));
        let t = TorusOptions::default();
        assert!(invalid(torus(&TorusOptions {
            minor_radius: 0.4,
            ..t
        })));
        assert!(invalid(torus(&TorusOptions {
            tube_segments: 2,
            ..t
        })));
        let a = ArrowOptions::default();
        assert!(invalid(arrow(&ArrowOptions {
            head_radius: 0.05,
            ..a
        })));
        assert!(invalid(arrow(&ArrowOptions {
            head_length: 1.0,
            ..a
        })));
        assert!(invalid(prism(&PrismOptions {
            sides: 2,
            ..Default::default()
        })));
        let p = PlaneOptions::default();
        assert!(invalid(plane(&PlaneOptions {
            segments: [0, 1],
            ..p
        })));
        assert!(invalid(plane(&PlaneOptions {
            size: [1.0, 0.0],
            ..p
        })));
        // More vertices than u32 ids can name, refused before allocating.
        let too_large = Err(OpsError::InvalidInput(TOO_LARGE));
        let huge = BoxOptions {
            segments: [u32::MAX; 3],
            ..b
        };
        assert_eq!(cuboid(&huge), too_large);
        assert_eq!(
            plane(&PlaneOptions {
                segments: [1 << 20; 2],
                ..p
            }),
            too_large
        );
        let huge = CylinderOptions {
            segments: u32::MAX,
            rings: u32::MAX,
            ..c
        };
        assert_eq!(cylinder(&huge), too_large);
        assert_eq!(
            uv_sphere(&UvSphereOptions {
                segments: u32::MAX,
                rings: u32::MAX,
                ..s
            }),
            too_large
        );
        assert_eq!(
            torus(&TorusOptions {
                segments: u32::MAX,
                tube_segments: u32::MAX,
                ..t
            }),
            too_large
        );
        assert_eq!(
            icosphere(&IcosphereOptions {
                subdivisions: 16,
                ..ico
            }),
            too_large
        );
    }

    #[test]
    fn running_out_of_memory_is_an_error() {
        let all = [
            Primitive::Box(BoxOptions::default()),
            Primitive::Cylinder(CylinderOptions::default()),
            Primitive::UvSphere(UvSphereOptions::default()),
            Primitive::Icosphere(IcosphereOptions::default()),
            Primitive::Torus(TorusOptions::default()),
            Primitive::Plane(PlaneOptions::default()),
        ];
        for p in &all {
            let mut n = 0;
            loop {
                fail_allocations_after(Some(n));
                let r = generate(p);
                fail_allocations_after(None);
                match r {
                    Ok(_) => break,
                    Err(e) => assert_eq!(e, OpsError::Store(StoreError::OutOfMemory)),
                }
                n += 1;
            }
            assert!(n > 0);
        }
    }

    #[test]
    fn outputs_are_deterministic() {
        let p = Primitive::Icosphere(IcosphereOptions {
            subdivisions: 4,
            ..Default::default()
        });
        assert_eq!(generate(&p).unwrap(), generate(&p).unwrap());
    }
}
