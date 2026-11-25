//! Height maps: orthographic depth images of scene meshes, and grids of
//! heights added to the scene as relief meshes.
//!
//! [`Scene::height_map_frame`] fits an image to the meshes, looking along a
//! direction, and [`Scene::height_map`] samples the meshes in it: one ray per
//! pixel, through its centre, whose nearest hit gives the pixel's height
//! above the frame's zero plane. The two are separate so that a caller can
//! sample a frame in bands of rows, several meshes or states of them in one
//! frame, or a frame of its own making.

use std::ops::Range;

use fast_mesh_ops::heightfield::{HeightfieldOptions, heightfield};
use fast_mesh_query::{RAY_TILE, RayGrid};
use fast_mesh_store::{Mesh, VertexId, par};

use crate::placement::{local_to_world, world_to_local};
use crate::query::{
    Aabb64, F32_SLACK, PLACEMENT_SLACK, cached_world_box, local_mag, oom, round_up, try_vec,
    unit_dir,
};
use crate::scene::Entry;
use crate::{MeshId, MeshOptions, Placement, Result, Scene, SceneError};

/// Most pixels of a height map: 2^28, a gigabyte of heights.
pub const MAX_HEIGHT_MAP_PIXELS: u64 = 1 << 28;

/// Pixels from which a height map is sampled in parallel, and (rounded to
/// whole tiles) that one parallel task samples. A ray costs around 100 ns.
const PAR_MIN_PIXELS: usize = 1 << 12;

/// Vertices per parallel task when fitting a frame, and from which to.
const BOUNDS_CHUNK: usize = 1 << 14;

/// How [`Scene::height_map_frame`] fits an image to the meshes.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HeightMapOptions {
    /// The world direction the image looks along; heights are measured
    /// against it. Any length. Default `[0, 0, -1]`, down the Z axis.
    pub direction: [f64; 3],
    /// The world direction that is up in the image, made perpendicular to
    /// `direction`. `None` (the default) takes `+Y`, or looking along the Y
    /// axis, `-Z` from above and `+Z` from below (`+X` is right in both).
    pub up: Option<[f64; 3]>,
    /// Pixels along the longer side of the image, at least 2. The other side
    /// follows from the meshes' proportions: pixels are square. Default 1024.
    pub resolution: u32,
    /// The world size of a pixel, instead of `resolution`.
    pub pixel_size: Option<f64>,
    /// World distance to leave free around the meshes. Default 0.
    pub margin: f64,
    /// The heights the image covers, `[low, high]`, as world coordinates
    /// along `-direction`: `low` is height 0, and only surfaces between the
    /// two are seen. `None` (the default) covers the meshes exactly.
    pub range: Option<[f64; 2]>,
}

impl Default for HeightMapOptions {
    fn default() -> Self {
        HeightMapOptions {
            direction: [0.0, 0.0, -1.0],
            up: None,
            resolution: 1024,
            pixel_size: None,
            margin: 0.0,
            range: None,
        }
    }
}

/// Where a height map's pixels are in the world.
///
/// Pixel `(col, row)`, rows from the top, at height `h` is the world point
/// `origin + col * pixel_size * right - row * pixel_size * up + h * normal`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HeightMapFrame {
    /// Pixels along a row, at least 1.
    pub width: u32,
    /// Rows, at least 1.
    pub height: u32,
    /// The centre of the top left pixel, at height 0.
    pub origin: [f64; 3],
    /// The image's right, a unit vector.
    pub right: [f64; 3],
    /// The image's up, a unit vector perpendicular to `right`.
    pub up: [f64; 3],
    /// Distance between the centres of neighbouring pixels.
    pub pixel_size: f64,
    /// The largest height, at least 0: the image covers heights from 0 to
    /// this.
    pub range: f64,
}

impl HeightMapFrame {
    /// The direction heights grow in, towards the viewer: `right × up`.
    pub fn normal(&self) -> [f64; 3] {
        cross(self.right, self.up)
    }

    /// The world point of pixel `(col, row)` (which may be a fraction, or
    /// outside the image) at `height`.
    pub fn point(&self, col: f64, row: f64, height: f64) -> [f64; 3] {
        let n = self.normal();
        let (u, v) = (col * self.pixel_size, -row * self.pixel_size);
        std::array::from_fn(|k| self.origin[k] + u * self.right[k] + v * self.up[k] + height * n[k])
    }

    /// The placement that puts a relief of the map's heights
    /// ([`Scene::add_heightfield`] with this frame's pixel size) where the
    /// map saw them.
    pub fn relief_placement(&self) -> Placement {
        let n = self.normal();
        let centre = self.point(
            0.5 * f64::from(self.width.max(1) - 1),
            0.5 * f64::from(self.height.max(1) - 1),
            0.0,
        );
        Placement {
            rotation: std::array::from_fn(|k| [self.right[k], self.up[k], n[k]]),
            translation: centre,
            scale: [1.0; 3],
        }
    }

    /// Checks that the frame is finite, its axes are orthonormal within
    /// `1e-6`, its pixel size is within `[1e-30, 1e30]`, its range is not
    /// negative, and it has from 1 to [`MAX_HEIGHT_MAP_PIXELS`] pixels.
    ///
    /// # Errors
    /// [`SceneError::InvalidInput`] otherwise.
    pub fn validate(&self) -> Result<()> {
        let bad = |what| Err(SceneError::InvalidInput(what));
        let finite = |v: &[f64; 3]| v.iter().all(|c| c.is_finite());
        if ![self.origin, self.right, self.up].iter().all(finite) {
            return bad("a height map frame must be finite");
        }
        let near = |x: f64, want: f64| (x - want).abs() <= 1e-6;
        if !near(dot(self.right, self.right), 1.0)
            || !near(dot(self.up, self.up), 1.0)
            || !near(dot(self.right, self.up), 0.0)
        {
            return bad("a height map frame's right and up must be perpendicular unit vectors");
        }
        if !(1e-30..=1e30).contains(&self.pixel_size) {
            return bad("a height map's pixel size must be within [1e-30, 1e30]");
        }
        if !(0.0..=1e30).contains(&self.range) {
            return bad("a height map's range must be within [0, 1e30]");
        }
        check_pixels(self.width, self.height)
    }
}

/// Outcome of [`Scene::add_heightfield`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HeightfieldReport {
    /// The new mesh holding the relief.
    pub added: MeshId,
    /// Vertices of the relief.
    pub vertices: usize,
    /// Triangles of the relief.
    pub triangles: usize,
}

fn check_pixels(width: u32, height: u32) -> Result<()> {
    if width == 0 || height == 0 {
        return Err(SceneError::InvalidInput(
            "a height map needs at least one pixel on each side",
        ));
    }
    if u64::from(width) * u64::from(height) > MAX_HEIGHT_MAP_PIXELS {
        return Err(SceneError::InvalidInput(
            "a height map has at most 2^28 pixels: lower its resolution",
        ));
    }
    Ok(())
}

fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

/// The image's right and up, looking along `direction` (a unit vector).
fn image_axes(direction: [f64; 3], up: Option<[f64; 3]>) -> Result<[[f64; 3]; 2]> {
    let hint = match up {
        Some(up) => unit_dir(up).map_err(|_| {
            SceneError::InvalidInput("a height map's up must be finite and non-zero")
        })?,
        // Within half a degree of the Y axis.
        None if direction[1].abs() > 0.99996 => [0.0, 0.0, direction[1].signum()],
        None => [0.0, 1.0, 0.0],
    };
    let right = cross(direction, hint);
    if dot(right, right) < 1e-12 {
        return Err(SceneError::InvalidInput(
            "a height map's up must not be parallel to its direction",
        ));
    }
    let right = unit_dir(right)?;
    Ok([right, unit_dir(cross(right, direction))?])
}

/// One mesh to sample: its ray grid and the pixels it can cover.
struct Target<'a> {
    grid: RayGrid<'a>,
    /// The height its rays start at: above everything of the mesh that the
    /// image shows.
    start: f64,
    /// The largest `t` that counts: just below height 0.
    max_t: f32,
    cols: Range<usize>,
    rows: Range<usize>,
}

impl Scene {
    /// Fits a height map's frame to the meshes `ids`: looking along
    /// [`HeightMapOptions::direction`], an image of square pixels around the
    /// vertices their live triangles use, centred on them, with their height
    /// range. At least 2 pixels on each side, so that a relief can be made of
    /// the map. A pixel's height is that of the surface at its centre, so the
    /// outermost samples lie up to half a pixel inside the meshes' outline.
    ///
    /// # Errors
    /// [`SceneError::UnknownMesh`] for a stale id;
    /// [`SceneError::InvalidInput`] for an empty list, meshes without a
    /// triangle or without an extent across the direction, a direction or up
    /// that is zero, not finite or parallel to each other, a resolution below
    /// 2, a pixel size that is not finite and positive, a margin that is
    /// negative or not finite, a range that is not finite and increasing, or
    /// an image over [`MAX_HEIGHT_MAP_PIXELS`]; [`SceneError::Store`] when
    /// out of memory.
    pub fn height_map_frame(
        &self,
        ids: &[MeshId],
        options: &HeightMapOptions,
    ) -> Result<HeightMapFrame> {
        let bad = |what| Err(SceneError::InvalidInput(what));
        if ids.is_empty() {
            return bad("a height map needs at least one mesh");
        }
        let direction = unit_dir(options.direction).map_err(|_| {
            SceneError::InvalidInput("a height map's direction must be finite and non-zero")
        })?;
        let [right, up] = image_axes(direction, options.up)?;
        let normal = cross(right, up);
        if options.resolution < 2 {
            return bad("a height map's resolution must be at least 2");
        }
        if options
            .pixel_size
            .is_some_and(|s| !(1e-30..=1e30).contains(&s))
        {
            return bad("a height map's pixel size must be within [1e-30, 1e30]");
        }
        if !(0.0..=1e30).contains(&options.margin) {
            return bad("a height map's margin must be finite and not negative");
        }
        if let Some([low, high]) = options.range
            && !(low.is_finite() && high.is_finite() && low < high)
        {
            return bad("a height map's range must be finite, its low below its high");
        }

        let mut bounds = Aabb64::EMPTY;
        for &id in ids {
            let b = frame_bounds(self.entry(id)?, [right, up, normal])?;
            for k in 0..3 {
                bounds.min[k] = bounds.min[k].min(b.min[k]);
                bounds.max[k] = bounds.max[k].max(b.max[k]);
            }
        }
        if bounds.is_empty() {
            return bad("the meshes have no triangles");
        }
        let extent: [f64; 2] =
            std::array::from_fn(|k| bounds.max[k] - bounds.min[k] + 2.0 * options.margin);
        let longer = extent[0].max(extent[1]);
        if longer.is_nan() || longer <= 0.0 {
            return bad("the meshes have no extent across the height map's direction");
        }
        let pixel_size = options
            .pixel_size
            .unwrap_or(longer / f64::from(options.resolution));
        // The slack keeps an extent that is a whole number of pixels from
        // rounding up to one more.
        let [width, height] = extent.map(|e| (e / pixel_size - 1e-9).ceil().clamp(2.0, 4e9) as u32);
        check_pixels(width, height)?;
        let [low, high] = options.range.unwrap_or([bounds.min[2], bounds.max[2]]);
        let centre: [f64; 2] = std::array::from_fn(|k| 0.5 * bounds.min[k] + 0.5 * bounds.max[k]);
        let u = centre[0] - 0.5 * pixel_size * f64::from(width - 1);
        let v = centre[1] + 0.5 * pixel_size * f64::from(height - 1);
        let frame = HeightMapFrame {
            width,
            height,
            origin: std::array::from_fn(|k| u * right[k] + v * up[k] + low * normal[k]),
            right,
            up,
            pixel_size,
            range: high - low,
        };
        frame.validate()?;
        Ok(frame)
    }

    /// Samples the meshes `ids` in `frame`: the heights of the pixels of
    /// `rows`, row by row from the left, in `[0, frame.range]`, NaN where no
    /// surface is seen. A pixel's ray runs through its centre against the
    /// frame's normal, and its height is that of the first surface it meets
    /// within the range; both sides of a triangle count, and triangles seen
    /// edge on do not. A ray exactly through an edge or vertex gets the
    /// nearest of the triangles meeting there, so a watertight mesh has no
    /// cracks. A surface within rounding of either end of the range counts.
    ///
    /// Rays run in parallel with the `parallel` feature, tiles of them
    /// sharing their walk of each BVH. A pixel depends on nothing but its own
    /// ray, so the heights are the same, bit for bit, with or without the
    /// feature, and whichever bands of rows a map is sampled in.
    ///
    /// The BVHs of the meshes named are brought up to date first (§15.4).
    ///
    /// # Errors
    /// [`SceneError::InvalidInput`] for an invalid frame (see
    /// [`HeightMapFrame::validate`]), an empty list or rows outside the
    /// frame; [`SceneError::UnknownMesh`] for a stale id; a BVH or memory
    /// error from a Lazy update, or [`SceneError::StaleBvh`] for the lowest
    /// stale Manual mesh named; [`SceneError::Store`] when out of memory.
    pub fn height_map(
        &mut self,
        ids: &[MeshId],
        frame: &HeightMapFrame,
        rows: Range<u32>,
    ) -> Result<Vec<f32>> {
        self.drop_field_unit();
        frame.validate()?;
        if ids.is_empty() {
            return Err(SceneError::InvalidInput(
                "a height map needs at least one mesh",
            ));
        }
        if rows.start > rows.end || rows.end > frame.height {
            return Err(SceneError::InvalidInput(
                "height map rows must lie within the frame",
            ));
        }
        for &id in ids {
            self.entry(id)?;
        }
        self.prepare(Some(ids))?;
        let mut boxes: Vec<(MeshId, Aabb64)> = try_vec(ids.len())?;
        for &id in ids {
            if boxes.iter().all(|&(seen, _)| seen != id) {
                boxes.push((id, cached_world_box(self.entry_mut(id)?).0));
            }
        }
        let mut targets: Vec<Target> = try_vec(boxes.len())?;
        for &(id, wbox) in &boxes {
            targets.extend(target(self.entry(id)?, &wbox, frame)?);
        }
        // Nearest first, so that the meshes behind are mostly skipped.
        targets.sort_by(|a, b| b.start.total_cmp(&a.start));

        let width = frame.width as usize;
        let count = rows.len() * width;
        let mut heights: Vec<f32> = try_vec(count)?;
        heights.resize(count, f32::NEG_INFINITY);
        let band = RAY_TILE * (PAR_MIN_PIXELS / (RAY_TILE * width)).max(1);
        let first = rows.start as usize;
        par::for_each_chunk_mut(&mut heights, band * width, PAR_MIN_PIXELS, |i, chunk| {
            let row = first + i * band;
            for (k, strip) in chunk.chunks_mut(RAY_TILE * width).enumerate() {
                sample_strip(&targets, frame, row + k * RAY_TILE, strip);
            }
            for h in chunk {
                if *h == f32::NEG_INFINITY {
                    *h = f32::NAN;
                }
            }
        });
        Ok(heights)
    }

    /// Adds the relief of `heights`, a `width` by `height` grid in rows from
    /// the top left sample, as a new closed, outward-wound mesh at
    /// `placement`: a solid from its base to the heights (see
    /// [`fast_mesh_ops::heightfield`]). With a height map's heights, pixel
    /// size and [`HeightMapFrame::relief_placement`], it lies on the surface
    /// the map saw.
    ///
    /// The new mesh is added as one undo step (an undoable add, see
    /// [`MeshOptions::undoable`]): undo removes it again and redo brings it
    /// back under the same id.
    ///
    /// # Errors
    /// [`SceneError::InvalidInput`] for an invalid placement;
    /// [`SceneError::Ops`] with
    /// [`OpsError::InvalidInput`](fast_mesh_ops::OpsError::InvalidInput) for an invalid
    /// grid (see [`heightfield`]); [`SceneError::Store`] when out of memory.
    /// The scene is unchanged on error.
    pub fn add_heightfield(
        &mut self,
        heights: &[f32],
        width: u32,
        height: u32,
        options: &HeightfieldOptions,
        placement: Placement,
    ) -> Result<HeightfieldReport> {
        self.drop_field_unit();
        placement.validate()?;
        let relief = heightfield(heights, width, height, options)?;
        let mesh = Mesh::from_buffers(
            relief.positions.as_flattened(),
            relief.triangles.as_flattened(),
        )?;
        let (vertices, triangles) = (relief.positions.len(), relief.triangles.len());
        drop(relief);
        let added = self.add_mesh(
            mesh,
            MeshOptions {
                placement,
                undoable: true,
                ..MeshOptions::default()
            },
        )?;
        Ok(HeightfieldReport {
            added,
            vertices,
            triangles,
        })
    }
}

/// The bounds, along `axes`, of the world vertices that the entry's live
/// triangles use (empty without any).
fn frame_bounds(e: &Entry, axes: [[f64; 3]; 3]) -> Result<Aabb64> {
    let (mesh, placement, origin) = (&e.mesh, &e.placement, e.mesh.origin());
    let positions = mesh.positions();
    let parts = par::map_range(positions.len(), BOUNDS_CHUNK, BOUNDS_CHUNK, oom, |range| {
        let mut b = Aabb64::EMPTY;
        for v in range {
            let id = v as VertexId;
            if mesh.is_vertex_alive(id) && mesh.corner_count(id) > 0 {
                let w = local_to_world(placement, origin, positions[v].map(f64::from));
                for (k, &axis) in axes.iter().enumerate() {
                    let x = dot(w, axis);
                    b.min[k] = b.min[k].min(x);
                    b.max[k] = b.max[k].max(x);
                }
            }
        }
        Ok(b)
    })?;
    Ok(parts.into_iter().fold(Aabb64::EMPTY, |a, b| Aabb64 {
        min: std::array::from_fn(|k| a.min[k].min(b.min[k])),
        max: std::array::from_fn(|k| a.max[k].max(b.max[k])),
    }))
}

/// What to cast at the entry to see it in `frame`: `None` when its world box
/// `wbox` is empty or outside the image.
fn target<'a>(e: &'a Entry, wbox: &Aabb64, frame: &HeightMapFrame) -> Result<Option<Target<'a>>> {
    if wbox.is_empty() {
        return Ok(None);
    }
    let normal = frame.normal();
    let (c, h) = wbox.centre_half();
    let from_origin: [f64; 3] = std::array::from_fn(|k| c[k] - frame.origin[k]);
    // The box along an axis of the frame: its middle and half length.
    let along = |axis: [f64; 3]| {
        let half: f64 = (0..3).map(|k| axis[k].abs() * h[k]).sum();
        (dot(from_origin, axis), half)
    };
    // How far a placed point can be from where the world box puts it, and
    // from where a ray meets it: the placement's rotation is orthonormal
    // only within 1e-6, and the rays run in `f32`.
    let size = h[0] + h[1] + h[2];
    let from_t: f64 = (0..3)
        .map(|k| (c[k] - e.placement.translation[k]).abs())
        .sum();
    let reach = e.placement.max_scale() * local_mag(e);
    let pad = PLACEMENT_SLACK * (size + from_t) + F32_SLACK * (size + reach) + f64::MIN_POSITIVE;
    let (n, half) = along(normal);
    // Rounding at the ends of the range, which keeps surfaces exactly on
    // them.
    let slack = F32_SLACK * (frame.range + size + reach) + f64::MIN_POSITIVE;
    let start = (n + half + pad).min(frame.range + slack);
    let bottom = n - half - pad;
    if n + half + pad < -slack || bottom > frame.range + slack {
        return Ok(None);
    }
    let span = |(mid, half): (f64, f64), len: u32| -> Option<Range<usize>> {
        let len = f64::from(len);
        let from = ((mid - half - pad) / frame.pixel_size - 1.0)
            .floor()
            .clamp(0.0, len);
        let to = ((mid + half + pad) / frame.pixel_size + 2.0)
            .ceil()
            .clamp(0.0, len);
        (from < to).then_some(from as usize..to as usize)
    };
    let down = frame.up.map(|c| -c);
    let Some(cols) = span(along(frame.right), frame.width) else {
        return Ok(None);
    };
    let Some(rows) = span(along(down), frame.height) else {
        return Ok(None);
    };
    let p = &e.placement;
    let local = |v: [f64; 3]| p.inverse_direction(v.map(|c| c * frame.pixel_size));
    let grid = RayGrid::new(
        &e.mesh,
        &e.bvh,
        world_to_local(p, e.mesh.origin(), frame.point(0.0, 0.0, start)),
        p.inverse_direction(normal.map(|c| -c)),
        local(frame.right),
        local(down),
    )?;
    Ok(Some(Target {
        grid,
        start,
        max_t: round_up(start + slack),
        cols,
        rows,
    }))
}

/// Raises the heights of `strip` (up to [`RAY_TILE`] rows from `row`,
/// `NEG_INFINITY` where nothing is seen yet) to what `targets` show.
fn sample_strip(targets: &[Target], frame: &HeightMapFrame, row: usize, strip: &mut [f32]) {
    let width = frame.width as usize;
    let rows = strip.len() / width;
    for col in (0..width).step_by(RAY_TILE) {
        let cols = RAY_TILE.min(width - col);
        let mut lowest = f32::NEG_INFINITY;
        for target in targets {
            // Nothing of it, or of any target after it, is above its start.
            if f64::from(lowest) >= target.start {
                break;
            }
            let misses = |range: &Range<usize>, at: usize, n: usize| {
                range.end <= at || range.start >= at + n
            };
            if misses(&target.cols, col, cols) || misses(&target.rows, row, rows) {
                continue;
            }
            let mut t = [0.0; RAY_TILE * RAY_TILE];
            for r in 0..rows {
                for c in 0..cols {
                    let best = strip[r * width + col + c];
                    t[r * RAY_TILE + c] = if best == f32::NEG_INFINITY {
                        target.max_t
                    } else {
                        round_up(target.start - f64::from(best)).min(target.max_t)
                    };
                }
            }
            let mut hit = target
                .grid
                .nearest_in_tile(col as u32, row as u32, cols, rows, 0.0, &mut t);
            while hit != 0 {
                let i = hit.trailing_zeros() as usize;
                hit &= hit - 1;
                let h = (target.start - f64::from(t[i])).clamp(0.0, frame.range) as f32;
                let best = &mut strip[(i / RAY_TILE) * width + col + i % RAY_TILE];
                *best = best.max(h);
            }
            lowest = f32::INFINITY;
            for line in strip.chunks(width) {
                lowest = line[col..col + cols].iter().fold(lowest, |m, &h| m.min(h));
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use fast_mesh_ops::OpsError;
    use fast_mesh_ops::measure;
    use fast_mesh_ops::primitives::{
        self, BoxOptions, IcosphereOptions, PlaneOptions, Primitive, TorusOptions,
    };

    use super::*;
    use crate::{SceneOptions, SceneRayOptions, UpdateMode};

    fn scene() -> Scene {
        Scene::new(SceneOptions::default())
    }

    fn mesh(primitive: &Primitive) -> Mesh {
        let src = primitives::generate(primitive).unwrap();
        Mesh::from_buffers(src.positions.as_flattened(), src.triangles.as_flattened()).unwrap()
    }

    fn cuboid(size: [f64; 3]) -> Mesh {
        mesh(&Primitive::Box(BoxOptions {
            size,
            segments: [4, 2, 1],
        }))
    }

    /// An icosphere, with the radius of its inscribed sphere.
    fn sphere(radius: f64) -> (Mesh, f64) {
        let m = mesh(&Primitive::Icosphere(IcosphereOptions {
            radius,
            subdivisions: 6,
        }));
        let nearest = m.triangles().iter().map(|t| {
            let [a, b, c] = t.map(|v| m.positions()[v as usize].map(f64::from));
            let e = |p: [f64; 3]| -> [f64; 3] { std::array::from_fn(|k| p[k] - a[k]) };
            let n = cross(e(b), e(c));
            dot(n, a) / dot(n, n).sqrt()
        });
        let inner = nearest.fold(f64::INFINITY, f64::min);
        (m, inner)
    }

    fn add(scene: &mut Scene, mesh: Mesh, placement: Placement) -> MeshId {
        let options = MeshOptions {
            placement,
            ..MeshOptions::default()
        };
        scene.add_mesh(mesh, options).unwrap()
    }

    fn at(translation: [f64; 3]) -> Placement {
        Placement {
            translation,
            ..Placement::IDENTITY
        }
    }

    /// A rotation by `deg` degrees about the axis `(1, 2, 2) / 3`.
    fn tilted(deg: f64) -> [[f64; 3]; 3] {
        let (s, c) = deg.to_radians().sin_cos();
        let u = [1.0 / 3.0, 2.0 / 3.0, 2.0 / 3.0];
        let skew = [[0.0, -u[2], u[1]], [u[2], 0.0, -u[0]], [-u[1], u[0], 0.0]];
        std::array::from_fn(|i| {
            std::array::from_fn(|j| {
                let id = if i == j { 1.0 } else { 0.0 };
                c * id + s * skew[i][j] + (1.0 - c) * u[i] * u[j]
            })
        })
    }

    fn all(scene: &mut Scene, ids: &[MeshId], frame: &HeightMapFrame) -> Vec<f32> {
        scene.height_map(ids, frame, 0..frame.height).unwrap()
    }

    fn bits(v: &[f32]) -> Vec<u32> {
        v.iter().map(|h| h.to_bits()).collect()
    }

    /// Each pixel's `(col, row)`, as numbers.
    fn pixels(frame: &HeightMapFrame) -> impl Iterator<Item = (f64, f64)> + use<> {
        let (w, h) = (frame.width, frame.height);
        (0..h).flat_map(move |r| (0..w).map(move |c| (f64::from(c), f64::from(r))))
    }

    const AXES: [[f64; 3]; 6] = [
        [1.0, 0.0, 0.0],
        [-1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, -1.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, 0.0, -1.0],
    ];

    #[test]
    fn boxes_have_their_analytic_depth_from_every_side() {
        let mut s = scene();
        let centre = [3.0, -2.0, 7.0];
        let size = [2.0, 3.0, 1.0];
        let id = add(&mut s, cuboid(size), at(centre));
        for direction in AXES {
            let axis = (0..3).find(|&k| direction[k] != 0.0).unwrap();
            let options = HeightMapOptions {
                direction,
                resolution: 48,
                ..HeightMapOptions::default()
            };
            let frame = s.height_map_frame(&[id], &options).unwrap();
            assert!((frame.range - size[axis]).abs() < 1e-12);
            assert_eq!(frame.normal(), direction.map(|c| 0.0 - c));
            assert_eq!(frame.width.max(frame.height), 48);
            let heights = all(&mut s, &[id], &frame);
            assert_eq!(heights.len(), (frame.width * frame.height) as usize);
            for &h in &heights {
                assert!(
                    (f64::from(h) - size[axis]).abs() < 1e-6,
                    "{direction:?} {h}"
                );
            }
            // The image covers the side seen, its middle on the box's.
            let extent = [frame.right, frame.up].map(|a| {
                let k = (0..3).find(|&k| a[k].abs() > 0.5).unwrap();
                size[k]
            });
            let covered = [frame.width, frame.height].map(|n| f64::from(n) * frame.pixel_size);
            for k in 0..2 {
                assert!(covered[k] >= extent[k] - 1e-9);
                assert!(covered[k] < extent[k] + frame.pixel_size);
            }
            let middle = frame.point(
                0.5 * f64::from(frame.width - 1),
                0.5 * f64::from(frame.height - 1),
                0.5 * frame.range,
            );
            for k in 0..3 {
                assert!((middle[k] - centre[k]).abs() < 1e-9);
            }
        }
        // With a margin, the pixels past the outline see nothing.
        let options = HeightMapOptions {
            resolution: 50,
            margin: 0.7,
            ..HeightMapOptions::default()
        };
        let frame = s.height_map_frame(&[id], &options).unwrap();
        let heights = all(&mut s, &[id], &frame);
        for ((c, r), h) in pixels(&frame).zip(heights) {
            let p = frame.point(c, r, 0.0);
            let d = [0, 1].map(|k| (p[k] - centre[k]).abs() - 0.5 * size[k]);
            if d[0].max(d[1]) < -1e-9 {
                assert!((h - 1.0).abs() < 1e-6);
            } else if d[0].max(d[1]) > 1e-9 {
                assert!(h.is_nan());
            }
        }
    }

    #[test]
    fn looking_down_keeps_the_usual_axes() {
        let mut s = scene();
        let id = add(&mut s, cuboid([1.0; 3]), Placement::IDENTITY);
        let axes = |direction, up| {
            let options = HeightMapOptions {
                direction,
                up,
                ..HeightMapOptions::default()
            };
            let f = s.height_map_frame(&[id], &options).unwrap();
            (f.right, f.up)
        };
        let (x, y, z) = ([1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]);
        let neg = |v: [f64; 3]| v.map(|c: f64| 0.0 - c);
        assert_eq!(axes(neg(z), None), (x, y));
        assert_eq!(axes(z, None), (neg(x), y));
        assert_eq!(axes(neg(y), None), (x, neg(z)));
        assert_eq!(axes(y, None), (x, z));
        assert_eq!(axes(neg(x), None), (neg(z), y));
        assert_eq!(axes([0.0, 0.0, -9.0], Some([5.0, 0.0, -3.0])), (neg(y), x));
    }

    #[test]
    fn spheres_have_their_analytic_depth_along_any_direction() {
        let (m, inner) = sphere(2.0);
        let mut s = scene();
        let placement = Placement {
            rotation: tilted(35.0),
            translation: [1.0, 2.0, 3.0],
            scale: [1.5; 3],
        };
        let id = add(&mut s, m, placement);
        let radius = 3.0;
        // The facets lie within this of the sphere, towards its centre.
        let sag = radius * (1.0 - inner / 2.0);
        for direction in [[0.0, 0.0, -1.0], [1.0, 2.0, 3.0], [-4.0, 1.0, 0.5]] {
            let options = HeightMapOptions {
                direction,
                resolution: 200,
                margin: 0.1,
                ..HeightMapOptions::default()
            };
            let frame = s.height_map_frame(&[id], &options).unwrap();
            assert!((frame.range - 2.0 * radius).abs() < 2.0 * sag + 1e-6);
            let heights = all(&mut s, &[id], &frame);
            let normal = frame.normal();
            let mut seen = 0;
            for ((c, r), h) in pixels(&frame).zip(heights) {
                let p = frame.point(c, r, 0.0);
                let d: [f64; 3] = std::array::from_fn(|k| p[k] - placement.translation[k]);
                let depth = dot(d, normal);
                let off = (dot(d, d) - depth * depth).max(0.0).sqrt();
                if off > radius {
                    assert!(h.is_nan(), "{off}");
                } else if off < 0.95 * radius {
                    seen += 1;
                    let top = (radius * radius - off * off).sqrt();
                    let got = f64::from(h) + depth;
                    // Along the ray, the sag is larger where it grazes.
                    let cos = (top - sag) / radius;
                    let slack = 2e-6 * radius;
                    assert!(got <= top + slack && got >= top - sag / cos - slack);
                }
            }
            assert!(seen > 20_000);
        }
    }

    #[test]
    fn the_nearest_mesh_wins_whatever_the_order() {
        let mut s = scene();
        let low = add(&mut s, cuboid([4.0, 4.0, 1.0]), Placement::IDENTITY);
        let high = add(&mut s, cuboid([2.0, 2.0, 1.0]), at([0.0, 0.0, 3.0]));
        let hidden = add(&mut s, cuboid([1.0, 1.0, 1.0]), at([0.0, 0.0, 0.25]));
        let aside = add(&mut s, cuboid([1.0; 3]), at([50.0, 0.0, 0.0]));
        let options = HeightMapOptions {
            resolution: 64,
            ..HeightMapOptions::default()
        };
        let frame = s.height_map_frame(&[low, high], &options).unwrap();
        assert_eq!((frame.width, frame.range), (64, 4.0));
        let want = all(&mut s, &[low, high], &frame);
        for ((c, r), &h) in pixels(&frame).zip(&want) {
            let p = frame.point(c, r, 0.0);
            let top = if p[0].abs() < 1.0 && p[1].abs() < 1.0 {
                4.0
            } else {
                1.0
            };
            assert!((h - top).abs() < 1e-6);
        }
        // A mesh hidden behind, one outside the image, repeats and order
        // change nothing.
        for ids in [
            vec![high, low],
            vec![low, high, low],
            vec![hidden, low, aside, high],
        ] {
            assert_eq!(bits(&all(&mut s, &ids, &frame)), bits(&want));
        }
        // On its own the mesh outside the image shows nothing.
        assert!(all(&mut s, &[aside], &frame).iter().all(|h| h.is_nan()));
        // In another mesh's frame: the box under, its top at 0.75 - 0.5 + 0.5.
        let h = all(&mut s, &[hidden], &frame);
        let seen: Vec<f32> = h.into_iter().filter(|h| !h.is_nan()).collect();
        assert_eq!(seen.len(), 16 * 16);
        assert!(seen.iter().all(|h| (h - 1.25).abs() < 1e-6));
    }

    #[test]
    fn heights_match_a_scene_raycast_per_pixel() {
        // Rotated, non-uniformly scaled and off-centre meshes, and one far
        // from the origin: the same surface as the scene's own raycast.
        for offset in [[0.0; 3], [1e6, -2e6, 5e5]] {
            let mut s = scene();
            let torus = mesh(&Primitive::Torus(TorusOptions::default()));
            let a = add(
                &mut s,
                torus,
                Placement {
                    rotation: tilted(50.0),
                    translation: [offset[0] + 0.3, offset[1], offset[2] - 0.2],
                    scale: [1.0, 2.5, 0.6],
                },
            );
            let b = add(
                &mut s,
                cuboid([0.5, 0.4, 0.3]),
                Placement {
                    rotation: tilted(-20.0),
                    translation: [offset[0], offset[1] + 0.4, offset[2]],
                    scale: [2.0; 3],
                },
            );
            for direction in [[0.0, 0.0, -1.0], [0.3, -1.0, 0.2], [-1.0, -1.0, 0.0]] {
                let options = HeightMapOptions {
                    direction,
                    resolution: 60,
                    margin: 0.05,
                    ..HeightMapOptions::default()
                };
                let frame = s.height_map_frame(&[a, b], &options).unwrap();
                let heights = all(&mut s, &[a, b], &frame);
                let back = frame.normal().map(|c| -c);
                let mut differ = 0;
                for ((c, r), &h) in pixels(&frame).zip(&heights) {
                    let from = frame.point(c, r, frame.range + 0.01);
                    let hit = s.raycast(from, back, &SceneRayOptions::default()).unwrap();
                    let want = hit.map(|hit| frame.range + 0.01 - hit.t);
                    match want {
                        Some(want) if (f64::from(h) - want).abs() < 1e-4 => {}
                        None if h.is_nan() => {}
                        _ => differ += 1,
                    }
                }
                // Only rays that graze an outline may fall either way.
                assert!(differ <= 6, "{differ} of {}", heights.len());
                assert!(heights.iter().any(|h| h.is_nan()));
            }
        }
    }

    #[test]
    fn bands_and_threads_do_not_change_a_bit() {
        let mut s = scene();
        let (m, _) = sphere(1.0);
        let a = add(&mut s, m, at([0.3, 0.0, 0.0]));
        let b = add(
            &mut s,
            mesh(&Primitive::Torus(TorusOptions::default())),
            Placement {
                rotation: tilted(70.0),
                scale: [3.0, 2.0, 2.0],
                ..Placement::IDENTITY
            },
        );
        let options = HeightMapOptions {
            direction: [0.2, 0.1, -1.0],
            resolution: 333,
            ..HeightMapOptions::default()
        };
        let frame = s.height_map_frame(&[a, b], &options).unwrap();
        let whole = bits(&all(&mut s, &[a, b], &frame));
        par::set_force_sequential(true);
        let sequential = bits(&all(&mut s, &[a, b], &frame));
        let again = s.height_map_frame(&[a, b], &options).unwrap();
        par::set_force_sequential(false);
        assert_eq!(whole, sequential);
        assert_eq!(frame, again);
        for band in [1, 5, 8, 100] {
            let mut rows = Vec::new();
            for start in (0..frame.height).step_by(band) {
                let end = (start + band as u32).min(frame.height);
                rows.extend(s.height_map(&[a, b], &frame, start..end).unwrap());
            }
            assert_eq!(whole, bits(&rows), "bands of {band}");
        }
        assert!(s.height_map(&[a], &frame, 7..7).unwrap().is_empty());
    }

    #[test]
    fn ranges_clip_what_is_seen() {
        let mut s = scene();
        let id = add(&mut s, cuboid([2.0; 3]), Placement::IDENTITY);
        let frame = |s: &Scene, range| {
            let options = HeightMapOptions {
                resolution: 8,
                range: Some(range),
                ..HeightMapOptions::default()
            };
            s.height_map_frame(&[id], &options).unwrap()
        };
        // All of it and more: the top, at 1, above a zero plane at -3.
        let f = frame(&s, [-3.0, 5.0]);
        assert_eq!(f.range, 8.0);
        assert!(
            all(&mut s, &[id], &f)
                .iter()
                .all(|h| (h - 4.0).abs() < 1e-6)
        );
        // From inside, the first surface below is the bottom's inner side.
        let f = frame(&s, [-2.0, 0.0]);
        assert!(
            all(&mut s, &[id], &f)
                .iter()
                .all(|h| (h - 1.0).abs() < 1e-6)
        );
        // Neither the top nor the bottom lies in between.
        let f = frame(&s, [-0.5, 0.5]);
        assert!(all(&mut s, &[id], &f).iter().all(|h| h.is_nan()));
        // Surfaces exactly at either end count.
        let f = frame(&s, [0.0, 1.0]);
        assert!(
            all(&mut s, &[id], &f)
                .iter()
                .all(|h| (h - 1.0).abs() < 1e-6)
        );
        let f = frame(&s, [1.0, 2.0]);
        assert!(all(&mut s, &[id], &f).iter().all(|h| h.abs() < 1e-6));
    }

    #[test]
    fn flat_sheets_map_to_height_zero() {
        let mut s = scene();
        let sheet = mesh(&Primitive::Plane(PlaneOptions::default()));
        let id = add(&mut s, sheet, at([5.0, 100.25, -3.0]));
        for direction in [[0.0, -1.0, 0.0], [0.0, 1.0, 0.0]] {
            let options = HeightMapOptions {
                direction,
                resolution: 16,
                ..HeightMapOptions::default()
            };
            let frame = s.height_map_frame(&[id], &options).unwrap();
            assert_eq!(frame.range, 0.0);
            assert_eq!(all(&mut s, &[id], &frame), vec![0.0; 256]);
        }
        // Edge on it has an outline but no surface.
        let options = HeightMapOptions {
            resolution: 16,
            ..HeightMapOptions::default()
        };
        let frame = s.height_map_frame(&[id], &options).unwrap();
        assert_eq!((frame.width, frame.height), (16, 2));
        assert!(all(&mut s, &[id], &frame).iter().all(|h| h.is_nan()));
    }

    #[test]
    fn a_relief_of_a_map_is_the_surface_it_saw() {
        let mut s = scene();
        let (m, _) = sphere(1.0);
        let ball = add(&mut s, m, at([0.2, 0.1, 0.4]));
        let slab = add(
            &mut s,
            cuboid([3.0, 2.0, 0.5]),
            Placement {
                rotation: tilted(25.0),
                translation: [0.0, 0.0, -0.5],
                scale: [1.0; 3],
            },
        );
        let ids = [ball, slab];
        for direction in [[0.0, 0.0, -1.0], [1.0, -2.0, -0.5]] {
            let options = HeightMapOptions {
                direction,
                resolution: 160,
                margin: 0.1,
                ..HeightMapOptions::default()
            };
            let frame = s.height_map_frame(&ids, &options).unwrap();
            let (w, h) = (frame.width as usize, frame.height as usize);
            let seen = all(&mut s, &ids, &frame);
            // As a 16-bit image holds them: no surface is height 0.
            let step = frame.range / 65535.0;
            let code = |h: f32| (f64::from(h) / step).round().clamp(0.0, 65535.0);
            let stored: Vec<f32> = seen
                .iter()
                .map(|&h| {
                    if h.is_nan() {
                        0.0
                    } else {
                        (code(h) * step) as f32
                    }
                })
                .collect();
            let field = HeightfieldOptions {
                pixel_size: [frame.pixel_size; 2],
                base: 0.3,
            };
            let added = s
                .add_heightfield(
                    &stored,
                    frame.width,
                    frame.height,
                    &field,
                    frame.relief_placement(),
                )
                .unwrap();
            let relief = added.added;
            assert!(measure(s.mesh(relief).unwrap()).unwrap().closed);

            // Sampled again in the same frame: the same image, within the
            // step of its codes.
            let back = all(&mut s, &[relief], &frame);
            for row in 1..h - 1 {
                for col in 1..w - 1 {
                    let i = row * w + col;
                    let got = f64::from(back[i]);
                    assert!(
                        (got - f64::from(stored[i])).abs() < 0.25 * step,
                        "{col} {row}: {got} vs {}",
                        stored[i]
                    );
                    assert_eq!(code(back[i]), code(stored[i]));
                    if !seen[i].is_nan() {
                        assert!((got - f64::from(seen[i])).abs() <= step);
                    }
                }
            }

            // Between the pixels: within the largest step in height between
            // neighbouring pixels around it of the surface that was seen.
            let fine = HeightMapFrame {
                width: 4 * frame.width - 3,
                height: 4 * frame.height - 3,
                pixel_size: 0.25 * frame.pixel_size,
                ..frame
            };
            let truth = all(&mut s, &ids, &fine);
            let built = all(&mut s, &[relief], &fine);
            let mut checked = 0;
            for row in 1..h - 2 {
                for col in 1..w - 2 {
                    let near = |dc: usize, dr: usize| seen[(row - 1 + dr) * w + col - 1 + dc];
                    let patch: [[f32; 4]; 4] =
                        std::array::from_fn(|r| std::array::from_fn(|c| near(c, r)));
                    if patch.iter().flatten().any(|h| h.is_nan()) {
                        continue;
                    }
                    let mut largest = 0.0f32;
                    for r in 0..4 {
                        for c in 0..4 {
                            if c < 3 {
                                largest = largest.max((patch[r][c] - patch[r][c + 1]).abs());
                            }
                            if r < 3 {
                                largest = largest.max((patch[r][c] - patch[r + 1][c]).abs());
                            }
                        }
                    }
                    for r in 0..4 {
                        for c in 0..4 {
                            let i = (4 * row + r) * fine.width as usize + 4 * col + c;
                            let d = f64::from((built[i] - truth[i]).abs());
                            assert!(d <= f64::from(largest) + step, "{d} over {largest}");
                            checked += 1;
                        }
                    }
                }
            }
            assert!(checked > 100_000, "{checked}");
            s.remove_mesh(relief).unwrap();
        }
    }

    #[test]
    fn reliefs_are_added_as_one_undo_step() {
        let mut s = scene();
        let field = HeightfieldOptions {
            pixel_size: [0.5, 0.25],
            base: 1.0,
        };
        let placement = Placement {
            rotation: tilted(10.0),
            translation: [1.0, 2.0, 3.0],
            scale: [1.0; 3],
        };
        let heights: Vec<f32> = (0..12).map(|i| i as f32 * 0.1).collect();
        let r = s
            .add_heightfield(&heights, 4, 3, &field, placement)
            .unwrap();
        assert_eq!((r.vertices, r.triangles), (12 + 10, 12 + 28));
        assert_eq!(s.placement(r.added).unwrap(), placement);
        let m = s.mesh(r.added).unwrap();
        assert_eq!(m.triangle_count(), r.triangles);
        let report = measure(m).unwrap();
        assert!(report.closed && report.volume.unwrap() > 0.0);
        assert!(s.undo().unwrap());
        assert_eq!(s.mesh(r.added).unwrap_err(), SceneError::UnknownMesh);
        assert!(s.redo().unwrap());
        assert_eq!(s.mesh(r.added).unwrap().triangle_count(), r.triangles);

        let count = s.entries.iter().count();
        let bad = s.add_heightfield(&heights, 5, 3, &field, placement);
        assert!(matches!(
            bad,
            Err(SceneError::Ops(OpsError::InvalidInput(_)))
        ));
        let mirrored = Placement {
            scale: [1.0, -1.0, 1.0],
            ..placement
        };
        let bad = s.add_heightfield(&heights, 4, 3, &field, mirrored);
        assert!(matches!(bad, Err(SceneError::InvalidInput(_))));
        assert_eq!(s.entries.iter().count(), count);
    }

    #[test]
    fn stale_and_unknown_meshes_are_handled_like_other_queries() {
        let mut s = scene();
        let lazy = s
            .add_mesh(
                cuboid([1.0; 3]),
                MeshOptions {
                    mode: UpdateMode::Lazy,
                    ..MeshOptions::default()
                },
            )
            .unwrap();
        let manual = s
            .add_mesh(
                cuboid([1.0; 3]),
                MeshOptions {
                    mode: UpdateMode::Manual,
                    ..MeshOptions::default()
                },
            )
            .unwrap();
        let options = HeightMapOptions {
            resolution: 8,
            ..HeightMapOptions::default()
        };
        let frame = s.height_map_frame(&[lazy], &options).unwrap();
        for id in [lazy, manual] {
            s.edit(id, |tx| {
                let moved: Vec<[f32; 3]> = tx
                    .mesh()
                    .positions()
                    .iter()
                    .map(|p| [p[0], p[1], p[2] * 0.5])
                    .collect();
                let ids: Vec<u32> = (0..moved.len() as u32).collect();
                tx.set_positions(&ids, &moved)
            })
            .unwrap();
        }
        // The lazy mesh is brought up to date, the stale manual one refused
        // only where it is named.
        let h = all(&mut s, &[lazy], &frame);
        assert!(h.iter().all(|h| (h - 0.75).abs() < 1e-6));
        assert_eq!(
            s.height_map(&[lazy, manual], &frame, 0..8),
            Err(SceneError::StaleBvh(manual))
        );
        s.remove_mesh(manual).unwrap();
        assert_eq!(
            s.height_map(&[lazy, manual], &frame, 0..8),
            Err(SceneError::UnknownMesh)
        );
        assert_eq!(
            s.height_map_frame(&[manual], &options),
            Err(SceneError::UnknownMesh)
        );
    }

    #[test]
    fn bad_options_and_frames_are_refused() {
        let mut s = scene();
        let id = add(&mut s, cuboid([1.0; 3]), Placement::IDENTITY);
        let empty = add(&mut s, Mesh::new(), Placement::IDENTITY);
        let o = HeightMapOptions::default();
        let invalid = |r: Result<HeightMapFrame>| {
            assert!(matches!(r, Err(SceneError::InvalidInput(_))), "{r:?}");
        };
        invalid(s.height_map_frame(&[], &o));
        invalid(s.height_map_frame(&[empty], &o));
        for direction in [[0.0; 3], [f64::NAN, 0.0, 1.0], [f64::INFINITY, 0.0, 0.0]] {
            invalid(s.height_map_frame(&[id], &HeightMapOptions { direction, ..o }));
        }
        for up in [[0.0; 3], [0.0, 0.0, 2.0], [0.0, f64::NAN, 0.0]] {
            let up = Some(up);
            invalid(s.height_map_frame(&[id], &HeightMapOptions { up, ..o }));
        }
        for resolution in [0, 1, 1 << 20, u32::MAX] {
            invalid(s.height_map_frame(&[id], &HeightMapOptions { resolution, ..o }));
        }
        for size in [0.0, -1.0, f64::NAN, f64::INFINITY, 1e-12] {
            let pixel_size = Some(size);
            invalid(s.height_map_frame(&[id], &HeightMapOptions { pixel_size, ..o }));
        }
        for margin in [-0.1, f64::NAN, f64::INFINITY] {
            invalid(s.height_map_frame(&[id], &HeightMapOptions { margin, ..o }));
        }
        for range in [
            [0.0, 0.0],
            [1.0, 0.0],
            [f64::NAN, 1.0],
            [0.0, f64::INFINITY],
        ] {
            let range = Some(range);
            invalid(s.height_map_frame(&[id], &HeightMapOptions { range, ..o }));
        }

        let frame = s.height_map_frame(&[id], &o).unwrap();
        let mut refuse = |frame: HeightMapFrame, rows: Range<u32>| {
            let r = s.height_map(&[id], &frame, rows);
            assert!(matches!(r, Err(SceneError::InvalidInput(_))), "{frame:?}");
        };
        refuse(frame, 0..frame.height + 1);
        #[allow(clippy::reversed_empty_ranges)]
        refuse(frame, 3..2);
        refuse(HeightMapFrame { width: 0, ..frame }, 0..0);
        refuse(
            HeightMapFrame {
                width: 1 << 20,
                height: 1 << 20,
                ..frame
            },
            0..1,
        );
        let right = [2.0, 0.0, 0.0];
        refuse(HeightMapFrame { right, ..frame }, 0..1);
        let up = [1.0, 0.0, 0.0];
        refuse(HeightMapFrame { up, ..frame }, 0..1);
        let origin = [f64::NAN, 0.0, 0.0];
        refuse(HeightMapFrame { origin, ..frame }, 0..1);
        for x in [0.0, -1.0, f64::NAN] {
            refuse(
                HeightMapFrame {
                    pixel_size: x,
                    ..frame
                },
                0..1,
            );
        }
        for range in [-1.0, f64::NAN, f64::INFINITY] {
            refuse(HeightMapFrame { range, ..frame }, 0..1);
        }
        assert!(matches!(
            s.height_map(&[], &frame, 0..1),
            Err(SceneError::InvalidInput(_))
        ));
        assert!(all(&mut s, &[empty], &frame).iter().all(|h| h.is_nan()));
    }
}
