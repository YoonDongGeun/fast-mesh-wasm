//! Planar section contour extraction, 2D metrology, and SVG/JSON/CSV exports.

use std::fmt::Write as _;

use fast_mesh_bvh::{BuildOptions, Bvh};
use fast_mesh_query::{Plane, Slice, contour_length, signed_area, slice};
use fast_mesh_store::Mesh;

use crate::allocation_allowed;
use crate::error::{OpsError, Result};
use crate::icp::{norm, sub};

/// Extracted 2D/3D planar section loop or open polyline chain.
#[derive(Debug, Clone, PartialEq)]
pub struct ContourLoop {
    /// Ordered vertices in 3D world coordinates.
    pub points_3d: Vec<[f32; 3]>,
    /// Ordered vertices projected into the slice plane's `(u, v)` frame.
    pub points_2d: Vec<[f64; 2]>,
    /// `true` if the contour forms a closed loop; `false` for an open chain.
    pub closed: bool,
    /// Arc length around the loop or along the open polyline.
    pub perimeter: f64,
    /// Enclosed area via Green's theorem (`0.0` when `closed == false`).
    pub enclosed_area: f64,
    /// Area centroid (closed) or arc-length centroid (open) in 2D `(u, v)` coordinates.
    pub centroid_2d: [f64; 2],
    /// Centroid lifted back into 3D world coordinates.
    pub centroid_3d: [f64; 3],
    /// Minimum Feret (caliper) width across the 2D convex hull.
    pub min_caliper_diameter: f64,
    /// Maximum Feret (caliper) diameter across the 2D convex hull.
    pub max_caliper_diameter: f64,
}

/// All contours extracted by slicing a mesh with a single plane.
#[derive(Debug, Clone, PartialEq)]
pub struct ContourSlice {
    /// Plane origin point in world coordinates.
    pub plane_origin: [f32; 3],
    /// Unit plane normal vector.
    pub plane_normal: [f32; 3],
    /// First in-plane orthonormal basis vector in world coordinates.
    pub u_axis: [f64; 3],
    /// Second in-plane orthonormal basis vector in world coordinates.
    pub v_axis: [f64; 3],
    /// Extracted closed loops and open chains on this slice plane.
    pub loops: Vec<ContourLoop>,
    /// Sum of perimeters across all extracted contours.
    pub total_perimeter: f64,
    /// Net enclosed area accounting for outer loops and nested holes (`even-odd` nesting).
    pub total_area: f64,
}

/// Extract planar cross-section contours from `mesh` cut by `(origin, normal)`.
pub fn slice_contours(mesh: &Mesh, origin: [f32; 3], normal: [f32; 3]) -> Result<ContourSlice> {
    let world_plane = build_plane(origin, normal)?;
    if mesh.triangle_count() == 0 {
        return Ok(empty_slice(origin, &world_plane));
    }

    allocation_allowed()?;
    let bvh = Bvh::build(mesh, &BuildOptions::default())
        .map_err(|_| OpsError::InvalidInput("failed to build BVH for contour slicing"))?;
    slice_with_bvh(mesh, &bvh, origin, &world_plane)
}

/// Extract `count` parallel planar slices spaced by `spacing` along normalized `normal`.
pub fn slice_contour_stack(
    mesh: &Mesh,
    origin: [f32; 3],
    normal: [f32; 3],
    spacing: f32,
    count: usize,
) -> Result<Vec<ContourSlice>> {
    let base_plane = build_plane(origin, normal)?;
    if !spacing.is_finite() || spacing <= 0.0 {
        return Err(OpsError::InvalidInput(
            "slice spacing must be positive and finite",
        ));
    }
    if count == 0 {
        return Ok(Vec::new());
    }

    allocation_allowed()?;
    let mut out = Vec::with_capacity(count);
    let step_n = base_plane.normal();
    let step_len = f64::from(spacing);
    let base_orig = [
        f64::from(origin[0]),
        f64::from(origin[1]),
        f64::from(origin[2]),
    ];

    if mesh.triangle_count() == 0 {
        for idx in 0..count {
            let offset = (idx as f64) * step_len;
            let o_k = [
                (base_orig[0] + offset * step_n[0]) as f32,
                (base_orig[1] + offset * step_n[1]) as f32,
                (base_orig[2] + offset * step_n[2]) as f32,
            ];
            let p_k = Plane::new(
                [f64::from(o_k[0]), f64::from(o_k[1]), f64::from(o_k[2])],
                step_n,
            )?;
            out.push(empty_slice(o_k, &p_k));
        }
        return Ok(out);
    }

    let bvh = Bvh::build(mesh, &BuildOptions::default())
        .map_err(|_| OpsError::InvalidInput("failed to build BVH for contour stack"))?;

    for idx in 0..count {
        let offset = (idx as f64) * step_len;
        let o_k = [
            (base_orig[0] + offset * step_n[0]) as f32,
            (base_orig[1] + offset * step_n[1]) as f32,
            (base_orig[2] + offset * step_n[2]) as f32,
        ];
        let p_k = Plane::new(
            [f64::from(o_k[0]), f64::from(o_k[1]), f64::from(o_k[2])],
            step_n,
        )?;
        out.push(slice_with_bvh(mesh, &bvh, o_k, &p_k)?);
    }

    Ok(out)
}

/// Export a slice collection as a standalone SVG document in 2D `(u, v)` slice coordinates.
pub fn contours_to_svg(slices: &[ContourSlice]) -> String {
    let mut min_x = f64::INFINITY;
    let mut min_y = f64::INFINITY;
    let mut max_x = f64::NEG_INFINITY;
    let mut max_y = f64::NEG_INFINITY;
    let mut any_pts = false;

    for s in slices {
        for l in &s.loops {
            for &p in &l.points_2d {
                any_pts = true;
                min_x = min_x.min(p[0]);
                min_y = min_y.min(p[1]);
                max_x = max_x.max(p[0]);
                max_y = max_y.max(p[1]);
            }
        }
    }

    let (vx, vy, vw, vh) = if any_pts {
        let span_x = (max_x - min_x).max(1e-3);
        let span_y = (max_y - min_y).max(1e-3);
        let pad = 0.05 * span_x.max(span_y);
        (
            min_x - pad,
            min_y - pad,
            span_x + 2.0 * pad,
            span_y + 2.0 * pad,
        )
    } else {
        (0.0, 0.0, 100.0, 100.0)
    };
    let stroke_w = (vw.max(vh) * 0.004).max(1e-4);

    let mut svg = String::new();
    let _ = writeln!(
        svg,
        "<svg xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"{:.6} {:.6} {:.6} {:.6}\">",
        clean_zero(vx),
        clean_zero(vy),
        clean_zero(vw),
        clean_zero(vh),
    );

    for (si, slice) in slices.iter().enumerate() {
        let _ = writeln!(svg, "  <g data-slice=\"{si}\">");
        for (li, loop_item) in slice.loops.iter().enumerate() {
            if loop_item.points_2d.len() < 2 {
                continue;
            }
            let tag = if loop_item.closed {
                "polygon"
            } else {
                "polyline"
            };
            let _ = write!(
                svg,
                "    <{tag} data-loop=\"{li}\" fill=\"none\" stroke=\"#0f172a\" stroke-width=\"{stroke_w:.6}\" points=\""
            );
            for (pi, &pt) in loop_item.points_2d.iter().enumerate() {
                if pi > 0 {
                    svg.push(' ');
                }
                let _ = write!(svg, "{:.6},{:.6}", clean_zero(pt[0]), clean_zero(pt[1]));
            }
            svg.push_str("\" />\n");
        }
        svg.push_str("  </g>\n");
    }

    svg.push_str("</svg>\n");
    svg
}

/// Export a slice collection as structured JSON metadata and contour coordinates.
pub fn contours_to_json(slices: &[ContourSlice]) -> String {
    let mut out = String::from("{\"slices\":[");
    for (si, s) in slices.iter().enumerate() {
        if si > 0 {
            out.push(',');
        }
        let _ = write!(
            out,
            "{{\"plane_origin\":[{:.6},{:.6},{:.6}],\"plane_normal\":[{:.6},{:.6},{:.6}],\"u_axis\":[{:.6},{:.6},{:.6}],\"v_axis\":[{:.6},{:.6},{:.6}],\"total_perimeter\":{:.6},\"total_area\":{:.6},\"loops\":[",
            clean_zero(f64::from(s.plane_origin[0])),
            clean_zero(f64::from(s.plane_origin[1])),
            clean_zero(f64::from(s.plane_origin[2])),
            clean_zero(f64::from(s.plane_normal[0])),
            clean_zero(f64::from(s.plane_normal[1])),
            clean_zero(f64::from(s.plane_normal[2])),
            clean_zero(s.u_axis[0]),
            clean_zero(s.u_axis[1]),
            clean_zero(s.u_axis[2]),
            clean_zero(s.v_axis[0]),
            clean_zero(s.v_axis[1]),
            clean_zero(s.v_axis[2]),
            clean_zero(s.total_perimeter),
            clean_zero(s.total_area),
        );
        for (li, l) in s.loops.iter().enumerate() {
            if li > 0 {
                out.push(',');
            }
            let _ = write!(
                out,
                "{{\"closed\":{},\"perimeter\":{:.6},\"enclosed_area\":{:.6},\"centroid_2d\":[{:.6},{:.6}],\"centroid_3d\":[{:.6},{:.6},{:.6}],\"min_caliper_diameter\":{:.6},\"max_caliper_diameter\":{:.6},\"points_2d\":[",
                l.closed,
                clean_zero(l.perimeter),
                clean_zero(l.enclosed_area),
                clean_zero(l.centroid_2d[0]),
                clean_zero(l.centroid_2d[1]),
                clean_zero(l.centroid_3d[0]),
                clean_zero(l.centroid_3d[1]),
                clean_zero(l.centroid_3d[2]),
                clean_zero(l.min_caliper_diameter),
                clean_zero(l.max_caliper_diameter),
            );
            for (pi, p2) in l.points_2d.iter().enumerate() {
                if pi > 0 {
                    out.push(',');
                }
                let _ = write!(out, "[{:.6},{:.6}]", clean_zero(p2[0]), clean_zero(p2[1]));
            }
            out.push_str("],\"points_3d\":[");
            for (pi, p3) in l.points_3d.iter().enumerate() {
                if pi > 0 {
                    out.push(',');
                }
                let _ = write!(
                    out,
                    "[{:.6},{:.6},{:.6}]",
                    clean_zero(f64::from(p3[0])),
                    clean_zero(f64::from(p3[1])),
                    clean_zero(f64::from(p3[2])),
                );
            }
            out.push_str("]}");
        }
        out.push_str("]}");
    }
    out.push_str("]}\n");
    out
}

/// Export per-loop contour measurements as CSV rows with header.
pub fn contours_to_csv(slices: &[ContourSlice]) -> String {
    let mut csv = String::from(
        "slice_index,loop_index,closed,point_count,perimeter,enclosed_area,centroid_u,centroid_v,centroid_x,centroid_y,centroid_z,min_caliper_diameter,max_caliper_diameter\n",
    );
    for (si, slice) in slices.iter().enumerate() {
        for (li, l) in slice.loops.iter().enumerate() {
            let _ = writeln!(
                csv,
                "{si},{li},{},{},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6}",
                l.closed,
                l.points_2d.len(),
                clean_zero(l.perimeter),
                clean_zero(l.enclosed_area),
                clean_zero(l.centroid_2d[0]),
                clean_zero(l.centroid_2d[1]),
                clean_zero(l.centroid_3d[0]),
                clean_zero(l.centroid_3d[1]),
                clean_zero(l.centroid_3d[2]),
                clean_zero(l.min_caliper_diameter),
                clean_zero(l.max_caliper_diameter),
            );
        }
    }
    csv
}

// ============================================================================
// Internal Geometry Helpers
// ============================================================================

fn build_plane(origin: [f32; 3], normal: [f32; 3]) -> Result<Plane> {
    if !origin.iter().all(|v| v.is_finite()) || !normal.iter().all(|v| v.is_finite()) {
        return Err(OpsError::InvalidInput(
            "slice plane origin and normal must be finite",
        ));
    }
    Ok(Plane::new(
        [
            f64::from(origin[0]),
            f64::from(origin[1]),
            f64::from(origin[2]),
        ],
        [
            f64::from(normal[0]),
            f64::from(normal[1]),
            f64::from(normal[2]),
        ],
    )?)
}

fn empty_slice(origin: [f32; 3], plane: &Plane) -> ContourSlice {
    let n = plane.normal();
    ContourSlice {
        plane_origin: origin,
        plane_normal: [n[0] as f32, n[1] as f32, n[2] as f32],
        u_axis: plane.u(),
        v_axis: plane.v(),
        loops: Vec::new(),
        total_perimeter: 0.0,
        total_area: 0.0,
    }
}

fn slice_with_bvh(
    mesh: &Mesh,
    bvh: &Bvh,
    world_origin: [f32; 3],
    world_plane: &Plane,
) -> Result<ContourSlice> {
    let mesh_orig = mesh.origin();
    let local_plane = Plane::new(
        [
            f64::from(world_origin[0]) - mesh_orig[0],
            f64::from(world_origin[1]) - mesh_orig[1],
            f64::from(world_origin[2]) - mesh_orig[2],
        ],
        world_plane.normal(),
    )?;

    let mut raw = Slice {
        contours: Vec::new(),
    };
    slice(mesh, bvh, &local_plane, &mut raw)?;

    let mut loops = Vec::with_capacity(raw.contours.len());
    for c in raw.contours {
        if let Some(contour_loop) = build_contour_loop(&c.points, c.closed, mesh_orig, world_plane)
        {
            loops.push(contour_loop);
        }
    }

    let total_perimeter = loops.iter().map(|l| l.perimeter).sum();
    let total_area = compute_nested_total_area(&loops);
    let pn = world_plane.normal();

    Ok(ContourSlice {
        plane_origin: world_origin,
        plane_normal: [pn[0] as f32, pn[1] as f32, pn[2] as f32],
        u_axis: world_plane.u(),
        v_axis: world_plane.v(),
        loops,
        total_perimeter,
        total_area,
    })
}

fn build_contour_loop(
    local_pts: &[[f64; 3]],
    raw_closed: bool,
    mesh_origin: [f64; 3],
    world_plane: &Plane,
) -> Option<ContourLoop> {
    if local_pts.is_empty() {
        return None;
    }

    // Lift to world space and deduplicate consecutive coincident vertices.
    let mut world_pts = Vec::with_capacity(local_pts.len());
    for &lp in local_pts {
        let wp = [
            lp[0] + mesh_origin[0],
            lp[1] + mesh_origin[1],
            lp[2] + mesh_origin[2],
        ];
        if world_pts
            .last()
            .is_some_and(|&prev| norm(sub(wp, prev)) <= 1e-9)
        {
            continue;
        }
        world_pts.push(wp);
    }

    if raw_closed
        && world_pts.len() >= 2
        && norm(sub(world_pts[0], *world_pts.last().unwrap_or(&[0.0; 3]))) <= 1e-9
    {
        world_pts.pop();
    }

    if world_pts.len() < 2 {
        return None;
    }

    let closed = raw_closed && world_pts.len() >= 3;
    let points_3d: Vec<[f32; 3]> = world_pts
        .iter()
        .map(|&p| [p[0] as f32, p[1] as f32, p[2] as f32])
        .collect();
    let points_2d: Vec<[f64; 2]> = world_pts.iter().map(|&p| world_plane.to_uv(p)).collect();

    let perimeter = contour_length(&points_2d, closed);
    let enclosed_area = if closed {
        signed_area(&points_2d).abs()
    } else {
        0.0
    };
    let centroid_2d = compute_centroid_2d(&points_2d, closed);
    let centroid_3d = world_plane.from_uv(centroid_2d);
    let (min_caliper_diameter, max_caliper_diameter) = caliper_diameters_2d(&points_2d);

    Some(ContourLoop {
        points_3d,
        points_2d,
        closed,
        perimeter,
        enclosed_area,
        centroid_2d,
        centroid_3d,
        min_caliper_diameter,
        max_caliper_diameter,
    })
}

fn compute_centroid_2d(pts: &[[f64; 2]], closed: bool) -> [f64; 2] {
    if pts.is_empty() {
        return [0.0, 0.0];
    }
    if pts.len() == 1 {
        return pts[0];
    }

    if closed && pts.len() >= 3 {
        let mut two_a = 0.0;
        let mut cx = 0.0;
        let mut cy = 0.0;
        let n = pts.len();
        for i in 0..n {
            let a = pts[i];
            let b = pts[(i + 1) % n];
            let cross = a[0] * b[1] - a[1] * b[0];
            two_a += cross;
            cx += (a[0] + b[0]) * cross;
            cy += (a[1] + b[1]) * cross;
        }
        if two_a.abs() > 1e-12 {
            return [cx / (3.0 * two_a), cy / (3.0 * two_a)];
        }
    }

    // Arc-length weighted polyline centroid fallback for open chains or zero-area loops.
    let seg_count = if closed { pts.len() } else { pts.len() - 1 };
    let mut len_sum = 0.0;
    let mut wx = 0.0;
    let mut wy = 0.0;
    for i in 0..seg_count {
        let a = pts[i];
        let b = pts[(i + 1) % pts.len()];
        let seg = (b[0] - a[0]).hypot(b[1] - a[1]);
        len_sum += seg;
        wx += 0.5 * (a[0] + b[0]) * seg;
        wy += 0.5 * (a[1] + b[1]) * seg;
    }
    if len_sum > 1e-14 {
        [wx / len_sum, wy / len_sum]
    } else {
        let inv = 1.0 / (pts.len() as f64);
        let sx: f64 = pts.iter().map(|p| p[0]).sum();
        let sy: f64 = pts.iter().map(|p| p[1]).sum();
        [sx * inv, sy * inv]
    }
}

fn convex_hull_2d(pts: &[[f64; 2]]) -> Vec<[f64; 2]> {
    if pts.len() <= 1 {
        return pts.to_vec();
    }
    let mut sorted = pts.to_vec();
    sorted.sort_by(|a, b| a[0].total_cmp(&b[0]).then_with(|| a[1].total_cmp(&b[1])));
    sorted.dedup_by(|a, b| (a[0] - b[0]).hypot(a[1] - b[1]) <= 1e-12);
    if sorted.len() <= 2 {
        return sorted;
    }

    let mut hull: Vec<[f64; 2]> = Vec::with_capacity(sorted.len() * 2);
    for &p in &sorted {
        while hull.len() >= 2 && cross_2d(hull[hull.len() - 2], hull[hull.len() - 1], p) <= 0.0 {
            hull.pop();
        }
        hull.push(p);
    }
    let lower_len = hull.len();
    for &p in sorted.iter().rev().skip(1) {
        while hull.len() > lower_len
            && cross_2d(hull[hull.len() - 2], hull[hull.len() - 1], p) <= 0.0
        {
            hull.pop();
        }
        hull.push(p);
    }
    hull.pop();
    hull
}

fn cross_2d(o: [f64; 2], a: [f64; 2], b: [f64; 2]) -> f64 {
    (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
}

fn caliper_diameters_2d(pts: &[[f64; 2]]) -> (f64, f64) {
    let hull = convex_hull_2d(pts);
    let m = hull.len();
    if m <= 1 {
        return (0.0, 0.0);
    }
    if m == 2 {
        let d = (hull[1][0] - hull[0][0]).hypot(hull[1][1] - hull[0][1]);
        return (0.0, d);
    }

    let mut min_width = f64::INFINITY;
    let mut max_diam = 0.0f64;
    let mut k = 1usize;

    for i in 0..m {
        let j = (i + 1) % m;
        let edge_len = (hull[j][0] - hull[i][0]).hypot(hull[j][1] - hull[i][1]);
        if edge_len <= 1e-15 {
            continue;
        }

        let mut steps = 0usize;
        while steps < m {
            let kn = (k + 1) % m;
            let cur = cross_2d(hull[i], hull[j], hull[k]).abs();
            let nxt = cross_2d(hull[i], hull[j], hull[kn]).abs();
            if nxt > cur && kn != i {
                k = kn;
                steps += 1;
            } else {
                break;
            }
        }

        let height = cross_2d(hull[i], hull[j], hull[k]).abs() / edge_len;
        min_width = min_width.min(height);

        let d_ik = (hull[i][0] - hull[k][0]).hypot(hull[i][1] - hull[k][1]);
        let d_jk = (hull[j][0] - hull[k][0]).hypot(hull[j][1] - hull[k][1]);
        let kn = (k + 1) % m;
        let d_ikn = (hull[i][0] - hull[kn][0]).hypot(hull[i][1] - hull[kn][1]);
        let d_jkn = (hull[j][0] - hull[kn][0]).hypot(hull[j][1] - hull[kn][1]);
        max_diam = max_diam.max(d_ik).max(d_jk).max(d_ikn).max(d_jkn);
    }

    if !min_width.is_finite() {
        min_width = 0.0;
    }
    (min_width, max_diam)
}

fn compute_nested_total_area(loops: &[ContourLoop]) -> f64 {
    let closed_indices: Vec<usize> = loops
        .iter()
        .enumerate()
        .filter_map(|(i, l)| (l.closed && l.points_2d.len() >= 3).then_some(i))
        .collect();
    if closed_indices.is_empty() {
        return 0.0;
    }
    if closed_indices.len() == 1 {
        return loops[closed_indices[0]].enclosed_area;
    }

    let mut total = 0.0;
    for &i in &closed_indices {
        let probe = boundary_probe_point(&loops[i].points_2d);
        let mut depth = 0usize;
        for &j in &closed_indices {
            if j == i || loops[j].enclosed_area <= loops[i].enclosed_area {
                continue;
            }
            if point_in_polygon_2d(&loops[j].points_2d, probe) {
                depth += 1;
            }
        }
        if (depth & 1) == 0 {
            total += loops[i].enclosed_area;
        } else {
            total -= loops[i].enclosed_area;
        }
    }
    total.max(0.0)
}

fn boundary_probe_point(pts: &[[f64; 2]]) -> [f64; 2] {
    [0.5 * (pts[0][0] + pts[1][0]), 0.5 * (pts[0][1] + pts[1][1])]
}

fn point_in_polygon_2d(poly: &[[f64; 2]], p: [f64; 2]) -> bool {
    let n = poly.len();
    if n < 3 {
        return false;
    }
    let mut inside = false;
    let mut j = n - 1;
    for i in 0..n {
        let yi = poly[i][1];
        let yj = poly[j][1];
        if (yi > p[1]) != (yj > p[1]) {
            let x_int = (poly[j][0] - poly[i][0]) * (p[1] - yi) / (yj - yi) + poly[i][0];
            if p[0] < x_int {
                inside = !inside;
            }
        }
        j = i;
    }
    inside
}

fn clean_zero(v: f64) -> f64 {
    if v == 0.0 || !v.is_finite() { 0.0 } else { v }
}
