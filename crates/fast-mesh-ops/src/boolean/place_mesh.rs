//! Surface-aligned stamp/decal placement onto a target mesh.
//!
//! [`compute_place_transform`] builds a scale + twist + surface-normal-aligned
//! rigid transform mapping the chosen [`PlaceUpAxis`] onto `hit_normal` with an
//! axial `embed_depth`. [`place_mesh`] applies that placement to a `stamp` mesh,
//! optionally projecting stamp vertices onto a curved target surface via BVH ray
//! and closest-point queries (`conform_to_surface`), then outputs either a
//! separate mesh, an appended shell, or an embedded/engraved CSG boolean result.

use fast_mesh_bvh::Bvh;
use fast_mesh_query::{
    ClosestHit, ClosestOptions, Cull, Ray, RayHit, RayOptions, Rigid, closest_point_many,
    raycast_all,
};
use fast_mesh_store::{Mesh, StoreError, Transaction, TriangleId, VertexId};

use super::{
    BooleanMethod, BooleanOp, BooleanOptions, apply_boolean_with_options, boolean,
    replace_tx_with_mesh,
};
use crate::align::rigid_to_matrix4;
use crate::append::{AppendSource, append_mesh};
use crate::offset::{CompactWorkMesh, build_work_bvh, compact_work_mesh};
use crate::topology::{cross, dot};
use crate::{OpsError, Result, allocation_allowed, filled, push, reserve};

/// Which local axis of the stamp mesh points outward along `hit_normal`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum PlaceUpAxis {
    /// Local `-Z` axis maps to `hit_normal`.
    NegZ,
    /// Local `+Z` axis maps to `hit_normal`.
    #[default]
    PosZ,
    /// Local `+Y` axis maps to `hit_normal`.
    PosY,
    /// Local `-Y` axis maps to `hit_normal`.
    NegY,
}

impl PlaceUpAxis {
    /// Returns the unit vector corresponding to this axis in the stamp's local space.
    #[must_use]
    pub const fn unit_vector(self) -> [f64; 3] {
        match self {
            Self::NegZ => [0.0, 0.0, -1.0],
            Self::PosZ => [0.0, 0.0, 1.0],
            Self::PosY => [0.0, 1.0, 0.0],
            Self::NegY => [0.0, -1.0, 0.0],
        }
    }
}

/// How [`place_mesh`] applies the transformed stamp relative to `tx`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum PlaceMode {
    /// Leaves `tx` unchanged and returns the transformed stamp in [`PlaceMeshReport::placed_mesh`].
    KeepSeparate,
    /// Appends the transformed stamp into `tx` as a separate closed/open shell without CSG.
    AppendShell,
    /// Unions the placed stamp into `tx` via [`BooleanOp::Union`].
    #[default]
    EmbedUnion,
    /// Subtracts the placed stamp from `tx` via [`BooleanOp::Subtract`].
    EngraveSubtract,
}

/// Placement parameters for [`compute_place_transform`] and [`place_mesh`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PlaceMeshParams {
    /// Anchor point on the target surface in `tx`'s local frame.
    pub hit_point: [f32; 3],
    /// Outward surface normal at `hit_point` (normalized internally; must be non-zero).
    pub hit_normal: [f32; 3],
    /// Local stamp axis aligned with `hit_normal`.
    pub up_axis: PlaceUpAxis,
    /// Twist rotation around `hit_normal` in radians after axis alignment.
    pub twist_angle_rad: f32,
    /// Uniform isotropic scale applied to the stamp (`> 0.0`).
    pub uniform_scale: f32,
    /// Axial shift inward along `-hit_normal` in local target units (`> 0` sinks into target).
    pub embed_depth: f32,
    /// Integration mode relative to `tx`.
    pub mode: PlaceMode,
    /// Projects vertices along `-hit_normal` onto curved target geometry before output/CSG.
    pub conform_to_surface: bool,
    /// Boolean backend used for [`PlaceMode::EmbedUnion`] and [`PlaceMode::EngraveSubtract`].
    pub boolean_method: BooleanMethod,
}

impl Default for PlaceMeshParams {
    fn default() -> Self {
        Self {
            hit_point: [0.0, 0.0, 0.0],
            hit_normal: [0.0, 0.0, 1.0],
            up_axis: PlaceUpAxis::PosZ,
            twist_angle_rad: 0.0,
            uniform_scale: 1.0,
            embed_depth: 0.0,
            mode: PlaceMode::EmbedUnion,
            conform_to_surface: false,
            boolean_method: BooleanMethod::Manifold,
        }
    }
}

/// Similarity transform (uniform isotropic scale + rigid rotation and translation).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PlaceTransform {
    /// Rigid rotation + translation mapping aligned, scaled stamp space to target local space.
    pub rigid: Rigid,
    /// Uniform scale factor applied in stamp space prior to `rigid`.
    pub scale: f64,
}

impl PlaceTransform {
    /// Applies the similarity transform `R * (scale * p) + t` to a local stamp point.
    #[must_use]
    pub fn apply(&self, point_local: [f32; 3]) -> [f64; 3] {
        let sx = self.scale * f64::from(point_local[0]);
        let sy = self.scale * f64::from(point_local[1]);
        let sz = self.scale * f64::from(point_local[2]);
        let r = &self.rigid.rotation;
        let t = self.rigid.translation;
        [
            r[0][0] * sx + r[0][1] * sy + r[0][2] * sz + t[0],
            r[1][0] * sx + r[1][1] * sy + r[1][2] * sz + t[1],
            r[2][0] * sx + r[2][1] * sy + r[2][2] * sz + t[2],
        ]
    }

    /// Converts the transform to a column-major `4x4` matrix `[col0, col1, col2, col3]`.
    #[must_use]
    pub fn to_matrix4(&self) -> [f64; 16] {
        rigid_to_matrix4(&self.rigid, self.scale)
    }
}

/// Outcome of [`place_mesh`].
#[derive(Debug)]
pub struct PlaceMeshReport {
    /// Non-empty placed stamp mesh when `params.mode == PlaceMode::KeepSeparate`.
    pub placed_mesh: Option<Mesh>,
    /// Rigid + uniform scale transform computed for placement.
    pub transform: PlaceTransform,
    /// Triangle count remaining on `tx` after the operation.
    pub kept_triangles: usize,
    /// Whether `UV_CHANNEL` on `tx` was dropped.
    pub uvs_dropped: bool,
}

/// Computes the minimal 3×3 proper rotation matrix aligning `up_axis` onto normalized `hit_normal`.
pub fn align_up_axis_rotation(up_axis: PlaceUpAxis, hit_normal: [f32; 3]) -> Result<[[f64; 3]; 3]> {
    let n = normalize_normal(hit_normal)?;
    let u = up_axis.unit_vector();
    Ok(minimal_align_rotation(u, n))
}

/// Computes the surface-aligned similarity transform for [`place_mesh`].
pub fn compute_place_transform(params: &PlaceMeshParams) -> Result<PlaceTransform> {
    if !params.hit_point.iter().all(|c| c.is_finite()) {
        return Err(OpsError::InvalidInput(
            "place_mesh hit_point must be finite",
        ));
    }
    if !params.twist_angle_rad.is_finite() {
        return Err(OpsError::InvalidInput(
            "place_mesh twist_angle_rad must be finite",
        ));
    }
    if !params.uniform_scale.is_finite() || params.uniform_scale <= 0.0 {
        return Err(OpsError::InvalidInput(
            "place_mesh uniform_scale must be finite and positive",
        ));
    }
    if !params.embed_depth.is_finite() {
        return Err(OpsError::InvalidInput(
            "place_mesh embed_depth must be finite",
        ));
    }

    let n = normalize_normal(params.hit_normal)?;
    let r_align = minimal_align_rotation(params.up_axis.unit_vector(), n);
    let r_twist = rotation_around_unit_axis(n, f64::from(params.twist_angle_rad));
    let r = orthonormalize_mat3(mul_mat3(r_twist, r_align));

    let embed = f64::from(params.embed_depth);
    let translation = [
        f64::from(params.hit_point[0]) - embed * n[0],
        f64::from(params.hit_point[1]) - embed * n[1],
        f64::from(params.hit_point[2]) - embed * n[2],
    ];
    if !translation.iter().all(|c| c.is_finite()) {
        return Err(OpsError::InvalidInput(
            "place_mesh translation leaves finite range",
        ));
    }

    let rigid = Rigid {
        rotation: r,
        translation,
    };
    rigid.validate()?;

    Ok(PlaceTransform {
        rigid,
        scale: f64::from(params.uniform_scale),
    })
}

/// Places `stamp` relative to `tx` according to `params`.
pub fn place_mesh(
    tx: &mut Transaction<'_>,
    stamp: &Mesh,
    params: &PlaceMeshParams,
) -> Result<PlaceMeshReport> {
    let transform = compute_place_transform(params)?;
    let n = normalize_normal(params.hit_normal)?;

    if stamp.triangle_count() == 0 {
        return Err(OpsError::InvalidInput(
            "place_mesh requires a non-empty stamp mesh",
        ));
    }
    let stamp_positions = stamp.positions();
    for (v, p) in stamp_positions.iter().enumerate() {
        if stamp.is_vertex_alive(v as VertexId) && !p.iter().all(|c| c.is_finite()) {
            return Err(OpsError::InvalidInput(
                "place_mesh stamp needs finite vertex positions",
            ));
        }
    }

    let stamp_compact = compact_work_mesh(stamp)?;
    if !stamp_compact.has_positive_area {
        return Err(OpsError::InvalidInput(
            "place_mesh stamp requires at least one non-degenerate triangle",
        ));
    }

    let need_target_query = params.conform_to_surface;
    let target_had_uvs = tx.mesh().uvs().is_some();
    let need_target_compact = need_target_query
        || (target_had_uvs
            && matches!(
                params.mode,
                PlaceMode::EmbedUnion | PlaceMode::EngraveSubtract
            ));

    let target_compact = if need_target_compact {
        let mesh_a = tx.mesh();
        if mesh_a.triangle_count() == 0 {
            return Err(OpsError::InvalidInput(
                "place_mesh target mesh must be non-empty for surface conforming or CSG",
            ));
        }
        for (v, p) in mesh_a.positions().iter().enumerate() {
            if mesh_a.is_vertex_alive(v as VertexId) && !p.iter().all(|c| c.is_finite()) {
                return Err(OpsError::InvalidInput(
                    "place_mesh target needs finite vertex positions",
                ));
            }
        }
        Some(compact_work_mesh(mesh_a)?)
    } else {
        None
    };

    let placed = build_placed_stamp(
        &stamp_compact.mesh,
        &transform,
        params,
        n,
        target_compact.as_ref(),
    )?;

    match params.mode {
        PlaceMode::KeepSeparate => Ok(PlaceMeshReport {
            placed_mesh: Some(placed),
            transform,
            kept_triangles: tx.mesh().triangle_count(),
            uvs_dropped: false,
        }),
        PlaceMode::AppendShell => {
            let source = mesh_to_append_source(&placed)?;
            let app = append_mesh(tx, &source)?;
            Ok(PlaceMeshReport {
                placed_mesh: None,
                transform,
                kept_triangles: tx.mesh().triangle_count(),
                uvs_dropped: app.uvs_dropped,
            })
        }
        PlaceMode::EmbedUnion | PlaceMode::EngraveSubtract => {
            if tx.mesh().triangle_count() == 0 {
                return Err(OpsError::InvalidInput(
                    "place_mesh CSG modes require a non-empty target mesh",
                ));
            }
            let op = match params.mode {
                PlaceMode::EmbedUnion => BooleanOp::Union,
                PlaceMode::EngraveSubtract => BooleanOp::Subtract,
                PlaceMode::KeepSeparate | PlaceMode::AppendShell => unreachable!(),
            };
            let bool_opts = BooleanOptions {
                op,
                method: params.boolean_method,
                coplanar_tolerance: 1e-5,
                snap_slivers: true,
                voxel_pitch: None,
                max_triangles: None,
            };

            let uvs_dropped = if let Some(ref tc) = target_compact {
                let (out, rep) = boolean(&tc.mesh, &placed, Rigid::IDENTITY, &bool_opts)?;
                replace_tx_with_mesh(tx, &out)?;
                target_had_uvs || rep.uvs_dropped
            } else {
                let rep = apply_boolean_with_options(tx, &placed, Rigid::IDENTITY, &bool_opts)?;
                target_had_uvs || rep.uvs_dropped
            };

            Ok(PlaceMeshReport {
                placed_mesh: None,
                transform,
                kept_triangles: tx.mesh().triangle_count(),
                uvs_dropped,
            })
        }
    }
}

fn to_f64(v: [f32; 3]) -> [f64; 3] {
    [f64::from(v[0]), f64::from(v[1]), f64::from(v[2])]
}

fn normalize_normal(hit_normal: [f32; 3]) -> Result<[f64; 3]> {
    if !hit_normal.iter().all(|c| c.is_finite()) {
        return Err(OpsError::InvalidInput(
            "place_mesh hit_normal must be finite",
        ));
    }
    let raw = to_f64(hit_normal);
    let len = dot(raw, raw).sqrt();
    if !len.is_finite() || len <= 1e-12 {
        return Err(OpsError::InvalidInput(
            "place_mesh hit_normal must be non-zero",
        ));
    }
    Ok([raw[0] / len, raw[1] / len, raw[2] / len])
}

fn minimal_align_rotation(u: [f64; 3], n: [f64; 3]) -> [[f64; 3]; 3] {
    let cos_theta = dot(u, n).clamp(-1.0, 1.0);
    if cos_theta > 1.0 - 1e-12 {
        return [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    }
    if cos_theta < -1.0 + 1e-12 {
        // Deterministic 180-degree rotation around an axis orthogonal to u.
        let helper = if u[0].abs() < 0.85 {
            [1.0, 0.0, 0.0]
        } else {
            [0.0, 1.0, 0.0]
        };
        let a_raw = cross(u, helper);
        let a_len = dot(a_raw, a_raw).sqrt();
        let a = [a_raw[0] / a_len, a_raw[1] / a_len, a_raw[2] / a_len];
        return [
            [
                2.0 * a[0] * a[0] - 1.0,
                2.0 * a[0] * a[1],
                2.0 * a[0] * a[2],
            ],
            [
                2.0 * a[1] * a[0],
                2.0 * a[1] * a[1] - 1.0,
                2.0 * a[1] * a[2],
            ],
            [
                2.0 * a[2] * a[0],
                2.0 * a[2] * a[1],
                2.0 * a[2] * a[2] - 1.0,
            ],
        ];
    }

    let v = cross(u, n);
    let inv = 1.0 / (1.0 + cos_theta);
    let vx = v[0];
    let vy = v[1];
    let vz = v[2];
    orthonormalize_mat3([
        [
            1.0 - (vy * vy + vz * vz) * inv,
            -vz + vx * vy * inv,
            vy + vx * vz * inv,
        ],
        [
            vz + vx * vy * inv,
            1.0 - (vx * vx + vz * vz) * inv,
            -vx + vy * vz * inv,
        ],
        [
            -vy + vx * vz * inv,
            vx + vy * vz * inv,
            1.0 - (vx * vx + vy * vy) * inv,
        ],
    ])
}

fn rotation_around_unit_axis(axis: [f64; 3], angle_rad: f64) -> [[f64; 3]; 3] {
    if angle_rad == 0.0 {
        return [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    }
    let s = angle_rad.sin();
    let half_s = (0.5 * angle_rad).sin();
    let one_minus_c = 2.0 * half_s * half_s;
    let x = axis[0];
    let y = axis[1];
    let z = axis[2];
    [
        [
            1.0 - (y * y + z * z) * one_minus_c,
            x * y * one_minus_c - z * s,
            x * z * one_minus_c + y * s,
        ],
        [
            x * y * one_minus_c + z * s,
            1.0 - (x * x + z * z) * one_minus_c,
            y * z * one_minus_c - x * s,
        ],
        [
            x * z * one_minus_c - y * s,
            y * z * one_minus_c + x * s,
            1.0 - (x * x + y * y) * one_minus_c,
        ],
    ]
}

fn mul_mat3(a: [[f64; 3]; 3], b: [[f64; 3]; 3]) -> [[f64; 3]; 3] {
    let mut out = [[0.0; 3]; 3];
    for r in 0..3 {
        for c in 0..3 {
            out[r][c] = a[r][0] * b[0][c] + a[r][1] * b[1][c] + a[r][2] * b[2][c];
        }
    }
    out
}

fn orthonormalize_mat3(m: [[f64; 3]; 3]) -> [[f64; 3]; 3] {
    let c0 = [m[0][0], m[1][0], m[2][0]];
    let len0 = dot(c0, c0).sqrt();
    let e0 = [c0[0] / len0, c0[1] / len0, c0[2] / len0];

    let c1 = [m[0][1], m[1][1], m[2][1]];
    let d01 = dot(e0, c1);
    let c1_orth = [
        c1[0] - d01 * e0[0],
        c1[1] - d01 * e0[1],
        c1[2] - d01 * e0[2],
    ];
    let len1 = dot(c1_orth, c1_orth).sqrt();
    let e1 = [c1_orth[0] / len1, c1_orth[1] / len1, c1_orth[2] / len1];
    let e2 = cross(e0, e1);

    [
        [e0[0], e1[0], e2[0]],
        [e0[1], e1[1], e2[1]],
        [e0[2], e1[2], e2[2]],
    ]
}

fn build_placed_stamp(
    stamp: &Mesh,
    transform: &PlaceTransform,
    params: &PlaceMeshParams,
    n: [f64; 3],
    target_compact: Option<&CompactWorkMesh>,
) -> Result<Mesh> {
    allocation_allowed()?;
    let stamp_positions = stamp.positions();
    let vslots = stamp.vertex_slots();

    let mut world_pts: Vec<[f64; 3]> = Vec::new();
    reserve(&mut world_pts, vslots)?;
    for p in stamp_positions {
        push(&mut world_pts, transform.apply(*p))?;
    }

    if params.conform_to_surface {
        let tc = target_compact.ok_or(OpsError::InvalidInput(
            "conform_to_surface requires target mesh",
        ))?;
        let target_bvh = build_work_bvh(&tc.mesh)?;
        conform_points_to_surface(
            &tc.mesh,
            &target_bvh,
            tc.min,
            tc.max,
            to_f64(params.hit_point),
            n,
            &mut world_pts,
        )?;
    }

    let mut out_positions: Vec<[f32; 3]> = Vec::new();
    reserve(&mut out_positions, world_pts.len())?;
    for q in &world_pts {
        let pf = [q[0] as f32, q[1] as f32, q[2] as f32];
        if !pf.iter().all(|c| c.is_finite()) {
            return Err(OpsError::InvalidInput(
                "placed stamp vertex position leaves f32 range",
            ));
        }
        push(&mut out_positions, pf)?;
    }

    let mut out_tris: Vec<[u32; 3]> = Vec::new();
    reserve(&mut out_tris, stamp.triangle_count())?;
    for t in 0..stamp.triangle_slots() as TriangleId {
        if let Some(tri) = stamp.triangle(t) {
            push(&mut out_tris, tri)?;
        }
    }

    let mut placed = Mesh::new();
    let mut ptx = placed.begin_edit();
    ptx.add_vertices(&out_positions)?;
    ptx.add_triangles(&out_tris)?;
    ptx.commit()?;
    Ok(placed)
}

fn conform_points_to_surface(
    target: &Mesh,
    target_bvh: &Bvh,
    target_min: [f32; 3],
    target_max: [f32; 3],
    hp: [f64; 3],
    n: [f64; 3],
    world_pts: &mut [[f64; 3]],
) -> Result<()> {
    let dx = f64::from(target_max[0]) - f64::from(target_min[0]);
    let dy = f64::from(target_max[1]) - f64::from(target_min[1]);
    let dz = f64::from(target_max[2]) - f64::from(target_min[2]);
    let target_diag = (dx * dx + dy * dy + dz * dz).sqrt();

    let mut max_rel = 0.0f64;
    for q in world_pts.iter() {
        let rel = [q[0] - hp[0], q[1] - hp[1], q[2] - hp[2]];
        let d = dot(rel, rel).sqrt();
        if d > max_rel {
            max_rel = d;
        }
    }
    let ray_pad = (target_diag + max_rel + 1.0).max(2.0);
    let ray_dir = [-n[0] as f32, -n[1] as f32, -n[2] as f32];
    let ray_opts = RayOptions {
        min_t: 0.0,
        max_t: (2.0 * ray_pad) as f32,
        cull: Cull::None,
    };

    let mut hits: Vec<RayHit> = Vec::new();
    let mut fallback_indices: Vec<usize> = Vec::new();
    let mut fallback_queries: Vec<[f32; 3]> = Vec::new();

    // Record tangent-plane projection offset per vertex: either resolved directly from ray hits
    // or queued for batch closest-point fallback near silhouette edges.
    let mut deltas = filled(world_pts.len(), 0.0f64)?;

    for (idx, q) in world_pts.iter().enumerate() {
        let rel = [q[0] - hp[0], q[1] - hp[1], q[2] - hp[2]];
        let along = dot(rel, n);
        let q_tangent = [
            q[0] - along * n[0],
            q[1] - along * n[1],
            q[2] - along * n[2],
        ];
        let ray_origin = [
            (q_tangent[0] + ray_pad * n[0]) as f32,
            (q_tangent[1] + ray_pad * n[1]) as f32,
            (q_tangent[2] + ray_pad * n[2]) as f32,
        ];
        hits.clear();
        let ray = Ray {
            origin: ray_origin,
            dir: ray_dir,
        };
        raycast_all(target, target_bvh, &ray, &ray_opts, &mut hits)?;

        let mut best_delta: Option<f64> = None;
        let mut best_abs_delta = f64::INFINITY;
        for hit in &hits {
            if dot(to_f64(hit.normal), n) > 0.05 {
                let p_hit = to_f64(hit.point);
                let delta = dot([p_hit[0] - hp[0], p_hit[1] - hp[1], p_hit[2] - hp[2]], n);
                let abs_d = delta.abs();
                if abs_d < best_abs_delta {
                    best_abs_delta = abs_d;
                    best_delta = Some(delta);
                }
            }
        }

        if let Some(delta) = best_delta {
            deltas[idx] = delta;
        } else {
            push(&mut fallback_indices, idx)?;
            push(
                &mut fallback_queries,
                [
                    q_tangent[0] as f32,
                    q_tangent[1] as f32,
                    q_tangent[2] as f32,
                ],
            )?;
        }
    }

    if !fallback_queries.is_empty() {
        let mut cp_hits: Vec<Option<ClosestHit>> = filled(fallback_queries.len(), None)?;
        let cp_opts = ClosestOptions {
            max_distance: f32::INFINITY,
        };
        closest_point_many(
            target,
            target_bvh,
            &fallback_queries,
            &cp_opts,
            &mut cp_hits,
        )?;
        for (&idx, hit_opt) in fallback_indices.iter().zip(cp_hits.iter()) {
            if let Some(hit) = hit_opt {
                let p_cp = to_f64(hit.point);
                deltas[idx] = dot([p_cp[0] - hp[0], p_cp[1] - hp[1], p_cp[2] - hp[2]], n);
            }
        }
    }

    for (q, delta) in world_pts.iter_mut().zip(deltas.iter()) {
        q[0] += delta * n[0];
        q[1] += delta * n[1];
        q[2] += delta * n[2];
    }

    Ok(())
}

fn mesh_to_append_source(mesh: &Mesh) -> Result<AppendSource> {
    let positions = mesh.positions();
    let mut remap: Vec<u32> = filled(mesh.vertex_slots(), u32::MAX)?;
    let mut out_pos: Vec<[f32; 3]> = Vec::new();
    let mut out_tris: Vec<[u32; 3]> = Vec::new();
    reserve(&mut out_pos, mesh.vertex_count())?;
    reserve(&mut out_tris, mesh.triangle_count())?;

    for t in 0..mesh.triangle_slots() as TriangleId {
        let Some(tri) = mesh.triangle(t) else {
            continue;
        };
        let mut mapped = [0u32; 3];
        for (k, v) in tri.into_iter().enumerate() {
            let idx = if remap[v as usize] != u32::MAX {
                remap[v as usize]
            } else {
                let fresh = u32::try_from(out_pos.len())
                    .map_err(|_| OpsError::Store(StoreError::OutOfMemory))?;
                push(&mut out_pos, positions[v as usize])?;
                remap[v as usize] = fresh;
                fresh
            };
            mapped[k] = idx;
        }
        push(&mut out_tris, mapped)?;
    }

    Ok(AppendSource {
        positions: out_pos,
        triangles: out_tris,
    })
}
