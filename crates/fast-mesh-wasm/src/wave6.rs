//! Flat-slice pack/unpack helpers and standalone bindings for Wave 6 domains
//! (`#233` selection, `#193` freeform deformation, `#194` inter-object collision,
//! `#234` metrology/contours/point-cloud, `#236` hollow/drain-holes/place-mesh).

use fast_mesh_ops::boolean::{
    BooleanMethod, DrainHoleOptions, DrainHoleReport, DrainHoleSpec, HollowBackend, HollowOptions,
    HollowReport, PlaceMeshParams, PlaceMeshReport, PlaceMode, PlaceUpAxis,
};
use fast_mesh_ops::deform::{
    ArapAnchor, ArapHandle, ArapOptions, ArapSolver, ArapWeights, FfdBasis, FfdFrame, FfdLattice,
    FfdOptions, FfdOutsideMode, FreeformOp, FreeformReport, KelvinletLoad, KelvinletOptions,
    KelvinletScale, Symmetry, XpbdOptions,
};
use fast_mesh_ops::inspect::{
    ContourLoop, ContourSlice, FitPrimitiveOptions, FitPrimitiveReport, FittedPrimitive,
    OutlierReport, PointCloud, PrimitiveKind, ReconstructMethod, SurfaceReconstructOptions,
    contours_to_csv, contours_to_json, contours_to_svg, estimate_normals, parse_pcd_ascii,
    parse_ply_points, parse_xyz, radius_outlier_removal, statistical_outlier_removal,
    voxel_downsample,
};
use fast_mesh_ops::remesh::Region;
use fast_mesh_ops::repair::{
    AdjacencyKind, FaceEditReport, ScreenSelectOptions, ScreenSelectionShape,
};
use fast_mesh_scene::{
    CollisionDistanceBackend, CollisionMode, CollisionRole, CollisionRule, CollisionStepReport,
    CutTrigger, MaterialMemory, MeshId, Placement, Scene, SceneError, SoftDeformSolver,
};
use fast_mesh_store::StoreError;
use wasm_bindgen::prelude::*;

use crate::edit::count_opt;
use crate::error::Error;

fn oom() -> Error {
    Error::from(StoreError::OutOfMemory)
}

pub(crate) fn count(n: usize) -> f64 {
    n as f64
}

pub(crate) fn req_count(v: f64, what: &str) -> Result<usize, Error> {
    count_opt(v, what)?.ok_or_else(|| Error::invalid(&format!("{what} must not be NaN")))
}

pub(crate) fn f64_to_f32(v: f64, what: &str) -> Result<f32, Error> {
    if !v.is_finite() {
        return Err(Error::invalid(&format!("{what} must be finite")));
    }
    let out = v as f32;
    if !out.is_finite() {
        return Err(Error::invalid(&format!("{what} out of f32 range")));
    }
    Ok(out)
}

pub(crate) fn vec3_f64(v: &[f64], what: &str) -> Result<[f64; 3], Error> {
    let arr: [f64; 3] = v
        .try_into()
        .map_err(|_| Error::invalid(&format!("{what} must have 3 values")))?;
    if !arr.iter().all(|c| c.is_finite()) {
        return Err(Error::invalid(&format!("{what} must be finite")));
    }
    Ok(arr)
}

fn vec3_f32(v: &[f64], what: &str) -> Result<[f32; 3], Error> {
    let d = vec3_f64(v, what)?;
    Ok([
        f64_to_f32(d[0], what)?,
        f64_to_f32(d[1], what)?,
        f64_to_f32(d[2], what)?,
    ])
}

fn parse_boolean_method(code: u8) -> Result<BooleanMethod, Error> {
    match code {
        0 => Ok(BooleanMethod::Manifold),
        1 => Ok(BooleanMethod::BvhSdf),
        2 => Ok(BooleanMethod::FhEdt),
        _ => Err(Error::invalid(
            "booleanMethod must be 0 (manifold), 1 (bvhSdf) or 2 (fhEdt)",
        )),
    }
}

pub(crate) fn parse_region_opt(
    center: Option<Vec<f64>>,
    radius: f64,
) -> Result<Option<Region>, Error> {
    match center {
        None => Ok(None),
        Some(c) => {
            let center = vec3_f32(&c, "region.center")?;
            let radius = f64_to_f32(radius, "region.radius")?;
            if radius < 0.0 {
                return Err(Error::invalid("region.radius must be non-negative"));
            }
            Ok(Some(Region { center, radius }))
        }
    }
}

pub(crate) fn apply_freeform_with_protection(
    scene: &mut Scene,
    id: MeshId,
    op: &FreeformOp,
    protected_verts: Option<Vec<u32>>,
    protection_weights: Option<Vec<f32>>,
) -> Result<FreeformReport, Error> {
    let rep = match (protected_verts, protection_weights) {
        (None, None) => scene.apply_freeform(id, op)?,
        (None, Some(_)) => {
            return Err(Error::invalid(
                "protectionWeights requires protectedVertices",
            ));
        }
        (Some(verts), weights) => {
            scene.apply_freeform_protected(id, op, &verts, weights.as_deref())?
        }
    };
    Ok(rep)
}

// ---------------------------------------------------------------------------
// #233 Face Selection Helpers
// ---------------------------------------------------------------------------

pub(crate) fn parse_screen_shape(kind: u8, data: &[f64]) -> Result<ScreenSelectionShape, Error> {
    match kind {
        0 => {
            let [x0, y0, x1, y1]: [f64; 4] = data
                .try_into()
                .map_err(|_| Error::invalid("screen rect requires 4 coordinates"))?;
            Ok(ScreenSelectionShape::Rect {
                min_ndc: [f64_to_f32(x0, "minNdc[0]")?, f64_to_f32(y0, "minNdc[1]")?],
                max_ndc: [f64_to_f32(x1, "maxNdc[0]")?, f64_to_f32(y1, "maxNdc[1]")?],
            })
        }
        1 => {
            let (pairs, rest) = data.as_chunks::<2>();
            if !rest.is_empty() || pairs.len() < 3 {
                return Err(Error::invalid(
                    "screen lasso requires at least 3 (x, y) NDC coordinate pairs",
                ));
            }
            let mut polygon_ndc = Vec::new();
            polygon_ndc.try_reserve(pairs.len()).map_err(|_| oom())?;
            for &[x, y] in pairs {
                polygon_ndc.push([f64_to_f32(x, "polygonNdc")?, f64_to_f32(y, "polygonNdc")?]);
            }
            Ok(ScreenSelectionShape::Lasso { polygon_ndc })
        }
        _ => Err(Error::invalid("shapeKind must be 0 (rect) or 1 (lasso)")),
    }
}

pub(crate) fn parse_screen_options(
    view_proj16: &[f64],
    camera_origin3: &[f64],
    camera_forward3: &[f64],
    is_orthographic: bool,
    visible_only: bool,
    reject_backfaces: bool,
    include_touching: bool,
) -> Result<ScreenSelectOptions, Error> {
    if view_proj16.len() != 16 {
        return Err(Error::invalid(
            "viewProj must be a 16-element column-major 4x4 matrix",
        ));
    }
    let mut view_proj = [[0.0_f32; 4]; 4];
    for c in 0..4 {
        for r in 0..4 {
            view_proj[c][r] = f64_to_f32(view_proj16[c * 4 + r], "viewProj")?;
        }
    }
    Ok(ScreenSelectOptions {
        view_proj,
        camera_origin: vec3_f32(camera_origin3, "cameraOrigin")?,
        camera_forward: vec3_f32(camera_forward3, "cameraForward")?,
        is_orthographic,
        visible_only,
        reject_backfaces,
        include_touching,
    })
}

pub(crate) fn parse_adjacency(code: u8) -> Result<AdjacencyKind, Error> {
    match code {
        0 => Ok(AdjacencyKind::Edge),
        1 => Ok(AdjacencyKind::Vertex),
        _ => Err(Error::invalid("adjacency must be 0 (edge) or 1 (vertex)")),
    }
}

pub(crate) fn encode_face_edit_report(r: &FaceEditReport) -> Vec<f64> {
    vec![count(r.removed_triangles), count(r.removed_vertices)]
}

// ---------------------------------------------------------------------------
// #193 Free-Form Deformation Helpers
// ---------------------------------------------------------------------------

pub(crate) fn parse_ffd_basis(code: u8) -> Result<FfdBasis, Error> {
    match code {
        0 => Ok(FfdBasis::Bernstein),
        1 => Ok(FfdBasis::CubicBSpline),
        _ => Err(Error::invalid(
            "basis must be 0 (bernstein) or 1 (cubicBSpline)",
        )),
    }
}

pub(crate) fn ffd_basis_code(basis: FfdBasis) -> f64 {
    match basis {
        FfdBasis::Bernstein => 0.0,
        FfdBasis::CubicBSpline => 1.0,
    }
}

pub(crate) fn encode_ffd_lattice(lattice: &FfdLattice) -> Result<Vec<f64>, Error> {
    let dims = lattice.dims();
    let frame = lattice.frame();
    let offsets = lattice.offsets();
    let mut out = Vec::new();
    out.try_reserve(19 + offsets.len() * 3).map_err(|_| oom())?;
    out.extend_from_slice(&[
        count(dims[0]),
        count(dims[1]),
        count(dims[2]),
        ffd_basis_code(lattice.basis()),
        f64::from(frame.center[0]),
        f64::from(frame.center[1]),
        f64::from(frame.center[2]),
        f64::from(frame.axes[0][0]),
        f64::from(frame.axes[0][1]),
        f64::from(frame.axes[0][2]),
        f64::from(frame.axes[1][0]),
        f64::from(frame.axes[1][1]),
        f64::from(frame.axes[1][2]),
        f64::from(frame.axes[2][0]),
        f64::from(frame.axes[2][1]),
        f64::from(frame.axes[2][2]),
        f64::from(frame.half_extents[0]),
        f64::from(frame.half_extents[1]),
        f64::from(frame.half_extents[2]),
    ]);
    for &o in offsets {
        out.extend_from_slice(&[f64::from(o[0]), f64::from(o[1]), f64::from(o[2])]);
    }
    Ok(out)
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn decode_ffd_op(
    dims: &[u32],
    basis: u8,
    frame15: &[f64],
    offsets: &[f64],
    outside_mode: u8,
    region_center: Option<Vec<f64>>,
    region_radius: f64,
    vertices: Option<Vec<u32>>,
) -> Result<FreeformOp, Error> {
    let [nx, ny, nz]: [u32; 3] = dims
        .try_into()
        .map_err(|_| Error::invalid("dims must be [nx, ny, nz]"))?;
    let dims = [nx as usize, ny as usize, nz as usize];
    let basis = parse_ffd_basis(basis)?;
    if frame15.len() != 15 {
        return Err(Error::invalid(
            "ffd frame must have 15 values (center3, axes9, halfExtents3)",
        ));
    }
    let center = vec3_f32(&frame15[0..3], "frame.center")?;
    let axes = [
        vec3_f32(&frame15[3..6], "frame.axes[0]")?,
        vec3_f32(&frame15[6..9], "frame.axes[1]")?,
        vec3_f32(&frame15[9..12], "frame.axes[2]")?,
    ];
    let half_extents = vec3_f32(&frame15[12..15], "frame.halfExtents")?;
    let frame = FfdFrame::new(center, axes, half_extents)?;
    let mut lattice = FfdLattice::new(dims, frame, basis)?;
    let expected = dims[0]
        .checked_mul(dims[1])
        .and_then(|v| v.checked_mul(dims[2]))
        .and_then(|v| v.checked_mul(3))
        .ok_or_else(|| Error::invalid("ffd dimensions overflow"))?;
    if offsets.len() != expected {
        return Err(Error::invalid(
            "ffd offsets length does not match 3 * nx * ny * nz",
        ));
    }
    for (slot, chunk) in lattice
        .offsets_mut()
        .iter_mut()
        .zip(offsets.as_chunks::<3>().0)
    {
        *slot = vec3_f32(chunk, "ffd.offsets")?;
    }
    let outside = match outside_mode {
        0 => FfdOutsideMode::Ignore,
        1 => FfdOutsideMode::Clamp,
        _ => {
            return Err(Error::invalid(
                "outsideMode must be 0 (ignore) or 1 (clamp)",
            ));
        }
    };
    let mut opts = FfdOptions::new(lattice).with_outside(outside);
    opts.region = parse_region_opt(region_center, region_radius)?;
    opts.vertices = vertices;
    Ok(FreeformOp::Ffd(opts))
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn decode_arap_op(
    anchor_verts: &[u32],
    anchor_pos: &[f64],
    anchor_has_pos: &[u8],
    handle_verts: &[u32],
    handle_targets: &[f64],
    weights: u8,
    solver: u8,
    iterations: f64,
    tolerance: f64,
    region_center: Option<Vec<f64>>,
    region_radius: f64,
    vertices: Option<Vec<u32>>,
) -> Result<FreeformOp, Error> {
    if anchor_pos.len() != anchor_verts.len() * 3 || anchor_has_pos.len() != anchor_verts.len() {
        return Err(Error::invalid("malformed ARAP anchors"));
    }
    if handle_targets.len() != handle_verts.len() * 3 {
        return Err(Error::invalid("malformed ARAP handles"));
    }
    let mut anchors = Vec::new();
    anchors.try_reserve(anchor_verts.len()).map_err(|_| oom())?;
    for (i, &v) in anchor_verts.iter().enumerate() {
        let pos = if anchor_has_pos[i] != 0 {
            Some(vec3_f32(&anchor_pos[i * 3..i * 3 + 3], "anchor.position")?)
        } else {
            None
        };
        anchors.push(ArapAnchor {
            vertex: v,
            position: pos,
        });
    }
    let mut handles = Vec::new();
    handles.try_reserve(handle_verts.len()).map_err(|_| oom())?;
    for (i, &v) in handle_verts.iter().enumerate() {
        let target = vec3_f32(&handle_targets[i * 3..i * 3 + 3], "handle.target")?;
        handles.push(ArapHandle { vertex: v, target });
    }
    let weights = match weights {
        0 => ArapWeights::Cotangent,
        1 => ArapWeights::Uniform,
        _ => {
            return Err(Error::invalid(
                "weights must be 0 (cotangent) or 1 (uniform)",
            ));
        }
    };
    let solver = match solver {
        0 => ArapSolver::Auto,
        1 => ArapSolver::Direct,
        2 => ArapSolver::Pcg,
        _ => {
            return Err(Error::invalid(
                "solver must be 0 (auto), 1 (direct) or 2 (pcg)",
            ));
        }
    };
    let mut opts = ArapOptions::new()
        .with_anchors(anchors)
        .with_handles(handles);
    opts.weights = weights;
    opts.solver = solver;
    opts.iterations = req_count(iterations, "iterations")?;
    opts.tolerance = f64_to_f32(tolerance, "tolerance")?;
    opts.region = parse_region_opt(region_center, region_radius)?;
    opts.vertices = vertices;
    Ok(FreeformOp::Arap(opts))
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn decode_kelvinlet_op(
    center3: &[f64],
    load_kind: u8,
    load_data: &[f64],
    epsilon: f64,
    nu: f64,
    mu: f64,
    scale_mode: u8,
    scale_alpha: f64,
    sym_x: bool,
    sym_y: bool,
    sym_z: bool,
    region_center: Option<Vec<f64>>,
    region_radius: f64,
    vertices: Option<Vec<u32>>,
) -> Result<FreeformOp, Error> {
    let center = vec3_f32(center3, "center")?;
    let load = match load_kind {
        0 => KelvinletLoad::Grab {
            force: vec3_f32(
                load_data
                    .get(..3)
                    .ok_or_else(|| Error::invalid("kelvinlet grab requires 3 force values"))?,
                "force",
            )?,
        },
        1 => {
            let m = *load_data
                .first()
                .ok_or_else(|| Error::invalid("kelvinlet scale requires magnitude"))?;
            KelvinletLoad::Scale {
                magnitude: f64_to_f32(m, "magnitude")?,
            }
        }
        2 => KelvinletLoad::Twist {
            torque: vec3_f32(
                load_data
                    .get(..3)
                    .ok_or_else(|| Error::invalid("kelvinlet twist requires 3 torque values"))?,
                "torque",
            )?,
        },
        3 => {
            let s = load_data
                .get(..9)
                .ok_or_else(|| Error::invalid("kelvinlet pinch requires 9 dipole values"))?;
            KelvinletLoad::Pinch {
                force_dipole: [
                    vec3_f32(&s[0..3], "forceDipole[0]")?,
                    vec3_f32(&s[3..6], "forceDipole[1]")?,
                    vec3_f32(&s[6..9], "forceDipole[2]")?,
                ],
            }
        }
        _ => {
            return Err(Error::invalid(
                "loadKind must be 0 (grab), 1 (scale), 2 (twist) or 3 (pinch)",
            ));
        }
    };
    let scale = match scale_mode {
        0 => KelvinletScale::Single,
        1 => KelvinletScale::BiScale {
            alpha: f64_to_f32(scale_alpha, "scaleAlpha")?,
        },
        2 => KelvinletScale::TriScale,
        _ => {
            return Err(Error::invalid(
                "scaleMode must be 0 (single), 1 (bi) or 2 (tri)",
            ));
        }
    };
    let mut opts = KelvinletOptions::new(center, load)
        .with_epsilon(f64_to_f32(epsilon, "epsilon")?)
        .with_nu(f64_to_f32(nu, "nu")?)
        .with_mu(f64_to_f32(mu, "mu")?)
        .with_scale(scale)
        .with_symmetry(Symmetry {
            x: sym_x,
            y: sym_y,
            z: sym_z,
        });
    opts.region = parse_region_opt(region_center, region_radius)?;
    opts.vertices = vertices;
    Ok(FreeformOp::Kelvinlet(opts))
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn decode_xpbd_op(
    substeps: f64,
    iterations: f64,
    dt: f64,
    stretch_compliance: f64,
    bend_compliance: f64,
    volume_compliance: f64,
    damping: f64,
    gravity3: &[f64],
    pin_verts: &[u32],
    pin_targets: &[f64],
    handle_verts: &[u32],
    handle_targets: &[f64],
    handle_compliances: &[f64],
    self_collision: bool,
    collision_thickness: f64,
) -> Result<FreeformOp, Error> {
    if pin_targets.len() != pin_verts.len() * 3 {
        return Err(Error::invalid("malformed XPBD pins"));
    }
    if handle_targets.len() != handle_verts.len() * 3
        || handle_compliances.len() != handle_verts.len()
    {
        return Err(Error::invalid("malformed XPBD handles"));
    }
    let mut pins = Vec::new();
    pins.try_reserve(pin_verts.len()).map_err(|_| oom())?;
    for (i, &v) in pin_verts.iter().enumerate() {
        pins.push((v, vec3_f32(&pin_targets[i * 3..i * 3 + 3], "pin.target")?));
    }
    let mut handles = Vec::new();
    handles.try_reserve(handle_verts.len()).map_err(|_| oom())?;
    for (i, &v) in handle_verts.iter().enumerate() {
        handles.push((
            v,
            vec3_f32(&handle_targets[i * 3..i * 3 + 3], "handle.target")?,
            f64_to_f32(handle_compliances[i], "handle.compliance")?,
        ));
    }
    let volume_compliance = if volume_compliance.is_nan() {
        None
    } else {
        Some(f64_to_f32(volume_compliance, "volumeCompliance")?)
    };
    let mut opts = XpbdOptions::new()
        .with_substeps(req_count(substeps, "substeps")?)
        .with_iterations(req_count(iterations, "iterations")?)
        .with_dt(f64_to_f32(dt, "dt")?)
        .with_stretch_compliance(f64_to_f32(stretch_compliance, "stretchCompliance")?)
        .with_bend_compliance(f64_to_f32(bend_compliance, "bendCompliance")?);
    opts.volume_compliance = volume_compliance;
    opts.damping = f64_to_f32(damping, "damping")?;
    opts.gravity = vec3_f32(gravity3, "gravity")?;
    opts.pins = pins;
    opts.handles = handles;
    opts.self_collision = self_collision;
    opts.collision_thickness = f64_to_f32(collision_thickness, "collisionThickness")?;
    Ok(FreeformOp::Xpbd(opts))
}

pub(crate) fn encode_freeform_report(r: &FreeformReport) -> Vec<f64> {
    vec![
        count(r.vertices_evaluated),
        count(r.vertices_moved),
        f64::from(r.max_displacement),
        count(r.iterations),
        if r.converged { 1.0 } else { 0.0 },
        count(r.self_collision_contacts_resolved),
    ]
}

// ---------------------------------------------------------------------------
// #194 Inter-Object Collision Helpers
// ---------------------------------------------------------------------------

fn collision_mode_code(m: CollisionMode) -> f64 {
    match m {
        CollisionMode::PassThrough => 0.0,
        CollisionMode::Rigid => 1.0,
        CollisionMode::Cut => 2.0,
        CollisionMode::SoftDeform => 3.0,
    }
}

fn collision_role_code(r: CollisionRole) -> f64 {
    match r {
        CollisionRole::Symmetric => 0.0,
        CollisionRole::MovingIsCutter => 1.0,
        CollisionRole::MovingIsTarget => 2.0,
    }
}

fn distance_backend_code(b: CollisionDistanceBackend) -> f64 {
    match b {
        CollisionDistanceBackend::BvhSdf => 0.0,
        CollisionDistanceBackend::FhEdt => 1.0,
    }
}

fn boolean_method_code(m: BooleanMethod) -> f64 {
    match m {
        BooleanMethod::Manifold => 0.0,
        BooleanMethod::BvhSdf => 1.0,
        BooleanMethod::FhEdt => 2.0,
    }
}

fn cut_trigger_code(t: CutTrigger) -> f64 {
    match t {
        CutTrigger::OnRelease => 0.0,
        CutTrigger::LiveStep => 1.0,
    }
}

fn soft_solver_code(s: SoftDeformSolver) -> f64 {
    match s {
        SoftDeformSolver::SdfLaplacian => 0.0,
        SoftDeformSolver::Kelvinlet => 1.0,
        SoftDeformSolver::Xpbd => 2.0,
    }
}

fn material_memory_code(m: MaterialMemory) -> f64 {
    match m {
        MaterialMemory::Elastic => 0.0,
        MaterialMemory::Plastic => 1.0,
    }
}

fn flag01(v: f64, what: &str) -> Result<bool, Error> {
    if v == 0.0 {
        Ok(false)
    } else if v == 1.0 {
        Ok(true)
    } else {
        Err(Error::invalid(&format!("{what} must be 0 or 1")))
    }
}

fn enum_u8(v: f64, what: &str) -> Result<u8, Error> {
    if !v.is_finite() || v.fract() != 0.0 || !(0.0..=255.0).contains(&v) {
        return Err(Error::invalid(&format!("invalid {what} code")));
    }
    Ok(v as u8)
}

/// Encodes a [`CollisionRule`] into its fixed 16-element representation.
pub(crate) fn encode_collision_rule(rule: &CollisionRule) -> Vec<f64> {
    vec![
        collision_mode_code(rule.mode),
        collision_role_code(rule.role),
        distance_backend_code(rule.distance_backend),
        boolean_method_code(rule.cut_method),
        cut_trigger_code(rule.cut_trigger),
        soft_solver_code(rule.soft_solver),
        material_memory_code(rule.material_memory),
        f64::from(rule.compliance),
        f64::from(rule.stiffness),
        f64::from(rule.recovery_rate),
        rule.contact_margin,
        if rule.self_collision_guard { 1.0 } else { 0.0 },
        if rule.rigid_slide { 1.0 } else { 0.0 },
        0.0,
        0.0,
        0.0,
    ]
}

/// Decodes a fixed 16-element flat array into a [`CollisionRule`].
pub(crate) fn decode_collision_rule(flat: &[f64]) -> Result<CollisionRule, Error> {
    if flat.len() != 16 {
        return Err(Error::invalid("collisionRule must have 16 elements"));
    }
    let mode = match enum_u8(flat[0], "mode")? {
        0 => CollisionMode::PassThrough,
        1 => CollisionMode::Rigid,
        2 => CollisionMode::Cut,
        3 => CollisionMode::SoftDeform,
        _ => {
            return Err(Error::invalid(
                "mode must be 0 (passThrough), 1 (rigid), 2 (cut) or 3 (softDeform)",
            ));
        }
    };
    let role = match enum_u8(flat[1], "role")? {
        0 => CollisionRole::Symmetric,
        1 => CollisionRole::MovingIsCutter,
        2 => CollisionRole::MovingIsTarget,
        _ => {
            return Err(Error::invalid(
                "role must be 0 (symmetric), 1 (movingIsCutter) or 2 (movingIsTarget)",
            ));
        }
    };
    let distance_backend = match enum_u8(flat[2], "distanceBackend")? {
        0 => CollisionDistanceBackend::BvhSdf,
        1 => CollisionDistanceBackend::FhEdt,
        _ => return Err(Error::invalid("distanceBackend must be 0 or 1")),
    };
    let cut_method = parse_boolean_method(enum_u8(flat[3], "cutMethod")?)?;
    let cut_trigger = match enum_u8(flat[4], "cutTrigger")? {
        0 => CutTrigger::OnRelease,
        1 => CutTrigger::LiveStep,
        _ => return Err(Error::invalid("cutTrigger must be 0 or 1")),
    };
    let soft_solver = match enum_u8(flat[5], "softSolver")? {
        0 => SoftDeformSolver::SdfLaplacian,
        1 => SoftDeformSolver::Kelvinlet,
        2 => SoftDeformSolver::Xpbd,
        _ => return Err(Error::invalid("softSolver must be 0, 1 or 2")),
    };
    let material_memory = match enum_u8(flat[6], "materialMemory")? {
        0 => MaterialMemory::Elastic,
        1 => MaterialMemory::Plastic,
        _ => return Err(Error::invalid("materialMemory must be 0 or 1")),
    };
    if !flat[10].is_finite() {
        return Err(Error::invalid("contactMargin must be finite"));
    }
    Ok(CollisionRule {
        mode,
        role,
        distance_backend,
        cut_method,
        cut_trigger,
        soft_solver,
        material_memory,
        compliance: f64_to_f32(flat[7], "compliance")?,
        stiffness: f64_to_f32(flat[8], "stiffness")?,
        recovery_rate: f64_to_f32(flat[9], "recoveryRate")?,
        contact_margin: flat[10],
        self_collision_guard: flag01(flat[11], "selfCollisionGuard")?,
        rigid_slide: flag01(flat[12], "rigidSlide")?,
    })
}

pub(crate) fn parse_placement(
    rotation: &[f64],
    translation: &[f64],
    scale: Option<Vec<f64>>,
) -> Result<Placement, Error> {
    let r: [f64; 9] = rotation
        .try_into()
        .map_err(|_| Error::invalid("rotation must have 9 values"))?;
    let t = vec3_f64(translation, "translation")?;
    let s: [f64; 3] = match scale {
        None => [1.0; 3],
        Some(sv) => vec3_f64(&sv, "scale")?,
    };
    Ok(Placement {
        rotation: [[r[0], r[1], r[2]], [r[3], r[4], r[5]], [r[6], r[7], r[8]]],
        translation: t,
        scale: s,
    })
}

pub(crate) fn encode_placement(p: &Placement) -> Vec<f64> {
    let mut out = p.rotation.as_flattened().to_vec();
    out.extend_from_slice(&p.translation);
    out.extend_from_slice(&p.scale);
    out
}

/// Converts a proper 3x3 rotation matrix to a unit quaternion `[x, y, z, w]`.
fn mat3_to_quat(r: [[f64; 3]; 3]) -> [f64; 4] {
    let trace = r[0][0] + r[1][1] + r[2][2];
    let (x, y, z, w) = if trace > 0.0 {
        let s = (trace + 1.0).sqrt() * 2.0;
        (
            (r[2][1] - r[1][2]) / s,
            (r[0][2] - r[2][0]) / s,
            (r[1][0] - r[0][1]) / s,
            0.25 * s,
        )
    } else if r[0][0] > r[1][1] && r[0][0] > r[2][2] {
        let s = (1.0 + r[0][0] - r[1][1] - r[2][2]).sqrt() * 2.0;
        (
            0.25 * s,
            (r[0][1] + r[1][0]) / s,
            (r[0][2] + r[2][0]) / s,
            (r[2][1] - r[1][2]) / s,
        )
    } else if r[1][1] > r[2][2] {
        let s = (1.0 + r[1][1] - r[0][0] - r[2][2]).sqrt() * 2.0;
        (
            (r[0][1] + r[1][0]) / s,
            0.25 * s,
            (r[1][2] + r[2][1]) / s,
            (r[0][2] - r[2][0]) / s,
        )
    } else {
        let s = (1.0 + r[2][2] - r[0][0] - r[1][1]).sqrt() * 2.0;
        (
            (r[0][2] + r[2][0]) / s,
            (r[1][2] + r[2][1]) / s,
            0.25 * s,
            (r[1][0] - r[0][1]) / s,
        )
    };
    let n = (x * x + y * y + z * z + w * w).sqrt();
    if n > 0.0 {
        [x / n, y / n, z / n, w / n]
    } else {
        [0.0, 0.0, 0.0, 1.0]
    }
}

/// Encodes `[appliedPose7, pairCount, ...pairs(13 fields each), modifiedCount, ...modifiedMeshIds(2 fields each), summary5]`.
pub(crate) fn encode_collision_step_report(
    scene: &Scene,
    moving: MeshId,
    report: &CollisionStepReport,
) -> Result<Vec<f64>, Error> {
    let q = mat3_to_quat(report.final_placement.rotation);
    let mut modified: Vec<MeshId> = Vec::new();
    let mut push_unique = |id: MeshId| {
        if scene.mesh(id).is_ok() && !modified.contains(&id) {
            modified.push(id);
        }
    };
    for p in &report.active_pairs {
        if p.cut_executed {
            let target = match p.role {
                CollisionRole::MovingIsTarget => moving,
                CollisionRole::Symmetric | CollisionRole::MovingIsCutter => p.other,
            };
            push_unique(target);
        }
        if p.soft_deformed_vertices > 0 || p.elastic_recovered_vertices > 0 {
            match p.role {
                CollisionRole::MovingIsCutter => push_unique(p.other),
                CollisionRole::MovingIsTarget => push_unique(moving),
                CollisionRole::Symmetric => {
                    push_unique(p.other);
                    push_unique(moving);
                }
            }
        }
    }

    let cap = 7 + 1 + report.active_pairs.len() * 13 + 1 + modified.len() * 2 + 5;
    let mut out = Vec::new();
    out.try_reserve(cap).map_err(|_| oom())?;
    out.extend_from_slice(&report.final_placement.translation);
    out.extend_from_slice(&q);
    out.push(count(report.active_pairs.len()));
    for p in &report.active_pairs {
        out.extend_from_slice(&[
            f64::from(p.other.index()),
            f64::from(p.other.generation()),
            collision_mode_code(p.mode),
            collision_role_code(p.role),
            if p.intersecting { 1.0 } else { 0.0 },
            p.min_distance,
            if p.contact_blocked { 1.0 } else { 0.0 },
            if p.slidden { 1.0 } else { 0.0 },
            if p.cut_executed { 1.0 } else { 0.0 },
            if p.preview_only { 1.0 } else { 0.0 },
            count(p.soft_deformed_vertices),
            count(p.elastic_recovered_vertices),
            if p.self_collision_guard_triggered {
                1.0
            } else {
                0.0
            },
        ]);
    }
    out.push(count(modified.len()));
    for id in modified {
        out.push(f64::from(id.index()));
        out.push(f64::from(id.generation()));
    }
    out.extend_from_slice(&[
        report.min_clearance,
        count(report.cuts_executed),
        count(report.soft_vertices_deformed),
        count(report.elastic_recovery_count),
        if report.self_intersection_guard_triggered {
            1.0
        } else {
            0.0
        },
    ]);
    Ok(out)
}

// ---------------------------------------------------------------------------
// #234 Primitive Fit, Contours & Point Cloud Helpers
// ---------------------------------------------------------------------------

pub(crate) fn parse_primitive_kind(code: i32) -> Result<Option<PrimitiveKind>, Error> {
    match code {
        -1 | 4 => Ok(Some(PrimitiveKind::Auto)),
        0 => Ok(Some(PrimitiveKind::Plane)),
        1 => Ok(Some(PrimitiveKind::Sphere)),
        2 => Ok(Some(PrimitiveKind::Cylinder)),
        3 => Ok(Some(PrimitiveKind::Cone)),
        _ => Err(Error::invalid(
            "primitiveKind must be 0 (plane), 1 (sphere), 2 (cylinder), 3 (cone) or 4 (auto)",
        )),
    }
}

pub(crate) fn parse_fit_options(
    inlier_threshold: f64,
    ransac_iterations: f64,
    lm_iterations: f64,
    seed: f64,
    min_inlier_ratio: f64,
    normal_weight: f64,
) -> Result<FitPrimitiveOptions, Error> {
    let defaults = FitPrimitiveOptions::default();
    let inlier_threshold = if inlier_threshold.is_nan() {
        defaults.inlier_threshold
    } else {
        inlier_threshold
    };
    let ransac_iterations =
        count_opt(ransac_iterations, "ransacIterations")?.unwrap_or(defaults.ransac_iterations);
    let lm_iterations = count_opt(lm_iterations, "lmIterations")?.unwrap_or(defaults.lm_iterations);
    let seed = if seed.is_nan() {
        defaults.seed
    } else if !seed.is_finite() || seed < 0.0 || seed.fract() != 0.0 {
        return Err(Error::invalid("seed must be a non-negative integer"));
    } else {
        seed as u64
    };
    let min_inlier_ratio = if min_inlier_ratio.is_nan() {
        defaults.min_inlier_ratio
    } else {
        min_inlier_ratio
    };
    let normal_weight = if normal_weight.is_nan() {
        defaults.normal_weight
    } else {
        normal_weight
    };
    Ok(FitPrimitiveOptions {
        inlier_threshold,
        ransac_iterations,
        lm_iterations,
        seed,
        min_inlier_ratio,
        normal_weight,
    })
}

pub(crate) fn encode_fitted_primitive(p: &FittedPrimitive) -> [f64; 9] {
    match *p {
        FittedPrimitive::Plane { origin, normal } => [
            0.0, origin[0], origin[1], origin[2], normal[0], normal[1], normal[2], 0.0, 0.0,
        ],
        FittedPrimitive::Sphere { center, radius } => [
            1.0, center[0], center[1], center[2], radius, 0.0, 0.0, 0.0, 0.0,
        ],
        FittedPrimitive::Cylinder {
            axis_origin,
            axis_dir,
            radius,
            height,
        } => [
            2.0,
            axis_origin[0],
            axis_origin[1],
            axis_origin[2],
            axis_dir[0],
            axis_dir[1],
            axis_dir[2],
            radius,
            height,
        ],
        FittedPrimitive::Cone {
            apex,
            axis_dir,
            half_angle_rad,
            height,
        } => [
            3.0,
            apex[0],
            apex[1],
            apex[2],
            axis_dir[0],
            axis_dir[1],
            axis_dir[2],
            half_angle_rad,
            height,
        ],
    }
}

pub(crate) fn decode_fitted_primitive(prim9: &[f64]) -> Result<FittedPrimitive, Error> {
    if prim9.len() != 9 || !prim9.iter().all(|v| v.is_finite()) {
        return Err(Error::invalid("primitive must have 9 finite values"));
    }
    let kind = enum_u8(prim9[0], "primitive.kind")?;
    match kind {
        0 => Ok(FittedPrimitive::Plane {
            origin: [prim9[1], prim9[2], prim9[3]],
            normal: [prim9[4], prim9[5], prim9[6]],
        }),
        1 => Ok(FittedPrimitive::Sphere {
            center: [prim9[1], prim9[2], prim9[3]],
            radius: prim9[4],
        }),
        2 => Ok(FittedPrimitive::Cylinder {
            axis_origin: [prim9[1], prim9[2], prim9[3]],
            axis_dir: [prim9[4], prim9[5], prim9[6]],
            radius: prim9[7],
            height: prim9[8],
        }),
        3 => Ok(FittedPrimitive::Cone {
            apex: [prim9[1], prim9[2], prim9[3]],
            axis_dir: [prim9[4], prim9[5], prim9[6]],
            half_angle_rad: prim9[7],
            height: prim9[8],
        }),
        _ => Err(Error::invalid(
            "primitive.kind must be 0 (plane), 1 (sphere), 2 (cylinder) or 3 (cone)",
        )),
    }
}

pub(crate) fn encode_fit_report(r: &FitPrimitiveReport) -> Result<Vec<f64>, Error> {
    let mut out = Vec::new();
    out.try_reserve(14 + r.residuals.len()).map_err(|_| oom())?;
    out.extend_from_slice(&encode_fitted_primitive(&r.primitive));
    out.extend_from_slice(&[
        r.rms_error,
        r.max_error,
        count(r.inlier_count),
        r.inlier_ratio,
        count(r.residuals.len()),
    ]);
    for &res in &r.residuals {
        out.push(f64::from(res));
    }
    Ok(out)
}

pub(crate) fn encode_contour_slices(slices: &[ContourSlice]) -> Result<Vec<f64>, Error> {
    let mut out = Vec::new();
    out.try_reserve(1 + slices.len() * 16).map_err(|_| oom())?;
    out.push(count(slices.len()));
    for s in slices {
        out.extend_from_slice(&[
            s.plane_origin[0],
            s.plane_origin[1],
            s.plane_origin[2],
            f64::from(s.plane_normal[0]),
            f64::from(s.plane_normal[1]),
            f64::from(s.plane_normal[2]),
            s.u_axis[0],
            s.u_axis[1],
            s.u_axis[2],
            s.v_axis[0],
            s.v_axis[1],
            s.v_axis[2],
            s.total_perimeter,
            s.total_area,
            count(s.loops.len()),
        ]);
        for l in &s.loops {
            let npts = l.points_2d.len().min(l.points_3d.len());
            out.try_reserve(11 + npts * 5).map_err(|_| oom())?;
            out.extend_from_slice(&[
                if l.closed { 1.0 } else { 0.0 },
                l.perimeter,
                l.enclosed_area,
                l.centroid_2d[0],
                l.centroid_2d[1],
                l.centroid_3d[0],
                l.centroid_3d[1],
                l.centroid_3d[2],
                l.min_caliper_diameter,
                l.max_caliper_diameter,
                count(npts),
            ]);
            for p2 in l.points_2d.iter().take(npts) {
                out.extend_from_slice(p2);
            }
            for p3 in l.points_3d.iter().take(npts) {
                out.extend_from_slice(p3);
            }
        }
    }
    Ok(out)
}

pub(crate) fn decode_contour_slices(flat: &[f64]) -> Result<Vec<ContourSlice>, Error> {
    let mut cur = 0usize;
    let next = |cur: &mut usize, what: &str| -> Result<f64, Error> {
        let v = *flat
            .get(*cur)
            .ok_or_else(|| Error::invalid(&format!("truncated contour {what}")))?;
        *cur += 1;
        Ok(v)
    };
    let next3_f32 = |cur: &mut usize, what: &str| -> Result<[f32; 3], Error> {
        let a = next(cur, what)?;
        let b = next(cur, what)?;
        let c = next(cur, what)?;
        Ok([
            f64_to_f32(a, what)?,
            f64_to_f32(b, what)?,
            f64_to_f32(c, what)?,
        ])
    };
    let next3_f64 = |cur: &mut usize, what: &str| -> Result<[f64; 3], Error> {
        let slice = flat
            .get(*cur..*cur + 3)
            .ok_or_else(|| Error::invalid(&format!("truncated contour {what}")))?;
        *cur += 3;
        vec3_f64(slice, what)
    };

    let slice_count = req_count(next(&mut cur, "sliceCount")?, "sliceCount")?;
    let mut slices = Vec::new();
    slices.try_reserve(slice_count).map_err(|_| oom())?;
    for _ in 0..slice_count {
        let plane_origin = next3_f64(&mut cur, "planeOrigin")?;
        let plane_normal = next3_f32(&mut cur, "planeNormal")?;
        let u_axis = next3_f64(&mut cur, "uAxis")?;
        let v_axis = next3_f64(&mut cur, "vAxis")?;
        let total_perimeter = next(&mut cur, "totalPerimeter")?;
        let total_area = next(&mut cur, "totalArea")?;
        let loop_count = req_count(next(&mut cur, "loopCount")?, "loopCount")?;
        let mut loops = Vec::new();
        loops.try_reserve(loop_count).map_err(|_| oom())?;
        for _ in 0..loop_count {
            let closed = flag01(next(&mut cur, "closed")?, "closed")?;
            let perimeter = next(&mut cur, "perimeter")?;
            let enclosed_area = next(&mut cur, "enclosedArea")?;
            let centroid_2d = [next(&mut cur, "centroid2d")?, next(&mut cur, "centroid2d")?];
            let centroid_3d = next3_f64(&mut cur, "centroid3d")?;
            let min_caliper_diameter = next(&mut cur, "minCaliperDiameter")?;
            let max_caliper_diameter = next(&mut cur, "maxCaliperDiameter")?;
            let point_count = req_count(next(&mut cur, "pointCount")?, "pointCount")?;
            let mut points_2d = Vec::new();
            points_2d.try_reserve(point_count).map_err(|_| oom())?;
            for _ in 0..point_count {
                points_2d.push([next(&mut cur, "point2d")?, next(&mut cur, "point2d")?]);
            }
            let mut points_3d = Vec::new();
            points_3d.try_reserve(point_count).map_err(|_| oom())?;
            for _ in 0..point_count {
                points_3d.push(next3_f64(&mut cur, "point3d")?);
            }
            loops.push(ContourLoop {
                points_3d,
                points_2d,
                closed,
                perimeter,
                enclosed_area,
                centroid_2d,
                centroid_3d,
                min_caliper_diameter,
                max_caliper_diameter,
            });
        }
        slices.push(ContourSlice {
            plane_origin,
            plane_normal,
            u_axis,
            v_axis,
            loops,
            total_perimeter,
            total_area,
        });
    }
    if cur != flat.len() {
        return Err(Error::invalid(
            "contour slice buffer contains trailing data",
        ));
    }
    Ok(slices)
}

/// Exports packed contour slices as SVG (`0`), JSON (`1`), or CSV (`2`).
#[wasm_bindgen(js_name = sliceContoursExport)]
pub fn slice_contours_export(packed_slices: &[f64], format: u8) -> Result<String, Error> {
    let slices = decode_contour_slices(packed_slices)?;
    match format {
        0 => Ok(contours_to_svg(&slices)),
        1 => Ok(contours_to_json(&slices)),
        2 => Ok(contours_to_csv(&slices)),
        _ => Err(Error::invalid(
            "format must be 0 (svg), 1 (json) or 2 (csv)",
        )),
    }
}

pub(crate) fn encode_point_cloud_f32(cloud: &PointCloud) -> Result<Vec<f32>, Error> {
    let n = cloud.positions.len();
    let has_n = cloud.normals.as_ref().is_some_and(|v| v.len() == n);
    let has_c = cloud.colors.as_ref().is_some_and(|v| v.len() == n);
    let total = 3 + n * 3 + if has_n { n * 3 } else { 0 } + if has_c { n * 3 } else { 0 };
    let mut out = Vec::new();
    out.try_reserve(total).map_err(|_| oom())?;
    out.push(n as f32);
    out.push(if has_n { 1.0 } else { 0.0 });
    out.push(if has_c { 1.0 } else { 0.0 });
    for p in &cloud.positions {
        out.extend_from_slice(p);
    }
    if has_n && let Some(norms) = &cloud.normals {
        for nr in norms {
            out.extend_from_slice(nr);
        }
    }
    if has_c && let Some(cols) = &cloud.colors {
        for c in cols {
            out.extend_from_slice(c);
        }
    }
    Ok(out)
}

pub(crate) fn decode_point_cloud_f32(packed: &[f32]) -> Result<PointCloud, Error> {
    if packed.len() < 3 {
        return Err(Error::invalid("point cloud header must have 3 values"));
    }
    let count_f = f64::from(packed[0]);
    let n = req_count(count_f, "pointCount")?;
    let has_n = flag01(f64::from(packed[1]), "hasNormals")?;
    let has_c = flag01(f64::from(packed[2]), "hasColors")?;
    let expected = 3usize
        .checked_add(n.checked_mul(3).ok_or_else(oom)?)
        .and_then(|s| if has_n { s.checked_add(n * 3) } else { Some(s) })
        .and_then(|s| if has_c { s.checked_add(n * 3) } else { Some(s) })
        .ok_or_else(oom)?;
    if packed.len() != expected {
        return Err(Error::invalid("point cloud buffer length mismatch"));
    }
    let mut cur = 3usize;
    let mut positions = Vec::new();
    positions.try_reserve(n).map_err(|_| oom())?;
    for _ in 0..n {
        positions.push([packed[cur], packed[cur + 1], packed[cur + 2]]);
        cur += 3;
    }
    let normals = if has_n {
        let mut v = Vec::new();
        v.try_reserve(n).map_err(|_| oom())?;
        for _ in 0..n {
            v.push([packed[cur], packed[cur + 1], packed[cur + 2]]);
            cur += 3;
        }
        Some(v)
    } else {
        None
    };
    let colors = if has_c {
        let mut v = Vec::new();
        v.try_reserve(n).map_err(|_| oom())?;
        for _ in 0..n {
            v.push([packed[cur], packed[cur + 1], packed[cur + 2]]);
            cur += 3;
        }
        Some(v)
    } else {
        None
    };
    Ok(PointCloud {
        positions,
        normals,
        colors,
    })
}

fn encode_outlier_and_cloud(rep: &OutlierReport, cloud: &PointCloud) -> Result<Vec<f64>, Error> {
    let packed = encode_point_cloud_f32(cloud)?;
    let mut out = Vec::new();
    out.try_reserve(5 + packed.len()).map_err(|_| oom())?;
    out.extend_from_slice(&[
        count(rep.kept_count),
        count(rep.removed_count),
        rep.mean_distance,
        rep.std_dev,
        rep.threshold,
    ]);
    for v in packed {
        out.push(f64::from(v));
    }
    Ok(out)
}

/// Parses an ASCII point cloud (`0 = xyz`, `1 = pcd`, `2 = ply`) into a packed `Float32Array`.
#[wasm_bindgen(js_name = parsePointCloudText)]
pub fn parse_point_cloud_text(text: &str, format: u8) -> Result<Vec<f32>, Error> {
    let cloud = match format {
        0 => parse_xyz(text)?,
        1 => parse_pcd_ascii(text)?,
        2 => parse_ply_points(text.as_bytes())?,
        _ => {
            return Err(Error::invalid(
                "point cloud text format must be 0 (xyz), 1 (pcd) or 2 (ply)",
            ));
        }
    };
    encode_point_cloud_f32(&cloud)
}

/// Parses binary/ASCII `.ply` point cloud bytes into a packed `Float32Array`.
#[wasm_bindgen(js_name = parsePointCloudBytes)]
pub fn parse_point_cloud_bytes(bytes: &[u8]) -> Result<Vec<f32>, Error> {
    let cloud = parse_ply_points(bytes)?;
    encode_point_cloud_f32(&cloud)
}

/// Runs Statistical Outlier Removal on a packed point cloud and returns
/// `[keptCount, removedCount, meanDistance, stdDev, threshold, ...packedCloud]`.
#[wasm_bindgen(js_name = pointCloudSor)]
pub fn point_cloud_sor(
    packed: &[f32],
    k_neighbors: f64,
    std_ratio: f64,
) -> Result<Vec<f64>, Error> {
    let cloud = decode_point_cloud_f32(packed)?;
    let k = req_count(k_neighbors, "kNeighbors")?;
    let ratio = f64_to_f32(std_ratio, "stdRatio")?;
    let (filtered, report) = statistical_outlier_removal(&cloud, k, ratio)?;
    encode_outlier_and_cloud(&report, &filtered)
}

/// Runs Radius Outlier Removal on a packed point cloud and returns
/// `[keptCount, removedCount, meanDistance, stdDev, threshold, ...packedCloud]`.
#[wasm_bindgen(js_name = pointCloudRor)]
pub fn point_cloud_ror(packed: &[f32], radius: f64, min_neighbors: f64) -> Result<Vec<f64>, Error> {
    let cloud = decode_point_cloud_f32(packed)?;
    let r = f64_to_f32(radius, "radius")?;
    let min_n = req_count(min_neighbors, "minNeighbors")?;
    let (filtered, report) = radius_outlier_removal(&cloud, r, min_n)?;
    encode_outlier_and_cloud(&report, &filtered)
}

/// Voxel-grid downsamples a packed point cloud into a packed `Float32Array`.
#[wasm_bindgen(js_name = pointCloudVoxelDownsample)]
pub fn point_cloud_voxel_downsample(packed: &[f32], voxel_size: f64) -> Result<Vec<f32>, Error> {
    let cloud = decode_point_cloud_f32(packed)?;
    let vs = f64_to_f32(voxel_size, "voxelSize")?;
    let out = voxel_downsample(&cloud, vs)?;
    encode_point_cloud_f32(&out)
}

/// Estimates and consistently orients per-point normals on a packed point cloud.
#[wasm_bindgen(js_name = pointCloudEstimateNormals)]
pub fn point_cloud_estimate_normals(
    packed: &[f32],
    k_neighbors: f64,
    camera_hint: Option<Vec<f64>>,
) -> Result<Vec<f32>, Error> {
    let mut cloud = decode_point_cloud_f32(packed)?;
    let k = req_count(k_neighbors, "kNeighbors")?;
    let hint = match camera_hint {
        None => None,
        Some(h) => Some(vec3_f32(&h, "cameraHint")?),
    };
    estimate_normals(&mut cloud, k, hint)?;
    encode_point_cloud_f32(&cloud)
}

pub(crate) fn parse_reconstruct_options(
    method: u8,
    voxel_size: f64,
    ball_radius: f64,
    k_neighbors: f64,
) -> Result<SurfaceReconstructOptions, Error> {
    let defaults = SurfaceReconstructOptions::default();
    let method = match method {
        0 => ReconstructMethod::ImplicitMlsMarchingCubes,
        1 => ReconstructMethod::BallPivoting,
        _ => {
            return Err(Error::invalid(
                "reconstruct method must be 0 (implicitMlsMarchingCubes) or 1 (ballPivoting)",
            ));
        }
    };
    let voxel_size = if voxel_size.is_nan() {
        defaults.voxel_size
    } else {
        f64_to_f32(voxel_size, "voxelSize")?
    };
    let ball_radius = if ball_radius.is_nan() {
        defaults.ball_radius
    } else {
        f64_to_f32(ball_radius, "ballRadius")?
    };
    let k_neighbors = count_opt(k_neighbors, "kNeighbors")?.unwrap_or(defaults.k_neighbors);
    Ok(SurfaceReconstructOptions {
        method,
        voxel_size,
        ball_radius,
        k_neighbors,
    })
}

// ---------------------------------------------------------------------------
// #236 Hollow, Drain Holes & Place Mesh Helpers
// ---------------------------------------------------------------------------

#[allow(clippy::too_many_arguments)]
pub(crate) fn parse_hollow_options(
    wall_thickness: f64,
    backend: u8,
    voxel_size: f64,
    max_voxels: f64,
    max_cavity_triangles: f64,
    min_cavity_volume: f64,
    modulation_verts: Option<Vec<u32>>,
    modulation_weights: Option<Vec<f32>>,
) -> Result<HollowOptions, Error> {
    let defaults = HollowOptions::default();
    let backend = match backend {
        0 => HollowBackend::BvhSdf,
        1 => HollowBackend::FhEdt,
        _ => return Err(Error::invalid("backend must be 0 (bvhSdf) or 1 (fhEdt)")),
    };
    let voxel_size = if voxel_size.is_nan() {
        defaults.voxel_size
    } else {
        Some(f64_to_f32(voxel_size, "voxelSize")?)
    };
    let max_voxels = count_opt(max_voxels, "maxVoxels")?.unwrap_or(defaults.max_voxels);
    let max_cavity_triangles = match count_opt(max_cavity_triangles, "maxCavityTriangles")? {
        None => None,
        Some(n) => Some(
            u32::try_from(n).map_err(|_| Error::invalid("maxCavityTriangles exceeds u32 range"))?,
        ),
    };
    let min_cavity_volume = if min_cavity_volume.is_nan() {
        defaults.min_cavity_volume
    } else if !min_cavity_volume.is_finite() {
        return Err(Error::invalid("minCavityVolume must be finite"));
    } else {
        min_cavity_volume
    };
    let wall_modulation_weights = match (modulation_verts, modulation_weights) {
        (None, None) => None,
        (Some(verts), Some(weights)) => {
            if verts.len() != weights.len() {
                return Err(Error::invalid(
                    "wallModulation vertices and weights must match in length",
                ));
            }
            let mut pairs = Vec::new();
            pairs.try_reserve(verts.len()).map_err(|_| oom())?;
            for (&v, &w) in verts.iter().zip(&weights) {
                pairs.push((v, w));
            }
            Some(pairs)
        }
        _ => {
            return Err(Error::invalid(
                "wallModulation requires both vertices and weights",
            ));
        }
    };
    Ok(HollowOptions {
        wall_thickness: f64_to_f32(wall_thickness, "wallThickness")?,
        backend,
        voxel_size,
        max_voxels,
        max_cavity_triangles,
        min_cavity_volume,
        wall_modulation_weights,
    })
}

pub(crate) fn encode_hollow_report(r: &HollowReport) -> Vec<f64> {
    vec![
        count(r.cavities_created),
        count(r.cavity_vertices),
        count(r.cavity_triangles),
        r.cavity_volume,
        f64::from(r.voxel_size),
        if r.uvs_dropped { 1.0 } else { 0.0 },
    ]
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn parse_drain_hole(
    surface_point3: &[f64],
    surface_normal3: &[f64],
    diameter: f64,
    depth: f64,
    outer_chamfer: f64,
    taper_angle_deg: f64,
    segments: u32,
    boolean_method: u8,
    fallback_bvh_sdf: bool,
    generate_plug: bool,
    plug_clearance: f64,
) -> Result<([f64; 3], DrainHoleSpec, DrainHoleOptions), Error> {
    let point = vec3_f64(surface_point3, "surfacePoint")?;
    let spec = DrainHoleSpec {
        surface_point: vec3_f32(surface_point3, "surfacePoint")?,
        surface_normal: vec3_f32(surface_normal3, "surfaceNormal")?,
        diameter: f64_to_f32(diameter, "diameter")?,
        depth: if depth.is_nan() {
            None
        } else {
            Some(f64_to_f32(depth, "depth")?)
        },
        outer_chamfer: f64_to_f32(outer_chamfer, "outerChamfer")?,
        taper_angle_deg: f64_to_f32(taper_angle_deg, "taperAngleDeg")?,
        segments,
    };
    let opts = DrainHoleOptions {
        boolean_method: parse_boolean_method(boolean_method)?,
        fallback_bvh_sdf,
        generate_plug,
        plug_clearance: f64_to_f32(plug_clearance, "plugClearance")?,
    };
    Ok((point, spec, opts))
}

pub(crate) fn encode_drain_report(
    report: &DrainHoleReport,
    plug_ids: &[u32],
) -> Result<Vec<f64>, Error> {
    let n_plugs = plug_ids.len() / 2;
    let mut out = Vec::new();
    out.try_reserve(6 + plug_ids.len()).map_err(|_| oom())?;
    out.extend_from_slice(&[
        count(report.holes_punched),
        count(report.kept_triangles),
        if report.used_fallback { 1.0 } else { 0.0 },
        count(n_plugs),
    ]);
    for &id_part in plug_ids {
        out.push(f64::from(id_part));
    }
    out.push(boolean_method_code(report.method_used));
    out.push(if report.uvs_dropped { 1.0 } else { 0.0 });
    Ok(out)
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn parse_place_params(
    hit_point3: &[f64],
    hit_normal3: &[f64],
    up_axis: u8,
    twist_angle_rad: f64,
    uniform_scale: f64,
    embed_depth: f64,
    mode: u8,
    conform_to_surface: bool,
    boolean_method: u8,
) -> Result<([f64; 3], [f64; 3], PlaceMeshParams), Error> {
    let hp_world = vec3_f64(hit_point3, "hitPoint")?;
    let hn_world = vec3_f64(hit_normal3, "hitNormal")?;
    let up_axis = match up_axis {
        0 => PlaceUpAxis::NegZ,
        1 => PlaceUpAxis::PosZ,
        2 => PlaceUpAxis::PosY,
        3 => PlaceUpAxis::NegY,
        _ => {
            return Err(Error::invalid(
                "upAxis must be 0 (negZ), 1 (posZ), 2 (posY) or 3 (negY)",
            ));
        }
    };
    let mode = match mode {
        0 => PlaceMode::KeepSeparate,
        1 => PlaceMode::AppendShell,
        2 => PlaceMode::EmbedUnion,
        3 => PlaceMode::EngraveSubtract,
        _ => {
            return Err(Error::invalid(
                "mode must be 0 (keepSeparate), 1 (appendShell), 2 (embedUnion) or 3 (engraveSubtract)",
            ));
        }
    };
    let params = PlaceMeshParams {
        hit_point: vec3_f32(hit_point3, "hitPoint")?,
        hit_normal: vec3_f32(hit_normal3, "hitNormal")?,
        up_axis,
        twist_angle_rad: f64_to_f32(twist_angle_rad, "twistAngleRad")?,
        uniform_scale: f64_to_f32(uniform_scale, "uniformScale")?,
        embed_depth: f64_to_f32(embed_depth, "embedDepth")?,
        mode,
        conform_to_surface,
        boolean_method: parse_boolean_method(boolean_method)?,
    };
    Ok((hp_world, hn_world, params))
}

pub(crate) fn encode_place_report(
    report: &PlaceMeshReport,
    added_pair: Option<[u32; 2]>,
) -> Vec<f64> {
    let mut out = Vec::with_capacity(19);
    out.extend_from_slice(report.transform.rigid.rotation.as_flattened());
    out.extend_from_slice(&report.transform.rigid.translation);
    out.push(report.transform.scale);
    out.push(count(report.kept_triangles));
    out.push(0.0); // usedFallback01
    out.push(0.0); // conformMaxDisp
    match added_pair {
        Some([idx, gen_id]) => {
            out.push(f64::from(idx));
            out.push(f64::from(gen_id));
        }
        None => {
            out.push(-1.0);
            out.push(0.0);
        }
    }
    out.push(if report.uvs_dropped { 1.0 } else { 0.0 });
    out
}

pub(crate) fn check_live_mesh(scene: &Scene, id: MeshId) -> Result<(), Error> {
    if scene.mesh(id).is_err() {
        return Err(SceneError::UnknownMesh.into());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scene::WasmScene;

    fn unit_box_at(cx: f64, cy: f64, cz: f64, half: f64) -> (Vec<f64>, Vec<u32>) {
        let p = [
            cx - half,
            cy - half,
            cz - half,
            cx + half,
            cy - half,
            cz - half,
            cx + half,
            cy + half,
            cz - half,
            cx - half,
            cy + half,
            cz - half,
            cx - half,
            cy - half,
            cz + half,
            cx + half,
            cy - half,
            cz + half,
            cx + half,
            cy + half,
            cz + half,
            cx - half,
            cy + half,
            cz + half,
        ];
        let i = [
            0, 2, 1, 0, 3, 2, // -z
            4, 5, 6, 4, 6, 7, // +z
            0, 1, 5, 0, 5, 4, // -y
            3, 7, 6, 3, 6, 2, // +y
            0, 4, 7, 0, 7, 3, // -x
            1, 2, 6, 1, 6, 5, // +x
        ];
        (p.to_vec(), i.to_vec())
    }

    fn grid_patch(n: usize) -> (Vec<f64>, Vec<u32>) {
        let mut p = Vec::new();
        for y in 0..=n {
            for x in 0..=n {
                p.extend_from_slice(&[x as f64 / n as f64, y as f64 / n as f64, 0.0]);
            }
        }
        let mut idx = Vec::new();
        let stride = (n + 1) as u32;
        for y in 0..n as u32 {
            for x in 0..n as u32 {
                let a = y * stride + x;
                idx.extend_from_slice(&[a, a + 1, a + stride + 1, a, a + stride + 1, a + stride]);
            }
        }
        (p, idx)
    }

    #[test]
    fn wave6_all_domains_roundtrip() {
        let mut ws = WasmScene::new();

        // 1. #233 Selection + split/extract/delete
        let (bp, bi) = unit_box_at(0.0, 0.0, 0.0, 1.0);
        let box_id = ws.add_mesh(&bp, &bi, None, 0, true).unwrap();
        let vp = [
            1.0, 0.0, 0.0, 0.0, // col 0
            0.0, 1.0, 0.0, 0.0, // col 1
            0.0, 0.0, 0.5, 0.0, // col 2
            0.0, 0.0, 0.0, 1.0, // col 3
        ];
        let scr = ws
            .select_faces_screen(
                box_id[0],
                box_id[1],
                0,
                &[-1.0, -1.0, 1.0, 1.0],
                &vp,
                &[0.0, 0.0, 5.0],
                &[0.0, 0.0, -1.0],
                true,
                true,
                false,
                true,
            )
            .unwrap();
        assert!(!scr.is_empty());
        let sph = ws
            .select_faces_in_sphere(box_id[0], box_id[1], &[0.0, 0.0, 1.0], 1.5, None, 0.0)
            .unwrap();
        assert!(!sph.is_empty());
        let shell = ws.select_connected_shell(box_id[0], box_id[1], 0).unwrap();
        assert_eq!(shell.len(), 12);
        let crease = ws
            .select_by_crease_angle(box_id[0], box_id[1], 0, 0.2)
            .unwrap();
        assert_eq!(crease.len(), 2);
        let grown = ws
            .grow_face_selection(box_id[0], box_id[1], &crease, 1.0, 0)
            .unwrap();
        assert!(grown.len() > crease.len());
        let shrunk = ws
            .shrink_face_selection(box_id[0], box_id[1], &grown, 1.0, 0)
            .unwrap();
        assert!(shrunk.len() <= grown.len());
        let mask = ws
            .face_selection_to_vertex_mask(box_id[0], box_id[1], &crease)
            .unwrap();
        let from_mask = ws
            .vertex_mask_to_face_selection(box_id[0], box_id[1], &mask, 0.5, true)
            .unwrap();
        assert_eq!(from_mask, crease);
        let extracted = ws
            .extract_selected_faces(box_id[0], box_id[1], &crease, true)
            .unwrap();
        assert_eq!(extracted.len(), 2);
        ws.remove_mesh(extracted[0], extracted[1]).unwrap();
        let split_res = ws
            .split_selected_faces(box_id[0], box_id[1], &crease, false)
            .unwrap();
        assert_eq!(split_res[2], 2);
        ws.undo().unwrap();

        // 2. #193 Deform: FFD + ARAP + Kelvinlet + XPBD
        let (gp, gi) = grid_patch(4);
        let gid = ws.add_mesh(&gp, &gi, None, 0, true).unwrap();
        let lat = ws
            .build_ffd_lattice(gid[0], gid[1], &[2, 2, 2], 0, 0.1)
            .unwrap();
        assert_eq!(lat.len(), 19 + 8 * 3);
        let mut offsets = lat[19..].to_vec();
        if let Some(last) = offsets.last_mut() {
            *last = 0.25;
        }
        let ffd_rep = ws
            .apply_ffd(
                gid[0],
                gid[1],
                &[2, 2, 2],
                0,
                &lat[4..19],
                &offsets,
                0,
                None,
                0.0,
                None,
                None,
                None,
            )
            .unwrap();
        assert!(ffd_rep[1] > 0.0);
        let arap_rep = ws
            .apply_arap(
                gid[0],
                gid[1],
                &[0, 4],
                &[0.0; 6],
                &[0, 0],
                &[24],
                &[1.0, 1.0, 0.4],
                0,
                0,
                4.0,
                1e-4,
                None,
                0.0,
                None,
                Some(vec![0]),
                Some(vec![0.0]),
            )
            .unwrap();
        assert!(arap_rep[1] > 0.0);
        let kel_rep = ws
            .apply_kelvinlet(
                gid[0],
                gid[1],
                &[0.5, 0.5, 0.0],
                0,
                &[0.0, 0.0, 0.2],
                0.5,
                0.3,
                1.0,
                0,
                2.0,
                false,
                false,
                false,
                None,
                0.0,
                None,
                None,
                None,
            )
            .unwrap();
        assert!(kel_rep[1] > 0.0);
        let xpbd_rep = ws
            .apply_xpbd(
                gid[0],
                gid[1],
                2.0,
                4.0,
                1.0 / 60.0,
                0.0,
                0.05,
                f64::NAN,
                0.02,
                &[0.0, 0.0, -1.0],
                &[0],
                &[0.0, 0.0, 0.0],
                &[24],
                &[1.0, 1.0, 0.3],
                &[0.0],
                false,
                0.01,
                None,
                None,
            )
            .unwrap();
        assert!(xpbd_rep[1] > 0.0);

        // 3. #194 Inter-Object Collision
        let (bp2, bi2) = unit_box_at(3.0, 0.0, 0.0, 1.0);
        let mover = ws.add_mesh(&bp2, &bi2, None, 0, true).unwrap();
        let mut rule_arr = ws.default_collision_rule();
        assert_eq!(rule_arr.len(), 16);
        rule_arr[0] = 1.0; // Rigid
        ws.set_collision_rule(box_id[0], box_id[1], mover[0], mover[1], &rule_arr)
            .unwrap();
        assert!(ws.has_active_collision_response(mover[0], mover[1]));
        let step_rep = ws
            .move_with_collision(
                mover[0],
                mover[1],
                &[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
                &[-3.0, 0.0, 0.0],
                None,
                false,
            )
            .unwrap();
        assert!(step_rep[0] > -2.9);
        assert!(
            ws.clear_collision_rule(box_id[0], box_id[1], mover[0], mover[1])
                .unwrap()
        );
        assert!(!ws.has_active_collision_response(mover[0], mover[1]));

        // 4. #234 Primitive Fit + Contours + Point Cloud
        let fit = ws
            .fit_primitive(
                box_id[0],
                box_id[1],
                None,
                0,
                f64::NAN,
                64.0,
                20.0,
                1.0,
                f64::NAN,
                f64::NAN,
            )
            .unwrap();
        assert_eq!(fit[0], 0.0);
        let aligned = ws
            .align_to_fitted_primitive(
                box_id[0],
                box_id[1],
                &fit[..9],
                &[0.0, 0.0, 1.0],
                &[0.0, 0.0, 0.0],
            )
            .unwrap();
        assert_eq!(aligned.len(), 15);
        ws.undo().unwrap();

        let slices = ws
            .slice_contour_stack(
                box_id[0],
                box_id[1],
                &[0.0, 0.0, -0.5],
                &[0.0, 0.0, 1.0],
                0.5,
                2.0,
            )
            .unwrap();
        let svg = slice_contours_export(&slices, 0).unwrap();
        assert!(svg.contains("<svg"));
        let json = slice_contours_export(&slices, 1).unwrap();
        assert!(json.contains("\"slices\""));
        let csv = slice_contours_export(&slices, 2).unwrap();
        assert!(csv.contains("slice_index"));

        let xyz = "0 0 0\n1 0 0\n0 1 0\n0 0 1\n1 1 0\n1 0 1\n0 1 1\n1 1 1\n";
        let pc = parse_point_cloud_text(xyz, 0).unwrap();
        let pc_n = point_cloud_estimate_normals(&pc, 4.0, Some(vec![0.5, 0.5, 5.0])).unwrap();
        assert_eq!(pc_n[1], 1.0);
        let sor = point_cloud_sor(&pc_n, 3.0, 2.0).unwrap();
        assert_eq!(sor[0], 8.0);
        let recon = ws
            .reconstruct_point_cloud(&pc_n, 0, 0.4, f64::NAN, 4.0, true)
            .unwrap();
        assert_eq!(recon.len(), 2);

        // 5. #236 Hollow, Drain Holes & Place Mesh
        let (sb_p, sb_i) = unit_box_at(10.0, 0.0, 0.0, 2.0);
        let solid = ws.add_mesh(&sb_p, &sb_i, None, 0, true).unwrap();
        let hol = ws
            .hollow_mesh(
                solid[0],
                solid[1],
                0.35,
                0,
                0.25,
                65536.0,
                f64::NAN,
                0.0,
                None,
                None,
            )
            .unwrap();
        assert_eq!(hol[0], 1.0);
        let dr = ws
            .punch_drain_hole(
                solid[0],
                solid[1],
                &[10.0, 0.0, 2.0],
                &[0.0, 0.0, 1.0],
                0.8,
                f64::NAN,
                0.0,
                0.0,
                16,
                0,
                true,
                true,
                0.02,
                true,
            )
            .unwrap();
        assert_eq!(dr[0], 1.0);
        assert_eq!(dr[3], 1.0);
        let (st_p, st_i) = unit_box_at(0.0, 0.0, 0.0, 0.4);
        let stamp = ws.add_mesh(&st_p, &st_i, None, 0, true).unwrap();
        let pl = ws
            .place_mesh(
                solid[0],
                solid[1],
                stamp[0],
                stamp[1],
                &[12.0, 0.0, 0.0],
                &[1.0, 0.0, 0.0],
                1,
                0.0,
                1.0,
                0.2,
                0,
                false,
                0,
                true,
            )
            .unwrap();
        assert!(pl[16] >= 0.0);
    }
}
