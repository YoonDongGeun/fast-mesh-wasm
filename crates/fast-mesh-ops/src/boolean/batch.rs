//! N-ary CSG tree batch evaluation via [`CsgNode`] and [`boolean_batch`].

use fast_mesh_bvh::{BuildOptions, Bvh};
use fast_mesh_fields::{DEFAULT_MAX_VOXELS, FieldsError, GridSpec};
use fast_mesh_query::Rigid;
use fast_mesh_store::{Mesh, StoreError};

use super::voxel::{
    choose_boolean_grid_spec, combine_grids_in_place, eval_signed_grid, extract_isosurface,
    live_bounds, place_mesh_compact,
};
use super::{
    BooleanMethod, BooleanOp, BooleanOptions, boolean, cap_mesh_triangles, check_input, map_tool,
    validate_options,
};
use crate::{OpsError, Result, filled, reserve};

/// Binary CSG expression tree referencing operands in a slice by leaf index.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CsgNode {
    /// Leaf operand referencing `meshes[index]`.
    Leaf(usize),
    /// Binary CSG operation combining `left` and `right` subtrees.
    Op {
        op: BooleanOp,
        left: Box<CsgNode>,
        right: Box<CsgNode>,
    },
}

impl CsgNode {
    /// Builds a height-balanced binary CSG reduction tree over `0..leaf_count`
    /// combining leaves with `op`.
    ///
    /// Returns `None` when `leaf_count == 0`, `Some(Leaf(0))` when `leaf_count == 1`.
    pub fn reduce_all(op: BooleanOp, leaf_count: usize) -> Option<CsgNode> {
        if leaf_count == 0 {
            return None;
        }
        Some(reduce_range(op, 0, leaf_count))
    }
}

fn reduce_range(op: BooleanOp, start: usize, len: usize) -> CsgNode {
    debug_assert!(len > 0);
    if len == 1 {
        return CsgNode::Leaf(start);
    }
    let mid = len / 2;
    CsgNode::Op {
        op,
        left: Box::new(reduce_range(op, start, mid)),
        right: Box::new(reduce_range(op, start + mid, len - mid)),
    }
}

fn collect_leaf_refs(node: &CsgNode, mesh_count: usize, ref_counts: &mut [usize]) -> Result<()> {
    match node {
        CsgNode::Leaf(idx) => {
            if *idx >= mesh_count {
                return Err(OpsError::InvalidInput("CSG tree leaf index out of bounds"));
            }
            ref_counts[*idx] = ref_counts[*idx]
                .checked_add(1)
                .ok_or(OpsError::Store(StoreError::OutOfMemory))?;
            Ok(())
        }
        CsgNode::Op { left, right, .. } => {
            collect_leaf_refs(left, mesh_count, ref_counts)?;
            collect_leaf_refs(right, mesh_count, ref_counts)
        }
    }
}

fn eval_grid_node(
    node: &CsgNode,
    ref_counts: &mut [usize],
    leaf_grids: &mut [Option<Vec<f32>>],
) -> Result<Vec<f32>> {
    match node {
        CsgNode::Leaf(idx) => {
            ref_counts[*idx] -= 1;
            if ref_counts[*idx] == 0 {
                Ok(leaf_grids[*idx]
                    .take()
                    .expect("leaf grid must exist for final reference"))
            } else {
                let src = leaf_grids[*idx]
                    .as_ref()
                    .expect("leaf grid must exist for shared reference");
                let mut copy = Vec::new();
                reserve(&mut copy, src.len())?;
                copy.extend_from_slice(src);
                Ok(copy)
            }
        }
        CsgNode::Op { op, left, right } => {
            let mut lhs = eval_grid_node(left, ref_counts, leaf_grids)?;
            let rhs = eval_grid_node(right, ref_counts, leaf_grids)?;
            combine_grids_in_place(&mut lhs, &rhs, *op);
            Ok(lhs)
        }
    }
}

fn eval_volumetric_batch(
    meshes: &[(&Mesh, Rigid)],
    tree: &CsgNode,
    ref_counts: &mut [usize],
    options: &BooleanOptions,
) -> Result<Mesh> {
    let mut union_bounds: Option<([f32; 3], [f32; 3])> = None;
    let mut placed_leaves: Vec<Option<Mesh>> = Vec::new();
    reserve(&mut placed_leaves, meshes.len())?;

    for (idx, (mesh, xf)) in meshes.iter().enumerate() {
        if ref_counts[idx] == 0 {
            placed_leaves.push(None);
            continue;
        }
        let placed = place_mesh_compact(mesh, xf)?;
        if let Some((bmin, bmax)) = live_bounds(&placed) {
            union_bounds = Some(match union_bounds {
                None => (bmin, bmax),
                Some((umin, umax)) => (
                    [
                        umin[0].min(bmin[0]),
                        umin[1].min(bmin[1]),
                        umin[2].min(bmin[2]),
                    ],
                    [
                        umax[0].max(bmax[0]),
                        umax[1].max(bmax[1]),
                        umax[2].max(bmax[2]),
                    ],
                ),
            });
        }
        placed_leaves.push(Some(placed));
    }

    let Some((min, max)) = union_bounds else {
        return Ok(Mesh::new());
    };

    let spec: GridSpec = choose_boolean_grid_spec(min, max, options.voxel_pitch)?;
    let total = spec.validate(DEFAULT_MAX_VOXELS)?;
    if total == 0 {
        return Err(FieldsError::InvalidInput("empty voxel grid").into());
    }

    let mut leaf_grids: Vec<Option<Vec<f32>>> = Vec::new();
    reserve(&mut leaf_grids, meshes.len())?;
    for slot in &mut placed_leaves {
        if let Some(mut placed) = slot.take() {
            let grid = eval_signed_grid(&mut placed, spec, options.method)?;
            leaf_grids.push(Some(grid.into_values()));
        } else {
            leaf_grids.push(None);
        }
    }
    drop(placed_leaves);

    let root_values = eval_grid_node(tree, ref_counts, &mut leaf_grids)?;
    drop(leaf_grids);

    let mut out = extract_isosurface(spec, root_values)?;
    if let Some(cap) = options.max_triangles {
        cap_mesh_triangles(&mut out, cap)?;
    }
    Ok(out)
}

fn eval_manifold_node(
    meshes: &[(&Mesh, Rigid)],
    node: &CsgNode,
    options: &BooleanOptions,
) -> Result<Mesh> {
    match node {
        CsgNode::Leaf(idx) => {
            let (mesh, xf) = &meshes[*idx];
            map_tool(mesh, xf)
        }
        CsgNode::Op { op, left, right } => {
            let lhs = eval_manifold_node(meshes, left, options)?;
            let rhs = eval_manifold_node(meshes, right, options)?;
            let step_opts = BooleanOptions {
                op: *op,
                method: BooleanMethod::Manifold,
                coplanar_tolerance: options.coplanar_tolerance,
                snap_slivers: options.snap_slivers,
                voxel_pitch: options.voxel_pitch,
                max_triangles: None,
            };
            let (combined, _) = boolean(&lhs, &rhs, Rigid::IDENTITY, &step_opts)?;
            Ok(combined)
        }
    }
}

/// Evaluates an arbitrary N-ary CSG [`CsgNode`] expression over `meshes`.
///
/// For volumetric modes (`BvhSdf`, `FhEdt`), constructs a single shared voxel grid
/// over the combined bounds of referenced leaves, caches leaf SDF evaluations across
/// repeated references, folds the CSG tree directly on `f32` grids, and extracts a
/// single watertight Marching Cubes surface at the root.
///
/// For `Manifold`, recursively evaluates binary operations in world coordinates with
/// optional post-step sliver snapping (`options.snap_slivers`), then applies
/// `options.max_triangles` once on the final mesh.
pub fn boolean_batch(
    meshes: &[(&Mesh, Rigid)],
    tree: &CsgNode,
    options: &BooleanOptions,
) -> Result<Mesh> {
    validate_options(options)?;
    for (_, xf) in meshes {
        xf.validate()?;
    }

    let mut ref_counts = filled(meshes.len(), 0usize)?;
    collect_leaf_refs(tree, meshes.len(), &mut ref_counts)?;

    match options.method {
        BooleanMethod::BvhSdf | BooleanMethod::FhEdt => {
            eval_volumetric_batch(meshes, tree, &mut ref_counts, options)
        }
        BooleanMethod::Manifold => {
            let mut out = eval_manifold_node(meshes, tree, options)?;
            if matches!(tree, CsgNode::Leaf(_)) {
                let bvh = Bvh::build(&out, &BuildOptions::default())?;
                check_input(&out, &bvh)?;
            }
            if let Some(cap) = options.max_triangles {
                cap_mesh_triangles(&mut out, cap)?;
            }
            Ok(out)
        }
    }
}
