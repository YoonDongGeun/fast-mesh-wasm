#![no_main]
//! Fuzzes [`boolean_plan`](fast_mesh_ops::boolean_plan) plus
//! [`apply_boolean`](fast_mesh_ops::apply_boolean). Random soup mostly
//! exercises input rejection; pairs of boxes (closed and manifold when their
//! corners are ordered) reach the split and classify paths.

#[path = "common.rs"]
mod common;

use arbitrary::{Arbitrary, Unstructured};
use common::{FuzzBox, FuzzBuffers};
use fast_mesh_bvh::{BuildOptions, Bvh};
use fast_mesh_ops::{apply_boolean, boolean_plan, BooleanOp};
use fast_mesh_query::Rigid;
use libfuzzer_sys::fuzz_target;

#[derive(Debug)]
struct BooleanInput {
    a: FuzzBuffers,
    b: FuzzBuffers,
    box_a: FuzzBox,
    box_b: FuzzBox,
    use_boxes: bool,
    op: u8,
    translation: [f64; 3],
}

impl<'a> Arbitrary<'a> for BooleanInput {
    fn arbitrary(u: &mut Unstructured<'a>) -> arbitrary::Result<Self> {
        let (a, b) = (FuzzBuffers::arbitrary(u)?, FuzzBuffers::arbitrary(u)?);
        let (box_a, box_b) = (FuzzBox::arbitrary(u)?, FuzzBox::arbitrary(u)?);
        let (use_boxes, op) = (bool::arbitrary(u)?, u8::arbitrary(u)?);
        let mut translation = [0.0; 3];
        for c in &mut translation {
            *c = f64::from(i16::arbitrary(u)?) * 0.01;
        }
        Ok(BooleanInput {
            a,
            b,
            box_a,
            box_b,
            use_boxes,
            op,
            translation,
        })
    }
}

fuzz_target!(|input: BooleanInput| {
    let (mut a, b) = if input.use_boxes {
        (input.box_a.buffers.mesh(), input.box_b.buffers.mesh())
    } else {
        (input.a.mesh(), input.b.mesh())
    };
    let options = BuildOptions::default();
    let (Ok(bvh_a), Ok(bvh_b)) = (Bvh::build(&a, &options), Bvh::build(&b, &options)) else {
        return;
    };
    let xf = Rigid {
        translation: input.translation,
        ..Rigid::IDENTITY
    };
    let op = match input.op % 3 {
        0 => BooleanOp::Union,
        1 => BooleanOp::Subtract,
        _ => BooleanOp::Intersect,
    };
    let Ok(plan) = boolean_plan(&a, &bvh_a, &b, &bvh_b, &xf, op) else {
        return;
    };
    let mut tx = a.begin_edit();
    let _ = apply_boolean(&mut tx, &plan);
    let _ = tx.commit();
});
