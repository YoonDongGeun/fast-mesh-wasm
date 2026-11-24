//! `WasmScene.generatePrimitive`: the shapes of
//! [`fast_mesh_ops::primitives`] from a kind code and a parameter list.
//!
//! A static of `WasmScene` rather than a free function: a Worker that runs
//! the threaded glue makes its scenes from that glue's `WasmScene`
//! (`useWasmScene`), and so reaches this through the same class, while free
//! functions would name the single-threaded glue it never instantiated.
//!
//! # Encoding
//!
//! `kind` and its parameters, in order; NaN takes the default of the
//! shape's options struct. Lengths are passed through (the generator checks
//! them), counts must be whole numbers in `u32` range, an axis is `0` X,
//! `1` Y or `2` Z.
//!
//! - `0` box: `size x, y, z, segments x, y, z`.
//! - `1` cylinder: `radius, height, segments, rings, cap_rings, axis`.
//! - `2` cone: `radius, top_radius, height, segments, rings, cap_rings, axis`.
//! - `3` UV sphere: `radius, segments, rings, axis`.
//! - `4` icosphere: `radius, subdivisions`.
//! - `5` torus: `major_radius, minor_radius, segments, tube_segments, axis`.
//! - `6` arrow: `length, shaft_radius, head_radius, head_length, segments, axis`.
//! - `7` prism: `sides, radius, height, axis`.
//! - `8` plane: `size u, v, segments u, v, axis`.
//!
//! The result is one `u32` buffer: `[vertices, triangles]`, then the
//! positions as `f32` bits (3 per vertex), then the indices (3 per triangle).

use fast_mesh_ops::append::AppendSource;
use fast_mesh_ops::primitives::{
    ArrowOptions, Axis, BoxOptions, ConeOptions, CylinderOptions, IcosphereOptions, PlaneOptions,
    Primitive, PrismOptions, TorusOptions, UvSphereOptions, generate,
};
use fast_mesh_store::StoreError;
use wasm_bindgen::prelude::*;

use crate::error::Error;
use crate::scene::WasmScene;

#[wasm_bindgen]
impl WasmScene {
    /// Generates primitive `kind` from `params` (see the module docs) and
    /// returns `[vertices, triangles, positions as f32 bits..., indices...]`.
    #[wasm_bindgen(js_name = generatePrimitive)]
    pub fn generate_primitive(kind: u8, params: &[f64]) -> Result<Vec<u32>, Error> {
        let shape = decode(kind, params)?;
        pack(&generate(&shape)?)
    }
}

/// The parameters of `kind`, NaN meaning the default.
struct Params<'a>(&'a [f64]);

impl Params<'_> {
    fn length(&self, i: usize, default: f64) -> f64 {
        let v = self.0[i];
        if v.is_nan() { default } else { v }
    }

    fn count(&self, i: usize, default: u32) -> Result<u32, Error> {
        let v = self.0[i];
        if v.is_nan() {
            return Ok(default);
        }
        if v.fract() != 0.0 || !(0.0..=f64::from(u32::MAX)).contains(&v) {
            return Err(Error::invalid(
                "primitive resolutions must be whole numbers in u32 range",
            ));
        }
        Ok(v as u32)
    }

    fn axis(&self, i: usize, default: Axis) -> Result<Axis, Error> {
        let v = self.0[i];
        match v {
            _ if v.is_nan() => Ok(default),
            0.0 => Ok(Axis::X),
            1.0 => Ok(Axis::Y),
            2.0 => Ok(Axis::Z),
            _ => Err(Error::invalid(
                "a primitive axis must be 0 (x), 1 (y) or 2 (z)",
            )),
        }
    }
}

/// The parameter count of each kind, by code.
const ARITY: [usize; 9] = [6, 6, 7, 4, 2, 5, 6, 4, 5];

fn decode(kind: u8, params: &[f64]) -> Result<Primitive, Error> {
    let Some(&arity) = ARITY.get(usize::from(kind)) else {
        return Err(Error::invalid("primitive kind must be 0 to 8"));
    };
    if params.len() != arity {
        return Err(Error::invalid("wrong number of primitive parameters"));
    }
    let p = Params(params);
    Ok(match kind {
        0 => {
            let d = BoxOptions::default();
            Primitive::Box(BoxOptions {
                size: [
                    p.length(0, d.size[0]),
                    p.length(1, d.size[1]),
                    p.length(2, d.size[2]),
                ],
                segments: [
                    p.count(3, d.segments[0])?,
                    p.count(4, d.segments[1])?,
                    p.count(5, d.segments[2])?,
                ],
            })
        }
        1 => {
            let d = CylinderOptions::default();
            Primitive::Cylinder(CylinderOptions {
                radius: p.length(0, d.radius),
                height: p.length(1, d.height),
                segments: p.count(2, d.segments)?,
                rings: p.count(3, d.rings)?,
                cap_rings: p.count(4, d.cap_rings)?,
                axis: p.axis(5, d.axis)?,
            })
        }
        2 => {
            let d = ConeOptions::default();
            Primitive::Cone(ConeOptions {
                radius: p.length(0, d.radius),
                top_radius: p.length(1, d.top_radius),
                height: p.length(2, d.height),
                segments: p.count(3, d.segments)?,
                rings: p.count(4, d.rings)?,
                cap_rings: p.count(5, d.cap_rings)?,
                axis: p.axis(6, d.axis)?,
            })
        }
        3 => {
            let d = UvSphereOptions::default();
            Primitive::UvSphere(UvSphereOptions {
                radius: p.length(0, d.radius),
                segments: p.count(1, d.segments)?,
                rings: p.count(2, d.rings)?,
                axis: p.axis(3, d.axis)?,
            })
        }
        4 => {
            let d = IcosphereOptions::default();
            Primitive::Icosphere(IcosphereOptions {
                radius: p.length(0, d.radius),
                subdivisions: p.count(1, d.subdivisions)?,
            })
        }
        5 => {
            let d = TorusOptions::default();
            Primitive::Torus(TorusOptions {
                major_radius: p.length(0, d.major_radius),
                minor_radius: p.length(1, d.minor_radius),
                segments: p.count(2, d.segments)?,
                tube_segments: p.count(3, d.tube_segments)?,
                axis: p.axis(4, d.axis)?,
            })
        }
        6 => {
            let d = ArrowOptions::default();
            Primitive::Arrow(ArrowOptions {
                length: p.length(0, d.length),
                shaft_radius: p.length(1, d.shaft_radius),
                head_radius: p.length(2, d.head_radius),
                head_length: p.length(3, d.head_length),
                segments: p.count(4, d.segments)?,
                axis: p.axis(5, d.axis)?,
            })
        }
        7 => {
            let d = PrismOptions::default();
            Primitive::Prism(PrismOptions {
                sides: p.count(0, d.sides)?,
                radius: p.length(1, d.radius),
                height: p.length(2, d.height),
                axis: p.axis(3, d.axis)?,
            })
        }
        _ => {
            let d = PlaneOptions::default();
            Primitive::Plane(PlaneOptions {
                size: [p.length(0, d.size[0]), p.length(1, d.size[1])],
                segments: [p.count(2, d.segments[0])?, p.count(3, d.segments[1])?],
                axis: p.axis(4, d.axis)?,
            })
        }
    })
}

/// `src` as `[vertices, triangles, positions as f32 bits..., indices...]`.
pub(crate) fn pack(src: &AppendSource) -> Result<Vec<u32>, Error> {
    let (v, t) = (src.positions.len(), src.triangles.len());
    let mut out = Vec::new();
    out.try_reserve_exact(2 + 3 * v + 3 * t)
        .map_err(|_| Error::from(StoreError::OutOfMemory))?;
    // The generator refuses shapes whose counts do not fit `u32`.
    out.extend([v as u32, t as u32]);
    out.extend(src.positions.iter().flatten().map(|c| c.to_bits()));
    out.extend(src.triangles.iter().flatten());
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::codes;

    fn unpack(raw: &[u32]) -> (Vec<f32>, Vec<u32>) {
        let (v, t) = (raw[0] as usize, raw[1] as usize);
        let positions = raw[2..2 + 3 * v]
            .iter()
            .map(|&b| f32::from_bits(b))
            .collect();
        let indices = raw[2 + 3 * v..].to_vec();
        assert_eq!(indices.len(), 3 * t);
        (positions, indices)
    }

    #[test]
    fn nan_takes_each_default_and_matches_the_generator() {
        let defaults = [
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
        for (kind, shape) in defaults.iter().enumerate() {
            let params = vec![f64::NAN; ARITY[kind]];
            let raw = WasmScene::generate_primitive(kind as u8, &params).unwrap();
            let want = generate(shape).unwrap();
            let (positions, indices) = unpack(&raw);
            let want_p: Vec<f32> = want.positions.iter().flatten().copied().collect();
            let want_i: Vec<u32> = want.triangles.iter().flatten().copied().collect();
            assert_eq!((positions, indices), (want_p, want_i));
        }
    }

    #[test]
    fn parameters_land_in_order() {
        let raw = WasmScene::generate_primitive(0, &[2.0, 4.0, 6.0, 1.0, 2.0, 3.0]).unwrap();
        assert_eq!(raw[1], 4 * (2 + 6 + 3));
        let (p, _) = unpack(&raw);
        let max = |k: usize| p.iter().skip(k).step_by(3).fold(f32::MIN, |m, &c| m.max(c));
        assert_eq!([max(0), max(1), max(2)], [1.0, 2.0, 3.0]);
        let cone = [1.0, 0.5, 2.0, 5.0, 2.0, 3.0, 2.0];
        let raw = WasmScene::generate_primitive(2, &cone).unwrap();
        let want = generate(&Primitive::Cone(ConeOptions {
            radius: 1.0,
            top_radius: 0.5,
            height: 2.0,
            segments: 5,
            rings: 2,
            cap_rings: 3,
            axis: Axis::Z,
        }))
        .unwrap();
        assert_eq!(raw[0] as usize, want.positions.len());
        assert_eq!(raw[1] as usize, want.triangles.len());
    }

    #[test]
    fn bad_encodings_are_invalid_arguments() {
        let invalid = |kind: u8, params: &[f64]| {
            let e = WasmScene::generate_primitive(kind, params).unwrap_err();
            assert_eq!(e.code, codes::INVALID_ARGUMENT, "{}", e.text);
        };
        invalid(9, &[]);
        invalid(0, &[1.0; 5]);
        invalid(3, &[0.5, 32.5, 16.0, 1.0]);
        invalid(3, &[0.5, -1.0, 16.0, 1.0]);
        invalid(3, &[0.5, 32.0, 16.0, 3.0]);
        // Out of range for the generator.
        invalid(3, &[0.0, 32.0, 16.0, 1.0]);
        invalid(5, &[0.1, 0.4, 8.0, 8.0, 1.0]);
    }
}
