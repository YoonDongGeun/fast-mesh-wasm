//! Colormaps and colour rules: turning a scalar field value into an RGBA8
//! colour.
//!
//! A [`ColorRule`] maps values in `range` onto a [`Colormap`]. Without a
//! `center` the mapping is linear, `t = (v - lo) / (hi - lo)`. With a
//! `center` `c` it is piecewise linear and pins `c` to the middle of the
//! colormap: `t = 0.5 (v - lo) / (c - lo)` on `[lo, c]` and
//! `t = 0.5 + 0.5 (v - c) / (hi - c)` on `[c, hi]`.
//!
//! The colormap is sampled at `t * 255` **with linear interpolation** between
//! the two neighbouring table entries (per channel, rounded to nearest).
//!
//! Outside the range, [`ColorRule::below`] decides for `v < lo` and
//! [`ColorRule::above`] for `v > hi`: [`OutOfRange::Clamp`] gives the end
//! colour of the colormap, [`OutOfRange::Base`] gives [`ColorRule::base`].
//! `-∞` follows `below` and `+∞` follows `above`; because the "far" value of
//! a distance kernel is `+∞`, it follows `above` (`Base` shows the base
//! colour, `Clamp` the end colour). NaN gives [`ColorRule::nan`].
//!
//! Built-in tables have 256 RGBA8 entries with alpha 255: Turbo is the
//! published table, the others are generated on first use:
//!
//! - **Turbo**: Anton Mikhailov's published 256-entry look-up table of the
//!   Turbo colormap (Google, 2019; Apache-2.0).
//! - **Viridis**: Matt Zucker's published degree-6 polynomial fit of
//!   matplotlib's viridis, evaluated at 256 points (within a few 8-bit
//!   levels of matplotlib's table).
//! - **Coolwarm**: Kenneth Moreland's diverging map ("Diverging Color Maps
//!   for Scientific Visualization", 2009), interpolated in Msh space between
//!   sRGB (59, 76, 192) and (180, 4, 38) through a white-point of M = 88.
//! - **Grayscale**: linear, entry `i` is `(i, i, i, 255)`.

use std::sync::OnceLock;

use crate::{FieldsError, Result};

/// Number of entries in each built-in colormap table.
const TABLE_LEN: usize = 256;

/// A colormap from `t` in `[0, 1]` to RGBA8.
#[derive(Debug, Clone, PartialEq)]
pub enum Colormap {
    /// Google's Turbo rainbow (published table).
    Turbo,
    /// Matplotlib's perceptually uniform Viridis (polynomial fit).
    Viridis,
    /// Moreland's blue-white-red diverging map.
    Coolwarm,
    /// Linear black to white.
    Grayscale,
    /// Colour stops `(position, rgba)`: at least two, positions in `[0, 1]`
    /// and strictly increasing. Colours are interpolated linearly per channel
    /// and rounded; `t` before the first or after the last stop takes that
    /// stop's colour.
    Custom(Vec<(f32, [u8; 4])>),
}

/// What to show for a value outside the colour range.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OutOfRange {
    /// The colormap's end colour (`t = 0` below, `t = 1` above).
    Clamp,
    /// The rule's [`ColorRule::base`] colour.
    Base,
}

/// How a scalar field is turned into colours.
#[derive(Debug, Clone, PartialEq)]
pub struct ColorRule {
    /// The colormap.
    pub map: Colormap,
    /// `(lo, hi)`: values mapped onto the colormap; `lo < hi`, both finite.
    pub range: (f32, f32),
    /// Colour rule for values below `lo` (and `-∞`).
    pub below: OutOfRange,
    /// Colour rule for values above `hi` (and `+∞`).
    pub above: OutOfRange,
    /// A value pinned to the middle of the colormap, within `range`.
    pub center: Option<f32>,
    /// The colour of [`OutOfRange::Base`].
    pub base: [u8; 4],
    /// The colour of NaN values.
    pub nan: [u8; 4],
}

impl ColorRule {
    /// Checks the rule.
    ///
    /// # Errors
    ///
    /// [`FieldsError::InvalidInput`] if `range` has a non-finite end or
    /// `lo >= hi`, if `center` is non-finite or outside `range`, or if a
    /// custom colormap has fewer than two stops or stop positions that are
    /// not finite, outside `[0, 1]` or not strictly increasing.
    pub fn validate(&self) -> Result<()> {
        let (lo, hi) = self.range;
        if !lo.is_finite() || !hi.is_finite() {
            return Err(FieldsError::InvalidInput("color range must be finite"));
        }
        if lo >= hi {
            return Err(FieldsError::InvalidInput("color range needs lo < hi"));
        }
        if let Some(c) = self.center
            && (!c.is_finite() || c < lo || c > hi)
        {
            return Err(FieldsError::InvalidInput(
                "color center must lie within the range",
            ));
        }
        if let Colormap::Custom(stops) = &self.map {
            if stops.len() < 2 {
                return Err(FieldsError::InvalidInput(
                    "custom colormap needs at least two stops",
                ));
            }
            let mut prev = -1.0_f32;
            for &(p, _) in stops {
                if !(0.0..=1.0).contains(&p) || p <= prev {
                    return Err(FieldsError::InvalidInput(
                        "custom stops must be in [0, 1] and strictly increasing",
                    ));
                }
                prev = p;
            }
        }
        Ok(())
    }
}

/// The colour of value `v` under `rule`.
///
/// `rule` should have passed [`ColorRule::validate`]; an invalid rule never
/// panics but gives unspecified colours.
pub fn color_of(rule: &ColorRule, v: f32) -> [u8; 4] {
    if v.is_nan() {
        return rule.nan;
    }
    let (lo, hi) = rule.range;
    if v < lo {
        return match rule.below {
            OutOfRange::Clamp => sample(&rule.map, 0.0),
            OutOfRange::Base => rule.base,
        };
    }
    if v > hi {
        return match rule.above {
            OutOfRange::Clamp => sample(&rule.map, 1.0),
            OutOfRange::Base => rule.base,
        };
    }
    let t = match rule.center {
        None => (v - lo) / (hi - lo),
        Some(c) => {
            if v == c {
                0.5
            } else if v < c {
                0.5 * (v - lo) / (c - lo)
            } else {
                0.5 + 0.5 * (v - c) / (hi - c)
            }
        }
    };
    sample(&rule.map, if t.is_nan() { 0.0 } else { t })
}

/// Samples a colormap at `t` (clamped to `[0, 1]`).
fn sample(map: &Colormap, t: f32) -> [u8; 4] {
    let t = t.clamp(0.0, 1.0);
    match map {
        Colormap::Custom(stops) => sample_stops(stops, t),
        builtin => {
            let table = table_of(builtin);
            let x = t * (TABLE_LEN - 1) as f32;
            let i = (x.floor() as usize).min(TABLE_LEN - 2);
            lerp_rgba(table[i], table[i + 1], x - i as f32)
        }
    }
}

fn sample_stops(stops: &[(f32, [u8; 4])], t: f32) -> [u8; 4] {
    let (Some(first), Some(last)) = (stops.first(), stops.last()) else {
        return [0, 0, 0, 0];
    };
    if t <= first.0 {
        return first.1;
    }
    if t >= last.0 {
        return last.1;
    }
    for w in stops.windows(2) {
        let (p0, c0) = w[0];
        let (p1, c1) = w[1];
        if t <= p1 {
            let span = p1 - p0;
            let f = if span > 0.0 { (t - p0) / span } else { 1.0 };
            return lerp_rgba(c0, c1, f);
        }
    }
    last.1
}

fn lerp_rgba(a: [u8; 4], b: [u8; 4], f: f32) -> [u8; 4] {
    let mut out = [0u8; 4];
    for k in 0..4 {
        let v = f32::from(a[k]) + (f32::from(b[k]) - f32::from(a[k])) * f;
        out[k] = v.round().clamp(0.0, 255.0) as u8;
    }
    out
}

/// Google's published 256-entry Turbo look-up table (Anton Mikhailov,
/// 2019; Apache-2.0), as 8-bit sRGB triplets with opaque alpha.
const TURBO_TABLE: [[u8; 4]; TABLE_LEN] = [
    [48, 18, 59, 255],
    [50, 21, 67, 255],
    [51, 24, 74, 255],
    [52, 27, 81, 255],
    [53, 30, 88, 255],
    [54, 33, 95, 255],
    [55, 36, 102, 255],
    [56, 39, 109, 255],
    [57, 42, 115, 255],
    [58, 45, 121, 255],
    [59, 47, 128, 255],
    [60, 50, 134, 255],
    [61, 53, 139, 255],
    [62, 56, 145, 255],
    [63, 59, 151, 255],
    [63, 62, 156, 255],
    [64, 64, 162, 255],
    [65, 67, 167, 255],
    [65, 70, 172, 255],
    [66, 73, 177, 255],
    [66, 75, 181, 255],
    [67, 78, 186, 255],
    [68, 81, 191, 255],
    [68, 84, 195, 255],
    [68, 86, 199, 255],
    [69, 89, 203, 255],
    [69, 92, 207, 255],
    [69, 94, 211, 255],
    [70, 97, 214, 255],
    [70, 100, 218, 255],
    [70, 102, 221, 255],
    [70, 105, 224, 255],
    [70, 107, 227, 255],
    [71, 110, 230, 255],
    [71, 113, 233, 255],
    [71, 115, 235, 255],
    [71, 118, 238, 255],
    [71, 120, 240, 255],
    [71, 123, 242, 255],
    [70, 125, 244, 255],
    [70, 128, 246, 255],
    [70, 130, 248, 255],
    [70, 133, 250, 255],
    [70, 135, 251, 255],
    [69, 138, 252, 255],
    [69, 140, 253, 255],
    [68, 143, 254, 255],
    [67, 145, 254, 255],
    [66, 148, 255, 255],
    [65, 150, 255, 255],
    [64, 153, 255, 255],
    [62, 155, 254, 255],
    [61, 158, 254, 255],
    [59, 160, 253, 255],
    [58, 163, 252, 255],
    [56, 165, 251, 255],
    [55, 168, 250, 255],
    [53, 171, 248, 255],
    [51, 173, 247, 255],
    [49, 175, 245, 255],
    [47, 178, 244, 255],
    [46, 180, 242, 255],
    [44, 183, 240, 255],
    [42, 185, 238, 255],
    [40, 188, 235, 255],
    [39, 190, 233, 255],
    [37, 192, 231, 255],
    [35, 195, 228, 255],
    [34, 197, 226, 255],
    [32, 199, 223, 255],
    [31, 201, 221, 255],
    [30, 203, 218, 255],
    [28, 205, 216, 255],
    [27, 208, 213, 255],
    [26, 210, 210, 255],
    [26, 212, 208, 255],
    [25, 213, 205, 255],
    [24, 215, 202, 255],
    [24, 217, 200, 255],
    [24, 219, 197, 255],
    [24, 221, 194, 255],
    [24, 222, 192, 255],
    [24, 224, 189, 255],
    [25, 226, 187, 255],
    [25, 227, 185, 255],
    [26, 228, 182, 255],
    [28, 230, 180, 255],
    [29, 231, 178, 255],
    [31, 233, 175, 255],
    [32, 234, 172, 255],
    [34, 235, 170, 255],
    [37, 236, 167, 255],
    [39, 238, 164, 255],
    [42, 239, 161, 255],
    [44, 240, 158, 255],
    [47, 241, 155, 255],
    [50, 242, 152, 255],
    [53, 243, 148, 255],
    [56, 244, 145, 255],
    [60, 245, 142, 255],
    [63, 246, 138, 255],
    [67, 247, 135, 255],
    [70, 248, 132, 255],
    [74, 248, 128, 255],
    [78, 249, 125, 255],
    [82, 250, 122, 255],
    [85, 250, 118, 255],
    [89, 251, 115, 255],
    [93, 252, 111, 255],
    [97, 252, 108, 255],
    [101, 253, 105, 255],
    [105, 253, 102, 255],
    [109, 254, 98, 255],
    [113, 254, 95, 255],
    [117, 254, 92, 255],
    [121, 254, 89, 255],
    [125, 255, 86, 255],
    [128, 255, 83, 255],
    [132, 255, 81, 255],
    [136, 255, 78, 255],
    [139, 255, 75, 255],
    [143, 255, 73, 255],
    [146, 255, 71, 255],
    [150, 254, 68, 255],
    [153, 254, 66, 255],
    [156, 254, 64, 255],
    [159, 253, 63, 255],
    [161, 253, 61, 255],
    [164, 252, 60, 255],
    [167, 252, 58, 255],
    [169, 251, 57, 255],
    [172, 251, 56, 255],
    [175, 250, 55, 255],
    [177, 249, 54, 255],
    [180, 248, 54, 255],
    [183, 247, 53, 255],
    [185, 246, 53, 255],
    [188, 245, 52, 255],
    [190, 244, 52, 255],
    [193, 243, 52, 255],
    [195, 241, 52, 255],
    [198, 240, 52, 255],
    [200, 239, 52, 255],
    [203, 237, 52, 255],
    [205, 236, 52, 255],
    [208, 234, 52, 255],
    [210, 233, 53, 255],
    [212, 231, 53, 255],
    [215, 229, 53, 255],
    [217, 228, 54, 255],
    [219, 226, 54, 255],
    [221, 224, 55, 255],
    [223, 223, 55, 255],
    [225, 221, 55, 255],
    [227, 219, 56, 255],
    [229, 217, 56, 255],
    [231, 215, 57, 255],
    [233, 213, 57, 255],
    [235, 211, 57, 255],
    [236, 209, 58, 255],
    [238, 207, 58, 255],
    [239, 205, 58, 255],
    [241, 203, 58, 255],
    [242, 201, 58, 255],
    [244, 199, 58, 255],
    [245, 197, 58, 255],
    [246, 195, 58, 255],
    [247, 193, 58, 255],
    [248, 190, 57, 255],
    [249, 188, 57, 255],
    [250, 186, 57, 255],
    [251, 184, 56, 255],
    [251, 182, 55, 255],
    [252, 179, 54, 255],
    [252, 177, 54, 255],
    [253, 174, 53, 255],
    [253, 172, 52, 255],
    [254, 169, 51, 255],
    [254, 167, 50, 255],
    [254, 164, 49, 255],
    [254, 161, 48, 255],
    [254, 158, 47, 255],
    [254, 155, 45, 255],
    [254, 153, 44, 255],
    [254, 150, 43, 255],
    [254, 147, 42, 255],
    [254, 144, 41, 255],
    [253, 141, 39, 255],
    [253, 138, 38, 255],
    [252, 135, 37, 255],
    [252, 132, 35, 255],
    [251, 129, 34, 255],
    [251, 126, 33, 255],
    [250, 123, 31, 255],
    [249, 120, 30, 255],
    [249, 117, 29, 255],
    [248, 114, 28, 255],
    [247, 111, 26, 255],
    [246, 108, 25, 255],
    [245, 105, 24, 255],
    [244, 102, 23, 255],
    [243, 99, 21, 255],
    [242, 96, 20, 255],
    [241, 93, 19, 255],
    [240, 91, 18, 255],
    [239, 88, 17, 255],
    [237, 85, 16, 255],
    [236, 83, 15, 255],
    [235, 80, 14, 255],
    [234, 78, 13, 255],
    [232, 75, 12, 255],
    [231, 73, 12, 255],
    [229, 71, 11, 255],
    [228, 69, 10, 255],
    [226, 67, 10, 255],
    [225, 65, 9, 255],
    [223, 63, 8, 255],
    [221, 61, 8, 255],
    [220, 59, 7, 255],
    [218, 57, 7, 255],
    [216, 55, 6, 255],
    [214, 53, 6, 255],
    [212, 51, 5, 255],
    [210, 49, 5, 255],
    [208, 47, 5, 255],
    [206, 45, 4, 255],
    [204, 43, 4, 255],
    [202, 42, 4, 255],
    [200, 40, 3, 255],
    [197, 38, 3, 255],
    [195, 37, 3, 255],
    [193, 35, 2, 255],
    [190, 33, 2, 255],
    [188, 32, 2, 255],
    [185, 30, 2, 255],
    [183, 29, 2, 255],
    [180, 27, 1, 255],
    [178, 26, 1, 255],
    [175, 24, 1, 255],
    [172, 23, 1, 255],
    [169, 22, 1, 255],
    [167, 20, 1, 255],
    [164, 19, 1, 255],
    [161, 18, 1, 255],
    [158, 16, 1, 255],
    [155, 15, 1, 255],
    [152, 14, 1, 255],
    [149, 13, 1, 255],
    [146, 11, 1, 255],
    [142, 10, 1, 255],
    [139, 9, 2, 255],
    [136, 8, 2, 255],
    [133, 7, 2, 255],
    [129, 6, 2, 255],
    [126, 5, 2, 255],
    [122, 4, 3, 255],
];

/// The 256-entry table of a built-in colormap.
fn table_of(map: &Colormap) -> &'static [[u8; 4]; TABLE_LEN] {
    static VIRIDIS: OnceLock<[[u8; 4]; TABLE_LEN]> = OnceLock::new();
    static COOLWARM: OnceLock<[[u8; 4]; TABLE_LEN]> = OnceLock::new();
    static GRAY: OnceLock<[[u8; 4]; TABLE_LEN]> = OnceLock::new();
    match map {
        Colormap::Turbo => &TURBO_TABLE,
        Colormap::Viridis => VIRIDIS.get_or_init(|| build_table(viridis)),
        Colormap::Coolwarm => COOLWARM.get_or_init(|| build_table(coolwarm)),
        Colormap::Grayscale => GRAY.get_or_init(|| build_table(grayscale)),
        // Custom colormaps sample their stops directly and never reach here.
        Colormap::Custom(_) => unreachable!("custom colormaps do not use tables"),
    }
}

fn build_table(f: fn(f64) -> [f64; 3]) -> [[u8; 4]; TABLE_LEN] {
    let mut table = [[0u8; 4]; TABLE_LEN];
    for (i, entry) in table.iter_mut().enumerate() {
        let c = f(i as f64 / (TABLE_LEN - 1) as f64);
        *entry = [to_u8(c[0]), to_u8(c[1]), to_u8(c[2]), 255];
    }
    table
}

fn to_u8(x: f64) -> u8 {
    (x * 255.0).round().clamp(0.0, 255.0) as u8
}

fn grayscale(t: f64) -> [f64; 3] {
    [t, t, t]
}

/// Matt Zucker's degree-6 polynomial fit of matplotlib's viridis.
fn viridis(t: f64) -> [f64; 3] {
    const C: [[f64; 3]; 7] = [
        [
            0.277_727_327_223_417_7,
            0.005_407_344_544_966_578,
            0.334_099_805_335_306_1,
        ],
        [
            0.105_093_043_108_577_4,
            1.404_613_529_898_575,
            1.384_590_162_594_685,
        ],
        [
            -0.330_861_828_725_556_3,
            0.214_847_559_468_213,
            0.095_095_163_028_236_59,
        ],
        [
            -4.634_230_498_983_486,
            -5.799_100_973_351_585,
            -19.332_440_956_279_87,
        ],
        [
            6.228_269_936_347_081,
            14.179_933_366_805_09,
            56.690_552_600_681_05,
        ],
        [
            4.776_384_997_670_288,
            -13.745_145_377_746_01,
            -65.353_032_633_372_34,
        ],
        [
            -5.435_455_855_934_631,
            4.645_852_612_178_535,
            26.312_435_249_583_2,
        ],
    ];
    let mut out = [0.0; 3];
    for (ch, o) in out.iter_mut().enumerate() {
        let mut acc = 0.0;
        for c in C.iter().rev() {
            acc = acc * t + c[ch];
        }
        *o = acc.clamp(0.0, 1.0);
    }
    out
}

/// Moreland's diverging map, interpolated in Msh space.
fn coolwarm(t: f64) -> [f64; 3] {
    static ENDS: OnceLock<(Msh, Msh)> = OnceLock::new();
    let (cool, warm) = ENDS.get_or_init(|| {
        (
            srgb_to_msh([59.0 / 255.0, 76.0 / 255.0, 192.0 / 255.0]),
            srgb_to_msh([180.0 / 255.0, 4.0 / 255.0, 38.0 / 255.0]),
        )
    });
    msh_to_srgb(msh_interpolate(*cool, *warm, t))
}

type Msh = [f64; 3];

fn srgb_to_msh(c: [f64; 3]) -> Msh {
    let lin = c.map(|u| {
        if u <= 0.040_45 {
            u / 12.92
        } else {
            ((u + 0.055) / 1.055).powf(2.4)
        }
    });
    let x = 0.412_456_4 * lin[0] + 0.357_576_1 * lin[1] + 0.180_437_5 * lin[2];
    let y = 0.212_672_9 * lin[0] + 0.715_152_2 * lin[1] + 0.072_175_0 * lin[2];
    let z = 0.019_333_9 * lin[0] + 0.119_192_0 * lin[1] + 0.950_304_1 * lin[2];
    let f = |u: f64| {
        if u > 216.0 / 24389.0 {
            u.cbrt()
        } else {
            (24389.0 / 27.0 * u + 16.0) / 116.0
        }
    };
    let (fx, fy, fz) = (f(x / 0.950_47), f(y), f(z / 1.088_83));
    let l = 116.0 * fy - 16.0;
    let a = 500.0 * (fx - fy);
    let b = 200.0 * (fy - fz);
    let m = (l * l + a * a + b * b).sqrt();
    let s = if m > 0.0 {
        (l / m).clamp(-1.0, 1.0).acos()
    } else {
        0.0
    };
    let h = b.atan2(a);
    [m, s, h]
}

fn msh_to_srgb(msh: Msh) -> [f64; 3] {
    let [m, s, h] = msh;
    let l = m * s.cos();
    let a = m * s.sin() * h.cos();
    let b = m * s.sin() * h.sin();
    let fy = (l + 16.0) / 116.0;
    let fx = fy + a / 500.0;
    let fz = fy - b / 200.0;
    let finv = |u: f64| {
        if u * u * u > 216.0 / 24389.0 {
            u * u * u
        } else {
            (116.0 * u - 16.0) * 27.0 / 24389.0
        }
    };
    let (x, y, z) = (finv(fx) * 0.950_47, finv(fy), finv(fz) * 1.088_83);
    let lin = [
        3.240_454_2 * x - 1.537_138_5 * y - 0.498_531_4 * z,
        -0.969_266_0 * x + 1.876_010_8 * y + 0.041_556_0 * z,
        0.055_643_4 * x - 0.204_025_9 * y + 1.057_225_2 * z,
    ];
    lin.map(|u| {
        let u = u.clamp(0.0, 1.0);
        if u <= 0.003_130_8 {
            12.92 * u
        } else {
            1.055 * u.powf(1.0 / 2.4) - 0.055
        }
    })
}

/// Moreland's hue adjustment for the unsaturated end of an interpolation.
fn adjust_hue(sat: Msh, m_unsat: f64) -> f64 {
    if sat[0] >= m_unsat {
        return sat[2];
    }
    let spin = sat[1] * (m_unsat * m_unsat - sat[0] * sat[0]).sqrt() / (sat[0] * sat[1].sin());
    if sat[2] > -std::f64::consts::FRAC_PI_3 {
        sat[2] + spin
    } else {
        sat[2] - spin
    }
}

fn msh_interpolate(mut a: Msh, mut b: Msh, mut t: f64) -> Msh {
    if a[1] > 0.05 && b[1] > 0.05 && (a[2] - b[2]).abs() > std::f64::consts::FRAC_PI_3 {
        let m_mid = a[0].max(b[0]).max(88.0);
        if t < 0.5 {
            b = [m_mid, 0.0, 0.0];
            t *= 2.0;
        } else {
            a = [m_mid, 0.0, 0.0];
            t = 2.0 * t - 1.0;
        }
    }
    if a[1] < 0.05 && b[1] > 0.05 {
        a[2] = adjust_hue(b, a[0]);
    } else if b[1] < 0.05 && a[1] > 0.05 {
        b[2] = adjust_hue(a, b[0]);
    }
    [
        (1.0 - t) * a[0] + t * b[0],
        (1.0 - t) * a[1] + t * b[1],
        (1.0 - t) * a[2] + t * b[2],
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rule(map: Colormap) -> ColorRule {
        ColorRule {
            map,
            range: (0.0, 10.0),
            below: OutOfRange::Clamp,
            above: OutOfRange::Clamp,
            center: None,
            base: [9, 8, 7, 255],
            nan: [1, 2, 3, 255],
        }
    }

    fn near(a: [u8; 4], b: [u8; 4], tol: i32) -> bool {
        a.iter()
            .zip(&b)
            .all(|(&x, &y)| (i32::from(x) - i32::from(y)).abs() <= tol)
    }

    #[test]
    fn range_ends_and_center_map_to_table_positions() {
        let r = rule(Colormap::Grayscale);
        assert_eq!(color_of(&r, 0.0), [0, 0, 0, 255]);
        assert_eq!(color_of(&r, 10.0), [255, 255, 255, 255]);
        assert_eq!(color_of(&r, 5.0), [128, 128, 128, 255]);

        // Center pinned off-middle: 2 maps to t = 0.5, 6 to t = 0.75.
        let mut c = rule(Colormap::Grayscale);
        c.center = Some(2.0);
        assert_eq!(color_of(&c, 0.0), [0, 0, 0, 255]);
        assert_eq!(color_of(&c, 2.0), [128, 128, 128, 255]);
        assert_eq!(color_of(&c, 6.0), [191, 191, 191, 255]);
        assert_eq!(color_of(&c, 10.0), [255, 255, 255, 255]);

        // Built-in tables: opaque, distinct ends, expected anchors.
        for map in [Colormap::Turbo, Colormap::Viridis, Colormap::Coolwarm] {
            let r = rule(map);
            let (a, b) = (color_of(&r, 0.0), color_of(&r, 10.0));
            assert_eq!((a[3], b[3]), (255, 255));
            assert_ne!(a, b);
        }
        assert!(near(
            color_of(&rule(Colormap::Viridis), 0.0),
            [68, 1, 84, 255],
            6
        ));
        assert!(near(
            color_of(&rule(Colormap::Viridis), 10.0),
            [253, 231, 37, 255],
            6
        ));
        assert!(near(
            color_of(&rule(Colormap::Coolwarm), 0.0),
            [59, 76, 192, 255],
            3
        ));
        assert!(near(
            color_of(&rule(Colormap::Coolwarm), 10.0),
            [180, 4, 38, 255],
            3
        ));
        let mid = color_of(&rule(Colormap::Coolwarm), 5.0);
        assert!(near(mid, [221, 221, 221, 255], 4), "{mid:?}");
    }

    #[test]
    fn turbo_matches_published_table() {
        // The embedded table is Google's published 256-entry Turbo LUT
        // (Mikhailov, 2019): dark end (48, 18, 59), bright end (122, 4, 3).
        let r = rule(Colormap::Turbo);
        assert_eq!(color_of(&r, 0.0), [48, 18, 59, 255]);
        assert_eq!(color_of(&r, 10.0), [122, 4, 3, 255]);
    }

    #[test]
    fn viridis_midpoints_match_reference() {
        // Mid-points against matplotlib's viridis table.
        let r = rule(Colormap::Viridis);
        assert!(
            near(color_of(&r, 2.5), [59, 82, 139, 255], 6),
            "{:?}",
            color_of(&r, 2.5)
        );
        assert!(
            near(color_of(&r, 5.0), [33, 145, 140, 255], 6),
            "{:?}",
            color_of(&r, 5.0)
        );
        assert!(
            near(color_of(&r, 7.5), [94, 201, 98, 255], 6),
            "{:?}",
            color_of(&r, 7.5)
        );
    }

    #[test]
    fn below_and_above_are_independent() {
        let mut r = rule(Colormap::Grayscale);
        r.below = OutOfRange::Clamp;
        r.above = OutOfRange::Base;
        assert_eq!(color_of(&r, -5.0), [0, 0, 0, 255]);
        assert_eq!(color_of(&r, 15.0), r.base);
        r.below = OutOfRange::Base;
        r.above = OutOfRange::Clamp;
        assert_eq!(color_of(&r, -5.0), r.base);
        assert_eq!(color_of(&r, 15.0), [255, 255, 255, 255]);
    }

    #[test]
    fn infinities_follow_below_and_above() {
        // +inf follows `above`, -inf follows `below`.
        let mut r = rule(Colormap::Grayscale);
        r.below = OutOfRange::Base;
        r.above = OutOfRange::Base;
        assert_eq!(color_of(&r, f32::INFINITY), r.base);
        assert_eq!(color_of(&r, f32::NEG_INFINITY), r.base);
        r.above = OutOfRange::Clamp;
        assert_eq!(color_of(&r, f32::INFINITY), [255, 255, 255, 255]);
        assert_eq!(color_of(&r, f32::NEG_INFINITY), r.base);
    }

    #[test]
    fn nan_is_nan_color() {
        let r = rule(Colormap::Viridis);
        assert_eq!(color_of(&r, f32::NAN), r.nan);
    }

    #[test]
    fn custom_stops_interpolate() {
        let r = rule(Colormap::Custom(vec![
            (0.0, [0, 0, 0, 255]),
            (0.5, [100, 200, 50, 255]),
            (1.0, [200, 200, 250, 255]),
        ]));
        assert_eq!(color_of(&r, 0.0), [0, 0, 0, 255]);
        assert_eq!(color_of(&r, 2.5), [50, 100, 25, 255]);
        assert_eq!(color_of(&r, 5.0), [100, 200, 50, 255]);
        assert_eq!(color_of(&r, 7.5), [150, 200, 150, 255]);
        assert_eq!(color_of(&r, 10.0), [200, 200, 250, 255]);
        assert!(r.validate().is_ok());
    }

    #[test]
    fn invalid_rules_rejected() {
        let ok = rule(Colormap::Turbo);
        assert!(ok.validate().is_ok());
        let mut r = ok.clone();
        r.range = (1.0, 1.0);
        assert!(r.validate().is_err());
        r.range = (2.0, 1.0);
        assert!(r.validate().is_err());
        r.range = (f32::NEG_INFINITY, 1.0);
        assert!(r.validate().is_err());
        r.range = (0.0, f32::NAN);
        assert!(r.validate().is_err());
        let mut r = ok.clone();
        r.center = Some(11.0);
        assert!(r.validate().is_err());
        r.center = Some(-1.0);
        assert!(r.validate().is_err());
        r.center = Some(f32::NAN);
        assert!(r.validate().is_err());
        r.center = Some(10.0);
        assert!(r.validate().is_ok());
        let bad = |stops: Vec<(f32, [u8; 4])>| rule(Colormap::Custom(stops)).validate().is_err();
        assert!(bad(vec![(0.0, [0; 4])]));
        assert!(bad(vec![(0.5, [0; 4]), (0.2, [0; 4])]));
        assert!(bad(vec![(0.5, [0; 4]), (0.5, [0; 4])]));
        assert!(bad(vec![(-0.1, [0; 4]), (0.5, [0; 4])]));
        assert!(bad(vec![(0.0, [0; 4]), (1.5, [0; 4])]));
    }
}
