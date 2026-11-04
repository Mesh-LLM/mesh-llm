//! The ggml tensor types a receipt can name: their ids, their names and
//! how many bytes a block of their values takes.
//!
//! A node names the types of its cache rows in a receipt, and a client
//! reads the types of the claimed GGUF's weights and of the rows a node
//! opens. Both sides take the ids, names and sizes from here, so a name in
//! a receipt means the same type to both.

/// ggml type ids of the formats [`crate::dequant`] decodes.
pub(crate) const GGML_F32: u32 = 0;
pub(crate) const GGML_F16: u32 = 1;
pub(crate) const GGML_Q4_0: u32 = 2;
pub(crate) const GGML_Q4_1: u32 = 3;
pub(crate) const GGML_Q5_0: u32 = 6;
pub(crate) const GGML_Q5_1: u32 = 7;
pub(crate) const GGML_Q8_0: u32 = 8;
pub(crate) const GGML_Q2_K: u32 = 10;
pub(crate) const GGML_Q3_K: u32 = 11;
pub(crate) const GGML_Q4_K: u32 = 12;
pub(crate) const GGML_Q5_K: u32 = 13;
pub(crate) const GGML_Q6_K: u32 = 14;
pub(crate) const GGML_IQ4_NL: u32 = 20;
pub(crate) const GGML_IQ4_XS: u32 = 23;
pub(crate) const GGML_BF16: u32 = 30;

/// (values per block, bytes per block) of every ggml type the pinned
/// runtime loads from a GGUF, so a tensor of any of them has a size: the
/// ones the verifier decodes ([`crate::dequant::supports`]) and the rest,
/// by their ggml type id. A type id that is not here was removed from ggml
/// or is unknown to it, and the runtime refuses the file.
pub(crate) fn type_layout(t: u32) -> Option<(u64, u64)> {
    Some(match t {
        GGML_F32 => (1, 4),
        GGML_F16 => (1, 2),
        GGML_Q4_0 => (32, 18),
        GGML_Q4_1 => (32, 20),
        GGML_Q5_0 => (32, 22),
        GGML_Q5_1 => (32, 24),
        GGML_Q8_0 => (32, 34),
        GGML_Q2_K => (256, 84),
        GGML_Q3_K => (256, 110),
        GGML_Q4_K => (256, 144),
        GGML_Q5_K => (256, 176),
        GGML_Q6_K => (256, 210),
        GGML_IQ4_NL => (32, 18),
        GGML_IQ4_XS => (256, 136),
        GGML_BF16 => (1, 2),
        // q8_1, q8_K
        9 => (32, 36),
        15 => (256, 292),
        // iq2_xxs, iq2_xs, iq3_xxs, iq1_s, iq3_s, iq2_s, iq1_m
        16 => (256, 66),
        17 => (256, 74),
        18 => (256, 98),
        19 => (256, 50),
        21 => (256, 110),
        22 => (256, 82),
        29 => (256, 56),
        // i8, i16, i32, i64, f64
        24 => (1, 1),
        25 => (1, 2),
        26 => (1, 4),
        27 | 28 => (1, 8),
        // tq1_0, tq2_0
        34 => (256, 54),
        35 => (256, 66),
        // mxfp4, nvfp4, q1_0, q2_0, f8_e4m3
        39 => (32, 17),
        40 => (64, 36),
        41 => (128, 18),
        42 => (64, 18),
        43 => (1, 1),
        _ => return None,
    })
}

/// How a receipt names the type with id `t`. An id without a name of its
/// own is named by its number.
pub(crate) fn type_name(t: u32) -> String {
    match t {
        GGML_F32 => "f32".into(),
        GGML_F16 => "f16".into(),
        GGML_Q4_0 => "q4_0".into(),
        GGML_Q4_1 => "q4_1".into(),
        GGML_Q5_0 => "q5_0".into(),
        GGML_Q5_1 => "q5_1".into(),
        GGML_Q8_0 => "q8_0".into(),
        GGML_Q2_K => "q2_K".into(),
        GGML_Q3_K => "q3_K".into(),
        GGML_Q4_K => "q4_K".into(),
        GGML_Q5_K => "q5_K".into(),
        GGML_Q6_K => "q6_K".into(),
        GGML_IQ4_NL => "iq4_nl".into(),
        GGML_IQ4_XS => "iq4_xs".into(),
        GGML_BF16 => "bf16".into(),
        _ => format!("type{t}"),
    }
}
