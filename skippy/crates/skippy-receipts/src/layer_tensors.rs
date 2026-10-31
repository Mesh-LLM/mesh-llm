//! The per-layer tensors a stage reports for receipts, found by graph
//! structure (see `mesh/docs/design/RECEIPTS_GLUE.md`).
//!
//! Every tensor is keyed by layer, kind and index: its ordinal among the
//! layer's tensors of that kind, in graph order. The same builder gives the
//! same order for a serving ubatch, a replay and an auditor's glue
//! evaluation, so the index names the same value in all three.

/// ggml type id of f32 values.
pub const GGML_F32: i32 = 0;
/// ggml type id of i32 values.
pub const GGML_I32: i32 = 26;

/// A reported per-layer tensor (`skippy_receipt_tensor_kind`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum TensorKind {
    /// The residual stream entering the layer; at `layer_end`, the stage output.
    Residual,
    /// An input of a token-mixing op that the layer computes, such as q.
    MixingInput,
    /// A row written to the KV cache or recurrent state. While serving and
    /// in a replay it is the row as stored, in the cache's type; in a glue
    /// evaluation it is the f32 value the layer computed for it.
    MixingStored,
    /// The output of a token-mixing region, such as attention's output.
    MixingOutput,
    /// The f32 input of a weight matmul.
    MatmulInput,
    MatmulOutput,
    /// The experts a `mul_mat_id` used, i32.
    ExpertIds,
    /// The ids an `argsort` or top-k picked from scores, i32.
    Selection,
    /// The scores a selection picked from.
    SelectionScores,
}

/// The kinds the token-mixing leaf commits to, in leaf order.
pub const MIXING_KINDS: [TensorKind; 3] = [
    TensorKind::MixingInput,
    TensorKind::MixingStored,
    TensorKind::MixingOutput,
];

impl TensorKind {
    pub fn from_code(code: i32) -> Option<Self> {
        Some(match code {
            0 => Self::Residual,
            1 => Self::MixingInput,
            2 => Self::MixingStored,
            3 => Self::MixingOutput,
            4 => Self::MatmulInput,
            5 => Self::MatmulOutput,
            6 => Self::ExpertIds,
            7 => Self::Selection,
            8 => Self::SelectionScores,
            _ => return None,
        })
    }

    pub fn code(self) -> i32 {
        match self {
            Self::Residual => 0,
            Self::MixingInput => 1,
            Self::MixingStored => 2,
            Self::MixingOutput => 3,
            Self::MatmulInput => 4,
            Self::MatmulOutput => 5,
            Self::ExpertIds => 6,
            Self::Selection => 7,
            Self::SelectionScores => 8,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Self::Residual => "residual",
            Self::MixingInput => "mixing input",
            Self::MixingStored => "stored row",
            Self::MixingOutput => "mixing output",
            Self::MatmulInput => "matmul input",
            Self::MatmulOutput => "matmul output",
            Self::ExpertIds => "expert ids",
            Self::Selection => "selection",
            Self::SelectionScores => "selection scores",
        }
    }
}

/// `skippy_receipt_tensor_flags`: the matmul is a router, whose output
/// feeds a selection.
pub const FLAG_ROUTER: i32 = 1;

/// One column of a reported tensor.
#[derive(Clone, Copy, Debug)]
pub struct TensorColumn<'a> {
    /// ggml type of the bytes: f32, i32, or a stored row's cache type.
    pub ggml_type: i32,
    pub flags: i32,
    /// The weight's name for a matmul, else the graph node's name.
    pub name: &'a str,
    /// Little-endian values.
    pub bytes: &'a [u8],
}

impl TensorColumn<'_> {
    pub fn f32_values(&self) -> Option<Vec<f32>> {
        (self.ggml_type == GGML_F32).then(|| decode_f32_bytes(self.bytes))
    }

    pub fn i32_values(&self) -> Option<Vec<i32>> {
        (self.ggml_type == GGML_I32).then(|| {
            self.bytes
                .as_chunks::<4>()
                .0
                .iter()
                .map(|c| i32::from_le_bytes(*c))
                .collect()
        })
    }
}

/// The reported tensors of a ubatch, by column.
pub trait UbatchColumns: Sync {
    /// Tensors of `kind` the ubatch reported for `layer`.
    fn count(&self, layer: u32, kind: TensorKind) -> u32;
    /// Column `col` of a tensor; `None` when the tensor holds no row for the
    /// column (the final layer of a final stage keeps only output rows) or
    /// was not reported.
    fn column(
        &self,
        layer: u32,
        kind: TensorKind,
        index: u32,
        col: usize,
    ) -> Option<TensorColumn<'_>>;
}

pub fn decode_f32_bytes(bytes: &[u8]) -> Vec<f32> {
    bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| f32::from_le_bytes(*c))
        .collect()
}

pub fn f32_bytes(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

pub fn i32_bytes(values: &[i32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kind_codes_round_trip() {
        for code in 0..9 {
            assert_eq!(TensorKind::from_code(code).unwrap().code(), code);
        }
        assert!(TensorKind::from_code(9).is_none());
    }
}
