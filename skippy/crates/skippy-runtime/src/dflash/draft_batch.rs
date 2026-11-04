//! Rust-owned `llama_batch` storage for the DFlash draft context.
//!
//! The draft receives two kinds of batches: target-feature rows that are
//! encoded and injected into its KV cache, and token noise blocks that it
//! denoises into draft tokens. Owning the arrays in Rust keeps the M-RoPE
//! position layout (four position rows per embedding) local and lets one
//! allocation serve every decode.

use skippy_ffi::llama_draft::{LlamaBatch, LlamaPos, LlamaSeqId, LlamaToken};

#[derive(Default)]
pub(crate) struct DraftBatch {
    tokens: Vec<LlamaToken>,
    embeddings: Vec<f32>,
    row_positions: Vec<LlamaPos>,
    positions: Vec<LlamaPos>,
    n_seq_id: Vec<i32>,
    seq_ids: Vec<LlamaSeqId>,
    seq_id_ptrs: Vec<*mut LlamaSeqId>,
    outputs: Vec<i8>,
}

impl DraftBatch {
    pub(crate) fn clear(&mut self) {
        self.tokens.clear();
        self.embeddings.clear();
        self.row_positions.clear();
        self.outputs.clear();
        self.seq_ids.clear();
    }

    pub(crate) fn push_token(
        &mut self,
        token: LlamaToken,
        position: LlamaPos,
        seq_id: LlamaSeqId,
        output: bool,
    ) {
        self.tokens.push(token);
        self.push_row(position, seq_id, output);
    }

    /// Appends one target-feature row. `features` must be the full encoder
    /// input width (every extracted target layer, concatenated).
    pub(crate) fn push_features(
        &mut self,
        features: impl IntoIterator<Item = f32>,
        position: LlamaPos,
        seq_id: LlamaSeqId,
    ) {
        self.embeddings.extend(features);
        self.push_row(position, seq_id, false);
    }

    fn push_row(&mut self, position: LlamaPos, seq_id: LlamaSeqId, output: bool) {
        self.row_positions.push(position);
        self.seq_ids.push(seq_id);
        self.outputs.push(i8::from(output));
    }

    /// Exposes the batch to `llama_decode`. The returned pointers borrow this
    /// storage and stay valid until the batch is next mutated.
    ///
    /// Embedding batches on an M-RoPE draft carry four position rows per
    /// token: three copies of the sequence position and a zero fourth axis.
    pub(crate) fn as_raw(&mut self, mrope_embeddings: bool) -> LlamaBatch {
        let rows = self.row_positions.len();
        self.positions.clear();
        if mrope_embeddings && !self.embeddings.is_empty() {
            for _axis in 0..3 {
                self.positions.extend_from_slice(&self.row_positions);
            }
            self.positions.extend(std::iter::repeat_n(0, rows));
        } else {
            self.positions.extend_from_slice(&self.row_positions);
        }
        self.n_seq_id.clear();
        self.n_seq_id.resize(rows, 1);
        self.seq_id_ptrs.clear();
        let seq_base = self.seq_ids.as_mut_ptr();
        self.seq_id_ptrs
            .extend((0..rows).map(|row| unsafe { seq_base.add(row) }));
        LlamaBatch {
            n_tokens: i32::try_from(rows).expect("draft batch rows fit in i32"),
            token: if self.tokens.is_empty() {
                std::ptr::null_mut()
            } else {
                self.tokens.as_mut_ptr()
            },
            embd: if self.embeddings.is_empty() {
                std::ptr::null_mut()
            } else {
                self.embeddings.as_mut_ptr()
            },
            pos: self.positions.as_mut_ptr(),
            n_seq_id: self.n_seq_id.as_mut_ptr(),
            seq_id: self.seq_id_ptrs.as_mut_ptr(),
            logits: self.outputs.as_mut_ptr(),
        }
    }
}

/// Concatenates one batch row's input across every extracted target layer.
///
/// `layers[k]` is the captured input of the k-th extracted layer for the last
/// target decode, `n_embd` floats per batch row.
pub(crate) fn target_feature_row<'a>(
    layers: &'a [&'a [f32]],
    n_embd: usize,
    row: usize,
) -> impl Iterator<Item = f32> + 'a {
    layers
        .iter()
        .flat_map(move |layer| layer[row * n_embd..(row + 1) * n_embd].iter().copied())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn feature_rows_concatenate_layers_in_extraction_order() {
        let layer_a = [1.0, 2.0, 3.0, 4.0];
        let layer_b = [10.0, 20.0, 30.0, 40.0];
        let layers = [&layer_a[..], &layer_b[..]];

        assert_eq!(
            target_feature_row(&layers, 2, 1).collect::<Vec<_>>(),
            vec![3.0, 4.0, 30.0, 40.0]
        );
    }

    #[test]
    fn mrope_embedding_batches_expand_positions_to_four_axes() {
        let mut batch = DraftBatch::default();
        batch.push_features([0.0, 0.0], 7, 1);
        batch.push_features([0.0, 0.0], 8, 1);

        let raw = batch.as_raw(true);

        assert_eq!(raw.n_tokens, 2);
        assert!(raw.token.is_null());
        let positions = unsafe { std::slice::from_raw_parts(raw.pos, 8) };
        assert_eq!(positions, &[7, 8, 7, 8, 7, 8, 0, 0]);
        let seq = unsafe { **raw.seq_id.add(1) };
        assert_eq!(seq, 1);
    }

    #[test]
    fn token_batches_keep_one_position_per_row_and_mark_outputs() {
        let mut batch = DraftBatch::default();
        batch.push_token(5, 3, 0, false);
        batch.push_token(9, 4, 0, true);

        let raw = batch.as_raw(true);

        assert!(raw.embd.is_null());
        let positions = unsafe { std::slice::from_raw_parts(raw.pos, 2) };
        assert_eq!(positions, &[3, 4]);
        let outputs = unsafe { std::slice::from_raw_parts(raw.logits, 2) };
        assert_eq!(outputs, &[0, 1]);

        batch.clear();
        assert_eq!(batch.as_raw(false).n_tokens, 0);
    }
}
