//! BigHead hidden-state provider over a llama context.
//!
//! [`HiddenStateProvider`](super::bighead_serve::HiddenStateProvider) needs one
//! hidden state per option line's final token plus one for the `<answer>`
//! marker. Per-token final hidden states come from a prefill pass:
//! `llama_set_embeddings(ctx, true)`, decode the prompt in batch-sized chunks,
//! and read rows of the last decoded batch with `llama_get_embeddings_ith`
//! (the pattern `media.rs` uses for speech synthesis). Rows are `n_embd` wide
//! via `llama_model_n_embd`.
//!
//! The seam stays testable without a loaded model: row access is a
//! [`EmbeddingsSource`] trait, chunk prefill is an injected closure, and
//! tokenization is an injected closure, so the position-resolution and
//! batching math run against synthetic callbacks in tests.

use super::bighead_serve::{HiddenStateProvider, RenderedQuestion};
use super::DecisionError;
use crate::native;
use crate::{StageModel, StageSession};

/// The `<answer>` marker [`super::bighead_serve::render`] closes every prompt
/// with; the provider resolves its token position.
const ANSWER_MARKER: &str = "<answer>";

/// Token indices of the hidden states one question reads.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct QuestionPositions {
    /// Token index of the `<answer>` marker's final token (the query side).
    pub answer: usize,
    /// Token index of the final token of each option's `- option` line, in
    /// option order.
    pub options: Vec<usize>,
}

/// Per-row hidden states of the last decoded batch
/// (`llama_get_embeddings_ith` semantics).
pub trait EmbeddingsSource {
    fn n_embd(&self) -> Result<usize, DecisionError>;
    /// Row `index` of the last decoded batch; `index` follows
    /// `llama_get_embeddings_ith` (negative counts from the batch end, though
    /// this provider only uses non-negative indices).
    fn row(&self, index: i32) -> Result<Vec<f32>, DecisionError>;
}

/// Maps each rendered question's option lines and `<answer>` marker to token
/// indices. `tokenize` is injected (native tokenizer in production, a fake in
/// tests) and receives the full prompt once per question plus each option line
/// standalone.
///
/// Positions are found by exact subsequence match: the full prompt is
/// tokenized with specials; each `- {label}\n` line is tokenized without
/// specials and located after the previous option's match. The `<answer>`
/// marker closes the prompt, so the query position is simply the final prompt
/// token. A line whose tokens never appear (tokenizer boundary effects) is an
/// invalid request, not a guessed position.
type QuestionPositionsResult = Result<QuestionPositions, DecisionError>;

type TokenizeFn<'a> = dyn FnMut(&str, bool) -> Result<Vec<i32>, DecisionError> + 'a;

fn resolve_one(
    question: &RenderedQuestion,
    tokenize: &mut TokenizeFn<'_>,
) -> QuestionPositionsResult {
            let full = tokenize(&question.prompt, true)?;
            if full.is_empty() {
                return Err(DecisionError::InvalidRequest(
                    "BigHead prompt produced no tokens".into(),
                ));
            }
            if !question.prompt.ends_with(ANSWER_MARKER) {
                return Err(DecisionError::Backend(anyhow::anyhow!(
                    "rendered BigHead prompt is missing the {ANSWER_MARKER} marker"
                )));
            }
            if question.option_labels.len() != question.n_options {
                return Err(DecisionError::Backend(anyhow::anyhow!(
                    "rendered question {:?} has {} labels for {} options",
                    question.kind,
                    question.option_labels.len(),
                    question.n_options
                )));
            }
            let answer = full.len() - 1;
            let mut options = Vec::with_capacity(question.option_labels.len());
            let mut cursor = 0usize;
            for label in &question.option_labels {
                let line = format!("- {label}\n");
                let line_tokens = tokenize(&line, false)?;
                let Some(start) = find_subsequence(&full[cursor..], &line_tokens) else {
                    return Err(DecisionError::InvalidRequest(format!(
                        "tokenizer could not place the option line {line:?} in the BigHead prompt"
                    )));
                };
                let base = cursor + start;
                options.push(base + line_tokens.len() - 1);
                cursor = base + line_tokens.len();
            }
            Ok(QuestionPositions { answer, options })
}

pub fn resolve_positions(
    rendered: &[RenderedQuestion],
    tokenize: &mut TokenizeFn<'_>,
) -> Result<Vec<QuestionPositions>, DecisionError> {
    rendered
        .iter()
        .map(|question| resolve_one(question, tokenize))
        .collect()
}

fn find_subsequence(haystack: &[i32], needle: &[i32]) -> Option<usize> {
    if needle.is_empty() || needle.len() > haystack.len() {
        return None;
    }
    haystack
        .windows(needle.len())
        .position(|window| window == needle)
}

/// Prefills `prompt_tokens` in `batch_size` chunks and reads the hidden-state
/// rows at `positions`, returning them as `[options..., answer]`.
///
/// Rows are read after each chunk decode (llama keeps only the last batch's
/// embeddings), so every target token is captured however the prompt spans
/// batches. `prefill_chunk` is injected (one session chunk decode in
/// production, a recorder in tests).
pub fn collect_hidden_states(
    source: &dyn EmbeddingsSource,
    prompt_tokens: &[i32],
    positions: &QuestionPositions,
    batch_size: usize,
    mut prefill_chunk: impl FnMut(&[i32]) -> Result<(), DecisionError>,
) -> Result<Vec<Vec<f32>>, DecisionError> {
    let n_embd = source.n_embd()?;
    let batch_size = batch_size.max(1);
    let mut targets: Vec<usize> = positions
        .options
        .iter()
        .copied()
        .chain([positions.answer])
        .collect();
    // Read rows in ascending token order as chunks decode; `order` maps the
    // sorted read order back to the caller's [options..., answer] order.
    let mut order: Vec<usize> = (0..targets.len()).collect();
    order.sort_by_key(|&index| targets[index]);
    targets.sort_unstable();

    let mut rows: Vec<Option<Vec<f32>>> = vec![None; targets.len()];
    let mut next_target = 0usize;
    let mut offset = 0usize;
    for chunk in prompt_tokens.chunks(batch_size) {
        prefill_chunk(chunk)?;
        while next_target < targets.len() && targets[next_target] < offset + chunk.len() {
            let position = targets[next_target];
            // Positions are sorted, so anything before `offset` was consumed
            // by an earlier chunk; a position at or past the prompt length is
            // caught by the unfinished-targets check below.
            let row_index = i32::try_from(position - offset)
                .map_err(|_| DecisionError::Backend(anyhow::anyhow!("token position overflows i32")))?;
            let row = source.row(row_index)?;
            if row.len() != n_embd {
                return Err(DecisionError::Backend(anyhow::anyhow!(
                    "hidden state row has {} floats, expected n_embd {n_embd}",
                    row.len()
                )));
            }
            rows[order[next_target]] = Some(row);
            next_target += 1;
        }
        offset += chunk.len();
    }
    let Some(position) = targets.get(next_target) else {
        return Ok(rows
            .into_iter()
            .map(|row| row.expect("every target row was read"))
            .collect());
    };
    Err(DecisionError::Backend(anyhow::anyhow!(
        "BigHead hidden-state target at token {position} was never prefilled \
         (prompt has {} tokens)",
        prompt_tokens.len()
    )))
}

/// [`EmbeddingsSource`] over a live llama context, matching the media.rs
/// speech pattern: `llama_get_embeddings_ith` rows read as `n_embd` floats.
struct LlamaEmbeddingsSource {
    context: *mut skippy_ffi::Opaque,
    n_embd: usize,
}

impl EmbeddingsSource for LlamaEmbeddingsSource {
    fn n_embd(&self) -> Result<usize, DecisionError> {
        Ok(self.n_embd)
    }

    fn row(&self, index: i32) -> Result<Vec<f32>, DecisionError> {
        // SAFETY: `context` is borrowed from the session held exclusively by
        // the caller of `LlamaHiddenStateProvider::hidden_states`; the row
        // pointer is owned by the context and copied out immediately.
        let pointer =
            unsafe { skippy_ffi::llama_get_embeddings_ith(self.context, index) }.cast_const();
        if pointer.is_null() {
            return Err(DecisionError::Backend(anyhow::anyhow!(
                "llama context returned no hidden state for row {index} (is embeddings mode on?)"
            )));
        }
        // SAFETY: llama.cpp keeps the embeddings buffer alive on the context
        // until the next decode; `n_embd` matches the model's embedding width.
        Ok(unsafe { std::slice::from_raw_parts(pointer, self.n_embd) }.to_vec())
    }
}

/// Restores logits output after hidden-state reads on every exit path.
struct EmbeddingsGuard {
    context: *mut skippy_ffi::Opaque,
}

impl Drop for EmbeddingsGuard {
    fn drop(&mut self) {
        // SAFETY: the borrowed session outlives the guard and owns the context.
        unsafe { skippy_ffi::llama_set_embeddings(self.context, false) };
    }
}

/// Hidden-state provider over a loaded model's llama context.
///
/// Captures the native handles for the duration of one serving call and reads
/// per-token hidden states after a chunked prefill. Construction validates
/// that the session exposes a llama context and the model a positive embedding
/// width, so a mismatched runtime fails at attach time rather than mid-read.
pub struct LlamaHiddenStateProvider {
    model: *mut skippy_ffi::Model,
    session: *mut StageSession,
    context: *mut skippy_ffi::Opaque,
    n_embd: usize,
}

impl LlamaHiddenStateProvider {
    /// Captures the handles backing `model`/`session`. The caller must hold
    /// exclusive access to the session for as long as the returned provider is
    /// used — the same contract as the media.rs speech path.
    pub fn capture(model: &StageModel, session: &mut StageSession) -> Result<Self, DecisionError> {
        // SAFETY: reads handles owned by the borrowed model/session; both
        // outlive the returned provider under the exclusivity contract above.
        let context = unsafe { skippy_ffi::skippy_session_llama_context(session.raw) };
        if context.is_null() {
            return Err(DecisionError::Backend(anyhow::anyhow!(
                "session did not expose a llama context for BigHead hidden states"
            )));
        }
        // SAFETY: `model` is borrowed and alive; skippy_model_llama_model only
        // reads the wrapper's inner pointer.
        let llama_model = unsafe { skippy_ffi::skippy_model_llama_model(model.raw_model()) };
        if llama_model.is_null() {
            return Err(DecisionError::Backend(anyhow::anyhow!(
                "model did not expose a llama model for BigHead hidden states"
            )));
        }
        // SAFETY: `llama_model` is non-null and owned by the borrowed model.
        let n_embd = unsafe { skippy_ffi::llama_model_n_embd(llama_model) };
        if n_embd <= 0 {
            return Err(DecisionError::Backend(anyhow::anyhow!(
                "llama model reported a non-positive embedding width: {n_embd}"
            )));
        }
        Ok(Self {
            model: model.raw_model(),
            session,
            context,
            n_embd: n_embd as usize,
        })
    }
}

// SAFETY: the provider is used under the StageSession's exclusive-access
// contract (one serving call at a time), exactly like the raw-pointer media
// paths; the pointed-to handles are not otherwise shared.
unsafe impl Send for LlamaHiddenStateProvider {}
unsafe impl Sync for LlamaHiddenStateProvider {}

impl HiddenStateProvider for LlamaHiddenStateProvider {
    fn hidden_states(
        &self,
        rendered: &mut [RenderedQuestion],
    ) -> Result<Vec<Vec<Vec<f32>>>, DecisionError> {
        if rendered.is_empty() {
            return Ok(Vec::new());
        }
        // SAFETY: exclusive session access per the capture contract; the
        // session borrow is reassembled only for the duration of this call.
        let session = unsafe { &mut *self.session };
        let batch_size = session.batch_size().map_err(DecisionError::Backend)?;
        session.reset().map_err(DecisionError::Backend)?;
        // SAFETY: `context` belongs to the exclusively held session.
        unsafe { skippy_ffi::llama_set_embeddings(self.context, true) };
        let _guard = EmbeddingsGuard {
            context: self.context,
        };

        let mut tokenize = |text: &str, add_special: bool| {
            // SAFETY: `self.model` outlives the closure call under the
            // capture contract; the native call copies out the tokens.
            native::tokenize(self.model, text, add_special).map_err(DecisionError::Backend)
        };
        let positions = resolve_positions(rendered, &mut tokenize)?;
        let source = LlamaEmbeddingsSource {
            context: self.context,
            n_embd: self.n_embd,
        };
        let mut states = Vec::with_capacity(rendered.len());
        for (question, positions) in rendered.iter_mut().zip(positions) {
            // Each question prefills independently: reset clears the previous
            // question's KV so its hidden states never bleed into this read.
            session.reset().map_err(DecisionError::Backend)?;
            let prompt_tokens = tokenize(&question.prompt, true)?;
            states.push(collect_hidden_states(
                &source,
                &prompt_tokens,
                &positions,
                batch_size,
                |chunk| session.prefill_chunk(chunk).map_err(DecisionError::Backend),
            )?);
            question.answer_pos = positions.answer;
            question.option_positions = positions.options;
        }
        Ok(states)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// One hidden state per absolute token position, deterministic: row p is
    /// `vec![p as f32; n_embd]`, so tests can verify which positions were read.
    struct SyntheticSource {
        n_embd: usize,
    }

    impl EmbeddingsSource for SyntheticSource {
        fn n_embd(&self) -> Result<usize, DecisionError> {
            Ok(self.n_embd)
        }

        fn row(&self, index: i32) -> Result<Vec<f32>, DecisionError> {
            Ok(vec![index as f32; self.n_embd])
        }
    }

    fn question(n_options: usize, labels: &[&str]) -> RenderedQuestion {
        RenderedQuestion {
            prompt: String::new(),
            answer_pos: usize::MAX,
            option_positions: vec![usize::MAX; n_options],
            kind: "choice".into(),
            option_labels: labels.iter().map(|label| (*label).to_string()).collect(),
            n_options,
        }
    }

    #[test]
    fn resolves_option_lines_and_answer_marker() {
        // Whitespace tokenizer: one token per word. The prompt layout renders
        // state, instructions, one "- label" line per option, then <answer>.
        let mut tokens = |text: &str, _add_special: bool| {
            Ok(text.split_whitespace().map(|word| word.len() as i32).collect())
        };
        let mut q = question(2, &["true", "false"]);
        q.prompt = "sky is blue\n\ncolor?\n- true\n- false\n<answer>".into();
        let positions = resolve_positions(&[q], &mut tokens).unwrap();
        // Tokens: [sky, is, blue, color?, -, true, -, false, <answer>]
        assert_eq!(positions[0].options, vec![5, 7]);
        assert_eq!(positions[0].answer, 8);
    }

    #[test]
    fn unplaceable_option_line_is_invalid_request() {
        // Tokenizer that never emits the "- x" line as written.
        let mut tokens = |text: &str, _add_special: bool| {
            if text.starts_with("- ") {
                return Ok(vec![-1]); // never matches the prompt's tokens
            }
            Ok(text.split_whitespace().map(|word| word.len() as i32).collect())
        };
        let mut q = question(1, &["true"]);
        q.prompt = "state\n- true\n<answer>".into();
        let error = resolve_positions(&[q], &mut tokens).unwrap_err();
        assert!(matches!(error, DecisionError::InvalidRequest(_)));
    }

    #[test]
    fn collect_reads_rows_across_batches_in_caller_order() {
        let source = SyntheticSource { n_embd: 3 };
        let positions = QuestionPositions {
            answer: 9,
            options: vec![1, 6],
        };
        let mut prefilled: Vec<Vec<i32>> = Vec::new();
        let states = collect_hidden_states(
            &source,
            &[10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
            &positions,
            4,
            |chunk| {
                prefilled.push(chunk.to_vec());
                Ok(())
            },
        )
        .unwrap();
        // Chunks of four over ten tokens: [0..4), [4..8), [8..10).
        assert_eq!(prefilled.len(), 3);
        assert_eq!(prefilled[2], vec![18, 19]);
        // [options..., answer]: row values are chunk-relative indices, so the
        // observed absolute positions are recoverable and correct.
        assert_eq!(states.len(), 3);
        assert_eq!(states[0], vec![1.0, 1.0, 1.0]); // option 0 at token 1, chunk 0 row 1
        assert_eq!(states[1], vec![2.0, 2.0, 2.0]); // option 1 at token 6, chunk 1 row 2
        assert_eq!(states[2], vec![1.0, 1.0, 1.0]); // answer at token 9, chunk 2 row 1
    }

    #[test]
    fn collect_fails_when_a_target_is_never_prefilled() {
        let source = SyntheticSource { n_embd: 3 };
        let positions = QuestionPositions {
            answer: 12, // past the end of the ten-token prompt
            options: vec![],
        };
        let error = collect_hidden_states(
            &source,
            &[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
            &positions,
            4,
            |_| Ok(()),
        )
        .unwrap_err();
        assert!(matches!(error, DecisionError::Backend(_)));
    }

    #[test]
    fn collect_rejects_wrong_width_rows() {
        struct WrongWidth;
        impl EmbeddingsSource for WrongWidth {
            fn n_embd(&self) -> Result<usize, DecisionError> {
                Ok(3)
            }
            fn row(&self, _index: i32) -> Result<Vec<f32>, DecisionError> {
                Ok(vec![0.0; 2])
            }
        }
        let positions = QuestionPositions {
            answer: 0,
            options: vec![],
        };
        let error =
            collect_hidden_states(&WrongWidth, &[0], &positions, 4, |_| Ok(())).unwrap_err();
        assert!(matches!(error, DecisionError::Backend(_)));
    }
}
