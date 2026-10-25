//! Token ids checked against the model vocabulary before native lookups.
//!
//! The native detokenizer indexes the vocabulary with every id it is given
//! and throws for an id outside it. The exception cannot cross the C ABI into
//! Rust, so one bad id aborted the whole process. Ids can come from API
//! callers and from peer stages, so they are range-checked here first.

use anyhow::{Result, anyhow, bail};
use skippy_ffi::Model as RawModel;

/// Number of tokens in the model's vocabulary; valid ids are `0..size`.
pub(crate) fn vocabulary_size(raw: *mut RawModel) -> Result<usize> {
    if raw.is_null() {
        bail!("model is not loaded");
    }
    // SAFETY: `raw` is a live Skippy model; the llama model and vocabulary it
    // returns are owned by it and only read here.
    let size = unsafe {
        let model = skippy_ffi::skippy_model_llama_model(raw);
        if model.is_null() {
            bail!("model has no native llama model");
        }
        let vocab = skippy_ffi::llama_model_get_vocab(model);
        if vocab.is_null() {
            bail!("model has no vocabulary");
        }
        skippy_ffi::llama_vocab_n_tokens(vocab)
    };
    usize::try_from(size).map_err(|_| anyhow!("model reported a negative vocabulary size"))
}

/// Fails if any id is negative or not below `vocabulary_size`.
pub(crate) fn ensure_in_vocabulary(tokens: &[i32], vocabulary_size: usize) -> Result<()> {
    let outside = tokens
        .iter()
        .find(|&&token| usize::try_from(token).map_or(true, |id| id >= vocabulary_size));
    if let Some(token) = outside {
        bail!("token id {token} is outside the vocabulary of {vocabulary_size} tokens");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn accepts_ids_inside_the_vocabulary() {
        ensure_in_vocabulary(&[0, 1, 31_999], 32_000).unwrap();
        ensure_in_vocabulary(&[], 0).unwrap();
    }

    #[test]
    fn rejects_negative_and_out_of_range_ids() {
        for token in [-1, i32::MIN, 32_000, i32::MAX] {
            let error = ensure_in_vocabulary(&[5, token], 32_000).unwrap_err();
            assert!(
                error.to_string().contains("outside the vocabulary"),
                "token={token}: {error}"
            );
        }
    }
}
