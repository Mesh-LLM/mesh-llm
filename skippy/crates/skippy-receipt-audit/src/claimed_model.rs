//! The claimed GGUF as an audit needs it: one layer at a time for the glue
//! check, and the vocabulary for the token check. The native loader
//! ([`crate::native_model`]) is the one a client runs; tests give a toy.

use std::path::PathBuf;

use anyhow::Result;
use skippy_receipts::glue::GlueEvaluator;

/// Loads the claimed model the native loader reads from a path: the GGUF,
/// or its sparse copy.
pub type LoadClaimedModel<'a> = &'a dyn Fn(PathBuf) -> Box<dyn ClaimedModel>;

/// The ggml types of the KV cache a stage stored one layer's rows in, by
/// the layer's stored rows: K, then V. `None` keeps the default. The
/// auditor builds the layer with them. Only the types an audit covers
/// reach it, f16 and f32 ([`skippy_receipts::cache_types::audited`]): a
/// receipt from a quantized cache is not audited.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct CacheTypes {
    pub k: Option<u32>,
    pub v: Option<u32>,
}

/// The claimed GGUF on this machine, as the audit needs it.
pub trait ClaimedModel {
    /// Runs `check` with one layer of the model loaded alone, with a KV
    /// cache of `cache` types.
    fn with_layer(
        &self,
        layer: u32,
        cache: CacheTypes,
        check: &mut dyn FnMut(&dyn GlueEvaluator) -> Result<()>,
    ) -> Result<()>;
    /// The text of `tokens`.
    fn detokenize(&self, tokens: &[i32]) -> Result<String>;
    /// The tokens of `text`, with special tokens parsed and none added, as
    /// the runtime reads a tag of its reasoning-budget sampler.
    fn tokenize(&self, text: &str) -> Result<Vec<i32>>;
    /// The bytes of the text of one token. A control token gives its name.
    fn token_piece(&self, token: i32) -> Result<Vec<u8>>;
    /// Whether `token` ends generation, so it adds no text to a response.
    fn is_end_of_generation(&self, token: i32) -> Result<bool>;
    /// The runtime the glue evaluation runs, as a receipt names it.
    fn runtime(&self) -> String;
}
