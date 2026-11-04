//! The metadata values on which the native loader, planner and graph
//! builders assert, checked in Rust before any of them sees the claimed
//! GGUF.
//!
//! A `GGML_ASSERT` or `GGML_ABORT` ends the process, and in Mesh the
//! process is the node. A node names the claimed GGUF, so each value below
//! is refused here, as no audit, instead of reaching an assert. The sites
//! are of the pinned llama.cpp. The list covers the code every model
//! reaches and the common rules of the graph builders; it does not cover
//! every builder of every architecture (see `mesh/docs/design/RECEIPTS.md`,
//! open limitations).

use anyhow::Result;
use skippy_receipts::weights::limits::refuse;
use skippy_receipts::weights::{MetaArray, MetaValue, TokenList, WeightFile};

/// `LLAMA_MAX_LAYERS`.
const MAX_LAYERS: u64 = 512;
/// `LLAMA_MAX_EXPERTS`.
const MAX_EXPERTS: u64 = 1024;
/// The longest training context accepted. The receipts glue sizes a region
/// from it; real models train on at most about a million positions.
const MAX_CONTEXT_LENGTH: u64 = 1 << 22;
/// The tokenizers whose load and tokenization paths were checked: a
/// SentencePiece vocabulary, a byte-level BPE one, and Gemma 4's.
const TOKENIZERS: [&str; 3] = ["llama", "gpt2", "gemma4"];
/// `LLAMA_ROPE_SCALING_TYPES`.
const ROPE_SCALING_TYPES: [&str; 4] = ["none", "linear", "yarn", "longrope"];
/// `LLM_CLS_ACT_TYPES_FROM_STRING`.
const CLASSIFIER_ACTIVATIONS: [&str; 3] = ["gelu", "silu", "tanh"];
/// The architectures whose graph divides by the head count of layer 0.
const DIVIDES_BY_HEAD_COUNT: [&str; 3] = ["gemma2", "gemma3", "deepseek2"];
/// Gemma 3n reads the KV cache of layer 19 from layer 20 on.
const GEMMA3N_SHARED_KV_FROM: u32 = 20;

/// What the native code may be given after the checks pass.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct Checked {
    /// Tokens of the vocabulary: a token id at or past it makes the
    /// tokenizer throw where nothing catches it.
    pub(super) vocabulary: u64,
}

/// Checks the metadata of the claimed GGUF the native loader asserts on.
pub(super) fn check(weights: &WeightFile) -> Result<Checked> {
    let meta = Meta { weights };
    let arch = meta.string("general.architecture")?;
    let layers = check_layers(&meta, &arch)?;
    check_experts(&meta, &arch)?;
    check_heads(&meta, &arch, layers)?;
    check_rope(&meta, &arch)?;
    check_options(&meta, &arch)?;
    let vocabulary = check_vocabulary(&meta)?;
    Ok(Checked { vocabulary })
}

/// Checks that a stage of `layer` alone loads: a layer that reads the KV
/// cache of another layer makes the cache assert when that layer is
/// outside the stage.
pub(super) fn check_layer(weights: &WeightFile, layer: u32) -> Result<()> {
    let meta = Meta { weights };
    let arch = meta.string("general.architecture")?;
    let shared_from = match arch.as_str() {
        "gemma3n" => Some(u64::from(GEMMA3N_SHARED_KV_FROM)),
        "gemma4" => {
            let layers = meta.int(&format!("{arch}.block_count"))?.unwrap_or(0);
            let shared = meta
                .int(&format!("{arch}.attention.shared_kv_layers"))?
                .unwrap_or(0);
            Some(layers.saturating_sub(shared))
        }
        _ => None,
    };
    match shared_from {
        Some(from) if u64::from(layer) >= from => refuse(format!(
            "layer {layer} of this {arch} model reads the KV cache of an earlier layer, which a one-layer stage does not hold"
        )),
        _ => Ok(()),
    }
}

/// The metadata of the file, read for the checks.
struct Meta<'a> {
    weights: &'a WeightFile,
}

impl Meta<'_> {
    fn value(&self, key: &str) -> Option<&MetaValue> {
        self.weights.metadata.get(key)
    }

    fn string(&self, key: &str) -> Result<String> {
        match self.value(key) {
            Some(MetaValue::Str(text)) => Ok(text.clone()),
            _ => refuse(format!("its metadata has no string {key}")),
        }
    }

    fn optional_string(&self, key: &str) -> Result<Option<String>> {
        match self.value(key) {
            None => Ok(None),
            Some(MetaValue::Str(text)) => Ok(Some(text.clone())),
            Some(_) => refuse(format!("its metadata {key} is not a string")),
        }
    }

    fn int(&self, key: &str) -> Result<Option<u64>> {
        match self.value(key) {
            None => Ok(None),
            Some(MetaValue::Int(value)) => Ok(Some(*value)),
            Some(_) => refuse(format!("its metadata {key} is not an integer")),
        }
    }

    /// A value that is one integer, or one per layer: the values, or none.
    fn per_layer(&self, key: &str) -> Result<Option<Vec<i64>>> {
        if let Some(value) = self.int(key)? {
            return Ok(Some(vec![i64::try_from(value).unwrap_or(i64::MAX)]));
        }
        match self.weights.arrays.get(key) {
            Some(MetaArray::Ints(values)) => Ok(Some(values.clone())),
            Some(MetaArray::Tokens(_)) => refuse(format!("its metadata {key} is not integers")),
            // an array of no integers, which the loader refuses with an error
            None => Ok(None),
        }
    }

    fn tokens(&self) -> Result<&TokenList> {
        match self.weights.arrays.get("tokenizer.ggml.tokens") {
            Some(MetaArray::Tokens(tokens)) => Ok(tokens),
            _ => refuse("its metadata has no list of tokens"),
        }
    }
}

/// `block_count` in 1 to 512 (`llama-model.cpp`, `load_hparams`), and
/// fewer `nextn_predict_layers`, each with its tensors.
fn check_layers(meta: &Meta<'_>, arch: &str) -> Result<u64> {
    let layers = meta.int(&format!("{arch}.block_count"))?.unwrap_or(0);
    if layers == 0 || layers > MAX_LAYERS {
        return refuse(format!(
            "its block_count is {layers}, not 1 to {MAX_LAYERS}"
        ));
    }
    let nextn = meta
        .int(&format!("{arch}.nextn_predict_layers"))?
        .unwrap_or(0);
    if nextn >= layers {
        return refuse(format!(
            "its nextn_predict_layers is {nextn}, not fewer than its {layers} layers"
        ));
    }
    // without these the multi-token head's tensors become optional, and a
    // graph reads them anyway
    for layer in layers - nextn..layers {
        let name = format!("blk.{layer}.nextn.eh_proj.weight");
        if !meta.weights.tensors.contains_key(&name) {
            return refuse(format!(
                "it names {nextn} next-token layers and has no {name}"
            ));
        }
    }
    if arch == "gemma4" {
        let shared = meta
            .int(&format!("{arch}.attention.shared_kv_layers"))?
            .unwrap_or(0);
        if shared.saturating_add(2) > layers {
            return refuse(format!(
                "its {shared} layers that share a KV cache leave fewer than 2 of its {layers}"
            ));
        }
    }
    if arch == "mistral-large4" {
        return refuse(
            "a layer of mistral-large4 may read the KV cache of another, which this client cannot tell from the header",
        );
    }
    Ok(layers)
}

/// The expert counts `load_hparams` asserts on.
fn check_experts(meta: &Meta<'_>, arch: &str) -> Result<()> {
    let experts = meta.int(&format!("{arch}.expert_count"))?.unwrap_or(0);
    let used = meta
        .per_layer(&format!("{arch}.expert_used_count"))?
        .unwrap_or_default();
    let groups = meta
        .int(&format!("{arch}.expert_group_count"))?
        .unwrap_or(0);
    let groups_used = meta
        .int(&format!("{arch}.expert_group_used_count"))?
        .unwrap_or(0);
    if experts > MAX_EXPERTS {
        return refuse(format!(
            "its expert_count is {experts}, more than {MAX_EXPERTS}"
        ));
    }
    if used.iter().any(|&n| n < 0 || n > i64::from(u32::MAX)) {
        return refuse(format!("its expert_used_count {used:?} is out of range"));
    }
    let most_used = used.iter().copied().max().unwrap_or(0) as u64;
    let fits = if experts > 0 {
        most_used > 0
            && most_used <= experts
            && groups < experts
            && (groups <= 1
                || (experts.is_multiple_of(groups) && groups_used > 0 && groups_used < groups))
    } else {
        most_used == 0 && groups == 0
    };
    if !fits {
        return refuse(format!(
            "its experts do not add up: {experts} experts, at most {most_used} used, {groups} groups of which {groups_used} used"
        ));
    }
    match meta.int(&format!("{arch}.expert_gating_func"))? {
        // `deepseek2` reads none as softmax; every other graph aborts on it
        Some(0) if arch == "deepseek2" => Ok(()),
        Some(1..=4) | None => Ok(()),
        Some(other) => refuse(format!("its expert_gating_func is {other}, not 1 to 4")),
    }
}

/// The head counts: a graph divides by them, and attention needs whole
/// groups of query heads per KV head.
fn check_heads(meta: &Meta<'_>, arch: &str, layers: u64) -> Result<()> {
    let heads = meta
        .per_layer(&format!("{arch}.attention.head_count"))?
        .unwrap_or_default();
    let kv_heads = meta
        .per_layer(&format!("{arch}.attention.head_count_kv"))?
        .unwrap_or_else(|| heads.clone());
    for (what, values) in [("head_count", &heads), ("head_count_kv", &kv_heads)] {
        let too_long = values.len() > 1 && values.len() as u64 != layers;
        if too_long || values.iter().any(|&n| n < 0 || n > i64::from(u32::MAX)) {
            return refuse(format!("its attention.{what} {values:?} is out of range"));
        }
    }
    let at = |values: &[i64], layer: usize| values.get(layer).or(values.first()).copied();
    for layer in 0..usize::try_from(layers)? {
        if let (Some(q), Some(kv)) = (at(&heads, layer), at(&kv_heads, layer))
            && q > 0
            && kv > 0
            && q % kv != 0
        {
            return refuse(format!(
                "its layer {layer} has {q} query heads, not a whole number per each of its {kv} KV heads"
            ));
        }
    }
    if DIVIDES_BY_HEAD_COUNT.contains(&arch) && at(&heads, 0).unwrap_or(0) <= 0 {
        return refuse(format!(
            "its {arch} graph divides by the head count of layer 0, which is 0"
        ));
    }
    Ok(())
}

/// The head width the rope rotates within: `key_length`, or the width per
/// head.
fn head_width(meta: &Meta<'_>, arch: &str) -> Result<Option<u64>> {
    if let Some(width) = meta.int(&format!("{arch}.attention.key_length"))? {
        return Ok(Some(width));
    }
    let embedding = meta.int(&format!("{arch}.embedding_length"))?;
    let heads = meta.int(&format!("{arch}.attention.head_count"))?;
    Ok(match (embedding, heads) {
        (Some(embedding), Some(heads)) if heads > 0 => Some(embedding / heads),
        _ => None,
    })
}

/// The rope: an even number of rotated dimensions within a head, sections
/// that fit half a head, and a scaling type the loader knows.
fn check_rope(meta: &Meta<'_>, arch: &str) -> Result<()> {
    if let Some(kind) = meta.optional_string(&format!("{arch}.rope.scaling.type"))?
        && !ROPE_SCALING_TYPES.contains(&kind.as_str())
    {
        return refuse(format!("its rope scaling type {kind:?} is unknown"));
    }
    let width = head_width(meta, arch)?;
    for key in [
        format!("{arch}.rope.dimension_count"),
        format!("{arch}.rope.dimension_count_swa"),
    ] {
        if let Some(rotated) = meta.int(&key)? {
            let fits = width.is_none_or(|width| rotated <= width);
            if !rotated.is_multiple_of(2) || !fits {
                return refuse(format!(
                    "its {key} is {rotated}, not an even number within a head of {width:?}"
                ));
            }
        }
    }
    let key = format!("{arch}.rope.dimension_sections");
    let sections = match meta.weights.arrays.get(&key) {
        Some(MetaArray::Ints(sections)) => sections,
        Some(MetaArray::Tokens(_)) => return refuse(format!("its {key} is not integers")),
        None => return Ok(()),
    };
    let sum = sections
        .iter()
        .try_fold(0i64, |sum, &s| (s >= 0).then(|| sum.saturating_add(s)));
    let half = width.map(|width| i64::try_from(width / 2).unwrap_or(i64::MAX));
    match (sum, half) {
        (Some(sum), Some(half)) if sum > 0 && sum <= half => Ok(()),
        (Some(sum), None) if sum > 0 => Ok(()),
        _ => refuse(format!(
            "its rope sections {sections:?} do not fit half a head of {width:?}"
        )),
    }
}

/// Options a graph aborts on when it does not know their value, and steps
/// and lengths it divides by or sizes from.
fn check_options(meta: &Meta<'_>, arch: &str) -> Result<()> {
    if let Some(activation) = meta.optional_string(&format!("{arch}.classifier.activation"))?
        && !CLASSIFIER_ACTIVATIONS.contains(&activation.as_str())
    {
        return refuse(format!(
            "its classifier activation {activation:?} is unknown"
        ));
    }
    // a pooled model builds a pooling graph that aborts on an unknown kind,
    // and is no chat model a receipt is for
    match meta.int(&format!("{arch}.pooling_type"))? {
        None | Some(0 | 0xffff_ffff) => {}
        Some(kind) => return refuse(format!("it pools its output (pooling type {kind})")),
    }
    for key in [
        format!("{arch}.interleave_moe_layer_step"),
        format!("{arch}.attention.indexer.top_k_frequency"),
    ] {
        if meta.int(&key)? == Some(0) {
            return refuse(format!("its {key} is 0, which a graph divides by"));
        }
    }
    if let Some(context) = meta.int(&format!("{arch}.context_length"))?
        && (context == 0 || context > MAX_CONTEXT_LENGTH)
    {
        return refuse(format!(
            "its context_length is {context}, not 1 to {MAX_CONTEXT_LENGTH}"
        ));
    }
    // a norm's epsilon below zero, or not a number, fails a kernel's assert
    for (key, value) in &meta.weights.metadata {
        let is_epsilon = key.ends_with("epsilon") || key.ends_with("_eps") || key.ends_with(".eps");
        if let (true, MetaValue::Float(eps)) = (is_epsilon, value)
            && !(eps.is_finite() && *eps >= 0.0)
        {
            return refuse(format!("its {key} is {eps}"));
        }
    }
    Ok(())
}

/// The vocabulary: a tokenizer whose paths were checked, each token text
/// once, a token for every byte where the tokenizer falls back to bytes,
/// and an embedding row per token.
fn check_vocabulary(meta: &Meta<'_>) -> Result<u64> {
    let model = meta.string("tokenizer.ggml.model")?;
    if !TOKENIZERS.contains(&model.as_str()) {
        return refuse(format!(
            "its tokenizer {model:?} is not one this client checks before loading: {TOKENIZERS:?}"
        ));
    }
    let tokens = meta.tokens()?;
    if let Some(index) = tokens.duplicate {
        return refuse(format!(
            "its token {index} has the text of an earlier token"
        ));
    }
    if model == "llama" && !tokens.has_every_byte() {
        return refuse(
            "its SentencePiece vocabulary has no token for some byte, which tokenizing falls back to",
        );
    }
    if let Ok(embedding) = meta.weights.tensor("token_embd.weight")
        && embedding.rows() < tokens.count
    {
        return refuse(format!(
            "its token embedding has {} rows for {} tokens",
            embedding.rows(),
            tokens.count
        ));
    }
    Ok(tokens.count)
}

#[cfg(test)]
mod tests;
