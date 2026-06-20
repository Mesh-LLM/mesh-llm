use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::Path;

use anyhow::{Context, Result, ensure};
use serde_json::Value;

use crate::gguf_writer::GgufKv;

const TOKEN_TYPE_NORMAL: i32 = 1;
const TOKEN_TYPE_CONTROL: i32 = 3;

pub(crate) fn push_tokenizer_metadata(
    metadata: &mut Vec<GgufKv>,
    source: &Path,
    config: &Value,
) -> Result<()> {
    let tokenizer_path = source.join("tokenizer.json");
    if !tokenizer_path.exists() {
        return Ok(());
    }
    let tokenizer: Value = serde_json::from_slice(
        &fs::read(&tokenizer_path).with_context(|| format!("read {}", tokenizer_path.display()))?,
    )
    .with_context(|| format!("parse {}", tokenizer_path.display()))?;
    let tokenizer_config = read_optional_json(&source.join("tokenizer_config.json"))?;
    let vocab = read_byte_level_bpe(&tokenizer, config)?;

    metadata.push(GgufKv::string("tokenizer.ggml.model", "gpt2"));
    metadata.push(GgufKv::string("tokenizer.ggml.pre", tokenizer_pre(config)?));
    metadata.push(GgufKv::array_string("tokenizer.ggml.tokens", vocab.tokens));
    metadata.push(GgufKv::array_i32(
        "tokenizer.ggml.token_type",
        vocab.token_types,
    ));
    metadata.push(GgufKv::array_f32("tokenizer.ggml.scores", vocab.scores));
    if !vocab.merges.is_empty() {
        metadata.push(GgufKv::array_string("tokenizer.ggml.merges", vocab.merges));
    }
    push_special_token_ids(metadata, config, &tokenizer_config, &vocab.added_tokens);
    push_chat_template(metadata, source, &tokenizer_config)?;
    Ok(())
}

pub(crate) fn ensure_native_tokenizer_metadata_supported(source: &Path) -> Result<()> {
    if source.join("tokenizer.json").exists() {
        return Ok(());
    }
    if source.join("tokenizer.model").exists() {
        anyhow::bail!(
            "native tokenizer metadata does not yet support SentencePiece tokenizer.model; use external convert_hf_to_gguf.py for this checkpoint"
        );
    }
    anyhow::bail!(
        "native tokenizer metadata requires tokenizer.json; use external convert_hf_to_gguf.py for this checkpoint"
    )
}

struct BpeVocabMetadata {
    tokens: Vec<String>,
    token_types: Vec<i32>,
    scores: Vec<f32>,
    merges: Vec<String>,
    added_tokens: BTreeMap<String, u32>,
}

fn read_byte_level_bpe(tokenizer: &Value, config: &Value) -> Result<BpeVocabMetadata> {
    let model = tokenizer
        .get("model")
        .and_then(Value::as_object)
        .context("tokenizer.json missing object field model")?;
    ensure!(
        model.get("type").and_then(Value::as_str) == Some("BPE"),
        "native tokenizer metadata currently supports tokenizer.json model.type=BPE only"
    );
    ensure!(
        tokenizer
            .get("decoder")
            .and_then(|decoder| decoder.get("type"))
            .and_then(Value::as_str)
            == Some("ByteLevel"),
        "native tokenizer metadata currently supports ByteLevel BPE decoders only"
    );
    let raw_vocab = model
        .get("vocab")
        .and_then(Value::as_object)
        .context("tokenizer.json model missing object field vocab")?;
    let vocab_size = config_vocab_size(config).unwrap_or(raw_vocab.len());
    let mut tokens = vec![String::new(); vocab_size];
    let mut token_types = vec![TOKEN_TYPE_NORMAL; vocab_size];
    let mut scores = vec![0.0_f32; vocab_size];
    for (token, id) in raw_vocab {
        let id = u32_value(id).with_context(|| format!("invalid vocab id for token {token:?}"))?;
        let index = usize::try_from(id).context("vocab id does not fit usize")?;
        ensure!(
            index < tokens.len(),
            "token {token:?} id {id} is outside configured vocab_size {vocab_size}"
        );
        tokens[index] = token.clone();
    }

    let added_tokens = collect_added_tokens(tokenizer);
    for added in &added_tokens {
        let index = usize::try_from(added.id).context("added token id does not fit usize")?;
        ensure!(
            index < tokens.len(),
            "added token {:?} id {} is outside configured vocab_size {vocab_size}",
            added.content,
            added.id
        );
        tokens[index] = added.content.clone();
        scores[index] = -1000.0;
        if added.special {
            token_types[index] = TOKEN_TYPE_CONTROL;
        }
    }

    let missing = tokens.iter().position(String::is_empty);
    ensure!(
        missing.is_none(),
        "tokenizer vocab has a gap at token id {}",
        missing.unwrap_or_default()
    );
    Ok(BpeVocabMetadata {
        tokens,
        token_types,
        scores,
        merges: normalize_merges(model)?,
        added_tokens: added_tokens
            .into_iter()
            .map(|token| (token.content, token.id))
            .collect(),
    })
}

#[derive(Debug)]
struct AddedToken {
    id: u32,
    content: String,
    special: bool,
}

fn collect_added_tokens(tokenizer: &Value) -> Vec<AddedToken> {
    let mut seen = BTreeSet::new();
    let mut tokens = tokenizer
        .get("added_tokens")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(|value| {
            let id = value.get("id").and_then(u32_value)?;
            let content = value.get("content")?.as_str()?.to_string();
            let special = value
                .get("special")
                .and_then(Value::as_bool)
                .unwrap_or(false);
            Some(AddedToken {
                id,
                content,
                special,
            })
        })
        .filter(|token| seen.insert(token.id))
        .collect::<Vec<_>>();
    tokens.sort_by_key(|token| token.id);
    tokens
}

fn normalize_merges(model: &serde_json::Map<String, Value>) -> Result<Vec<String>> {
    let Some(merges) = model.get("merges").and_then(Value::as_array) else {
        return Ok(Vec::new());
    };
    merges
        .iter()
        .map(|merge| {
            if let Some(value) = merge.as_str() {
                return Ok(value.to_string());
            }
            let pair = merge
                .as_array()
                .context("tokenizer merge entries must be strings or 2-item arrays")?;
            ensure!(pair.len() == 2, "tokenizer merge pair must contain 2 items");
            let left = pair[0]
                .as_str()
                .context("tokenizer merge left item must be a string")?;
            let right = pair[1]
                .as_str()
                .context("tokenizer merge right item must be a string")?;
            Ok(format!(
                "{} {}",
                encode_merge_spaces(left),
                encode_merge_spaces(right)
            ))
        })
        .collect()
}

fn encode_merge_spaces(value: &str) -> String {
    value
        .chars()
        .map(|ch| {
            if ch == ' ' {
                char::from_u32(u32::from(ch) + 256).unwrap_or(ch)
            } else {
                ch
            }
        })
        .collect()
}

fn tokenizer_pre(config: &Value) -> Result<&'static str> {
    let model_type = config
        .get("model_type")
        .and_then(Value::as_str)
        .unwrap_or_default();
    if model_type.starts_with("glm") {
        return Ok("glm4");
    }
    if model_type.starts_with("qwen2") || model_type.starts_with("qwen3") {
        return Ok("qwen2");
    }
    if matches!(model_type, "llama" | "mistral") {
        return Ok("llama-bpe");
    }
    anyhow::bail!(
        "native tokenizer metadata needs an explicit tokenizer.ggml.pre mapping for model_type={model_type:?}"
    )
}

fn push_special_token_ids(
    metadata: &mut Vec<GgufKv>,
    config: &Value,
    tokenizer_config: &Value,
    added_tokens: &BTreeMap<String, u32>,
) {
    let model_type = config
        .get("model_type")
        .and_then(Value::as_str)
        .unwrap_or_default();
    if model_type.starts_with("glm") {
        push_added_token_id(
            metadata,
            "tokenizer.ggml.bos_token_id",
            added_tokens,
            "[gMASK]",
        );
        push_added_token_id(
            metadata,
            "tokenizer.ggml.eot_token_id",
            added_tokens,
            "<|user|>",
        );
        push_added_token_id(
            metadata,
            "tokenizer.ggml.eom_token_id",
            added_tokens,
            "<|observation|>",
        );
        push_added_token_id(
            metadata,
            "tokenizer.ggml.unknown_token_id",
            added_tokens,
            "<|endoftext|>",
        );
    }
    for (key, config_key) in [
        ("tokenizer.ggml.eos_token_id", "eos_token"),
        ("tokenizer.ggml.padding_token_id", "pad_token"),
        ("tokenizer.ggml.mask_token_id", "mask_token"),
    ] {
        if let Some(content) = tokenizer_config.get(config_key).and_then(token_content)
            && let Some(id) = added_tokens.get(content)
        {
            metadata.push(GgufKv::u32(key, *id));
        }
    }
    push_tokenizer_bool(metadata, tokenizer_config, "add_bos_token");
    push_tokenizer_bool(metadata, tokenizer_config, "add_eos_token");
}

fn push_added_token_id(
    metadata: &mut Vec<GgufKv>,
    key: &str,
    added_tokens: &BTreeMap<String, u32>,
    content: &str,
) {
    if let Some(id) = added_tokens.get(content) {
        metadata.push(GgufKv::u32(key, *id));
    }
}

fn push_chat_template(
    metadata: &mut Vec<GgufKv>,
    source: &Path,
    tokenizer_config: &Value,
) -> Result<()> {
    if let Some(template) = tokenizer_config
        .get("chat_template")
        .and_then(Value::as_str)
    {
        metadata.push(GgufKv::string("tokenizer.chat_template", template));
        return Ok(());
    }
    let template_path = source.join("chat_template.jinja");
    if template_path.exists() {
        metadata.push(GgufKv::string(
            "tokenizer.chat_template",
            &fs::read_to_string(&template_path)
                .with_context(|| format!("read {}", template_path.display()))?,
        ));
    }
    Ok(())
}

fn token_content(value: &Value) -> Option<&str> {
    value
        .as_str()
        .or_else(|| value.get("content").and_then(Value::as_str))
}

fn push_tokenizer_bool(metadata: &mut Vec<GgufKv>, tokenizer_config: &Value, config_key: &str) {
    if let Some(value) = tokenizer_config.get(config_key).and_then(Value::as_bool) {
        metadata.push(GgufKv::bool(&format!("tokenizer.ggml.{config_key}"), value));
    }
}

fn read_optional_json(path: &Path) -> Result<Value> {
    if !path.exists() {
        return Ok(Value::Null);
    }
    serde_json::from_slice(&fs::read(path).with_context(|| format!("read {}", path.display()))?)
        .with_context(|| format!("parse {}", path.display()))
}

fn config_vocab_size(config: &Value) -> Option<usize> {
    config
        .get("vocab_size")
        .and_then(Value::as_u64)
        .and_then(|value| usize::try_from(value).ok())
}

fn u32_value(value: &Value) -> Option<u32> {
    value.as_u64().and_then(|value| u32::try_from(value).ok())
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use super::*;

    #[test]
    fn builds_glm_byte_level_bpe_tokenizer_metadata() {
        let root = unique_temp_dir();
        fs::create_dir_all(&root).unwrap();
        fs::write(
            root.join("tokenizer.json"),
            r#"{
              "model": {
                "type": "BPE",
                "vocab": {"a": 0, "b": 1, "[gMASK]": 2, "<|endoftext|>": 3, "<|user|>": 4, "<|observation|>": 5},
                "merges": [["a", "b"]]
              },
              "decoder": {"type": "ByteLevel"},
              "added_tokens": [
                {"id": 2, "content": "[gMASK]", "special": true},
                {"id": 3, "content": "<|endoftext|>", "special": true},
                {"id": 4, "content": "<|user|>", "special": true},
                {"id": 5, "content": "<|observation|>", "special": true}
              ]
            }"#,
        )
        .unwrap();
        fs::write(
            root.join("tokenizer_config.json"),
            r#"{"eos_token": "<|endoftext|>", "pad_token": "<|endoftext|>"}"#,
        )
        .unwrap();
        let config: Value =
            serde_json::from_str(r#"{"model_type":"glm4_moe_lite","vocab_size":6}"#).unwrap();
        let mut metadata = Vec::new();

        push_tokenizer_metadata(&mut metadata, &root, &config).unwrap();
        let text = format!("{metadata:?}");

        assert!(text.contains("tokenizer.ggml.tokens"));
        assert!(text.contains("tokenizer.ggml.pre"));
        assert!(text.contains("glm4"));
        assert!(text.contains("tokenizer.ggml.bos_token_id"));
        assert!(text.contains("tokenizer.ggml.eot_token_id"));
        assert!(text.contains("tokenizer.ggml.eom_token_id"));
        assert!(text.contains("tokenizer.ggml.merges"));
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn builds_qwen_byte_level_bpe_tokenizer_metadata() {
        let root = unique_temp_dir();
        fs::create_dir_all(&root).unwrap();
        fs::write(
            root.join("tokenizer.json"),
            r#"{
              "model": {
                "type": "BPE",
                "vocab": {"a": 0, "b": 1, "<|endoftext|>": 2, "<|im_end|>": 3},
                "merges": ["a b"]
              },
              "decoder": {"type": "ByteLevel"},
              "added_tokens": [
                {"id": 2, "content": "<|endoftext|>", "special": true},
                {"id": 3, "content": "<|im_end|>", "special": true}
              ]
            }"#,
        )
        .unwrap();
        fs::write(
            root.join("tokenizer_config.json"),
            r#"{"eos_token": "<|im_end|>", "pad_token": "<|endoftext|>", "add_bos_token": false}"#,
        )
        .unwrap();
        let config: Value =
            serde_json::from_str(r#"{"model_type":"qwen3","vocab_size":4}"#).unwrap();
        let mut metadata = Vec::new();

        push_tokenizer_metadata(&mut metadata, &root, &config).unwrap();
        let text = format!("{metadata:?}");

        assert!(text.contains("tokenizer.ggml.pre"));
        assert!(text.contains("qwen2"));
        assert!(text.contains("tokenizer.ggml.eos_token_id"));
        assert!(text.contains("tokenizer.ggml.padding_token_id"));
        assert!(text.contains("tokenizer.ggml.add_bos_token"));
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn builds_llama_byte_level_bpe_tokenizer_metadata() {
        let root = unique_temp_dir();
        fs::create_dir_all(&root).unwrap();
        fs::write(
            root.join("tokenizer.json"),
            r#"{
              "model": {
                "type": "BPE",
                "vocab": {"a": 0, "b": 1, "<|end_of_text|>": 2},
                "merges": ["a b"]
              },
              "decoder": {"type": "ByteLevel"},
              "added_tokens": [
                {"id": 2, "content": "<|end_of_text|>", "special": true}
              ]
            }"#,
        )
        .unwrap();
        fs::write(
            root.join("tokenizer_config.json"),
            r#"{"eos_token": "<|end_of_text|>", "add_bos_token": true}"#,
        )
        .unwrap();
        let config: Value =
            serde_json::from_str(r#"{"model_type":"llama","vocab_size":3}"#).unwrap();
        let mut metadata = Vec::new();

        push_tokenizer_metadata(&mut metadata, &root, &config).unwrap();
        let text = format!("{metadata:?}");

        assert!(text.contains("tokenizer.ggml.pre"));
        assert!(text.contains("llama-bpe"));
        assert!(text.contains("tokenizer.ggml.eos_token_id"));
        assert!(text.contains("tokenizer.ggml.add_bos_token"));
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn rejects_missing_tokenizer_json() {
        let root = unique_temp_dir();
        fs::create_dir_all(&root).unwrap();
        let config: Value = serde_json::from_str(r#"{"model_type":"qwen3"}"#).unwrap();
        let mut metadata = Vec::new();

        push_tokenizer_metadata(&mut metadata, &root, &config).unwrap();
        let error = ensure_native_tokenizer_metadata_supported(&root)
            .unwrap_err()
            .to_string();

        assert!(error.contains("requires tokenizer.json"));
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn rejects_sentencepiece_tokenizer_model_without_tokenizer_json() {
        let root = unique_temp_dir();
        fs::create_dir_all(&root).unwrap();
        fs::write(root.join("tokenizer.model"), b"not-a-real-spm").unwrap();
        let config: Value = serde_json::from_str(r#"{"model_type":"llama"}"#).unwrap();
        let mut metadata = Vec::new();

        push_tokenizer_metadata(&mut metadata, &root, &config).unwrap();
        let error = ensure_native_tokenizer_metadata_supported(&root)
            .unwrap_err()
            .to_string();

        assert!(error.contains("SentencePiece tokenizer.model"));
        fs::remove_dir_all(root).unwrap();
    }

    fn unique_temp_dir() -> PathBuf {
        static NEXT_ID: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let id = NEXT_ID.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        std::env::temp_dir().join(format!("skippy-tokenizer-metadata-{nanos}-{id}"))
    }
}
