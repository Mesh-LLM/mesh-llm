use super::family::{config_supports_mlx, detect_architecture_from_safetensors_header};
use anyhow::{bail, Context, Result};
use mlx_rs::Array;
use serde_json::Value;
use std::collections::HashMap;
use std::path::Path;

#[derive(Debug, Clone)]
pub struct TokenizerSpacingPatch {
    pub special_tokens: Vec<(String, u32)>,
    pub space_token_id: u32,
}

pub(super) struct TensorPrefixes {
    pub model: String,
    pub lm_head: Option<String>,
}

pub(super) fn tensor_prefixes(tensors: &HashMap<String, Array>) -> Result<TensorPrefixes> {
    if tensors.contains_key("model.embed_tokens.weight") {
        return Ok(TensorPrefixes {
            model: "model".to_string(),
            lm_head: Some("lm_head".to_string()),
        });
    }
    if tensors.contains_key("language_model.model.embed_tokens.weight") {
        return Ok(TensorPrefixes {
            model: "language_model.model".to_string(),
            lm_head: Some("language_model.lm_head".to_string()),
        });
    }
    bail!("unsupported MLX tensor prefix layout")
}

pub(super) fn load_all_safetensors(dir: &Path) -> Result<HashMap<String, Array>> {
    let index_path = dir.join("model.safetensors.index.json");
    if index_path.exists() {
        let index: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(&index_path)?)?;
        let weight_map = index["weight_map"]
            .as_object()
            .context("missing weight_map in index")?;
        let mut tensors = HashMap::new();
        let mut seen = std::collections::HashSet::new();
        for file in weight_map.values() {
            let file = file.as_str().context("weight_map value not a string")?;
            if seen.insert(file.to_string()) {
                tensors.extend(Array::load_safetensors(dir.join(file))?);
            }
        }
        Ok(tensors)
    } else {
        let st_path = dir.join("model.safetensors");
        if st_path.exists() {
            return Ok(Array::load_safetensors(st_path)?);
        }

        let mut shard_paths = std::fs::read_dir(dir)
            .with_context(|| format!("reading MLX model directory {}", dir.display()))?
            .filter_map(|entry| entry.ok().map(|entry| entry.path()))
            .filter(|path| {
                path.file_name()
                    .and_then(|name| name.to_str())
                    .is_some_and(|name| {
                        name.starts_with("model-") && name.ends_with(".safetensors")
                    })
            })
            .collect::<Vec<_>>();
        shard_paths.sort();
        if shard_paths.is_empty() {
            bail!("no MLX safetensors weights found in {}", dir.display());
        }

        let mut tensors = HashMap::new();
        for shard_path in shard_paths {
            tensors.extend(Array::load_safetensors(shard_path)?);
        }
        Ok(tensors)
    }
}

fn normalize_model_dir(path: &Path) -> Option<&Path> {
    if path.is_dir() {
        return Some(path);
    }
    let name = path.file_name()?.to_str()?;
    match name {
        "config.json" | "chat_template.jinja" | "tokenizer.json" | "tokenizer_config.json" => {
            path.parent()
        }
        _ if name.ends_with(".safetensors") || name == "model.safetensors.index.json" => {
            path.parent()
        }
        _ => None,
    }
}

fn has_required_model_files(dir: &Path) -> bool {
    let has_config = dir.join("config.json").exists();
    let has_tokenizer =
        dir.join("tokenizer_config.json").exists() || dir.join("tokenizer.json").exists();
    let has_sharded_weights = std::fs::read_dir(dir).ok().is_some_and(|entries| {
        entries.filter_map(|entry| entry.ok()).any(|entry| {
            entry
                .file_name()
                .to_str()
                .is_some_and(|name| name.starts_with("model-") && name.ends_with(".safetensors"))
        })
    });
    let has_weights = dir.join("model.safetensors").exists()
        || dir.join("model.safetensors.index.json").exists()
        || has_sharded_weights;
    has_config && has_tokenizer && has_weights
}

pub(super) fn read_model_config(dir: &Path) -> Option<Value> {
    let text = std::fs::read_to_string(dir.join("config.json")).ok()?;
    serde_json::from_str(&text).ok()
}

pub(super) fn patch_phi3_special_token_whitespace(tokenizer_json: &mut Value, config_json: &Value) {
    let is_phi3 = config_json
        .get("model_type")
        .and_then(|value| value.as_str())
        .is_some_and(|value| value.eq_ignore_ascii_case("phi3"));
    if is_phi3 {
        let preserve_following_whitespace = ["<|assistant|>", "<|user|>", "<|system|>", "<|end|>"];
        if let Some(tokens) = tokenizer_json
            .get_mut("added_tokens")
            .and_then(|value| value.as_array_mut())
        {
            for token in tokens {
                let should_patch = token
                    .get("content")
                    .and_then(|value| value.as_str())
                    .is_some_and(|value| preserve_following_whitespace.contains(&value));
                if should_patch {
                    token["rstrip"] = Value::Bool(false);
                }
            }
        }
    }
}

fn mistral_tokenizer_spacing_patch(
    tokenizer: &tokenizers::Tokenizer,
    tokenizer_json: &Value,
    config_json: &Value,
) -> Result<Option<TokenizerSpacingPatch>> {
    let is_mistral = config_json
        .get("model_type")
        .and_then(|value| value.as_str())
        .is_some_and(|value| value.eq_ignore_ascii_case("mistral"));
    if !is_mistral {
        return Ok(None);
    }
    let mut special_tokens = tokenizer_json
        .get("added_tokens")
        .and_then(|value| value.as_array())
        .into_iter()
        .flatten()
        .filter(|token| token.get("special").and_then(|value| value.as_bool()) == Some(true))
        .filter_map(|token| {
            Some((
                token.get("content")?.as_str()?.to_string(),
                token.get("id")?.as_u64()? as u32,
            ))
        })
        .collect::<Vec<_>>();
    if special_tokens.is_empty() {
        return Ok(None);
    }
    special_tokens.sort_by(|(lhs, _), (rhs, _)| rhs.len().cmp(&lhs.len()));
    let space_token_id = tokenizer
        .encode(" ", false)
        .map_err(|e| anyhow::anyhow!("loading mistral spacing patch: {e}"))?
        .get_ids()
        .first()
        .copied()
        .context("loading mistral spacing patch: tokenizer encoded space to zero tokens")?;
    Ok(Some(TokenizerSpacingPatch {
        special_tokens,
        space_token_id,
    }))
}

pub(super) fn load_tokenizer(
    dir: &Path,
    config_json: &Value,
) -> Result<(tokenizers::Tokenizer, Option<TokenizerSpacingPatch>)> {
    let tokenizer_path = dir.join("tokenizer.json");
    let mut tokenizer_json: Value = serde_json::from_str(
        &std::fs::read_to_string(&tokenizer_path).context("reading tokenizer.json")?,
    )
    .context("parsing tokenizer.json")?;
    patch_phi3_special_token_whitespace(&mut tokenizer_json, config_json);

    let tokenizer = tokenizers::Tokenizer::from_bytes(
        serde_json::to_vec(&tokenizer_json).context("serializing patched tokenizer.json")?,
    )
    .map_err(|e| anyhow::anyhow!("loading tokenizer: {e}"))?;
    let spacing_patch = mistral_tokenizer_spacing_patch(&tokenizer, &tokenizer_json, config_json)?;
    Ok((tokenizer, spacing_patch))
}

pub fn mlx_model_dir(path: &Path) -> Option<&Path> {
    let dir = normalize_model_dir(path)?;
    if has_required_model_files(dir) {
        Some(dir)
    } else {
        None
    }
}

pub fn is_mlx_model_dir(path: &Path) -> bool {
    let Some(dir) = mlx_model_dir(path) else {
        return false;
    };

    if let Some(config) = read_model_config(dir) {
        if config_supports_mlx(&config) {
            return true;
        }
    }

    detect_architecture_from_safetensors_header(dir).is_some()
}
