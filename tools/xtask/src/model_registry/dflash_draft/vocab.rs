//! The target's byte-level BPE tokenizer as llama.cpp's converter writes it
//! (`_set_vocab_gpt2` plus `SpecialVocab`), for the tokenizers we convert.
//! Anything outside that subset is refused rather than approximated.

use std::collections::{BTreeMap, HashMap};
use std::path::Path;

use serde_json::Value as Json;

use super::gguf::Value;
use crate::command::DynResult;

const NORMAL: i32 = 1;
const CONTROL: i32 = 3;
const USER_DEFINED: i32 = 4;
const UNUSED: i32 = 5;

/// `tokenizer.ggml.pre` by the target's `model_type`. llama.cpp derives it
/// from a hash of the tokenizer's output; add a model here only with a
/// conversion that checks against llama.cpp's.
fn pre_tokenizer(model_type: &str) -> DynResult<&'static str> {
    match model_type {
        "laguna" => Ok("laguna"),
        other => Err(format!("no tokenizer.ggml.pre known for model_type {other:?}").into()),
    }
}

fn read_json(path: &Path) -> DynResult<Json> {
    let bytes = std::fs::read(path).map_err(|e| format!("read {}: {e}", path.display()))?;
    Ok(serde_json::from_slice(&bytes).map_err(|e| format!("parse {}: {e}", path.display()))?)
}

fn token_content(value: &Json) -> Option<&str> {
    value
        .as_str()
        .or_else(|| value.get("content").and_then(Json::as_str))
}

/// llama.cpp treats some added tokens that are not marked special as control
/// tokens anyway.
fn looks_special(token: &str) -> bool {
    matches!(token, "<pad>" | "<mask>" | "<2mass>" | "[@BOS@]")
        || (token.starts_with("<|") && token.ends_with("|>"))
        || (token.starts_with("<\u{ff5c}") && token.ends_with("\u{ff5c}>"))
        || (token.starts_with("<unused") && token.ends_with('>'))
}

/// Tokenizer metadata in llama.cpp's key order. `target` holds the target's
/// `config.json`, `tokenizer.json` and `tokenizer_config.json`.
pub(super) fn metadata(target: &Path, vocab_size: usize) -> DynResult<Vec<(String, Value)>> {
    let config = read_json(&target.join("config.json"))?;
    let tokenizer = read_json(&target.join("tokenizer.json"))?;
    let tokenizer_config = read_json(&target.join("tokenizer_config.json"))?;
    let model_type = config
        .get("model_type")
        .and_then(Json::as_str)
        .ok_or("target config.json has no model_type")?;
    let model = tokenizer
        .get("model")
        .ok_or("tokenizer.json has no model")?;
    if model.get("type").and_then(Json::as_str) != Some("BPE")
        || tokenizer.pointer("/decoder/type").and_then(Json::as_str) != Some("ByteLevel")
    {
        return Err("only byte-level BPE tokenizers are supported".into());
    }

    // Ids to tokens: the BPE vocabulary, then the added tokens over it.
    let mut by_id: BTreeMap<usize, String> = BTreeMap::new();
    let mut ids: HashMap<String, usize> = HashMap::new();
    for (token, id) in model
        .get("vocab")
        .and_then(Json::as_object)
        .ok_or("tokenizer.json has no vocab")?
    {
        let id = id.as_u64().ok_or("non-integer token id")? as usize;
        by_id.insert(id, token.clone());
        ids.insert(token.clone(), id);
    }
    let mut added: HashMap<String, bool> = HashMap::new();
    for entry in tokenizer
        .get("added_tokens")
        .and_then(Json::as_array)
        .into_iter()
        .flatten()
    {
        let id = entry
            .get("id")
            .and_then(Json::as_u64)
            .ok_or("added token without id")? as usize;
        let content = entry
            .get("content")
            .and_then(Json::as_str)
            .ok_or("added token without content")?;
        let special = entry
            .get("special")
            .and_then(Json::as_bool)
            .unwrap_or(false);
        by_id.insert(id, content.to_string());
        ids.insert(content.to_string(), id);
        added.insert(content.to_string(), special);
    }
    if by_id
        .keys()
        .next_back()
        .is_some_and(|max| *max >= vocab_size)
    {
        return Err("tokenizer has ids beyond the model's vocab_size".into());
    }
    let mut tokens = Vec::with_capacity(vocab_size);
    let mut types = Vec::with_capacity(vocab_size);
    for id in 0..vocab_size {
        match by_id.get(&id) {
            None => {
                tokens.push(format!("[PAD{id}]"));
                types.push(UNUSED);
            }
            Some(token) => match added.get(token) {
                Some(special) if *special || looks_special(token) => {
                    tokens.push(token.clone());
                    types.push(CONTROL);
                }
                Some(_) => {
                    tokens.push(token.replace('\u{2581}', " "));
                    types.push(USER_DEFINED);
                }
                None => {
                    tokens.push(token.clone());
                    types.push(NORMAL);
                }
            },
        }
    }

    let mut merges = Vec::new();
    for merge in model
        .get("merges")
        .and_then(Json::as_array)
        .into_iter()
        .flatten()
    {
        match merge {
            Json::String(merge) => merges.push(merge.clone()),
            Json::Array(pair) => {
                let parts: Vec<&str> = pair.iter().filter_map(Json::as_str).collect();
                if parts.len() != 2 || parts.iter().any(|part| part.contains(' ')) {
                    return Err("unsupported merge in tokenizer.json".into());
                }
                merges.push(parts.join(" "));
            }
            _ => return Err("unsupported merge in tokenizer.json".into()),
        }
    }

    let mut out = vec![
        (
            "tokenizer.ggml.model".to_string(),
            Value::Str("gpt2".into()),
        ),
        (
            "tokenizer.ggml.pre".to_string(),
            Value::Str(pre_tokenizer(model_type)?.into()),
        ),
        ("tokenizer.ggml.tokens".to_string(), Value::Strs(tokens)),
        ("tokenizer.ggml.token_type".to_string(), Value::I32s(types)),
    ];
    if !merges.is_empty() {
        out.push(("tokenizer.ggml.merges".to_string(), Value::Strs(merges)));
    }

    // Special token ids: tokenizer_config.json first, then config.json for
    // any still missing. llama.cpp writes no cls id.
    let mut special: Vec<(&str, u32)> = Vec::new();
    for (kind, field) in [
        ("bos", "bos_token"),
        ("eos", "eos_token"),
        ("unk", "unk_token"),
        ("sep", "sep_token"),
        ("pad", "pad_token"),
        ("mask", "mask_token"),
    ] {
        if let Some(id) = tokenizer_config
            .get(field)
            .and_then(token_content)
            .and_then(|content| ids.get(content))
        {
            special.push((kind, *id as u32));
        }
    }
    for kind in ["bos", "eos", "unk", "sep", "pad", "mask"] {
        if special.iter().any(|(k, _)| *k == kind) {
            continue;
        }
        if let Some(id) = config
            .get(format!("{kind}_token_id"))
            .and_then(Json::as_u64)
            && (id as usize) < vocab_size
        {
            special.push((kind, id as u32));
        }
    }
    for (kind, id) in special {
        let key = match kind {
            "bos" => "bos",
            "eos" => "eos",
            "unk" => "unknown",
            "sep" => "seperator",
            "pad" => "padding",
            _ => "mask",
        };
        out.push((format!("tokenizer.ggml.{key}_token_id"), Value::U32(id)));
    }

    for (kind, add) in add_special_tokens(&tokenizer, &tokenizer_config)? {
        out.push((format!("tokenizer.ggml.add_{kind}_token"), Value::Bool(add)));
    }

    let template = match tokenizer_config.get("chat_template") {
        Some(Json::String(template)) => Some(template.clone()),
        Some(_) => return Err("only a single string chat_template is supported".into()),
        None => {
            let jinja = target.join("chat_template.jinja");
            jinja
                .is_file()
                .then(|| std::fs::read_to_string(&jinja))
                .transpose()?
        }
    };
    if let Some(template) = template {
        out.push(("tokenizer.chat_template".to_string(), Value::Str(template)));
    }
    Ok(out)
}

/// Which special tokens the tokenizer adds, read from a `TemplateProcessing`
/// post-processor as llama.cpp's `SpecialVocab` reads it, then any explicit
/// `add_*_token` in tokenizer_config.json.
fn add_special_tokens(
    tokenizer: &Json,
    tokenizer_config: &Json,
) -> DynResult<Vec<(&'static str, bool)>> {
    let mut adds: Vec<(&'static str, bool)> = Vec::new();
    let content = |field: &str| tokenizer_config.get(field).and_then(token_content);
    let (bos, eos, sep) = (
        content("bos_token"),
        content("eos_token"),
        content("sep_token"),
    );
    fn special(step: &Json) -> Option<&str> {
        step.pointer("/SpecialToken/id").and_then(Json::as_str)
    }
    fn sequence(step: &Json) -> Option<&str> {
        step.pointer("/Sequence/id").and_then(Json::as_str)
    }
    match tokenizer.get("post_processor") {
        None | Some(Json::Null) => {}
        Some(processor)
            if processor.get("type").and_then(Json::as_str) == Some("TemplateProcessing") =>
        {
            let single = processor
                .get("single")
                .and_then(Json::as_array)
                .cloned()
                .unwrap_or_default();
            let pair = processor
                .get("pair")
                .and_then(Json::as_array)
                .cloned()
                .unwrap_or_default();
            // The supported shape: [BOS?] A [EOS?] and [BOS?] A [SEP] B [EOS?].
            if single.len() < 2 {
                return Err("unsupported TemplateProcessing<single>".into());
            }
            let first = special(&single[0]);
            let last = special(&single[single.len() - 1]);
            if let Some(first) = first {
                if Some(first) != bos {
                    return Err("TemplateProcessing starts with a token other than bos".into());
                }
                adds.push(("bos", true));
            }
            if let Some(last) = last {
                if Some(last) != eos {
                    return Err("TemplateProcessing ends with a token other than eos".into());
                }
                adds.push(("eos", true));
            }
            if !pair.is_empty() {
                let start = usize::from(first.is_some() && special(&pair[0]) == first);
                let stop = if last.is_some() && special(&pair[pair.len() - 1]) == last {
                    pair.len() - 1
                } else {
                    pair.len()
                };
                let inner = &pair[start..stop];
                if inner.len() < 3
                    || sequence(&inner[0]) != Some("A")
                    || sequence(&inner[inner.len() - 1]) != Some("B")
                    || inner.len() > 4
                {
                    return Err("unsupported TemplateProcessing<pair>".into());
                }
                let middle = &inner[1..inner.len() - 1];
                let separator = |step: &Json| {
                    special(step).is_some_and(|id| Some(id) == sep || Some(id) == eos)
                };
                if middle.iter().any(|step| !separator(step)) {
                    return Err("unsupported separator in TemplateProcessing<pair>".into());
                }
                let add_sep = (last.is_none() && separator(&middle[0])) || middle.len() == 2;
                adds.push(("sep", add_sep));
            }
        }
        Some(_) => return Err("unsupported tokenizer post_processor".into()),
    }
    for kind in ["bos", "eos", "sep"] {
        if let Some(add) = tokenizer_config
            .get(format!("add_{kind}_token"))
            .and_then(Json::as_bool)
        {
            match adds.iter_mut().find(|(k, _)| *k == kind) {
                Some(entry) => entry.1 = add,
                None => adds.push((kind, add)),
            }
        }
    }
    Ok(adds)
}
