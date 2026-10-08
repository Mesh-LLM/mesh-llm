use super::contract::{Case, check, digest, tree};
use anyhow::{Result, bail};
use serde_json::{Value, json};
use std::{collections::BTreeMap, fs, path::Path, time::Instant};
use tokenizers::Tokenizer;
fn engine(bytes: &[u8]) -> Result<Tokenizer> {
    Tokenizer::from_bytes(bytes).map_err(|_| anyhow::anyhow!("fast tokenizer JSON unsupported"))
}
fn bounded(path: &Path) -> Result<Vec<u8>> {
    let mut f = super::read::open(path, 64 * 1024 * 1024, false)?;
    super::read::read(&mut f, 64 * 1024 * 1024)
}
pub(super) fn fast(
    source: &Path,
    output: &Path,
    expected: &str,
    cases: &[Case],
    deadline: Instant,
) -> Result<Value> {
    check(deadline)?;
    if cases.is_empty()
        || cases.len() > 128
        || cases
            .iter()
            .any(|c| c.text.len() > 65536 || c.decode_ids.len() > 16384)
    {
        bail!("finite tokenizer semantic cases required");
    }
    let original = bounded(&source.join("tokenizer.json"))?;
    super::json::unique(&original)?;
    let mut config: Value = super::json::unique(&bounded(&source.join("tokenizer_config.json"))?)?;
    if !config.is_object() {
        bail!("tokenizer configuration object required");
    }
    match fs::symlink_metadata(source.join("special_tokens_map.json")) {
        Ok(_) => {
            let special = super::json::unique(&bounded(&source.join("special_tokens_map.json"))?)?;
            for (key, value) in special
                .as_object()
                .ok_or_else(|| anyhow::anyhow!("special token map object"))?
            {
                config
                    .as_object_mut()
                    .unwrap()
                    .entry(key.clone())
                    .or_insert_with(|| value.clone());
            }
        }
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => (),
        Err(_) => bail!("special token map custody refused"),
    };
    let source_engine = engine(&original)?;
    let serialized = source_engine
        .to_string(false)
        .map_err(|_| anyhow::anyhow!("native tokenizer serialization refused"))?;
    let derived_engine = engine(serialized.as_bytes())?;
    let mut evidence = Vec::new();
    for c in cases {
        check(deadline)?;
        let before = source_engine
            .encode(c.text.as_str(), c.add_special_tokens)
            .map_err(|_| anyhow::anyhow!("source encoding refused"))?;
        let after = derived_engine
            .encode(c.text.as_str(), c.add_special_tokens)
            .map_err(|_| anyhow::anyhow!("export encoding refused"))?;
        let before_decoded = source_engine
            .decode(&c.decode_ids, c.skip_special_tokens)
            .map_err(|_| anyhow::anyhow!("source decoding refused"))?;
        let after_decoded = derived_engine
            .decode(&c.decode_ids, c.skip_special_tokens)
            .map_err(|_| anyhow::anyhow!("export decoding refused"))?;
        check(deadline)?;
        if before.get_ids() != c.expected_ids
            || digest(before_decoded.as_bytes()) != c.expected_decoded_sha256
            || before.get_ids() != after.get_ids()
            || before.get_type_ids() != after.get_type_ids()
            || before.get_attention_mask() != after.get_attention_mask()
            || before_decoded != after_decoded
        {
            bail!("native tokenizer semantic export mismatch");
        }
        evidence.push(json!({"text_sha256":digest(c.text.as_bytes()),"ids_sha256":digest(&serde_json::to_vec(before.get_ids())?),"decoded_sha256":digest(before_decoded.as_bytes()),"matched":true}));
    }
    // Engine JSON owns tokenization; no remote Python auto_map is executed/exported.
    let object = config
        .as_object_mut()
        .ok_or_else(|| anyhow::anyhow!("config object"))?;
    object.remove("auto_map");
    object.insert("tokenizer_class".into(), json!("PreTrainedTokenizerFast"));
    let template = match fs::symlink_metadata(source.join("chat_template.jinja")) {
        Ok(_) => bounded(&source.join("chat_template.jinja"))?,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => config["chat_template"]
            .as_str()
            .ok_or_else(|| anyhow::anyhow!("single explicit chat template required"))?
            .as_bytes()
            .to_vec(),
        Err(_) => bail!("chat template custody refused"),
    };
    if template.len() > 1024 * 1024 || std::str::from_utf8(&template).is_err() {
        bail!("chat template bound/encoding refused");
    }
    let outputs: BTreeMap<String, Vec<u8>> = BTreeMap::from([
        ("tokenizer.json".into(), serialized.into_bytes()),
        (
            "tokenizer_config.json".into(),
            serde_json::to_vec_pretty(&config)?,
        ),
        ("chat_template.jinja".into(), template),
    ]);
    let pins = outputs
        .iter()
        .map(|(n, b)| (n.clone(), digest(b)))
        .collect();
    let derived = tree(&pins)?;
    if derived != expected {
        bail!("explicit derived tokenizer tree pin mismatch");
    }
    check(deadline)?;
    fs::create_dir(output)?;
    for (name, bytes) in outputs {
        check(deadline)?;
        super::publish(&output.join(name), &bytes)?;
    }
    if bounded(&source.join("tokenizer.json"))? != original {
        bail!("source tokenizer changed during export");
    }
    Ok(
        json!({"producer":"tokenizers-0.23.2-native-fast-export-v1","tree_sha256":derived,"files":pins,"cases":evidence,"chat_template_rendering_qualified":false,"transformers_layout_identity_claimed":false}),
    )
}
pub(super) fn granite(
    source: &Path,
    output: &Path,
    rows: &BTreeMap<String, String>,
    expected: &str,
    deadline: Instant,
) -> Result<Value> {
    let mut selected = BTreeMap::new();
    for (name, pin) in rows {
        let basename = Path::new(name)
            .file_name()
            .and_then(|n| n.to_str())
            .ok_or_else(|| anyhow::anyhow!("basename"))?;
        if basename == "README.md" {
            continue;
        }
        if selected.insert(basename.to_string(), (name, pin)).is_some() {
            bail!("Granite flattened basename collision");
        }
    }
    let pins = selected
        .iter()
        .map(|(n, (_, p))| (n.clone(), (*p).clone()))
        .collect();
    if tree(&pins)? != expected {
        bail!("Granite complete flattened export pin mismatch");
    }
    fs::create_dir(output)?;
    for (name, (relative, pin)) in selected {
        check(deadline)?;
        copy_pin(&source.join(relative), &output.join(name), pin, deadline)?;
    }
    Ok(
        json!({"producer":"immutable-complete-snapshot-flatten-v1","tree_sha256":expected,"files":pins,"README_basename_excluded":true}),
    )
}

fn copy_pin(source: &Path, output: &Path, expected: &str, deadline: Instant) -> Result<()> {
    use sha2::Digest as _;
    use std::io::{Read as _, Write as _};
    let mut source = super::read::open(source, 1024 * 1024 * 1024 * 1024, false)?;
    let mut stage = tempfile::NamedTempFile::new_in(output.parent().unwrap())?;
    let mut hash = sha2::Sha256::new();
    let mut b = [0; 65536];
    loop {
        check(deadline)?;
        let n = source.read(&mut b)?;
        if n == 0 {
            break;
        }
        hash.update(&b[..n]);
        stage.write_all(&b[..n])?;
    }
    let observed: String = hash.finalize().iter().map(|b| format!("{b:02x}")).collect();
    if observed != expected {
        bail!("Granite source drift");
    }
    check(deadline)?;
    stage.as_file().sync_all()?;
    stage
        .persist_noclobber(output)
        .map_err(|_| anyhow::anyhow!("fresh Granite output refused"))?;
    Ok(())
}
