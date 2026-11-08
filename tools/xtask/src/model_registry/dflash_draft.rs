//! `models dflash-draft-convert`: converts one draft listed in
//! `mesh/crates/mesh-llm-skippy-adapter/data/dflash-drafts/manifest.json`
//! from its pinned SafeTensors to GGUF and checks the result against the
//! manifest.
//!
//! The source and target-model directories must hold the files the entry
//! lists; each is hashed before use. The output is written next to `--output`
//! and moved into place only if it hashes to the manifest's `sha256` and has
//! its `bytes`; otherwise the command fails and reports what it produced.

mod gguf;
mod laguna;
mod safetensors;
#[cfg(test)]
mod tests;
mod vocab;

use std::fs::File;
use std::io::{BufWriter, Read, Write};
use std::path::{Path, PathBuf};

use serde_json::Value as Json;
use sha2::{Digest, Sha256};

use crate::command::DynResult;

pub(super) const USAGE: &str = "models dflash-draft-convert --manifest FILE --draft NAME --source DIR [--target-model DIR] --output FILE";

/// The conversions this command performs, by the manifest's `convert.converter`.
const CONVERTERS: &[&str] = &["dflash-laguna"];

struct Args {
    manifest: PathBuf,
    draft: String,
    source: PathBuf,
    target_model: Option<PathBuf>,
    output: PathBuf,
}

fn parse(args: &[String]) -> DynResult<Args> {
    let (mut manifest, mut draft, mut source, mut target_model, mut output) =
        (None, None, None, None, None);
    let mut rest = args.iter();
    while let Some(flag) = rest.next() {
        let value = rest.next().ok_or(USAGE)?;
        let slot = match flag.as_str() {
            "--manifest" => &mut manifest,
            "--draft" => &mut draft,
            "--source" => &mut source,
            "--target-model" => &mut target_model,
            "--output" => &mut output,
            _ => return Err(USAGE.into()),
        };
        if slot.replace(value.clone()).is_some() {
            return Err(USAGE.into());
        }
    }
    Ok(Args {
        manifest: manifest.ok_or(USAGE)?.into(),
        draft: draft.ok_or(USAGE)?,
        source: source.ok_or(USAGE)?.into(),
        target_model: target_model.map(PathBuf::from),
        output: output.ok_or(USAGE)?.into(),
    })
}

fn file_sha256(path: &Path) -> DynResult<(String, u64)> {
    let mut file = File::open(path).map_err(|e| format!("open {}: {e}", path.display()))?;
    let mut hasher = Sha256::new();
    let mut buffer = vec![0u8; 1 << 20];
    let mut bytes = 0u64;
    loop {
        let read = file.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
        bytes += read as u64;
    }
    Ok((hex::encode(hasher.finalize()), bytes))
}

/// Checks that `dir` holds every file `source` lists, with its SHA-256, and
/// returns the listed names.
fn verify_source(source: &Json, dir: &Path, what: &str) -> DynResult<Vec<String>> {
    let files = source
        .get("files")
        .and_then(Json::as_array)
        .filter(|files| !files.is_empty())
        .ok_or_else(|| format!("{what} lists no files"))?;
    let mut names = Vec::new();
    for file in files {
        let path = file
            .get("path")
            .and_then(Json::as_str)
            .ok_or("source file without path")?;
        let expected = file
            .get("sha256")
            .and_then(Json::as_str)
            .ok_or("source file without sha256")?;
        if path.contains('/') || path.contains('\\') || path.starts_with('.') {
            return Err(format!("{what} file {path} is not a top-level file").into());
        }
        let (actual, _) = file_sha256(&dir.join(path))?;
        if actual != expected {
            return Err(
                format!("{what} {path} hashes to {actual}, the manifest says {expected}").into(),
            );
        }
        names.push(path.to_string());
    }
    Ok(names)
}

fn require(listed: &[String], needed: &[&str], what: &str) -> DynResult<()> {
    match needed
        .iter()
        .find(|name| !listed.iter().any(|listed| listed == *name))
    {
        Some(name) => Err(format!("{what} must list {name}").into()),
        None => Ok(()),
    }
}

pub(super) fn run(args: &[String]) -> DynResult<()> {
    if args == ["--help"] {
        println!("{USAGE}");
        return Ok(());
    }
    let args = parse(args)?;
    let manifest: Json = serde_json::from_slice(&std::fs::read(&args.manifest)?)?;
    if manifest.get("schema_version").and_then(Json::as_u64) != Some(1) {
        return Err("the draft manifest is not schema_version 1".into());
    }
    let entry = manifest
        .get("drafts")
        .and_then(Json::as_array)
        .into_iter()
        .flatten()
        .find(|entry| entry.get("name").and_then(Json::as_str) == Some(args.draft.as_str()))
        .ok_or_else(|| format!("no draft named {} in the manifest", args.draft))?;
    let converter = entry
        .pointer("/convert/converter")
        .and_then(Json::as_str)
        .unwrap_or("");
    let outtype = entry
        .pointer("/convert/outtype")
        .and_then(Json::as_str)
        .unwrap_or("");
    if !CONVERTERS.contains(&converter) || outtype != "q8_0" {
        return Err(format!("unsupported conversion {converter}/{outtype}").into());
    }
    let expected_sha256 = entry
        .pointer("/output/sha256")
        .and_then(Json::as_str)
        .ok_or("output needs sha256")?;
    let expected_bytes = entry
        .pointer("/output/bytes")
        .and_then(Json::as_u64)
        .ok_or("output needs bytes")?;

    let source_files = verify_source(
        entry.get("source").unwrap_or(&Json::Null),
        &args.source,
        "source",
    )?;
    require(
        &source_files,
        &["config.json", "model.safetensors"],
        "source",
    )?;
    let target = args
        .target_model
        .as_deref()
        .ok_or("dflash-laguna needs --target-model")?;
    let target_files = verify_source(
        entry.get("target_model").unwrap_or(&Json::Null),
        target,
        "target_model",
    )?;
    require(
        &target_files,
        &["config.json", "tokenizer.json", "tokenizer_config.json"],
        "target_model",
    )?;

    // The model is named after its source directory, as llama.cpp names it.
    let name = args
        .source
        .file_name()
        .and_then(|name| name.to_str())
        .ok_or("source directory has no name")?
        .replace('-', " ");
    let partial = args.output.with_extension("gguf.partial");
    convert(&args.source, target, &name, &partial)?;
    let (sha256, bytes) = file_sha256(&partial)?;
    println!(
        "{}",
        serde_json::json!({"draft": args.draft, "sha256": sha256, "bytes": bytes})
    );
    if sha256 != expected_sha256 || bytes != expected_bytes {
        std::fs::remove_file(&partial)?;
        return Err(format!(
            "{} converted to {sha256} ({bytes} bytes); the manifest expects {expected_sha256} ({expected_bytes} bytes)",
            args.draft
        )
        .into());
    }
    std::fs::rename(&partial, &args.output)?;
    Ok(())
}

/// Converts a Laguna DFlash drafter in `source`, with the target's tokenizer
/// in `target`, to GGUF at `output`.
pub(super) fn convert(source: &Path, target: &Path, name: &str, output: &Path) -> DynResult<()> {
    let mut checkpoint = safetensors::Safetensors::open(&source.join("model.safetensors"))?;
    let plan = laguna::plan(source, &checkpoint, name)?;
    let mut metadata = plan.metadata;
    metadata.extend(vocab::metadata(target, plan.vocab_size)?);
    // The draft's own mask token wins over the tokenizer's, in its place.
    if let Some(mask) = plan.mask_token_id {
        let key = "tokenizer.ggml.mask_token_id";
        match metadata.iter_mut().find(|(k, _)| k == key) {
            Some(entry) => entry.1 = gguf::Value::U32(mask),
            None => metadata.push((key.into(), gguf::Value::U32(mask))),
        }
    }
    let entries: Vec<gguf::TensorEntry> = plan
        .tensors
        .iter()
        .map(|(entry, _)| entry.clone())
        .collect();
    let mut out = BufWriter::new(
        File::create(output).map_err(|e| format!("create {}: {e}", output.display()))?,
    );
    out.write_all(&gguf::header(&metadata, &entries))?;
    for (entry, source) in &plan.tensors {
        let values = match source {
            laguna::Source::Tensor(name) => checkpoint.read_f32(name)?,
            laguna::Source::Stack(names) => {
                let mut values = Vec::new();
                for name in names {
                    values.extend(checkpoint.read_f32(name)?);
                }
                values
            }
            laguna::Source::Rows { name, start, rows } => {
                let all = checkpoint.read_f32(name)?;
                let width = all.len() as u64 / checkpoint.tensors()[name].shape[0];
                all[(start * width) as usize..((start + rows) * width) as usize].to_vec()
            }
        };
        gguf::write_tensor(&mut out, entry, &values)?;
    }
    out.flush()?;
    Ok(())
}
