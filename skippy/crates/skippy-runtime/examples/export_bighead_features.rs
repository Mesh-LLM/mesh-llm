//! Export Strands-rendered examples through the production Q4+LoRA hidden-state path.
//!
//! Input is JSONL with `sample_id`, `task`, `prompt`, `kind`, `option_lines`,
//! `label`, `weight`, and optional `label_dist`/`teacher`. Output is a compact
//! little-endian f32 feature stream plus JSONL row metadata and a hash manifest.

use std::fs::{self, File, OpenOptions};
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{Context, Result, anyhow, ensure};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use skippy_runtime::decision::bighead_serve::RenderedQuestion;
use skippy_runtime::{RuntimeConfig, StageModel, gguf_shard_paths};

const PROGRESS_INTERVAL: usize = 100;

struct Progress<'a> {
    examples: usize,
    float_values: u64,
    hidden_size: usize,
    records_bytes: u64,
    input_sha256: &'a str,
    model_sha256: &'a str,
    adapter_sha256: &'a str,
}

struct Args {
    model: PathBuf,
    input: PathBuf,
    output: PathBuf,
}

fn main() -> Result<()> {
    let args = parse_args()?;
    ensure!(
        args.model.is_file(),
        "model file does not exist: {}",
        args.model.display()
    );
    ensure!(
        args.input.is_file(),
        "input JSONL does not exist: {}",
        args.input.display()
    );
    let adapter = args.model.with_extension("lora.gguf");
    ensure!(
        adapter.is_file(),
        "required decision LoRA sidecar is missing: {}",
        adapter.display()
    );

    eprintln!(
        "[features] hashing model and LoRA; input={}",
        args.input.display()
    );
    let model_parts = gguf_shard_paths(&args.model).context("discover Q4 GGUF shards")?;
    ensure!(!model_parts.is_empty(), "no Q4 GGUF model parts were found");
    let model_part_digests = model_parts
        .iter()
        .map(|path| {
            Ok((
                path.file_name()
                    .unwrap_or_default()
                    .to_string_lossy()
                    .to_string(),
                sha256_file(path)?,
            ))
        })
        .collect::<Result<Vec<_>>>()?;
    let model_sha256 = combined_model_sha256(&model_part_digests);
    let adapter_sha256 = sha256_file(&adapter)?;
    let input_sha256 = sha256_file(&args.input)?;

    let manifest_path = args.output.join("manifest.json");
    let progress_path = args.output.join("progress.json");
    ensure!(
        !manifest_path.is_file(),
        "completed output already exists: {}",
        manifest_path.display()
    );
    let (mut count, mut float_count, mut hidden_size) = resume_or_initialize(
        &args.output,
        &progress_path,
        &input_sha256,
        &model_sha256,
        &adapter_sha256,
    )?;

    eprintln!("[features] loading Q4 model on GPU; ctx=3072; decision LoRA required");
    let config = RuntimeConfig {
        ctx_size: 3072,
        n_gpu_layers: -1,
        ..RuntimeConfig::default()
    };
    let model = StageModel::open_from_parts(&model_parts, &config)
        .context("load Q4 BigHead model parts")?;
    ensure!(
        model.has_bighead_lora_adapter(),
        "runtime did not attach the BigHead LoRA adapter"
    );
    eprintln!("[features] model loaded; BigHead and LoRA are attached");

    let input = BufReader::new(File::open(&args.input).context("open input JSONL")?);
    let feature_path = args.output.join("features.f32le");
    let records_path = args.output.join("records.jsonl");
    let feature_file = OpenOptions::new()
        .append(true)
        .open(&feature_path)
        .context("open feature stream for append")?;
    let records_file = OpenOptions::new()
        .append(true)
        .open(&records_path)
        .context("open record stream for append")?;
    let mut features = BufWriter::with_capacity(1024 * 1024, feature_file);
    let mut records = BufWriter::new(records_file);
    let start = Instant::now();
    let mut seen = 0usize;

    for (line_number, line) in input.lines().enumerate() {
        let line = line.with_context(|| format!("read input line {}", line_number + 1))?;
        if line.trim().is_empty() {
            continue;
        }
        seen += 1;
        if seen <= count {
            continue;
        }
        let row: Value = serde_json::from_str(&line)
            .with_context(|| format!("parse input JSON on line {}", line_number + 1))?;
        let prompt = required_string(&row, "prompt", line_number)?;
        let kind = required_string(&row, "kind", line_number)?;
        ensure!(
            matches!(kind, "noul" | "choice" | "score"),
            "line {} has unsupported kind {kind:?}",
            line_number + 1
        );
        let split = row.get("split").and_then(Value::as_str).unwrap_or("train");
        ensure!(
            matches!(split, "train" | "validation"),
            "line {} has unsupported split {split:?}",
            line_number + 1
        );
        let option_lines = required_strings(&row, "option_lines", line_number)?;
        ensure!(
            !option_lines.is_empty(),
            "line {} has no options",
            line_number + 1
        );
        let token_count = model
            .tokenize(prompt, true)
            .with_context(|| format!("tokenize sample on line {}", line_number + 1))?
            .len();
        ensure!(
            token_count <= config.ctx_size as usize,
            "line {} is {token_count} tokens, exceeding runtime context {}",
            line_number + 1,
            config.ctx_size
        );
        let n_options = option_lines.len();
        let label = row
            .get("label")
            .and_then(Value::as_u64)
            .ok_or_else(|| anyhow!("line {} has no non-negative integer label", line_number + 1))?;
        ensure!(
            label < n_options as u64,
            "line {} label is outside its option range",
            line_number + 1
        );

        let mut rendered = [RenderedQuestion {
            prompt: prompt.to_string(),
            answer_pos: usize::MAX,
            option_positions: vec![usize::MAX; n_options],
            prompt_token_count: 0,
            kind: kind.to_string(),
            option_lines: option_lines.clone(),
            n_options,
        }];
        let hidden = model
            .collect_bighead_hidden_states(&mut rendered)
            .with_context(|| format!("extract features for sample on line {}", line_number + 1))?;
        ensure!(
            hidden.len() == 1,
            "runtime returned an unexpected question count"
        );
        let states = &hidden[0];
        ensure!(
            states.len() == n_options + 1,
            "runtime returned {} rows for {} options",
            states.len(),
            n_options
        );
        let width = states.first().map(Vec::len).unwrap_or_default();
        ensure!(width > 0, "runtime returned an empty hidden state");
        ensure!(
            states.iter().all(|state| state.len() == width),
            "runtime returned inconsistent hidden widths"
        );
        if let Some(expected) = hidden_size {
            ensure!(
                expected == width,
                "hidden size changed from {expected} to {width}"
            );
        } else {
            hidden_size = Some(width);
        }

        for state in states {
            for value in state {
                features
                    .write_all(&value.to_le_bytes())
                    .context("write hidden-state feature")?;
            }
        }
        let record = json!({
            "sample_id": row.get("sample_id").cloned().unwrap_or(json!(count)),
            "task": row.get("task").cloned().unwrap_or(json!("unknown")),
            "split": split,
            "kind": kind,
            "n_options": n_options,
            "label": label,
            "weight": row.get("weight").cloned().unwrap_or(json!(1.0)),
            "label_dist": row.get("label_dist"),
            "teacher": row.get("teacher"),
            "float_offset": float_count,
            "float_count": states.len() * width,
            "prompt_tokens": rendered[0].prompt_token_count,
        });
        serde_json::to_writer(&mut records, &record).context("write record metadata")?;
        records
            .write_all(b"\n")
            .context("terminate record metadata")?;
        float_count += (states.len() * width) as u64;
        count += 1;
        if count.is_multiple_of(PROGRESS_INTERVAL) {
            features.flush().context("flush feature progress")?;
            records.flush().context("flush record progress")?;
            write_progress(
                &progress_path,
                Progress {
                    examples: count,
                    float_values: float_count,
                    hidden_size: hidden_size.unwrap_or_default(),
                    records_bytes: fs::metadata(&records_path)?.len(),
                    input_sha256: &input_sha256,
                    model_sha256: &model_sha256,
                    adapter_sha256: &adapter_sha256,
                },
            )?;
            eprintln!(
                "[features] {count} examples; {float_count} f32 values; {:.1}s",
                start.elapsed().as_secs_f64()
            );
        }
    }
    ensure!(
        seen >= count,
        "input has fewer examples than the saved progress checkpoint"
    );
    ensure!(count > 0, "input JSONL contained no examples");
    features.flush().context("flush features")?;
    records.flush().context("flush records")?;
    let hidden_size = hidden_size.ok_or_else(|| anyhow!("no hidden-state rows were written"))?;
    let manifest = json!({
        "schema_version": 1,
        "complete": true,
        "feature_dtype": "float32-le",
        "feature_row_order": "options_then_answer",
        "context_size": 3072,
        "hidden_size": hidden_size,
        "examples": count,
        "float_values": float_count,
        "features_bytes": fs::metadata(&feature_path)?.len(),
        "records_bytes": fs::metadata(&records_path)?.len(),
        "features_sha256": sha256_file(&feature_path)?,
        "records_sha256": sha256_file(&records_path)?,
        "input_sha256": input_sha256,
        "q4_model_sha256": model_sha256,
        "q4_model_parts": model_part_digests
            .iter()
            .map(|(name, digest)| json!({"file": name, "sha256": digest}))
            .collect::<Vec<_>>(),
        "decision_lora_sha256": adapter_sha256,
        "extractor": "skippy-runtime/examples/export_bighead_features.rs",
        "elapsed_seconds": start.elapsed().as_secs_f64(),
    });
    fs::write(&manifest_path, serde_json::to_vec_pretty(&manifest)?)
        .context("write completion manifest")?;
    fs::remove_file(&progress_path).context("remove completed progress checkpoint")?;
    eprintln!(
        "[features] complete: {count} examples, hidden_size={}, {:.1}s",
        hidden_size,
        start.elapsed().as_secs_f64()
    );
    Ok(())
}

fn resume_or_initialize(
    output: &Path,
    progress_path: &Path,
    input_sha256: &str,
    model_sha256: &str,
    adapter_sha256: &str,
) -> Result<(usize, u64, Option<usize>)> {
    if !output.exists() {
        fs::create_dir_all(output).context("create output directory")?;
        File::create(output.join("features.f32le"))?;
        File::create(output.join("records.jsonl"))?;
        write_progress(
            progress_path,
            Progress {
                examples: 0,
                float_values: 0,
                hidden_size: 0,
                records_bytes: 0,
                input_sha256,
                model_sha256,
                adapter_sha256,
            },
        )?;
        return Ok((0, 0, None));
    }

    ensure!(
        progress_path.is_file(),
        "output directory exists without a resumable progress file: {}",
        output.display()
    );
    let progress: Value = serde_json::from_slice(&fs::read(progress_path)?)?;
    for (key, expected) in [
        ("input_sha256", input_sha256),
        ("q4_model_sha256", model_sha256),
        ("decision_lora_sha256", adapter_sha256),
    ] {
        ensure!(
            progress.get(key).and_then(Value::as_str) == Some(expected),
            "saved progress {key} does not match current inputs"
        );
    }
    let count = progress
        .get("examples")
        .and_then(Value::as_u64)
        .ok_or_else(|| anyhow!("progress file has no valid example count"))?
        as usize;
    let float_count = progress
        .get("float_values")
        .and_then(Value::as_u64)
        .ok_or_else(|| anyhow!("progress file has no valid float count"))?;
    let records_bytes = progress
        .get("records_bytes")
        .and_then(Value::as_u64)
        .ok_or_else(|| anyhow!("progress file has no valid records byte count"))?;
    let hidden_size = progress
        .get("hidden_size")
        .and_then(Value::as_u64)
        .filter(|size| *size > 0)
        .map(|size| size as usize);
    let feature_path = output.join("features.f32le");
    let records_path = output.join("records.jsonl");
    ensure!(
        feature_path.is_file() && records_path.is_file(),
        "progress checkpoint files are missing"
    );
    ensure!(
        fs::metadata(&feature_path)?.len() >= float_count * 4,
        "feature stream is shorter than its checkpoint"
    );
    ensure!(
        fs::metadata(&records_path)?.len() >= records_bytes,
        "record stream is shorter than its checkpoint"
    );
    fs::OpenOptions::new()
        .write(true)
        .open(&feature_path)?
        .set_len(float_count * 4)?;
    fs::OpenOptions::new()
        .write(true)
        .open(&records_path)?
        .set_len(records_bytes)?;
    eprintln!("[features] resuming at {count} examples and {float_count} f32 values");
    Ok((count, float_count, hidden_size))
}

fn write_progress(path: &Path, progress: Progress<'_>) -> Result<()> {
    let progress = json!({
        "schema_version": 1,
        "examples": progress.examples,
        "float_values": progress.float_values,
        "hidden_size": progress.hidden_size,
        "records_bytes": progress.records_bytes,
        "input_sha256": progress.input_sha256,
        "q4_model_sha256": progress.model_sha256,
        "decision_lora_sha256": progress.adapter_sha256,
    });
    let temporary = path.with_extension("json.tmp");
    fs::write(&temporary, serde_json::to_vec_pretty(&progress)?)?;
    fs::rename(temporary, path).context("atomically write progress checkpoint")
}

fn parse_args() -> Result<Args> {
    let mut values = std::env::args_os().skip(1);
    let mut model = None;
    let mut input = None;
    let mut output = None;
    while let Some(flag) = values.next() {
        let value = values
            .next()
            .ok_or_else(|| anyhow!("missing value for {}", flag.to_string_lossy()))?;
        match flag.to_string_lossy().as_ref() {
            "--model" => model = Some(PathBuf::from(value)),
            "--input" => input = Some(PathBuf::from(value)),
            "--out" => output = Some(PathBuf::from(value)),
            other => return Err(anyhow!("unknown argument {other:?}")),
        }
    }
    Ok(Args {
        model: model.ok_or_else(|| anyhow!("required: --model <Q4.gguf>"))?,
        input: input.ok_or_else(|| anyhow!("required: --input <rendered.jsonl>"))?,
        output: output.ok_or_else(|| anyhow!("required: --out <new-directory>"))?,
    })
}

fn required_string<'a>(row: &'a Value, field: &str, line: usize) -> Result<&'a str> {
    row.get(field)
        .and_then(Value::as_str)
        .ok_or_else(|| anyhow!("line {} is missing string field {field:?}", line + 1))
}

fn required_strings(row: &Value, field: &str, line: usize) -> Result<Vec<String>> {
    row.get(field)
        .and_then(Value::as_array)
        .ok_or_else(|| anyhow!("line {} is missing array field {field:?}", line + 1))?
        .iter()
        .map(|value| {
            value.as_str().map(str::to_owned).ok_or_else(|| {
                anyhow!(
                    "line {} field {field:?} must contain only strings",
                    line + 1
                )
            })
        })
        .collect()
}

fn sha256_file(path: &Path) -> Result<String> {
    use std::io::Read;
    let mut file =
        File::open(path).with_context(|| format!("open {} for hashing", path.display()))?;
    let mut hasher = Sha256::new();
    let mut buffer = [0u8; 1024 * 1024];
    loop {
        let count = file.read(&mut buffer).context("read file for hashing")?;
        if count == 0 {
            break;
        }
        hasher.update(&buffer[..count]);
    }
    Ok(hex::encode(hasher.finalize()))
}

fn combined_model_sha256(parts: &[(String, String)]) -> String {
    let mut hasher = Sha256::new();
    for (name, digest) in parts {
        hasher.update(name.as_bytes());
        hasher.update([0]);
        hasher.update(digest.as_bytes());
        hasher.update([0]);
    }
    hex::encode(hasher.finalize())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resume_truncates_uncheckpointed_feature_and_record_tails() -> Result<()> {
        let directory = tempfile::tempdir()?;
        let output = directory.path().join("features");
        let progress_path = output.join("progress.json");
        let input_hash = "input";
        let model_hash = "model";
        let adapter_hash = "adapter";

        assert_eq!(
            resume_or_initialize(
                &output,
                &progress_path,
                input_hash,
                model_hash,
                adapter_hash
            )?,
            (0, 0, None)
        );
        let feature_path = output.join("features.f32le");
        let records_path = output.join("records.jsonl");
        fs::write(&feature_path, [1u8; 16])?;
        fs::write(&records_path, b"one\n")?;
        write_progress(
            &progress_path,
            Progress {
                examples: 1,
                float_values: 4,
                hidden_size: 2,
                records_bytes: 4,
                input_sha256: input_hash,
                model_sha256: model_hash,
                adapter_sha256: adapter_hash,
            },
        )?;
        OpenOptions::new()
            .append(true)
            .open(&feature_path)?
            .write_all(&[9u8; 8])?;
        OpenOptions::new()
            .append(true)
            .open(&records_path)?
            .write_all(b"partial")?;

        assert_eq!(
            resume_or_initialize(
                &output,
                &progress_path,
                input_hash,
                model_hash,
                adapter_hash
            )?,
            (1, 4, Some(2))
        );
        assert_eq!(fs::metadata(feature_path)?.len(), 16);
        assert_eq!(fs::read(records_path)?, b"one\n");
        Ok(())
    }

    #[test]
    fn resume_refuses_changed_model_or_input_hashes() -> Result<()> {
        let directory = tempfile::tempdir()?;
        let output = directory.path().join("features");
        let progress_path = output.join("progress.json");
        resume_or_initialize(&output, &progress_path, "input", "model", "adapter")?;

        let error =
            resume_or_initialize(&output, &progress_path, "input", "other-model", "adapter");
        assert!(error.is_err());
        Ok(())
    }
}
