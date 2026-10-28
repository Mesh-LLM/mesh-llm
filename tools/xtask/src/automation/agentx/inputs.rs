use super::{Model, Prepared, Profile, command, write};
use crate::command::DynResult;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
pub(super) const DATA_REPO: &str = "semianalysisai/cc-traces-weka-062126-256k";
pub(super) const DATA_REVISION: &str = "8fecd2fc56694469f758f0afbbb6335ad3043740";
pub(super) const DATA_SHA: &str =
    "e39cd2ff3eba21d4a3664be51da743ac3d2149a1933898cafc7bfeac8147eeef";

pub(super) fn prepare(
    root: PathBuf,
    output: PathBuf,
    key: &str,
    profile: Profile,
) -> DynResult<Prepared> {
    let model = select_model(&root, key)?;
    let mut weights = None;
    for (index, file) in model.files.iter().enumerate() {
        let path = download(
            &root,
            &model.repo,
            &model.revision,
            &file.name,
            false,
            &output,
            &format!("download-weight-{index}"),
        )?;
        // Preparation verifies each pin once before any serving cell starts.
        verify(&path, &file.sha256)?;
        weights.get_or_insert(path);
    }
    download(
        &root,
        &model.tokenizer,
        &model.tokenizer_revision,
        "",
        false,
        &output,
        "download-tokenizer",
    )?;
    let dataset = download(
        &root,
        DATA_REPO,
        DATA_REVISION,
        "traces.jsonl",
        true,
        &output,
        "download-dataset",
    )?;
    verify(&dataset, DATA_SHA)?;
    let traces = output.join("traces");
    materialize(&dataset, &traces, profile)?;
    let prepared = Prepared {
        root,
        output,
        weights: weights.ok_or("empty model files")?,
        dataset: traces,
        model,
        profile,
    };
    write(&prepared.output.join("inputs.json"), &prepared)?;
    Ok(prepared)
}

pub(super) fn select_model(root: &Path, key: &str) -> DynResult<Model> {
    let models: Vec<Model> = serde_json::from_slice(&std::fs::read(
        root.join("skippy/evals/agentx/models.json"),
    )?)?;
    models
        .into_iter()
        .find(|model| model.key == key || model.registry_ref() == key)
        .ok_or_else(|| "unknown AgentX model".into())
}
fn download(
    root: &Path,
    repo: &str,
    revision: &str,
    file: &str,
    dataset: bool,
    output: &Path,
    log_name: &str,
) -> DynResult<PathBuf> {
    let log = output.join(log_name);
    if std::env::var("HF_HUB_OFFLINE").as_deref() == Ok("1") {
        let home = std::env::var_os("HF_HOME").ok_or("offline AgentX requires HF_HOME")?;
        let path = offline_cached_path(Path::new(&home), repo, revision, file, dataset)?;
        std::fs::write(
            log.with_extension("stdout.log"),
            format!("{}\n", path.display()),
        )?;
        std::fs::write(log.with_extension("stderr.log"), b"")?;
        return Ok(path);
    }
    let hf = root.join("skippy/evals/agentx/.venv/bin/hf");
    // Standard HF cache resolution reuses the host's blobs/snapshots and writes
    // only missing pins into that same mounted HF_HOME. Never use --local-dir.
    let mut args = command::strings(&["download", repo, "--revision", revision, "--quiet"]);
    if dataset {
        args.extend(command::strings(&["--repo-type", "dataset"]));
    }
    if !file.is_empty() {
        args.push(file.into());
    } else {
        args.extend(command::strings(&[
            "--include",
            "tokenizer*",
            "--include",
            "vocab.json",
            "--include",
            "merges.txt",
            "--include",
            "config.json",
            "--include",
            "chat_template*",
        ]));
    }
    let mut spec = command::spec(root, hf.to_str().ok_or("hf path")?, &args)?;
    spec.environment.insert(
        "HF_HUB_OFFLINE".into(),
        crate::process::Value::Public("1".into()),
    );
    command::run(&spec, 60, &log)?;
    cached_path(&std::fs::read_to_string(log.with_extension("stdout.log"))?)
}

fn offline_cached_path(
    home: &Path,
    repo: &str,
    revision: &str,
    file: &str,
    dataset: bool,
) -> DynResult<PathBuf> {
    if !home.is_absolute()
        || !revision.bytes().all(|byte| byte.is_ascii_hexdigit())
        || revision.len() != 40
        || repo.split('/').count() != 2
        || repo
            .split('/')
            .any(|part| part.is_empty() || part == "." || part == "..")
    {
        return Err("invalid pinned offline Hugging Face model reference".into());
    }
    let kind = if dataset { "datasets" } else { "models" };
    let snapshot = home
        .join("hub")
        .join(format!("{kind}--{}", repo.replace('/', "--")))
        .join("snapshots")
        .join(revision);
    if !file.is_empty() {
        let relative = Path::new(file);
        if relative.is_absolute()
            || relative
                .components()
                .any(|component| matches!(component, std::path::Component::ParentDir))
        {
            return Err("invalid pinned offline Hugging Face file path".into());
        }
        let path = snapshot.join(relative);
        if path.is_file() {
            return Ok(path);
        }
        return Err(format!(
            "pinned offline Hugging Face file is missing: {}",
            path.display()
        )
        .into());
    }
    // All pinned AgentX models use the Qwen tokenizer bundle. A directory
    // alone can be a partially downloaded snapshot, so check the common
    // files the harness needs before accepting a read-only mount.
    if [
        "config.json",
        "merges.txt",
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.json",
    ]
    .iter()
    .all(|name| snapshot.join(name).is_file())
    {
        return Ok(snapshot);
    }
    Err(format!(
        "pinned offline tokenizer snapshot is missing: {}",
        snapshot.display()
    )
    .into())
}

fn cached_path(stdout: &str) -> DynResult<PathBuf> {
    let text = stdout.trim();
    let path = PathBuf::from(text);
    if text.contains(['\n', '\r']) || !path.is_absolute() || !path.exists() {
        return Err("HF download did not return one existing absolute cache path".into());
    }
    // Retain the snapshot path rather than canonicalizing to blobs: split GGUF
    // loaders and shard verification resolve sibling files in the snapshot.
    Ok(path)
}
pub(super) fn verify(path: &Path, expected: &str) -> DynResult<()> {
    let actual = crate::product::digest::file_sha256(path).map_err(|failure| failure.error)?;
    if actual != expected {
        return Err(format!("SHA-256 mismatch for {}", path.display()).into());
    }
    Ok(())
}
fn materialize(source: &Path, destination: &Path, profile: Profile) -> DynResult<()> {
    std::fs::create_dir(destination)?;
    if profile == Profile::Comparison {
        return super::comparison_cohort::materialize(source, destination);
    }
    if profile == Profile::Contention {
        return super::contention_cohort::materialize(source, destination);
    }
    let limit = if profile == Profile::Smoke { 128 } else { 393 };
    let mut ids = Vec::new();
    let mut trace_context = Vec::new();
    for (index, line) in BufReader::new(std::fs::File::open(source)?)
        .lines()
        .take(limit)
        .enumerate()
    {
        let line = line?;
        let trace: serde_json::Value = serde_json::from_str(&line)?;
        ids.push(trace["id"].as_str().ok_or("trace ID missing")?.to_owned());
        trace_context.push(super::trace_context::describe(&trace, line.as_bytes())?);
        std::fs::write(destination.join(format!("{index:04}.json")), line)?;
    }
    if ids.len() != limit {
        return Err("AgentX corpus has fewer traces than requested".into());
    }
    write(
        &destination
            .parent()
            .ok_or("trace parent")?
            .join("cohort.json"),
        &serde_json::json!({"dataset_repo":DATA_REPO,"dataset_revision":DATA_REVISION,"dataset_sha256":DATA_SHA,"trace_ids":ids,"trace_context":trace_context}),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn smoke_preserves_complete_rows_and_selects_exact_prefix() {
        let temp = tempfile::tempdir().unwrap();
        let source = temp.path().join("corpus.jsonl");
        let rows: Vec<_> = (0..129).map(|index| format!(r#"{{"id":"trace-{index}","requests":[{{"type":"subagent","id":"child","requests":[{{"type":"s","in":120000,"out":20000}}]}},{{"type":"n","in":10,"out":100000}}]}}"#)).collect();
        std::fs::write(&source, rows.join("\n")).unwrap();
        let target = temp.path().join("selected");
        materialize(&source, &target, Profile::Smoke).unwrap();
        assert_eq!(std::fs::read_dir(&target).unwrap().count(), 128);
        for (index, row) in rows.iter().take(128).enumerate() {
            assert_eq!(
                std::fs::read_to_string(target.join(format!("{index:04}.json"))).unwrap(),
                *row
            );
        }
        let cohort: serde_json::Value =
            serde_json::from_slice(&std::fs::read(temp.path().join("cohort.json")).unwrap())
                .unwrap();
        assert_eq!(cohort["trace_ids"].as_array().unwrap().len(), 128);
        assert_eq!(cohort["trace_context"][0]["status"], "unsupported_context");
    }
    #[test]
    fn corrupted_download_fails_before_use() {
        let temp = tempfile::NamedTempFile::new().unwrap();
        assert!(verify(temp.path(), DATA_SHA).is_err());
    }
    #[cfg(unix)]
    #[test]
    fn consecutive_input_commands_retain_independent_outputs() {
        use std::os::unix::fs::PermissionsExt;
        let root = tempfile::tempdir().unwrap();
        let bin = root.path().join("skippy/evals/agentx/.venv/bin");
        std::fs::create_dir_all(&bin).unwrap();
        let fixture = root.path().join("cached.gguf");
        std::fs::write(&fixture, b"cached weights").unwrap();
        let hf = bin.join("hf");
        std::fs::write(
            &hf,
            format!("#!/bin/sh\nprintf '%s\\n' '{}'\n", fixture.display()),
        )
        .unwrap();
        std::fs::set_permissions(&hf, std::fs::Permissions::from_mode(0o755)).unwrap();
        for name in [
            "download-weight-0",
            "download-tokenizer",
            "download-dataset",
        ] {
            assert_eq!(
                download(
                    root.path(),
                    "org/model",
                    "pin",
                    "file",
                    false,
                    root.path(),
                    name
                )
                .unwrap(),
                fixture
            );
            assert!(root.path().join(format!("{name}.stdout.log")).is_file());
        }
    }
    #[cfg(unix)]
    #[test]
    fn hf_cache_resolution_preserves_snapshot_symlinks_and_rejects_ambiguous_output() {
        let cache = tempfile::tempdir().unwrap();
        let blob = cache.path().join("blob");
        std::fs::write(&blob, b"weights").unwrap();
        let snapshot = cache.path().join("snapshot.gguf");
        std::os::unix::fs::symlink(&blob, &snapshot).unwrap();
        assert_eq!(
            cached_path(&format!("{}\n", snapshot.display())).unwrap(),
            snapshot
        );
        assert!(cached_path(&format!("{}\n{}", blob.display(), snapshot.display())).is_err());
        assert!(cached_path("relative/model.gguf").is_err());
        assert!(cached_path(cache.path().join("missing").to_str().unwrap()).is_err());
    }

    #[cfg(unix)]
    #[test]
    fn offline_snapshot_uses_pinned_read_only_cache_without_refs() {
        let home = tempfile::tempdir().unwrap();
        let revision = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a";
        let snapshot = home
            .path()
            .join("hub/models--Qwen--Qwen3.5-4B/snapshots")
            .join(revision);
        std::fs::create_dir_all(&snapshot).unwrap();
        let blob = home.path().join("tokenizer-blob");
        std::fs::write(&blob, b"tokenizer").unwrap();
        for name in [
            "config.json",
            "merges.txt",
            "tokenizer.json",
            "tokenizer_config.json",
            "vocab.json",
        ] {
            std::os::unix::fs::symlink(&blob, snapshot.join(name)).unwrap();
        }
        assert_eq!(
            offline_cached_path(home.path(), "Qwen/Qwen3.5-4B", revision, "", false).unwrap(),
            snapshot
        );
        assert!(
            offline_cached_path(home.path(), "Qwen/Qwen3.5-4B", revision, "../blob", false)
                .is_err()
        );
        std::fs::remove_file(snapshot.join("tokenizer.json")).unwrap();
        assert!(offline_cached_path(home.path(), "Qwen/Qwen3.5-4B", revision, "", false).is_err());
        std::fs::create_dir_all(
            home.path()
                .join("hub/datasets--some--corpus/snapshots")
                .join(revision),
        )
        .unwrap();
        let dataset = home
            .path()
            .join("hub/datasets--some--corpus/snapshots")
            .join(revision)
            .join("traces.jsonl");
        std::os::unix::fs::symlink(&blob, &dataset).unwrap();
        assert_eq!(
            offline_cached_path(home.path(), "some/corpus", revision, "traces.jsonl", true)
                .unwrap(),
            dataset
        );
    }
}
