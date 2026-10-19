use super::{FilePin, Model, Prepared, Profile, command, write};
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
    let models: Vec<Model> = serde_json::from_slice(&std::fs::read(
        root.join("skippy/evals/agentx/models.json"),
    )?)?;
    let model = models
        .into_iter()
        .find(|model| model.key == key || model.registry_ref() == key)
        .ok_or("unknown AgentX model")?;
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
fn download(
    root: &Path,
    repo: &str,
    revision: &str,
    file: &str,
    dataset: bool,
    output: &Path,
    log_name: &str,
) -> DynResult<PathBuf> {
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
    let log = output.join(log_name);
    let mut spec = command::spec(root, hf.to_str().ok_or("hf path")?, &args)?;
    spec.environment.insert(
        "HF_HUB_OFFLINE".into(),
        crate::process::Value::Public("1".into()),
    );
    command::run(&spec, 60, &log)?;
    cached_path(&std::fs::read_to_string(log.with_extension("stdout.log"))?)
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
pub(super) fn recheck(prepared: &Prepared) -> DynResult<()> {
    let parent = prepared.weights.parent().ok_or("weights parent")?;
    let directory = parent;
    for FilePin { name, sha256 } in &prepared.model.files {
        verify(&directory.join(name), sha256)?;
    }
    Ok(())
}
fn materialize(source: &Path, destination: &Path, profile: Profile) -> DynResult<()> {
    std::fs::create_dir(destination)?;
    let limit = if profile == Profile::Smoke { 128 } else { 393 };
    let mut ids = Vec::new();
    for (index, line) in BufReader::new(std::fs::File::open(source)?)
        .lines()
        .take(limit)
        .enumerate()
    {
        let line = line?;
        let trace: serde_json::Value = serde_json::from_str(&line)?;
        ids.push(trace["id"].as_str().ok_or("trace ID missing")?.to_owned());
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
        &serde_json::json!({"dataset_repo":DATA_REPO,"dataset_revision":DATA_REVISION,"dataset_sha256":DATA_SHA,"trace_ids":ids}),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn smoke_preserves_complete_rows_and_selects_exact_prefix() {
        let temp = tempfile::tempdir().unwrap();
        let source = temp.path().join("corpus.jsonl");
        let rows: Vec<_> = (0..129).map(|index| format!(r#"{{"id":"trace-{index}","requests":[{{"type":"subagent","id":"child"}},{{"out":100000}}]}}"#)).collect();
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
}
