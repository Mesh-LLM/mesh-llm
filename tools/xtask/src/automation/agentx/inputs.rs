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
        .find(|model| model.key == key)
        .ok_or("unknown AgentX model")?;
    let cache = std::env::var_os("AGENTX_CACHE")
        .map(PathBuf::from)
        .unwrap_or(root.join(".deps/agentx-inputs"));
    std::fs::create_dir_all(&cache)?;
    let weights = cache.join(&model.key).join(&model.revision);
    for file in &model.files {
        download(
            &root,
            &model.repo,
            &model.revision,
            &file.name,
            &weights,
            false,
            &output,
        )?;
        verify(&weights.join(&file.name), &file.sha256)?;
    }
    download(
        &root,
        &model.tokenizer,
        &model.tokenizer_revision,
        "",
        &weights,
        false,
        &output,
    )?;
    let dataset = cache.join(DATA_REVISION);
    download(
        &root,
        DATA_REPO,
        DATA_REVISION,
        "traces.jsonl",
        &dataset,
        true,
        &output,
    )?;
    verify(&dataset.join("traces.jsonl"), DATA_SHA)?;
    let traces = output.join("traces");
    materialize(&dataset.join("traces.jsonl"), &traces, profile)?;
    let prepared = Prepared {
        root,
        output,
        weights: weights.join(&model.files.first().ok_or("empty model files")?.name),
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
    destination: &Path,
    dataset: bool,
    output: &Path,
) -> DynResult<()> {
    let hf = root.join("skippy/evals/agentx/.venv/bin/hf");
    let mut args = command::strings(&["download", repo, "--revision", revision]);
    if !file.is_empty() {
        args.push("--local-dir".into());
        args.push(destination.to_string_lossy().into());
    }
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
    command::run(
        &command::spec(root, hf.to_str().ok_or("hf path")?, &args)?,
        7200,
        &output.join("download"),
    )
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
    let directory = if prepared.model.files[0].name.starts_with("BF16/") {
        parent.parent().ok_or("shard parent")?
    } else {
        parent
    };
    for FilePin { name, sha256 } in &prepared.model.files {
        verify(&directory.join(name), sha256)?;
    }
    Ok(())
}
fn materialize(source: &Path, destination: &Path, profile: Profile) -> DynResult<()> {
    std::fs::create_dir(destination)?;
    let limit = if profile == Profile::Smoke { 4 } else { 393 };
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
        let rows: Vec<_> = (0..5).map(|index| format!(r#"{{"id":"trace-{index}","requests":[{{"type":"subagent","id":"child"}},{{"out":100000}}]}}"#)).collect();
        std::fs::write(&source, rows.join("\n")).unwrap();
        let target = temp.path().join("selected");
        materialize(&source, &target, Profile::Smoke).unwrap();
        assert_eq!(std::fs::read_dir(&target).unwrap().count(), 4);
        for (index, row) in rows.iter().take(4).enumerate() {
            assert_eq!(
                std::fs::read_to_string(target.join(format!("{index:04}.json"))).unwrap(),
                *row
            );
        }
        let cohort: serde_json::Value =
            serde_json::from_slice(&std::fs::read(temp.path().join("cohort.json")).unwrap())
                .unwrap();
        assert_eq!(
            cohort["trace_ids"],
            serde_json::json!(["trace-0", "trace-1", "trace-2", "trace-3"])
        );
    }
    #[test]
    fn corrupted_download_fails_before_use() {
        let temp = tempfile::NamedTempFile::new().unwrap();
        assert!(verify(temp.path(), DATA_SHA).is_err());
    }
}
