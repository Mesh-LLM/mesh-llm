use crate::command::DynResult;
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

#[derive(Deserialize, Serialize)]
pub(super) struct Dataset {
    pub file: PathBuf,
    pub sha256: String,
    pub revision: String,
    pub python: PathBuf,
    pub timeout_seconds: u64,
    pub sessions_per_cohort: usize,
    pub min_isl: u64,
    pub max_isl: u64,
    pub min_turns: usize,
    pub frameworks: Vec<String>,
    pub source_datasets: Vec<String>,
}

pub(super) fn prepare(
    root: Option<&Path>,
    dataset: &Dataset,
    selection: (&Path, &super::manifest_preflight::Requirements),
) -> DynResult<()> {
    verify(dataset)?;
    let digest = dataset.sha256.clone();
    if dataset.sessions_per_cohort == 0
        || dataset.min_turns == 0
        || dataset.min_isl >= dataset.max_isl
    {
        return Err("invalid trajectory selection window".into());
    }
    if selection.0.try_exists()? {
        return Err("generated trajectory manifest already exists".into());
    }
    if let Some(parent) = selection.0.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let mut arguments = vec![
        "--python".into(),
        dataset
            .python
            .to_str()
            .ok_or("non-Unicode interpreter path")?
            .into(),
        "--timeout".into(),
        dataset.timeout_seconds.to_string(),
        "--dataset-file".into(),
        dataset
            .file
            .to_str()
            .ok_or("non-Unicode dataset path")?
            .into(),
        "--dataset-revision".into(),
        dataset.revision.clone(),
        "--output".into(),
        selection
            .0
            .to_str()
            .ok_or("non-Unicode manifest path")?
            .into(),
        "--sessions-per-cohort".into(),
        dataset.sessions_per_cohort.to_string(),
        "--min-isl".into(),
        dataset.min_isl.to_string(),
        "--max-isl".into(),
        dataset.max_isl.to_string(),
        "--min-turns".into(),
        dataset.min_turns.to_string(),
        "--cohort".into(),
        "warmup".into(),
    ];
    for level in &selection.1.concurrency {
        arguments.extend(["--cohort".into(), level.to_string()]);
    }
    for framework in &dataset.frameworks {
        arguments.extend(["--framework".into(), framework.clone()]);
    }
    for source in &dataset.source_datasets {
        arguments.extend(["--source-dataset".into(), source.clone()]);
    }
    super::trajectory_reader::run(root, &arguments)?;
    let after = crate::product::digest::file_sha256(&dataset.file).map_err(|error| error.error)?;
    if after != digest {
        return Err("trajectory dataset changed during selection".into());
    }
    Ok(())
}

pub(super) fn verify(dataset: &Dataset) -> DynResult<()> {
    let digest = crate::product::digest::file_sha256(&dataset.file).map_err(|error| error.error)?;
    if digest != dataset.sha256 {
        return Err("replay dataset SHA-256 mismatch".into());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wrong_dataset_digest_precedes_reader_or_output_creation() {
        let state = tempfile::tempdir().unwrap();
        let file = state.path().join("input.parquet");
        std::fs::write(&file, b"fixture").unwrap();
        let dataset = Dataset {
            file,
            sha256: "wrong".into(),
            revision: "fixture".into(),
            python: state.path().join("must-not-execute"),
            timeout_seconds: 1,
            sessions_per_cohort: 2,
            min_isl: 1,
            max_isl: 100,
            min_turns: 1,
            frameworks: Vec::new(),
            source_datasets: Vec::new(),
        };
        let requirements = super::super::manifest_preflight::Requirements {
            concurrency: vec![1],
            minimum_worker_waves: 2,
            warmup_turns: 1,
            required_frameworks: Vec::new(),
        };
        let output = state.path().join("generated/manifest.json");
        let result = prepare(None, &dataset, (&output, &requirements));
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("dataset SHA-256 mismatch")
        );
        assert!(!output.parent().unwrap().exists());
    }
}
