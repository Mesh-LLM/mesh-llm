use super::*;
use sha2::{Digest, Sha256};

fn model(path: &Path) -> String {
    let mut bytes = b"GGUF".to_vec();
    bytes.extend(3_u32.to_le_bytes());
    bytes.extend(0_u64.to_le_bytes());
    bytes.extend(2_u64.to_le_bytes());
    for (key, value) in [
        ("general.architecture", Some("fixture")),
        ("fixture.context_length", None),
    ] {
        bytes.extend(u64::try_from(key.len()).unwrap().to_le_bytes());
        bytes.extend(key.as_bytes());
        if let Some(value) = value {
            bytes.extend(8_u32.to_le_bytes());
            bytes.extend(u64::try_from(value.len()).unwrap().to_le_bytes());
            bytes.extend(value.as_bytes());
        } else {
            bytes.extend(4_u32.to_le_bytes());
            bytes.extend(131072_u32.to_le_bytes());
        }
    }
    let digest = hex::encode(Sha256::digest(&bytes));
    std::fs::write(path, bytes).unwrap();
    digest
}

fn case(change: impl FnOnce(&mut serde_json::Value), refs: &[&str], good: bool) {
    let state = tempfile::tempdir().unwrap();
    let model_file = state.path().join("model.gguf");
    let digest = model(&model_file);
    let dataset = state.path().join("dataset.parquet");
    std::fs::write(&dataset, b"fixture dataset").unwrap();
    let matrix_path = state.path().join("matrix.json");
    let mut matrix: serde_json::Value = serde_json::from_slice(include_bytes!(
        "../../../../ci/agentic-replay-nightly/matrix.json"
    ))
    .unwrap();
    matrix["models"][0]["sha256"] = digest.into();
    matrix["replay"]["dataset_sha256"] = hex::encode(Sha256::digest(b"fixture dataset")).into();
    change(&mut matrix);
    std::fs::write(&matrix_path, serde_json::to_vec(&matrix).unwrap()).unwrap();
    let output = state.path().join("final");
    let worktrees = state.path().join("worktrees");
    let executable = std::env::current_exe().unwrap();
    let refs = refs
        .iter()
        .map(|reference| (*reference).to_owned())
        .collect::<Vec<_>>();
    let context = Context {
        matrix: &matrix_path,
        family: "granite-3.1-2b",
        model: &model_file,
        dataset: &dataset,
        python: &executable,
        output: &output,
        refs: &refs,
        repo: state.path(),
        worktree_root: &worktrees,
        git: &executable,
        just: &executable,
        timeout_seconds: 3600,
    };
    let prepared = prepare(&context);
    assert_eq!(
        prepared.is_ok(),
        good,
        "{}",
        prepared
            .as_ref()
            .err()
            .map_or(String::new(), ToString::to_string)
    );
    assert!(!output.exists());
    assert!(!worktrees.exists());
    if let Ok(prepared) = prepared {
        let serialized = serde_json::to_value(&prepared.input).unwrap();
        assert_eq!(
            serialized["build_jobs"].as_array().unwrap().len(),
            refs.len()
        );
        for (job, reference) in prepared.input.build_jobs.iter().zip(&refs) {
            let (label, reference) = reference.split_once('=').unwrap();
            assert_eq!(job.label, label);
            assert_eq!(job.reference, reference);
            assert_eq!(job.backend, "metal");
        }
        assert_eq!(
            prepared.input.require_recurrent_restores,
            matrix["models"][0]["class"] == "hybrid-recurrent"
        );
        assert_eq!(
            prepared.input.passes,
            u32::try_from(matrix["replay"]["passes"].as_u64().unwrap()).unwrap()
        );
        assert_eq!(
            prepared.input.dataset.as_ref().unwrap().frameworks,
            ["swe-agent", "mini-swe-agent", "openhands"]
        );
        assert_eq!(
            prepared.input.dataset.as_ref().unwrap().source_datasets,
            [
                "swe-smith-claude-3-7-sonnet",
                "kwai-klear-swe-smith-mini",
                "nebius-swe-rebench-openhands"
            ]
        );
        assert_eq!(
            prepared.input.model_reference.as_deref(),
            Some(
                "bartowski/granite-3.1-2b-instruct-GGUF@e47b8b46c04cede00f9e19d5a846551b14b2efce/granite-3.1-2b-instruct-Q4_K_M.gguf"
            )
        );
        let staging = prepared.staging.root().to_path_buf();
        assert!(staging.is_dir());
        assert!(!prepared.input.manifest.exists());
        assert!(!prepared.input.manifest.starts_with(&output));
        drop(prepared);
        assert!(!staging.exists());
    }
}

#[test]
fn single_family_keeps_pins_defaults_and_owned_staging_outside_output() {
    case(|_| {}, &["main=HEAD"], true);
}

#[test]
fn repair_refs_keep_order_and_matrix_passes_with_recurrent_requirement() {
    case(
        |matrix| {
            matrix["models"][0]["class"] = "hybrid-recurrent".into();
            matrix["replay"]["passes"] = 3.into();
        },
        &["fixed=HEAD", "base=prior"],
        true,
    );
}

#[test]
fn model_digest_context_and_dataset_digest_reject_before_children_or_output() {
    case(
        |matrix| matrix["models"][0]["sha256"] = "0".repeat(64).into(),
        &["main=HEAD"],
        false,
    );
    case(
        |matrix| matrix["models"][0]["native_context_tokens"] = 4096.into(),
        &["main=HEAD"],
        false,
    );
    case(
        |matrix| matrix["replay"]["dataset_sha256"] = "0".repeat(64).into(),
        &["main=HEAD"],
        false,
    );
}

#[test]
fn absent_duplicate_family_and_ref_labels_fail_preparation() {
    case(
        |matrix| matrix["models"][0]["family"] = "other".into(),
        &["main=HEAD"],
        false,
    );
    case(
        |matrix| {
            let copy = matrix["models"][0].clone();
            matrix["models"].as_array_mut().unwrap().push(copy);
        },
        &["main=HEAD"],
        false,
    );
    case(|_| {}, &["main=HEAD", "main=prior"], false);
}
