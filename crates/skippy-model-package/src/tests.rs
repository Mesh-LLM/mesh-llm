use crate::gguf_header::activation_width;
use crate::package::{
    ArtifactHook, ExplicitSourceIdentity, model_distribution_id, native_mtp_layer_indices,
    package_generation, resolve_local_package_input, run_artifact_hook,
    should_resume_package_artifact,
};
use crate::write::{local_artifact_files, resolve_gguf_shard_paths};
use skippy_ffi::TensorRole;
use skippy_runtime::TensorInfo;
use std::path::{Path, PathBuf};

#[cfg(unix)]
#[test]
fn artifact_hook_tolerates_a_hook_that_deletes_the_uploaded_file() {
    // The production upload hook (split-model-job.sh) uploads each artifact
    // and then unlinks it locally to stay under the HF Jobs ephemeral
    // storage limit. write_package_artifact must therefore read all artifact
    // metadata before invoking the hook; this test locks in that the hook is
    // allowed to remove the file and still report success.
    let dir = std::env::temp_dir().join(format!("skippy-hook-test-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();

    let artifact = dir.join("shared").join("metadata.gguf");
    std::fs::create_dir_all(artifact.parent().unwrap()).unwrap();
    std::fs::write(&artifact, b"artifact-bytes").unwrap();

    let record = dir.join("hook-record.txt");
    let hook = dir.join("delete-hook.sh");
    std::fs::write(
        &hook,
        format!(
            "#!/bin/bash\nset -euo pipefail\n\
             printf '%s\\n%s\\n' \"$SKIPPY_PACKAGE_ARTIFACT_PATH\" \
             \"$SKIPPY_PACKAGE_ARTIFACT_RELATIVE_PATH\" > {record}\n\
             rm -f \"$SKIPPY_PACKAGE_ARTIFACT_PATH\"\n",
            record = record.display()
        ),
    )
    .unwrap();
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(&hook, std::fs::Permissions::from_mode(0o755)).unwrap();
    }

    let result = run_artifact_hook(
        &ArtifactHook {
            command: Some(hook),
        },
        &artifact,
        "shared/metadata.gguf",
    );
    assert!(result.is_ok(), "hook run failed: {result:?}");
    assert!(!artifact.exists(), "hook should have deleted the artifact");

    let recorded = std::fs::read_to_string(&record).unwrap();
    let mut lines = recorded.lines();
    assert_eq!(lines.next().unwrap(), artifact.display().to_string());
    assert_eq!(lines.next().unwrap(), "shared/metadata.gguf");

    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn model_distribution_id_uses_shared_gguf_stem_normalization() {
    assert_eq!(
        model_distribution_id(Path::new("UD-IQ2_M/GLM-5.1-UD-IQ2_M-00001-of-00006.gguf")),
        Some("GLM-5.1-UD-IQ2_M".to_string())
    );
    assert_eq!(
        model_distribution_id(Path::new("Qwen3-8B-Q4_K_M.gguf")),
        Some("Qwen3-8B-Q4_K_M".to_string())
    );
    assert_eq!(model_distribution_id(Path::new("README.md")), None);
}

#[test]
fn local_package_input_requires_explicit_identity() {
    let error = resolve_local_package_input("model.gguf".into(), ExplicitSourceIdentity::default())
        .unwrap_err();

    assert!(error.to_string().contains("requires --model-id"));
}

#[test]
fn local_package_input_uses_explicit_coordinate_identity() {
    let input = resolve_local_package_input(
        "local.gguf".into(),
        ExplicitSourceIdentity {
            model_id: Some("org/repo:Q4_K_M".to_string()),
            source_repo: None,
            source_revision: Some("abc123".to_string()),
            source_file: Some("Qwen3-8B-Q4_K_M.gguf".to_string()),
        },
    )
    .unwrap();

    assert_eq!(input.model_id, "org/repo:Q4_K_M");
    assert_eq!(input.source_identity.repo.as_deref(), Some("org/repo"));
    assert_eq!(input.source_identity.revision.as_deref(), Some("abc123"));
    assert_eq!(
        input.source_identity.canonical_ref.as_deref(),
        Some("org/repo@abc123/Qwen3-8B-Q4_K_M.gguf")
    );
    assert_eq!(
        input.source_identity.distribution_id.as_deref(),
        Some("Qwen3-8B-Q4_K_M")
    );
}

#[test]
fn package_generation_is_absent_without_native_mtp_tensors() {
    let tensors = vec![tensor("blk.0.attn_norm.weight", Some(0))];

    assert!(package_generation(&tensors).is_none());
}

#[test]
fn package_generation_advertises_mtp_strategy() {
    let tensors = vec![
        tensor("blk.0.attn_norm.weight", Some(0)),
        tensor("blk.47.nextn.eh_proj.weight", Some(47)),
        tensor("blk.47.nextn.enorm.weight", Some(47)),
        tensor("blk.47.nextn.hnorm.weight", Some(47)),
    ];

    assert_eq!(native_mtp_layer_indices(&tensors), vec![47]);
    let generation = package_generation(&tensors).expect("MTP tensors should enable generation");
    let speculative = generation
        .speculative_decoding
        .expect("MTP generation should configure speculative decoding");
    assert_eq!(speculative.default, "mtp");
    let proposer = speculative
        .proposers
        .get("mtp")
        .expect("native MTP proposer should be present");
    assert_eq!(proposer.proposer_type, "native-mtp");
    assert_eq!(proposer.prediction_depth, Some(1));
    assert_eq!(proposer.layer_indices, vec![47]);
    let strategy = speculative
        .strategies
        .get("mtp")
        .expect("default strategy should be present");
    assert_eq!(strategy.strategy_type, "native-mtp");
    assert_eq!(strategy.proposer.as_deref(), Some("mtp"));
    assert_eq!(strategy.prediction_depth, Some(1));
    assert_eq!(strategy.layer_indices, vec![47]);
    let window = strategy
        .window_policy
        .as_ref()
        .expect("native MTP should declare its fixed window");
    assert_eq!(window.default, "fixed");
    assert_eq!(window.initial_window, 1);
    assert_eq!(window.min_window, 1);
    assert_eq!(window.max_window, 1);
}

#[test]
fn split_gguf_path_resolves_sibling_shards() {
    let dir = unique_test_dir("split-gguf-path");
    std::fs::create_dir_all(&dir).unwrap();
    for part in 1..=3 {
        std::fs::write(
            dir.join(format!("MiniMax-M2.7-UD-Q2_K_XL-{part:05}-of-00003.gguf")),
            b"",
        )
        .unwrap();
    }

    let input = dir.join("MiniMax-M2.7-UD-Q2_K_XL-00002-of-00003.gguf");
    let paths = resolve_gguf_shard_paths(&input).unwrap();
    let names = paths
        .iter()
        .map(|path| path.file_name().unwrap().to_string_lossy().to_string())
        .collect::<Vec<_>>();

    assert_eq!(
        names,
        vec![
            "MiniMax-M2.7-UD-Q2_K_XL-00001-of-00003.gguf",
            "MiniMax-M2.7-UD-Q2_K_XL-00002-of-00003.gguf",
            "MiniMax-M2.7-UD-Q2_K_XL-00003-of-00003.gguf",
        ]
    );
    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
fn local_artifact_files_preserve_shard_subdirectory() {
    let dir = unique_test_dir("split-gguf-files");
    let shard_dir = dir.join("UD-Q2_K_XL");
    std::fs::create_dir_all(&shard_dir).unwrap();
    for part in 1..=2 {
        std::fs::write(
            shard_dir.join(format!("MiniMax-M2.7-UD-Q2_K_XL-{part:05}-of-00002.gguf")),
            b"",
        )
        .unwrap();
    }

    let input = shard_dir.join("MiniMax-M2.7-UD-Q2_K_XL-00001-of-00002.gguf");
    let files = local_artifact_files(
        &input,
        "UD-Q2_K_XL/MiniMax-M2.7-UD-Q2_K_XL-00001-of-00002.gguf",
    )
    .unwrap()
    .into_iter()
    .map(|file| file.path)
    .collect::<Vec<_>>();

    assert_eq!(
        files,
        vec![
            "UD-Q2_K_XL/MiniMax-M2.7-UD-Q2_K_XL-00001-of-00002.gguf",
            "UD-Q2_K_XL/MiniMax-M2.7-UD-Q2_K_XL-00002-of-00002.gguf",
        ]
    );
    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
fn activation_width_reads_arch_embedding_length_from_gguf_metadata() {
    let dir = unique_test_dir("activation-width");
    std::fs::create_dir_all(&dir).unwrap();
    let model = dir.join("model.gguf");
    let mut bytes = gguf_header(2);
    push_string_kv(&mut bytes, "general.architecture", "qwen2");
    push_u32_kv(&mut bytes, "qwen2.embedding_length", 3584);
    std::fs::write(&model, bytes).unwrap();

    assert_eq!(activation_width(&model).unwrap(), 3584);
    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
fn activation_width_accepts_smaller_and_signed_integer_metadata() {
    let dir = unique_test_dir("activation-width-int-forms");
    std::fs::create_dir_all(&dir).unwrap();
    let u16_model = dir.join("u16.gguf");
    let i32_model = dir.join("i32.gguf");

    let mut u16_bytes = gguf_header(2);
    push_string_kv(&mut u16_bytes, "general.architecture", "tiny");
    push_u16_kv(&mut u16_bytes, "tiny.embedding_length", 1024);
    std::fs::write(&u16_model, u16_bytes).unwrap();

    let mut i32_bytes = gguf_header(2);
    push_string_kv(&mut i32_bytes, "general.architecture", "qwen2");
    push_i32_kv(&mut i32_bytes, "qwen2.embedding_length", 4096);
    std::fs::write(&i32_model, i32_bytes).unwrap();

    assert_eq!(activation_width(&u16_model).unwrap(), 1024);
    assert_eq!(activation_width(&i32_model).unwrap(), 4096);
    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
fn activation_width_rejects_zero_embedding_length() {
    let dir = unique_test_dir("activation-width-zero");
    std::fs::create_dir_all(&dir).unwrap();
    let model = dir.join("model.gguf");
    let mut bytes = gguf_header(2);
    push_string_kv(&mut bytes, "general.architecture", "qwen2");
    push_u32_kv(&mut bytes, "qwen2.embedding_length", 0);
    std::fs::write(&model, bytes).unwrap();

    let error = activation_width(&model).unwrap_err().to_string();
    assert!(error.contains("embedding_length 0"), "{error}");
    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
fn activation_width_rejects_oversized_metadata_string() {
    let dir = unique_test_dir("activation-width-big-string");
    std::fs::create_dir_all(&dir).unwrap();
    let model = dir.join("model.gguf");
    let mut bytes = gguf_header(3);
    push_string_kv(&mut bytes, "general.architecture", "qwen2");
    push_oversized_string_kv(&mut bytes, "junk");
    push_u32_kv(&mut bytes, "qwen2.embedding_length", 3584);
    std::fs::write(&model, bytes).unwrap();

    let error = activation_width(&model).unwrap_err().to_string();
    assert!(error.contains("exceeds safety limit"), "{error}");
    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
fn activation_width_rejects_too_deep_metadata_arrays() {
    let dir = unique_test_dir("activation-width-deep-array");
    std::fs::create_dir_all(&dir).unwrap();
    let model = dir.join("model.gguf");
    let mut bytes = gguf_header(3);
    push_string_kv(&mut bytes, "general.architecture", "qwen2");
    push_deep_array_kv(&mut bytes, "junk", 65);
    push_u32_kv(&mut bytes, "qwen2.embedding_length", 3584);
    std::fs::write(&model, bytes).unwrap();

    let error = activation_width(&model).unwrap_err().to_string();
    assert!(error.contains("array nesting exceeds"), "{error}");
    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
fn resumes_only_existing_artifacts_when_requested() {
    let dir = unique_test_dir("resume-artifact");
    std::fs::create_dir_all(&dir).unwrap();
    let artifact = dir.join("layer-000.gguf");
    std::fs::write(&artifact, b"existing").unwrap();

    assert!(should_resume_package_artifact(&artifact, true));
    assert!(!should_resume_package_artifact(&artifact, false));
    assert!(!should_resume_package_artifact(
        &dir.join("missing.gguf"),
        true
    ));
    std::fs::remove_dir_all(dir).unwrap();
}

fn tensor(name: &str, layer_index: Option<u32>) -> TensorInfo {
    TensorInfo {
        name: name.to_string(),
        layer_index,
        role: TensorRole::Layer,
        ggml_type: 0,
        byte_size: 1,
        element_count: 1,
    }
}

fn gguf_header(kv_count: u64) -> Vec<u8> {
    let mut bytes = Vec::new();
    bytes.extend_from_slice(b"GGUF");
    bytes.extend_from_slice(&2_u32.to_le_bytes());
    bytes.extend_from_slice(&0_i64.to_le_bytes());
    bytes.extend_from_slice(&(kv_count as i64).to_le_bytes());
    bytes
}

fn push_gguf_string(bytes: &mut Vec<u8>, value: &str) {
    bytes.extend_from_slice(&(value.len() as u64).to_le_bytes());
    bytes.extend_from_slice(value.as_bytes());
}

fn push_string_kv(bytes: &mut Vec<u8>, key: &str, value: &str) {
    push_gguf_string(bytes, key);
    bytes.extend_from_slice(&8_u32.to_le_bytes());
    push_gguf_string(bytes, value);
}

fn push_u32_kv(bytes: &mut Vec<u8>, key: &str, value: u32) {
    push_gguf_string(bytes, key);
    bytes.extend_from_slice(&4_u32.to_le_bytes());
    bytes.extend_from_slice(&value.to_le_bytes());
}

fn push_i32_kv(bytes: &mut Vec<u8>, key: &str, value: i32) {
    push_gguf_string(bytes, key);
    bytes.extend_from_slice(&5_u32.to_le_bytes());
    bytes.extend_from_slice(&value.to_le_bytes());
}

fn push_u16_kv(bytes: &mut Vec<u8>, key: &str, value: u16) {
    push_gguf_string(bytes, key);
    bytes.extend_from_slice(&2_u32.to_le_bytes());
    bytes.extend_from_slice(&value.to_le_bytes());
}

fn push_oversized_string_kv(bytes: &mut Vec<u8>, key: &str) {
    push_gguf_string(bytes, key);
    bytes.extend_from_slice(&8_u32.to_le_bytes());
    bytes.extend_from_slice(&(crate::gguf_header::MAX_GGUF_STRING_BYTES + 1).to_le_bytes());
}

fn push_deep_array_kv(bytes: &mut Vec<u8>, key: &str, depth: usize) {
    push_gguf_string(bytes, key);
    bytes.extend_from_slice(&9_u32.to_le_bytes());
    for _ in 0..depth {
        bytes.extend_from_slice(&9_u32.to_le_bytes());
        bytes.extend_from_slice(&1_u64.to_le_bytes());
    }
    bytes.extend_from_slice(&4_u32.to_le_bytes());
    bytes.extend_from_slice(&0_u64.to_le_bytes());
}

fn unique_test_dir(name: &str) -> PathBuf {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    std::env::temp_dir().join(format!(
        "skippy-model-package-{name}-{}-{nanos}",
        std::process::id()
    ))
}
