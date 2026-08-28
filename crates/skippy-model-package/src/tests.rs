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
fn activation_width_for_qwen4exp_is_the_wide_hyper_connected_boundary() {
    // QWEN4EXP moves hc parallel residual streams across a stage boundary, so a
    // stage exchanges hc*embedding_length floats per token. embedding_length
    // alone would under-size every activation frame by a factor of hc.
    let dir = unique_test_dir("activation-width-qwen4exp");
    std::fs::create_dir_all(&dir).unwrap();
    let model = dir.join("model.gguf");
    let mut bytes = gguf_header(3);
    push_string_kv(&mut bytes, "general.architecture", "qwen4exp");
    push_u32_kv(&mut bytes, "qwen4exp.embedding_length", 2048);
    push_u32_kv(&mut bytes, "qwen4exp.hyper_connection.count", 4);
    std::fs::write(&model, bytes).unwrap();

    assert_eq!(activation_width(&model).unwrap(), 8192);
    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
fn activation_width_for_qwen4exp_accepts_a_matching_declared_output_width() {
    let dir = unique_test_dir("activation-width-qwen4exp-declared");
    std::fs::create_dir_all(&dir).unwrap();
    let model = dir.join("model.gguf");
    let mut bytes = gguf_header(4);
    push_string_kv(&mut bytes, "general.architecture", "qwen4exp");
    push_u32_kv(&mut bytes, "qwen4exp.embedding_length", 2048);
    push_u32_kv(&mut bytes, "qwen4exp.hyper_connection.count", 4);
    push_u32_kv(&mut bytes, "qwen4exp.embedding_length_out", 8192);
    std::fs::write(&model, bytes).unwrap();

    assert_eq!(activation_width(&model).unwrap(), 8192);
    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
fn activation_width_for_qwen4exp_rejects_a_contradictory_declared_output_width() {
    let dir = unique_test_dir("activation-width-qwen4exp-mismatch");
    std::fs::create_dir_all(&dir).unwrap();
    let model = dir.join("model.gguf");
    let mut bytes = gguf_header(4);
    push_string_kv(&mut bytes, "general.architecture", "qwen4exp");
    push_u32_kv(&mut bytes, "qwen4exp.embedding_length", 2048);
    push_u32_kv(&mut bytes, "qwen4exp.hyper_connection.count", 4);
    push_u32_kv(&mut bytes, "qwen4exp.embedding_length_out", 2048);
    std::fs::write(&model, bytes).unwrap();

    let error = activation_width(&model).unwrap_err().to_string();
    assert!(
        error.contains("disagrees with hyper_connection.count"),
        "{error}"
    );
    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
fn activation_width_for_qwen4exp_requires_the_hyper_connection_count() {
    // Failing loudly is the point: silently falling back to embedding_length
    // would produce a package whose manifest under-sizes every stage boundary.
    let dir = unique_test_dir("activation-width-qwen4exp-missing-hc");
    std::fs::create_dir_all(&dir).unwrap();
    let model = dir.join("model.gguf");
    let mut bytes = gguf_header(2);
    push_string_kv(&mut bytes, "general.architecture", "qwen4exp");
    push_u32_kv(&mut bytes, "qwen4exp.embedding_length", 2048);
    std::fs::write(&model, bytes).unwrap();

    let error = activation_width(&model).unwrap_err().to_string();
    assert!(error.contains("hyper_connection.count"), "{error}");
    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
fn activation_width_for_other_architectures_is_unchanged_by_a_stray_hc_count() {
    // The hyper-connected derivation must be scoped to qwen4exp only.
    let dir = unique_test_dir("activation-width-non-qwen4exp");
    std::fs::create_dir_all(&dir).unwrap();
    let model = dir.join("model.gguf");
    let mut bytes = gguf_header(3);
    push_string_kv(&mut bytes, "general.architecture", "qwen2");
    push_u32_kv(&mut bytes, "qwen2.embedding_length", 3584);
    push_u32_kv(&mut bytes, "qwen2.hyper_connection.count", 4);
    std::fs::write(&model, bytes).unwrap();

    assert_eq!(activation_width(&model).unwrap(), 3584);
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

#[test]
fn manifest_rejects_a_plan_that_disagrees_with_the_written_artifact() {
    // The observed drift: the native slice writer retained per_layer_token_embd
    // while the plan did not, so the manifest reported 610 tensors for an
    // artifact holding 611 and under-reported tensor_bytes by 26.8 GiB. That
    // number feeds split planning's per-layer cost estimate.
    let error = crate::write::check_manifest_matches_artifact(
        1,
        (610, 22_060_312_320),
        (611, 50_860_450_560),
        Path::new("/tmp/stage-001.gguf"),
    )
    .unwrap_err()
    .to_string();

    assert!(error.contains("disagree on tensor selection"), "{error}");
    assert!(error.contains("610"), "{error}");
    assert!(error.contains("611"), "{error}");
}

#[test]
fn manifest_accepts_a_plan_that_matches_the_written_artifact() {
    assert!(
        crate::write::check_manifest_matches_artifact(
            0,
            (617, 50_482_128_640),
            (617, 50_482_128_640),
            Path::new("/tmp/stage-000.gguf"),
        )
        .is_ok()
    );
}

#[test]
fn per_layer_token_embd_is_kept_only_by_stages_that_retain_a_ple_layer() {
    // Qwen3.8-Flash-Next declares ple.layers = [1]. Layer 1 is in stage 0, so
    // stage 1 can never gather from the 26.8 GiB table and must not ship it.
    let tensors = qwen4exp_tensors(48, &[1]);
    let plan = crate::plan::build_plan_from_tensors(2, &tensors).unwrap();

    let stage0 = &plan.stages[0];
    let stage1 = &plan.stages[1];

    assert!(
        stage_selects(&tensors, stage0.layer_start, stage0.layer_end, true, false),
        "stage 0 holds the PLE layer and must retain per_layer_token_embd.weight"
    );
    assert!(
        !stage_selects(&tensors, stage1.layer_start, stage1.layer_end, false, true),
        "stage 1 holds no PLE layer and must not ship per_layer_token_embd.weight"
    );

    // And the cost must be absent from the accounting, not merely from the file.
    assert!(
        stage1.tensor_bytes < 28_800_138_240,
        "stage 1 tensor_bytes {} still includes the per-layer embedding table",
        stage1.tensor_bytes
    );
}

#[test]
fn per_layer_token_embd_is_kept_by_a_mid_stage_that_holds_the_ple_layer() {
    // The case the loader exemption exists for: a PLE layer on a stage that does
    // not own the token embeddings.
    let tensors = qwen4exp_tensors(48, &[30]);
    let plan = crate::plan::build_plan_from_tensors(2, &tensors).unwrap();
    let stage1 = &plan.stages[1];

    assert!(
        stage_selects(&tensors, stage1.layer_start, stage1.layer_end, false, true),
        "a mid stage holding a PLE layer must retain per_layer_token_embd.weight \
         even though it does not include embeddings"
    );
}

#[test]
fn per_layer_token_embd_is_kept_by_every_stage_for_a_gemma_shaped_artifact() {
    // Gemma3n/Gemma4 gather the same table through per-block tensors named
    // `blk.N.inp_gate` / `blk.N.proj` / `blk.N.post_norm` (llama-arch.cpp:568-570)
    // -- none of which carry a `ple_` or `per_layer_` prefix. A name-based
    // consumer scan therefore finds nothing for them, and the qwen4exp rule must
    // NOT fail closed: every stage must retain the shared table.
    let mut tensors = vec![
        sized_tensor("token_embd.weight", None, TensorRole::Embedding, 100),
        sized_tensor(
            "per_layer_token_embd.weight",
            None,
            TensorRole::Embedding,
            5_000,
        ),
    ];
    for layer in 0..8u32 {
        tensors.push(sized_tensor(
            &format!("blk.{layer}.attn_norm.weight"),
            Some(layer),
            TensorRole::Layer,
            7,
        ));
        tensors.push(sized_tensor(
            &format!("blk.{layer}.inp_gate.weight"),
            Some(layer),
            TensorRole::Layer,
            5,
        ));
    }

    let plan = crate::plan::build_plan_from_tensors(4, &tensors).unwrap();
    for stage in &plan.stages {
        assert!(
            stage.includes_per_layer_token_embd
                && stage_selects(
                    &tensors,
                    stage.layer_start,
                    stage.layer_end,
                    stage.includes_embeddings,
                    stage.includes_output
                ),
            "gemma-shaped stage {} must retain per_layer_token_embd.weight",
            stage.stage_index
        );
    }
}

#[test]
fn cross_shard_ple_ownership_uses_complete_source_tensor_counts_and_bytes() {
    // The shared table is in shard 0 while the only sparse PLE consumer is in
    // shard 1. Planning joins both inventories before it decides which stage
    // owns the table, and the resulting ownership bit is passed to every
    // shard-local native slice plan.
    let table_bytes = 28_800_138_240;
    let table_shard = vec![
        sized_tensor("token_embd.weight", None, TensorRole::Embedding, 100),
        sized_tensor(
            "per_layer_token_embd.weight",
            None,
            TensorRole::Embedding,
            table_bytes,
        ),
    ];
    let mut consumer_shard = Vec::new();
    for layer in 0..4u32 {
        consumer_shard.push(sized_tensor(
            &format!("blk.{layer}.attn_norm.weight"),
            Some(layer),
            TensorRole::Layer,
            10,
        ));
    }
    consumer_shard.push(sized_tensor(
        "blk.1.ple_mlp.weight",
        Some(1),
        TensorRole::Layer,
        7,
    ));

    let tensors = table_shard
        .into_iter()
        .chain(consumer_shard)
        .collect::<Vec<_>>();
    let plan = crate::plan::build_plan_from_tensors(2, &tensors).unwrap();
    let stage0 = &plan.stages[0];
    let stage1 = &plan.stages[1];

    assert!(stage0.includes_per_layer_token_embd);
    assert!(!stage1.includes_per_layer_token_embd);
    assert_eq!(stage0.tensor_count, 5);
    assert_eq!(stage0.tensor_bytes, table_bytes + 127);
    assert_eq!(stage1.tensor_count, 2);
    assert_eq!(stage1.tensor_bytes, 20);
}

fn stage_selects(
    tensors: &[TensorInfo],
    layer_start: u32,
    layer_end: u32,
    includes_embeddings: bool,
    includes_output: bool,
) -> bool {
    let with_table = crate::plan::stage_plan_from_tensors(
        0,
        layer_start,
        layer_end,
        includes_embeddings,
        includes_output,
        tensors,
    );
    let without_table: Vec<TensorInfo> = tensors
        .iter()
        .filter(|tensor| tensor.name != "per_layer_token_embd.weight")
        .cloned()
        .collect();
    let baseline = crate::plan::stage_plan_from_tensors(
        0,
        layer_start,
        layer_end,
        includes_embeddings,
        includes_output,
        &without_table,
    );
    with_table.tensor_count > baseline.tensor_count
}

/// A qwen4exp-shaped tensor list: the PLE consumer blocks are a sparse subset of
/// layers (`ple.layers = [1]` on Qwen3.8-Flash-Next), so only the stage holding
/// layer 1 can ever gather from `per_layer_token_embd.weight`.
fn qwen4exp_tensors(layer_count: u32, ple_layers: &[u32]) -> Vec<TensorInfo> {
    let mut tensors = vec![
        sized_tensor("token_embd.weight", None, TensorRole::Embedding, 100),
        sized_tensor(
            "per_layer_token_embd.weight",
            None,
            TensorRole::Embedding,
            28_800_138_240,
        ),
        sized_tensor("output_hc_norm.weight", None, TensorRole::FinalNorm, 10),
    ];
    for layer in 0..layer_count {
        tensors.push(sized_tensor(
            &format!("blk.{layer}.hc_attn_norm.weight"),
            Some(layer),
            TensorRole::Layer,
            7,
        ));
        if ple_layers.contains(&layer) {
            tensors.push(sized_tensor(
                &format!("blk.{layer}.ple_key.weight"),
                Some(layer),
                TensorRole::Layer,
                5,
            ));
        }
    }
    tensors
}

fn sized_tensor(
    name: &str,
    layer_index: Option<u32>,
    role: TensorRole,
    byte_size: u64,
) -> TensorInfo {
    TensorInfo {
        name: name.to_string(),
        layer_index,
        role,
        ggml_type: 0,
        byte_size,
        element_count: 1,
    }
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

/// Regression cover for loading a stage that does not start at layer 0.
///
/// A mid-stage artifact legitimately contains no `blk.0.*` tensors. The native
/// loader must consult the skippy stage filter *before* looking a tensor up;
/// when the filter runs after the lookup instead, opening this artifact fails
/// with `check_tensor_dims: tensor 'blk.0.attn_norm.weight' not found`.
///
/// Writing a real mid-stage artifact and opening it through the runtime is what
/// makes this catch the bug: a full GGUF still contains block 0, so a filtered
/// config over an unfiltered file exercises none of this.
#[test]
fn mid_stage_artifact_opens_with_the_stage_filter_applied() -> anyhow::Result<()> {
    use skippy_runtime::{
        FlashAttentionType, GGML_TYPE_F16, RuntimeConfig, RuntimeLoadMode, StageModel,
    };

    let Some(model_path) = std::env::var_os("SKIPPY_CORRECTNESS_MODEL").map(PathBuf::from) else {
        eprintln!("skipping mid-stage load: SKIPPY_CORRECTNESS_MODEL is not set");
        return Ok(());
    };

    let source = crate::write::ModelSource::open(&model_path)?;
    let layer_count = crate::plan::layer_count(&source.tensors)?;
    if layer_count < 2 {
        eprintln!("skipping mid-stage load: model has fewer than 2 layers");
        return Ok(());
    }
    let layer_start = layer_count / 2;

    // RAII: the directory is removed when `dir` drops, on every exit path.
    let dir = tempfile::tempdir()?;
    let artifact = dir.path().join("stage-mid.gguf");
    let stage = crate::plan::stage_plan_from_tensors(
        1,
        layer_start,
        layer_count,
        true,
        true,
        &source.tensors,
    );
    crate::write::write_stage_artifact(&source, &stage, &artifact)?;

    let config = RuntimeConfig {
        stage_index: 1,
        layer_start,
        layer_end: layer_count,
        ctx_size: 256,
        n_gpu_layers: 0,
        cache_type_k: GGML_TYPE_F16,
        cache_type_v: GGML_TYPE_F16,
        flash_attn_type: FlashAttentionType::Auto,
        load_mode: RuntimeLoadMode::RuntimeSlice,
        include_embeddings: true,
        include_output: true,
        filter_tensors_on_load: true,
        ..RuntimeConfig::default()
    };

    let model = StageModel::open(&artifact, &config).map_err(|error| {
        anyhow::anyhow!("mid-stage load of layers {layer_start}..{layer_count} failed: {error}")
    })?;
    // Close the model before the temp dir is removed: on Windows a mapped file
    // cannot be deleted while it is still open.
    drop(model);
    dir.close()?;
    Ok(())
}
