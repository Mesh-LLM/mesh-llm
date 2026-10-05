use super::test_support::*;
use super::*;
use anyhow::Context as _;
use serde_json::Value;
use skippy_protocol::LoadMode;
use std::{io::Write, path::Path};
use tempfile::NamedTempFile;

/// Resolves `toml_str` for `model_id` and returns the resulting stage config
/// as stable (time-varying-identifier-free) JSON, so hardware field wiring
/// tests can assert on materially different downstream state in a few lines.
fn hardware_stage_json(toml_str: &str, model_id: &str, model_path: &Path) -> Value {
    let mesh_config = parse_config(toml_str);
    let resolved = resolve_skippy_config(SkippyConfigResolveRequest {
        mesh_config: &mesh_config,
        model_id,
        model_path,
        model_bytes: 10 * 1024 * 1024 * 1024,
        allocatable_memory_bytes: None,
        request_defaults: None,
        package_generation: None,
        compact_meta: None,
    })
    .unwrap_or_else(|error| panic!("config should resolve: {error}"));
    let stage = resolved
        .to_stage_config(Some(fake_package_identity(28)), LoadMode::RuntimeSlice)
        .expect("stage config should build");
    stage_config_stable_json(&stage)
}

const HARDWARE_TEST_MODEL_ID: &str = "ggml-org/gemma-3-270m-it-GGUF:Q8_0";

#[test]
fn hardware_op_offload_true_and_false_reach_different_stage_configs() {
    let model_file = temp_model_file();
    let stage_true = hardware_stage_json(
        "[defaults.hardware]\nop_offload = true\n",
        HARDWARE_TEST_MODEL_ID,
        model_file.path(),
    );
    let stage_false = hardware_stage_json(
        "[defaults.hardware]\nop_offload = false\n",
        HARDWARE_TEST_MODEL_ID,
        model_file.path(),
    );
    assert_ne!(
        stage_true, stage_false,
        "hardware.op_offload=true and =false must reach the stage config differently"
    );
}

#[test]
fn hardware_op_offload_per_model_only_override_differs_from_unset() {
    let model_file = temp_model_file();
    let stage_set = hardware_stage_json(
        &format!(
            "[[models]]\nmodel = \"{HARDWARE_TEST_MODEL_ID}\"\n\n[models.hardware]\nop_offload = true\n"
        ),
        HARDWARE_TEST_MODEL_ID,
        model_file.path(),
    );
    let stage_unset = hardware_stage_json(
        &format!("[[models]]\nmodel = \"{HARDWARE_TEST_MODEL_ID}\"\n"),
        HARDWARE_TEST_MODEL_ID,
        model_file.path(),
    );
    assert_ne!(
        stage_set, stage_unset,
        "a per-model hardware.op_offload override with no defaults.hardware value set \
         must reach the stage config differently than leaving it unset"
    );
}

#[test]
fn hardware_op_offload_per_model_override_beats_default() {
    let model_file = temp_model_file();
    let stage_override = hardware_stage_json(
        &format!(
            "[defaults.hardware]\nop_offload = false\n\n[[models]]\nmodel = \"{HARDWARE_TEST_MODEL_ID}\"\n\n[models.hardware]\nop_offload = true\n"
        ),
        HARDWARE_TEST_MODEL_ID,
        model_file.path(),
    );
    let stage_default_only = hardware_stage_json(
        &format!(
            "[defaults.hardware]\nop_offload = false\n\n[[models]]\nmodel = \"{HARDWARE_TEST_MODEL_ID}\"\n"
        ),
        HARDWARE_TEST_MODEL_ID,
        model_file.path(),
    );
    assert_ne!(
        stage_override, stage_default_only,
        "a per-model hardware.op_offload override must beat the defaults.hardware value"
    );
}

#[test]
fn hardware_no_host_buffer_true_and_false_reach_different_stage_configs() {
    let model_file = temp_model_file();
    let stage_true = hardware_stage_json(
        "[defaults.hardware]\nno_host_buffer = true\n",
        HARDWARE_TEST_MODEL_ID,
        model_file.path(),
    );
    let stage_false = hardware_stage_json(
        "[defaults.hardware]\nno_host_buffer = false\n",
        HARDWARE_TEST_MODEL_ID,
        model_file.path(),
    );
    assert_ne!(
        stage_true, stage_false,
        "hardware.no_host_buffer=true and =false must reach the stage config differently"
    );
}

#[test]
fn hardware_no_host_buffer_per_model_only_override_differs_from_unset() {
    let model_file = temp_model_file();
    let stage_set = hardware_stage_json(
        &format!(
            "[[models]]\nmodel = \"{HARDWARE_TEST_MODEL_ID}\"\n\n[models.hardware]\nno_host_buffer = true\n"
        ),
        HARDWARE_TEST_MODEL_ID,
        model_file.path(),
    );
    let stage_unset = hardware_stage_json(
        &format!("[[models]]\nmodel = \"{HARDWARE_TEST_MODEL_ID}\"\n"),
        HARDWARE_TEST_MODEL_ID,
        model_file.path(),
    );
    assert_ne!(
        stage_set, stage_unset,
        "a per-model hardware.no_host_buffer override with no defaults.hardware value set \
         must reach the stage config differently than leaving it unset"
    );
}

#[test]
fn hardware_no_host_buffer_per_model_override_beats_default() {
    let model_file = temp_model_file();
    let stage_override = hardware_stage_json(
        &format!(
            "[defaults.hardware]\nno_host_buffer = false\n\n[[models]]\nmodel = \"{HARDWARE_TEST_MODEL_ID}\"\n\n[models.hardware]\nno_host_buffer = true\n"
        ),
        HARDWARE_TEST_MODEL_ID,
        model_file.path(),
    );
    let stage_default_only = hardware_stage_json(
        &format!(
            "[defaults.hardware]\nno_host_buffer = false\n\n[[models]]\nmodel = \"{HARDWARE_TEST_MODEL_ID}\"\n"
        ),
        HARDWARE_TEST_MODEL_ID,
        model_file.path(),
    );
    assert_ne!(
        stage_override, stage_default_only,
        "a per-model hardware.no_host_buffer override must beat the defaults.hardware value"
    );
}

#[test]
fn hardware_check_tensors_true_and_false_reach_different_stage_configs() {
    let model_file = temp_model_file();
    let stage_true = hardware_stage_json(
        "[defaults.hardware]\ncheck_tensors = true\n",
        HARDWARE_TEST_MODEL_ID,
        model_file.path(),
    );
    let stage_false = hardware_stage_json(
        "[defaults.hardware]\ncheck_tensors = false\n",
        HARDWARE_TEST_MODEL_ID,
        model_file.path(),
    );
    assert_ne!(
        stage_true, stage_false,
        "hardware.check_tensors=true and =false must reach the stage config differently"
    );
}

#[test]
fn hardware_checkpoint_quantization_reaches_stage_config() {
    let model_file = temp_model_file();
    let stage = hardware_stage_json(
        "[defaults.hardware]\ncheckpoint_quantization = \"Q4_K_M\"\n",
        HARDWARE_TEST_MODEL_ID,
        model_file.path(),
    );

    assert_eq!(
        stage.get("checkpoint_quantization"),
        Some(&Value::String("Q4_K_M".to_string()))
    );
}

/// Exercises the same Mesh configuration -> resolver -> stage-config ->
/// skippy-serving model-open and inference path used by the embedded host. CI
/// supplies a pinned Hugging Face checkpoint directory rather than checking
/// model bytes into the repository.
#[test]
#[ignore = "requires SKIPPY_SAFETENSORS_SMOKE_DIR with a complete checkpoint"]
fn safetensors_checkpoint_reaches_mesh_host_runtime() -> anyhow::Result<()> {
    let checkpoint = std::env::var_os("SKIPPY_SAFETENSORS_SMOKE_DIR")
        .map(std::path::PathBuf::from)
        .ok_or_else(|| anyhow::anyhow!("SKIPPY_SAFETENSORS_SMOKE_DIR is not set"))?;
    let quantization = std::env::var("SKIPPY_SAFETENSORS_SMOKE_QUANTIZATION")
        .unwrap_or_else(|_| "preserve".to_string());
    let gpu_layers = std::env::var("SKIPPY_SAFETENSORS_SMOKE_GPU_LAYERS")
        .unwrap_or_else(|_| "0".to_string())
        .parse::<i32>()
        .context("parse SKIPPY_SAFETENSORS_SMOKE_GPU_LAYERS")?;
    let mut synthetic_imatrix = None;
    let checkpoint_imatrix = match std::env::var("SKIPPY_SAFETENSORS_SMOKE_IMATRIX").ok() {
        Some(value) if value == "synthetic" => {
            let mut file = NamedTempFile::new().context("create synthetic importance matrix")?;
            let direct =
                skippy_model::gguf_writer::DirectCheckpoint::open(&checkpoint, 1024 * 1024)?;
            let layout = direct.imatrix_layout()?;
            let entry_count =
                i32::try_from(layout.len()).context("imatrix entry count exceeds i32")?;
            file.write_all(&entry_count.to_le_bytes())?;
            for entry in layout {
                let name = entry.name.as_bytes();
                file.write_all(&i32::try_from(name.len())?.to_le_bytes())?;
                file.write_all(name)?;
                file.write_all(&1_i32.to_le_bytes())?;
                file.write_all(&i32::try_from(entry.value_count)?.to_le_bytes())?;
                for _ in 0..entry.value_count {
                    file.write_all(&1_f32.to_le_bytes())?;
                }
            }
            file.flush()?;
            let path = file.path().to_string_lossy().into_owned();
            synthetic_imatrix = Some(file);
            Some(path)
        }
        Some(path) => Some(path),
        None => None,
    };
    let quantization_toml = toml::Value::String(quantization.clone()).to_string();
    let imatrix_toml = checkpoint_imatrix
        .as_ref()
        .map(|path| {
            format!(
                "checkpoint_imatrix = {}\n",
                toml::Value::String(path.clone())
            )
        })
        .unwrap_or_default();
    let mesh_config = parse_config(&format!(
        "[defaults.model_fit]\nctx_size = 128\nbatch = 128\nubatch = 128\n\
         \n[defaults.hardware]\ngpu_layers = {gpu_layers}\ncheckpoint_quantization = {quantization_toml}\n{imatrix_toml}"
    ));
    let identity = crate::synthetic_direct_gguf_package("safetensors-smoke", &checkpoint)?;
    let resolved = resolve_skippy_config(SkippyConfigResolveRequest {
        mesh_config: &mesh_config,
        model_id: "safetensors-smoke",
        model_path: &checkpoint,
        model_bytes: identity.source_model_bytes,
        allocatable_memory_bytes: None,
        request_defaults: None,
        package_generation: None,
        compact_meta: None,
    })?;
    let stage = resolved.to_stage_config(Some(identity), LoadMode::RuntimeSlice)?;

    assert_eq!(stage.model_path.as_deref(), checkpoint.to_str());
    assert_eq!(
        stage.checkpoint_quantization.as_deref(),
        Some(quantization.as_str())
    );
    let canonical_checkpoint_imatrix = checkpoint_imatrix
        .as_deref()
        .map(std::fs::canonicalize)
        .transpose()?
        .map(|path| path.to_string_lossy().into_owned());
    assert_eq!(
        stage.checkpoint_imatrix.as_deref(),
        canonical_checkpoint_imatrix.as_deref()
    );
    if checkpoint_imatrix.is_some() {
        anyhow::ensure!(
            stage.checkpoint_imatrix_sha256.is_some(),
            "importance matrix digest was not included in stage identity"
        );
    }
    let runtime = skippy_serving::runtime_state::load_runtime(&stage)?
        .ok_or_else(|| anyhow::anyhow!("Mesh host did not open the checkpoint"))?;
    let mut runtime = runtime
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let tokens = runtime.model.tokenize("Hello from Mesh", true)?;
    anyhow::ensure!(
        !tokens.is_empty(),
        "checkpoint tokenizer returned no tokens"
    );
    let first = runtime.prefill_chunked_sampled("safetensors-smoke", &tokens, None)?;
    anyhow::ensure!(first >= 0, "sampled prefill returned invalid token {first}");
    let second = runtime.decode("safetensors-smoke", first)?;
    anyhow::ensure!(second >= 0, "decode returned invalid token {second}");
    drop(synthetic_imatrix);
    Ok(())
}

#[test]
fn hardware_check_tensors_per_model_only_override_differs_from_unset() {
    let model_file = temp_model_file();
    let stage_set = hardware_stage_json(
        &format!(
            "[[models]]\nmodel = \"{HARDWARE_TEST_MODEL_ID}\"\n\n[models.hardware]\ncheck_tensors = true\n"
        ),
        HARDWARE_TEST_MODEL_ID,
        model_file.path(),
    );
    let stage_unset = hardware_stage_json(
        &format!("[[models]]\nmodel = \"{HARDWARE_TEST_MODEL_ID}\"\n"),
        HARDWARE_TEST_MODEL_ID,
        model_file.path(),
    );
    assert_ne!(
        stage_set, stage_unset,
        "a per-model hardware.check_tensors override with no defaults.hardware value set \
         must reach the stage config differently than leaving it unset"
    );
}

#[test]
fn hardware_check_tensors_per_model_override_beats_default() {
    let model_file = temp_model_file();
    let stage_override = hardware_stage_json(
        &format!(
            "[defaults.hardware]\ncheck_tensors = false\n\n[[models]]\nmodel = \"{HARDWARE_TEST_MODEL_ID}\"\n\n[models.hardware]\ncheck_tensors = true\n"
        ),
        HARDWARE_TEST_MODEL_ID,
        model_file.path(),
    );
    let stage_default_only = hardware_stage_json(
        &format!(
            "[defaults.hardware]\ncheck_tensors = false\n\n[[models]]\nmodel = \"{HARDWARE_TEST_MODEL_ID}\"\n"
        ),
        HARDWARE_TEST_MODEL_ID,
        model_file.path(),
    );
    assert_ne!(
        stage_override, stage_default_only,
        "a per-model hardware.check_tensors override must beat the defaults.hardware value"
    );
}

#[test]
fn hardware_direct_io_true_and_false_reach_different_stage_configs() {
    let model_file = temp_model_file();
    let stage_true = hardware_stage_json(
        "[defaults.hardware]\ndirect_io = true\n",
        HARDWARE_TEST_MODEL_ID,
        model_file.path(),
    );
    let stage_false = hardware_stage_json(
        "[defaults.hardware]\ndirect_io = false\n",
        HARDWARE_TEST_MODEL_ID,
        model_file.path(),
    );
    assert_ne!(
        stage_true, stage_false,
        "hardware.direct_io=true and =false must reach the stage config differently"
    );
}

#[test]
fn hardware_direct_io_per_model_only_override_differs_from_unset() {
    let model_file = temp_model_file();
    let stage_set = hardware_stage_json(
        &format!(
            "[[models]]\nmodel = \"{HARDWARE_TEST_MODEL_ID}\"\n\n[models.hardware]\ndirect_io = true\n"
        ),
        HARDWARE_TEST_MODEL_ID,
        model_file.path(),
    );
    let stage_unset = hardware_stage_json(
        &format!("[[models]]\nmodel = \"{HARDWARE_TEST_MODEL_ID}\"\n"),
        HARDWARE_TEST_MODEL_ID,
        model_file.path(),
    );
    assert_ne!(
        stage_set, stage_unset,
        "a per-model hardware.direct_io override with no defaults.hardware value set \
         must reach the stage config differently than leaving it unset"
    );
}

#[test]
fn hardware_direct_io_per_model_override_beats_default() {
    let model_file = temp_model_file();
    let stage_override = hardware_stage_json(
        &format!(
            "[defaults.hardware]\ndirect_io = false\n\n[[models]]\nmodel = \"{HARDWARE_TEST_MODEL_ID}\"\n\n[models.hardware]\ndirect_io = true\n"
        ),
        HARDWARE_TEST_MODEL_ID,
        model_file.path(),
    );
    let stage_default_only = hardware_stage_json(
        &format!(
            "[defaults.hardware]\ndirect_io = false\n\n[[models]]\nmodel = \"{HARDWARE_TEST_MODEL_ID}\"\n"
        ),
        HARDWARE_TEST_MODEL_ID,
        model_file.path(),
    );
    assert_ne!(
        stage_override, stage_default_only,
        "a per-model hardware.direct_io override must beat the defaults.hardware value"
    );
}

#[test]
fn hardware_main_gpu_variants_reach_different_stage_configs() {
    let model_file = temp_model_file();
    let stage_zero = hardware_stage_json(
        "[defaults.hardware]\nmain_gpu = 0\n",
        HARDWARE_TEST_MODEL_ID,
        model_file.path(),
    );
    let stage_one = hardware_stage_json(
        "[defaults.hardware]\nmain_gpu = 1\n",
        HARDWARE_TEST_MODEL_ID,
        model_file.path(),
    );
    let stage_unset = hardware_stage_json("", HARDWARE_TEST_MODEL_ID, model_file.path());
    assert_ne!(
        stage_zero, stage_one,
        "hardware.main_gpu=0 and =1 must reach the stage config differently"
    );
    assert_ne!(
        stage_zero, stage_unset,
        "an explicit hardware.main_gpu=0 must reach the stage config differently than \
         leaving it unset (auto)"
    );
}

#[test]
fn hardware_main_gpu_per_model_only_override_differs_from_unset() {
    let model_file = temp_model_file();
    let stage_set = hardware_stage_json(
        &format!(
            "[[models]]\nmodel = \"{HARDWARE_TEST_MODEL_ID}\"\n\n[models.hardware]\nmain_gpu = 2\n"
        ),
        HARDWARE_TEST_MODEL_ID,
        model_file.path(),
    );
    let stage_unset = hardware_stage_json(
        &format!("[[models]]\nmodel = \"{HARDWARE_TEST_MODEL_ID}\"\n"),
        HARDWARE_TEST_MODEL_ID,
        model_file.path(),
    );
    assert_ne!(
        stage_set, stage_unset,
        "a per-model hardware.main_gpu override with no defaults.hardware value set must \
         reach the stage config differently than leaving it unset"
    );
}

#[test]
fn hardware_main_gpu_per_model_override_beats_default() {
    let model_file = temp_model_file();
    let stage_override = hardware_stage_json(
        &format!(
            "[defaults.hardware]\nmain_gpu = 0\n\n[[models]]\nmodel = \"{HARDWARE_TEST_MODEL_ID}\"\n\n[models.hardware]\nmain_gpu = 3\n"
        ),
        HARDWARE_TEST_MODEL_ID,
        model_file.path(),
    );
    let stage_default_only = hardware_stage_json(
        &format!(
            "[defaults.hardware]\nmain_gpu = 0\n\n[[models]]\nmodel = \"{HARDWARE_TEST_MODEL_ID}\"\n"
        ),
        HARDWARE_TEST_MODEL_ID,
        model_file.path(),
    );
    assert_ne!(
        stage_override, stage_default_only,
        "a per-model hardware.main_gpu override must beat the defaults.hardware value"
    );
}

#[test]
fn hardware_split_mode_variants_reach_different_stage_configs() {
    let model_file = temp_model_file();
    let stage_layer = hardware_stage_json(
        "[defaults.hardware]\nsplit_mode = \"layer\"\n",
        HARDWARE_TEST_MODEL_ID,
        model_file.path(),
    );
    let stage_row = hardware_stage_json(
        "[defaults.hardware]\nsplit_mode = \"row\"\n",
        HARDWARE_TEST_MODEL_ID,
        model_file.path(),
    );
    let stage_none = hardware_stage_json(
        "[defaults.hardware]\nsplit_mode = \"none\"\n",
        HARDWARE_TEST_MODEL_ID,
        model_file.path(),
    );
    let stage_auto = hardware_stage_json(
        "[defaults.hardware]\nsplit_mode = \"auto\"\n",
        HARDWARE_TEST_MODEL_ID,
        model_file.path(),
    );
    assert_ne!(
        stage_layer, stage_row,
        "hardware.split_mode=\"layer\" and =\"row\" must reach the stage config differently"
    );
    assert_ne!(
        stage_none, stage_auto,
        "hardware.split_mode=\"none\" and =\"auto\" must reach the stage config differently"
    );
}

#[test]
fn hardware_split_mode_per_model_only_override_differs_from_unset() {
    let model_file = temp_model_file();
    let stage_set = hardware_stage_json(
        &format!(
            "[[models]]\nmodel = \"{HARDWARE_TEST_MODEL_ID}\"\n\n[models.hardware]\nsplit_mode = \"row\"\n"
        ),
        HARDWARE_TEST_MODEL_ID,
        model_file.path(),
    );
    let stage_unset = hardware_stage_json(
        &format!("[[models]]\nmodel = \"{HARDWARE_TEST_MODEL_ID}\"\n"),
        HARDWARE_TEST_MODEL_ID,
        model_file.path(),
    );
    assert_ne!(
        stage_set, stage_unset,
        "a per-model hardware.split_mode override with no defaults.hardware value set must \
         reach the stage config differently than leaving it unset"
    );
}

#[test]
fn hardware_split_mode_per_model_override_beats_default() {
    let model_file = temp_model_file();
    let stage_override = hardware_stage_json(
        &format!(
            "[defaults.hardware]\nsplit_mode = \"layer\"\n\n[[models]]\nmodel = \"{HARDWARE_TEST_MODEL_ID}\"\n\n[models.hardware]\nsplit_mode = \"row\"\n"
        ),
        HARDWARE_TEST_MODEL_ID,
        model_file.path(),
    );
    let stage_default_only = hardware_stage_json(
        &format!(
            "[defaults.hardware]\nsplit_mode = \"layer\"\n\n[[models]]\nmodel = \"{HARDWARE_TEST_MODEL_ID}\"\n"
        ),
        HARDWARE_TEST_MODEL_ID,
        model_file.path(),
    );
    assert_ne!(
        stage_override, stage_default_only,
        "a per-model hardware.split_mode override must beat the defaults.hardware value"
    );
}

#[test]
fn hardware_split_mode_rejects_invalid_value() {
    let model_file = temp_model_file();
    let mesh_config = parse_config("[defaults.hardware]\nsplit_mode = \"bogus\"\n");
    let error = resolve_skippy_config(SkippyConfigResolveRequest {
        mesh_config: &mesh_config,
        model_id: HARDWARE_TEST_MODEL_ID,
        model_path: model_file.path(),
        model_bytes: 10 * 1024 * 1024 * 1024,
        allocatable_memory_bytes: None,
        request_defaults: None,
        package_generation: None,
        compact_meta: None,
    })
    .expect_err("an unrecognized hardware.split_mode value should be rejected");
    assert!(error.to_string().contains("split_mode"));
}
