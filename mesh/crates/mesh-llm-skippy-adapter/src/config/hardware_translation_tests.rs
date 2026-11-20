use super::test_support::*;
use super::*;
use serde_json::Value;
use skippy_protocol::LoadMode;
use std::path::Path;

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

#[test]
fn hardware_cpu_moe_policy_reaches_stage_config() {
    let model_file = temp_model_file();
    let all_blocks = hardware_stage_json(
        "[defaults.hardware]\ncpu_moe = true\n",
        HARDWARE_TEST_MODEL_ID,
        model_file.path(),
    );
    assert_eq!(
        all_blocks["host_expert_layers"],
        serde_json::json!({ "start": 0, "end": 28 }),
        "cpu_moe = true moves every block's routed experts into host memory"
    );

    let first_four = hardware_stage_json(
        "[defaults.hardware]\nn_cpu_moe = 4\n",
        HARDWARE_TEST_MODEL_ID,
        model_file.path(),
    );
    assert_eq!(
        first_four["host_expert_layers"],
        serde_json::json!({ "start": 0, "end": 4 }),
        "n_cpu_moe counts blocks from block 0, like llama.cpp --n-cpu-moe"
    );

    for toml in [
        "[defaults.hardware]\ncpu_moe = false\nn_cpu_moe = 4\n",
        "[defaults.hardware]\ncpu_moe = \"auto\"\n",
        "",
    ] {
        let stage = hardware_stage_json(toml, HARDWARE_TEST_MODEL_ID, model_file.path());
        assert!(
            stage["host_expert_layers"].is_null(),
            "{toml:?} must keep routed experts on the accelerator"
        );
    }
}

#[test]
fn hardware_n_cpu_moe_per_model_override_beats_default() {
    let model_file = temp_model_file();
    let stage = hardware_stage_json(
        &format!(
            "[defaults.hardware]\nn_cpu_moe = 2\n\n[[models]]\nmodel = \"{HARDWARE_TEST_MODEL_ID}\"\n\n[models.hardware]\nn_cpu_moe = 6\n"
        ),
        HARDWARE_TEST_MODEL_ID,
        model_file.path(),
    );
    assert_eq!(
        stage["host_expert_layers"],
        serde_json::json!({ "start": 0, "end": 6 })
    );
}
