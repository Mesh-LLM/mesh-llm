use super::*;

fn fixture() -> String {
    let mut source = String::from(
        "static enum skippy_status skippy_finish_model_open(void) {\nif (skippy_runtime_has_stage_plan(config)) {\n",
    );
    for (condition, message) in [
        (
            "config->layer_end > n_layer",
            "layer_end exceeds model layer count",
        ),
        (
            "skippy_runtime_is_source_stage(config) != (config->layer_start == 0)",
            "admitted activation imports disagree with the source-stage range",
        ),
        (
            "skippy_runtime_is_terminal_stage(config) != (config->layer_end == n_layer)",
            "admitted activation exports disagree with the terminal-stage range",
        ),
    ] {
        source.push_str(&format!("if ({condition}) {{ llama_model_free(model); const char * message = \"{message}\"; if (out_event) {{ emit(); }} skippy_set_error(out_error,SKIPPY_STATUS_INVALID_ARGUMENT,message); return SKIPPY_STATUS_INVALID_ARGUMENT; }}\n"));
    }
    source.push_str("}\nif (!skippy_context_capacity(config, lane_count, per_lane, total)) { return SKIPPY_STATUS_INVALID_ARGUMENT; }\nskippy_model * stage_model = nullptr;\nauto helper = []() { if (bad) return false; for (auto item : values) { inspect(item); } return true; };\n");
    for (outer, condition, message) in [
        (
            "skippy_runtime_has_stage_plan(config) && !skippy_runtime_is_terminal_stage(config)",
            "!build_boundary(false,stage_model->output_activation_boundary)",
            "stage graph output frontier does not match its admitted planner identities",
        ),
        (
            "skippy_runtime_has_stage_plan(config) && !skippy_runtime_is_source_stage(config)",
            "!build_boundary(true,stage_model->input_activation_boundary)",
            "stage graph input frontier does not match its admitted planner identities",
        ),
    ] {
        source.push_str(&format!(
            "if ({outer}) {{ if ({condition}) {{ return fail_boundary_load(\"{message}\"); }} }}\n"
        ));
    }
    source.push_str("}\nenum skippy_status skippy_model_open_impl(void) {}\n");
    source
}

#[test]
fn realized_failure_paths_accept_spacing_comments_and_opaque_event_blocks() {
    let source = fixture();
    assert!(validate(&source).is_ok());
    assert!(
        validate(&source.replace(
            "llama_model_free(model)",
            "llama_model_free /* unicode é */ ( model )"
        ))
        .is_ok()
    );
    let decorated = source.replace("skippy_model * stage_model", "/* model->arch == LLM_ARCH_FAKE */ auto text=R\"x(model->arch == LLM_ARCH_FAKE #if)x\"; skippy_model * stage_model");
    assert!(validate(&decorated).is_ok());
}

#[test]
fn realized_failure_paths_reject_fake_conditional_unreachable_and_wrong_contracts() {
    let source = fixture();
    for mutant in [
        source.replace("llama_model_free(model);", "if (false) llama_model_free(model);"),
        source.replace("llama_model_free(model);", "if (false) { llama_model_free(model); }"),
        source.replace("llama_model_free(model);", "return SKIPPY_STATUS_OK; llama_model_free(model);"),
        source.replace("layer_end exceeds model layer count", "wrong failure message"),
        source.replace("skippy_model * stage_model", "if (model->arch == LLM_ARCH_LLAMA) { return SKIPPY_STATUS_OK; } skippy_model * stage_model"),
        source.replace("skippy_model * stage_model", "#if 0\n skipped();\n#endif\nskippy_model * stage_model"),
        source.replace("return fail_boundary_load(", "if (false) return fail_boundary_load("),
        source.replace("return fail_boundary_load(", "return SKIPPY_STATUS_OK; return fail_boundary_load("),
        format!("/* {} */",source),
        format!("auto fake=R\"raw({})raw\";",source),
    ] { assert!(validate(&mutant).is_err(),"must reject unrealized contract: {mutant}"); }
}

#[test]
fn runtime_source_parent_escape_is_rejected() {
    let native = tempfile::tempdir().unwrap();
    let outside = tempfile::tempdir().unwrap();
    fs::create_dir_all(native.path().join("src")).unwrap();
    fs::write(outside.path().join("model_loading.cpp"), fixture()).unwrap();
    #[cfg(unix)]
    {
        std::os::unix::fs::symlink(outside.path(), native.path().join("src/skippy")).unwrap();
        assert!(verify(native.path()).is_err());
    }
}

#[test]
fn exact_capability_ancestry_rejects_wrappers_and_preceding_terminals() {
    let source = fixture();
    for (old, new) in [
        (
            "if (skippy_runtime_has_stage_plan(config)) {",
            "if (false) { if (skippy_runtime_has_stage_plan(config)) {",
        ),
        (
            "if (config->layer_end > n_layer)",
            "if (false) if (config->layer_end > n_layer)",
        ),
        (
            "if (config->layer_end > n_layer)",
            "return SKIPPY_STATUS_OK; if (config->layer_end > n_layer)",
        ),
        (
            "if (skippy_runtime_has_stage_plan(config)) {",
            "return SKIPPY_STATUS_OK; if (skippy_runtime_has_stage_plan(config)) {",
        ),
        (
            "if (!build_boundary(false,stage_model->output_activation_boundary))",
            "return SKIPPY_STATUS_OK; if (!build_boundary(false,stage_model->output_activation_boundary))",
        ),
    ] {
        let mut mutant = source.replacen(old, new, 1);
        if new.starts_with("if (false) {") {
            mutant = mutant.replacen(
                "skippy_model * stage_model",
                "} skippy_model * stage_model",
                1,
            );
        }
        assert!(
            validate(&mutant).is_err(),
            "must reject guard context: {new}"
        );
    }
    let wrapped = source
        .replace(
            "if (config->layer_end > n_layer)",
            "auto hidden = []() { if (config->layer_end > n_layer)",
        )
        .replace(
            "if (skippy_runtime_is_source_stage(config)",
            "}; if (skippy_runtime_is_source_stage(config)",
        );
    assert!(validate(&wrapped).is_err());
    let hidden = source
        .replace(
            "if (!build_boundary(false,stage_model->output_activation_boundary)) {",
            "if (false) { if (!build_boundary(false,stage_model->output_activation_boundary)) {",
        )
        .replace(
            "stage graph output frontier does not match its admitted planner identities\"); } }",
            "stage graph output frontier does not match its admitted planner identities\"); } } }",
        );
    assert!(validate(&hidden).is_err());
}

#[test]
fn real_prepared_function_accepts_legitimate_workload_and_lambda_control_flow() {
    assert!(validate(include_str!("prepared_admission.cpp")).is_ok());
}

// Append to the existing runtime_slice/tests.rs module; fixture()/validate() are owned there.
#[test]
fn architecture_specific_implementation_after_admission_remains_allowed() {
    let source = fixture().replace(
        "skippy_model * stage_model = nullptr;",
        "skippy_model * stage_model = nullptr;\nif (model->arch == LLM_ARCH_GLM_DSA) { configure_graph(); }",
    );
    assert!(validate(&source).is_ok());
}

#[test]
fn detached_commented_and_missing_input_frontier_contracts_are_refused() {
    let source = fixture();
    let guard = "if (config->layer_end > n_layer) {";
    for mutant in [
        source.replace(guard, "if (config->layer_end > n_layer) {} {"),
        source.replace(guard, "/* if (config->layer_end > n_layer) */ {"),
        source.replace(
            "stage graph input frontier does not match its admitted planner identities",
            "input frontier diagnostics without an admitted identity contract",
        ),
    ] {
        assert!(validate(&mutant).is_err(), "{mutant}");
    }
}
