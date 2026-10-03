//! Mesh console destinations for Skippy-owned model commands.
pub fn model_output_context() -> skippy_commands::models::ModelCommandContext {
    skippy_commands::models::ModelCommandContext {
        program: "mesh-llm",
        cache_root: skippy_model_hf::application_cache_dir(),
        fit_budget_bytes: skippy_hardware_profile::model_capacity::local_model_fit_budget_bytes(),
        terminal_progress: !mesh_llm_events::json_mode_enabled()
            && skippy_commands::console::stderr_is_terminal(),
        byte_progress: None,
        console_out: || Box::new(mesh_llm_events::console_out()),
        console_err: || Box::new(mesh_llm_events::console_err()),
        machine_out: || Box::new(mesh_llm_events::machine_out()),
    }
}
