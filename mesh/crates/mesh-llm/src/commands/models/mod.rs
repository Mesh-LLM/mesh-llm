//! Mesh console wiring for Skippy's model-command contract.
pub async fn dispatch_models_command(
    command: &mesh_llm_cli::models::ModelsCommand,
) -> anyhow::Result<()> {
    skippy_commands::models::run(
        command,
        mesh_llm_commands::model_output::model_output_context(),
    )
    .await
}
