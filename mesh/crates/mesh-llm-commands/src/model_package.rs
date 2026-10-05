//! Compatibility dispatch for the Skippy-owned model package workflow.
pub use skippy_commands::models::package::ModelPrepareArgs;
pub async fn dispatch_model_package(args: ModelPrepareArgs<'_>) -> anyhow::Result<()> {
    skippy_commands::models::run_package(args, crate::model_output::model_output_context()).await
}
