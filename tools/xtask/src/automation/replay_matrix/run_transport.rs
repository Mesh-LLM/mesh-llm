use super::{run_budget::Budget, run_workload::Build};
use crate::command::DynResult;
use std::path::Path;

pub(super) fn request(
    workload: (&super::run_workload::Input, &Build),
    manifest: &Path,
    output: &Path,
    budget: &Budget,
) -> DynResult<serde_json::Value> {
    let (input, build) = workload;
    let reservation = std::net::TcpListener::bind(("127.0.0.1", 0))?;
    let port = reservation.local_addr()?.port();
    let timeout = budget.seconds(input.timeout_seconds)?;
    let startup = input.startup_timeout_seconds.min(timeout.saturating_sub(1));
    if startup == 0 {
        return Err("insufficient replay startup budget".into());
    }
    let mut request = serde_json::json!({"manifest":manifest,"requirements":input.requirements,
        "model":input.model,"model_sha256":input.model_sha256,"minimum_context_tokens":input.minimum_context_tokens,
        "minimum_session_prompt_tokens":input.minimum_session_prompt_tokens,"max_output_tokens":input.max_output_tokens,
        "request_timeout_seconds":input.request_timeout_seconds.min(timeout),"startup_timeout_seconds":startup,
        "timeout_seconds":timeout,"port":port,"output":output});
    request["replay_mode"] = serde_json::to_value(input.replay_mode)?;
    request["hf_home"] = serde_json::to_value(&input.hf_home)?;
    match build {
        Build::Mesh(build) => {
            request["binary"] = serde_json::to_value(&build.binary)?;
            request["native_runtime_root"] = serde_json::to_value(&build.runtime_root)?;
        }
        Build::External(build) => request["external"] = serde_json::to_value(build)?,
    }
    Ok(request)
}

pub(super) fn invoke(
    root: Option<&Path>,
    command: &str,
    input: &Path,
    output: &Path,
) -> DynResult<()> {
    crate::automation::run_replay_matrix(
        &[
            command.into(),
            "--input".into(),
            input.to_str().ok_or("non-Unicode input path")?.into(),
            "--output".into(),
            output.to_str().ok_or("non-Unicode output path")?.into(),
        ],
        root,
    )
}

pub(super) fn verify_build(build: &Build, budget: &Budget) -> DynResult<()> {
    match build {
        Build::Mesh(build) => verify_mesh(build),
        Build::External(build) => {
            let fresh = super::external_probe::verify_with_budget(
                &build.external_engine,
                budget.remaining(super::external_probe::VERSION_TIMEOUT)?,
            )?;
            if serde_json::to_value(build)? != serde_json::to_value(fresh)? {
                return Err("external replay arm version or provenance changed".into());
            }
            Ok(())
        }
    }
}

pub(super) fn verify_mesh(build: &super::run_workload::MeshBuild) -> DynResult<()> {
    if !build.engine.is_mesh()
        || !build.binary.is_absolute()
        || !build.runtime.is_absolute()
        || !build.runtime_root.is_absolute()
    {
        return Err("Mesh build requires its own absolute host/runtime identity".into());
    }
    let binary = crate::product::digest::file_sha256(&build.binary).map_err(|error| error.error)?;
    let runtime =
        crate::product::digest::tree_sha256(&build.runtime).map_err(|error| error.error)?;
    if binary != build.binary_sha256 || runtime != build.runtime_sha256 {
        return Err("replay arm build digest changed".into());
    }
    if !build
        .runtime
        .canonicalize()?
        .starts_with(build.runtime_root.canonicalize()?)
    {
        return Err("replay runtime is outside its bundle root".into());
    }
    Ok(())
}
