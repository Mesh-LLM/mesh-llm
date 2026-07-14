use super::super::{
    registry::definition,
    run::{mcp_atlas_output_path, mcp_atlas_score_dir},
    *,
};

pub(in crate::evals) fn mcp_atlas_command(
    args: &EvalRunArgs,
    root: &Path,
    run_dir: &Path,
) -> Result<CommandSpec> {
    let harness = harness_dir(root, definition(EvalId::McpAtlas));
    let script = run_dir.join("raw/mcp-atlas-run.sh");
    write_mcp_atlas_run_script(&script, args, root, &harness, run_dir)?;
    Ok(CommandSpec::new("zsh")
        .args([script.display().to_string()])
        .secret_env("SKIPPY_BENCH_API_KEY", args.api_key.clone()))
}

fn write_mcp_atlas_run_script(
    path: &Path,
    args: &EvalRunArgs,
    cache_root: &Path,
    harness: &Path,
    run_dir: &Path,
) -> Result<()> {
    let raw_dir = run_dir.join("raw");
    let completion_dir = harness.join("services/mcp_eval");
    let default_output_name = format!("skippybench-{}-completion-results.csv", unix_millis()?);
    let model_label = model_label(&args.model);
    let score_dir = mcp_atlas_score_dir(run_dir);
    let completion_concurrency = env_concurrency_matching_endpoint(
        "MCP_ATLAS_COMPLETION_CONCURRENCY",
        args.endpoint_concurrency,
    )?;
    let script = format!(
        include_str!("templates/mcp_atlas_run.sh"),
        harness = shell_quote(&harness.display().to_string()),
        completion_dir = shell_quote(&completion_dir.display().to_string()),
        raw_dir = shell_quote(&raw_dir.display().to_string()),
        base_url = shell_quote(&args.base_url),
        model = shell_quote(&litellm_model_name(&args.model)),
        output = shell_quote(&mcp_atlas_output_path(run_dir).display().to_string()),
        output_name = shell_quote(&default_output_name),
        model_label = shell_quote(&model_label),
        score_dir = shell_quote(&score_dir.display().to_string()),
        completion_concurrency = shell_quote(&completion_concurrency),
        hf_home = shell_quote(&cache_root.join("hf").display().to_string()),
        hf_datasets_cache = shell_quote(&cache_root.join("hf-datasets").display().to_string()),
        uv_cache_dir = shell_quote(&cache_root.join("uv").display().to_string()),
        xdg_cache_home = shell_quote(&cache_root.join("xdg").display().to_string()),
    );
    fs::write(path, script).with_context(|| format!("write {}", path.display()))
}
