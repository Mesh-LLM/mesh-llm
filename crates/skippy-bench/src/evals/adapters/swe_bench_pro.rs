use super::super::{registry::definition, *};

pub(in crate::evals) fn swe_bench_pro_command(
    args: &EvalRunArgs,
    root: &Path,
    run_dir: &Path,
) -> Result<CommandSpec> {
    let harness = harness_dir(root, definition(EvalId::SweBenchPro));
    let script = run_dir.join("raw/swe-bench-pro-run.sh");
    write_swe_bench_pro_run_script(&script, args, root, &harness, run_dir)?;
    Ok(CommandSpec::new("zsh")
        .args([script.display().to_string()])
        .secret_env("SKIPPY_BENCH_API_KEY", args.api_key.clone()))
}

fn write_swe_bench_pro_run_script(
    path: &Path,
    args: &EvalRunArgs,
    cache_root: &Path,
    harness: &Path,
    run_dir: &Path,
) -> Result<()> {
    let raw_dir = run_dir.join("raw/swe-bench-pro");
    let instances = raw_dir.join("instances.yaml");
    let expert_instances = raw_dir.join("instances-expert.yaml");
    let sweagent_output = raw_dir.join("sweagent-results");
    let patches = raw_dir.join("patches.json");
    let eval_dir = raw_dir.join("eval");
    let dockerhub_username =
        env::var("SWE_BENCH_PRO_DOCKERHUB_USERNAME").unwrap_or_else(|_| "jefzda".to_string());
    let deployment_type =
        env::var("SWE_BENCH_PRO_DEPLOYMENT_TYPE").unwrap_or_else(|_| "docker".to_string());
    let num_workers =
        env_concurrency_matching_endpoint("SWE_BENCH_PRO_NUM_WORKERS", args.endpoint_concurrency)?;
    let eval_workers = env::var("SWE_BENCH_PRO_EVAL_WORKERS").unwrap_or_else(|_| "100".to_string());
    let docker_platform =
        env::var("SWE_BENCH_PRO_DOCKER_PLATFORM").unwrap_or_else(|_| "linux/amd64".to_string());
    let parse_function = env::var("SWE_BENCH_PRO_PARSE_FUNCTION").ok();
    let sweagent_python = env::var("SWE_BENCH_PRO_PYTHON").unwrap_or_else(|_| "3.12".to_string());
    let swerex_spec = env::var("SWE_BENCH_PRO_SWEREX_SPEC").ok();
    let swerex_pip_index_url = env::var("SWE_BENCH_PRO_SWEREX_PIP_INDEX_URL")
        .unwrap_or_else(|_| "https://pypi.org/simple".to_string());
    let use_local_eval = env::var("SWE_BENCH_PRO_USE_LOCAL_DOCKER")
        .map(|value| matches!(value.as_str(), "1" | "true" | "yes"))
        .unwrap_or(true);
    let local_eval_flag = if use_local_eval {
        "--use_local_docker"
    } else {
        ""
    };
    let model = litellm_model_name(&args.model);
    let script = format!(
        include_str!("templates/swe_bench_pro_run.sh"),
        harness = shell_quote(&harness.display().to_string()),
        raw_dir = shell_quote(&raw_dir.display().to_string()),
        instances = shell_quote(&instances.display().to_string()),
        expert_instances = shell_quote(&expert_instances.display().to_string()),
        sweagent_output = shell_quote(&sweagent_output.display().to_string()),
        patches = shell_quote(&patches.display().to_string()),
        eval_dir = shell_quote(&eval_dir.display().to_string()),
        model = shell_quote(&model),
        base_url = shell_quote(&args.base_url),
        dockerhub_username = shell_quote(&dockerhub_username),
        deployment_type = shell_quote(&deployment_type),
        num_workers = shell_quote(&num_workers),
        eval_workers = shell_quote(&eval_workers),
        docker_platform = shell_quote(&docker_platform),
        parse_function = shell_quote(parse_function.as_deref().unwrap_or("")),
        sweagent_python = shell_quote(&sweagent_python),
        swerex_spec = shell_quote(swerex_spec.as_deref().unwrap_or("")),
        swerex_pip_index_url = shell_quote(&swerex_pip_index_url),
        local_eval_flag = shell_quote(local_eval_flag),
        hf_home = shell_quote(&cache_root.join("hf").display().to_string()),
        hf_datasets_cache = shell_quote(&cache_root.join("hf-datasets").display().to_string()),
        uv_cache_dir = shell_quote(&cache_root.join("uv").display().to_string()),
        xdg_cache_home = shell_quote(&cache_root.join("xdg").display().to_string()),
    );
    fs::write(path, script).with_context(|| format!("write {}", path.display()))
}
