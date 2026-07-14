use super::super::{run::terminal_bench_output_path, *};

pub(in crate::evals) fn terminal_bench_command(args: &EvalRunArgs, run_dir: &Path) -> CommandSpec {
    let model = litellm_model_name(&args.model);
    CommandSpec::new("tb")
        .args([
            "run".to_string(),
            "--dataset".to_string(),
            "terminal-bench-core==0.1.1".to_string(),
            "--agent".to_string(),
            "terminus".to_string(),
            "--model".to_string(),
            model,
            "--n-concurrent".to_string(),
            args.endpoint_concurrency.to_string(),
            "--output-path".to_string(),
            terminal_bench_output_path(run_dir).display().to_string(),
            "--global-agent-timeout-sec".to_string(),
            args.timeout_secs.to_string(),
            "--global-test-timeout-sec".to_string(),
            "60".to_string(),
            "--no-upload-results".to_string(),
            "--no-livestream".to_string(),
        ])
        .env("OPENAI_BASE_URL", args.base_url.clone())
        .secret_env("OPENAI_API_KEY", args.api_key.clone())
}
