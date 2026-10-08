use super::super::{
    run::{speed_bench_output_path, speed_bench_response_timings_path},
    *,
};

pub(in crate::evals) fn speed_bench_command(
    definition: EvalDefinition,
    args: &EvalRunArgs,
    root: &Path,
    run_dir: &Path,
) -> Result<CommandSpec> {
    let harness = harness_dir(root, definition);

    let script = harness.join("tools/server/bench/speed-bench/speed_bench.py");
    let launcher = super::super::external_sdk_source::leaf("speed-bench-auth.py")?;
    let cache_root = super::super::speed_environment::base(root);
    let command = CommandSpec::new(
        super::super::speed_environment::runtime_python(root)
            .display()
            .to_string(),
    )
    .args([
        "-I".to_string(),
        "-B".to_string(),
        launcher.display().to_string(),
        script.display().to_string(),
        "--url".to_string(),
        args.base_url.clone(),
        "--model".to_string(),
        args.model.clone(),
        "--bench".to_string(),
        "qualitative".to_string(),
        "--category".to_string(),
        "all".to_string(),
        "--osl".to_string(),
        "1024".to_string(),
        "--concurrency".to_string(),
        args.endpoint_concurrency.to_string(),
        "--timeout".to_string(),
        args.timeout_secs.to_string(),
        "--output".to_string(),
        speed_bench_output_path(run_dir).display().to_string(),
    ])
    .env(
        "XDG_CACHE_HOME",
        cache_root.join("xdg").display().to_string(),
    )
    .env("HF_HOME", cache_root.join("hf").display().to_string())
    .env(
        "HF_DATASETS_CACHE",
        cache_root.join("hf-datasets").display().to_string(),
    )
    .env("UV_CACHE_DIR", cache_root.join("uv").display().to_string())
    .env("SKIPPY_BENCH_BASE_URL", args.base_url.clone())
    .env(
        "SKIPPY_BENCH_RESPONSE_TIMINGS_PATH",
        speed_bench_response_timings_path(run_dir)
            .display()
            .to_string(),
    )
    .secret_env("SKIPPY_BENCH_API_KEY", args.api_key.clone());
    Ok(super::super::speed_environment::isolated(command, root)
        .env("HF_HUB_OFFLINE", "1")
        .env("HF_DATASETS_OFFLINE", "1")
        .env(
            "SKIPPY_BENCH_SPEED_DATASET",
            super::super::speed_environment::dataset(root)
                .display()
                .to_string(),
        )
        .env(
            "SKIPPY_BENCH_SPEED_DATASET_CACHE",
            cache_root.join("dataset-cache").display().to_string(),
        ))
}

#[cfg(test)]
mod tests {
    #[test]
    fn response_timings_are_written_as_json_lines() {
        let bytes = super::super::super::external_sdk_source::read("speed-bench-auth.py").unwrap();
        let launcher = std::str::from_utf8(&bytes).unwrap();
        assert!(launcher.contains(r#"sort_keys=True) + "\n")"#));
        assert!(!launcher.contains(r#"sort_keys=True) + "\\n")"#));
        assert!(launcher.contains("request_sha256"));
        assert!(launcher.contains("response_sha256"));
    }
}
