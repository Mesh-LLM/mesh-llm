use super::*;
fn receipt() -> Receipt {
    Receipt {
        schema_version: 1,
        source: SOURCE.into(),
        dataset_revision: DATASET_REVISION.into(),
        dataset_sha256: DATASET_SHA.into(),
        python: PathBuf::from("/admitted/python"),
        tool_pins: BTreeMap::new(),
        environment_pins: BTreeMap::new(),
        benchmark_qualified: false,
    }
}
#[test]
fn prepared_contract_refuses_source_dataset_and_qualification_drift() {
    validate_receipt(&receipt()).unwrap();
    for field in ["source", "revision", "digest", "qualification", "schema"] {
        let mut changed = receipt();
        match field {
            "source" => changed.source = "master".into(),
            "revision" => changed.dataset_revision = "main".into(),
            "digest" => changed.dataset_sha256 = "0".repeat(64),
            "qualification" => changed.benchmark_qualified = true,
            _ => changed.schema_version = 2,
        }
        assert!(validate_receipt(&changed).is_err(), "{field}");
    }
}
#[test]
fn dataset_refuses_wrong_size_and_same_size_corruption() {
    assert!(validate_dataset(b"short").is_err());
    assert!(validate_dataset(&vec![0; usize::try_from(DATASET_BYTES).unwrap()]).is_err());
}
#[test]
fn missing_preparation_refuses_without_creating_environment_or_outputs() {
    let root = tempfile::tempdir().unwrap();
    assert!(admit(root.path()).is_err());
    assert!(!base(root.path()).exists());
}
#[test]
fn runtime_offline_command_retains_exact_upstream_arguments_and_private_cache() {
    use clap::Parser as _;
    let cli = crate::cli::Cli::try_parse_from([
        "skippy-bench",
        "eval",
        "run",
        "speed-bench",
        "--model",
        "fixture-model",
        "--base-url",
        "http://127.0.0.1:19337/v1",
        "--endpoint-concurrency",
        "3",
        "--timeout-secs",
        "7",
        "--dry-run",
    ])
    .unwrap();
    let crate::cli::CommandKind::Eval(eval) = cli.command else {
        panic!("eval");
    };
    let crate::cli::EvalCommandKind::Run(args) = eval.command else {
        panic!("run");
    };
    let root = tempfile::tempdir().unwrap();
    let spec = adapters::speed_bench_command(
        registry::definition(EvalId::SpeedBench),
        &args,
        root.path(),
        &root.path().join("run"),
    )
    .unwrap();
    assert_eq!(
        spec.program,
        runtime_python(root.path()).display().to_string()
    );
    assert_eq!(&spec.args[..2], ["-I", "-B"]);
    assert!(
        spec.args
            .windows(2)
            .any(|w| w == ["--bench", "qualitative"])
    );
    assert!(spec.args.windows(2).any(|w| w == ["--category", "all"]));
    assert!(spec.args.windows(2).any(|w| w == ["--concurrency", "3"]));
    assert!(spec.args.windows(2).any(|w| w == ["--timeout", "7"]));
    assert!(
        !spec
            .args
            .iter()
            .any(|a| matches!(a.as_str(), "uv" | "sync" | "--with-requirements"))
    );
    assert!(spec.clear_environment);
    assert_eq!(
        spec.envs
            .iter()
            .rev()
            .find(|(key, _)| key == "HF_HUB_OFFLINE")
            .map(|(_, value)| value.as_str()),
        Some("1")
    );
    assert_eq!(
        spec.envs
            .iter()
            .rev()
            .find(|(key, _)| key == "HF_DATASETS_OFFLINE")
            .map(|(_, value)| value.as_str()),
        Some("1")
    );
    assert_eq!(
        &spec
            .envs
            .iter()
            .rev()
            .find(|(key, _)| key == "SKIPPY_BENCH_SPEED_DATASET")
            .unwrap()
            .1,
        &dataset(root.path()).display().to_string()
    );
}
#[test]
fn pinned_sync_defers_submodule_acquisition_until_exact_source_checkout() {
    let definition = registry::definition(EvalId::SpeedBench);
    assert_eq!(definition.repo_ref, SOURCE);
    let steps = sync::new_eval_repo_sync_steps(Path::new("/private/speed-source"), definition);
    assert!(
        !steps[0]
            .args
            .iter()
            .any(|arg| arg == "--recurse-submodules")
    );
    assert_eq!(steps[1].args.last().unwrap(), SOURCE);
    assert_eq!(&steps[2].args[2..], ["checkout", "--detach", "FETCH_HEAD"]);
}
