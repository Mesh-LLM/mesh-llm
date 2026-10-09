use super::*;
use crate::cli::Command;
use skippy_protocol::StageConfig;

fn parse_test(args: &[&str]) -> Result<Cli> {
    parse_with_environment(args.iter().map(OsString::from), |_| None)
}

#[test]
fn complete_help_groups_every_catalog_option() {
    let mut command = command();
    let help = command
        .find_subcommand_mut("serve")
        .unwrap()
        .render_long_help()
        .to_string();
    for spec in OPTIONS {
        assert!(help.contains(&format!("--{}", spec.name)), "{}", spec.name);
    }
    for section in [
        "Model loading:",
        "Devices and execution:",
        "Sampling defaults:",
        "Chat and reasoning:",
        "Speculative decoding:",
        "Multimodal:",
        "Prompt caching:",
        "Distributed stages:",
    ] {
        assert!(help.contains(section), "{section}");
    }
    assert!(!help.contains("Binary stage tuning:"));
    let sections = [
        "Settings:",
        "Model loading:",
        "Devices and execution:",
        "Context and KV memory:",
        "Prompt caching:",
        "Scheduling and prefill:",
        "Sampling defaults:",
        "Chat and reasoning:",
        "Speculative decoding:",
        "Multimodal:",
        "API and compatibility:",
        "Distributed stages:",
        "Native runtime:",
        "Diagnostics:",
        "Network simulation (testing):",
    ];
    for pair in sections.windows(2) {
        assert!(help.find(pair[0]).unwrap() < help.find(pair[1]).unwrap());
    }
}

#[test]
fn cli_and_environment_override_file_including_false() {
    let dir = tempfile::tempdir().unwrap();
    let settings = dir.path().join("serve.toml");
    std::fs::write(&settings, "version = 1\n[model]\nmodel_path = 'tiny.gguf'\nmlock = true\n[sampling]\ntemperature = 0.2\n[execution]\nthreads = 2\n").unwrap();
    let cli = parse_with_environment(
        [
            "skippy",
            "serve",
            "--settings",
            settings.to_str().unwrap(),
            "--mlock=false",
            "--temperature",
            "0.7",
        ]
        .into_iter()
        .map(OsString::from),
        |name| (name == "SKIPPY_SERVE_THREADS").then(|| "5".into()),
    )
    .unwrap();
    let Command::Serve(args) = cli.command else {
        panic!()
    };
    assert_eq!(args.public.model_path, Some(dir.path().join("tiny.gguf")));
    assert_eq!(args.public.settings.values["mlock"], false);
    let tuning = args
        .public
        .settings
        .tuning(args.public.openai_guardrails)
        .unwrap();
    assert_eq!(tuning.n_threads, Some(5));
    assert_eq!(tuning.request_defaults.temperature, Some(0.7));
    assert_eq!(args.public.settings.sources["temperature"], "cli");
    assert_eq!(
        args.public.settings.sources["threads"],
        "environment:SKIPPY_SERVE_THREADS"
    );
}

#[test]
fn explicit_worker_false_does_not_require_a_stage_configuration() {
    let Command::Serve(args) = parse_test(&[
        "skippy",
        "serve",
        "--model",
        "missing.gguf",
        "--worker-only=false",
    ])
    .unwrap()
    .command
    else {
        panic!()
    };
    assert!(!args.worker_only);
    args.public.settings.validate_mode(&args).unwrap();
}

#[test]
fn structured_values_and_text_files_are_relative_to_settings_file() {
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(dir.path().join("prompt.txt"), "You are concise.").unwrap();
    std::fs::write(dir.path().join("schema.json"), r#"{"type":"object"}"#).unwrap();
    let path = dir.path().join("serve.toml");
    std::fs::write(&path, "[model]\nmodel = 'some/repo'\n[chat]\nsystem_prompt = '@prompt.txt'\njson_schema = '@schema.json'\n[sampling]\nstop = ['END']\ndry_multiplier = 0.4\n").unwrap();
    let cli = parse_test(&["skippy", "serve", "--settings", path.to_str().unwrap()]).unwrap();
    let Command::Serve(args) = cli.command else {
        panic!()
    };
    let tuning = args
        .public
        .settings
        .tuning(args.public.openai_guardrails)
        .unwrap();
    assert_eq!(
        tuning.request_defaults.system_prompt.as_deref(),
        Some("You are concise.")
    );
    assert_eq!(
        tuning.request_defaults.json_schema.unwrap()["type"],
        "object"
    );
    assert_eq!(tuning.request_defaults.stop, Some(vec!["END".into()]));
    let dry = tuning.request_defaults.dry.unwrap();
    assert_eq!(dry.multiplier, 0.4);
    assert_eq!(dry.base, skippy_runtime::SamplingConfig::default().dry.base);
}

#[test]
fn file_rejects_unknown_keys_and_wrong_sections() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("bad.toml");
    for (content, message) in [
        ("[sampling]\ntemperatur = 0.5", "unknown serving setting"),
        ("[model]\ntemperature = 0.5", "belongs in [sampling]"),
    ] {
        std::fs::write(&path, content).unwrap();
        let error = parse_test(&["skippy", "serve", "--settings", path.to_str().unwrap()])
            .err()
            .unwrap();
        assert!(error.to_string().contains(message), "{error}");
    }
}

fn stage_file(dir: &Path) -> std::path::PathBuf {
    let path = dir.join("stage.json");
    std::fs::write(&path, skippy_config::example_config().to_string()).unwrap();
    path
}

#[test]
fn local_and_binary_settings_reach_runtime_consumers() {
    let dir = tempfile::tempdir().unwrap();
    let path = stage_file(dir.path());
    let cli = parse_test(&[
        "skippy",
        "serve",
        "--config",
        path.to_str().unwrap(),
        "--threads",
        "3",
        "--batch-size",
        "256",
        "--mmap=false",
        "--projector-gpu=false",
        "--temperature",
        "0.4",
        "--continuous-batching=false",
        "--compact=false",
        "--guardrails",
        "enforce",
        "--prefix-cache",
        "on",
        "--prefix-cache-max-entries",
        "5",
    ])
    .unwrap();
    let Command::Serve(mut args) = cli.command else {
        panic!()
    };
    let binary_settings = args.public.settings.clone();
    let local = crate::conversion::local_openai_options(args.public).unwrap();
    assert_eq!(local.config.n_batch, Some(256));
    assert_eq!(local.config.mmap, Some(false));
    assert_eq!(local.config.projector_use_gpu, Some(false));
    assert_eq!(local.config.kv_cache.as_ref().unwrap().max_entries, 5);
    assert_eq!(local.tuning.n_threads, Some(3));
    let frontend = local.resolved_openai_options().unwrap();
    assert_eq!(frontend.request_defaults.temperature, Some(0.4));
    assert!(!frontend.continuous_batching);
    assert!(
        !local
            .tuning
            .guardrails
            .as_ref()
            .unwrap()
            .compaction
            .unwrap()
            .enabled
    );
    args.stage.config = path;
    args.stage.api_bind_addr = Some("127.0.0.1:9337".parse().unwrap());
    args.stage.settings = binary_settings;
    let binary = crate::conversion::binary_stage_options(args.stage).unwrap();
    assert_eq!(binary.config.n_batch, local.config.n_batch);
    assert_eq!(
        binary.tuning.request_defaults,
        local.tuning.request_defaults
    );
    assert_eq!(binary.tuning.n_threads, Some(3));
    assert!(!binary.continuous_batching);
    assert_eq!(
        binary.tuning.guardrails.unwrap().policy.snapshot().mode,
        skippy_inference_api::GuardrailMode::Enforce
    );
}

#[test]
fn local_explicit_draft_controls_and_speculative_plan_are_consumed() {
    let dir = tempfile::tempdir().unwrap();
    let path = stage_file(dir.path());
    let cli = parse_test(&[
        "skippy",
        "serve",
        "--config",
        path.to_str().unwrap(),
        "--draft-model-path",
        "draft.gguf",
        "--speculative-window",
        "7",
        "--draft-n-gpu-layers",
        "-1",
        "--draft-threads",
        "2",
        "--speculative-strategy",
        "draft-model",
    ])
    .unwrap();
    let Command::Serve(args) = cli.command else {
        panic!()
    };
    let local = crate::conversion::local_openai_options(args.public).unwrap();
    let frontend = local.resolved_openai_options().unwrap();
    assert_eq!(
        frontend.draft_model_path.unwrap(),
        std::path::PathBuf::from("draft.gguf")
    );
    assert_eq!(frontend.speculative_window, 7);
    assert_eq!(frontend.draft_n_gpu_layers, Some(-1));
    assert_eq!(frontend.speculative.draft_threads, Some(2));
    assert!(!frontend.native_mtp_enabled);
}

#[test]
fn disabled_speculation_suppresses_configured_draft_model_path() {
    assert_disabled_speculation_suppresses_draft_path("--draft-model-path");
}

#[test]
fn disabled_speculation_suppresses_configured_native_mtp_draft_model_path() {
    assert_disabled_speculation_suppresses_draft_path("--native-mtp-draft-model-path");
}

fn assert_disabled_speculation_suppresses_draft_path(draft_flag: &str) {
    let dir = tempfile::tempdir().unwrap();
    let path = stage_file(dir.path());
    let missing_draft = dir.path().join("unavailable-draft.gguf");
    assert!(!missing_draft.exists());
    let Command::Serve(args) = parse_test(&[
        "skippy",
        "serve",
        "--config",
        path.to_str().unwrap(),
        draft_flag,
        missing_draft.to_str().unwrap(),
        "--speculative-strategy",
        "disabled",
    ])
    .unwrap()
    .command
    else {
        panic!()
    };
    let local = crate::conversion::local_openai_options(args.public).unwrap();
    assert_eq!(local.tuning.draft_model_path, None);
    assert_eq!(local.tuning.native_mtp_draft_model_path, None);
    let frontend = local.resolved_openai_options().unwrap();
    assert_eq!(frontend.speculative.effective_strategy, "disabled");
    assert_eq!(frontend.draft_model_path, None);
    assert_eq!(frontend.native_mtp_draft_model_path, None);
    assert!(!frontend.native_mtp_enabled);
}

#[test]
fn disabled_speculation_suppresses_binary_configured_draft_model_path() {
    assert_binary_disabled_speculation_suppresses_draft_path("--draft-model-path");
}

#[test]
fn disabled_speculation_suppresses_binary_configured_native_mtp_draft_model_path() {
    assert_binary_disabled_speculation_suppresses_draft_path("--native-mtp-draft-model-path");
}

fn binary_test_options(argv: &[&str]) -> skippy_serving::binary_transport::BinaryStageOptions {
    let Command::Serve(mut args) = parse_test(argv).unwrap().command else {
        panic!()
    };
    crate::serve::apply_public_frontend_tuning(&args.public, &mut args.stage);
    args.stage.settings = args.public.settings;
    args.stage.config = args.public.config.unwrap();
    args.stage.api_bind_addr = Some("127.0.0.1:9337".parse().unwrap());
    crate::conversion::binary_stage_options(args.stage).unwrap()
}

fn assert_binary_disabled_speculation_suppresses_draft_path(draft_flag: &str) {
    let dir = tempfile::tempdir().unwrap();
    let path = stage_file(dir.path());
    let missing_draft = dir.path().join("unavailable-draft.gguf");
    assert!(!missing_draft.exists());
    let binary = binary_test_options(&[
        "skippy",
        "serve",
        "--config",
        path.to_str().unwrap(),
        "--stage-transport",
        "binary",
        draft_flag,
        missing_draft.to_str().unwrap(),
        "--speculative-strategy",
        "disabled",
    ]);
    let frontend = binary.openai.unwrap();
    assert_eq!(frontend.speculative.effective_strategy, "disabled");
    assert_eq!(frontend.draft_model_path, None);
    assert_eq!(frontend.native_mtp_draft_model_path, None);
    assert!(!binary.native_mtp_enabled);
}

fn stage_file_with_discoverable_sibling_draft(dir: &Path, native_mtp: bool) -> std::path::PathBuf {
    // Only the architecture header is needed for automatic pairing; no weights are loaded.
    let mut metadata = b"GGUF".to_vec();
    metadata.extend_from_slice(&3_u32.to_le_bytes());
    metadata.extend_from_slice(&0_u64.to_le_bytes());
    metadata.extend_from_slice(&1_u64.to_le_bytes());
    let key = "general.architecture";
    metadata.extend_from_slice(&(key.len() as u64).to_le_bytes());
    metadata.extend_from_slice(key.as_bytes());
    metadata.extend_from_slice(&8_u32.to_le_bytes());
    metadata.extend_from_slice(&5_u64.to_le_bytes());
    metadata.extend_from_slice(b"llama");
    let target = dir.join("target.gguf");
    let draft = dir.join("sibling-draft.gguf");
    std::fs::write(&target, &metadata).unwrap();
    std::fs::write(&draft, &metadata).unwrap();
    let mut automatic = skippy_api::serving::InferenceOptions::direct_single_stage_defaults(
        "test-model".into(),
        32,
        1,
        native_mtp,
    );
    skippy_api::speculative::apply_auto_speculation(&mut automatic, &target);
    if native_mtp {
        assert_eq!(automatic.native_mtp_draft_model_path, Some(draft));
    } else {
        assert_eq!(automatic.draft_model_path, Some(draft));
    }
    let mut config = skippy_config::example_config();
    config["model_path"] = serde_json::json!(target);
    config["native_mtp_enabled"] = serde_json::json!(native_mtp);
    let path = dir.join("stage.json");
    std::fs::write(&path, config.to_string()).unwrap();
    path
}

#[test]
fn disabled_speculation_suppresses_local_automatically_discovered_draft_paths() {
    for native_mtp in [false, true] {
        let dir = tempfile::tempdir().unwrap();
        let path = stage_file_with_discoverable_sibling_draft(dir.path(), native_mtp);
        let Command::Serve(args) = parse_test(&[
            "skippy",
            "serve",
            "--config",
            path.to_str().unwrap(),
            "--speculative-strategy",
            "disabled",
        ])
        .unwrap()
        .command
        else {
            panic!()
        };
        let local = crate::conversion::local_openai_options(args.public).unwrap();
        let frontend = local.resolved_openai_options().unwrap();
        assert_eq!(frontend.speculative.effective_strategy, "disabled");
        assert_eq!(frontend.draft_model_path, None, "native_mtp={native_mtp}");
        assert_eq!(
            frontend.native_mtp_draft_model_path, None,
            "native_mtp={native_mtp}"
        );
        assert!(!frontend.native_mtp_enabled);
    }
}

#[test]
fn disabled_speculation_suppresses_binary_automatically_discovered_draft_paths() {
    for native_mtp in [false, true] {
        let dir = tempfile::tempdir().unwrap();
        let path = stage_file_with_discoverable_sibling_draft(dir.path(), native_mtp);
        let binary = binary_test_options(&[
            "skippy",
            "serve",
            "--config",
            path.to_str().unwrap(),
            "--stage-transport",
            "binary",
            "--speculative-strategy",
            "disabled",
        ]);
        let frontend = binary.openai.unwrap();
        assert_eq!(frontend.speculative.effective_strategy, "disabled");
        assert_eq!(frontend.draft_model_path, None, "native_mtp={native_mtp}");
        assert_eq!(
            frontend.native_mtp_draft_model_path, None,
            "native_mtp={native_mtp}"
        );
        assert!(!binary.native_mtp_enabled);
    }
}

#[test]
fn auto_speculation_preserves_local_and_binary_discovered_draft_paths() {
    for native_mtp in [false, true] {
        let dir = tempfile::tempdir().unwrap();
        let path = stage_file_with_discoverable_sibling_draft(dir.path(), native_mtp);
        let expected_draft = dir.path().join("sibling-draft.gguf");
        let argv = [
            "skippy",
            "serve",
            "--config",
            path.to_str().unwrap(),
            "--speculative-strategy",
            "auto",
        ];
        let Command::Serve(args) = parse_test(&argv).unwrap().command else {
            panic!()
        };
        let local = crate::conversion::local_openai_options(args.public).unwrap();
        let frontend = local.resolved_openai_options().unwrap();
        let binary = binary_test_options(&argv);
        let binary_frontend = binary.openai.unwrap();
        if native_mtp {
            assert_eq!(
                frontend.native_mtp_draft_model_path,
                Some(expected_draft.clone())
            );
            assert_eq!(
                binary_frontend.native_mtp_draft_model_path,
                Some(expected_draft)
            );
        } else {
            assert_eq!(frontend.draft_model_path, Some(expected_draft.clone()));
            assert_eq!(binary_frontend.draft_model_path, Some(expected_draft));
        }
    }
}

#[test]
fn binary_default_strategy_preserves_explicit_draft_paths() {
    let dir = tempfile::tempdir().unwrap();
    let path = stage_file(dir.path());
    for draft_flag in ["--draft-model-path", "--native-mtp-draft-model-path"] {
        let draft = dir.path().join("configured-draft.gguf");
        let binary = binary_test_options(&[
            "skippy",
            "serve",
            "--config",
            path.to_str().unwrap(),
            "--stage-transport",
            "binary",
            draft_flag,
            draft.to_str().unwrap(),
        ]);
        let frontend = binary.openai.unwrap();
        if draft_flag == "--draft-model-path" {
            assert_eq!(frontend.draft_model_path, Some(draft));
        } else {
            assert_eq!(frontend.native_mtp_draft_model_path, Some(draft));
        }
    }
}

#[test]
fn other_speculative_strategies_preserve_configured_draft_paths() {
    for strategy in ["auto", "draft-model", "native-mtp", "ngram", "mtp-ngram"] {
        let Command::Serve(args) = parse_test(&[
            "skippy",
            "serve",
            "--model",
            "missing.gguf",
            "--draft-model-path",
            "draft.gguf",
            "--native-mtp-draft-model-path",
            "mtp-draft.gguf",
            "--speculative-strategy",
            strategy,
        ])
        .unwrap()
        .command
        else {
            panic!()
        };
        let tuning = args
            .public
            .settings
            .tuning(args.public.openai_guardrails)
            .unwrap();
        assert_eq!(
            tuning.draft_model_path.as_deref(),
            Some(Path::new("draft.gguf")),
            "{strategy}"
        );
        assert_eq!(
            tuning.native_mtp_draft_model_path.as_deref(),
            Some(Path::new("mtp-draft.gguf")),
            "{strategy}"
        );
    }
}

#[test]
fn speculative_strategy_native_mtp_conflicts_are_rejected() {
    for (strategy, native_mtp) in [
        ("disabled", "--native-mtp=true"),
        ("draft-model", "--native-mtp=true"),
        ("ngram", "--native-mtp=true"),
        ("native-mtp", "--native-mtp=false"),
        ("mtp-ngram", "--native-mtp=false"),
    ] {
        let Command::Serve(args) = parse_test(&[
            "skippy",
            "serve",
            "--model",
            "missing.gguf",
            "--draft-model-path",
            "draft.gguf",
            "--native-mtp-draft-model-path",
            "mtp-draft.gguf",
            "--speculative-strategy",
            strategy,
            native_mtp,
        ])
        .unwrap()
        .command
        else {
            panic!()
        };
        let error = args
            .public
            .settings
            .speculative(skippy_serving::SpeculativeDecodeConfig::default(), true)
            .unwrap_err();
        assert!(
            error.to_string().contains("conflicts"),
            "{strategy}: {error}"
        );
    }
}

#[test]
fn invalid_controls_fail_before_loading_models() {
    for (flags, message) in [
        (vec!["--threads", "0"], "threads"),
        (vec!["--temperature", "-1"], "sampling"),
        (
            vec![
                "--compact-target-percent",
                "99",
                "--compact-trigger-percent",
                "90",
            ],
            "compaction",
        ),
    ] {
        let mut argv = vec!["skippy", "serve", "--model", "missing.gguf"];
        argv.extend(flags);
        let Command::Serve(args) = parse_test(&argv).unwrap().command else {
            panic!()
        };
        let error = args
            .public
            .settings
            .tuning(args.public.openai_guardrails)
            .unwrap_err();
        assert!(error.to_string().contains(message), "{error}");
    }
    let Command::Serve(args) = parse_test(&[
        "skippy",
        "serve",
        "--model",
        "missing.gguf",
        "--prefix-cache-ram",
        "1GiB",
    ])
    .unwrap()
    .command
    else {
        panic!()
    };
    let mut stage: StageConfig = serde_json::from_value(skippy_config::example_config()).unwrap();
    args.public.settings.apply_stage(&mut stage).unwrap();
    assert!(
        args.public
            .settings
            .validate_cache_dependencies(&stage, false)
            .unwrap_err()
            .to_string()
            .contains("requires")
    );
}

#[test]
fn request_default_catalog_covers_all_operator_fields() {
    let fields =
        serde_json::to_value(skippy_serving::EmbeddedOpenAiRequestDefaults::default()).unwrap();
    for field in fields.as_object().unwrap().keys() {
        let target = format!("request.{field}");
        assert!(OPTIONS.iter().any(|spec| spec.target == target || spec.target.starts_with(&format!("{target}."))), "missing control for {field}");
    }
}

#[test]
fn global_booleans_and_automatic_forwarding_can_be_explicitly_disabled() {
    let cli = parse_test(&[
        "skippy",
        "serve",
        "--model",
        "missing.gguf",
        "--debug=false",
    ])
    .unwrap();
    assert!(!cli.debug);
    let Command::Serve(args) = cli.command else {
        panic!()
    };
    assert!(!args.prompt);
    let dir = tempfile::tempdir().unwrap();
    let path = stage_file(dir.path());
    let Command::Serve(mut args) = parse_test(&[
        "skippy",
        "serve",
        "--config",
        path.to_str().unwrap(),
        "--stage-transport",
        "binary",
        "--async-prefill-forward=false",
        "--adaptive-speculative-window=false",
    ])
    .unwrap()
    .command
    else {
        panic!()
    };
    args.stage.config = path;
    args.stage.settings = args.public.settings;
    let options = crate::conversion::binary_stage_options(args.stage).unwrap();
    assert!(!options.async_prefill_forward);
}

#[test]
fn modes_and_budget_dependencies_produce_actionable_errors() {
    let Command::Serve(args) = parse_test(&[
        "skippy",
        "serve",
        "--model",
        "missing.gguf",
        "--activation-codec",
        "f16-rne-v1",
    ])
    .unwrap()
    .command
    else {
        panic!()
    };
    assert!(
        args.public
            .settings
            .validate_mode(&args)
            .unwrap_err()
            .to_string()
            .contains("requires --stage-transport")
    );
    let Command::Serve(args) = parse_test(&[
        "skippy",
        "serve",
        "--config",
        "stage.json",
        "--stage-transport",
        "binary",
        "--worker-only",
        "--temperature",
        "0.2",
    ])
    .unwrap()
    .command
    else {
        panic!()
    };
    assert!(
        args.public
            .settings
            .validate_mode(&args)
            .unwrap_err()
            .to_string()
            .contains("cannot be used with --worker-only")
    );
    let Command::Serve(args) = parse_test(&[
        "skippy",
        "serve",
        "--model",
        "missing.gguf",
        "--prefix-cache-exact-budget",
        "2GiB",
    ])
    .unwrap()
    .command
    else {
        panic!()
    };
    let mut stage: StageConfig = serde_json::from_value(skippy_config::example_config()).unwrap();
    args.public.settings.apply_stage(&mut stage).unwrap();
    assert_eq!(
        stage.kv_cache.unwrap().exact_max_bytes,
        Some(2 * 1024 * 1024 * 1024)
    );
}

#[test]
fn documented_example_is_a_valid_complete_settings_file() {
    let package_root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    let path = package_root.join("../../docs/SERVING_SETTINGS.md");
    // The repository guide is outside the published crate package.
    if !path.exists() {
        assert!(
            !package_root.join("../../../.git").exists(),
            "missing repository serving settings guide"
        );
        return;
    }
    let guide = std::fs::read_to_string(path).unwrap();
    let (_, text) = guide.split_once("```toml\n").unwrap();
    let (text, _) = text.split_once("```").unwrap();
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("serve.toml");
    std::fs::write(&path, text).unwrap();
    let Command::Serve(args) =
        parse_test(&["skippy", "serve", "--settings", path.to_str().unwrap()])
            .unwrap()
            .command
    else {
        panic!()
    };
    let tuning = args
        .public
        .settings
        .tuning(args.public.openai_guardrails)
        .unwrap();
    assert_eq!(tuning.n_threads, Some(4));
    assert_eq!(
        tuning.request_defaults.reasoning_budget,
        Some(skippy_serving::EmbeddedReasoningBudget::Auto)
    );
    for spec in OPTIONS {
        assert!(
            text.contains("[model]") && guide.contains(&format!("`--{}`", spec.name)),
            "undocumented {}",
            spec.name
        );
    }
}
