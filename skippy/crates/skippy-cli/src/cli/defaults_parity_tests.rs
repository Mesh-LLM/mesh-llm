use super::*;
use skippy_api::serving::InferenceOptions;

#[test]
fn standalone_frontends_use_skippy_defaults_for_every_exposed_tuning_control() {
    let cli = Cli::try_parse_from(["skippy", "serve", "--model", "model.gguf"]).unwrap();
    let Command::Serve(args) = cli.command else {
        panic!("expected serve");
    };
    let defaults = InferenceOptions::direct_single_stage_defaults(
        "model".into(),
        skippy_config::local_serving::MAX_OUTPUT_TOKENS,
        skippy_config::local_serving::PARALLEL,
        true,
    );
    let public = args.public;
    let stage = args.stage;
    assert_eq!(public.default_max_tokens, defaults.default_max_tokens);
    assert_eq!(stage.openai_default_max_tokens, defaults.default_max_tokens);
    assert_eq!(public.generation_concurrency, None);
    assert_eq!(stage.openai_generation_concurrency, None);
    assert_eq!(public.generation_queue_capacity, None);
    assert_eq!(stage.openai_generation_queue_capacity, None);
    assert_eq!(
        public.adaptive_generation_concurrency,
        defaults.adaptive_generation_min_concurrency.is_some()
    );
    assert_eq!(
        stage.openai_adaptive_generation_concurrency,
        defaults.adaptive_generation_min_concurrency.is_some()
    );
    assert_eq!(
        public.generation_admission_timeout_secs,
        defaults.generation_admission_timeout_secs
    );
    assert_eq!(
        stage.openai_generation_admission_timeout_secs,
        defaults.generation_admission_timeout_secs
    );
    assert_eq!(public.prefill_chunk_size, defaults.prefill_chunk_size);
    assert_eq!(stage.openai_prefill_chunk_size, defaults.prefill_chunk_size);
    assert_eq!(public.prefill_chunk_policy, defaults.prefill_chunk_policy);
    assert_eq!(
        stage.openai_prefill_chunk_policy,
        defaults.prefill_chunk_policy
    );
    assert_eq!(
        public.prefill_adaptive_start,
        defaults.prefill_adaptive_start
    );
    assert_eq!(
        stage.openai_prefill_adaptive_start,
        defaults.prefill_adaptive_start
    );
    assert_eq!(public.prefill_adaptive_step, defaults.prefill_adaptive_step);
    assert_eq!(
        stage.openai_prefill_adaptive_step,
        defaults.prefill_adaptive_step
    );
    assert_eq!(public.prefill_adaptive_max, defaults.prefill_adaptive_max);
    assert_eq!(
        stage.openai_prefill_adaptive_max,
        defaults.prefill_adaptive_max
    );
    assert_eq!(
        public.prefill_adaptive_target_ms,
        defaults.prefill_adaptive_target_ms
    );
    assert_eq!(
        stage.openai_prefill_adaptive_target_ms,
        defaults.prefill_adaptive_target_ms
    );
    assert_eq!(
        stage.downstream_connect_timeout_secs,
        defaults.downstream_connect_timeout_secs
    );
    assert_eq!(public.openai_guardrails, OpenAiGuardrailsCliMode::Disabled);
    assert_eq!(
        stage.openai_speculative_window,
        skippy_config::local_serving::DRAFT_MODEL_TOKENS
    );
    assert_eq!(public.ctx_size, None);
    assert_eq!(public.n_gpu_layers, None);
}

#[test]
fn prefix_free_serving_flags_reach_binary_frontend() {
    let cli = Cli::try_parse_from([
        "skippy",
        "serve",
        "--config",
        "stage.json",
        "--stage-transport",
        "binary",
        "--bind-addr",
        "127.0.0.1:9444",
        "--model-id",
        "custom-model",
        "--default-max-tokens",
        "2048",
        "--generation-concurrency",
        "2",
        "--prefill-chunk-size",
        "128",
        "--speculative-config",
        "plan.json",
        "--draft-model-path",
        "draft.gguf",
        "--speculative-window",
        "5",
    ])
    .unwrap();
    let Command::Serve(mut args) = cli.command else {
        panic!("expected serve");
    };
    crate::serve::apply_public_frontend_tuning(&args.public, &mut args.stage);
    assert_eq!(args.public.bind_addr.unwrap().port(), 9444);
    assert_eq!(args.stage.openai_model_id.as_deref(), Some("custom-model"));
    assert_eq!(args.stage.openai_default_max_tokens, 2048);
    assert_eq!(args.stage.openai_generation_concurrency, Some(2));
    assert_eq!(args.stage.openai_prefill_chunk_size, 128);
    assert_eq!(
        args.stage.openai_speculative_config,
        Some(PathBuf::from("plan.json"))
    );
    assert_eq!(
        args.stage.openai_draft_model_path,
        Some(PathBuf::from("draft.gguf"))
    );
    assert_eq!(args.stage.openai_speculative_window, 5);
}

#[test]
fn serve_rejects_prefixed_tuning_flags() {
    for flag in [
        "--api-generation-concurrency",
        "--openai-generation-concurrency",
    ] {
        assert!(
            Cli::try_parse_from(["skippy", "serve", "--model", "model.gguf", flag, "2",]).is_err()
        );
    }
}

#[test]
fn worker_transport_only_accepts_binary() {
    for (transport, accepted) in [("binary", true), ("http", false)] {
        let result = Cli::try_parse_from([
            "skippy",
            "serve",
            "--config",
            "stage.json",
            "--worker-only",
            "--stage-transport",
            transport,
        ]);
        if accepted {
            assert!(result.is_ok());
        } else {
            assert_eq!(
                result.err().unwrap().kind(),
                clap::error::ErrorKind::InvalidValue
            );
        }
    }
}
