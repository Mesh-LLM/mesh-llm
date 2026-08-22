use super::SummaryAssembly;

pub(super) fn format_benchmark(
    command: &mesh_llm_cli::benchmark::BenchmarkCommand,
    assembly: &mut SummaryAssembly,
) {
    use mesh_llm_cli::benchmark::BenchmarkCommand;
    match command {
        BenchmarkCommand::Tune(tune) => {
            assembly.command.push_str(" benchmark tune");
            assembly.redact("--model", tune.model.is_some());
            assembly.redact("--models", !tune.models.is_empty());
            assembly.flag("json", tune.json);
            assembly.redact("--ctx-sizes", !tune.ctx_sizes.is_empty());
            assembly.redact("--batch-sizes", !tune.batch_sizes.is_empty());
            assembly.redact("--ubatch-sizes", !tune.ubatch_sizes.is_empty());
            assembly.redact("--mmap-values", !tune.mmap_values.is_empty());
            assembly.redact("--mlock-values", !tune.mlock_values.is_empty());
            assembly.redact("--flash-attention", !tune.flash_attention.is_empty());
            assembly.redact("--speculative-types", !tune.speculative_types.is_empty());
            assembly.flag("no-speculative-tune", tune.no_speculative_tune);
            assembly.redact("--spec-draft-models", !tune.spec_draft_models.is_empty());
            assembly.redact(
                "--spec-draft-max-tokens",
                !tune.spec_draft_max_tokens.is_empty(),
            );
            assembly.redact(
                "--spec-draft-min-tokens",
                !tune.spec_draft_min_tokens.is_empty(),
            );
            assembly.redact("--spec-ngram-min", !tune.spec_ngram_min.is_empty());
            assembly.redact("--spec-ngram-max", !tune.spec_ngram_max.is_empty());
            assembly.redact(
                "--spec-draft-acceptance-threshold",
                !tune.spec_draft_acceptance_threshold.is_empty(),
            );
            assembly.redact(
                "--spec-draft-split-probability",
                !tune.spec_draft_split_probability.is_empty(),
            );
            assembly.flag("apply", tune.apply);
            assembly.flag("replace-existing", tune.replace_existing);
            assembly.flag("launch-args", tune.launch_args);
            assembly.redact(
                "--throughput-tolerance-pct",
                tune.throughput_tolerance_pct != 10.0,
            );
            assembly.redact("--max-tokens", tune.max_tokens != 128);
            assembly.redact("--startup-timeout-secs", tune.startup_timeout_secs != 600);
            assembly.redact("--request-timeout-secs", tune.request_timeout_secs != 600);
            assembly.flag("debug-telemetry", tune.debug_telemetry);
            assembly.redact(
                "--prompt",
                tune.prompt != "Write a concise paragraph about distributed GPU inference.",
            );
        }
        BenchmarkCommand::ImportPrompts {
            source: _,
            limit,
            max_tokens,
            output: _,
        } => {
            assembly.command.push_str(" benchmark import-prompts");
            assembly.redact("--source", true);
            assembly.redact("--limit", *limit != 20);
            assembly.redact("--max-tokens", max_tokens.is_some());
            assembly.redact("--output", true);
        }
    }
}
