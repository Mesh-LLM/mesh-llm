use super::{Descriptor, NONE, REDACTED_NAME, RawKind, descriptor, descriptor_with_conflicts};

const PLUGIN_INSTALL_CONFLICTS: &[&[&str]] = &[&["reference", "--archive"]];
const TUNE_FLAGS: &[&str] = &[
    "--json",
    "--no-speculative-tune",
    "--apply",
    "--replace-existing",
    "--launch-args",
    "--debug-telemetry",
];
const TUNE_CONFLICTS: &[&[&str]] = &[
    &["--model", "--models"],
    &["--no-speculative-tune", "--speculative-types"],
    &["--no-speculative-tune", "--spec-draft-models"],
    &["--no-speculative-tune", "--spec-draft-max-tokens"],
    &["--no-speculative-tune", "--spec-draft-min-tokens"],
    &["--no-speculative-tune", "--spec-draft-acceptance-threshold"],
    &["--no-speculative-tune", "--spec-draft-split-probability"],
    &["--no-speculative-tune", "--spec-ngram-min"],
    &["--no-speculative-tune", "--spec-ngram-max"],
];

pub(super) const PLUGIN_DESCRIPTORS: &[Descriptor] = &[
    descriptor_with_conflicts(
        &["mesh-llm", "plugins", "install"],
        NONE,
        &["reference", "--archive", "--name", "--version"],
        PLUGIN_INSTALL_CONFLICTS,
        false,
        RawKind::None,
    ),
    descriptor(
        &["mesh-llm", "plugins", "update"],
        NONE,
        REDACTED_NAME,
        false,
        RawKind::None,
    ),
    descriptor(
        &["mesh-llm", "plugins", "enable"],
        NONE,
        REDACTED_NAME,
        false,
        RawKind::None,
    ),
    descriptor(
        &["mesh-llm", "plugins", "disable"],
        NONE,
        REDACTED_NAME,
        false,
        RawKind::None,
    ),
    descriptor(
        &["mesh-llm", "plugins", "delete"],
        NONE,
        REDACTED_NAME,
        false,
        RawKind::None,
    ),
    descriptor(
        &["mesh-llm", "plugins", "info"],
        NONE,
        REDACTED_NAME,
        false,
        RawKind::None,
    ),
    descriptor(
        &["mesh-llm", "plugins", "search"],
        NONE,
        &["query"],
        false,
        RawKind::None,
    ),
    descriptor(
        &["mesh-llm", "plugins", "list"],
        NONE,
        NONE,
        false,
        RawKind::None,
    ),
];

pub(super) const BENCHMARK_DESCRIPTORS: &[Descriptor] = &[
    descriptor_with_conflicts(
        &["mesh-llm", "benchmark", "tune"],
        TUNE_FLAGS,
        &[
            "--model",
            "--models",
            "--ctx-sizes",
            "--batch-sizes",
            "--ubatch-sizes",
            "--mmap-values",
            "--mlock-values",
            "--flash-attention",
            "--speculative-types",
            "--spec-draft-models",
            "--spec-draft-max-tokens",
            "--spec-draft-min-tokens",
            "--spec-ngram-min",
            "--spec-ngram-max",
            "--spec-draft-acceptance-threshold",
            "--spec-draft-split-probability",
            "--throughput-tolerance-pct",
            "--max-tokens",
            "--startup-timeout-secs",
            "--request-timeout-secs",
            "--prompt",
        ],
        TUNE_CONFLICTS,
        false,
        RawKind::None,
    ),
    descriptor(
        &["mesh-llm", "benchmark", "import-prompts"],
        NONE,
        &["--source", "--limit", "--max-tokens", "--output"],
        false,
        RawKind::None,
    ),
];
