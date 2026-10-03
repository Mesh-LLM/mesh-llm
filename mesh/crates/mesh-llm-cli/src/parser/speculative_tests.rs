//! Command-line surface for the split performance settings.

use super::commands::SpeculativeNgramFallbackCli;
use super::{Cli, normalize_runtime_surface_args};
use clap::Parser;

/// Both keys existed in `[models.speculative]` with no command-line
/// spelling, which left run-ahead admission — the strongest measured
/// speculation arm — reachable only from a config file.
#[test]
fn serve_parses_runahead_and_ngram_fallback() {
    let normalized = normalize_runtime_surface_args([
        "mesh-llm",
        "serve",
        "--speculative-verify-window-runahead-tokens",
        "96",
        "--speculative-ngram-fallback",
        "draft",
    ]);
    let cli = Cli::try_parse_from(normalized.normalized).expect("clap parse");

    assert_eq!(cli.speculative_verify_window_runahead_tokens, Some(96));
    assert_eq!(
        cli.speculative_ngram_fallback,
        Some(SpeculativeNgramFallbackCli::Draft)
    );
}

/// `none` is the documented off-switch, so a command line can countermand a
/// fallback that the config file or model defaults turned on.
#[test]
fn ngram_fallback_accepts_none_and_rejects_unknown_values() {
    let cli = Cli::parse_from([
        "mesh-llm",
        "--speculative-ngram-fallback",
        "none",
        "--model",
        "x.gguf",
    ]);
    assert_eq!(
        cli.speculative_ngram_fallback,
        Some(SpeculativeNgramFallbackCli::Off)
    );
    assert_eq!(
        cli.speculative_ngram_fallback.map(|value| value.as_str()),
        Some("none")
    );

    // `simple` was #1054's original fallback; it was consolidated away and
    // `model_validation` now allows only `draft` and `none`.
    Cli::try_parse_from([
        "mesh-llm",
        "--speculative-ngram-fallback",
        "simple",
        "--model",
        "x.gguf",
    ])
    .expect_err("simple is no longer a supported fallback");
}
