use std::num::{NonZeroU32, NonZeroUsize};

use super::Cli;
use clap::{CommandFactory, error::ErrorKind};

#[test]
fn load_accepts_context_size_and_lane_count() {
    let matches = Cli::command()
        .try_get_matches_from([
            "mesh-llm",
            "load",
            "model.gguf",
            "--ctx-size",
            "8192",
            "--parallel",
            "1",
        ])
        .expect("load should accept --ctx-size and --parallel");
    let load = matches
        .subcommand_matches("load")
        .expect("load subcommand matches");
    assert_eq!(
        load.get_one::<NonZeroU32>("ctx_size")
            .map(|value| value.get()),
        Some(8192)
    );
    assert_eq!(
        load.get_one::<NonZeroUsize>("parallel")
            .map(|value| value.get()),
        Some(1)
    );
}

#[test]
fn load_rejects_zero_context_size_and_lanes() {
    for flag in ["--ctx-size", "--parallel"] {
        let error = Cli::command()
            .try_get_matches_from(["mesh-llm", "load", "model.gguf", flag, "0"])
            .expect_err("zero is not a usable context size or lane count");
        assert_eq!(error.kind(), ErrorKind::ValueValidation, "{flag}: {error}");
    }
}
