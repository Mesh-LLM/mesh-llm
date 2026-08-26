use super::command_summary;
use clap::Parser;
use mesh_llm_events::CliCommandSummary;

fn assert_parsed_summaries(cases: &[(&[&str], &str)]) {
    for (args, marker) in cases {
        let cli = mesh_llm_cli::Cli::parse_from(*args);
        let command = cli.command.as_ref().expect("parsed command");
        let summary = command_summary(command)
            .map(|summary| summary.as_str().to_owned())
            .unwrap_or_else(|| panic!("producer summary for {args:?}"));
        assert!(summary.contains(marker), "missing {marker} in {summary}");
        assert!(
            CliCommandSummary::sanitize(&summary).is_some(),
            "rejected: {summary}"
        );
    }
}

#[test]
fn parsed_commands_emit_local_raw_options() {
    assert_parsed_summaries(&[
        (&["mesh-llm", "status", "--port", "41731"], "--port 41731"),
        (
            &["mesh-llm", "load", "private/model", "--port", "41731"],
            "--port 41731",
        ),
        (
            &["mesh-llm", "unload", "private/model", "--port", "41731"],
            "--port 41731",
        ),
        (&["mesh-llm", "goose", "--port", "41731"], "--port 41731"),
        (&["mesh-llm", "claude", "--port", "41731"], "--port 41731"),
        (
            &[
                "mesh-llm",
                "doctor",
                "split",
                "--model-ref",
                "private/model",
                "--port",
                "41731",
            ],
            "--port 41731",
        ),
        (
            &[
                "mesh-llm",
                "doctor",
                "--json",
                "split",
                "--model-ref",
                "private/model",
                "--port",
                "41731",
            ],
            "--port 41731",
        ),
        (
            &["mesh-llm", "gpus", "run-benchmark", "--backend", "cuda"],
            "--backend cuda",
        ),
        (
            &[
                "mesh-llm",
                "gpus",
                "--json",
                "run-benchmark",
                "--backend",
                "cuda",
            ],
            "--backend cuda",
        ),
    ]);
}

#[test]
fn parsed_commands_emit_runtime_raw_options() {
    assert_parsed_summaries(&[
        (
            &[
                "mesh-llm",
                "runtime",
                "guardrails",
                "--mode",
                "metrics",
                "--port",
                "41731",
            ],
            "--mode metrics",
        ),
        (
            &[
                "mesh-llm",
                "runtime",
                "guardrails",
                "--mode",
                "metrics",
                "--json",
                "--port",
                "41731",
            ],
            "--mode metrics",
        ),
        (
            &["mesh-llm", "runtime", "status", "--port", "41731"],
            "--port 41731",
        ),
        (
            &[
                "mesh-llm",
                "runtime",
                "load",
                "private/model",
                "--port",
                "41731",
            ],
            "--port 41731",
        ),
        (
            &[
                "mesh-llm",
                "runtime",
                "unload",
                "private/model",
                "--port",
                "41731",
            ],
            "--port 41731",
        ),
        (
            &["mesh-llm", "runtime", "bootstrap", "--port", "41731"],
            "--port 41731",
        ),
        (
            &[
                "mesh-llm",
                "runtime",
                "bootstrap",
                "--json",
                "--port",
                "41731",
            ],
            "--port 41731",
        ),
    ]);
}

#[test]
fn parsed_commands_emit_remote_raw_options() {
    assert_parsed_summaries(&[
        (
            &[
                "mesh-llm",
                "runtime",
                "get-config",
                "--endpoint",
                "private/endpoint",
                "--port",
                "41731",
            ],
            "--port 41731",
        ),
        (
            &[
                "mesh-llm",
                "runtime",
                "get-config",
                "--endpoint",
                "private/endpoint",
                "--json",
                "--port",
                "41731",
            ],
            "--port 41731",
        ),
        (
            &[
                "mesh-llm",
                "runtime",
                "load-model",
                "--endpoint",
                "private/endpoint",
                "--model",
                "private/model",
                "--port",
                "41731",
            ],
            "--port 41731",
        ),
        (
            &[
                "mesh-llm",
                "runtime",
                "load-model",
                "--endpoint",
                "private/endpoint",
                "--model",
                "private/model",
                "--profile",
                "profile-a",
                "--json",
                "--port",
                "41731",
            ],
            "--port 41731",
        ),
        (
            &[
                "mesh-llm",
                "runtime",
                "unload-model",
                "--endpoint",
                "private/endpoint",
                "--instance-id",
                "instance-a",
                "--json",
                "--port",
                "41731",
            ],
            "--port 41731",
        ),
        (
            &[
                "mesh-llm",
                "runtime",
                "apply-config",
                "--endpoint",
                "private/endpoint",
                "--expected-revision",
                "7",
                "--config",
                "/private/config.toml",
                "--port",
                "41731",
            ],
            "--port 41731",
        ),
        (
            &[
                "mesh-llm",
                "runtime",
                "apply-config",
                "--endpoint",
                "private/endpoint",
                "--expected-revision",
                "7",
                "--config",
                "/private/config.toml",
                "--json",
                "--port",
                "41731",
            ],
            "--port 41731",
        ),
    ]);
}
