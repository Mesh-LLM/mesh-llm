use super::{Prepared, Profile, command};
use crate::command::DynResult;
use std::{
    path::Path,
    time::{Duration, Instant},
};

pub(super) fn arguments(input: &Prepared, port: u16, concurrency: u32) -> Vec<String> {
    let mut args = command::strings(&[
        "profile",
        "--url",
        &format!("http://127.0.0.1:{port}"),
        "--endpoint",
        "/v1/chat/completions",
        "--model",
        &input.model.key,
        "--tokenizer",
    ]);
    args.push(input.model.tokenizer.clone());
    args.extend(command::strings(&[
        "--tokenizer-revision",
        &input.model.tokenizer_revision,
    ]));
    args.extend(command::strings(&[
        "--custom-dataset-type",
        "weka_trace",
        "--input-file",
    ]));
    args.push(input.dataset.to_string_lossy().into());
    args.extend(command::strings(&[
        "--session-header",
        "X-Session-ID",
        "--random-seed",
        "42",
        "--concurrency",
        &concurrency.to_string(),
        "--unsafe-override",
        "--output-artifact-dir",
    ]));
    args.push(input.output.join("upstream").to_string_lossy().into());
    if input.profile == Profile::Smoke {
        args.extend(command::strings(&[
            "--num-dataset-entries",
            "4",
            "--benchmark-duration",
            "60",
            "--warmup-requests-per-lane",
            "1",
            "--trajectory-start-min-ratio",
            "0",
            "--trajectory-start-max-ratio",
            "0",
        ]));
    }
    args
}
pub(super) fn run(input: &Prepared, port: u16, concurrency: u32) -> DynResult<()> {
    ready(&input.root, port, &input.output)?;
    let executable = input.root.join("skippy/evals/agentx/.venv/bin/agentx");
    let spec = command::spec(
        &input.root,
        executable.to_str().ok_or("AgentX executable")?,
        &arguments(input, port, concurrency),
    )?;
    command::run(
        &spec,
        if input.profile == Profile::Smoke {
            3600
        } else {
            9000
        },
        &input.output.join("agentx"),
    )
}
fn ready(root: &Path, port: u16, output: &Path) -> DynResult<()> {
    let spec = command::spec(
        root,
        "curl",
        &command::strings(&[
            "--fail",
            "--silent",
            "--show-error",
            "--max-time",
            "5",
            &format!("http://127.0.0.1:{port}/health"),
        ]),
    )?;
    let deadline = Instant::now() + Duration::from_secs(900);
    let mut attempt = 0;
    while Instant::now() < deadline {
        let log = output.join(format!("health-{attempt}"));
        if command::run(&spec, 10, &log).is_ok() {
            return Ok(());
        }
        attempt += 1;
        std::thread::sleep(Duration::from_secs(2));
    }
    Err("server never became healthy within 900s".into())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn smoke_changes_workload_size_without_shortening_context_or_outputs() {
        let models: Vec<super::super::Model> = serde_json::from_str(include_str!(
            "../../../../../skippy/evals/agentx/models.json"
        ))
        .unwrap();
        let mut input = Prepared {
            root: "/repo".into(),
            output: "/results".into(),
            weights: "/weights/model.gguf".into(),
            dataset: "/traces".into(),
            model: models[1].clone(),
            profile: Profile::Smoke,
        };
        let smoke = arguments(&input, 8080, 1);
        for pair in [
            ["--benchmark-duration", "60"],
            ["--num-dataset-entries", "4"],
            ["--warmup-requests-per-lane", "1"],
            ["--trajectory-start-min-ratio", "0"],
        ] {
            assert!(smoke.windows(2).any(|args| args == pair));
        }
        assert!(
            !smoke
                .iter()
                .any(|arg| arg == "--max-context-length" || arg == "--output-tokens")
        );
        input.profile = Profile::Benchmark;
        let full = arguments(&input, 8080, 8);
        assert!(
            !full
                .iter()
                .any(|arg| arg == "--benchmark-duration" || arg == "--num-dataset-entries")
        );
        assert!(full.windows(2).any(|args| args == ["--concurrency", "8"]));
    }
}
