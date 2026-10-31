use super::{Prepared, Profile, command};
use crate::command::DynResult;
use std::{
    path::Path,
    time::{Duration, Instant},
};

pub(super) fn arguments(input: &Prepared, port: u16, concurrency: u32) -> Vec<String> {
    if matches!(input.profile, Profile::Comparison | Profile::Contention) {
        let name = if input.profile == Profile::Comparison {
            "comparison-config.json"
        } else {
            "contention-config.json"
        };
        return vec![
            "profile".into(),
            "--config".into(),
            input.output.join(name).to_string_lossy().into_owned(),
        ];
    }
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
    // The locked client's offline resolver accepts repository IDs, not paths.
    args.push(input.model.tokenizer.clone());
    args.extend(command::strings(&[
        "--tokenizer-revision",
        &input.model.tokenizer_revision,
    ]));
    args.extend(command::strings(&[
        "--custom-dataset-type",
        "weka_trace",
        // Upstream drops whole traces before selecting the first N eligible
        // entries, including parent and child input + requested output.
        "--max-context-length",
        &super::trace_context::LIMIT.to_string(),
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
        "--export-level",
        "records",
        "--export-http-trace",
        "--output-artifact-dir",
    ]));
    args.push(input.output.join("upstream").to_string_lossy().into());
    args.extend(command::strings(&["--benchmark-grace-period", "inf"]));
    if input.profile == Profile::Smoke {
        args.extend(command::strings(&[
            "--num-dataset-entries",
            &concurrency.max(4).to_string(),
            "--benchmark-duration",
            "60",
            "--warmup-requests-per-lane",
            "1",
            // Recorded first-request spacing can exceed the entire smoke
            // window (81.4s in the pinned c4 cohort). Admit every lane at the
            // phase boundary while preserving later inter-turn delays.
            "--burst-phase-starts",
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
    match input.profile {
        Profile::Comparison => super::write(
            &input.output.join("comparison-config.json"),
            &super::comparison_cohort::configuration(input, port, concurrency)?,
        )?,
        Profile::Contention => super::write(
            &input.output.join("contention-config.json"),
            &super::contention_cohort::configuration(input, port, concurrency)?,
        )?,
        _ => {}
    }
    // JY's macOS work directory exceeds sockaddr_un's 103-byte path limit
    // after AgentX appends its temporary directory and ZMQ socket names.
    // Own a short private directory until the supervised harness has stopped.
    let ipc = ipc_directory()?;
    let executable = input.tools.as_ref().map_or_else(
        || input.root.join("skippy/evals/agentx/.venv/bin/agentx"),
        |tools| tools.client.path.clone(),
    );
    let mut spec = command::spec(
        &input.root,
        executable.to_str().ok_or("AgentX executable")?,
        &arguments(input, port, concurrency),
    )?;
    // A 1.0.7 mmap cache hit can finish dataset configuration before record
    // processors subscribe, leaving completed requests without exported metrics.
    // Rebuild the dataset before measurement for both engines; keep HF weights.
    spec.environment.insert(
        "AIPERF_DATASET_MMAP_CACHE_ENABLED".into(),
        crate::process::Value::Public("false".into()),
    );
    if input.profile == Profile::Contention {
        // The 16-process Weka reconstruction path spent minutes before
        // dispatch on Apple Silicon. Serial reconstruction completed in
        // seconds for this capped cohort and avoids competing with serving.
        spec.environment.insert(
            "AIPERF_DATASET_WEKA_PARALLEL_WORKERS".into(),
            crate::process::Value::Public("1".into()),
        );
    }
    spec.environment.insert(
        "TMPDIR".into(),
        crate::process::Value::Public(ipc.path().as_os_str().to_owned()),
    );
    command::run(
        &spec,
        if matches!(input.profile, Profile::Smoke | Profile::Contention) {
            3600
        } else {
            9000
        },
        &input.output.join("agentx"),
    )
}
fn ipc_directory() -> DynResult<tempfile::TempDir> {
    Ok(tempfile::Builder::new()
        .prefix("agentx-")
        .tempdir_in("/tmp")?)
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
    let seconds = super::lifecycle::STARTUP_TIMEOUT_SECS;
    let deadline = Instant::now() + Duration::from_secs(seconds);
    let mut attempt = 0;
    while Instant::now() < deadline {
        let log = output.join(format!("health-{attempt}"));
        if command::run(&spec, 10, &log).is_ok() {
            return Ok(());
        }
        attempt += 1;
        std::thread::sleep(Duration::from_secs(2));
    }
    Err(format!("server never became healthy within {seconds}s").into())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[cfg(unix)]
    #[test]
    fn private_ipc_directory_fits_agentx_unix_socket_paths() {
        let directory = ipc_directory().unwrap();
        let path = directory.path().to_owned();
        let socket = path.join("tmp01234567/aiperf/event_bus_proxy_frontend.ipc");
        assert!(socket.as_os_str().as_encoded_bytes().len() < 103);
        std::fs::create_dir_all(socket.parent().unwrap()).unwrap();
        let listener = std::os::unix::net::UnixListener::bind(&socket).unwrap();
        drop(listener);
        drop(directory);
        assert!(!path.exists());
    }

    #[test]
    fn smoke_changes_workload_size_without_shortening_context_or_outputs() {
        let models: Vec<super::super::Model> = serde_json::from_str(include_str!(
            "../../../../../skippy/evals/kv-validation/models.json"
        ))
        .unwrap();
        let mut input = Prepared {
            tools: None,
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
        assert!(!smoke.iter().any(|arg| arg == "--synthesis-max-isl"
            || arg == "--synthesis-max-osl"
            || arg == "--output-tokens"));
        assert!(
            smoke
                .windows(2)
                .any(|args| args == ["--max-context-length", "131072"])
        );
        assert!(smoke.iter().any(|arg| arg == "--burst-phase-starts"));
        let wide = arguments(&input, 8080, 128);
        assert!(
            wide.windows(2)
                .any(|args| args == ["--num-dataset-entries", "128"])
        );
        assert!(wide.windows(2).any(|args| args == ["--concurrency", "128"]));
        assert!(
            wide.windows(2)
                .any(|args| args == ["--benchmark-grace-period", "inf"])
        );
        input.profile = Profile::Benchmark;

        let full = arguments(&input, 8080, 8);
        assert!(!full.iter().any(|arg| matches!(
            arg.as_str(),
            "--benchmark-duration" | "--num-dataset-entries" | "--burst-phase-starts"
        )));
        assert!(full.windows(2).any(|args| args == ["--concurrency", "8"]));
        assert!(
            full.windows(2)
                .any(|args| args == ["--max-context-length", "131072"])
        );
        input.profile = Profile::Contention;
        for concurrency in [4, 8, 16] {
            let diagnostic = arguments(&input, 8080, concurrency);
            assert_eq!(
                diagnostic,
                ["profile", "--config", "/results/contention-config.json"]
            );
        }
    }
}
