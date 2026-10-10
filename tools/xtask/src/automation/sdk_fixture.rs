mod coordinator;
mod options;
mod readiness;
use super::{daemon_readiness::ports::Reservation, private_state::PrivateState};
use crate::{
    command::DynResult,
    process::{
        self, ProcessSpec, Value,
        retained::{Launch, MemberId},
    },
};
use coordinator::Owner;
use std::{path::Path, time::Duration};

#[derive(thiserror::Error)]
enum Failure {
    #[error("SDK fixture process session failed")]
    Session {
        report: crate::process::retained::Report<String>,
    },
    #[error("SDK readiness worker panicked; preceding result retained")]
    Worker {
        preceding: Result<
            crate::process::retained::Report<String>,
            super::retained_session::Error<String>,
        >,
    },
    #[error("SDK state cleanup failed ({kind:?}, OS code {code:?})")]
    State {
        kind: std::io::ErrorKind,
        code: Option<i32>,
        preceding: Option<Box<Box<dyn std::error::Error>>>,
    },
}

impl std::fmt::Debug for Failure {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Display::fmt(self, formatter)
    }
}

pub(crate) fn run(root: &Path, args: &[String]) -> DynResult<()> {
    let options = options::Options::parse(args)?;
    let state = PrivateState::create(&std::env::temp_dir(), "sdk-fixture")?;
    state.prepare()?;
    let ports = Reservation::acquire_at((options.api, options.console))?;
    let installer = options
        .artifact
        .as_ref()
        .map(|artifact| -> DynResult<Launch> {
            Ok(Launch {
                member: MemberId::new("runtime-install", 0)?,
                spec: ProcessSpec {
                    executable: options.binary.clone(),
                    cwd: root.to_owned(),
                    environment: state.environment(&options.native),
                    arguments: [
                        "--log-format".into(),
                        "json".into(),
                        "runtime".into(),
                        "install".into(),
                        "--bundle-dir".into(),
                        artifact.to_string_lossy().into_owned(),
                        "--cache-dir".into(),
                        options.cache.to_string_lossy().into_owned(),
                    ]
                    .into_iter()
                    .map(|argument: String| Value::Public(argument.into()))
                    .collect(),
                },
                files: Default::default(),
                readiness_deadline: options.consumer_deadline,
            })
        })
        .transpose()?;
    let mut environment = state.environment(&options.native);
    environment.insert(
        "MESH_LLM_NATIVE_RUNTIME_CACHE_DIR".into(),
        Value::Public(options.cache.clone().into()),
    );
    let daemon = Launch {
        member: MemberId::Seed,
        spec: ProcessSpec {
            executable: options.binary.clone(),
            cwd: root.to_owned(),
            environment,
            arguments: [
                "--log-format".into(),
                "json".into(),
                "serve".into(),
                "--model".into(),
                options.model.to_string_lossy().into_owned(),
                "--no-draft".into(),
                "--device".into(),
                "CPU".into(),
                "--ctx-size".into(),
                options.context.to_string(),
                "--port".into(),
                options.api.to_string(),
                "--console".into(),
                options.console.to_string(),
                "--bind-port".into(),
                ports.ports.quic.to_string(),
            ]
            .into_iter()
            .map(|argument: String| Value::Public(argument.into()))
            .collect(),
        },
        files: state.output_files(),
        readiness_deadline: options.wait,
    };
    let mut command_environment = std::env::vars_os()
        .map(|(key, value)| {
            let sensitive = key.to_string_lossy().to_ascii_uppercase();
            let secret = ["TOKEN", "SECRET", "PASSWORD", "API_KEY"]
                .iter()
                .any(|part| sensitive.contains(part));
            (
                key,
                if secret && !value.is_empty() {
                    Value::Secret(value)
                } else {
                    Value::Public(value)
                },
            )
        })
        .collect::<std::collections::BTreeMap<_, _>>();
    for (key, value) in [
        ("MESH_SDK_API_PORT", options.api.to_string()),
        ("MESH_SDK_CONSOLE_PORT", options.console.to_string()),
        (
            "MESH_CLIENT_API_BASE",
            format!("http://127.0.0.1:{}", options.api),
        ),
    ] {
        command_environment.insert(key.into(), Value::Public(value.into()));
    }
    let cancellation = process::Cancellation::default();
    command_environment.insert(
        "MESH_LLM_NATIVE_RUNTIME_CACHE_DIR".into(),
        Value::Public(options.cache.clone().into()),
    );
    let result = std::thread::scope(|scope| -> DynResult<()> {
        let (sender, ready) = std::sync::mpsc::sync_channel(1);
        let (begin_readiness, begin) = std::sync::mpsc::sync_channel(1);
        let worker_options = &options;
        let worker_cancellation = &cancellation;
        let worker = scope.spawn(move || {
            loop {
                match begin.recv_timeout(Duration::from_millis(100)) {
                    Ok(()) => break,
                    Err(std::sync::mpsc::RecvTimeoutError::Timeout)
                        if !worker_cancellation.is_cancelled() => {}
                    Err(_) => return,
                }
            }
            let result = readiness::wait(worker_options, worker_cancellation);
            if let Err(error) = sender.send(result) {
                drop(error);
            }
        });
        let mut owner = Owner {
            installer,
            installation_started: false,
            daemon: Some(daemon),
            command: Some(ProcessSpec {
                executable: options.command.clone(),
                arguments: options
                    .arguments
                    .iter()
                    .map(|argument| Value::Public(argument.into()))
                    .collect(),
                cwd: std::env::current_dir()?,
                environment: command_environment,
            }),
            ready,
            begin_readiness: Some(begin_readiness),
            consumer_deadline: options.consumer_deadline,
            admitted: false,
            launched: false,
        };
        drop(ports);
        let result = super::retained_session::run(
            &mut owner,
            &process::Limits {
                execution: options.wait + options.consumer_deadline * 2 + Duration::from_secs(30),
                graceful_shutdown: Duration::from_secs(5),
                forced_shutdown: Duration::from_secs(5),
                retained_bytes_per_stream: 65536,
                readiness: process::Readiness::None,
                completion: process::Completion::Exit,
            },
        );
        cancellation.cancel();
        if worker.join().is_err() {
            return Err(Failure::Worker { preceding: result }.into());
        }
        let report = result?;
        if let Some(consumer) = report
            .members
            .iter()
            .find(|member| member.member == MemberId::WorkerOne)
        {
            use std::io::Write;
            crate::cli_output::stdout().write_all(&consumer.process.stdout.bytes_retained)?;
            crate::cli_output::stderr().write_all(&consumer.process.stderr.bytes_retained)?;
        }
        if !report.success() {
            return Err(Failure::Session { report }.into());
        }
        Ok(())
    });
    match state.finish(result) {
        Ok(()) => Ok(()),
        Err(super::private_state::FinishError::Prior(error)) => Err(error),
        Err(super::private_state::FinishError::Deletion {
            kind,
            code,
            preceding,
        }) => Err(Failure::State {
            kind,
            code,
            preceding,
        }
        .into()),
    }
}
