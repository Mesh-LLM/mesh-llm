mod args;
pub(crate) mod correlation;
mod error;
pub(crate) mod http;
pub(crate) mod observer;
pub(crate) mod ports;
mod receipt;

use super::command_interrupt::{Interrupt, Reason};
use super::private_state::{self, PrivateState};
use crate::process::{self, Completion, Limits, ProcessSpec, Readiness, Value};
use args::Options;
use correlation::RequestId;
pub(crate) use error::Error;
use observer::Observer;
use ports::{Ports, Reservation};
use std::io::{self, Write};
use std::path::Path;
use std::sync::mpsc;
use std::time::Duration;

mod status;

#[cfg(test)]
#[path = "../../tests/migration_lifecycle/daemon/owned.rs"]
mod tests;

#[cfg(test)]
#[path = "../../tests/migration_lifecycle/daemon/completion.rs"]
mod completion_tests;

pub(crate) const USAGE: &str = "cargo xtool automation daemon-readiness --binary <absolute executable> --native-runtime-root <absolute directory> [--ready-max-wait <seconds:1..86400>] [--shutdown-max-wait <seconds:1..86400>] [--state-parent <absolute directory>] (bounded zero-model subset)";

#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub(crate) enum Rejection {
    #[error("malformed_status")]
    MalformedStatus,
    #[error("ownership_mismatch")]
    OwnershipMismatch,
    #[error("models_transfer_failed")]
    ModelsTransferFailed,
    #[error("response_limit")]
    ResponseLimit,
    #[error("worker_failed")]
    WorkerFailed,
}

pub(crate) fn run(root: &Path, args: &[String]) -> Result<(), Error> {
    if args == ["--help"] {
        return writeln!(io::stdout().lock(), "{USAGE}")
            .map_err(|error| Error::io("write help", error));
    }
    let options = Options::parse(args)?;
    let interrupt = Interrupt::install()?;
    interrupt.check()?;
    let state = PrivateState::create(&options.state_parent, "mld-state")?;
    let result = (|| {
        state.prepare()?;
        let reservation = Reservation::acquire()?;
        let ports = reservation.ports;
        let id = RequestId::generate()?;
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .map_err(|error| Error::io("create HTTP runtime", error))?;
        let spec = daemon_spec(root, &options, (&state, ports));
        let accepted = std::thread::scope(|scope| {
            let (requests, work) = mpsc::sync_channel(1);
            let (results, responses) = mpsc::sync_channel(1);
            let worker = std::thread::Builder::new()
                .name("daemon-http".into())
                .spawn_scoped(scope, move || http::work(ports, (work, results), runtime))
                .map_err(|error| Error::io("start HTTP worker", error))?;
            let mut observer =
                Observer::new(requests, responses, id, options.ready_max_wait.as_secs());
            drop(reservation);
            let result = process::supervise_with_probe(
                &spec,
                &limits(&options),
                &interrupt.cancellation(),
                state.output_files(),
                process::Probe {
                    observer: &mut observer,
                    deadline: options.ready_max_wait,
                },
            )
            .map_err(Error::from);
            let facts = observer.facts();
            drop(observer);
            let joined = worker.join().is_ok();
            match result {
                Ok(report) => receipt::accept(
                    report,
                    facts,
                    (interrupt.cancellation().is_cancelled(), joined),
                ),
                Err(error) if joined => Err(error),
                Err(error) => Err(Error::WorkerJoin(Some(Box::new(error)))),
            }
        })?;
        Ok((ports, accepted))
    })();
    let result = state.finish(result).map_err(|error| match error {
        private_state::FinishError::Prior(error) => error,
        private_state::FinishError::Deletion {
            kind,
            code,
            preceding,
        } => Error::StateDeletion {
            kind,
            code,
            preceding,
        },
    });
    let interruption = interrupt.finish();
    let ports = finish(result, interruption)?;
    let mut output = result_output().map_err(|error| Error::io("open readiness result", error))?;
    writeln!(
        output,
        "zero_model_serve_ready: bounded subset observed on API {} and console {}",
        ports.api, ports.console
    )
    .and_then(|()| output.flush())
    .map_err(|error| Error::io("write readiness result", error))
}

fn finish(
    result: Result<(Ports, receipt::Accepted), Error>,
    interruption: Result<(), Reason>,
) -> Result<Ports, Error> {
    match (result, interruption) {
        (Ok((ports, accepted)), interruption) => {
            accepted.finish(interruption)?;
            Ok(ports)
        }
        (Err(error), Ok(())) => Err(error),
        (Err(error), Err(reason)) => Err(Error::Interruption {
            preceding: Box::new(error),
            cause: Box::new(Error::from(reason)),
        }),
    }
}

fn result_output() -> io::Result<std::fs::File> {
    #[cfg(unix)]
    {
        use std::os::fd::AsFd;
        io::stdout()
            .as_fd()
            .try_clone_to_owned()
            .map(std::fs::File::from)
    }
    #[cfg(windows)]
    {
        use std::os::windows::io::AsHandle;
        io::stdout()
            .as_handle()
            .try_clone_to_owned()
            .map(std::fs::File::from)
    }
}

fn daemon_spec(root: &Path, options: &Options, endpoint: (&PrivateState, Ports)) -> ProcessSpec {
    let (state, ports) = endpoint;
    let mut environment = state.environment(&options.native_runtime_root);
    environment.insert("MESH_LLM_EPHEMERAL_KEY".into(), Value::Public("1".into()));
    ProcessSpec {
        executable: options.binary.clone(),
        arguments: [
            "--log-format".into(),
            "json".into(),
            "--port".into(),
            ports.api.to_string(),
            "--console".into(),
            ports.console.to_string(),
            "--bind-port".into(),
            ports.quic.to_string(),
            "--bind-ip".into(),
            "127.0.0.1".into(),
            "--headless".into(),
            "serve".into(),
            "--mesh-discovery-mode".into(),
            "mdns".into(),
        ]
        .into_iter()
        .map(|value: String| Value::Public(value.into()))
        .collect(),
        cwd: root.to_path_buf(),
        environment,
    }
}

fn limits(options: &Options) -> Limits {
    Limits {
        execution: options.ready_max_wait,
        graceful_shutdown: options.shutdown_max_wait,
        forced_shutdown: Duration::from_secs(5),
        retained_bytes_per_stream: 65536,
        readiness: Readiness::None,
        completion: Completion::Exit,
    }
}
