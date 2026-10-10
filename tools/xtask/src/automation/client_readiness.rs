mod args;
pub(crate) mod event;
pub(crate) mod receipt;

use super::command_interrupt::{self, Interrupt};
use super::private_state::{self, PrivateState};
use crate::process::{self, Completion, Limits, ProcessSpec, Readiness, Value};
use args::Options;
use std::io::{self, Write};
use std::net::{Ipv4Addr, TcpListener};
use std::path::Path;
use std::time::Duration;

pub(crate) const USAGE: &str = "cargo xtool automation client-readiness --binary <absolute executable> --native-runtime-root <absolute directory> [--ready-max-wait <seconds:1..86400>] [--shutdown-max-wait <seconds:1..86400>] [--state-parent <absolute directory>]";

#[derive(Debug, thiserror::Error)]
pub(crate) enum Error {
    #[error("client readiness: {0}")]
    Invalid(&'static str),
    #[error("client readiness {operation} failed ({kind:?}, OS code {code:?})")]
    Io {
        operation: &'static str,
        kind: io::ErrorKind,
        code: Option<i32>,
    },
    #[error("client readiness: {0}")]
    Process(#[from] process::Failure),
    #[error("client readiness: {0}")]
    Lifecycle(Box<receipt::Rejected>),
    #[error(
        "private state deletion failed ({kind:?}, OS code {code:?}); preceding failure: {preceding:?}"
    )]
    StateDeletion {
        kind: io::ErrorKind,
        code: Option<i32>,
        preceding: Option<Box<Error>>,
    },
}

impl Error {
    fn io(operation: &'static str, error: io::Error) -> Self {
        Self::Io {
            operation,
            kind: error.kind(),
            code: error.raw_os_error(),
        }
    }
}

impl From<private_state::Error> for Error {
    fn from(error: private_state::Error) -> Self {
        match error {
            private_state::Error::EntropyUnavailable => {
                Self::Invalid("private state entropy unavailable")
            }
            #[cfg(windows)]
            private_state::Error::Invalid(message) => Self::Invalid(message),
            private_state::Error::Io {
                operation,
                kind,
                code,
            } => Self::Io {
                operation,
                kind,
                code,
            },
        }
    }
}

impl From<private_state::FinishError<Error>> for Error {
    fn from(error: private_state::FinishError<Error>) -> Self {
        match error {
            private_state::FinishError::Prior(error) => error,
            private_state::FinishError::Deletion {
                kind,
                code,
                preceding,
            } => Self::StateDeletion {
                kind,
                code,
                preceding,
            },
        }
    }
}

impl From<command_interrupt::Reason> for Error {
    fn from(reason: command_interrupt::Reason) -> Self {
        use command_interrupt::Reason;
        match reason {
            Reason::ScopeBusy => Self::Invalid("client readiness signal scope already active"),
            Reason::Interrupted => {
                Self::Invalid("client readiness cancelled by command interruption")
            }
            #[cfg(unix)]
            Reason::ExistingHandler => {
                Self::Invalid("client readiness refuses an existing signal handler")
            }
            #[cfg(unix)]
            Reason::BlockedSigint => Self::Invalid("client readiness refuses blocked SIGINT"),
            #[cfg(unix)]
            Reason::BlockedSigterm => Self::Invalid("client readiness refuses blocked SIGTERM"),
            Reason::Io {
                operation,
                kind,
                code,
            } => Self::Io {
                operation,
                kind,
                code,
            },
        }
    }
}

pub(crate) fn run(root: &Path, args: &[String]) -> Result<(), Error> {
    if args == ["--help"] {
        return writeln!(io::stdout().lock(), "{USAGE}")
            .map_err(|error| Error::io("write help", error));
    }
    let options = Options::parse(args)?;
    let interrupt = Interrupt::install()?;
    interrupt.check()?;
    let state = PrivateState::create(&options.state_parent, "mlc-state")?;
    let result = (|| {
        state.prepare()?;
        let listener = TcpListener::bind((Ipv4Addr::LOCALHOST, 0))
            .map_err(|error| Error::io("select loopback port", error))?;
        let port = listener
            .local_addr()
            .map_err(|error| Error::io("read loopback port", error))?
            .port();
        let spec = client_spec(root, &options, (&state, port));
        drop(listener);
        let report = process::supervise(
            &spec,
            &limits(&options),
            &interrupt.cancellation(),
            state.output_files(),
        )?;
        if interrupt.cancellation().is_cancelled() {
            return Err(receipt::interrupted(report));
        }
        receipt::accept(report)?;
        Ok(port)
    })();
    let result = state.finish(result).map_err(Error::from);
    let interruption = interrupt.finish();
    let port = result?;
    interruption?;
    writeln!(
        io::stdout().lock(),
        "client readiness observed on port {port}"
    )
    .map_err(|error| Error::io("write readiness result", error))
}

fn client_spec(root: &Path, options: &Options, endpoint: (&PrivateState, u16)) -> ProcessSpec {
    let (state, port) = endpoint;
    ProcessSpec {
        executable: options.binary.clone(),
        arguments: [
            "--log-format".to_owned(),
            "json".to_owned(),
            "--port".to_owned(),
            port.to_string(),
            "--no-console".to_owned(),
            "client".to_owned(),
            "--mesh-discovery-mode".to_owned(),
            "mdns".to_owned(),
        ]
        .into_iter()
        .map(|value| Value::Public(value.into()))
        .collect(),
        cwd: root.to_path_buf(),
        environment: state.environment(&options.native_runtime_root),
    }
}

fn limits(options: &Options) -> Limits {
    Limits {
        execution: options.ready_max_wait,
        graceful_shutdown: options.shutdown_max_wait,
        forced_shutdown: Duration::from_secs(5),
        retained_bytes_per_stream: 64 * 1024,
        readiness: Readiness::ObservedLines {
            deadline: options.ready_max_wait,
            matcher: event::matches,
        },
        completion: Completion::StopAfterReady,
    }
}
