use super::{
    adapter_args::{NativeBudget, Options},
    adapter_error::{self, Error},
    native_lint::{NativeFailure, NativeLint},
};
use crate::{automation::command_interrupt::Reason, process::Cancellation};
use std::{
    io::{self, Write},
    num::NonZeroUsize,
    path::{Path, PathBuf},
    time::Duration,
};

struct RejectDelivery {
    flush_only: bool,
}

impl Write for RejectDelivery {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        if self.flush_only {
            Ok(bytes.len())
        } else {
            Err(io::ErrorKind::BrokenPipe.into())
        }
    }

    fn flush(&mut self) -> io::Result<()> {
        Err(io::ErrorKind::BrokenPipe.into())
    }
}

fn failure_with_delivery_error(flush_only: bool) -> Error {
    let root = tempfile::tempdir().unwrap();
    let tool = PathBuf::from(
        std::env::var_os("SPV_ADAPTER_FIXTURE").expect("set the inert fixture executable"),
    );
    std::fs::write(
        root.path().join("lint-plan.json"),
        br#"{"behavior":"sensitive-fail","fail_path":null}"#,
    )
    .unwrap();
    let options = Options {
        template: Path::new("template.xcprivacy"),
        xcframework: None,
        plutil: &tool,
        budget: NativeBudget {
            execution: Duration::from_secs(2),
            grace: Duration::from_millis(200),
            cleanup: Duration::from_secs(2),
            output: NonZeroUsize::new(4096).unwrap(),
        },
    };
    let cancellation = Cancellation::default();
    let native = NativeLint {
        options: &options,
        cwd: root.path(),
        cancellation: &cancellation,
    };
    native
        .lint(options.template, &mut RejectDelivery { flush_only })
        .unwrap_err()
}

#[test]
fn retains_child_failure_when_diagnostic_write_fails() {
    let error = failure_with_delivery_error(false);
    assert!(matches!(error, Error::Diagnostic {
        primary: NativeFailure::Child { exit: Some(23), complete_raw_stderr: true, .. }, output
    } if output.kind() == io::ErrorKind::BrokenPipe));
}

#[test]
fn retains_child_failure_when_diagnostic_flush_fails() {
    let error = failure_with_delivery_error(true);
    assert!(matches!(error, Error::Diagnostic {
        primary: NativeFailure::Child { exit: Some(23), .. }, output
    } if output.kind() == io::ErrorKind::BrokenPipe));
}

#[test]
fn retains_child_and_delivery_failures_when_signal_finalization_fails() {
    let primary: Result<(), Error> = Err(failure_with_delivery_error(false));
    let error = adapter_error::finish(primary, Err(Reason::Interrupted)).unwrap_err();
    assert!(
        matches!(error, Error::Finalization { primary, secondary: Reason::Interrupted }
        if matches!(*primary, Error::Diagnostic { primary: NativeFailure::Child { exit: Some(23), .. }, .. }))
    );
}

#[test]
fn omits_raw_diagnostic_from_debug_when_delivery_fails() {
    let error = failure_with_delivery_error(false);
    assert!(matches!(error, Error::Diagnostic { .. }));
    assert!(!format!("{error:?}").contains("synthetic-raw"));
}
