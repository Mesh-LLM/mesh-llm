use super::super::*;
use crate::process::Cancellation;
mod failure;
mod overlap_tests;

pub(super) fn options(directory: &std::path::Path, scenario: &str) -> args::Options {
    std::fs::write(directory.join("scenario"), scenario).unwrap();
    let fixture = std::env::current_exe()
        .unwrap()
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .join("examples")
        .join(format!(
            "migration_smoke_fixture{}",
            std::env::consts::EXE_SUFFIX
        ));
    args::Options {
        binary: fixture,
        native: directory.to_owned(),
        parent: directory.to_owned(),
        model: "fixture-model".into(),
        device: "CPU".into(),
        projector: None,
        key: None,
        expected: "missing".into(),
        variant: Variant {
            model: Model::Dense,
            stack: Stack::Default,
        },
        readiness: std::time::Duration::from_secs(2),
        shutdown: std::time::Duration::from_secs(1),
        context_size: None,
        batch_sizes: None,
        endpoints: None,
    }
}

#[test]
fn smoke_passes_when_fixture_serves_each_required_variant() {
    for variant in Variant::REQUIRED {
        let directory = tempfile::tempdir().unwrap();
        let mut options = options(directory.path(), "success");
        options.variant = variant;
        let receipt =
            command::execute(directory.path(), &options, &Cancellation::default()).unwrap();
        let receipt = serde_json::to_value(receipt).unwrap();
        assert_eq!(receipt["status"], "passed");
        assert_eq!(receipt["processes"][0]["exit_code"], 0);
        assert_ne!(
            receipt["processes"][0]["pid"],
            receipt["processes"][1]["pid"]
        );
        let audit: serde_json::Value = serde_json::from_slice(
            &std::fs::read(directory.path().join("primary-audit.json")).unwrap(),
        )
        .unwrap();
        assert_eq!(
            audit["stack"],
            serde_json::to_value(variant.stack.bytes().map(|bytes| bytes.to_string())).unwrap()
        );
        match variant.model {
            Model::Dense => assert!(audit["config"].is_null()),
            Model::Recurrent => assert!(audit["config"].as_str().unwrap().contains("batch = 128")),
        }
        assert!(std::fs::read_dir(directory.path()).unwrap().all(|entry| {
            !entry
                .unwrap()
                .file_name()
                .to_string_lossy()
                .starts_with("required-smoke.")
        }));
    }
}

macro_rules! rejects {
    ($name:ident, $scenario:literal, $reason:expr) => {
        #[test]
        fn $name() {
            let directory = tempfile::tempdir().unwrap();
            let options = options(directory.path(), $scenario);
            let error =
                command::execute(directory.path(), &options, &Cancellation::default()).unwrap_err();
            assert_eq!(error.downcast_ref::<Rejected>().unwrap().reason, $reason);
        }
    };
}
rejects!(
    rejects_chat_when_malformed,
    "malformed",
    Rejection::Evidence(Check::Chat)
);
rejects!(
    rejects_chat_when_transfer_fails,
    "transfer",
    Rejection::Transfer(Check::Chat)
);
rejects!(
    rejects_chat_when_oversized,
    "oversized",
    Rejection::ResponseLimit(Check::Chat)
);
rejects!(
    rejects_stream_when_marker_missing,
    "stream",
    Rejection::Evidence(Check::Stream)
);
rejects!(
    rejects_auto_when_content_missing,
    "auto",
    Rejection::Evidence(Check::Auto)
);
rejects!(
    rejects_daemon_when_identity_foreign,
    "ownership",
    Rejection::ProcessFailure
);
rejects!(
    rejects_runtime_when_deadline_expires,
    "timeout",
    Rejection::Deadline(Check::Runtime)
);
rejects!(
    rejects_process_when_exits_early,
    "early",
    Rejection::EarlyExit
);
rejects!(
    rejects_cleanup_when_forced,
    "stubborn",
    Rejection::ProcessCleanup
);
rejects!(
    rejects_cleanup_when_exit_nonzero,
    "nonzero",
    Rejection::ProcessCleanup
);

#[test]
fn rejects_cancel_when_already_requested() {
    let directory = tempfile::tempdir().unwrap();
    let options = options(directory.path(), "success");
    let cancel = Cancellation::default();
    cancel.cancel();
    let error = command::execute(directory.path(), &options, &cancel).unwrap_err();
    assert_eq!(
        error.downcast_ref::<Rejection>(),
        Some(&Rejection::Cancelled)
    );
}

#[test]
fn rejects_cleanup_when_private_state_obstructed() {
    let directory = tempfile::tempdir().unwrap();
    let options = options(directory.path(), "delete");
    let error = command::execute(directory.path(), &options, &Cancellation::default()).unwrap_err();
    assert_eq!(
        output::reason(error.as_ref()),
        Rejection::StateCleanup.to_string()
    );
    let reports = output::reports(error.as_ref());
    assert_eq!(reports.len(), 2);
    assert!(reports[0].pid > 0);
    assert!(reports[0].status.is_some());
}

#[test]
fn rejects_cancel_when_http_request_started() {
    let directory = tempfile::tempdir().unwrap();
    let options = options(directory.path(), "cancel");
    let cancel = Cancellation::default();
    std::thread::scope(|scope| {
        let worker = scope.spawn(|| {
            command::execute(directory.path(), &options, &cancel).map_err(|error| error.to_string())
        });
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
        while !directory.path().join("chat.armed").exists() {
            assert!(std::time::Instant::now() < deadline);
            std::thread::yield_now();
        }
        cancel.cancel();
        let result = worker.join().unwrap();
        assert_eq!(result.unwrap_err(), Rejection::Cancelled.to_string());
    });
}

#[test]
fn smoke_passes_when_missing_attestation_expected() {
    let directory = tempfile::tempdir().unwrap();
    let mut options = options(directory.path(), "success");
    let key = directory.path().join("key.json");
    std::fs::write(&key, b"{}").unwrap();
    options.key = Some(key);
    let receipt = command::execute(directory.path(), &options, &Cancellation::default()).unwrap();
    assert_eq!(serde_json::to_value(receipt).unwrap()["status"], "passed");
}

#[test]
fn rejects_attestation_when_runtime_differs_from_inspection() {
    let directory = tempfile::tempdir().unwrap();
    let mut options = options(directory.path(), "attestation");
    let key = directory.path().join("key.json");
    std::fs::write(&key, b"{}").unwrap();
    options.key = Some(key);
    let error = command::execute(directory.path(), &options, &Cancellation::default()).unwrap_err();
    assert_eq!(
        error.downcast_ref::<Rejected>().unwrap().reason,
        Rejection::Evidence(Check::RuntimeAttestation)
    );
}

#[test]
fn rejects_headless_when_models_fail() {
    let directory = tempfile::tempdir().unwrap();
    let options = options(directory.path(), "headless");
    let error = command::execute(directory.path(), &options, &Cancellation::default()).unwrap_err();
    assert_eq!(
        error.downcast_ref::<Rejected>().unwrap().reason,
        Rejection::ProcessFailure
    );
}
