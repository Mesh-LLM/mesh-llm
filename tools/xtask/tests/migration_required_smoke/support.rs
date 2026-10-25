use super::*;
use crate::process::{self, ProcessReport};
use std::time::Duration;

pub(super) fn variant(model: Model) -> Variant {
    Variant {
        model,
        stack: Stack::Default,
    }
}

pub(super) fn session() -> Session {
    Session::new(
        variant(Model::Dense),
        Attestation::Disabled,
        Budget::default(),
    )
}

pub(super) fn response(body: &[u8]) -> Transfer<'_> {
    Transfer::Complete(Response { status: 200, body })
}

pub(super) fn completed(variant: Variant) -> Session {
    let mut session = before_headless(variant);
    for check in [Check::HeadlessModels, Check::HeadlessStatus] {
        session
            .observe((check, response(b"")), Duration::from_secs(1))
            .unwrap();
    }
    session
}

pub(super) fn before_headless(variant: Variant) -> Session {
    let mut session = Session::new(variant, Attestation::Disabled, Budget::default());
    for (check, body) in [
        (Check::Runtime, br#"{"llama_ready":true}"#.as_slice()),
        (Check::Models, br#"{"data":[{"id":"fixture"}]}"#),
        (
            Check::Chat,
            br#"{"object":"chat.completion","choices":[{"message":{"content":"hi"}}]}"#,
        ),
        (
            Check::Stream,
            b"data: {\"role\":\"assistant\"}\n\ndata: [DONE]\n",
        ),
        (
            Check::Auto,
            br#"{"choices":[{"message":{"content":"hi"}}]}"#,
        ),
    ] {
        session
            .observe((check, response(body)), Duration::from_secs(1))
            .unwrap();
    }
    session
}

pub(super) fn report(pid: u32) -> ProcessReport {
    ProcessReport {
        pid,
        outcome: process::Outcome::Ready,
        status: Some(exit(0)),
        ready: true,
        readiness_stop: process::ReadinessStop::ProbeAdmitted {
            elapsed: Duration::from_secs(2),
            request: process::GracefulRequest::RequestedAfterLiveObservation,
        },
        elapsed: Duration::from_secs(3),
        stdout: process::StreamReport::default(),
        stderr: process::StreamReport::default(),
        cleanup: process::Cleanup {
            complete: true,
            ..process::Cleanup::default()
        },
        failure: None,
    }
}

pub(super) fn exit(code: i32) -> std::process::ExitStatus {
    #[cfg(unix)]
    {
        use std::os::unix::process::ExitStatusExt;
        std::process::ExitStatus::from_raw(code << 8)
    }
    #[cfg(windows)]
    {
        use std::os::windows::process::ExitStatusExt;
        std::process::ExitStatus::from_raw(u32::try_from(code).unwrap())
    }
}
