use super::{support::*, *};
use std::time::Duration;

#[test]
fn all_four_required_variants_need_the_entire_standalone_sequence() {
    for variant in Variant::REQUIRED {
        let session = completed(variant);
        let result: Completed = session
            .finish((report(41), Some(report(42))), Ok(()))
            .unwrap();
        assert_eq!(result.variant(), variant);
        assert_eq!(result.process_reports().0.pid, 41);
        assert_eq!(result.process_reports().1.pid, 42);
    }
}

#[test]
fn variant_defaults_preserve_recurrent_sizing_and_constrained_stack() {
    assert_eq!(Model::Dense.artifact_id(), "smollm2-q8-inference");
    assert_eq!(Model::Recurrent.artifact_id(), "family-granite-hybrid");
    assert_eq!(
        (Model::Dense.context_size(), Model::Dense.batch_sizes()),
        (256, None)
    );
    assert_eq!(
        (
            Model::Recurrent.context_size(),
            Model::Recurrent.batch_sizes()
        ),
        (128, Some((128, 128)))
    );
    assert_eq!(Stack::Default.bytes(), None);
    assert_eq!(Stack::Constrained.bytes(), Some(2_097_152));
}

#[test]
fn readiness_retries_do_not_reset_the_deadline() {
    let mut session = session();
    session
        .observe((Check::Runtime, response(b"{")), Duration::from_secs(179))
        .unwrap();
    let result = session.observe(
        (Check::Runtime, response(br#"{"llama_ready":true}"#)),
        Duration::from_secs(180),
    );
    assert_eq!(result, Err(Rejection::Deadline(Check::Runtime)));
}

#[test]
fn models_get_their_own_sixty_second_budget() {
    let mut session = session();
    session
        .observe(
            (Check::Runtime, response(br#"{"llama_ready":true}"#)),
            Duration::from_secs(179),
        )
        .unwrap();
    let result = session.observe(
        (Check::Models, response(br#"{"data":[{"id":"chosen"}]}"#)),
        Duration::from_secs(238),
    );
    assert_eq!(result, Ok(Progress::Advanced(Check::Chat)));
    assert_eq!(session.model_id(), Some("chosen"));
}

#[test]
fn skipped_stream_check_is_terminal_even_if_auto_response_is_valid() {
    let mut session = session();
    session
        .observe(
            (Check::Runtime, response(br#"{"llama_ready":true}"#)),
            Duration::ZERO,
        )
        .unwrap();
    session
        .observe(
            (Check::Models, response(br#"{"data":[{"id":"chosen"}]}"#)),
            Duration::ZERO,
        )
        .unwrap();
    session
        .observe(
            (
                Check::Chat,
                response(
                    br#"{"object":"chat.completion","choices":[{"message":{"content":"hi"}}]}"#,
                ),
            ),
            Duration::ZERO,
        )
        .unwrap();
    let result = session.observe(
        (
            Check::Auto,
            response(br#"{"choices":[{"message":{"content":"hi"}}]}"#),
        ),
        Duration::ZERO,
    );
    assert_eq!(result, Err(Rejection::OutOfOrder));
    assert_eq!(session.expected(), Err(Rejection::OutOfOrder));
}

#[test]
fn malformed_chat_cannot_be_retried_into_success() {
    let mut session = session();
    session
        .observe(
            (Check::Runtime, response(br#"{"llama_ready":true}"#)),
            Duration::ZERO,
        )
        .unwrap();
    session
        .observe(
            (Check::Models, response(br#"{"data":[{"id":"chosen"}]}"#)),
            Duration::ZERO,
        )
        .unwrap();
    let result = session.observe((Check::Chat, response(b"{")), Duration::ZERO);
    assert_eq!(result, Err(Rejection::Evidence(Check::Chat)));
    assert_eq!(session.tick(Duration::from_secs(1)), result);
}

#[test]
fn headless_status_retry_rechecks_models_without_renewing_the_budget() {
    let mut session = before_headless(variant(Model::Dense));
    session
        .observe(
            (Check::HeadlessModels, response(b"")),
            Duration::from_secs(179),
        )
        .unwrap();
    session
        .observe(
            (Check::HeadlessStatus, Transfer::Failed),
            Duration::from_secs(179),
        )
        .unwrap();
    assert_eq!(session.expected(), Ok(Some(Check::HeadlessModels)));
    assert_eq!(
        session.tick(Duration::from_secs(181)),
        Err(Rejection::Deadline(Check::HeadlessModels))
    );
}

#[test]
fn enabled_attestation_cannot_skip_either_served_status() {
    let mut session = Session::new(
        variant(Model::Dense),
        Attestation::Required {
            expected: "valid".into(),
        },
        Budget::default(),
    );
    for (check, body) in [
        (
            Check::InspectAttestation,
            br#"{"status":"valid"}"#.as_slice(),
        ),
        (Check::Runtime, br#"{"llama_ready":true}"#),
        (Check::Models, br#"{"data":[{"id":"chosen"}]}"#),
    ] {
        session
            .observe((check, response(body)), Duration::ZERO)
            .unwrap();
    }
    assert_eq!(session.expected(), Ok(Some(Check::RuntimeAttestation)));
    for (check, body) in [
        (
            Check::RuntimeAttestation,
            br#"{"release_attestation":{"status":"valid"}}"#.as_slice(),
        ),
        (
            Check::Chat,
            br#"{"object":"chat.completion","choices":[{"message":{"content":"hi"}}]}"#,
        ),
        (Check::Stream, b"\"role\":\"assistant\" data: [DONE]"),
        (
            Check::Auto,
            br#"{"choices":[{"message":{"content":"hi"}}]}"#,
        ),
        (Check::HeadlessModels, b""),
        (Check::HeadlessStatus, b""),
    ] {
        session
            .observe((check, response(body)), Duration::ZERO)
            .unwrap();
    }
    assert_eq!(session.expected(), Ok(Some(Check::HeadlessAttestation)));
    assert_eq!(
        session.observe((Check::HeadlessAttestation, response(b"{")), Duration::ZERO),
        Err(Rejection::Evidence(Check::HeadlessAttestation))
    );
}

#[test]
fn attestation_preflight_is_required_before_runtime_when_enabled() {
    let mut session = Session::new(
        variant(Model::Recurrent),
        Attestation::Required {
            expected: "valid".into(),
        },
        Budget::default(),
    );
    let result = session.observe(
        (Check::Runtime, response(br#"{"llama_ready":true}"#)),
        Duration::ZERO,
    );
    assert_eq!(result, Err(Rejection::OutOfOrder));
}

#[test]
fn backwards_clock_cannot_extend_smoke_deadlines() {
    let mut session = session();
    session.tick(Duration::from_secs(2)).unwrap();
    assert_eq!(
        session.tick(Duration::from_secs(1)),
        Err(Rejection::ClockRegression)
    );
}

#[test]
fn interruption_after_decisions_prevents_completion() {
    let mut session = completed(variant(Model::Dense));
    let result = session.cancel();
    assert_eq!(result, Err(Rejection::Cancelled));
    assert!(
        session
            .finish((report(41), Some(report(42))), Ok(()))
            .is_err()
    );
}

#[test]
fn readiness_budget_rejects_zero_and_unbounded_values() {
    assert!(Budget::new(Duration::ZERO).is_err());
    assert!(Budget::new(Duration::from_secs(86_401)).is_err());
    assert!(Budget::new(Duration::from_secs(86_400)).is_ok());
}
