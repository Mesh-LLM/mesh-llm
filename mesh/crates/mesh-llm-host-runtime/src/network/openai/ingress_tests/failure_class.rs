//! The `failure_class` a terminal `openai.exchange.v1` event states for an
//! exchange that did not complete: the classifier, the envelope member and the
//! remote-mesh publisher's wiring (the raw-proxy publisher's is in
//! `durable_artifacts`).
use crate::network::openai::ServedModelHeaders;
use crate::plugin::openai_exchange::OpenAiExchangeEnvelope;

use super::{RemoteDeliveredFacts, exchange_failure_class, proxy, remote_delivered_terminal};

// A terminal event states why an exchange did not complete. `status` cannot
// say it alone: a stream cut after its 200 headers is a `FailedWithStatus`
// with status 200, and a drop or transport failure has no status at all.
#[test]
fn exchange_failure_class_names_every_way_an_exchange_does_not_complete() {
    use crate::plugin::openai_exchange::ExchangeFailureClass::{
        Dropped, PaymentRequired, Refused, Timeout, UpstreamError,
    };
    use proxy::RouteDispatchOutcome as O;
    let digests = Default::default();
    let cases = [
        (O::PolicyDenied { status_code: 402 }, Some(PaymentRequired)),
        (O::PolicyDenied { status_code: 403 }, Some(Refused)),
        (O::RequiredHookFailed, Some(Refused)),
        // A peer's 402 relayed to the client.
        (O::Responded(402), Some(PaymentRequired)),
        (
            O::RespondedWithDigests {
                status_code: 402,
                output_digests: digests,
            },
            Some(PaymentRequired),
        ),
        (O::Responded(429), Some(Refused)),
        (O::Responded(408), Some(Timeout)),
        (
            O::RespondedWithUsage {
                status_code: 500,
                usage: mesh_llm_events::logging::events::TokenUsage::default(),
                output_digests: digests,
            },
            Some(UpstreamError),
        ),
        (O::Responded(500), Some(UpstreamError)),
        (O::Responded(504), Some(Timeout)),
        (
            O::FailedWithStatus {
                status_code: 504,
                reason: "timeout",
            },
            Some(Timeout),
        ),
        (
            O::FailedWithStatus {
                status_code: 200,
                reason: "upstream_stream_incomplete",
            },
            Some(UpstreamError),
        ),
        (O::Failed("endpoint_transport_failed"), Some(UpstreamError)),
        (O::Dropped("client_disconnected"), Some(Dropped)),
        (O::Responded(200), None),
        (
            O::RespondedWithDigests {
                status_code: 204,
                output_digests: digests,
            },
            None,
        ),
    ];
    for (outcome, want) in cases {
        assert_eq!(exchange_failure_class(&outcome), want, "{outcome:?}");
    }
}

#[test]
fn failure_class_rides_the_terminal_envelope_as_an_additive_member() {
    use crate::plugin::openai_exchange::{ExchangeFailureClass, OpenAiExchangeDispatchPath};
    let served = OpenAiExchangeEnvelope::terminal(
        "exch-served",
        OpenAiExchangeDispatchPath::RawProxy,
        "acme/model",
        Some(200),
        None,
        None,
    );
    assert!(
        serde_json::to_value(&served)
            .unwrap()
            .get("failure_class")
            .is_none(),
        "a completed exchange serializes exactly as before"
    );
    let refused = served
        .clone()
        .with_failure_class(Some(ExchangeFailureClass::PaymentRequired));
    assert_eq!(
        serde_json::to_value(&refused).unwrap()["failure_class"],
        "payment_required"
    );
}

/// The `RemoteMesh` terminal states why an exchange did not complete, including
/// a stream cut after its 200 headers, whose status says nothing.
#[test]
fn remote_delivered_terminal_states_the_failure_class() {
    use crate::plugin::openai_exchange::ExchangeFailureClass;
    let terminal_for = |outcome: proxy::RouteDispatchOutcome| {
        remote_delivered_terminal(
            RemoteDeliveredFacts {
                exchange_id: "exchange-failed".to_string(),
                model_name: "acme/remote-model",
                nonce: None,
                nonce_source: None,
                peer_capsule_id: None,
                target: None,
                observed_served_by_hex: None,
                peer_served_model: ServedModelHeaders::default(),
                request_digest: None,
                twin_bracket_id: None,
            },
            &outcome,
        )
    };
    let cut = terminal_for(proxy::RouteDispatchOutcome::FailedWithStatus {
        status_code: 200,
        reason: "upstream_stream_incomplete",
    });
    assert_eq!(cut.status, Some(200));
    assert_eq!(cut.failure_class, Some(ExchangeFailureClass::UpstreamError));
    assert_eq!(
        terminal_for(proxy::RouteDispatchOutcome::Responded(402)).failure_class,
        Some(ExchangeFailureClass::PaymentRequired)
    );
    assert_eq!(
        terminal_for(proxy::RouteDispatchOutcome::Responded(200)).failure_class,
        None
    );
}
