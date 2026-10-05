use super::*;

fn push_sse_event(
    logging: &crate::logging::LoggingRuntimeState,
    channel: mesh_llm_events::logging::replay::ReplayChannel,
    request_id: mesh_llm_events::logging::identifiers::RequestId,
) -> String {
    use mesh_llm_events::logging::{envelope::CanonicalEnvelope, events::LifecycleEvent};

    logging
        .service_for_test()
        .expect("installed logging service")
        .enqueue_event(
            request_id,
            channel,
            serde_json::to_string(&LifecycleEvent::Admitted {
                model: Some("/private/operator/model.gguf?token=secret".into()),
                method: None,
            })
            .expect("serialize test lifecycle event"),
        )
        .expect("enqueue test lifecycle event");

    let record = logging
        .replay_bus()
        .expect("installed replay bus")
        .replay_window()
        .records
        .into_iter()
        .rev()
        .find(|record| {
            serde_json::from_str::<serde_json::Value>(&record.entry.payload)
                .ok()
                .and_then(|payload| payload.get("canonical_envelope").cloned())
                .and_then(|value| CanonicalEnvelope::from_json_str(&value.to_string()).ok())
                .is_some_and(|envelope| {
                    envelope.request_id == request_id && envelope.channel == channel
                })
        })
        .expect("enqueued test lifecycle event retained in replay window");
    format!(
        "id: v1:{}.{}.{}",
        record
            .cursor
            .sequence(mesh_llm_events::logging::replay::ReplayChannel::Requests),
        record
            .cursor
            .sequence(mesh_llm_events::logging::replay::ReplayChannel::Operations),
        record
            .cursor
            .sequence(mesh_llm_events::logging::replay::ReplayChannel::System),
    )
}

async fn install_sse_logging() -> tempfile::TempDir {
    let temporary_directory = tempfile::tempdir().unwrap();
    crate::initialize_logging_foundation(&mesh_llm_config::LoggingConfig {
        application_state_root: Some(temporary_directory.path().join("logging")),
        replay_capacity: 8,
        ..Default::default()
    })
    .await;
    temporary_directory
}

async fn disable_sse_logging() {
    crate::initialize_logging_foundation(&mesh_llm_config::LoggingConfig {
        enabled: false,
        ..Default::default()
    })
    .await;
}

fn cleanup_post(path: &str, body: &str) -> String {
    format!(
        "POST {path} HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{body}",
        body.len()
    )
}

fn delete_post(request_id: &str, body: &str) -> String {
    format!(
        "POST /api/logs/requests/{request_id}/delete HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{body}",
        body.len()
    )
}

fn webhook_retry_post(delivery_id: &str, body: &str) -> String {
    format!(
        "POST /api/logs/webhooks/{delivery_id}/retry HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{body}",
        body.len()
    )
}

fn seed_terminal_summary(store: &mesh_llm_log_store::LogStore, request_id: &str, created_at: &str) {
    store
        .insert_summary(
            request_id,
            Some("safe-model"),
            Some("management"),
            None,
            None,
            created_at,
            None,
            None,
            None,
        )
        .unwrap();
    store
        .conn()
        .execute(
            "UPDATE summaries SET state = 'completed' WHERE request_id = ?1",
            [request_id],
        )
        .unwrap();
}

mod access_and_mutation;
mod event_stream;
mod read_and_export;
