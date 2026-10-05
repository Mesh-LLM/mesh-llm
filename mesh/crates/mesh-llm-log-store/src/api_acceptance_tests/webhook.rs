use super::*;

#[test]
fn webhook_delivery_insert_and_count() {
    let (store, clock, _tmp) = open_store();

    store
        .insert_summary(
            "req-1",
            None,
            None,
            None,
            None,
            &clock.now(),
            None,
            None,
            None,
        )
        .unwrap();
    store
        .insert_webhook_delivery("wh-1", Some("req-1"), &clock.now(), 1, Some(200))
        .expect("insert webhook delivery");

    assert_eq!(store.count_table("webhook_deliveries").unwrap(), 1);

    // Duplicate PK fails.
    let err = store
        .insert_webhook_delivery("wh-1", Some("req-1"), &clock.now(), 2, None)
        .unwrap_err();
    assert!(matches!(err, LogStoreError::AlreadyExists { .. }));

    // Without request_id.
    store
        .insert_webhook_delivery("wh-2", None, &clock.now(), 1, Some(500))
        .expect("insert webhook without request_id");

    assert_eq!(store.count_table("webhook_deliveries").unwrap(), 2);
}

#[test]
fn webhook_delivery_state_machine_is_idempotent_fenced_and_restart_safe() {
    let (store, clock, _tmp) = open_store();
    let created_at = "2026-08-04T12:00:00Z";
    seed_terminal_webhook_request(&store, created_at);
    assert_enqueue_requires_terminal_event(&store, created_at);
    enqueue_webhook_delivery_idempotently(&store, created_at);
    assert_webhook_delivery_intent_collisions_are_rejected(&store, created_at);
    let first = claim_initial_webhook_delivery(&store);
    let second = retry_and_claim_webhook_delivery(&store, first.claim_generation);
    let manual = dead_letter_and_claim_manual_retry(&store, second.claim_generation);
    complete_webhook_delivery_with_fencing(
        &store,
        manual.claim_generation,
        second.claim_generation,
    );
    assert_webhook_delivery_is_private_and_restart_safe(&store, clock);
}

#[test]
fn expired_exhausted_claim_continues_to_the_next_eligible_delivery() {
    let (store, _clock, _tmp) = open_store();
    for (request_id, event_id) in [
        ("request-stale", "event-stale"),
        ("request-ready", "event-ready"),
    ] {
        store
            .insert_summary(
                request_id,
                None,
                None,
                None,
                None,
                "2026-08-04T12:00:00Z",
                None,
                None,
                None,
            )
            .expect("insert terminal webhook owner");
        store
            .write_terminal_event(
                request_id,
                event_id,
                r#"{"type":"completed"}"#,
                "completed",
                Some(200),
                "2026-08-04T12:00:00Z",
            )
            .expect("write terminal webhook event");
    }
    store
        .enqueue_webhook_delivery(
            "stale-exhausted",
            "request-stale",
            "2026-08-04T12:00:00Z",
            1,
        )
        .expect("enqueue stale delivery");
    store
        .claim_next_webhook_delivery("2026-08-04T12:00:01Z", "2026-08-04T12:00:02Z")
        .expect("claim stale delivery")
        .expect("stale delivery is initially eligible");
    store
        .enqueue_webhook_delivery("ready-delivery", "request-ready", "2026-08-04T12:00:01Z", 1)
        .expect("enqueue ready delivery");

    let claimed = store
        .claim_next_webhook_delivery("2026-08-04T12:00:03Z", "2026-08-04T12:00:04Z")
        .expect("claim continues after stale dead letter")
        .expect("ready delivery is claimed in the same scheduler turn");
    assert_eq!(claimed.delivery_id, "ready-delivery");
    assert_eq!(
        store
            .webhook_delivery("stale-exhausted")
            .expect("load stale delivery")
            .expect("stale delivery exists")
            .state,
        WebhookDeliveryState::DeadLetter
    );
}

fn seed_terminal_webhook_request(store: &LogStore, created_at: &str) {
    store
        .insert_summary(
            "request-terminal",
            None,
            None,
            None,
            None,
            created_at,
            None,
            None,
            None,
        )
        .expect("insert terminal summary owner");
}

fn assert_enqueue_requires_terminal_event(store: &LogStore, created_at: &str) {
    assert!(matches!(
        store.enqueue_webhook_delivery("before-terminal", "request-terminal", created_at, 2),
        Err(LogStoreError::InvalidQuery(message)) if message.contains("durable terminal event")
    ));
    store
        .write_terminal_event(
            "request-terminal",
            "event-terminal",
            r#"{"type":"completed"}"#,
            "completed",
            Some(201),
            created_at,
        )
        .expect("commit terminal before webhook enqueue");
}

fn enqueue_webhook_delivery_idempotently(store: &LogStore, created_at: &str) {
    let created = store
        .enqueue_webhook_delivery("delivery-terminal", "request-terminal", created_at, 2)
        .expect("enqueue terminal webhook");
    assert!(matches!(
        created,
        WebhookDeliveryInsertOutcome::Created(WebhookDeliveryRecord {
            terminal_outcome: WebhookTerminalOutcome::Completed,
            terminal_status_code: Some(201),
            response_status_code: None,
            state: WebhookDeliveryState::Pending,
            attempt_number: 0,
            max_attempts: 2,
            ..
        })
    ));
    assert!(matches!(
        store
            .enqueue_webhook_delivery("delivery-terminal", "request-terminal", created_at, 2)
            .expect("idempotent enqueue"),
        WebhookDeliveryInsertOutcome::Existing(_)
    ));
}

fn assert_webhook_delivery_intent_collisions_are_rejected(store: &LogStore, created_at: &str) {
    fn assert_intent_conflict(result: Result<WebhookDeliveryInsertOutcome, LogStoreError>) {
        assert!(matches!(
            result,
            Err(LogStoreError::InvalidQuery(message))
                if message.contains("immutable delivery intent")
        ));
    }

    assert_intent_conflict(store.enqueue_webhook_delivery(
        "delivery-terminal",
        "request-terminal",
        created_at,
        3,
    ));

    store
        .insert_summary(
            "request-collision",
            None,
            None,
            None,
            None,
            created_at,
            None,
            None,
            None,
        )
        .expect("collision summary");
    for (column, replacement, original) in [
        ("request_id", "request-collision", "request-terminal"),
        ("terminal_outcome", "failed", "completed"),
        ("terminal_status_code", "202", "201"),
    ] {
        store
            .conn()
            .execute(
                &format!(
                    "UPDATE webhook_deliveries SET {column} = ? WHERE delivery_id = 'delivery-terminal'"
                ),
                [replacement],
            )
            .expect("mutate one immutable intent field");
        assert_intent_conflict(store.enqueue_webhook_delivery(
            "delivery-terminal",
            "request-terminal",
            created_at,
            2,
        ));
        store
            .conn()
            .execute(
                &format!(
                    "UPDATE webhook_deliveries SET {column} = ? WHERE delivery_id = 'delivery-terminal'"
                ),
                [original],
            )
            .expect("restore immutable intent field");
    }
}

fn claim_initial_webhook_delivery(store: &LogStore) -> WebhookDeliveryRecord {
    let first = store
        .claim_next_webhook_delivery("2026-08-04T12:00:01Z", "2026-08-04T12:01:01Z")
        .expect("claim first attempt")
        .expect("pending delivery is claimable");
    assert_eq!(first.state, WebhookDeliveryState::InFlight);
    assert_eq!(first.terminal_outcome, WebhookTerminalOutcome::Completed);
    assert_eq!(first.terminal_status_code, Some(201));
    assert_eq!(first.response_status_code, None);
    assert_eq!(first.attempt_number, 1);
    assert_eq!(first.claim_generation, 1);
    assert!(
        store
            .claim_next_webhook_delivery("2026-08-04T12:00:02Z", "2026-08-04T12:01:02Z")
            .expect("second claim")
            .is_none(),
        "an active lease excludes duplicate worker wakeups"
    );
    first
}

fn retry_and_claim_webhook_delivery(
    store: &LogStore,
    claim_generation: u64,
) -> WebhookDeliveryRecord {
    assert_eq!(
        store
            .retry_or_dead_letter_webhook_delivery(
                "delivery-terminal",
                claim_generation,
                "2026-08-04T12:00:03Z",
                "2026-08-04T12:00:10Z",
                WebhookDeliveryErrorCode::Timeout,
                None,
            )
            .expect("schedule retry"),
        Some(WebhookRetryOutcome::RetryScheduled)
    );
    assert!(
        store
            .claim_next_webhook_delivery("2026-08-04T12:00:09Z", "2026-08-04T12:01:09Z")
            .expect("claim before retry")
            .is_none()
    );

    let second = store
        .claim_next_webhook_delivery("2026-08-04T12:00:10Z", "2026-08-04T12:01:10Z")
        .expect("claim retry")
        .expect("retry is eligible");
    assert_eq!(second.attempt_number, 2);
    assert_eq!(second.terminal_outcome, WebhookTerminalOutcome::Completed);
    assert_eq!(second.terminal_status_code, Some(201));
    assert_eq!(second.response_status_code, None);
    second
}

fn dead_letter_and_claim_manual_retry(
    store: &LogStore,
    claim_generation: u64,
) -> WebhookDeliveryRecord {
    assert_eq!(
        store
            .retry_or_dead_letter_webhook_delivery(
                "delivery-terminal",
                claim_generation,
                "2026-08-04T12:00:11Z",
                "2026-08-04T12:00:20Z",
                WebhookDeliveryErrorCode::Http5xx,
                Some(503),
            )
            .expect("dead-letter exhausted delivery"),
        Some(WebhookRetryOutcome::DeadLettered)
    );
    assert_eq!(
        store
            .manually_retry_webhook_delivery("delivery-terminal", "2026-08-04T12:00:12Z")
            .expect("manual retry dead letter"),
        WebhookManualRetryOutcome::Scheduled
    );

    let manual = store
        .claim_next_webhook_delivery("2026-08-04T12:00:12Z", "2026-08-04T12:01:12Z")
        .expect("claim manual retry")
        .expect("manual retry is eligible");
    assert_eq!(manual.state, WebhookDeliveryState::InFlight);
    assert_eq!(manual.terminal_outcome, WebhookTerminalOutcome::Completed);
    assert_eq!(manual.terminal_status_code, Some(201));
    assert_eq!(manual.response_status_code, Some(503));
    assert_eq!(manual.attempt_number, 1);
    assert_eq!(manual.claim_generation, 3);
    assert_eq!(
        store
            .manually_retry_webhook_delivery("delivery-terminal", "2026-08-04T12:00:12Z")
            .expect("manual retry remains idempotent after scheduler claim"),
        WebhookManualRetryOutcome::AlreadyScheduled
    );
    manual
}

fn complete_webhook_delivery_with_fencing(
    store: &LogStore,
    winning_claim_generation: u64,
    stale_claim_generation: u64,
) {
    assert!(
        store
            .complete_webhook_delivery(
                "delivery-terminal",
                winning_claim_generation,
                "2026-08-04T12:00:13Z",
                204,
            )
            .expect("complete fenced delivery")
    );
    assert!(
        !store
            .complete_webhook_delivery(
                "delivery-terminal",
                stale_claim_generation,
                "2026-08-04T12:00:14Z",
                205,
            )
            .expect("stale completion is harmless"),
        "a displaced worker cannot overwrite the fenced completion"
    );
}

fn assert_webhook_delivery_is_private_and_restart_safe(
    store: &LogStore,
    clock: Arc<dyn ClockTrait>,
) {
    let record = store
        .webhook_delivery("delivery-terminal")
        .expect("load delivery")
        .expect("delivery persisted");
    assert_eq!(record.state, WebhookDeliveryState::Succeeded);
    assert_eq!(record.terminal_outcome, WebhookTerminalOutcome::Completed);
    assert_eq!(record.terminal_status_code, Some(201));
    assert_eq!(record.response_status_code, Some(204));
    assert_eq!(record.last_error_code, None);
    let (target, body, error): (String, Option<String>, Option<String>) = store
        .conn()
        .query_row(
            "SELECT target_url, response_body, error_msg FROM webhook_deliveries WHERE delivery_id = ?",
            ["delivery-terminal"],
            |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
        )
        .expect("inspect privacy-safe storage");
    assert_eq!(target, "configured_webhook");
    assert!(body.is_none());
    assert!(error.is_none());

    let reopened = store.reopen(clock).expect("reopen webhook database");
    let reopened_record = reopened
        .webhook_delivery("delivery-terminal")
        .expect("load after restart")
        .expect("record after restart");
    assert_eq!(reopened_record.state, WebhookDeliveryState::Succeeded);
    assert_eq!(
        reopened_record.terminal_outcome,
        WebhookTerminalOutcome::Completed
    );
    assert_eq!(reopened_record.terminal_status_code, Some(201));
    assert_eq!(reopened_record.response_status_code, Some(204));
}

#[test]
fn cleanup_run_insert_and_count() {
    let (store, clock, _tmp) = open_store();

    store
        .insert_cleanup_run(
            "cr-1",
            &clock.now(),
            "daily-cleanup",
            "2025-01-01T00:00:00Z",
            42,
            Some(150),
        )
        .expect("insert cleanup run");

    assert_eq!(store.count_table("cleanup_runs").unwrap(), 1);

    // Duplicate PK fails.
    let err = store
        .insert_cleanup_run(
            "cr-1",
            &clock.now(),
            "other-policy",
            "2025-02-01T00:00:00Z",
            10,
            None,
        )
        .unwrap_err();
    assert!(matches!(err, LogStoreError::AlreadyExists { .. }));
}
