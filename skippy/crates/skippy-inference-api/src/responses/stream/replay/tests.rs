use std::sync::Arc;

use serde_json::{Value, json};

use super::*;
use crate::responses::{
    continuation::ResponseContinuation,
    store::{CallerScope, ResponseSessionCache, ResponseStore},
    turn::ResponsesTurn,
};

fn completed_chat(text: &str) -> Value {
    json!({
        "choices": [{
            "message": {"role": "assistant", "content": text},
            "finish_reason": "stop"
        }]
    })
}

fn turn_with_both_continuations() -> (ResponsesTurn, Arc<ResponseStore>, Arc<ResponseSessionCache>)
{
    let store = Arc::new(ResponseStore::default());
    let session = Arc::new(ResponseSessionCache::new(1024 * 1024));
    let continuation = ResponseContinuation::for_session(
        Some(store.clone()),
        Some(CallerScope::LocalHost),
        session.clone(),
        true,
    );
    let mut request = json!({"model": "m", "input": "hello"});
    let turn = ResponsesTurn::prepare(&mut request, continuation).unwrap();
    (turn, store, session)
}

fn assert_failed_lifecycle(events: &[ResponsesStreamEvent]) {
    assert_eq!(
        events.iter().map(|event| event.event).collect::<Vec<_>>(),
        ["response.created", "error", "response.failed"]
    );
    for (sequence, event) in events.iter().enumerate() {
        assert_eq!(event.data["sequence_number"], sequence as u64);
    }
    let failed = &events.last().unwrap().data["response"];
    assert_eq!(failed["status"], "failed");
    assert_eq!(failed["store"], false);
    assert_eq!(failed["output"], json!([]));
}

#[test]
fn event_or_byte_rejection_commits_neither_shared_nor_session_state() {
    for limits in [
        ReplayLimits {
            max_bytes: usize::MAX,
            max_events: 1,
        },
        ReplayLimits {
            max_bytes: 1,
            max_events: usize::MAX,
        },
    ] {
        let (turn, store, session) = turn_with_both_continuations();
        let id = turn.id().to_string();
        let events = stream_events_for_chat_completion_with_limits(
            &turn,
            &completed_chat("accepted only if replay fits"),
            limits,
        );

        assert_failed_lifecycle(&events);
        assert!(store.get(&CallerScope::LocalHost, &id).is_err());
        assert!(session.get(&id).is_none());
    }
}

#[test]
fn admitted_replay_commits_both_continuation_locations() {
    let (turn, store, session) = turn_with_both_continuations();
    let id = turn.id().to_string();
    let events = stream_events_for_chat_completion(&turn, &completed_chat("accepted"));

    let completed = events.last().unwrap();
    assert_eq!(completed.event, "response.completed");
    assert_eq!(completed.data["response"]["store"], true);
    assert!(store.get(&CallerScope::LocalHost, &id).is_ok());
    assert!(session.get(&id).is_some());
}

#[test]
fn whole_resource_replay_rejects_incrementally_without_partial_lifecycle() {
    let marker = "rejected-output-marker";
    let response = json!({
        "id": "resp_large",
        "created_at": 7,
        "model": "m",
        "status": "completed",
        "output": [{
            "id": "msg_large",
            "type": "message",
            "status": "completed",
            "role": "assistant",
            "content": [{
                "type": "output_text",
                "text": format!("{marker}{}", "x".repeat(16 * 1024)),
                "annotations": []
            }]
        }]
    });

    let mut rejected = response;
    rejected["id"] = Value::String("x".repeat(1024));
    let failed = stream_events_for_response_with_limits(
        &rejected,
        ReplayLimits {
            max_bytes: 1024,
            max_events: usize::MAX,
        },
    );
    assert_failed_lifecycle(&failed);
    let serialized = failed
        .iter()
        .map(|event| event.data.to_string())
        .collect::<String>();
    assert!(!serialized.contains(marker));
    assert!(!serialized.contains(&"x".repeat(1024)));
    assert!(failed[0].data["response"].get("tools").is_some());
    assert!(failed[2].data["response"].get("tool_choice").is_some());
}

#[test]
fn event_count_rejection_does_not_return_truncated_item_indices() {
    let response = json!({
        "id": "resp_many",
        "model": "m",
        "status": "completed",
        "output": (0..4).map(|index| json!({
            "id": format!("msg_{index}"),
            "type": "message",
            "status": "completed",
            "role": "assistant",
            "content": [{"type": "output_text", "text": "ok", "annotations": []}]
        })).collect::<Vec<_>>()
    });

    let failed = stream_events_for_response_with_limits(
        &response,
        ReplayLimits {
            max_bytes: usize::MAX,
            max_events: 4,
        },
    );
    assert_failed_lifecycle(&failed);
}
