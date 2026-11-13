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

fn turn_with_obfuscation() -> (ResponsesTurn, Arc<ResponseStore>, Arc<ResponseSessionCache>) {
    let store = Arc::new(ResponseStore::default());
    let session = Arc::new(ResponseSessionCache::new(1024 * 1024));
    let continuation = ResponseContinuation::for_session(
        Some(store.clone()),
        Some(CallerScope::LocalHost),
        session.clone(),
        true,
    );
    let mut request = json!({
        "model":"m",
        "input":"hello",
        "stream_options":{"include_obfuscation":true},
    });
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

fn replay_bytes(turn: &ResponsesTurn, chat: &Value) -> usize {
    let staged = turn.stage_chat_completion(chat);
    let events = try_stream_events_for_response(
        staged.response(),
        ReplayLimits::default(),
        turn.obfuscation_policy(),
        0,
        true,
    )
    .expect("valid response fits default replay limits");
    assert!(events.iter().any(|event| {
        event.event == "response.output_text.delta" && event.data["obfuscation"].as_str().is_some()
    }));
    let padded_bytes = events
        .iter()
        .map(|event| serde_json::to_vec(&event.data).unwrap().len())
        .sum();
    let unpadded_bytes = events
        .iter()
        .map(|event| {
            let mut data = event.data.clone();
            data.as_object_mut().unwrap().remove("obfuscation");
            serde_json::to_vec(&data).unwrap().len()
        })
        .sum();
    assert!(padded_bytes > unpadded_bytes);
    padded_bytes
}

#[test]
fn exact_padding_bytes_are_admitted_before_any_continuation_commit() {
    let chat = completed_chat("padding changes this replay budget");
    let (accepted_turn, accepted_store, accepted_session) = turn_with_obfuscation();
    let accepted_id = accepted_turn.id().to_string();
    assert!(
        accepted_store
            .get(&CallerScope::LocalHost, &accepted_id)
            .is_err()
    );
    assert!(accepted_session.get(&accepted_id).is_none());
    let exact_padded_bytes = replay_bytes(&accepted_turn, &chat);
    let accepted = stream_events_for_chat_completion_with_limits(
        &accepted_turn,
        &chat,
        ReplayLimits {
            max_bytes: exact_padded_bytes,
            max_events: usize::MAX,
        },
    );

    assert_eq!(accepted.last().unwrap().event, "response.completed");
    assert!(
        accepted_store
            .get(&CallerScope::LocalHost, &accepted_id)
            .is_ok()
    );
    assert!(accepted_session.get(&accepted_id).is_some());

    let (rejected_turn, rejected_store, rejected_session) = turn_with_obfuscation();
    let rejected_id = rejected_turn.id().to_string();
    assert!(
        rejected_store
            .get(&CallerScope::LocalHost, &rejected_id)
            .is_err()
    );
    assert!(rejected_session.get(&rejected_id).is_none());
    let rejected_padded_bytes = replay_bytes(&rejected_turn, &chat);
    assert_eq!(rejected_padded_bytes, exact_padded_bytes);
    let rejected = stream_events_for_chat_completion_with_limits(
        &rejected_turn,
        &chat,
        ReplayLimits {
            max_bytes: rejected_padded_bytes - 1,
            max_events: usize::MAX,
        },
    );

    assert_failed_lifecycle(&rejected);
    assert!(
        rejected_store
            .get(&CallerScope::LocalHost, &rejected_id)
            .is_err()
    );
    assert!(rejected_session.get(&rejected_id).is_none());
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

#[test]
fn max_tool_calls_rejects_buffered_replay_before_calls_or_continuation_commit() {
    let store = Arc::new(ResponseStore::default());
    let session = Arc::new(ResponseSessionCache::new(1024 * 1024));
    let continuation = ResponseContinuation::for_session(
        Some(store.clone()),
        Some(CallerScope::LocalHost),
        session.clone(),
        true,
    );
    let mut request = json!({
        "model": "m",
        "input": "hello",
        "max_tool_calls": 1,
        "tools": [{"type": "function", "name": "same"}],
    });
    let turn = ResponsesTurn::prepare(&mut request, continuation).unwrap();
    let id = turn.id().to_string();
    let chat = json!({
        "choices": [{
            "message": {"role": "assistant", "content": null, "tool_calls": [
                {"id":"c0","type":"function","function":{"name":"same","arguments":"{}"}},
                {"id":"c1","type":"function","function":{"name":"same","arguments":"{}"}}
            ]},
            "finish_reason": "tool_calls"
        }]
    });

    let events = stream_events_for_chat_completion(&turn, &chat);
    assert_eq!(
        events.iter().map(|event| event.event).collect::<Vec<_>>(),
        [
            "response.created",
            "response.in_progress",
            "response.failed"
        ]
    );
    for (sequence, event) in events.iter().enumerate() {
        assert_eq!(event.data["sequence_number"], sequence as u64);
    }
    assert_eq!(events.last().unwrap().data["response"]["store"], false);
    assert_eq!(events.last().unwrap().data["response"]["output"], json!([]));
    assert_eq!(
        events.last().unwrap().data["response"]["error"]["code"],
        "tool_call_limit_exceeded"
    );
    assert!(events.iter().all(|event| {
        !matches!(
            event.event,
            "response.output_item.added"
                | "response.function_call_arguments.delta"
                | "response.function_call_arguments.done"
        )
    }));
    assert!(store.get(&CallerScope::LocalHost, &id).is_err());
    assert!(session.get(&id).is_none());
}
