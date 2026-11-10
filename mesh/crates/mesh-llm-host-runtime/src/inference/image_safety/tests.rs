use super::*;

fn status() -> GuardStatus {
    GuardStatus {
        protocol_version: 1,
        ready: true,
        screen_received_images: true,
        model: "guard".into(),
        policy_revision: "child-safety-v1".into(),
    }
}

fn verdict(decision: GuardDecision) -> GuardVerdict {
    GuardVerdict {
        decision,
        model: "guard".into(),
        policy_revision: "child-safety-v1".into(),
        reason_code: "safe".into(),
    }
}

#[test]
fn only_explicit_matching_allow_permits_release() {
    let status = status();
    assert_eq!(
        check_verdict(verdict(GuardDecision::Allow), &status),
        Ok(())
    );
    assert_eq!(
        check_verdict(verdict(GuardDecision::Block), &status),
        Err(ImageSafetyResponseError::Denied)
    );
    assert_eq!(
        check_verdict(verdict(GuardDecision::Unavailable), &status),
        Err(ImageSafetyResponseError::Unavailable)
    );
    let mut wrong = verdict(GuardDecision::Allow);
    wrong.policy_revision = "different".into();
    assert_eq!(
        check_verdict(wrong, &status),
        Err(ImageSafetyResponseError::Unavailable)
    );
    let mut unbounded_reason = verdict(GuardDecision::Allow);
    unbounded_reason.reason_code = "raw classifier rationale".into();
    assert_eq!(
        check_verdict(unbounded_reason, &status),
        Err(ImageSafetyResponseError::Unavailable)
    );
}

#[test]
fn effective_input_rejects_duplicate_fields_instead_of_screening_only_the_last() {
    for body in [
        r#"{"model":"demo","prompt":"first","prompt":"second"}"#,
        r#"{"model":"demo","messages":[{"role":"user","content":"first","content":"second"}]}"#,
        r#"{"model":"demo","messages":[{"role":"user","content":"first","cont\u0065nt":"second"}]}"#,
        r#"{"model":"demo","prompt":"hello","chat_template_kwargs":{"instruction":"first","instruction":"second"}}"#,
    ] {
        assert_eq!(
            effective_text(body.as_bytes()),
            Err(ImageSafetyResponseError::Unavailable)
        );
    }
}

#[test]
fn effective_input_preserves_all_turns_instructions_and_tools() {
    let body = serde_json::json!({"model":"demo", "instructions":"policy", "messages":[
        {"role":"system","content":"system instructions"},
        {"role":"user","content":"first turn"},
        {"role":"assistant","tool_calls":[{"id":"call","function":{"name":"lookup","arguments":"argument"}}]},
        {"role":"tool","tool_call_id":"call","content":"tool result"},
        {"role":"user","content":"latest turn"}],
        "tools":[{"type":"function","function":{"name":"lookup","description":"tool policy"}}],
        "chat_template_kwargs":{"provider_instruction":"template input"},
        "prefill_assistant":"assistant prefill", "negative_prompt":"negative input", "system_prompt":"provider system"});
    let text = effective_text(&serde_json::to_vec(&body).unwrap()).unwrap();
    for expected in [
        "policy",
        "system instructions",
        "first turn",
        "argument",
        "tool result",
        "latest turn",
        "tool policy",
        "template input",
        "assistant prefill",
        "negative input",
        "provider system",
    ] {
        assert!(text.contains(expected));
    }
}

#[test]
fn oversized_or_media_input_never_gets_silently_truncated() {
    let large = serde_json::json!({"input":"x".repeat(MAX_TEXT_BYTES)});
    assert!(effective_text(&serde_json::to_vec(&large).unwrap()).is_err());
    for kind in [
        "input_image",
        "image",
        "input_audio",
        "audio",
        "input_file",
        "video",
        "video_url",
        "input_video",
        "document",
    ] {
        let media = serde_json::json!({"input":[{"type":kind,"data":"fixture"}]});
        assert!(effective_text(&serde_json::to_vec(&media).unwrap()).is_err());
    }
    assert!(effective_text(b"{}").is_err());
    assert!(effective_text(b"invalid").is_err());
    assert!(effective_text(br#"{"messages":[]}"#).is_err());
    assert!(effective_text(br#"{"input":"   "}"#).is_err());
    assert!(
        effective_text(
            br#"{"messages":[{"role":"user"}],"tools":[{"description":"not an input"}]}"#
        )
        .is_err()
    );
    assert!(effective_text(br#"{"input":"benign","chat_template_kwargs":{"hidden":{"type":"input_image","data":"fixture"}}}"#).is_err());
}
