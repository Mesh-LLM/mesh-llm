use serde_json::{Value, json};
use std::{
    fs,
    process::{Command, Output},
};

const FACTS: &str =
    "CODEWORD=signal-7429\nCHECKSUM=FS-319-DELTA\nPRIME_SUM=10\nQUESTION=facts/signal.md\n";

fn invoke(verb: &str, bytes: &[u8]) -> Output {
    let state = tempfile::tempdir().unwrap();
    let path = state
        .path()
        .join("actual client event log with spaces.jsonl");
    fs::write(&path, bytes).unwrap();
    Command::new(env!("CARGO_BIN_EXE_xtask"))
        .args(["automation", "agent-fixture-evidence", verb])
        .arg(path)
        .env("PATH", "")
        .output()
        .unwrap()
}

fn check(output: Output, success: bool) -> Output {
    assert_eq!(output.status.success(), success, "{output:?}");
    if !success {
        assert!(output.stdout.is_empty());
        assert!(!output.stderr.is_empty());
    }
    output
}

fn tool(name: &str) -> Value {
    json!({"type":"tool_use","part":{"tool":name,"state":{"status":"completed"}}})
}

fn document(events: &[Value], plain: &str) -> Vec<u8> {
    let mut text = String::from("non-event client log\n{broken\n[]\nnull\n");
    for event in events {
        text.push_str(&event.to_string());
        text.push('\n');
    }
    text.push_str(plain);
    text.into_bytes()
}

#[test]
fn session_is_the_first_actual_nonempty_string_and_never_synthesized_from_noise() {
    let input = document(
        &[
            json!({"sessionID":null}),
            json!({"sessionID":42}),
            json!({"sessionID":""}),
            json!({"sessionID":"ses_literal:雪"}),
            json!({"sessionID":"ses_later"}),
        ],
        "",
    );
    let output = check(invoke("opencode-session", &input), true);
    assert_eq!(
        String::from_utf8(output.stdout).unwrap(),
        "ses_literal:雪\n"
    );
    for invalid in [
        b"not-json\n{broken\n".as_slice(),
        b"[]\n42\n",
        b"{\"sessionID\":false}\n",
        b"{\"sessionID\":\" \"}\n",
        b"{\"sessionID\":\"line\\nsecond\"}\n",
        b"{\"sessionID\":\"\xff\"}\n",
    ] {
        check(invoke("opencode-session", invalid), false);
    }
}

#[test]
fn actual_failure_event_rejects_even_after_a_valid_session() {
    let bytes = document(
        &[
            json!({"sessionID":"ses_actual"}),
            json!({"type":"error","error":{"message":"client failed"}}),
        ],
        "",
    );
    check(invoke("opencode-session", &bytes), false);
    let bytes = document(
        &[
            json!({"sessionID":"ses_actual"}),
            json!({"type":"tool_use","part":{"tool":"edit","state":{"status":"error","error":"write failed"}}}),
        ],
        "",
    );
    check(invoke("opencode-session", &bytes), false);
}

#[test]
fn opencode_requires_four_filesystem_events_and_its_own_edit_names() {
    for name in ["edit", "write", "apply_patch"] {
        let events = [
            tool("read"),
            tool("glob"),
            tool("bash"),
            tool(name),
            json!({"type":"text","part":{"text":FACTS}}),
        ];
        let output = check(invoke("opencode-result", &document(&events, "")), true);
        assert_eq!(
            String::from_utf8(output.stdout).unwrap(),
            format!("OpenCode multi-turn coding smoke passed\n  tools: read, glob, bash, {name}\n")
        );
    }
    for names in [
        vec!["read", "glob", "edit"],
        vec!["read", "glob", "search", "edit"],
        vec!["read", "glob", "bash", "grep"],
        vec!["read", "glob", "bash", "developer__text_editor"],
    ] {
        check(
            invoke(
                "opencode-result",
                &document(&names.into_iter().map(tool).collect::<Vec<_>>(), FACTS),
            ),
            false,
        );
    }
}

#[test]
fn tool_and_name_aliases_count_once_but_repeated_real_calls_remain_events() {
    let alias = json!({"type":"tool_use","toolName":"edit","part":{"tool":"edit","name":"write"}});
    check(
        invoke(
            "opencode-result",
            &document(&[alias.clone(), alias.clone()], FACTS),
        ),
        false,
    );
    check(
        invoke(
            "opencode-result",
            &document(&[alias.clone(), tool("read")], FACTS),
        ),
        false,
    );
    let output = check(
        invoke(
            "opencode-result",
            &document(&[alias.clone(), alias.clone(), alias.clone(), alias], FACTS),
        ),
        true,
    );
    assert_eq!(
        String::from_utf8(output.stdout).unwrap(),
        "OpenCode multi-turn coding smoke passed\n  tools: edit, edit, edit, edit\n"
    );
    let name = json!({"type":"tool_use","part":{"name":"apply_patch"}});
    check(
        invoke(
            "opencode-result",
            &document(
                &[
                    tool("read"),
                    tool("read"),
                    tool("read"),
                    json!({"type":"tool_use","part":{"tool":"read","name":"edit"}}),
                ],
                FACTS,
            ),
        ),
        false,
    );
    check(
        invoke(
            "opencode-result",
            &document(&[tool("read"), tool("read"), tool("read"), name], FACTS),
        ),
        true,
    );
    check(
        invoke(
            "opencode-result",
            &document(
                &[
                    json!({"toolName":"edit","nested":{"type":"tool_use","part":{"tool":"edit"}}}),
                    tool("read"),
                    tool("read"),
                    tool("read"),
                ],
                FACTS,
            ),
        ),
        false,
    );
}

#[test]
fn literal_facts_are_required_in_answer_text_or_plain_output() {
    let calls = [tool("read"), tool("glob"), tool("bash"), tool("edit")];
    for answer in [
        json!({"type":"assistant","message":{"content":[{"text":FACTS}]}}),
        json!({"type":"message","delta":FACTS}),
    ] {
        let mut events = calls.to_vec();
        events.push(answer);
        check(invoke("opencode-result", &document(&events, "")), true);
    }
    check(invoke("opencode-result", &document(&calls, FACTS)), true);
    for text in [
        FACTS.replace("PRIME_SUM=10", "PRIME_SUM=31"),
        FACTS.replace("CODEWORD=signal-7429", "prefixCODEWORD=signal-7429"),
        FACTS.replace("QUESTION=facts/signal.md", "QUESTION=facts/signal.md.extra"),
    ] {
        check(invoke("opencode-result", &document(&calls, &text)), false);
    }
    let mut events = calls.to_vec();
    events.push(json!({"type":"tool_use","part":{"tool":"read","output":FACTS}}));
    check(invoke("opencode-result", &document(&events, "")), false);
}

#[test]
fn completed_tool_counts_and_facts_cannot_hide_client_or_tool_failure() {
    let base = [tool("read"), tool("glob"), tool("bash"), tool("edit")];
    for failure in [
        json!({"type":"error","error":{"message":"failed"}}),
        json!({"type":"tool_use","part":{"tool":"edit","state":{"status":"failed"}}}),
        json!({"type":"tool_use","part":{"tool":"write","state":{"success":false}}}),
        json!({"type":"tool_use","part":{"tool":"apply_patch","state":{"error":"patch failed"}}}),
    ] {
        let mut events = base.to_vec();
        events.push(failure);
        check(invoke("opencode-result", &document(&events, FACTS)), false);
    }
    check(invoke("opencode-result", b"not-json\n{broken\n"), false);
    check(
        invoke("opencode-result", &vec![b' '; 8 * 1024 * 1024 + 1]),
        false,
    );
}
