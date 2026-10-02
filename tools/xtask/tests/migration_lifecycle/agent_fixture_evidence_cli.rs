use serde_json::json;
use std::{
    fs,
    process::{Command, Output},
};

const FACTS: &str =
    "CODEWORD=signal-7429\nCHECKSUM=FS-319-DELTA\nPRIME_SUM=10\nQUESTION=facts/signal.md\n";

fn invoke(verb: &str, bytes: &[u8], require: Option<&str>) -> Output {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("client evidence with spaces.jsonl");
    fs::write(&path, bytes).unwrap();
    let mut command = Command::new(env!("CARGO_BIN_EXE_xtask"));
    command
        .args(["automation", "agent-fixture-evidence", verb])
        .arg(path)
        .arg("Pi fixture");
    if let Some(require) = require {
        command.arg(require);
    }
    command.output().unwrap()
}

fn expect(output: Output, success: bool) {
    assert_eq!(
        output.status.success(),
        success,
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    if !success {
        assert!(output.stdout.is_empty());
    }
}

#[test]
fn soak_response_requires_first_choice_sentinels_and_rejects_failed_response() {
    let good = json!({"choices":[{"message":{"content":"LONG_SOAK=ALPHA-719|MID-482|OMEGA-503"}}]});
    expect(
        invoke("soak", &serde_json::to_vec(&good).unwrap(), None),
        true,
    );
    for value in [
        json!({}),
        json!({"choices":[]}),
        json!({"choices":[{"message":{"content":"ALPHA-719"}}]}),
        json!({"choices":[{"message":{"content":["LONG_SOAK=ALPHA-719|MID-482|OMEGA-503"]}}]}),
        json!({"error":{"message":"failed"},"choices":good["choices"]}),
    ] {
        expect(
            invoke("soak", &serde_json::to_vec(&value).unwrap(), None),
            false,
        );
    }
    expect(invoke("soak", b"not-json", None), false);
}

fn successful_tools() -> String {
    format!(
        "{}\n{}\n{}\n",
        json!({"nested":[{"toolName":"read"}]}),
        json!({"tool_name":"glob"}),
        json!({"type":"toolRequest","toolRequest":{"name":"developer__text_editor"}})
    )
}

#[test]
fn result_preserves_nested_tools_literal_facts_and_malformed_log_tolerance() {
    let data = format!(
        "non-json client log\n{{broken\n{}{FACTS}",
        successful_tools()
    );
    let result = invoke("result", data.as_bytes(), Some("true"));
    assert!(String::from_utf8_lossy(&result.stdout).contains("read, glob, developer__text_editor"));
    expect(result, true);
    expect(invoke("result", FACTS.as_bytes(), Some("false")), true);
}

#[test]
fn tool_aliases_count_once_per_event_and_repeated_actual_calls_still_count() {
    let edit = json!({"type":"tool_call","toolName":"edit","toolCall":{"name":"edit"}});
    let read = json!({"type":"tool_call","toolName":"read","toolCall":{"name":"read"}});
    expect(
        invoke(
            "result",
            format!("{edit}\n{read}\n{FACTS}").as_bytes(),
            Some("true"),
        ),
        false,
    );
    expect(
        invoke(
            "result",
            format!("{edit}\n{edit}\n{edit}\n{FACTS}").as_bytes(),
            Some("true"),
        ),
        true,
    );
}

#[test]
fn result_rejects_missing_or_wrong_facts_no_tools_no_edit_and_explicit_failed_results() {
    expect(
        invoke(
            "result",
            format!(
                "{{\"tool_name\":\"\"}}\n{{\"tool_name\":\"\"}}\n{{\"toolName\":\"edit\"}}\n{FACTS}"
            )
            .as_bytes(),
            Some("true"),
        ),
        false,
    );
    let valid = format!("{}{FACTS}", successful_tools());
    for removed in FACTS.lines() {
        expect(
            invoke(
                "result",
                valid.replace(removed, "WRONG_FACT").as_bytes(),
                Some("true"),
            ),
            false,
        );
    }
    expect(invoke("result", FACTS.as_bytes(), Some("true")), false);
    expect(
        invoke(
            "result",
            valid.replace("developer__text_editor", "read").as_bytes(),
            Some("true"),
        ),
        false,
    );
    for failure in [
        json!({"type":"error","message":"client failed"}),
        json!({"type":"tool_result","isError":true}),
        json!({"type":"toolResult","success":false}),
        json!({"type":"tool_result","status":"failed"}),
    ] {
        expect(
            invoke(
                "result",
                format!("{valid}{failure}\n").as_bytes(),
                Some("true"),
            ),
            false,
        );
    }
    expect(invoke("result", valid.as_bytes(), Some("maybe")), false);
}

#[test]
fn input_line_and_nesting_limits_fail_closed() {
    expect(
        invoke("result", &vec![b'x'; 8 * 1024 * 1024 + 1], Some("false")),
        false,
    );
    expect(
        invoke(
            "result",
            format!("{}{FACTS}", "\n".repeat(100_001)).as_bytes(),
            Some("false"),
        ),
        false,
    );
    let nested = format!("{}0{}\n{FACTS}", "[".repeat(256), "]".repeat(256));
    expect(invoke("result", nested.as_bytes(), Some("false")), false);
}

#[test]
fn aggregate_json_node_limit_is_enforced_without_byte_limit() {
    let input = format!("[{}0]\n{FACTS}", "0,".repeat(1_000_001));
    expect(invoke("result", input.as_bytes(), Some("false")), false);
}

#[test]
fn shared_caller_preserves_isolated_home_and_propagates_real_cli_rejection() {
    let root = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    let directory = tempfile::tempdir().unwrap();
    let evidence = directory.path().join("fixture with spaces.jsonl");
    let isolated = directory.path().join("isolated client home");
    fs::create_dir(&isolated).unwrap();
    for (bytes, success) in [
        (FACTS.as_bytes(), false),
        (format!("{}{FACTS}", successful_tools()).as_bytes(), true),
    ] {
        fs::write(&evidence, bytes).unwrap();
        let result = Command::new("/bin/bash").args(["-c",
            "source \"$1\"; HOME=\"$2\"; export HOME; agent_smoke_evidence result \"$3\" 'Pi fixture' true; status=$?; [[ \"$HOME\" == \"$2\" ]] || exit 99; exit $status", "fixture"])
            .arg(root.join("scripts/ci-agent-live-fixture-lib.sh")).arg(&isolated).arg(&evidence)
            .env("MESH_LLM_AUTOMATION_BIN", env!("CARGO_BIN_EXE_xtask")).output().unwrap();
        expect(result, success);
    }
}

#[test]
fn compatibility_probe_requires_chat_completion_and_choices_without_error() {
    let good = json!({"object":"chat.completion","choices":[{"message":{"content":"ok"}}]});
    expect(
        invoke("probe", &serde_json::to_vec(&good).unwrap(), None),
        true,
    );
    for value in [
        json!({}),
        json!({"object":"wrong","choices":good["choices"]}),
        json!({"object":"chat.completion","choices":[]}),
        json!({"object":"chat.completion","choices":{}}),
        json!({"object":"chat.completion","choices":good["choices"],"error":{"message":"failed"}}),
    ] {
        expect(
            invoke("probe", &serde_json::to_vec(&value).unwrap(), None),
            false,
        );
    }
    expect(invoke("probe", b"not-json", None), false);
    expect(invoke("probe", b"\xff", None), false);
}
