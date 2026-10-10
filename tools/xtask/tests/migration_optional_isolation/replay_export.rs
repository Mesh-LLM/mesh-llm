use super::export_cases;
use super::export_process;
use super::support::{SHELL, TestResult};

#[test]
fn export_writes_json_and_appends_environment_fields() -> TestResult {
    let mut case = export_cases::complete_export();
    let input: serde_json::Value =
        serde_json::from_slice(case.input.as_deref().ok_or("missing input")?)?;
    case.initial
        .push(("github.env".to_owned(), b"PRIOR=kept\n".to_vec()));
    let actual = export_process::observe(&case)?;
    assert_eq!(actual.status, 0);
    let json = actual
        .files
        .get("params.json")
        .and_then(Option::as_ref)
        .ok_or("missing JSON")?;
    let document: serde_json::Value = serde_json::from_slice(json)?;
    assert_eq!(document, input["replay"]);
    assert!(json.ends_with(b"\n"));
    let environment = actual
        .files
        .get("github.env")
        .and_then(Option::as_ref)
        .ok_or("missing environment")?;
    assert_eq!(
        std::str::from_utf8(environment)?,
        concat!(
            "PRIOR=kept\n",
            "AGENTIC_REPLAY_MODE=all\n",
            "AGENTIC_REPLAY_SESSIONS_PER_CONCURRENCY=16\n",
            "AGENTIC_REPLAY_MINIMUM_WORKER_WAVES=2\n",
            "AGENTIC_REPLAY_MINIMUM_CONTEXT_TOKENS=131072\n",
            "AGENTIC_REPLAY_MINIMUM_SESSION_PROMPT_TOKENS=32768\n",
            "AGENTIC_REPLAY_MIN_ISL=32768\n",
            "AGENTIC_REPLAY_MAX_ISL=131072\n",
            "AGENTIC_REPLAY_MIN_TURNS=5\n",
            "AGENTIC_REPLAY_PASSES=2\n",
            "AGENTIC_REPLAY_WARMUP_TURNS=4\n",
            "AGENTIC_REPLAY_MAX_OUTPUT_TOKENS=2048\n",
            "AGENTIC_REPLAY_CONCURRENCY=1,2,4,8\n"
        )
    );
    assert_eq!(actual.stdout, SHELL.as_bytes());
    assert!(actual.stderr.is_empty());
    Ok(())
}
