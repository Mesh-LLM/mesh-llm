use super::*;
use serde_json::json;

fn emit(progress: &mut Progress, value: Value, output: &mut Vec<u8>) {
    progress
        .line(&serde_json::to_vec(&value).unwrap(), output)
        .unwrap();
}

#[test]
fn narration_and_tool_lifecycle_replace_verbose_success_output() {
    let mut progress = Progress::new(false);
    let mut output = Vec::new();
    emit(
        &mut progress,
        json!({"type":"message","message":{"role":"assistant","content":[
            {"type":"text","text":"Checking the queue because preparation failed."},
            {"type":"thinking","thinking":"private reasoning"},
            {"type":"toolRequest","id":"call-1","toolCall":{"status":"success","value":{"name":"developer__shell","arguments":{"command":"check-family --all"}}}}
        ]}}),
        &mut output,
    );
    progress.heartbeat(&mut output).unwrap();
    emit(
        &mut progress,
        json!({"type":"message","message":{"role":"user","content":[
            {"type":"text","text":"not assistant narration"},
            {"type":"toolResponse","id":"call-1","toolResult":{"status":"success","value":{"isError":false,"content":[{"type":"text","text":"verbose compiler output"}]}}}
        ]}}),
        &mut output,
    );
    let text = String::from_utf8(output).unwrap();
    assert!(text.contains("Checking the queue because preparation failed."));
    assert!(text.contains("goose running: developer__shell: check-family --all"));
    assert!(text.contains("goose still running:"));
    assert!(text.contains("goose returned:"));
    assert!(!text.contains("verbose compiler") && !text.contains("private reasoning"));
    assert!(!text.contains("not assistant narration"));
    assert!(progress.tools.is_empty());
}

#[test]
fn tool_and_agent_errors_remain_visible_and_never_execute_workflow_commands() {
    let mut progress = Progress::new(false);
    let mut output = Vec::new();
    for result in [
        json!({"status":"error","error":"transport failed"}),
        json!({"status":"success","value":{"isError":true,"content":[{"type":"text","text":"test failed\n::error::injected"}]}}),
    ] {
        emit(
            &mut progress,
            json!({"type":"message","message":{"role":"user","content":[{"type":"toolResponse","id":"x","toolResult":result}]}}),
            &mut output,
        );
    }
    emit(
        &mut progress,
        json!({"type":"error","error":"provider unavailable"}),
        &mut output,
    );
    progress
        .line(b"::error::plain diagnostic\n", &mut output)
        .unwrap();
    let text = String::from_utf8(output).unwrap();
    assert!(
        text.contains("transport failed")
            && text.contains("test failed")
            && text.contains("provider unavailable")
    );
    assert!(!text.lines().any(|line| line.starts_with("::")));
}

#[test]
fn oversized_event_is_drained_without_losing_following_events() {
    let mut bytes = vec![b'x'; MAX_EVENT + 10];
    bytes.extend_from_slice(b"\n{\"type\":\"complete\"}");
    let (send, receive) = mpsc::sync_channel(4);
    events::read_events(io::Cursor::new(bytes), send, &AtomicBool::new(false)).unwrap();
    let lines: Vec<_> = receive.into_iter().collect();
    assert_eq!(lines.len(), 2);
    assert!(String::from_utf8_lossy(&lines[0]).contains("display limit"));
    assert_eq!(lines[1], b"{\"type\":\"complete\"}");
}

#[test]
fn command_log_can_be_observed_before_tool_returns_and_completion_is_not_assumed() {
    let directory = tempfile::tempdir().unwrap();
    let command = directory.path().join("check-family.abcd");
    std::fs::create_dir(&command).unwrap();
    let log = command.join("output.log");
    std::fs::write(&log, "started_epoch: 1\nchecking shard 37/84\n").unwrap();
    let mut commands = commands::Commands::default();
    let mut output = Vec::new();
    commands.report(directory.path(), &mut output).unwrap();
    let text = String::from_utf8(output).unwrap();
    assert!(text.contains("running") && text.contains("checking shard 37/84"));
    assert!(text.contains("output.log"));
    std::fs::OpenOptions::new()
        .append(true)
        .open(&log)
        .unwrap()
        .write_all(b"failed shard 38\nexit_status: 73\n")
        .unwrap();
    std::fs::write(
        command.join("completion"),
        "73
",
    )
    .unwrap();
    let mut output = Vec::new();
    commands.report(directory.path(), &mut output).unwrap();
    commands.report(directory.path(), &mut output).unwrap();
    let text = String::from_utf8(output).unwrap();
    assert_eq!(text.matches("canary command:").count(), 1);
    assert!(text.contains("exit_status: 73"));
}

#[cfg(unix)]
#[test]
fn command_monitor_does_not_follow_foreign_log_symlinks() {
    let directory = tempfile::tempdir().unwrap();
    let external = tempfile::tempdir().unwrap();
    std::fs::write(external.path().join("output.log"), "foreign data").unwrap();
    std::os::unix::fs::symlink(external.path(), directory.path().join("foreign")).unwrap();
    let command = directory.path().join("local");
    std::fs::create_dir(&command).unwrap();
    std::os::unix::fs::symlink(
        external.path().join("output.log"),
        command.join("output.log"),
    )
    .unwrap();
    let mut output = Vec::new();
    commands::Commands::default()
        .report(directory.path(), &mut output)
        .unwrap();
    assert!(output.is_empty());
}

#[test]
fn github_details_can_be_expanded_without_executing_model_workflow_commands() {
    let mut progress = Progress::new(true);
    let mut output = Vec::new();
    emit(
        &mut progress,
        json!({"type":"message","message":{"role":"user","content":[
            {"type":"toolResponse","id":"call","toolResult":{"status":"success","value":{"isError":false,"content":[{"type":"text","text":"full diagnostic ::error::injected"}]}}}
        ]}}),
        &mut output,
    );
    let text = String::from_utf8(output).unwrap();
    assert!(text.contains("::group::Goose tool result"));
    assert!(text.contains("full diagnostic : :error: :injected"));
    assert!(text.contains("::endgroup::"));
    assert!(!text.contains("::error::"));
    assert!(!compact("::::error::::injected", 100).contains("::"));
    assert_eq!(
        compact("prefix ::set-env name=x::y", 100),
        "prefix : :set-env name=x: :y"
    );
}

#[test]
fn oversized_tool_response_clears_pending_lifecycle() {
    let mut progress = Progress::new(false);
    let mut output = Vec::new();
    emit(
        &mut progress,
        json!({"type":"message","message":{"role":"assistant","content":[
            {"type":"toolRequest","id":"large","toolCall":{"status":"success","value":{"name":"shell"}}}
        ]}}),
        &mut output,
    );
    let bytes = serde_json::to_vec(&json!({"type":"message","message":{"role":"user","content":[
        {"type":"toolResponse","id":"large","toolResult":{"status":"success","value":{"content":[{"type":"text","text":"x".repeat(MAX_EVENT + 1)}]}}}
    ]}})).unwrap();
    let (send, receive) = mpsc::sync_channel(4);
    events::read_events(io::Cursor::new(bytes), send, &AtomicBool::new(false)).unwrap();
    for line in receive {
        progress.line(&line, &mut output).unwrap();
    }
    assert!(progress.tools.is_empty());
    let text = String::from_utf8(output).unwrap();
    assert!(text.contains("goose returned: shell"));
    assert!(text.contains("lifecycle retained"));
}

#[test]
fn malformed_tool_call_does_not_become_pending() {
    let mut progress = Progress::new(false);
    let mut output = Vec::new();
    emit(
        &mut progress,
        json!({"type":"message","message":{"role":"assistant","content":[
            {"type":"toolRequest","id":"bad","toolCall":{"status":"error","error":"invalid arguments"}}
        ]}}),
        &mut output,
    );
    assert!(progress.tools.is_empty());
    assert!(
        String::from_utf8(output)
            .unwrap()
            .contains("invalid arguments")
    );
}

#[test]
fn command_output_cannot_signal_wrapper_completion() {
    let directory = tempfile::tempdir().unwrap();
    let command = directory.path().join("running");
    std::fs::create_dir(&command).unwrap();
    std::fs::write(command.join("output.log"), "exit_status: 0\n").unwrap();
    let mut commands = commands::Commands::default();
    let mut output = Vec::new();
    commands.report(directory.path(), &mut output).unwrap();
    std::fs::write(
        command.join("output.log"),
        "later output\nexit_status: 73\n",
    )
    .unwrap();
    std::fs::write(command.join("completion"), "73\n").unwrap();
    commands.report(directory.path(), &mut output).unwrap();
    let text = String::from_utf8(output).unwrap();
    assert_eq!(text.matches("canary command:").count(), 2);
    assert!(text.contains("running;"));
    assert!(text.contains("exit_status: 73") && text.contains("later output"));
}

#[test]
fn quiet_active_command_survives_sixteen_newer_completions() {
    let directory = tempfile::tempdir().unwrap();
    let active = directory.path().join("active");
    std::fs::create_dir(&active).unwrap();
    let log = std::fs::File::create(active.join("output.log")).unwrap();
    log.set_times(std::fs::FileTimes::new().set_modified(std::time::UNIX_EPOCH))
        .unwrap();
    for index in 0..16 {
        let command = directory.path().join(format!("complete-{index}"));
        std::fs::create_dir(&command).unwrap();
        std::fs::write(command.join("output.log"), "done\n").unwrap();
        std::fs::write(command.join("completion"), "0\n").unwrap();
    }
    let mut output = Vec::new();
    commands::Commands::default()
        .report(directory.path(), &mut output)
        .unwrap();
    assert!(
        String::from_utf8(output)
            .unwrap()
            .contains("active — running")
    );
}

#[test]
fn projection_failure_cancels_quiet_reader() {
    struct Quiet;
    impl Read for Quiet {
        fn read(&mut self, _: &mut [u8]) -> io::Result<usize> {
            Err(io::ErrorKind::WouldBlock.into())
        }
    }
    struct FailedOutput;
    impl Write for FailedOutput {
        fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
            Ok(bytes.len())
        }
        fn flush(&mut self) -> io::Result<()> {
            Err(io::ErrorKind::BrokenPipe.into())
        }
    }
    let directory = tempfile::tempdir().unwrap();
    // Without cancellation this returns only after the reader's pipe closes.
    assert!(project(Quiet, directory.path(), &mut FailedOutput).is_err());
}

#[test]
fn legacy_workflow_commands_are_neutralized_in_details_and_summaries() {
    assert!(!compact("###[error]fake ::error::fake", 100).contains("##["));
    let mut output = Vec::new();
    Progress::new(true)
        .details("trusted title", "##[error]fake", &mut output)
        .unwrap();
    assert!(!String::from_utf8(output).unwrap().contains("##["));
}
