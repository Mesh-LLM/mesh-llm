//! `mesh-llm stop` and client processes: issue #2352, papercut 1.
//!
//! A plain `client` never registers a runtime instance, so `stop` cannot see
//! it. Its help says "Stop running mesh-llm processes." — while a client
//! keeps serving, a bare "Nothing running." is false. The command's output
//! must say what it does not cover.

use std::process::Command;

#[test]
fn stopping_nothing_discloses_the_clients_stop_does_not_cover() {
    let root = tempfile::tempdir().expect("temporary runtime root");
    let output = Command::new(env!("CARGO_BIN_EXE_mesh-llm"))
        .env("MESH_LLM_RUNTIME_ROOT", root.path())
        .args(["stop"])
        .output()
        .expect("run mesh-llm stop");

    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        output.status.success(),
        "stopping nothing exits 0: {stderr}"
    );
    assert!(
        stderr.contains("Nothing running."),
        "an empty runtime root stops nothing: {stderr}"
    );
    assert!(
        stderr.to_lowercase().contains("client"),
        "stop's output must say that client processes are not covered: {stderr}"
    );
}
