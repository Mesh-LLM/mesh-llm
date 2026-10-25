use super::support::Fixture;
use std::{fs, process::Command};

pub(super) fn verify(fixture: &Fixture) -> Result<(), Box<dyn std::error::Error>> {
    let root = fixture.root.path();
    let runs = root.join("history-inputs");
    fs::create_dir(&runs)?;
    fs::rename(root.join("result"), runs.join("granite-3.1-2b"))?;
    let hardware = root.join("hardware.json");
    fs::write(
        &hardware,
        serde_json::to_vec(&serde_json::json!({
            "machine_model":"fixture", "chip":"fixture", "gpu_cores":1,
            "unified_memory_bytes":274877906944_u64, "os_version":"fixture"
        }))?,
    )?;
    let output = root.join("history.jsonl");
    let status = Command::new(env!("CARGO_BIN_EXE_xtask"))
        .args(["automation", "replay-matrix", "history", "--matrix"])
        .arg(root.join("matrix.json"))
        .arg("--replay-dir")
        .arg(&runs)
        .args([
            "--label",
            "main",
            "--source-sha",
            "0123456789abcdef0123456789abcdef01234567",
        ])
        .arg("--hardware")
        .arg(&hardware)
        .arg("--replay")
        .arg(&fixture.json)
        .arg("--output")
        .arg(&output)
        .arg("--gate")
        .output()?;
    assert!(
        status.status.success(),
        "{}",
        String::from_utf8_lossy(&status.stderr)
    );
    let rows = fs::read_to_string(output)?;
    let rows: Vec<serde_json::Value> = rows
        .lines()
        .map(serde_json::from_str)
        .collect::<Result<_, _>>()?;
    assert_eq!(rows.len(), 1);
    assert_eq!(rows[0]["schema_version"], 3);
    assert_eq!(rows[0]["complete"], true);
    assert_eq!(rows[0]["model"]["family"], "granite-3.1-2b");
    assert!(rows[0]["session_cohort_sha256"].is_array());
    Ok(())
}
