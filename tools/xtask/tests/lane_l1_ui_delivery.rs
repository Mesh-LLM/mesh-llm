#![cfg(unix)]

use std::error::Error;
use std::fs;
use std::process::Command;

type TestResult = Result<(), Box<dyn Error>>;

#[test]
fn restore_release_ui_verifies_when_only_the_prepared_tool_is_available() -> TestResult {
    restore(false)
}

#[test]
fn restore_release_ui_rejects_when_a_downloaded_asset_changed() -> TestResult {
    restore(true)
}

fn restore(tampered: bool) -> TestResult {
    let scratch = tempfile::tempdir()?;
    let dist = scratch.path().join("console/dist");
    fs::create_dir_all(dist.join("assets"))?;
    fs::write(dist.join("assets/app.js"), "console.log('release');")?;
    fs::write(
        dist.join("index.html"),
        r#"<script type="module" src="/assets/app.js"></script>"#,
    )?;
    let source = "a".repeat(40);
    let tool = env!("CARGO_BIN_EXE_xtask");
    let stamp = Command::new(tool)
        .args(["prepared-input", "ui-distribution", "stamp", "--dist"])
        .arg(&dist)
        .args(["--source-sha", &source, "--release-tag", "v1.2.3"])
        .output()?;
    assert!(stamp.status.success(), "{:?}", stamp);
    if tampered {
        fs::write(dist.join("assets/app.js"), "changed")?;
    }
    let action = include_str!("../../../.github/actions/restore-release-ui/action.yml");
    let body = action
        .split_once("      run: |\n")
        .ok_or("missing run step")?
        .1;
    let script = body
        .lines()
        .map(|line| line.strip_prefix("        ").unwrap_or(line))
        .collect::<Vec<_>>()
        .join("\n");
    let output = Command::new("/bin/bash")
        .current_dir(scratch.path())
        .env("PATH", "")
        .env("MESH_LLM_AUTOMATION_BIN", tool)
        .env("UI_DIR", "console")
        .env("UI_SOURCE_SHA", source)
        .env("UI_RELEASE_TAG", "v1.2.3")
        .args(["-c", &script])
        .output()?;
    assert_eq!(output.status.success(), !tampered, "{:?}", output);
    Ok(())
}
