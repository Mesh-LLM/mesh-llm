//! Attempt-local linker evidence after target-cache restoration.
use super::{Node, assert_mutation_rejected, document, input, job, steps, support, text};
use std::{fs, process::Command};

fn reset_precedes_builds(doc: &Node) -> bool {
    let sequence = steps(job(doc, "windows_host"));
    let position = |predicate: &dyn Fn(&Node) -> bool| sequence.iter().position(predicate);
    let restore =
        position(&|s| text(s, "uses").is_some_and(|v| v.starts_with("Swatinem/rust-cache@")));
    let reset = position(&|s| text(s, "id") == Some("linker_traces"));
    let skippy = position(&|s| text(s, "name") == Some("Prepare standalone Skippy CLI"));
    let mesh = position(&|s| text(s, "name") == Some("Prepare immutable Windows host"));
    let upload = position(&|s| text(s, "name") == Some("Upload Windows compilation timings"));
    let (Some(restore), Some(reset), Some(skippy), Some(mesh), Some(upload)) =
        (restore, reset, skippy, mesh, upload)
    else {
        return false;
    };
    restore < reset
        && reset < skippy
        && reset < mesh
        && mesh < upload
        && input(&sequence[restore], "workspaces") == Some(". -> target")
        && input(&sequence[upload], "path").is_some_and(|paths| {
            paths.trim() == "target/cargo-timings/\ntarget/**/*.time-trace.json"
        })
        && text(&sequence[reset], "run") == Some("just ci-reset-linker-traces")
        && text(&sequence[reset], "if").is_none()
        && text(&sequence[upload], "if")
            == Some("${{ !cancelled() && steps.linker_traces.outcome == 'success' }}")
}

#[test]
fn windows_timing_upload_requires_reset_after_restore_and_before_both_builds() {
    let file = "ci-windows-host-slice.yml";
    assert!(reset_precedes_builds(&document(file)));
    assert_mutation_rejected(
        file,
        "run: just ci-reset-linker-traces",
        "run: echo skipped",
        reset_precedes_builds,
    );
    assert_mutation_rejected(
        file,
        "if: ${{ !cancelled() && steps.linker_traces.outcome == 'success' }}",
        "if: ${{ !cancelled() }}",
        reset_precedes_builds,
    );
}

#[test]
fn actual_trace_reset_cli_preserves_compiler_outputs_and_new_attempt_traces() {
    let fixture = support::Fixture::new();
    fs::create_dir_all(fixture.path().join("tools/xtask")).unwrap();
    fs::write(
        fixture.path().join("Cargo.toml"),
        "[workspace]\nmembers = []\n",
    )
    .unwrap();
    fs::write(
        fixture.path().join("tools/xtask/Cargo.toml"),
        "[package]\nname = \"xtask\"\nversion = \"0.0.0\"\nedition = \"2024\"\n",
    )
    .unwrap();
    let directory = fixture.path().join("target/release/build/dependency/out");
    fs::create_dir_all(&directory).unwrap();
    let old = directory.join("old.time-trace.json");
    fs::write(&old, "cached trace").unwrap();
    let binary = directory.join("build-script.exe");
    fs::write(&binary, "compiler output").unwrap();
    let mut command = Command::new(env!("CARGO_BIN_EXE_xtask"));
    command.args([
        "--repo-root",
        fixture.path().to_str().unwrap(),
        "ci-ops",
        "reset-linker-traces",
    ]);
    let result = fixture.run(command);
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    assert_eq!(
        String::from_utf8(result.stdout).unwrap(),
        "cleared_linker_traces=1\n"
    );
    assert!(!old.exists());
    assert_eq!(fs::read_to_string(binary).unwrap(), "compiler output");
    let fresh = directory.join("new.time-trace.json");
    fs::write(&fresh, "current attempt").unwrap();
    assert_eq!(fs::read_to_string(fresh).unwrap(), "current attempt");
}
