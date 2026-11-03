//! Drive the real planner's placement signal into the real selector action.
use super::support;
use serde_json::{Value, json};
use std::io::Write;
use std::process::{Command, Stdio};

fn plan(profile: &str, event: &str, base: &str, path: &str) -> Value {
    let request = json!({
        "profile": profile,
        "event_name": event,
        "source_sha": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        "base_sha": base,
        "changed_files": [path],
        "workspace_packages": [{"name": "mesh-llm", "path": "mesh/crates/mesh-llm"}]
    });
    let mut child = Command::new(env!("CARGO_BIN_EXE_xtask"))
        .current_dir(support::root())
        .args(["ci", "plan"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    child
        .stdin
        .take()
        .unwrap()
        .write_all(request.to_string().as_bytes())
        .unwrap();
    let output = child.wait_with_output().unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    serde_json::from_slice(&output.stdout).unwrap()
}

#[test]
fn exhaustive_coverage_and_provider_selection_remain_separate() {
    let base = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";
    let cases = [
        (
            "main",
            "push",
            base,
            "README.md",
            "true",
            "false",
            false,
            true,
        ),
        (
            "main",
            "push",
            base,
            "README.md",
            "false",
            "false",
            false,
            false,
        ),
        (
            "main",
            "push",
            base,
            "README.md",
            "true",
            "false",
            true,
            false,
        ),
        (
            "main",
            "push",
            base,
            ".github/workflows/main_linux.yml",
            "true",
            "false",
            false,
            false,
        ),
        (
            "main",
            "push",
            "",
            "__force_all__",
            "true",
            "false",
            false,
            false,
        ),
        (
            "manual-full",
            "workflow_dispatch",
            base,
            "README.md",
            "false",
            "true",
            false,
            true,
        ),
        (
            "manual-full",
            "workflow_dispatch",
            base,
            "README.md",
            "false",
            "false",
            false,
            false,
        ),
    ];
    for (profile, event, base, path, main_gate, manual_gate, override_hosted, depot) in cases {
        let plan = plan(profile, event, base, path);
        assert_eq!(plan["signals"]["runner_contract_required"], true);
        let force = override_hosted || plan["signals"]["force_hosted"].as_bool().unwrap();
        let fixture = support::Fixture::new();
        let output = fixture.selector(&[
            ("INPUT_EVENT_NAME", event),
            ("INPUT_ORIGINAL_EVENT_NAME", event),
            ("INPUT_REF", "refs/heads/main"),
            ("INPUT_DEPOT_MAIN_ENABLED", main_gate),
            ("INPUT_MANUAL_USE_DEPOT", manual_gate),
            ("INPUT_FORCE_HOSTED", if force { "true" } else { "false" }),
            ("FIXTURE_DATE", "2026-11-03"),
        ]);
        assert!(output.status.success(), "{profile} {path}: {output:?}");
        let selected = fixture.outputs();
        assert_eq!(
            selected["depot_enabled"],
            depot.to_string(),
            "{profile} {path}"
        );
        if depot {
            assert_eq!(selected["allow_depot_remote_cache"], "false");
            assert_eq!(selected["allow_native_github_cache"], "false");
        }
        fixture.0.close().unwrap();
    }
}
