//! Production ownership must select qualification and its mandatory host row.
use serde_json::{Value, json};
use std::io::Write as _;
use std::process::{Command, Stdio};

fn plan(path: &str) -> Value {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let mut child = Command::new(env!("CARGO_BIN_EXE_xtask"))
        .current_dir(root)
        .args(["ci", "plan"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    let request = json!({
        "profile": "pr-ready", "event_name": "pull_request",
        "source_sha": super::support::SOURCE_SHA,
        "base_sha": "b".repeat(40), "changed_files": [path],
        "affected_crates": [],
    });
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
        super::support::stderr(&output)
    );
    serde_json::from_slice(&output.stdout).unwrap()
}

#[test]
fn qualification_and_composition_changes_select_every_core_row() {
    let expected = [
        "linux-cpu",
        "linux-cuda",
        "linux-rocm",
        "linux-vulkan",
        "macos-metal",
        "windows-cpu",
        "windows-cuda",
        "windows-rocm",
        "windows-vulkan",
    ];
    for path in [
        "tools/xtask/src/qualification/contract.rs",
        "tools/xtask/src/product/compose.rs",
        ".github/workflows/ci-skippy-product-slice.yml",
    ] {
        let plan = plan(path);
        assert!(
            plan["domains"]
                .as_array()
                .unwrap()
                .contains(&json!("ci-control"))
        );
        let rows: Vec<_> = plan["matrices"]["runtime_products"]
            .as_array()
            .unwrap()
            .iter()
            .map(|row| row["id"].as_str().unwrap())
            .collect();
        assert_eq!(rows, expected, "{path}");
    }
}

#[test]
fn backend_plans_include_the_cpu_receipt_required_by_each_host() {
    for path in [
        "skippy/scripts/detect-cuda-arch.sh",
        "skippy/scripts/build-linux.sh",
    ] {
        let plan = plan(path);
        let rows = plan["matrices"]["runtime_products"].as_array().unwrap();
        for platform in ["linux", "windows"] {
            if rows.iter().any(|row| row["platform"] == platform) {
                assert!(
                    rows.iter()
                        .any(|row| row["id"] == format!("{platform}-cpu")),
                    "{path}"
                );
            }
        }
    }
}
