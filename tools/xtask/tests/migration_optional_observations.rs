//! Real native projections used by optional research callers, without starting their servers.
use std::io::Write;
use std::process::{Command, Stdio};
fn native(args: &[&str], input: &[u8]) -> std::process::Output {
    let mut child = Command::new(env!("CARGO_BIN_EXE_xtask"))
        .args(args)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    child.stdin.take().unwrap().write_all(input).unwrap();
    child.wait_with_output().unwrap()
}
#[test]
fn actual_model_count_and_latency_input_are_typed_and_bounded() {
    for (input, expected) in [
        (br#"{"data":[]}"#.as_slice(), "0\n"),
        (
            br#"{"data":[{"id":"first"},{"id":"second"}],"object":"list"}"#.as_slice(),
            "2\n",
        ),
    ] {
        let output = native(&["automation", "smoke-observation", "model-count"], input);
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        assert_eq!(output.stdout, expected.as_bytes());
    }
    for input in [
        b"[]".as_slice(),
        br#"{"data":null}"#,
        br#"{"data":[{}]}"#,
        b"{",
    ] {
        let output = native(&["automation", "smoke-observation", "model-count"], input);
        assert!(!output.status.success());
        assert!(output.stdout.is_empty());
    }
    let output = native(
        &["automation", "smoke-observation", "model-count"],
        &vec![b' '; 1024 * 1024 + 1],
    );
    assert!(!output.status.success());
    assert!(output.stdout.is_empty());
    let output = native(
        &["automation", "wan-observation", "latency-summary"],
        br#"{"ttft_ms":10,"total_ms":22,"tok_s":4}"#,
    );
    assert!(output.status.success());
    assert_eq!(output.stdout, b"10ms\t22ms\t4.0\n");
    assert!(
        !native(&["automation", "wan-observation", "tensor-split", "0"], b"")
            .status
            .success()
    );
    assert_eq!(
        native(&["automation", "wan-observation", "tensor-split", "3"], b"").stdout,
        b"0.3333,0.3333,0.3334\n"
    );
}
#[cfg(unix)]
fn shell(script: &str, input: &str) -> std::process::Output {
    let child = Command::new("bash")
        .args(["-c", script])
        .env("MESH_LLM_AUTOMATION_BIN", env!("CARGO_BIN_EXE_xtask"))
        .env("FIXTURE_INPUT", input)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    child.wait_with_output().unwrap()
}
#[cfg(unix)]
#[test]
fn actual_research_caller_fragments_delegate_and_preserve_fallbacks() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let ab = std::fs::read_to_string(root.join("mesh/evals/ab-test.sh")).unwrap();
    let start = ab.find("# Frozen automation selection begins.").unwrap();
    let end = ab.find("# Frozen automation selection ends.").unwrap()
        + "# Frozen automation selection ends.".len();
    let selection = &ab[start..end];
    let lines = ab
        .lines()
        .filter(|l| l.contains("model_count=$(curl"))
        .collect::<Vec<_>>();
    assert_eq!(lines.len(), 2);
    for line in lines {
        let script = format!(
            "set -o pipefail\nROOT=/\n{selection}\ncurl() {{ printf '%s' \"$FIXTURE_INPUT\"; }}\n{line}\nprintf '%s\\n' \"$model_count\"\n"
        );
        assert_eq!(
            shell(&script, r#"{"data":[{"id":"a"},{"id":"b"}]}"#).stdout,
            b"2\n"
        );
        assert_eq!(shell(&script, "invalid").stdout, b"0\n");
    }
    for file in ["bench.sh", "bench-b2b.sh"] {
        let source =
            std::fs::read_to_string(root.join("skippy/evals/latency-benchmarking").join(file))
                .unwrap();
        assert!(source.contains("python3 \"$SCRIPT_DIR/latency-proxy.py\""));
        let split = source
            .lines()
            .find(|l| l.contains("split=$(\"${automation[@]}\""))
            .unwrap();
        let script = format!(
            "set -euo pipefail\nROOT=/\n{selection}\nnodes=3\n{split}\nprintf '%s\\n' \"$split\"\n"
        );
        assert_eq!(shell(&script, "").stdout, b"0.3333,0.3333,0.3334\n");
        assert!(
            !shell(&script.replace("nodes=3", "nodes=0"), "")
                .status
                .success()
        );
        if file == "bench-b2b.sh" {
            assert!(source.contains("python3 \"$SCRIPT_DIR/measure.py\""));
            let summary = source
                .lines()
                .find(|l| l.trim_start().starts_with("summary=$(printf"))
                .unwrap();
            let read = source
                .lines()
                .find(|l| l.contains("read -r ttft total tps"))
                .unwrap();
            let script = format!(
                "set -euo pipefail\nROOT=/\n{selection}\nresult=$FIXTURE_INPUT\n{summary}\n{read}\nprintf '%s|%s|%s\\n' \"$ttft\" \"$total\" \"$tps\"\n"
            );
            assert_eq!(
                shell(&script, r#"{"ttft_ms":10,"tok_s":4}"#).stdout,
                b"10ms|err|4.0\n"
            );
            assert_eq!(shell(&script, "invalid").stdout, b"err|err|err\n");
        }
    }
}
