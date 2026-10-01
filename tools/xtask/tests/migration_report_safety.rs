use std::fs;
use std::path::PathBuf;
use std::process::{Command, Output, Stdio};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

static SEQUENCE: AtomicU64 = AtomicU64::new(0);

struct Scratch(PathBuf);

impl Scratch {
    fn new() -> Self {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let sequence = SEQUENCE.fetch_add(1, Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!(
            "xtask-report-safety-{}-{nanos}-{sequence}",
            std::process::id()
        ));
        fs::create_dir(&path).unwrap();
        Self(path)
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        fs::remove_dir_all(&self.0).expect("remove owned report scratch");
    }
}

fn run(body: &str, mode: &str) -> Output {
    let scratch = Scratch::new();
    let report = scratch.0.join("report.json");
    fs::write(&report, body).unwrap();
    let stdout = scratch.0.join("stdout");
    let stderr = scratch.0.join("stderr");
    let mut child = Command::new(env!("CARGO_BIN_EXE_xtask"))
        .args(["automation", "rewriter-report", "--report"])
        .arg(report)
        .args(["--mode", mode])
        .stdin(Stdio::null())
        .stdout(fs::File::create(&stdout).unwrap())
        .stderr(fs::File::create(&stderr).unwrap())
        .spawn()
        .unwrap();
    let deadline = Instant::now() + Duration::from_secs(5);
    let status = loop {
        match child.try_wait() {
            Ok(Some(status)) => break status,
            Ok(None) if Instant::now() < deadline => {
                std::thread::sleep(Duration::from_millis(10));
            }
            other => {
                child.kill().expect("kill bounded report child");
                child.wait().expect("reap bounded report child");
                panic!("report child failed to finish: {other:?}");
            }
        }
    };
    Output {
        status,
        stdout: fs::read(stdout).unwrap(),
        stderr: fs::read(stderr).unwrap(),
    }
}

fn report(builders: &str, summary: &str) -> String {
    format!(
        r#"{{"schema_version":1,"llama_cpp_commit":"c","generator_version":"g","builders":{builders},"summary":{summary}}}"#
    )
}

fn assert_output(output: Output, expected: (i32, &str, &str)) {
    assert_eq!(output.status.code(), Some(expected.0), "{output:?}");
    assert_eq!(output.stdout, expected.1.as_bytes());
    assert_eq!(output.stderr, expected.2.as_bytes());
}

#[test]
fn omitted_file_uses_distinct_index_identities() {
    let body = report(
        r#"[{"verdict":"already_transformed"},{"verdict":"already_transformed"}]"#,
        r#"{"already_transformed":2}"#,
    );
    assert_output(
        run(&body, "validate"),
        (0, "ok: report valid (validate); 2 builders checked\n", ""),
    );
}

#[test]
fn explicit_null_file_is_a_typed_contract_violation() {
    let body = report(
        r#"[{"verdict":"already_transformed"},{"file":null,"verdict":"already_transformed"}]"#,
        r#"{"already_transformed":2}"#,
    );
    assert_output(
        run(&body, "validate"),
        (1, "", "fail: builders[1].file: expected string\n"),
    );
}

#[test]
fn omitted_file_collides_with_its_literal_index_label() {
    for (builders, label) in [
        (
            r#"[{"verdict":"already_transformed"},{"file":"<builders[0]>","verdict":"already_transformed"}]"#,
            0,
        ),
        (
            r#"[{"file":"<builders[1]>","verdict":"already_transformed"},{"verdict":"already_transformed"}]"#,
            1,
        ),
    ] {
        let body = report(builders, r#"{"already_transformed":2}"#);
        let stderr = format!("fail: duplicate builder record for <builders[{label}]>::\n");
        assert_output(run(&body, "validate"), (1, "", &stderr));
    }
}

#[test]
fn summary_overflow_rejects_without_panicking() {
    for (summary, expected) in [
        (
            format!(
                r#"{{"transformable":{},"already_transformed":1,"supported_auxiliary":-{}}}"#,
                i128::MAX,
                i128::MAX
            ),
            "fail: summary.transformable: expected integer in 0..=9223372036854775807\nfail: summary.supported_auxiliary: expected integer in 0..=9223372036854775807\n",
        ),
        (
            format!(
                r#"{{"transformable":{},"already_transformed":1}}"#,
                i128::MAX
            ),
            "fail: summary.transformable: expected integer in 0..=9223372036854775807\n",
        ),
        (
            format!(
                r#"{{"transformable":{},"already_transformed":true}}"#,
                i128::MAX
            ),
            "fail: summary.transformable: expected integer in 0..=9223372036854775807\nfail: summary.already_transformed: expected integer in 0..=9223372036854775807\n",
        ),
        (
            format!(
                r#"{{"transformable":{},"already_transformed":-1}}"#,
                i128::MIN
            ),
            "fail: summary.transformable: expected integer in 0..=9223372036854775807\nfail: summary.already_transformed: expected integer in 0..=9223372036854775807\n",
        ),
    ] {
        let body = report(
            r#"[{"file":"f","verdict":"already_transformed"}]"#,
            &summary,
        );
        assert_output(run(&body, "validate"), (1, "", expected));
    }
}

#[test]
fn idempotence_still_ignores_summary_overflow() {
    let body = format!(
        r#"{{"builders":[{{"verdict":"already_transformed","edits":[1]}}],"summary":{{"transformable":{},"already_transformed":1}}}}"#,
        i128::MAX
    );
    assert_output(
        run(&body, "idempotence"),
        (
            0,
            "ok: report valid (idempotence); 1 builders checked\n",
            "",
        ),
    );
}

#[test]
fn deep_json_rejects_before_decoding() {
    for depth in [256, 6000, 9998, 10000] {
        for (open, close) in [("[", "]"), (r#"{"nested":"#, "}")] {
            let body = format!(
                r#"{{"builders":[],"ignored":{}0{}}}"#,
                open.repeat(depth),
                close.repeat(depth)
            );
            for mode in ["validate", "idempotence"] {
                assert_output(
                    run(&body, mode),
                    (
                        2,
                        "",
                        "error: cannot load report: report JSON nesting exceeds safety limit of 256 containers\n",
                    ),
                );
            }
        }
    }
}

#[test]
fn nesting_boundary_decodes_and_drops_safely() {
    for (open, close) in [("[", "]"), (r#"{"nested":"#, "}")] {
        let body = format!(
            r#"{{"builders":[],"ignored":{}0{}}}"#,
            open.repeat(255),
            close.repeat(255)
        );
        assert_output(
            run(&body, "idempotence"),
            (
                0,
                "ok: report valid (idempotence); 0 builders checked\n",
                "",
            ),
        );
        let malformed = format!("{body} trailing");
        let output = run(&malformed, "idempotence");
        assert_eq!(output.status.code(), Some(2), "{output:?}");
        assert!(
            output
                .stderr
                .starts_with(b"error: cannot load report: trailing characters at line 1 column ")
        );
    }
}

#[test]
fn nesting_guard_ignores_delimiters_inside_escaped_strings() {
    let ignored = serde_json::to_string(&format!("\\\"{}\"\\", "[{]}".repeat(10000))).unwrap();
    let body = format!(r#"{{"builders":[],"ignored":{ignored}}}"#);
    assert_output(
        run(&body, "idempotence"),
        (
            0,
            "ok: report valid (idempotence); 0 builders checked\n",
            "",
        ),
    );
}
