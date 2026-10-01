use serde::Deserialize;
use sha2::{Digest, Sha256};
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};

struct Campaign {
    directory: PathBuf,
    _temporary: Option<tempfile::TempDir>,
}

struct ProducerCopy(PathBuf);
impl Drop for ProducerCopy {
    fn drop(&mut self) {
        fs::remove_dir_all(&self.0).expect("remove only owned producer copy");
    }
}

impl Campaign {
    fn new(name: &str) -> Self {
        let retained = std::env::var_os("REPORT_CONTRACT_EVIDENCE");
        let (directory, temporary) = match retained {
            Some(parent) => {
                let directory = PathBuf::from(parent).join(name);
                fs::create_dir(&directory).unwrap();
                (directory, None)
            }
            None => {
                let temporary = tempfile::Builder::new()
                    .prefix("report-contract-")
                    .tempdir()
                    .unwrap();
                let directory = temporary.path().join(name);
                fs::create_dir(&directory).unwrap();
                (directory, Some(temporary))
            }
        };
        Self {
            directory,
            _temporary: temporary,
        }
    }

    fn run(&self, name: &str, body: &[u8], args: &[&str]) -> Output {
        let case = self.directory.join(name);
        fs::create_dir(&case).unwrap();
        save(&case.join("input.json"), body);
        let output = Command::new("gtimeout")
            .args([
                "-k",
                "1",
                "5",
                env!("CARGO_BIN_EXE_xtask"),
                "automation",
                "rewriter-report",
                "--report",
            ])
            .arg(case.join("input.json"))
            .args(args)
            .output()
            .unwrap();
        save(&case.join("stdout"), &output.stdout);
        save(&case.join("stderr"), &output.stderr);
        save(
            &case.join("status"),
            format!("{:?}\n", output.status.code()).as_bytes(),
        );
        save(&case.join("argv.json"), &serde_json::to_vec(args).unwrap());
        assert!(
            matches!(output.status.code(), Some(0..=2)),
            "{name}: {output:?}"
        );
        output
    }

    fn legacy(&self, name: &str, args: &[&str]) -> Output {
        let case = self.directory.join(name);
        let output = Command::new("gtimeout")
            .args([
                "-k",
                "1",
                "5",
                "/usr/bin/python3",
                "scripts/skippy-rewriter-harness.py",
                "--report",
            ])
            .current_dir(Path::new(env!("CARGO_MANIFEST_DIR")).join("../.."))
            .env("PYTHONDONTWRITEBYTECODE", "1")
            .env("PYTHONHASHSEED", "0")
            .arg(case.join("input.json"))
            .args(args)
            .output()
            .unwrap();
        save(&case.join("legacy.stdout"), &output.stdout);
        save(&case.join("legacy.stderr"), &output.stderr);
        save(
            &case.join("legacy.status"),
            format!("{:?}\n", output.status.code()).as_bytes(),
        );
        assert!(
            matches!(output.status.code(), Some(0..=2)),
            "{name}: {output:?}"
        );
        output
    }
}

fn save(path: &Path, bytes: &[u8]) {
    OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)
        .unwrap()
        .write_all(bytes)
        .unwrap();
}

fn report(summary: &str) -> String {
    format!(
        r#"{{"schema_version":1,"llama_cpp_commit":"c","generator_version":"g","builders":[{{"file":"f","verdict":"already_transformed"}}],"summary":{summary}}}"#
    )
}

fn exact(output: &Output, expected: (i32, &str, &str)) {
    assert_eq!(output.status.code(), Some(expected.0), "{output:?}");
    assert_eq!(output.stdout, expected.1.as_bytes());
    assert_eq!(output.stderr, expected.2.as_bytes());
}

fn fixtures() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/migration/rewriter-report")
}

#[test]
fn approved_controls_reject_coercions_and_accept_ignored_numbers() {
    let campaign = Campaign::new("regression-controls");
    let controls = [
        (
            "boolean-counter",
            report(r#"{"already_transformed":true}"#),
            vec![],
            1,
        ),
        (
            "numeric-file",
            report(r#"{"already_transformed":1}"#).replace(r#""file":"f""#, r#""file":12"#),
            vec![],
            1,
        ),
        (
            "digits-4301",
            format!(r#"{{"ignored":{}}}"#, "9".repeat(4301)),
            vec!["--mode", "idempotence"],
            0,
        ),
        ("help-prefix", "{}".into(), vec!["--hel"], 2),
        ("unknown-help", "{}".into(), vec!["--unknown", "--help"], 0),
    ];
    let mut differences = Vec::new();
    for (name, body, args, expected) in controls {
        let output = campaign.run(name, body.as_bytes(), &args);
        if output.status.code() != Some(expected) {
            differences.push(format!(
                "{name}: expected {expected}, got {:?}",
                output.status.code()
            ));
        }
    }
    assert!(differences.is_empty(), "{}", differences.join("\n"));
}

#[test]
fn producer_counter_domain_has_exact_totals() {
    let campaign = Campaign::new("exact-arithmetic");
    for (name, summary, total) in [
        ("one", r#"{"already_transformed":1}"#, "1"),
        (
            "above-f64",
            r#"{"already_transformed":9007199254740993}"#,
            "9007199254740993",
        ),
        (
            "above-i64",
            r#"{"transformable":9223372036854775807,"already_transformed":1}"#,
            "9223372036854775808",
        ),
        (
            "all-max",
            r#"{"transformable":9223372036854775807,"already_transformed":9223372036854775807,"supported_auxiliary":9223372036854775807,"supported_whole_model":9223372036854775807,"unsupported_shape":9223372036854775807,"error":9223372036854775807}"#,
            "55340232221128654842",
        ),
    ] {
        let body = report(summary);
        let output = campaign.run(name, body.as_bytes(), &[]);
        if name == "one" {
            exact(
                &output,
                (0, "ok: report valid (validate); 1 builders checked\n", ""),
            );
        } else {
            exact(
                &output,
                (
                    1,
                    "",
                    &format!("fail: summary counts ({total}) != builder records (1)\n"),
                ),
            );
        }
    }
}

#[test]
fn nine_invalid_counter_tokens_reject_individually() {
    let campaign = Campaign::new("invalid-counters");
    for (index, token) in [
        "-1",
        "0.5",
        "1.0",
        "true",
        "null",
        r#""1""#,
        "[]",
        "{}",
        "9223372036854775808",
    ]
    .iter()
    .enumerate()
    {
        let body = report(&format!(r#"{{"already_transformed":{token}}}"#));
        let output = campaign.run(&format!("counter-{index}"), body.as_bytes(), &[]);
        exact(
            &output,
            (
                1,
                "",
                "fail: summary.already_transformed: expected integer in 0..=9223372036854775807\n",
            ),
        );
    }
}

#[test]
fn invalid_fields_keep_forwarded_gates() {
    let campaign = Campaign::new("field-gates");
    let output = campaign.run(
        "counter",
        report(r#"{"already_transformed":true}"#).as_bytes(),
        &[
            "--graph-verify-result",
            "fail",
            "--compile-result",
            "fail",
            "--patch-check",
            "fail",
            "--patch-drift-gate",
            "fail",
        ],
    );
    exact(
        &output,
        (
            1,
            "",
            "fail: summary.already_transformed: expected integer in 0..=9223372036854775807\nfail: generated family patch drifts from checked-in patch\nfail: transformed tree failed to compile\nfail: no-allocation graph verifier failed on transformed tree\n",
        ),
    );
}

#[test]
fn schema_and_identity_types_are_mode_specific() {
    let campaign = Campaign::new("schema-identity");
    let ready = report(r#"{"already_transformed":1}"#);
    for (index, token) in ["true", "1.0"].iter().enumerate() {
        let body = ready.replace(
            "\"schema_version\":1",
            &format!("\"schema_version\":{token}"),
        );
        exact(
            &campaign.run(&format!("schema-{index}-validate"), body.as_bytes(), &[]),
            (
                1,
                "",
                "fail: schema_version: expected unsigned integer schema version\n",
            ),
        );
        exact(
            &campaign.run(
                &format!("schema-{index}-idempotence"),
                body.as_bytes(),
                &["--mode", "idempotence"],
            ),
            (
                0,
                "ok: report valid (idempotence); 1 builders checked\n",
                "",
            ),
        );
    }
    for (index, token) in ["12", "null", "{}"].iter().enumerate() {
        let body = ready.replace(r#""file":"f""#, &format!(r#""file":{token}"#));
        exact(
            &campaign.run(&format!("identity-{index}"), body.as_bytes(), &[]),
            (1, "", "fail: builders[0].file: expected string\n"),
        );
    }
    let body = ready.replace(r#""file":"f""#, r#""file":"f","constructor":{}"#);
    exact(
        &campaign.run("constructor-object", body.as_bytes(), &[]),
        (1, "", "fail: builders[0].constructor: expected string\n"),
    );
    let body = ready.replace(r#"{"file":"f","verdict":"already_transformed"}"#, r#"{"file":"f","constructor":"one","verdict":"already_transformed"},{"file":"f","constructor":"two","verdict":"already_transformed"}"#).replace(r#""already_transformed":1"#, r#""already_transformed":2"#);
    exact(
        &campaign.run("same-file-constructors", body.as_bytes(), &[]),
        (0, "ok: report valid (validate); 2 builders checked\n", ""),
    );
}

#[test]
fn unicode_scalars_are_preserved_and_surrogates_reject() {
    let campaign = Campaign::new("unicode");
    let ready = report(r#"{"already_transformed":1}"#);
    let body = ready.replace(r#""file":"f""#, r#""file":"\ud800""#);
    let output = campaign.run("unpaired", body.as_bytes(), &[]);
    assert_eq!(output.status.code(), Some(2));
    assert!(output.stdout.is_empty());
    assert!(
        String::from_utf8(output.stderr).unwrap().starts_with(
            "error: cannot load report: unexpected end of hex escape at line 1 column "
        )
    );
    for (name, token) in [
        ("replacement", "\"\u{fffd}\""),
        ("supplementary", r#""\ud83d\ude00""#),
    ] {
        let body = ready.replace(r#""file":"f""#, &format!(r#""file":{token}"#));
        exact(
            &campaign.run(name, body.as_bytes(), &[]),
            (0, "ok: report valid (validate); 1 builders checked\n", ""),
        );
    }
}

#[test]
fn builders_container_and_record_types_reject_in_both_modes() {
    let campaign = Campaign::new("builders");
    let ready = report(r#"{"already_transformed":1}"#);
    for (name, builders, field) in [
        ("null", "null", "builders: expected array"),
        ("null-record", "[null]", "builders[0]: expected object"),
    ] {
        let body = ready.replace(
            r#"[{"file":"f","verdict":"already_transformed"}]"#,
            builders,
        );
        for mode in ["validate", "idempotence"] {
            let output = campaign.run(
                &format!("{name}-{mode}"),
                body.as_bytes(),
                &["--mode", mode],
            );
            let suffix = if name == "null" && mode == "validate" {
                "fail: missing non-empty 'builders' array\n"
            } else {
                ""
            };
            exact(&output, (1, "", &format!("fail: {field}\n{suffix}")));
        }
    }
    for (name, body) in [("absent", "{}"), ("empty", r#"{"builders":[]}"#)] {
        exact(
            &campaign.run(name, body.as_bytes(), &["--mode", "idempotence"]),
            (
                0,
                "ok: report valid (idempotence); 0 builders checked\n",
                "",
            ),
        );
    }
}

#[test]
fn ignored_large_integers_do_not_construct_numbers() {
    let campaign = Campaign::new("ignored-integers");
    for digits in [4300, 4301] {
        let body = format!(r#"{{"builders":[],"ignored":{}}}"#, "9".repeat(digits));
        exact(
            &campaign.run(
                &format!("digits-{digits}"),
                body.as_bytes(),
                &["--mode", "idempotence"],
            ),
            (
                0,
                "ok: report valid (idempotence); 0 builders checked\n",
                "",
            ),
        );
    }
}

#[test]
fn boundary_admission_applies_before_typed_decode() {
    let campaign = Campaign::new("depth-boundary");
    for (shape, open, close) in [("array", "[", "]"), ("object", r#"{"nested":"#, "}")] {
        for depth in [255, 256] {
            let body = format!(
                r#"{{"ignored":{}0{}}}"#,
                open.repeat(depth),
                close.repeat(depth)
            );
            for mode in ["validate", "idempotence"] {
                let output = campaign.run(
                    &format!("{shape}-{depth}-{mode}"),
                    body.as_bytes(),
                    &["--mode", mode],
                );
                if depth == 256 {
                    exact(
                        &output,
                        (
                            2,
                            "",
                            "error: cannot load report: report JSON nesting exceeds safety limit of 256 containers\n",
                        ),
                    );
                } else if mode == "idempotence" {
                    exact(
                        &output,
                        (
                            0,
                            "ok: report valid (idempotence); 0 builders checked\n",
                            "",
                        ),
                    );
                } else {
                    exact(
                        &output,
                        (
                            1,
                            "",
                            "fail: schema_version None != supported 1\nfail: missing required string field 'llama_cpp_commit'\nfail: missing required string field 'generator_version'\nfail: missing non-empty 'builders' array\n",
                        ),
                    );
                }
            }
        }
    }
    let body = format!(r#"{{"ignored":{}"#, "[".repeat(6000));
    exact(
        &campaign.run("malformed-6000", body.as_bytes(), &[]),
        (
            2,
            "",
            "error: cannot load report: report JSON nesting exceeds safety limit of 256 containers\n",
        ),
    );
}

#[test]
fn malformed_and_duplicate_members_reject_deterministically() {
    let campaign = Campaign::new("malformed");
    for (name, body, code, stderr) in [
        (
            "duplicate",
            &b"{\"builders\":[],\"builders\":[]}"[..],
            1,
            "fail: builders: duplicate consumed member\n",
        ),
        (
            "trailing-comma",
            &b"{\"builders\":[],}"[..],
            2,
            "error: cannot load report: trailing comma at line 1 column 16\n",
        ),
        (
            "nonobject",
            &b"[]"[..],
            2,
            "error: cannot load report: report must be a JSON object\n",
        ),
        (
            "invalid-utf8",
            &b"{\"ignored\":\"\xff\"}"[..],
            2,
            "error: cannot load report: invalid UTF-8 at byte 12\n",
        ),
        (
            "nan",
            &b"{\"ignored\":NaN}"[..],
            2,
            "error: cannot load report: expected value at line 1 column 12\n",
        ),
    ] {
        exact(
            &campaign.run(name, body, &["--mode", "idempotence"]),
            (code, "", stderr),
        );
    }
}

#[derive(Deserialize)]
struct Observation {
    code: i32,
    stdout: String,
    stderr: String,
}
#[derive(Deserialize)]
struct Receipt {
    name: String,
    report: PathBuf,
    args: Vec<String>,
    python: Observation,
    rust: Observation,
}

fn matches_observation(output: &Output, expected: &Observation) {
    exact(output, (expected.code, &expected.stdout, &expected.stderr));
}

#[test]
fn retained_gate72_and_reverse_order_preserve_all_streams() {
    let campaign = Campaign::new("gate72");
    let frozen: serde_json::Value =
        serde_json::from_slice(&fs::read(fixtures().join("legacy-cli.json")).unwrap()).unwrap();
    let retained: Vec<Receipt> = std::env::var_os("REPORT_RETAINED_GATES")
        .map_or_else(Vec::new, |path| {
            serde_json::from_slice(&fs::read(path).unwrap()).unwrap()
        });
    let mut count = 0;
    for fixture in ["ready", "invalid"] {
        let body = fs::read(fixtures().join(format!("{fixture}.json"))).unwrap();
        for mode in ["validate", "idempotence"] {
            for gate in ["warn", "fail"] {
                for compile in ["pass", "fail", "skipped"] {
                    for graph in ["pass", "fail", "skipped"] {
                        let name = format!("gates-{fixture}-{mode}-{gate}-{compile}-{graph}");
                        let args = [
                            "--mode",
                            mode,
                            "--patch-check",
                            "fail",
                            "--patch-drift-gate",
                            gate,
                            "--compile-result",
                            compile,
                            "--graph-verify-result",
                            graph,
                        ];
                        let output = campaign.run(&name, &body, &args);
                        let mut stderr = if fixture == "invalid" {
                            if mode == "validate" {
                                frozen["cases"][8]["stderr"].as_str().unwrap().to_owned()
                            } else {
                                "fail: src/models/llama.cpp: idempotence violation -- transformable on second run\n".into()
                            }
                        } else {
                            String::new()
                        };
                        if gate == "fail" {
                            stderr.push_str(
                                "fail: generated family patch drifts from checked-in patch\n",
                            );
                        }
                        if compile == "fail" {
                            stderr.push_str("fail: transformed tree failed to compile\n");
                        }
                        if graph == "fail" {
                            stderr.push_str(
                                "fail: no-allocation graph verifier failed on transformed tree\n",
                            );
                        }
                        let mut stdout = if gate == "warn" {
                            "warn: generated family patch drifts from checked-in patch\n".to_owned()
                        } else {
                            String::new()
                        };
                        if stderr.is_empty() {
                            stdout.push_str(&format!(
                                "ok: report valid ({mode}); 1 builders checked\n"
                            ));
                        }
                        exact(&output, (i32::from(!stderr.is_empty()), &stdout, &stderr));
                        if !retained.is_empty() {
                            let receipt = retained.iter().find(|row| row.name == name).unwrap();
                            assert_eq!(fs::read(&receipt.report).unwrap(), body);
                            matches_observation(&output, &receipt.python);
                            let legacy = campaign.legacy(&name, &args);
                            matches_observation(&legacy, &receipt.python);
                        }
                        count += 1;
                    }
                }
            }
        }
    }
    assert_eq!(count, 72);
    let args = [
        "--graph-verify-result",
        "fail",
        "--compile-result",
        "fail",
        "--patch-drift-gate",
        "fail",
        "--patch-check",
        "fail",
    ];
    let output = campaign.run(
        "reverse",
        &fs::read(fixtures().join("ready.json")).unwrap(),
        &args,
    );
    exact(
        &output,
        (
            1,
            "",
            "fail: generated family patch drifts from checked-in patch\nfail: transformed tree failed to compile\nfail: no-allocation graph verifier failed on transformed tree\n",
        ),
    );
}

#[test]
fn retained_cli20_preserves_success_and_usage_decisions() {
    let campaign = Campaign::new("cli20");
    let body = report(r#"{"already_transformed":1}"#);
    let rows: &[(&[&str], i32)] = &[
        (&["--help"], 0),
        (&["-h"], 0),
        (&["--hel"], 2),
        (&["--h"], 2),
        (&["--m", "idempotence"], 2),
        (&["--mod=idempotence"], 2),
        (&["--mode=idempotence"], 0),
        (&["--mo=idempotence"], 2),
        (&["--compile", "pass"], 2),
        (&["--graph", "pass"], 2),
        (&["--patch", "pass"], 2),
        (&["--p=pass"], 2),
        (&["--unknown", "--help"], 0),
        (&["--mode", "--help"], 2),
        (&["--report", "--help"], 2),
        (&["--help=1"], 2),
        (&["--", "--help"], 2),
        (&["-hfoo"], 2),
        (&["--mode=other", "--help"], 2),
        (&["--help", "--mode=other"], 0),
    ];
    let retained: Vec<Receipt> = std::env::var_os("REPORT_RETAINED_EXTRA")
        .map_or_else(Vec::new, |path| {
            serde_json::from_slice(&fs::read(path).unwrap()).unwrap()
        });
    for (index, (args, expected)) in rows.iter().enumerate() {
        let name = format!("cli-{}", index + 99);
        let output = campaign.run(&name, body.as_bytes(), args);
        assert_eq!(output.status.code(), Some(*expected), "{name}: {output:?}");
        if *expected == 0 {
            assert!(output.stderr.is_empty());
            assert!(!output.stdout.is_empty());
        } else {
            assert!(output.stdout.is_empty());
            assert!(output.stderr.starts_with(b"usage: "));
        }
        if !retained.is_empty() {
            let receipt = &retained[index + 99];
            assert_eq!(receipt.name, name);
            assert_eq!(receipt.args, *args);
            assert_eq!(fs::read(&receipt.report).unwrap(), body.as_bytes());
            assert_eq!(receipt.python.code, *expected);
            if *expected == 0 {
                matches_observation(&output, &receipt.python);
            }
            let legacy = campaign.legacy(&name, args);
            assert_eq!(legacy.status.code(), Some(*expected));
            if *expected == 0 {
                matches_observation(&legacy, &receipt.python);
            }
        }
    }
}

#[test]
#[ignore = "reads retained evidence only in the explicitly bounded local campaign"]
fn retained45_have_exactly_37_domain_dispositions_and_eight_fixes() {
    let path = std::env::var_os("REPORT_RETAINED_EXTRA").expect("retained119 receipt required");
    let bytes = fs::read(path).unwrap();
    let retained: Vec<Receipt> = serde_json::from_slice(&bytes).unwrap();
    let expected = [
        3, 4, 6, 9, 10, 11, 12, 15, 17, 18, 20, 30, 32, 35, 37, 39, 41, 42, 44, 54, 56, 59, 61, 63,
        65, 67, 69, 79, 81, 83, 85, 87, 88, 89, 90, 91, 92, 94, 101, 102, 103, 104, 107, 108, 111,
    ];
    let actual: Vec<_> = retained
        .iter()
        .enumerate()
        .filter(|(_, row)| row.python.code != row.rust.code)
        .map(|(index, _)| index)
        .collect();
    assert_eq!(actual, expected);
    let campaign = Campaign::new("retained45-disposition");
    let mut dispositions = Vec::new();
    for index in expected {
        let row = &retained[index];
        let class = if [94, 101, 102, 103, 104, 107, 108, 111].contains(&index) {
            "fix"
        } else {
            "domain/admission"
        };
        dispositions.push(serde_json::json!({"index":index,"name":row.name,"args":row.args,"class":class,"input_sha256":hex::encode(Sha256::digest(fs::read(&row.report).unwrap()))}));
    }
    assert_eq!(
        dispositions
            .iter()
            .filter(|row| row["class"] == "fix")
            .count(),
        8
    );
    save(
        &campaign.directory.join("dispositions.json"),
        &serde_json::to_vec_pretty(&dispositions).unwrap(),
    );
    save(
        &campaign.directory.join("receipt.sha256"),
        hex::encode(Sha256::digest(bytes)).as_bytes(),
    );
    for index in [11, 12, 16, 40, 64] {
        let row = &retained[index];
        let body = fs::read(&row.report).unwrap();
        let output = campaign.run(&format!("extra-{index}"), &body, &[]);
        let fields: &[&str] = match index {
            11 => &["transformable", "already_transformed"],
            12 => &["transformable", "supported_auxiliary"],
            16 => &["already_transformed", "supported_auxiliary"],
            40 => &["transformable", "already_transformed"],
            64 => &[
                "transformable",
                "already_transformed",
                "supported_auxiliary",
            ],
            _ => unreachable!(),
        };
        let stderr: String = fields
            .iter()
            .map(|field| {
                format!("fail: summary.{field}: expected integer in 0..=9223372036854775807\n")
            })
            .collect();
        exact(&output, (1, "", &stderr));
    }
}

#[test]
fn captured_producer_reports_preserve_valid_output_bytes() {
    let campaign = Campaign::new("captured-producer");
    for (pass, digest) in [
        (
            "first",
            "97eece363002f7a39935fabef84737d0c55af4231afbfccab905eaba4ba8ad7c",
        ),
        (
            "second",
            "0f3c9c67bdd59510a3497aeb34fe2ebd7e5a865e7cfc2fbdc6178e382bda53e1",
        ),
    ] {
        let bytes = fs::read(fixtures().join(format!("producer-{pass}.json"))).unwrap();
        assert_eq!(hex::encode(Sha256::digest(&bytes)), digest);
        for mode in ["validate", "idempotence"] {
            let output = campaign.run(&format!("{pass}-{mode}"), &bytes, &["--mode", mode]);
            if pass == "first" && mode == "idempotence" {
                exact(
                    &output,
                    (
                        1,
                        "",
                        "fail: src/models/continue-path.cpp: idempotence violation -- transformable on second run\nfail: src/models/continue-unbraced.cpp: idempotence violation -- transformable on second run\nfail: src/models/conventional.cpp: idempotence violation -- transformable on second run\n",
                    ),
                );
            } else {
                exact(
                    &output,
                    (
                        0,
                        &format!("ok: report valid ({mode}); 6 builders checked\n"),
                        "",
                    ),
                );
            }
        }
    }
}

#[test]
#[ignore = "requires a source-identified existing native producer; never builds or downloads it"]
fn actual_producer_pair_preserves_four_gate_pairs() {
    let tool = std::env::var_os("REPORT_PRODUCER").expect("source-identified producer required");
    let campaign = Campaign::new("producer-pair");
    let root = campaign.directory.join("source");
    fs::create_dir_all(root.join("src/models")).unwrap();
    let _source_cleanup = ProducerCopy(root.clone());
    let names = [
        "conventional.cpp",
        "continue-path.cpp",
        "continue-unbraced.cpp",
        "auxiliary.cpp",
        "multiple-domains.cpp",
        "delegated-opaque.cpp",
    ];
    let source = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../skippy-stage-rewriter/tests/source/src/models");
    let mut inputs = Vec::new();
    for name in names {
        let bytes = fs::read(source.join(name)).unwrap();
        save(&root.join("src/models").join(name), &bytes);
        inputs.push(serde_json::json!({"file":name,"sha256":hex::encode(Sha256::digest(bytes))}));
    }
    save(
        &campaign.directory.join("source-inputs.json"),
        &serde_json::to_vec_pretty(&inputs).unwrap(),
    );
    for (pass, apply) in [("first", true), ("second", false)] {
        let report_path = campaign.directory.join(format!("{pass}.json"));
        assert!(!report_path.exists());
        let mut command = Command::new("gtimeout");
        command
            .args(["-k", "1", "30"])
            .arg(&tool)
            .arg("--source-root")
            .arg(&root)
            .args(["--llama-commit", "fixture", "--report"])
            .arg(&report_path);
        if apply {
            command.arg("--apply");
        }
        command
            .args(names.map(|name| root.join("src/models").join(name)))
            .args(["--", "-std=c++17"]);
        save(
            &campaign.directory.join(format!("{pass}.argv")),
            format!("{command:?}\n").as_bytes(),
        );
        let output = command.output().unwrap();
        save(
            &campaign.directory.join(format!("{pass}.stdout")),
            &output.stdout,
        );
        save(
            &campaign.directory.join(format!("{pass}.stderr")),
            &output.stderr,
        );
        save(
            &campaign.directory.join(format!("{pass}.status")),
            format!("{:?}\n", output.status.code()).as_bytes(),
        );
        assert_eq!(output.status.code(), Some(0), "{output:?}");
        let bytes = fs::read(&report_path).unwrap();
        let report: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        let builders = report["builders"].as_array().unwrap();
        assert_eq!(builders.len(), 6);
        assert_eq!(report["llama_cpp_commit"], "fixture");
        assert_eq!(
            report["summary"][if apply {
                "transformable"
            } else {
                "already_transformed"
            }],
            3
        );
        for verdict in [
            "supported_auxiliary",
            "supported_whole_model",
            "unsupported_shape",
        ] {
            assert_eq!(report["summary"][verdict], 1);
        }
        let refused = builders
            .iter()
            .find(|builder| builder["verdict"] == "unsupported_shape")
            .unwrap();
        assert_eq!(refused["unsupported_reason"], "no layer block loop");
        let kinds: std::collections::BTreeSet<_> = builders
            .iter()
            .flat_map(|builder| builder["edits"].as_array().unwrap())
            .map(|edit| edit["kind"].as_str().unwrap())
            .collect();
        if apply {
            assert_eq!(
                kinds,
                [
                    "insert_begin_block",
                    "insert_end_block",
                    "insert_end_block_before_continue",
                    "wrap_end_block_before_continue"
                ]
                .into_iter()
                .collect()
            );
        } else {
            assert!(kinds.is_empty());
        }
        for mode in ["validate", "idempotence"] {
            let name = format!("{pass}-{mode}");
            let rust = campaign.run(&name, &bytes, &["--mode", mode]);
            let legacy = campaign.legacy(&name, &["--mode", mode]);
            assert_eq!(rust.status.code(), legacy.status.code());
            assert_eq!(rust.stdout, legacy.stdout);
            assert_eq!(rust.stderr, legacy.stderr);
            if apply && mode == "idempotence" {
                exact(
                    &rust,
                    (
                        1,
                        "",
                        "fail: src/models/continue-path.cpp: idempotence violation -- transformable on second run\nfail: src/models/continue-unbraced.cpp: idempotence violation -- transformable on second run\nfail: src/models/conventional.cpp: idempotence violation -- transformable on second run\n",
                    ),
                );
            } else {
                exact(
                    &rust,
                    (
                        0,
                        &format!("ok: report valid ({mode}); 6 builders checked\n"),
                        "",
                    ),
                );
            }
        }
        let mut stats = ProducerStats::default();
        stats.visit(&report, 0);
        save(&campaign.directory.join(format!("{pass}.stats.json")), &serde_json::to_vec_pretty(&serde_json::json!({"bytes":bytes.len(),"builders":builders.len(),"edits":builders.iter().map(|builder| builder["edits"].as_array().unwrap().len()).sum::<usize>(),"maximum_container_depth":stats.depth,"maximum_string_bytes":stats.string_bytes,"maximum_numeric_token":stats.number,"sha256":hex::encode(Sha256::digest(&bytes))})).unwrap());
    }
}

#[derive(Default)]
struct ProducerStats {
    depth: usize,
    string_bytes: usize,
    number: u64,
}
impl ProducerStats {
    fn visit(&mut self, value: &serde_json::Value, depth: usize) {
        match value {
            serde_json::Value::Object(entries) => {
                self.depth = self.depth.max(depth + 1);
                for (key, value) in entries {
                    self.string_bytes = self.string_bytes.max(key.len());
                    self.visit(value, depth + 1);
                }
            }
            serde_json::Value::Array(items) => {
                self.depth = self.depth.max(depth + 1);
                for value in items {
                    self.visit(value, depth + 1);
                }
            }
            serde_json::Value::String(text) => {
                self.string_bytes = self.string_bytes.max(text.len())
            }
            serde_json::Value::Number(number) => {
                self.number = self.number.max(number.as_u64().unwrap())
            }
            serde_json::Value::Bool(_) | serde_json::Value::Null => {}
        }
    }
}
