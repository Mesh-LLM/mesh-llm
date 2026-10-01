use super::support::{self, *};
use serde::Deserialize;
use std::collections::BTreeSet;
use std::fs;

#[derive(Deserialize)]
struct Receipt {
    cases: Vec<Case>,
}

#[derive(Deserialize)]
struct Case {
    name: String,
    status: i32,
    stdout: String,
    stderr: String,
}

#[test]
fn migration_optional_isolation_frozen_replay_contract() -> TestResult {
    let receipt: Receipt =
        serde_json::from_slice(&fs::read(fixtures().join("legacy-print-shell.json"))?)?;
    let stage = Stage::new()?;
    let mut names = BTreeSet::new();
    let mut mismatches = Vec::new();
    let (mut successes, mut policies, mut gaps) = (0, 0, 0);
    for case in receipt.cases {
        assert!(names.insert(case.name.clone()), "duplicate receipt name");
        let path = fixtures().join(format!("{}.json", case.name));
        let actual = stage.run(&["--matrix", path.to_str().ok_or("fixture path")?])?;
        let stderr = if case.name == "matrix-array" {
            gaps += 1;
            assert_ne!(
                case.stderr, ROOT_ERROR,
                "unapproved diagnostic gap retained"
            );
            ROOT_ERROR
        } else {
            if case.status == 0 {
                successes += 1;
            } else {
                policies += 1;
            }
            &case.stderr
        };
        if actual.code != case.status
            || actual.stdout != case.stdout.as_bytes()
            || (case.status == 0 && actual.stderr != stderr.as_bytes())
            || (case.status != 0 && actual.stderr.is_empty())
        {
            mismatches.push(format!("{}: {actual:?}", case.name));
        }
        eprintln!(
            "frozen {}: {}",
            case.name,
            if case.name == "matrix-array" {
                "unapproved stderr gap, rejection only"
            } else {
                "exact status/stdout/stderr comparison"
            }
        );
    }
    assert_eq!((names.len(), successes, policies, gaps), (31, 3, 27, 1));
    assert!(mismatches.is_empty(), "{}", mismatches.join("\n"));
    Ok(())
}

#[test]
fn migration_optional_isolation_integer_identity_and_waves() -> TestResult {
    let stage = Stage::new()?;
    let raw = mutate(&[("\"passes\":2", "\"passes\":9007199254740993")]);
    assert_outcome(
        "passes above 2^53",
        &stage.input(raw.as_bytes())?,
        (
            0,
            "all\t16\t2\t131072\t32768\t32768\t131072\t5\t9007199254740993\t4\t2048\t1,2,4,8\n",
            "",
        ),
    );
    for (levels, sessions, expected, accepted) in [
        (
            "[9007199254740992,9007199254740993]",
            "18014398509481986",
            "all\t18014398509481986\t2\t131072\t32768\t32768\t131072\t5\t2\t4\t2048\t9007199254740992,9007199254740993\n",
            true,
        ),
        (
            "[9223372036854775808]",
            "18446744073709551616",
            "all\t18446744073709551616\t2\t131072\t32768\t32768\t131072\t5\t2\t4\t2048\t9223372036854775808\n",
            false,
        ),
        ("[9223372036854775808]", "18446744073709551615", "", false),
        (
            "[340282366920938463463374607431768211456]",
            "680564733841876926926749214863536422912",
            "all\t680564733841876926926749214863536422912\t2\t131072\t32768\t32768\t131072\t5\t2\t4\t2048\t340282366920938463463374607431768211456\n",
            false,
        ),
        (
            "[340282366920938463463374607431768211456]",
            "680564733841876926926749214863536422911",
            "",
            false,
        ),
    ] {
        let raw = mutate(&[
            ("[1,2,4,8]", levels),
            (
                "\"sessions_per_concurrency\":16",
                &format!("\"sessions_per_concurrency\":{sessions}"),
            ),
        ]);
        let actual = stage.input(raw.as_bytes())?;
        assert_eq!(actual.code, i32::from(!accepted));
        if accepted {
            assert_eq!(actual.stdout, expected.as_bytes());
        } else {
            assert!(actual.stdout.is_empty());
            assert!(!actual.stderr.is_empty());
        }
    }
    Ok(())
}

#[test]
fn migration_optional_isolation_policy_order() -> TestResult {
    let stage = Stage::new()?;
    for (name, original) in [
        ("sessions_per_concurrency", "16"),
        ("minimum_worker_waves", "2"),
        ("minimum_context_tokens", "131072"),
        ("minimum_session_prompt_tokens", "32768"),
        ("min_isl", "32768"),
        ("max_isl", "131072"),
        ("min_turns", "5"),
        ("passes", "2"),
        ("warmup_turns", "4"),
        ("max_output_tokens", "2048"),
    ] {
        let raw = mutate(&[(&format!("\"{name}\":{original}"), &format!("\"{name}\":0"))]);
        assert_outcome(
            name,
            &stage.input(raw.as_bytes())?,
            (1, "", &format!("{name} must be a positive integer\n")),
        );
    }
    let cases: &[(&[(&str, &str)], &str)] = &[
        (
            &[
                ("\"min_turns\":5", "\"min_turns\":0"),
                ("\"passes\":2", "\"passes\":0"),
            ],
            "min_turns must be a positive integer\n",
        ),
        (
            &[("\"passes\":2", "\"passes\":0"), ("[1,2,4,8]", "[1,1]")],
            "passes must be a positive integer\n",
        ),
        (
            &[
                (
                    "\"sessions_per_concurrency\":16",
                    "\"sessions_per_concurrency\":2",
                ),
                (
                    "\"minimum_context_tokens\":131072",
                    "\"minimum_context_tokens\":1",
                ),
            ],
            WAVES_ERROR,
        ),
        (
            &[
                (
                    "\"sessions_per_concurrency\":16",
                    "\"sessions_per_concurrency\":2",
                ),
                ("[1,2,4,8]", "[1]"),
                (
                    "\"minimum_context_tokens\":131072",
                    "\"minimum_context_tokens\":1",
                ),
            ],
            "session count must cover all three frameworks\n",
        ),
        (
            &[
                (
                    "\"minimum_context_tokens\":131072",
                    "\"minimum_context_tokens\":1",
                ),
                ("\"max_isl\":131072", "\"max_isl\":32768"),
            ],
            "nightly requires at least 128K effective context\n",
        ),
        (
            &[
                ("\"max_isl\":131072", "\"max_isl\":32768"),
                ("\"seed\":42", "\"seed\":43"),
            ],
            "invalid selection window\n",
        ),
        (
            &[
                ("\"seed\":42", "\"seed\":43"),
                ("\"backend\":\"metal\"", "\"backend\":\"cuda\""),
            ],
            SAMPLING_ERROR,
        ),
    ];
    for (changes, expected) in cases {
        assert_outcome(
            expected,
            &stage.input(mutate(changes).as_bytes())?,
            (1, "", expected),
        );
    }
    Ok(())
}

#[test]
fn migration_optional_isolation_sampling_numeric_kinds() -> TestResult {
    let stage = Stage::new()?;
    for (field, original, token, accepted) in [
        ("temperature", "0", "0.0", true),
        ("temperature", "0", "-0.0", true),
        ("temperature", "0", "0e10", true),
        ("temperature", "0", "1e-4000", true),
        ("temperature", "0", "false", true),
        ("temperature", "0", "true", false),
        ("temperature", "0", "\"0\"", false),
        ("temperature", "0", "9007199254740993", false),
        ("seed", "42", "42.0", true),
        ("seed", "42", "42.0000000000000001", true),
        ("seed", "42", "42.00000000000001", false),
        ("seed", "42", "true", false),
        ("seed", "42", "false", false),
        ("seed", "42", "\"42\"", false),
        ("seed", "42", "9007199254740993", false),
    ] {
        let raw = mutate(&[(
            &format!("\"{field}\":{original}"),
            &format!("\"{field}\":{token}"),
        )]);
        let expected = if accepted {
            (0, support::SHELL, "")
        } else {
            (1, "", SAMPLING_ERROR)
        };
        assert_outcome(
            &format!("{field}={token}"),
            &stage.input(raw.as_bytes())?,
            expected,
        );
    }
    Ok(())
}

#[test]
fn migration_optional_isolation_duplicate_and_marker_identity() -> TestResult {
    let stage = Stage::new()?;
    let valid = valid();
    let replay = valid
        .strip_prefix("{\"replay\":")
        .unwrap()
        .strip_suffix('}')
        .unwrap();
    for (raw, accepted) in [
        (format!("{{\"replay\":null,\"replay\":{replay}}}"), true),
        (format!("{{\"replay\":{replay},\"replay\":null}}"), false),
    ] {
        let expected = if accepted {
            (0, SHELL, "")
        } else {
            (1, "", "matrix replay block is missing\n")
        };
        assert_outcome("duplicate replay", &stage.input(raw.as_bytes())?, expected);
    }
    for (replacement, accepted) in [
        (r#""passes":0,"passes":2"#, true),
        (r#""passes":2,"passes":0"#, false),
        (r#""passes":0,"pa\u0073ses":2"#, true),
        (r#""passes":2,"pa\u0073ses":0"#, false),
    ] {
        let raw = mutate(&[(r#""passes":2"#, replacement)]);
        let expected = if accepted {
            (0, SHELL, "")
        } else {
            (1, "", "passes must be a positive integer\n")
        };
        assert_outcome(replacement, &stage.input(raw.as_bytes())?, expected);
    }
    let raw = mutate(&[(r#""mode":"all""#, r#""mode":{"b":1,"a":2,"b":3}"#)]);
    assert_outcome(
        "mode key order",
        &stage.input(raw.as_bytes())?,
        (
            1,
            "",
            "replay mode must be all (got {\"a\": 2, \"b\": 3})\n",
        ),
    );
    for marker in [
        r#"{"\u0000exact-number":"42"}"#,
        r#"{"\u0000ci-metrics-object":null}"#,
        r#"{"\u0000ci-metrics-constant":"NaN"}"#,
        r#"{"\u0000family-object":null}"#,
        r#"{"$serde_json::private::Number":"42"}"#,
    ] {
        for (field, original, expected) in [
            ("passes", "2", "passes must be a positive integer\n"),
            ("seed", "42", SAMPLING_ERROR),
        ] {
            let raw = mutate(&[(
                &format!("\"{field}\":{original}"),
                &format!("\"{field}\":{marker}"),
            )]);
            assert_outcome(marker, &stage.input(raw.as_bytes())?, (1, "", expected));
        }
    }
    Ok(())
}
