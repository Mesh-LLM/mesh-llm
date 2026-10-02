use super::support::{self, *};

fn rejected_fixtures(names: &[&str]) -> TestResult {
    let stage = Stage::new()?;
    for name in names {
        let path = fixtures().join(format!("{name}.json"));
        let actual = stage.run(&["--matrix", path.to_str().ok_or("fixture path")?])?;
        assert_eq!(actual.code, 1, "{name}");
        assert!(actual.stdout.is_empty(), "{name}");
        assert!(!actual.stderr.is_empty(), "{name}");
    }
    Ok(())
}

#[test]
fn nightly_all_profile_preserves_shell_fields_and_requested_concurrency_order() -> TestResult {
    let stage = Stage::new()?;
    for (name, expected) in [
        ("valid", SHELL),
        (
            "concurrency-reordered",
            "all\t16\t2\t131072\t32768\t32768\t131072\t5\t2\t4\t2048\t8,1,4,2\n",
        ),
    ] {
        let path = fixtures().join(format!("{name}.json"));
        let actual = stage.run(&["--matrix", path.to_str().ok_or("fixture path")?])?;
        assert_outcome(name, &actual, (0, expected, ""));
    }
    Ok(())
}

#[test]
fn nightly_profile_requires_a_replay_object_and_all_mode() -> TestResult {
    rejected_fixtures(&[
        "matrix-array",
        "replay-absent",
        "replay-array",
        "mode-checkpoint",
        "mode-final",
        "mode-missing",
        "mode-array",
    ])
}

#[test]
fn nightly_counts_require_positive_integer_values_and_unique_concurrency() -> TestResult {
    rejected_fixtures(&[
        "positive-bool",
        "positive-float",
        "positive-null",
        "positive-missing",
        "positive-zero",
        "positive-negative",
        "concurrency-empty",
        "concurrency-duplicate",
        "concurrency-bool",
        "concurrency-float",
        "concurrency-null",
        "concurrency-missing",
        "concurrency-zero",
    ])
}

#[test]
fn nightly_profile_requires_worker_waves_frameworks_context_and_selection_window() -> TestResult {
    rejected_fixtures(&[
        "sessions-waves",
        "sessions-frameworks",
        "context-short",
        "window-equal",
    ])
}

#[test]
fn nightly_profile_pins_sampling_backend_and_selection() -> TestResult {
    rejected_fixtures(&[
        "backend-cuda",
        "selection-unknown",
        "temperature-drift",
        "seed-drift",
    ])
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
fn nightly_positive_fields_reject_zero() -> TestResult {
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
    Ok(())
}

#[test]
fn pinned_sampling_accepts_exact_numeric_values_and_rejects_wrong_types() -> TestResult {
    let stage = Stage::new()?;
    for (field, original, token, accepted) in [
        ("temperature", "0", "0.0", true),
        ("temperature", "0", "-0.0", true),
        ("temperature", "0", "0e10", true),
        ("temperature", "0", "true", false),
        ("temperature", "0", "\"0\"", false),
        ("temperature", "0", "9007199254740993", false),
        ("seed", "42", "42.0", true),
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
