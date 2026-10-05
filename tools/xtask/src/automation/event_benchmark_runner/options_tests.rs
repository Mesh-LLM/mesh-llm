use super::*;

fn flags() -> Vec<String> {
    [
        "--binary",
        "current",
        "--model",
        "local.gguf",
        "--output-dir",
        "evidence",
        "--pairs-primary",
        "20",
        "--pairs-scenario",
        "10",
        "--seed",
        "42",
        "--mode",
        "production",
        "--mode",
        "event-disabled",
        "--scenario",
        "fixture",
    ]
    .into_iter()
    .map(String::from)
    .collect()
}

fn set(args: &mut [String], flag: &str, value: &str) {
    let index = args.iter().position(|arg| arg == flag).unwrap();
    args[index + 1] = value.into();
}

#[test]
fn comparison_a_preserves_frozen_flags_defaults_and_local_model_reference() {
    let command = Command::parse(&flags()).unwrap();
    assert_eq!(command.spec.seed, 42);
    assert_eq!(command.spec.pairs_primary, 20);
    assert_eq!(command.spec.pairs_scenario, 10);
    assert_eq!(command.sides[0].mode, Mode::Production);
    assert_eq!(command.sides[1].mode, Mode::EventDisabled);
    assert_eq!(command.sides[0].binary, command.sides[1].binary);
    assert_eq!(command.model, PathBuf::from("local.gguf"));
    assert_eq!(command.output_dir, PathBuf::from("evidence"));
    assert_eq!(
        (
            command.attempt,
            command.max_tokens,
            command.readiness_secs,
            command.request_secs,
            command.shutdown_secs,
            command.execution_secs
        ),
        (1, 64, 120, 120, 15, 86400)
    );
    for flag in REQUIRED.iter().chain(OPTIONAL) {
        assert!(USAGE.contains(flag));
    }
}

#[test]
fn comparison_b_requires_exactly_one_mode_and_preserves_baseline_identity() {
    let mut args = flags();
    let second = args.iter().rposition(|value| value == "--mode").unwrap();
    args.drain(second..second + 2);
    args.extend(["--baseline-binary".into(), "baseline".into()]);
    let command = Command::parse(&args).unwrap();
    assert_eq!(command.sides[0].side_id, "current");
    assert_eq!(command.sides[1].side_id, "baseline");
    assert_eq!(command.sides[1].binary, PathBuf::from("baseline"));
    assert_eq!(command.sides[0].mode, command.sides[1].mode);
    args.extend(["--mode".into(), "event-disabled".into()]);
    assert!(Command::parse(&args).is_err());
}

#[test]
fn malformed_duplicate_and_unknown_flags_refuse_without_launching() {
    for extra in [
        vec!["--seed", "43"],
        vec!["--unknown", "x"],
        vec!["--attempt"],
        vec!["--mode", "invented"],
    ] {
        let mut args = flags();
        args.extend(extra.into_iter().map(String::from));
        assert!(Command::parse(&args).is_err());
    }
    for flag in REQUIRED {
        let mut args = flags();
        let index = args.iter().position(|arg| arg == flag).unwrap();
        args.drain(index..index + 2);
        assert!(Command::parse(&args).is_err());
    }
}

#[test]
fn bounded_counts_seed_and_retry_attempts_refuse_outside_the_contract() {
    for (flag, value) in [
        ("--seed", "-1"),
        ("--seed", "18446744073709551616"),
        ("--pairs-primary", "0"),
        ("--pairs-scenario", "0"),
        ("--pairs-primary", "10001"),
    ] {
        let mut args = flags();
        set(&mut args, flag, value);
        assert!(Command::parse(&args).is_err());
    }
    for (flag, value) in [
        ("--attempt", "0"),
        ("--attempt", "3"),
        ("--max-tokens", "0"),
        ("--shutdown-timeout-secs", "0"),
        ("--execution-timeout-secs", "360"),
    ] {
        let mut args = flags();
        args.extend([flag.into(), value.into()]);
        assert!(Command::parse(&args).is_err());
    }
    let mut args = flags();
    set(&mut args, "--seed", "18446744073709551615");
    args.extend(["--attempt".into(), "2".into()]);
    assert_eq!(Command::parse(&args).unwrap().spec.seed, u64::MAX);
}

#[test]
fn cli_budget_matches_retained_trial_and_refuses_shutdown_before_preflight() {
    let mut args = flags();
    args.extend([
        "--readiness-timeout-secs".into(),
        "40".into(),
        "--request-timeout-secs".into(),
        "10".into(),
        "--shutdown-timeout-secs".into(),
        "1".into(),
        "--execution-timeout-secs".into(),
        "70".into(),
    ]);
    assert!(Command::parse(&args).is_err());
    set(&mut args, "--execution-timeout-secs", "107");
    assert!(Command::parse(&args).is_err());
    set(&mut args, "--execution-timeout-secs", "108");
    assert!(Command::parse(&args).is_ok());
    set(&mut args, "--shutdown-timeout-secs", "301");
    set(&mut args, "--execution-timeout-secs", "86400");
    assert!(Command::parse(&args).is_err());
}
