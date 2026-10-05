use super::super::fixture;
use super::*;

fn args(directory: &Path) -> Vec<String> {
    let mut args = Vec::new();
    for (flag, value) in [
        ("--production", "production.json"),
        ("--event-disabled", "reference.json"),
        ("--baseline", "baseline.json"),
        ("--output", "report.json"),
    ] {
        args.push(flag.into());
        args.push(directory.join(value).to_string_lossy().into_owned());
    }
    for (flag, value) in [
        ("--bootstrap-samples", "100"),
        ("--seed", "42"),
        ("--max-degradation-percent", "5"),
        ("--min-primary-pairs", "20"),
        ("--min-scenario-pairs", "10"),
        ("--max-mdd-percent", "5"),
    ] {
        args.push(flag.into());
        args.push(value.into());
    }
    args
}

fn inputs(directory: &Path) {
    for (name, mode) in [
        ("production.json", "production"),
        ("reference.json", "event-disabled"),
        ("baseline.json", "production"),
    ] {
        fs::write(
            directory.join(name),
            serde_json::to_vec(&fixture::manifest(mode)).unwrap(),
        )
        .unwrap();
    }
}

#[test]
fn frozen_flags_remain_advertised_and_required() {
    let directory = tempfile::tempdir().unwrap();
    let args = args(directory.path());
    assert!(Command::parse(&args).is_ok());
    for flag in VALUE_FLAGS {
        assert!(USAGE.contains(flag));
        let index = args.iter().position(|arg| arg == flag).unwrap();
        let mut incomplete = args.clone();
        incomplete.drain(index..index + 2);
        assert!(Command::parse(&incomplete).is_err());
    }
    assert!(USAGE.contains("--report-holm"));
}

#[test]
fn duplicated_unknown_or_missing_options_refuse_admission() {
    let directory = tempfile::tempdir().unwrap();
    let base = args(directory.path());
    for extra in [
        vec!["--seed", "42"],
        vec!["--unknown", "42"],
        vec!["--report-holm", "--report-holm"],
    ] {
        let mut args = base.clone();
        args.extend(extra.into_iter().map(String::from));
        assert!(Command::parse(&args).is_err());
    }
    assert!(Command::parse(&["--production".into()]).is_err());
}

#[test]
fn malformed_input_preserves_existing_report_and_leaves_no_staging_file() {
    let directory = tempfile::tempdir().unwrap();
    let output = directory.path().join("report.json");
    fs::write(&output, b"previous report").unwrap();
    inputs(directory.path());
    fs::write(directory.path().join("baseline.json"), b"{}").unwrap();
    assert!(
        Command::parse(&args(directory.path()))
            .unwrap()
            .execute()
            .is_err()
    );
    assert_eq!(fs::read(output).unwrap(), b"previous report");
    assert_eq!(fs::read_dir(directory.path()).unwrap().count(), 4);
}

#[test]
fn output_alias_cannot_overwrite_input_manifest() {
    let directory = tempfile::tempdir().unwrap();
    inputs(directory.path());
    let mut args = args(directory.path());
    let index = args.iter().position(|arg| arg == "--output").unwrap();
    args[index + 1] = directory
        .path()
        .join("production.json")
        .to_string_lossy()
        .into_owned();
    let before = fs::read(&args[index + 1]).unwrap();
    assert!(Command::parse(&args).unwrap().execute().is_err());
    assert_eq!(fs::read(&args[index + 1]).unwrap(), before);
}

#[test]
fn healthy_fixture_publishes_complete_report_and_machine_readable_summary() {
    let directory = tempfile::tempdir().unwrap();
    inputs(directory.path());
    let summary = Command::parse(&args(directory.path()))
        .unwrap()
        .execute()
        .unwrap();
    assert_eq!(summary["certification_status"], "pass");
    let bytes = fs::read(directory.path().join("report.json")).unwrap();
    assert_eq!(bytes.last(), Some(&b'\n'));
    let report: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(
        report["resampling_algorithm"],
        super::super::resampling::ALGORITHM
    );
    assert_eq!(fs::read_dir(directory.path()).unwrap().count(), 4);
}

#[test]
fn blocked_fixture_publishes_diagnostics_and_command_returns_failure() {
    let directory = tempfile::tempdir().unwrap();
    inputs(directory.path());
    let mut input = fixture::manifest("production");
    input["health"] = serde_json::Value::Null;
    fs::write(
        directory.path().join("production.json"),
        serde_json::to_vec(&input).unwrap(),
    )
    .unwrap();
    assert!(run(&args(directory.path())).is_err());
    let report: serde_json::Value =
        serde_json::from_slice(&fs::read(directory.path().join("report.json")).unwrap()).unwrap();
    assert_eq!(report["certification_status"], "blocked");
    assert!(
        report["blocking_reasons"]
            .as_array()
            .unwrap()
            .iter()
            .any(|reason| reason == "health_unavailable")
    );
}

#[test]
fn oversized_input_refuses_before_report_publication() {
    let directory = tempfile::tempdir().unwrap();
    inputs(directory.path());
    fs::File::create(directory.path().join("production.json"))
        .unwrap()
        .set_len(MAX_INPUT_BYTES + 1)
        .unwrap();
    assert!(
        Command::parse(&args(directory.path()))
            .unwrap()
            .execute()
            .is_err()
    );
    assert!(!directory.path().join("report.json").exists());
}
