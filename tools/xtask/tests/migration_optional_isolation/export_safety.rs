use super::export_cases::{Comparison, ExportCase};
use super::export_process::observe;
use super::support::{TestResult, valid};

#[test]
fn migration_optional_isolation_export_deep_render_and_drop() -> TestResult {
    let nested = format!("{}0{}", "[".repeat(5000), "]".repeat(5000));
    let raw = valid().replace(
        r#""mode":"all""#,
        &format!(r#""deep":{nested},"mode":"all""#),
    );
    let case = ExportCase {
        id: "rust-deep-safety".into(),
        description: "Rust-only worker render/drop safety, not Python recursion parity".into(),
        input: Some(raw.into_bytes()),
        args: [
            "--matrix",
            "matrix.json",
            "--json-output",
            "params.json",
            "--github-env",
            "github.env",
            "--print-shell",
        ]
        .map(str::to_owned)
        .to_vec(),
        initial: Vec::new(),
        directories: Vec::new(),
        comparison: Comparison::Exact,
    };
    let actual = observe(&case, None)?;
    assert_eq!(actual.status, 1);
    assert!(actual.stdout.is_empty());
    assert!(!actual.files.contains_key("params.json"));
    Ok(())
}

#[test]
fn migration_optional_isolation_export_rejects_unowned_options() -> TestResult {
    let case = ExportCase {
        id: "unowned-option".into(),
        description: "run-family remains legacy-owned".into(),
        input: Some(valid().into_bytes()),
        args: [
            "--matrix",
            "matrix.json",
            "--json-output",
            "params.json",
            "--run-family",
            "dense",
        ]
        .map(str::to_owned)
        .to_vec(),
        initial: vec![("params.json".into(), b"UNCHANGED".to_vec())],
        directories: Vec::new(),
        comparison: Comparison::UsageDiagnostic,
    };
    let actual = observe(&case, None)?;
    assert_eq!(actual.status, 2);
    assert!(actual.stdout.is_empty());
    assert!(
        String::from_utf8(actual.stderr)?
            .ends_with("error: unrecognized arguments: --run-family\n")
    );
    assert_eq!(
        actual.files.get("params.json"),
        Some(&Some(b"UNCHANGED".to_vec()))
    );
    Ok(())
}
