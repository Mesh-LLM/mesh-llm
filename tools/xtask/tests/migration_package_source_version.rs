use std::error::Error;
use std::process::Command;

type TestResult = Result<(), Box<dyn Error>>;

#[path = "migration_package_source_version/encoding_approval.rs"]
mod encoding_approval;

struct Case {
    kind: &'static str,
    source: String,
    stdout: String,
    success: bool,
}

fn cases() -> Vec<Case> {
    let huge = "009".repeat(2000);
    let workspace = [
        (
            "[workspace.package]\nversion=\"first\" tail\nversion=\"last\"",
            "first",
        ),
        (
            "[workspace.package]\n[bad\nversion=\"wrong\"\n[workspace.package]\nversion=\"again\"",
            "again",
        ),
        (
            " [workspace.package]\u{a0}\r\nversion =\u{2003}\"arbitrary\\text\"",
            "arbitrary\\text",
        ),
        (
            "[workspace.package]\nversion=\"\"\nversion='wrong'\nversion=\"value\"",
            "value",
        ),
        (
            "[workspace.package]\rversion=\"left\u{2028}right\"\r",
            "left\u{2028}right",
        ),
        ("[workspace.package]\nversion=\"lf\"\n", "lf"),
        ("", ""),
        ("[package]\nversion=\"wrong\"", ""),
        ("[workspace.package] # comment\nversion=\"wrong\"", ""),
    ];
    let mut cases: Vec<Case> = workspace
        .into_iter()
        .map(|(source, value)| Case {
            kind: "workspace",
            source: source.to_owned(),
            stdout: if value.is_empty() {
                String::new()
            } else {
                format!("{value}\n")
            },
            success: !value.is_empty(),
        })
        .collect();
    for (source, value) in [
        ("pub const ABI_VERSION_PATCH: u32 = 3;\npub const ABI_VERSION_MINOR: u32 = 2;\npub const ABI_VERSION_MAJOR: u32 = 1;\npub const ABI_VERSION_MAJOR: u32 = 004; tail".to_owned(), "004.2.3".to_owned()),
        (" pub const ABI_VERSION_MAJOR: u32 = 01;\u{a0}\r\npub const ABI_VERSION_MINOR: u32 = 02;\rpub const ABI_VERSION_PATCH: u32 = 03;\n".to_owned(), "01.02.03".to_owned()),
        (format!("pub const ABI_VERSION_MAJOR: u32 = {huge};\npub const ABI_VERSION_MINOR: u32 = 00;\npub const ABI_VERSION_PATCH: u32 = 000;"), format!("{huge}.00.000")),
        ("pub const ABI_VERSION_MAJOR: u32 = 1;\npub const ABI_VERSION_MINOR: u32 = 2;\npub const ABI_VERSION_PATCH: u32 = 3;\npub const ABI_VERSION_PATCH: u32 = 4 ;\npub const ABI_VERSION_MINOR: u64 = 9;\npub const ABI_VERSION_MAJOR: u32 = ９;".to_owned(), "1.2.3".to_owned()),
        (String::new(), String::new()),
        ("pub const ABI_VERSION_PATCH: u32 = 3;".to_owned(), String::new()),
        ("pub const ABI_VERSION_MAJOR: u32 = 1;".to_owned(), String::new()),
        ("pub const ABI_VERSION_MAJOR: u32 = 1;\npub const ABI_VERSION_MINOR: u32 = 2;".to_owned(), String::new()),
    ] {
        cases.push(Case {
            kind: "abi",
            source,
            stdout: if value.is_empty() { String::new() } else { format!("{value}\n") },
            success: !value.is_empty(),
        });
    }
    cases
}

#[test]
fn source_version_cli_matches_when_textual_fixtures_are_supplied() -> TestResult {
    let directory = tempfile::tempdir()?;
    let path = directory.path().join("source");
    for case in cases() {
        std::fs::write(&path, &case.source)?;
        let actual = Command::new(env!("CARGO_BIN_EXE_xtask"))
            .args(["native", "package-source-version", case.kind])
            .arg(&path)
            .output()?;
        assert_eq!(actual.status.success(), case.success, "{}", case.source);
        assert_eq!(actual.stdout, case.stdout.as_bytes());
    }
    Ok(())
}

#[test]
fn source_version_cli_fails_when_source_is_invalid_utf8() -> TestResult {
    let directory = tempfile::tempdir()?;
    let path = directory.path().join("source");
    std::fs::write(&path, b"\xff")?;
    let actual = Command::new(env!("CARGO_BIN_EXE_xtask"))
        .args(["native", "package-source-version", "abi"])
        .arg(path)
        .output()?;
    assert_eq!(actual.status.code(), Some(1));
    assert!(actual.stdout.is_empty());
    Ok(())
}

#[test]
fn source_version_cli_fails_when_source_cannot_be_read() -> TestResult {
    let directory = tempfile::tempdir()?;
    let actual = Command::new(env!("CARGO_BIN_EXE_xtask"))
        .args(["native", "package-source-version", "workspace"])
        .arg(directory.path().join("missing"))
        .output()?;
    assert_eq!(actual.status.code(), Some(1));
    assert!(actual.stdout.is_empty());
    Ok(())
}

#[test]
fn source_version_is_discoverable_when_help_is_requested() -> TestResult {
    for (args, status) in [
        (vec!["--help"], 0),
        (vec!["native", "--help"], 1),
        (vec!["native", "package-source-version", "--help"], 2),
    ] {
        let actual = Command::new(env!("CARGO_BIN_EXE_xtask"))
            .args(&args)
            .output()?;
        assert_eq!(actual.status.code(), Some(status), "{args:?}");
        let usage = [actual.stdout, actual.stderr].concat();
        assert!(
            usage
                .windows(b"native package-source-version".len())
                .any(|window| window == b"native package-source-version"),
            "{args:?}"
        );
    }
    Ok(())
}
