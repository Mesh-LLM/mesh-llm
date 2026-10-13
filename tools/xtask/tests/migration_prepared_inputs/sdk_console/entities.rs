use super::{fixture, invocation, verified};
use crate::support::TestResult;
use crate::support::snapshot;

#[test]
fn sdk_console_resolves_standard_escaped_asset_paths() -> TestResult {
    let scratch = fixture(
        "<img src='assets/a&amp;b.js'><img src='assets/&#97;.js'>",
        "index.html\n",
    )?;
    scratch.write("console/assets/a&b.js", b"escaped asset")?;
    scratch.write("console/assets/a.js", b"numeric asset")?;
    verified(&scratch)
}

#[test]
fn sdk_console_rejects_unknown_named_entity() -> TestResult {
    let scratch = fixture("<img src='assets/a&copy;.js'>", "index.html\n")?;
    let output = invocation(&scratch, "sdk-console-verify").run(scratch.path())?;
    assert_eq!(output.status.code(), Some(1));
    assert!(output.stdout.is_empty());
    Ok(())
}

#[test]
fn sdk_console_rejects_missing_asset_without_mutation() -> TestResult {
    let scratch = fixture("<img src='assets/missing.js'>", "index.html\n")?;
    let before = snapshot(&scratch.join("console"))?;
    let output = invocation(&scratch, "sdk-console-verify").run(scratch.path())?;
    assert_eq!(output.status.code(), Some(1));
    assert!(output.stdout.is_empty());
    assert_eq!(snapshot(&scratch.join("console"))?, before);
    Ok(())
}

#[test]
fn sdk_console_rejects_invalid_utf8_without_mutation() -> TestResult {
    for filename in ["index.html", "manifest.txt"] {
        let scratch = fixture("", "index.html\n")?;
        scratch.write(&format!("console/{filename}"), b"\xff")?;
        let before = snapshot(&scratch.join("console"))?;
        let output = invocation(&scratch, "sdk-console-verify")
            .status_only()
            .run(scratch.path())?;
        assert_eq!(output.status.code(), Some(1));
        assert!(output.stdout.is_empty());
        assert!(!output.stderr.is_empty());
        assert_eq!(snapshot(&scratch.join("console"))?, before);
    }
    Ok(())
}
