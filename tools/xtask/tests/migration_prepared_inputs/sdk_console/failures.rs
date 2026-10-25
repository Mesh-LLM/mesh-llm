use super::{fixture, invocation};
use crate::support::{TestResult, assert_output};
use std::fs;

#[test]
fn sdk_console_verify_when_duplicate_missing_refs_precede_unsafe_manifest() -> TestResult {
    let scratch = fixture("<img href=z src=b><img src=b>", "../unsafe\n")?;
    let output = invocation(&scratch, "sdk-console-verify").run(scratch.path())?;
    assert_output(
        &output,
        1,
        "",
        "console index references missing assets: b, b, z\n",
    );
    Ok(())
}

#[test]
fn sdk_console_verify_when_manifest_entries_are_invalid() -> TestResult {
    for (manifest, expected) in [
        ("\n", "console manifest.txt must include index.html\n"),
        (
            "index.html\n/absolute\n",
            "unsafe console manifest path: /absolute\n",
        ),
        ("index.html\na//b\n", "unsafe console manifest path: a//b\n"),
        (
            "index.html\na/../b\n",
            "unsafe console manifest path: a/../b\n",
        ),
        ("index.html\na\\b\n", "unsafe console manifest path: a\\b\n"),
        (
            "index.html\nz\na\n",
            "console manifest references missing asset: z\n",
        ),
        (
            "index.html\nassets\n",
            "console manifest references missing asset: assets\n",
        ),
    ] {
        let scratch = fixture("", manifest)?;
        let output = invocation(&scratch, "sdk-console-verify").run(scratch.path())?;
        assert_output(&output, 1, "", expected);
    }
    Ok(())
}

#[test]
fn sdk_console_verify_when_required_files_are_directories() -> TestResult {
    for name in ["index.html", "manifest.txt"] {
        let scratch = fixture("", "index.html\n")?;
        let path = scratch.join(&format!("console/{name}"));
        fs::remove_file(&path)?;
        fs::create_dir(&path)?;
        let output = invocation(&scratch, "sdk-console-verify").run(scratch.path())?;
        assert_output(
            &output,
            1,
            "",
            &format!("missing console {name}: {}\n", path.display()),
        );
    }
    Ok(())
}

#[test]
fn sdk_console_verify_when_only_nested_javascript_exists() -> TestResult {
    let scratch = fixture("", "index.html\n")?;
    fs::remove_file(scratch.join("console/assets/app.js"))?;
    scratch.write("console/assets/nested/app.js", b"nested")?;
    let output = invocation(&scratch, "sdk-console-verify").run(scratch.path())?;
    assert_output(
        &output,
        1,
        "",
        "console assets must include at least one JavaScript asset under assets/\n",
    );
    Ok(())
}

#[test]
fn sdk_console_verify_when_referenced_css_is_outside_assets() -> TestResult {
    let scratch = fixture("<link href=app.css>", "index.html\napp.css\n")?;
    scratch.write("console/app.css", b"css")?;
    let output = invocation(&scratch, "sdk-console-verify").run(scratch.path())?;
    assert_output(
        &output,
        1,
        "",
        "console index references CSS, but no CSS asset exists under assets/\n",
    );
    Ok(())
}
