use crate::support::{Case, Legacy, Scratch, TestResult, assert_output, snapshot, text};
use std::fs;
use std::path::Path;

const SCRIPT: Legacy = Legacy::Script("scripts/ui-distribution.py");
const SOURCE: &str = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
const TAG: &str = "v0.74.0-rc.1";
const MANIFEST: &str = ".mesh-llm-ui-release.json";
const INDEX: &str = r#"<script type="module" src="/assets/app.js"></script>"#;
const APP: &str = "console.log('release');";
const MISMATCH: &str = "UI release identity or file checksums do not match\n";

/// `json.dumps(sort_keys=True, indent=2) + "\n"` of the legacy stamp.
const GOLDEN: &str = r#"{
  "files": {
    "assets/app.js": "dc1c5eca6b249db6d52558df3e91d29c7f131b7cd793bb108b850bd8f24daf50",
    "index.html": "64f76d91ba098c675d30272f10094c5ecef3f7c427ec3636310f92a26d077d51"
  },
  "release_tag": "v0.74.0-rc.1",
  "schema": 1,
  "source_sha": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
}
"#;

fn raw_dist(scratch: &Scratch) -> Result<std::path::PathBuf, Box<dyn std::error::Error>> {
    scratch.write("dist/assets/app.js", APP.as_bytes())?;
    scratch.write("dist/index.html", INDEX.as_bytes())?;
    Ok(scratch.join("dist"))
}

fn ui(operation: &str, dist: &Path, source: &str, tag: &str) -> Case {
    let dist = dist.to_str().expect("UTF-8 scratch path");
    let args = [
        operation,
        "--dist",
        dist,
        "--source-sha",
        source,
        "--release-tag",
        tag,
    ];
    let case = Case::same(&["ui-distribution"], &args, SCRIPT);
    match operation {
        "stamp" => case.writing(&Path::new(dist).join(MANIFEST)),
        _ => case,
    }
}

fn stamped(scratch: &Scratch) -> Result<std::path::PathBuf, Box<dyn std::error::Error>> {
    let dist = raw_dist(scratch)?;
    assert_output(
        &ui("stamp", &dist, SOURCE, TAG).run(scratch.path())?,
        0,
        "",
        "",
    );
    Ok(dist)
}

#[test]
fn migration_prepared_inputs_ui_stamp_writes_exact_manifest_and_verifies() -> TestResult {
    let scratch = Scratch::new("ui-stamp")?;
    let dist = stamped(&scratch)?;
    assert_eq!(fs::read_to_string(dist.join(MANIFEST))?, GOLDEN);
    assert_output(
        &ui("verify", &dist, SOURCE, TAG).run(scratch.path())?,
        0,
        "",
        "",
    );
    Ok(())
}

/// The lean producer split stamps the raw artifact in a later job: the final
/// artifact must be the raw bytes plus exactly one manifest file.
#[test]
fn migration_prepared_inputs_ui_raw_to_final_adds_only_the_manifest() -> TestResult {
    let scratch = Scratch::new("ui-raw-final")?;
    let dist = raw_dist(&scratch)?;
    let mut expected = snapshot(&dist)?;
    stamped_in_place(&scratch, &dist)?;
    expected.insert(MANIFEST.to_owned(), GOLDEN.as_bytes().to_vec());
    assert_eq!(snapshot(&dist)?, expected);
    Ok(())
}

fn stamped_in_place(scratch: &Scratch, dist: &Path) -> TestResult {
    assert_output(
        &ui("stamp", dist, SOURCE, TAG).run(scratch.path())?,
        0,
        "",
        "",
    );
    Ok(())
}

#[test]
fn migration_prepared_inputs_ui_rejects_changed_source_or_tag() -> TestResult {
    let scratch = Scratch::new("ui-identity")?;
    let dist = stamped(&scratch)?;
    for (source, tag) in [(&"b".repeat(40)[..], TAG), (SOURCE, "v0.74.0")] {
        let output = ui("verify", &dist, source, tag).run(scratch.path())?;
        assert_output(&output, 1, "", MISMATCH);
    }
    Ok(())
}

#[test]
fn migration_prepared_inputs_ui_rejects_checksum_drift_and_extra_files() -> TestResult {
    let scratch = Scratch::new("ui-drift")?;
    let dist = stamped(&scratch)?;
    fs::write(dist.join("assets/app.js"), "changed")?;
    assert_output(
        &ui("verify", &dist, SOURCE, TAG).run(scratch.path())?,
        1,
        "",
        MISMATCH,
    );
    stamped_in_place(&scratch, &dist)?;
    fs::write(dist.join("assets/.unexpected.js"), "extra")?;
    assert_output(
        &ui("verify", &dist, SOURCE, TAG).run(scratch.path())?,
        1,
        "",
        MISMATCH,
    );
    Ok(())
}

#[test]
fn migration_prepared_inputs_ui_requires_a_local_module_reference() -> TestResult {
    let expected = "UI index must reference a built local JavaScript module\n";
    for index in [
        "<html></html>",
        r#"<script type="text/javascript" src="/assets/app.js"></script>"#,
        r#"<!-- <script type="module" src="/assets/app.js"></script> -->"#,
        r#"<script type="module" src="/assets/missing.js"></script>"#,
        r#"<script type="module" src="https://cdn.example/assets/app.js"></script>"#,
    ] {
        let scratch = Scratch::new("ui-module")?;
        let dist = raw_dist(&scratch)?;
        fs::write(dist.join("index.html"), index)?;
        let output = ui("stamp", &dist, SOURCE, TAG).run(scratch.path())?;
        assert_output(&output, 1, "", expected);
        assert!(!dist.join(MANIFEST).exists(), "{index}");
    }
    Ok(())
}

#[test]
fn migration_prepared_inputs_ui_parses_module_tags_like_html_parser() -> TestResult {
    let scratch = Scratch::new("ui-html")?;
    let dist = raw_dist(&scratch)?;
    let index = "<!doctype html><SCRIPT>var a = '<script type=\"module\">';</SCRIPT>\
                 <Script TYPE=module src='assets/app.js' defer></Script>";
    fs::write(dist.join("index.html"), index)?;
    assert_output(
        &ui("stamp", &dist, SOURCE, TAG).run(scratch.path())?,
        0,
        "",
        "",
    );
    Ok(())
}

/// Python raises `AttributeError` (a traceback, status 1) on a valueless
/// `src`; the port reports it as a rejection with the same status.
#[test]
fn migration_prepared_inputs_ui_rejects_valueless_module_src() -> TestResult {
    let scratch = Scratch::new("ui-bare-src")?;
    let dist = raw_dist(&scratch)?;
    fs::write(
        dist.join("index.html"),
        r#"<script type="module" src></script>"#,
    )?;
    let output = ui("stamp", &dist, SOURCE, TAG)
        .status_only()
        .run(scratch.path())?;
    assert_eq!(output.status.code(), Some(1));
    assert!(!dist.join(MANIFEST).exists());
    Ok(())
}

#[test]
fn migration_prepared_inputs_ui_rejects_missing_manifest_index_and_directory() -> TestResult {
    let scratch = Scratch::new("ui-missing")?;
    let dist = raw_dist(&scratch)?;
    let missing = "UI distribution is missing its release manifest\n";
    assert_output(
        &ui("verify", &dist, SOURCE, TAG).run(scratch.path())?,
        1,
        "",
        missing,
    );
    fs::remove_file(dist.join("index.html"))?;
    let missing_index = "UI distribution is missing index.html\n";
    assert_output(
        &ui("stamp", &dist, SOURCE, TAG).run(scratch.path())?,
        1,
        "",
        missing_index,
    );
    let absent = scratch.join("absent");
    let not_dir = "UI distribution must be a real directory\n";
    assert_output(
        &ui("stamp", &absent, SOURCE, TAG).run(scratch.path())?,
        1,
        "",
        not_dir,
    );
    Ok(())
}

#[test]
fn migration_prepared_inputs_ui_checks_identity_shape_first() -> TestResult {
    let scratch = Scratch::new("ui-shape")?;
    let absent = scratch.join("absent");
    let sha = "source SHA must be 40 lowercase hexadecimal characters\n";
    let tag = "release tag must be a versioned v-prefixed tag\n";
    for (source, release, expected) in [
        ("main", TAG, sha),
        (&"A".repeat(40)[..], TAG, sha),
        (SOURCE, "main", tag),
        (SOURCE, "v1.2", tag),
        (SOURCE, "v1.2.3-", tag),
    ] {
        let output = ui("stamp", &absent, source, release).run(scratch.path())?;
        assert_output(&output, 1, "", expected);
    }
    Ok(())
}

#[cfg(unix)]
#[test]
fn migration_prepared_inputs_ui_rejects_symlinks() -> TestResult {
    let scratch = Scratch::new("ui-symlink")?;
    let dist = raw_dist(&scratch)?;
    crate::support::symlink(&dist.join("assets/app.js"), &dist.join("assets/link.js"))?;
    let expected = "UI distribution must not contain symbolic links\n";
    assert_output(
        &ui("stamp", &dist, SOURCE, TAG).run(scratch.path())?,
        1,
        "",
        expected,
    );
    let linked = scratch.join("linked");
    crate::support::symlink(&scratch.join("dist"), &linked)?;
    let not_dir = "UI distribution must be a real directory\n";
    assert_output(
        &ui("verify", &linked, SOURCE, TAG).run(scratch.path())?,
        1,
        "",
        not_dir,
    );
    Ok(())
}

#[test]
fn migration_prepared_inputs_ui_rejects_malformed_manifest_json() -> TestResult {
    let scratch = Scratch::new("ui-json")?;
    let dist = stamped(&scratch)?;
    fs::write(dist.join(MANIFEST), "{not json")?;
    let output = ui("verify", &dist, SOURCE, TAG)
        .status_only()
        .run(scratch.path())?;
    assert_eq!(output.status.code(), Some(1));
    assert!(text(&output.stderr).starts_with("UI release manifest is not valid JSON: "));
    Ok(())
}

#[test]
fn migration_prepared_inputs_ui_usage_errors_exit_two() -> TestResult {
    let scratch = Scratch::new("ui-usage")?;
    for args in [
        &["publish", "--dist", "d"][..],
        &["verify", "--dist", "d"][..],
    ] {
        let output = Case::same(&["ui-distribution"], args, SCRIPT)
            .status_only()
            .run(scratch.path())?;
        assert_eq!(output.status.code(), Some(2), "{args:?}");
    }
    Ok(())
}
