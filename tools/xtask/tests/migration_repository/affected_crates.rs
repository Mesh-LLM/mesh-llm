//! `repository affected-crates` parity with `scripts/affected-crates.sh`.

use crate::support::{
    Invocation, Legacy, LegacyKind, Scratch, TestResult, assert_output, repository_root, text,
};
use std::path::Path;

const SCRIPT: &str = "scripts/affected-crates.sh";

/// alpha <- nested (normal), alpha <- beta (normal), beta <- gamma (normal),
/// alpha <- gamma (dev). `nested` lives inside alpha's directory.
fn fixture_workspace(scratch: &Scratch) -> TestResult {
    scratch.write(
        "Cargo.toml",
        "[workspace]\nresolver = \"2\"\nmembers = [\"crates/alpha\", \"crates/alpha/nested\", \"crates/beta\", \"crates/gamma\", \"tools/xtask\"]\n",
    )?;
    let crates = [
        ("crates/alpha", "alpha", ""),
        (
            "crates/alpha/nested",
            "nested",
            "alpha = { path = \"..\" }\n",
        ),
        ("crates/beta", "beta", "alpha = { path = \"../alpha\" }\n"),
        (
            "crates/gamma",
            "gamma",
            "beta = { path = \"../beta\" }\n\n[dev-dependencies]\nalpha = { path = \"../alpha\" }\n",
        ),
        ("tools/xtask", "xtask", ""),
    ];
    for (dir, name, deps) in crates {
        scratch.write(
            &format!("{dir}/Cargo.toml"),
            &format!(
                "[package]\nname = \"{name}\"\nversion = \"0.1.0\"\nedition = \"2021\"\n\n[dependencies]\n{deps}"
            ),
        )?;
        scratch.write(&format!("{dir}/src/lib.rs"), "")?;
    }
    Ok(())
}

fn affected(
    cwd: &Path,
    args: &[&str],
    stdin: &str,
) -> Result<std::process::Output, Box<dyn std::error::Error>> {
    let mut ported = vec!["repository", "affected-crates"];
    ported.extend_from_slice(args);
    Invocation {
        cwd,
        args: &ported,
        stdin: Some(stdin),
        env: &[],
    }
    .run_with_legacy(Legacy {
        kind: LegacyKind::Bash,
        script: SCRIPT,
        args,
    })
}

fn pretty(affected: &[&str], test_crates: &[&str], ui: bool, website: bool) -> String {
    let list = |names: &[&str]| {
        if names.is_empty() {
            return "[]".to_owned();
        }
        let items = names
            .iter()
            .map(|name| format!("    \"{name}\""))
            .collect::<Vec<_>>()
            .join(",\n");
        format!("[\n{items}\n  ]")
    };
    format!(
        "{{\n  \"affected\": {},\n  \"test_crates\": {},\n  \"all_rust\": false,\n  \"ui_changed\": {ui},\n  \"website_changed\": {website}\n}}\n",
        list(affected),
        list(test_crates)
    )
}

/// The all-workspace document, built from the legacy script's own list.
fn all_workspace(ui: bool, website: bool) -> Result<String, Box<dyn std::error::Error>> {
    let script = std::fs::read_to_string(repository_root().join(SCRIPT))?;
    let (_, rest) = script
        .split_once("WORKSPACE_MEMBERS=(\n")
        .ok_or("missing member list")?;
    let (body, _) = rest.split_once("\n)").ok_or("unterminated member list")?;
    let members = body.lines().map(str::trim).collect::<Vec<_>>().join(",");
    Ok(format!(
        "{{\n  \"affected\": [{members}],\n  \"test_crates\": [],\n  \"all_rust\": true,\n  \"ui_changed\": {ui},\n  \"website_changed\": {website}\n}}\n"
    ))
}

#[test]
fn migration_repository_affected_direct_crate_plus_reverse_deps() -> TestResult {
    // Given: a workspace where alpha is depended on by nested, beta and gamma.
    let scratch = Scratch::new("affected-closure")?;
    fixture_workspace(&scratch)?;
    // When: alpha, a UI file, a doc and the nested crate change (alpha twice).
    let output = affected(
        scratch.path(),
        &["--stdin"],
        "crates/alpha/src/lib.rs\ncrates/mesh-llm-ui/src/app.tsx\ndocs/guide.md\ncrates/alpha/nested/src/lib.rs\ncrates/alpha/src/lib.rs\n",
    )?;
    // Then: owners by longest prefix, then breadth-first reverse closure.
    assert_output(
        &output,
        0,
        &pretty(
            &["alpha", "nested", "beta", "gamma"],
            &["alpha", "nested"],
            true,
            false,
        ),
        "",
    );
    Ok(())
}

#[test]
fn migration_repository_affected_mid_graph_change_skips_upstream() -> TestResult {
    // Given: the fixture workspace.
    let scratch = Scratch::new("affected-mid")?;
    fixture_workspace(&scratch)?;
    // When: beta changes, and an unterminated final gamma line follows.
    let output = affected(
        scratch.path(),
        &["--stdin"],
        "crates/beta/src/lib.rs\ncrates/gamma/src/lib.rs",
    )?;
    // Then: only beta and its dependents; the unterminated line is dropped.
    assert_output(
        &output,
        0,
        &pretty(&["beta", "gamma"], &["beta"], false, false),
        "",
    );
    Ok(())
}

#[test]
fn migration_repository_affected_positional_files_and_website() -> TestResult {
    // Given: the fixture workspace.
    let scratch = Scratch::new("affected-positional")?;
    fixture_workspace(&scratch)?;
    // When: files are passed as arguments, including a website input.
    let output = affected(
        scratch.path(),
        &["tools/xtask/src/main.rs", "install.ps1"],
        "",
    )?;
    // Then: tools/ crates are owned and the website flag is raised.
    assert_output(&output, 0, &pretty(&["xtask"], &["xtask"], false, true), "");
    Ok(())
}

#[test]
fn migration_repository_affected_empty_input_is_empty() -> TestResult {
    // Given: the fixture workspace.
    let scratch = Scratch::new("affected-empty")?;
    fixture_workspace(&scratch)?;
    // When: nothing changed.
    let output = affected(scratch.path(), &["--stdin"], "")?;
    // Then: an empty non-escalated selection.
    assert_output(&output, 0, &pretty(&[], &[], false, false), "");
    Ok(())
}

#[test]
fn migration_repository_affected_escalation_selects_whole_workspace() -> TestResult {
    // Given: the fixture workspace (escalation never consults it).
    let scratch = Scratch::new("affected-escalate")?;
    fixture_workspace(&scratch)?;
    // When: Cargo.lock changes alongside UI and website inputs.
    let output = affected(
        scratch.path(),
        &["--stdin"],
        "Cargo.lock\ncrates/mesh-llm-ui/a\nwebsite/index.md\n",
    )?;
    // Then: the hardcoded all-workspace list with both flags preserved.
    assert_output(&output, 0, &all_workspace(true, true)?, "");
    Ok(())
}

#[test]
fn migration_repository_affected_malformed_manifest_falls_back() -> TestResult {
    // Given: a workspace whose manifest Cargo cannot parse.
    let scratch = Scratch::new("affected-malformed")?;
    scratch.write("Cargo.toml", "[workspace\n")?;
    // When: a crate file changes.
    let output = affected(
        scratch.path(),
        &["--stdin"],
        "crates/a/src/lib.rs\ncrates/mesh-llm-ui/x\n",
    )?;
    // Then: the fail-open warning carries Cargo's status and all crates run.
    assert_output(
        &output,
        0,
        &all_workspace(true, false)?,
        "WARNING: affected-crates.sh encountered an error (exit=101), falling back to all_rust=true\n",
    );
    Ok(())
}

#[cfg(unix)]
#[test]
fn migration_repository_affected_unparseable_metadata_falls_back() -> TestResult {
    use std::os::unix::fs::PermissionsExt;
    // Given: a `cargo` on PATH whose metadata output is not JSON.
    let scratch = Scratch::new("affected-not-json")?;
    let stub = scratch.write("bin/cargo", "#!/bin/sh\nprintf 'not json\\n'\n")?;
    std::fs::set_permissions(&stub, std::fs::Permissions::from_mode(0o755))?;
    let path = format!("{}:/usr/bin:/bin", scratch.path().join("bin").display());
    // When: a crate file changes.
    let output = Invocation {
        cwd: scratch.path(),
        args: &["repository", "affected-crates", "--stdin"],
        stdin: Some("crates/a/src/lib.rs\n"),
        env: &[("PATH", &path)],
    }
    .run()?;
    // Then: the JSON-parse failure takes the fallback with jq's status 5.
    assert_eq!(text(&output.stdout), all_workspace(false, false)?);
    assert!(
        text(&output.stderr).ends_with(
            "WARNING: affected-crates.sh encountered an error (exit=5), falling back to all_rust=true\n"
        ),
        "{}",
        text(&output.stderr)
    );
    assert_eq!(output.status.code(), Some(0));
    Ok(())
}
