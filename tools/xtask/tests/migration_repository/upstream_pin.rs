//! `repository llama-upstream-pin` parity with
//! `scripts/check-llama-upstream-pin.py`.

use crate::support::{
    Invocation, Legacy, LegacyKind, Scratch, TestResult, assert_output, commit_all, git, git_init,
    text,
};
use std::path::Path;
use std::process::Output;

const SCRIPT: &str = "scripts/check-llama-upstream-pin.py";
const PIN_PATH: &str = "third_party/llama.cpp/upstream.txt";

struct Upstream {
    base: String,
    forward: String,
    latest: String,
}

fn create_upstream(repo: &Path) -> Result<Upstream, Box<dyn std::error::Error>> {
    git_init(repo)?;
    std::fs::write(repo.join("history.txt"), "base\n")?;
    let base = commit_all(repo, "base")?;
    std::fs::write(repo.join("history.txt"), "base\nforward\n")?;
    let forward = commit_all(repo, "forward")?;
    std::fs::write(repo.join("history.txt"), "base\nforward\nlatest\n")?;
    let latest = commit_all(repo, "latest")?;
    Ok(Upstream {
        base,
        forward,
        latest,
    })
}

fn write_pin(mesh: &Path, pin: &str) -> TestResult {
    let path = mesh.join(PIN_PATH);
    std::fs::create_dir_all(path.parent().ok_or("pin parent")?)?;
    std::fs::write(path, format!("{pin}\n"))?;
    Ok(())
}

/// Linear mesh history: base commit pins `base_pin`, head pins `proposed`.
fn mesh_history(
    mesh: &Path,
    base_pin: &str,
    proposed: &str,
) -> Result<(String, String), Box<dyn std::error::Error>> {
    git_init(mesh)?;
    write_pin(mesh, base_pin)?;
    let base = commit_all(mesh, "base")?;
    write_pin(mesh, proposed)?;
    std::fs::write(mesh.join("change.txt"), "proposed\n")?;
    let head = commit_all(mesh, "propose pin")?;
    Ok((base, head))
}

fn guard(scratch: &Scratch, base: &str, head: &str) -> Result<Output, Box<dyn std::error::Error>> {
    let mesh = scratch.path().join("mesh");
    let upstream = scratch.path().join("upstream");
    let mesh = mesh.to_str().ok_or("non-UTF8 mesh")?;
    let upstream = upstream.to_str().ok_or("non-UTF8 upstream")?;
    let args = ["--repository", mesh, "--upstream-url", upstream, base, head];
    let mut ported = vec!["repository", "llama-upstream-pin"];
    ported.extend_from_slice(&args);
    Invocation {
        cwd: scratch.path(),
        args: &ported,
        stdin: None,
        env: &[],
    }
    .run_with_legacy(Legacy {
        kind: LegacyKind::Python,
        script: SCRIPT,
        args: &args,
    })
}

fn header(merge_base: &str, base_pin: &str, proposed: &str) -> String {
    format!(
        "PR merge-base commit:   {merge_base}\nmerge-base llama.cpp pin: {base_pin}\nproposed llama.cpp pin: {proposed}\n"
    )
}

#[test]
fn migration_repository_pin_unchanged_and_forward_pass() -> TestResult {
    // Given: an upstream with three linear commits.
    let scratch = Scratch::new("pin-pass")?;
    let pins = create_upstream(&scratch.path().join("upstream"))?;
    let mesh = scratch.path().join("mesh");
    // When: the PR moves the pin forward.
    let (base, head) = mesh_history(&mesh, &pins.base, &pins.latest)?;
    let forward = guard(&scratch, &base, &head)?;
    // Then: the three identity lines and a forward verdict on stdout.
    let expected =
        header(&base, &pins.base, &pins.latest) + "llama.cpp upstream pin moves forward\n";
    assert_output(&forward, 0, &expected, "");
    // When: a follow-up commit keeps the pin.
    std::fs::write(mesh.join("change.txt"), "again\n")?;
    let next = commit_all(&mesh, "unrelated")?;
    let unchanged = guard(&scratch, &head, &next)?;
    let expected =
        header(&head, &pins.latest, &pins.latest) + "llama.cpp upstream pin is unchanged\n";
    assert_output(&unchanged, 0, &expected, "");
    Ok(())
}

#[test]
fn migration_repository_pin_backward_is_rejected() -> TestResult {
    let scratch = Scratch::new("pin-backward")?;
    let pins = create_upstream(&scratch.path().join("upstream"))?;
    let (base, head) = mesh_history(&scratch.path().join("mesh"), &pins.latest, &pins.forward)?;
    let output = guard(&scratch, &base, &head)?;
    assert_output(
        &output,
        1,
        &header(&base, &pins.latest, &pins.forward),
        &format!(
            "ERROR: PR moves third_party/llama.cpp/upstream.txt backward: {} is an ancestor of the base pin {}\n",
            pins.forward, pins.latest
        ),
    );
    Ok(())
}

#[test]
fn migration_repository_pin_stale_pr_compares_merge_base() -> TestResult {
    // Given: target advanced the pin after the PR branched; PR keeps old pin.
    let scratch = Scratch::new("pin-stale")?;
    let pins = create_upstream(&scratch.path().join("upstream"))?;
    let mesh = scratch.path().join("mesh");
    git_init(&mesh)?;
    write_pin(&mesh, &pins.base)?;
    let merge_base = commit_all(&mesh, "merge base")?;
    git(&mesh, &["checkout", "--quiet", "-b", "pr"])?;
    std::fs::write(mesh.join("pr.txt"), "pr\n")?;
    let head = commit_all(&mesh, "PR change")?;
    git(&mesh, &["checkout", "--quiet", "main"])?;
    write_pin(&mesh, &pins.forward)?;
    let base = commit_all(&mesh, "target advances pin")?;
    // When: guarded against the advanced target.
    let output = guard(&scratch, &base, &head)?;
    // Then: the merge-base pin is used, so the PR is unchanged.
    let expected =
        header(&merge_base, &pins.base, &pins.base) + "llama.cpp upstream pin is unchanged\n";
    assert_output(&output, 0, &expected, "");
    Ok(())
}

#[test]
fn migration_repository_pin_unknown_upstream_fails_closed() -> TestResult {
    let scratch = Scratch::new("pin-missing-upstream")?;
    let pins = create_upstream(&scratch.path().join("upstream"))?;
    let missing = "f".repeat(40);
    let (base, head) = mesh_history(&scratch.path().join("mesh"), &pins.base, &missing)?;
    let output = guard(&scratch, &base, &head)?;
    assert_eq!(output.status.code(), Some(1));
    assert_eq!(text(&output.stdout), header(&base, &pins.base, &missing));
    let stderr = text(&output.stderr);
    assert!(
        stderr.starts_with("ERROR: unable to fetch both llama.cpp upstream pins; the guard cannot prove ancestry and will fail closed: "),
        "{stderr}"
    );
    Ok(())
}

#[test]
fn migration_repository_pin_missing_base_and_bad_revisions() -> TestResult {
    let scratch = Scratch::new("pin-missing-base")?;
    let pins = create_upstream(&scratch.path().join("upstream"))?;
    let (_, head) = mesh_history(&scratch.path().join("mesh"), &pins.base, &pins.latest)?;
    // When: the base revision is absent from the checkout (incomplete history).
    let absent = "a".repeat(40);
    let output = guard(&scratch, &absent, &head)?;
    assert_output(
        &output,
        1,
        "",
        &format!("ERROR: repository is missing {absent}; cannot inspect PR pin\n"),
    );
    // When: the base revision is not a full lowercase SHA.
    let output = guard(&scratch, "HEAD", &head)?;
    assert_output(
        &output,
        1,
        "",
        "ERROR: base revision must be a lowercase 40-character SHA: 'HEAD'\n",
    );
    Ok(())
}

#[cfg(unix)]
#[test]
fn migration_repository_pin_rejects_symlink_and_invalid_pin() -> TestResult {
    let scratch = Scratch::new("pin-symlink")?;
    let pins = create_upstream(&scratch.path().join("upstream"))?;
    let mesh = scratch.path().join("mesh");
    let (base, _) = mesh_history(&mesh, &pins.latest, "not-a-sha")?;
    let head = git(&mesh, &["rev-parse", "HEAD"])?;
    // When: the proposed pin file content is not a SHA.
    let output = guard(&scratch, &base, &head)?;
    assert_output(
        &output,
        1,
        "",
        &format!("ERROR: head commit {head} has an invalid {PIN_PATH} value: 'not-a-sha'\n"),
    );
    // When: the pin path becomes a symlink to sibling content.
    std::fs::remove_file(mesh.join(PIN_PATH))?;
    std::fs::write(
        mesh.join("third_party/llama.cpp/real.txt"),
        format!("{}\n", pins.base),
    )?;
    std::os::unix::fs::symlink("real.txt", mesh.join(PIN_PATH))?;
    let link = commit_all(&mesh, "symlink pin")?;
    let output = guard(&scratch, &base, &link)?;
    assert_output(
        &output,
        1,
        "",
        &format!("ERROR: head commit {link} {PIN_PATH} must be a regular 100644 blob\n"),
    );
    Ok(())
}
