use super::*;
use std::sync::atomic::{AtomicU64, Ordering};
static NEXT: AtomicU64 = AtomicU64::new(0);
struct Fixture(PathBuf);
impl Fixture {
    fn new() -> Self {
        let path = env::temp_dir().join(format!(
            "eval-pin-fixture-{}-{}-{}",
            std::process::id(),
            unix_millis().unwrap(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir(&path).unwrap();
        Self(path)
    }
    fn repo(&self, name: &str) -> PathBuf {
        let path = self.0.join(name);
        fs::create_dir(&path).unwrap();
        checked(&path, &["init"]);
        checked(&path, &["config", "user.email", "fixture@example.invalid"]);
        checked(&path, &["config", "user.name", "Fixture"]);
        fs::write(path.join("source.py"), "original\n").unwrap();
        checked(&path, &["add", "."]);
        checked(&path, &["commit", "-m", "finite fixture"]);
        path
    }
}
impl Drop for Fixture {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}
fn checked(path: &Path, args: &[&str]) -> Vec<u8> {
    git(path, args).unwrap()
}
fn head(path: &Path) -> String {
    String::from_utf8(checked(path, &["rev-parse", "HEAD"]))
        .unwrap()
        .trim()
        .into()
}
#[test]
fn finite_candidate_pins_and_sync_fetch_head_are_exact() {
    assert_eq!(
        registry::definition(EvalId::McpAtlas).repo_ref,
        "b290e672645791fea0bcb23e2c0f4fec50715cca"
    );
    assert_eq!(
        registry::definition(EvalId::SweBenchPro).repo_ref,
        "66f92766bba642462d4bbe5479e83f91f9211862"
    );
    assert_eq!(
        registry::SWE_AGENT_REF,
        "402a7b8fdac8193f3f255bb53859ba274234f596"
    );
    assert_eq!(
        registry::MCP_ATLAS_IMAGE,
        "ghcr.io/scaleapi/mcp-atlas@sha256:415a532f1aeae911fbe2d337cde0657c345a937fab295989f6ef8c70c09c740f"
    );
    for id in [EvalId::McpAtlas, EvalId::SweBenchPro] {
        let definition = registry::definition(id);
        let new = super::super::sync::new_eval_repo_sync_steps(Path::new("fixture"), definition);
        assert_eq!(new[0].args, ["clone", definition.repo_url, "fixture"]);
        assert_eq!(new[1].args.last().unwrap(), definition.repo_ref);
        assert_eq!(new[2].args.last().unwrap(), "FETCH_HEAD");
        let steps =
            super::super::sync::existing_repo_sync_steps(Path::new("fixture"), definition.repo_ref);
        assert_eq!(steps[0].args.last().unwrap(), definition.repo_ref);
        assert_eq!(
            steps[1].args,
            ["-C", "fixture", "checkout", "--detach", "FETCH_HEAD"]
        );
    }
}
#[test]
fn exact_head_and_tracked_source_are_required_even_when_ignore_config_is_set() {
    let fixture = Fixture::new();
    let repo = fixture.repo("root");
    let pin = head(&repo);
    assert_eq!(admit(&repo, &pin, None).unwrap(), pin);
    assert!(admit(&repo, "0000000000000000000000000000000000000000", None).is_err());
    fs::write(repo.join("source.py"), "changed\n").unwrap();
    assert!(admit(&repo, &pin, None).is_err());
    checked(&repo, &["update-index", "--assume-unchanged", "source.py"]);
    assert!(admit(&repo, &pin, None).is_err());
    checked(
        &repo,
        &["update-index", "--no-assume-unchanged", "source.py"],
    );
    checked(&repo, &["add", "source.py"]);
    assert!(admit(&repo, &pin, None).is_err());
}
#[test]
fn recursive_gitlink_checkout_dirty_source_and_uninitialized_submodule_refuse() {
    let fixture = Fixture::new();
    let nested = fixture.repo("nested");
    let agent = fixture.repo("agent");
    checked(
        &agent,
        &[
            "-c",
            "protocol.file.allow=always",
            "submodule",
            "add",
            nested.to_str().unwrap(),
            "nested",
        ],
    );
    checked(&agent, &["commit", "-am", "nested link"]);
    let agent_pin = head(&agent);
    let root = fixture.repo("root");
    checked(
        &root,
        &[
            "-c",
            "protocol.file.allow=always",
            "submodule",
            "add",
            agent.to_str().unwrap(),
            "SWE-agent",
        ],
    );
    checked(&root, &["commit", "-am", "agent link"]);
    checked(
        &root,
        &[
            "-c",
            "protocol.file.allow=always",
            "submodule",
            "update",
            "--init",
            "--recursive",
        ],
    );
    let pin = head(&root);
    assert!(admit(&root, &pin, Some(&agent_pin)).is_ok());
    assert!(
        admit(
            &root,
            &pin,
            Some("0000000000000000000000000000000000000000")
        )
        .is_err()
    );
    checked(&root, &["config", "submodule.SWE-agent.ignore", "all"]);
    checked(
        &root.join("SWE-agent"),
        &["config", "submodule.nested.ignore", "all"],
    );
    let child = root.join("SWE-agent/nested");
    fs::write(child.join("untracked-output.json"), "harmless output\n").unwrap();
    assert!(admit(&root, &pin, Some(&agent_pin)).is_ok());
    fs::write(child.join("source.py"), "nested changed\n").unwrap();
    assert!(admit(&root, &pin, Some(&agent_pin)).is_err());
    checked(&child, &["checkout", "--", "source.py"]);
    checked(&child, &["config", "user.email", "fixture@example.invalid"]);
    checked(&child, &["config", "user.name", "Fixture"]);
    checked(
        &child,
        &["commit", "--allow-empty", "-m", "wrong nested HEAD"],
    );
    assert!(admit(&root, &pin, Some(&agent_pin)).is_err());
    checked(&root, &["submodule", "deinit", "--force", "--all"]);
    assert!(admit(&root, &pin, Some(&agent_pin)).is_err());
}
fn run_args(id: EvalId, root: &Path, output: &Path) -> EvalRunArgs {
    EvalRunArgs {
        eval: id,
        base_url: "http://127.0.0.1:1/v1".into(),
        model: "fixture".into(),
        api_key: "unused".into(),
        task_id: None,
        dataset: "lite".into(),
        agent: "terminus-2".into(),
        harbor_endpoint_url: None,
        session_id: None,
        cacheline_state: None,
        cache_root: Some(root.into()),
        output_dir: Some(output.into()),
        timeout_secs: 1,
        harness_timeout_secs: Some(1),
        endpoint_concurrency: 1,
        run_id: None,
        metrics_http: "http://127.0.0.1:1".into(),
        metrics_run_id: None,
        metrics_finalize_only: false,
        dry_run: false,
    }
}
#[test]
fn actual_run_refuses_wrong_source_before_template_metrics_or_child_effects() {
    let fixture = Fixture::new();
    for id in [EvalId::McpAtlas, EvalId::SweBenchPro] {
        let definition = registry::definition(id);
        let harness = harness_dir(&fixture.0, definition);
        fs::create_dir_all(harness.parent().unwrap()).unwrap();
        let local = fixture.repo(definition.cache_name);
        fs::rename(local, &harness).unwrap();
        let output = fixture.0.join(format!("output-{}", id.as_str()));
        let error = super::super::run::run_eval(run_args(id, &fixture.0, &output))
            .unwrap_err()
            .to_string();
        assert!(error.contains("HEAD differs from immutable pin"), "{error}");
        assert!(!output.exists());
    }
}

#[test]
fn mcp_rendered_consumer_uses_same_immutable_image_without_stale_alias() {
    let fixture = Fixture::new();
    let output = fixture.0.join("render");
    fs::create_dir_all(output.join("raw")).unwrap();
    let args = run_args(EvalId::McpAtlas, &fixture.0, &output);
    super::super::adapters::run_command(
        registry::definition(EvalId::McpAtlas),
        &args,
        &fixture.0,
        &output,
    )
    .unwrap();
    let script = fs::read_to_string(output.join("raw/mcp-atlas-run.sh")).unwrap();
    assert!(script.contains(registry::MCP_ATLAS_IMAGE));
    assert!(!script.contains("agent-environment:latest"));
    assert!(!script.contains("mcp-atlas:1.2.5"));
}

#[test]
fn replacement_objects_cannot_admit_a_substituted_tree_under_the_expected_head() {
    let fixture = Fixture::new();
    let repo = fixture.repo("replacement");
    let pin = head(&repo);
    fs::write(repo.join("source.py"), "substituted\n").unwrap();
    checked(&repo, &["add", "source.py"]);
    checked(&repo, &["commit", "-m", "substituted tree"]);
    let substitute = head(&repo);
    checked(&repo, &["replace", &pin, &substitute]);
    // Normal Git now sees the replacement tree with the original HEAD identity.
    let ordinary = Command::new("git")
        .env_remove("GIT_NO_REPLACE_OBJECTS")
        .args(["-C", repo.to_str().unwrap(), "checkout", "--detach", &pin])
        .output()
        .unwrap();
    assert!(ordinary.status.success());
    assert_eq!(
        fs::read_to_string(repo.join("source.py")).unwrap(),
        "substituted\n"
    );
    assert_eq!(head(&repo), pin);
    assert!(admit(&repo, &pin, None).is_err());
    checked(&repo, &["checkout", "--force", &pin]);
    assert_eq!(
        fs::read_to_string(repo.join("source.py")).unwrap(),
        "original\n"
    );
    assert!(admit(&repo, &pin, None).is_ok());
}
#[cfg(unix)]
#[test]
fn inherited_stdout_writer_after_direct_exit_does_not_block_capture() {
    let fixture = Fixture::new();
    let pid = fixture.0.join("writer.pid");
    let mut command = Command::new("/bin/bash");
    command
        .args([
            "-c",
            "sleep 30 & printf '%s' $! > \"$1\"; printf ready",
            "fixture",
        ])
        .arg(&pid);
    let start = Instant::now();
    assert_eq!(
        capture_git(command, Duration::from_secs(2)).unwrap(),
        b"ready"
    );
    assert!(start.elapsed() < Duration::from_secs(5));
    let writer: libc::pid_t = fs::read_to_string(pid).unwrap().parse().unwrap();
    // The group was signalled even though the direct shell had already exited.
    // A non-child zombie can remain briefly; its inherited writer no longer runs.
    let observed = Instant::now();
    loop {
        let status = Command::new("ps")
            .args(["-o", "stat=", "-p", &writer.to_string()])
            .output()
            .unwrap();
        let state = String::from_utf8(status.stdout).unwrap();
        if state.trim().is_empty() || state.trim().starts_with('Z') {
            break;
        }
        assert!(
            observed.elapsed() < Duration::from_secs(1),
            "writer remains live: {state}"
        );
        thread::sleep(Duration::from_millis(10));
    }
}

#[cfg(unix)]
#[test]
fn capture_disappearance_error_cleans_the_live_owned_child_before_returning() {
    let fixture = Fixture::new();
    let pid = fixture.0.join("live.pid");
    let mut command = Command::new("/bin/bash");
    command
        .args(["-c", "printf '%s' $$ > \"$1\"; sleep 30", "fixture"])
        .arg(&pid);
    let error = capture_git_observed(command, Duration::from_secs(3), |output| {
        let until = Instant::now() + Duration::from_secs(1);
        let live = loop {
            if let Some(live) = fs::read_to_string(&pid)
                .ok()
                .and_then(|value| value.parse::<libc::pid_t>().ok())
            {
                break live;
            }
            if Instant::now() >= until {
                bail!("fixture child did not start");
            }
            thread::sleep(Duration::from_millis(5));
        };
        // SAFETY: signal zero observes this fixture PID without modifying it.
        assert_eq!(unsafe { libc::kill(live, 0) }, 0);
        fs::remove_file(output)?;
        Ok(())
    })
    .unwrap_err();
    assert_eq!(
        error.downcast_ref::<std::io::Error>().unwrap().kind(),
        std::io::ErrorKind::NotFound
    );
    let owned: libc::pid_t = fs::read_to_string(pid).unwrap().parse().unwrap();
    // SAFETY: signal zero observes the already reaped fixture child.
    assert_eq!(unsafe { libc::kill(owned, 0) }, -1);
    assert_eq!(
        std::io::Error::last_os_error().raw_os_error(),
        Some(libc::ESRCH)
    );
}
#[cfg(unix)]
#[test]
fn private_sdk_clone_preserves_exact_commit_without_parent_worktree_or_untracked_files() {
    let fixture = Fixture::new();
    let source = fixture.repo("source");
    let expected = String::from_utf8(checked(&source, &["rev-parse", "HEAD"]))
        .unwrap()
        .trim()
        .to_owned();
    fs::write(source.join("ambient.py"), b"untracked import").unwrap();
    let destination = fixture.0.join("private-agent");
    clone_snapshot_expected(
        &source,
        &destination,
        Instant::now() + Duration::from_secs(30),
        &expected,
    )
    .unwrap();
    assert_eq!(
        checked(&destination, &["rev-parse", "HEAD"]),
        format!("{expected}\n").as_bytes()
    );
    assert!(destination.join(".git").is_dir());
    assert!(!destination.join("ambient.py").exists());
    assert_eq!(
        fs::read(destination.join("source.py")).unwrap(),
        b"original\n"
    );
    fs::write(source.join("source.py"), b"tracked drift").unwrap();
    assert!(
        clone_snapshot_expected(
            &source,
            &fixture.0.join("refused"),
            Instant::now() + Duration::from_secs(30),
            &expected
        )
        .is_err()
    );
    assert!(!fixture.0.join("refused").exists());
}
#[cfg(unix)]
#[test]
fn sdk_source_mode_drift_is_not_hidden_by_local_core_filemode_false() {
    use std::os::unix::fs::PermissionsExt;
    let fixture = Fixture::new();
    let source = fixture.repo("mode");
    let expected = String::from_utf8(checked(&source, &["rev-parse", "HEAD"]))
        .unwrap()
        .trim()
        .to_owned();
    checked(&source, &["config", "core.filemode", "false"]);
    fs::set_permissions(source.join("source.py"), fs::Permissions::from_mode(0o755)).unwrap();
    assert!(admit(&source, &expected, None).is_err());
}
#[cfg(unix)]
#[test]
fn exact_parent_metadata_capture_succeeds_above_one_mib_and_refuses_above_two() {
    // Causal finite producer: this exercises capture bounds, not a fabricated pinned checkout.
    let command = || {
        let mut c = Command::new("/bin/sh");
        c.args(["-c", "head -c 1500000 /dev/zero"]);
        c
    };
    assert!(capture_git(command(), Duration::from_secs(3)).is_err());
    assert_eq!(
        capture_git_limit(command(), Duration::from_secs(3), 2 * 1048576)
            .unwrap()
            .len(),
        1500000
    );
    let mut overflow = Command::new("/bin/sh");
    overflow.args(["-c", "head -c 2097153 /dev/zero"]);
    assert!(capture_git_limit(overflow, Duration::from_secs(3), 2 * 1048576).is_err());
    assert!(source_metadata(Path::new("/unused"), &["status", "--porcelain"], true).is_err());
}
