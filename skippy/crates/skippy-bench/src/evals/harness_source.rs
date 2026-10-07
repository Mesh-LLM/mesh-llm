//! Admit only the two fixed upstream SDK source capsules before external execution.
use super::{registry, *};
use std::io::Read;
const GIT_OUTPUT_LIMIT: u64 = 1024 * 1024;
#[cfg(test)]
#[path = "harness_source/tests.rs"]
mod tests;

struct GitCapture(PathBuf);
impl GitCapture {
    fn new() -> Result<Self> {
        use std::sync::atomic::{AtomicU64, Ordering};
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let path = env::temp_dir().join(format!(
            "skippy-harness-source-{}-{}-{}",
            std::process::id(),
            unix_millis()?,
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        let mut builder = fs::DirBuilder::new();
        #[cfg(unix)]
        {
            use std::os::unix::fs::DirBuilderExt;
            builder.mode(0o700);
        }
        builder
            .create(&path)
            .context("create private harness Git capture")?;
        Ok(Self(path))
    }
    fn output(&self) -> PathBuf {
        self.0.join("stdout")
    }
}
impl Drop for GitCapture {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}
fn git(harness: &Path, args: &[&str]) -> Result<Vec<u8>> {
    let mut command = snapshot_git();
    command
        .arg("-C")
        .arg(harness)
        .args(["-c", "core.fsmonitor=false", "-c", "core.filemode=true"])
        .args(args);
    for key in [
        "GIT_DIR",
        "GIT_WORK_TREE",
        "GIT_INDEX_FILE",
        "GIT_OBJECT_DIRECTORY",
        "GIT_ALTERNATE_OBJECT_DIRECTORIES",
        "GIT_CONFIG_COUNT",
        "GIT_CONFIG_PARAMETERS",
    ] {
        command.env_remove(key);
    }
    command
        .env("GIT_MASTER", "1")
        .env("GIT_TERMINAL_PROMPT", "0")
        .env("GIT_NO_REPLACE_OBJECTS", "1");
    capture_git_limit(command, Duration::from_secs(10), GIT_OUTPUT_LIMIT)
}
fn capture_git(command: Command, budget: Duration) -> Result<Vec<u8>> {
    capture_git_observed(command, budget, |_| Ok(()))
}
fn capture_git_observed(
    command: Command,
    budget: Duration,
    observe: impl FnOnce(&Path) -> Result<()>,
) -> Result<Vec<u8>> {
    capture_git_with_observer(command, budget, GIT_OUTPUT_LIMIT, observe)
}
fn capture_git_limit(command: Command, budget: Duration, limit: u64) -> Result<Vec<u8>> {
    capture_git_with_observer(command, budget, limit, |_| Ok(()))
}
fn capture_git_with_observer(
    mut command: Command,
    budget: Duration,
    limit: u64,
    observe: impl FnOnce(&Path) -> Result<()>,
) -> Result<Vec<u8>> {
    let capture = GitCapture::new()?;
    let output = fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(capture.output())?;
    command.stdout(Stdio::from(output)).stderr(Stdio::null());
    configure_child_group(&mut command);
    let mut child = command
        .spawn()
        .context("start finite harness Git admission")?;
    let deadline = Instant::now() + budget;
    let execution_deadline = deadline - budget.min(Duration::from_millis(500));
    let polled = observe(&capture.output())
        .and_then(|()| poll_git(&mut child, &capture.output(), execution_deadline, limit));
    // Always clean the process after spawn, including observation/stat/wait errors.
    // File capture has no inherited-pipe EOF join; Unix cleanup owns its group.
    cleanup_git(&mut child, deadline)?;
    let status = polled?;
    if !status.success() {
        bail!("harness source Git admission refused");
    }
    let mut bytes = Vec::new();
    fs::File::open(capture.output())?
        .take(limit + 1)
        .read_to_end(&mut bytes)?;
    if bytes.len() as u64 > limit {
        bail!("harness Git output exceeds finite bound");
    }
    Ok(bytes)
}
fn poll_git(
    child: &mut Child,
    output: &Path,
    deadline: Instant,
    limit: u64,
) -> Result<std::process::ExitStatus> {
    loop {
        if fs::metadata(output)?.len() > limit || Instant::now() >= deadline {
            bail!("harness Git admission exceeded finite output/deadline bound");
        }
        if let Some(status) = child.try_wait().context("poll harness Git admission")? {
            return Ok(status);
        }
        thread::sleep(Duration::from_millis(10));
    }
}

fn cleanup_git(child: &mut Child, deadline: Instant) -> Result<()> {
    #[cfg(unix)]
    {
        let group = -libc::pid_t::try_from(child.id()).context("Git child PID overflow")?;
        // SAFETY: the freshly spawned process owns this group; no pointers are used.
        if unsafe { libc::kill(group, libc::SIGKILL) } == -1 {
            let error = std::io::Error::last_os_error();
            if error.raw_os_error() != Some(libc::ESRCH) {
                return Err(error.into());
            }
        }
    }
    if child.try_wait()?.is_none() {
        child.kill().context("kill Git admission child")?;
    }
    loop {
        if child.try_wait()?.is_some() {
            return Ok(());
        }
        if Instant::now() >= deadline {
            bail!("Git admission cleanup exceeded deadline");
        }
        thread::sleep(Duration::from_millis(5));
    }
}

fn admit(harness: &Path, expected: &str, agent: Option<&str>) -> Result<String> {
    let head = String::from_utf8(git(harness, &["rev-parse", "--verify", "HEAD"])?)
        .context("Git HEAD is not UTF-8")?;
    if head.trim() != expected {
        bail!("harness source HEAD differs from immutable pin {expected}");
    }
    tracked_tree(harness, 0, &mut 0, expected == registry::SWE_BENCH_PRO_REF)?;
    if let Some(expected_agent) = agent {
        let link = git(harness, &["rev-parse", "--verify", "HEAD:SWE-agent"])?;
        if link != format!("{expected_agent}\n").as_bytes() {
            bail!("SWE-agent gitlink differs from admitted upstream pin");
        }
        let actual = git(
            &harness.join("SWE-agent"),
            &["rev-parse", "--verify", "HEAD"],
        )?;
        if actual != link {
            bail!("SWE-agent checkout differs from admitted gitlink");
        }
    }
    Ok(expected.into())
}

fn tracked_tree(harness: &Path, depth: usize, nodes: &mut usize, large_parent: bool) -> Result<()> {
    *nodes += 1;
    if depth > 32 || *nodes > 256 {
        bail!("harness recursive gitlink exceeds finite tree bound");
    }
    // Refuse local index flags that can hide tracked worktree changes.
    let flags = source_metadata(harness, &["ls-files", "-v", "-z"], large_parent)?;
    if flags
        .split(|b| *b == 0)
        .filter(|e| !e.is_empty())
        .any(|entry| entry.first() != Some(&b'H'))
    {
        bail!("harness tracked source has hidden or unresolved index flags");
    }
    git(
        harness,
        &[
            "diff",
            "--quiet",
            "--no-ext-diff",
            "--no-textconv",
            "--ignore-submodules=untracked",
            "HEAD",
            "--",
        ],
    )?;
    let tree = source_metadata(harness, &["ls-tree", "-r", "-z", "HEAD"], large_parent)?;
    for entry in tree
        .split(|byte| *byte == 0)
        .filter(|entry| entry.starts_with(b"160000 commit "))
    {
        let (identity, path) = entry
            .iter()
            .position(|byte| *byte == b'\t')
            .map(|offset| (&entry[..offset], &entry[offset + 1..]))
            .ok_or_else(|| anyhow::anyhow!("invalid harness gitlink record"))?;
        let expected = std::str::from_utf8(&identity[14..]).context("invalid gitlink identity")?;
        let path = Path::new(std::str::from_utf8(path).context("gitlink path is not UTF-8")?);
        if !path
            .components()
            .all(|part| matches!(part, std::path::Component::Normal(_)))
        {
            bail!("unsafe harness gitlink path");
        }
        let child = harness.join(path);
        let head = git(&child, &["rev-parse", "--verify", "HEAD"])?;
        if head != format!("{expected}\n").as_bytes() {
            bail!("harness gitlink checkout is absent or differs from admitted tree");
        }
        tracked_tree(&child, depth + 1, nodes, false)?;
    }
    Ok(())
}

pub(super) fn admit_run(root: &Path, definition: EvalDefinition) -> Result<Option<String>> {
    let agent = match definition.id {
        EvalId::McpAtlas => None,
        EvalId::SweBenchPro => Some(registry::SWE_AGENT_REF),
        _ => return Ok(None),
    };
    admit(&harness_dir(root, definition), definition.repo_ref, agent).map(Some)
}

pub(super) fn admit_swe_agent_snapshot(path: &Path) -> Result<()> {
    admit(path, registry::SWE_AGENT_REF, None)?;
    Ok(())
}

pub(super) fn clone_swe_agent_snapshot(
    source: &Path,
    destination: &Path,
    deadline: Instant,
) -> Result<()> {
    clone_snapshot_expected(source, destination, deadline, registry::SWE_AGENT_REF)
}
fn clone_snapshot_expected(
    source: &Path,
    destination: &Path,
    deadline: Instant,
    expected: &str,
) -> Result<()> {
    if !source.is_absolute() || !destination.is_absolute() || destination.exists() {
        bail!("SWE snapshot needs an admitted absolute local source and fresh destination");
    }
    admit(source, expected, None)?;
    let mut clone = snapshot_git();
    clone
        .args([
            "clone",
            "--quiet",
            "--local",
            "--no-hardlinks",
            "--no-checkout",
            "--no-recurse-submodules",
            "--",
        ])
        .arg(source)
        .arg(destination);
    snapshot_capture(clone, deadline)?;
    let mut checkout = snapshot_git();
    checkout
        .arg("-C")
        .arg(destination)
        .args(["checkout", "--quiet", "--detach", expected]);
    snapshot_capture(checkout, deadline)?;
    admit(source, expected, None)?;
    admit(destination, expected, None)?;
    Ok(())
}

fn snapshot_git() -> Command {
    let mut command = Command::new("/usr/bin/git");
    command
        .env_clear()
        .env("PATH", "/usr/bin:/bin")
        .env("GIT_MASTER", "1")
        .env("GIT_TERMINAL_PROMPT", "0")
        .env("GIT_NO_REPLACE_OBJECTS", "1")
        .env("GIT_CONFIG_NOSYSTEM", "1")
        .env("GIT_CONFIG_GLOBAL", "/dev/null")
        .args([
            "-c",
            "core.fsmonitor=false",
            "-c",
            "core.hooksPath=/dev/null",
            "-c",
            "protocol.allow=never",
            "-c",
            "protocol.file.allow=always",
        ]);
    command
}
fn snapshot_capture(command: Command, deadline: Instant) -> Result<()> {
    let remaining = deadline.saturating_duration_since(Instant::now());
    if remaining < Duration::from_secs(1) {
        bail!("SWE snapshot preparation deadline");
    }
    capture_git(command, remaining.min(Duration::from_secs(30)))?;
    Ok(())
}
// Exact current SWE parent has 1,923,079 tree bytes and 1,360,800 index flag bytes.
// Only these two parent metadata commands receive 2MiB; all other Git remains 1MiB.
fn source_metadata(harness: &Path, args: &[&str], large_parent: bool) -> Result<Vec<u8>> {
    if !large_parent {
        return git(harness, args);
    }
    if !matches!(
        args,
        ["ls-files", "-v", "-z"] | ["ls-tree", "-r", "-z", "HEAD"]
    ) {
        bail!("unsupported large SWE metadata command");
    }
    let mut command = snapshot_git();
    command
        .arg("-C")
        .arg(harness)
        .args(["-c", "core.filemode=true"])
        .args(args);
    capture_git_limit(command, Duration::from_secs(10), 2 * 1048576)
}
