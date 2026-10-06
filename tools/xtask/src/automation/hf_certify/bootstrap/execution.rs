use super::super::admission;
use super::contract::{Input, Tool};
use crate::{
    command::DynResult,
    process::{
        self, Cancellation, Completion, Limits, OutputFiles, ProcessSpec, RawCaptureOptions,
        Readiness, Value,
    },
};
use serde_json::{Value as Json, json};
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeMap,
    io::Read,
    num::NonZeroUsize,
    path::Path,
    time::{Duration, Instant},
};
#[derive(serde::Serialize)]
pub(in crate::automation::hf_certify) struct ObservedBootstrap {
    pub binary: admission::Artifact,
    pub mesh_commit: String,
    pub prepared_llama_commit: String,
}
fn check(deadline: Instant, cancel: &Cancellation) -> DynResult<()> {
    if cancel.is_cancelled() {
        return Err("bootstrap cancelled".into());
    }
    if Instant::now() >= deadline {
        return Err("bootstrap deadline expired".into());
    }
    Ok(())
}
pub(in crate::automation::hf_certify) fn observe(
    path: &Path,
    deadline: Instant,
    cancel: &Cancellation,
) -> DynResult<String> {
    observe_bounded(path, deadline, cancel, 256 * 1024 * 1024)
}
/// The owned coordinator includes all automation owners in debug builds. Keep
/// its separate bounded identity admission without expanding bootstrap tool bounds.
pub(in crate::automation::hf_certify) fn observe_runner(
    path: &Path,
    deadline: Instant,
    cancel: &Cancellation,
) -> DynResult<String> {
    observe_bounded(path, deadline, cancel, 1024 * 1024 * 1024)
}
fn observe_bounded(
    path: &Path,
    deadline: Instant,
    cancel: &Cancellation,
    byte_limit: u64,
) -> DynResult<String> {
    check(deadline, cancel)?;
    if !std::fs::symlink_metadata(path)?.is_file() {
        return Err("bootstrap tool must be a regular file".into());
    }
    let mut options = std::fs::OpenOptions::new();
    options.read(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt as _;
        options.custom_flags(libc::O_NONBLOCK | libc::O_NOFOLLOW);
    }
    let mut file = options.open(path)?;
    let metadata = file.metadata()?;
    if !metadata.is_file() || metadata.len() > byte_limit {
        return Err("bootstrap tool file/type bound refused".into());
    }
    let mut hash = Sha256::new();
    let mut seen = 0_u64;
    let mut bytes = vec![0; 1048576];
    loop {
        check(deadline, cancel)?;
        let count = file.read(&mut bytes)?;
        if count == 0 {
            break;
        }
        seen += count as u64;
        if seen > byte_limit {
            return Err("bootstrap file grew past byte bound".into());
        }
        hash.update(&bytes[..count]);
    }
    Ok(hex::encode(hash.finalize()))
}
fn pin(tool: &Tool, deadline: Instant, cancel: &Cancellation) -> DynResult<()> {
    if observe(&tool.path, deadline, cancel)? != tool.sha256 {
        return Err("bootstrap tool byte pin mismatch".into());
    }
    Ok(())
}
fn environment(input: &Input) -> DynResult<BTreeMap<std::ffi::OsString, Value>> {
    for dir in &input.path_directories {
        if !std::fs::symlink_metadata(dir)?.is_dir() {
            return Err("bootstrap PATH entry must be directory".into());
        }
    }
    let mut env = BTreeMap::from([(
        "PATH".into(),
        Value::Public(std::env::join_paths(&input.path_directories)?),
    )]);
    for (name, value) in [
        ("GIT_MASTER", "1"),
        ("GIT_OPTIONAL_LOCKS", "0"),
        ("GIT_TERMINAL_PROMPT", "0"),
        ("GIT_NO_REPLACE_OBJECTS", "1"),
        ("GIT_CONFIG_NOSYSTEM", "1"),
        ("GIT_CONFIG_GLOBAL", "/dev/null"),
        ("LC_ALL", "C"),
    ] {
        env.insert(name.into(), Value::Public(value.into()));
    }
    for tool in &input.tools {
        let selected = input
            .path_directories
            .iter()
            .map(|directory| directory.join(&tool.name))
            .find(|path| path.is_file())
            .ok_or("bootstrap named tool absent from declared PATH")?;
        if selected.canonicalize()? != tool.path.canonicalize()? {
            return Err("bootstrap PATH selects a different tool than its pin".into());
        }
    }
    Ok(env)
}
struct Runner<'a> {
    input: &'a Input,
    evidence: &'a Path,
    deadline: Instant,
    cancel: &'a Cancellation,
    rows: &'a mut Vec<Json>,
}
impl Runner<'_> {
    fn phase(&mut self, tool: &str, args: &[&str], cwd: &Path, label: &str) -> DynResult<Vec<u8>> {
        let input = self.input;
        let evidence = self.evidence;
        let deadline = self.deadline;
        let cancel = self.cancel;
        let rows = &mut *self.rows;
        check(deadline, cancel)?;
        let executable = input.tool(tool)?;
        pin(executable, deadline, cancel)?;
        let execution = deadline
            .checked_duration_since(Instant::now())
            .and_then(|d| d.checked_sub(Duration::from_secs(3)))
            .filter(|d| !d.is_zero())
            .ok_or("bootstrap has no execution allowance after cleanup reserve")?;
        let report = process::supervise_raw_with_files(
            &ProcessSpec {
                executable: executable.path.clone(),
                arguments: args.iter().map(|v| Value::Public((*v).into())).collect(),
                cwd: cwd.into(),
                environment: environment(input)?,
            },
            &Limits {
                execution,
                graceful_shutdown: Duration::from_secs(1),
                forced_shutdown: Duration::from_secs(1),
                retained_bytes_per_stream: 16777216,
                readiness: Readiness::None,
                completion: Completion::Exit,
            },
            cancel,
            OutputFiles {
                stdout: Some(evidence.join(format!("{label}-stdout.log"))),
                stderr: Some(evidence.join(format!("{label}-stderr.log"))),
            },
            RawCaptureOptions {
                stdout: NonZeroUsize::new(16777216),
                stderr: NonZeroUsize::new(16777216),
            },
        )?;
        let clean = super::super::execution::clean(&report);
        rows.push(json!({"phase":label,"outcome":format!("{:?}",report.process.outcome),"status":report.process.status.and_then(|s|s.code()),"cleanup_complete":report.process.cleanup.complete,"forced":report.process.cleanup.forced,"clean":clean,"stdout_sha256":report.stdout.as_ref().map(|v|admission::digest(v.as_bytes())),"stdout_bytes":report.process.stdout.bytes_seen,"stderr_bytes":report.process.stderr.bytes_seen}));
        if !clean {
            return Err("bootstrap process failed or incomplete; phase evidence retained".into());
        }
        check(deadline, cancel)?;
        pin(executable, deadline, cancel)?;
        Ok(report
            .stdout
            .ok_or("bootstrap stdout missing")?
            .as_bytes()
            .to_vec())
    }
    fn prepared(&mut self, root: &Path, label: &str) -> DynResult<String> {
        let dependency = root.join(".deps/llama.cpp");
        if !std::fs::symlink_metadata(&dependency)?.is_dir() {
            return Err("prepared llama checkout directory refused".into());
        }
        let upstream = admission::read(&dependency.join(".mesh-llm-upstream-sha"), 128)?;
        let patched = admission::read(&dependency.join(".mesh-llm-patched-sha"), 128)?;
        let patched = std::str::from_utf8(&patched)?.trim().to_owned();
        if std::str::from_utf8(&upstream)?.trim() != self.input.llama_commit
            || !super::contract::hex(&patched, 40)
        {
            return Err("prepared llama stamp refused".into());
        }
        let observed = self.phase(
            "git",
            &["rev-parse", "HEAD"],
            &dependency,
            &format!("{label}-llama-head"),
        )?;
        if std::str::from_utf8(&observed)?.trim() != patched {
            return Err("prepared llama HEAD/stamp mismatch".into());
        }
        self.phase(
            "git",
            &[
                "merge-base",
                "--is-ancestor",
                &self.input.llama_commit.clone(),
                "HEAD",
            ],
            &dependency,
            &format!("{label}-llama-base"),
        )?;
        self.phase(
            "git",
            &["diff-index", "--quiet", "HEAD", "--"],
            &dependency,
            &format!("{label}-llama-clean"),
        )?;
        Ok(patched)
    }
    fn source(&mut self, root: &Path, label: &str) -> DynResult<()> {
        let input = self.input;
        for (suffix, args, expected) in [
            (
                "head",
                vec!["rev-parse", "HEAD"],
                input.mesh_commit.as_str(),
            ),
            (
                "tree",
                vec!["rev-parse", "HEAD^{tree}"],
                input.git_tree.as_str(),
            ),
        ] {
            let output = self.phase("git", &args, root, &format!("{label}-{suffix}"))?;
            if std::str::from_utf8(&output)?.trim() != expected {
                return Err("bootstrap selected source identity mismatch".into());
            }
        }
        self.phase(
            "git",
            &["diff-index", "--quiet", "HEAD", "--"],
            root,
            &format!("{label}-clean"),
        )?;
        let path = root.join("third_party/llama.cpp/upstream.txt");
        let bytes = admission::read(&path, 4096)?;
        if admission::digest(&bytes) != input.upstream_file_sha256
            || !std::str::from_utf8(&bytes)?
                .lines()
                .any(|line| line.trim() == input.llama_commit)
        {
            return Err("bootstrap llama upstream identity mismatch".into());
        }
        Ok(())
    }
}
pub(in crate::automation::hf_certify) fn execute(
    input: &Input,
    evidence: &Path,
    deadline: Instant,
    cancel: &Cancellation,
    rows: &mut Vec<Json>,
) -> DynResult<ObservedBootstrap> {
    input.validate()?;
    check(deadline, cancel)?;
    if !cfg!(target_os = "linux") {
        return Err(
            "HF container bootstrap is Linux-only; no other platform support claimed".into(),
        );
    }
    for tool in &input.tools {
        pin(tool, deadline, cancel)?;
    }
    let mut runner = Runner {
        input,
        evidence,
        deadline,
        cancel,
        rows,
    };
    for name in [
        "git", "just", "cargo", "rustc", "cmake", "c++", "ld.lld", "curl",
    ] {
        runner.phase(
            name,
            &["--version"],
            evidence,
            &format!("version-{}", name.replace('+', "p")),
        )?;
    }
    let root = evidence.join("mesh-source");
    runner.phase(
        "git",
        &[
            "clone",
            "--no-checkout",
            "--filter=blob:none",
            "--",
            "https://github.com/Mesh-LLM/mesh-llm.git",
            root.to_str().ok_or("bootstrap Unicode path")?,
        ],
        evidence,
        "clone",
    )?;
    runner.phase(
        "git",
        &["checkout", "--detach", &input.mesh_commit],
        &root,
        "checkout",
    )?;
    runner.source(&root, "before")?;
    let binary = root.join("target/release/skippy-quantize");
    match std::fs::symlink_metadata(&binary) {
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => (),
        _ => return Err("bootstrap fresh checkout contains unexpected binary".into()),
    }
    runner.phase("just", &["llama-prepare"], &root, "prepare")?;
    let prepared = runner.prepared(&root, "before-build")?;
    runner.phase(
        "just",
        &["skippy-quantize-standalone-release-build", "cpu"],
        &root,
        "build",
    )?;
    runner.source(&root, "after")?;
    if runner.prepared(&root, "after-build")? != prepared {
        return Err("prepared llama identity changed during build".into());
    }
    for tool in &input.tools {
        pin(tool, deadline, cancel)?;
    }
    let artifact_sha256 = observe(&binary, deadline, cancel)?;
    runner.rows.push(json!({"phase":"binary_observation","sha256":artifact_sha256,"path":binary,"fresh_build_output_observed":true,"prepared_llama_commit":prepared}));
    check(deadline, cancel)?;
    Ok(ObservedBootstrap {
        binary: admission::Artifact {
            path: binary,
            sha256: artifact_sha256,
        },
        mesh_commit: input.mesh_commit.clone(),
        prepared_llama_commit: prepared,
    })
}

#[cfg(test)]
mod runner_identity_tests {
    use super::*;
    #[test]
    fn coordinator_identity_retains_hash_type_byte_and_terminal_bounds() {
        let root = tempfile::tempdir().unwrap();
        let file = root.path().join("coordinator");
        std::fs::write(&file, b"owned coordinator bytes").unwrap();
        let cancel = Cancellation::default();
        let until = Instant::now() + Duration::from_secs(5);
        assert_eq!(
            observe_runner(&file, until, &cancel).unwrap(),
            admission::digest(b"owned coordinator bytes")
        );
        assert!(observe_bounded(&file, until, &cancel, 4).is_err());
        assert!(observe_runner(root.path(), until, &cancel).is_err());
        assert!(observe_runner(&file, Instant::now(), &cancel).is_err());
        cancel.cancel();
        assert!(observe_runner(&file, until, &cancel).is_err());
        let sparse = std::fs::File::create(root.path().join("oversize")).unwrap();
        sparse.set_len(1024 * 1024 * 1024 + 1).unwrap();
        assert!(
            observe_runner(
                &root.path().join("oversize"),
                until,
                &Cancellation::default()
            )
            .is_err()
        );
    }
}
