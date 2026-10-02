use crate::automation::command_interrupt::Interrupt;
use crate::command::DynResult;
use crate::process::{Completion, Limits, OutputFiles, ProcessSpec, Readiness, Value, supervise};
use crate::repository::{check_args::Grammar, check_report::CheckReport};
use serde::{Deserialize, Serialize};
use std::{
    path::{Path, PathBuf},
    time::Duration,
};

#[derive(Deserialize)]
struct Input {
    repo: PathBuf,
    worktree_root: PathBuf,
    label: String,
    #[serde(rename = "ref")]
    reference: String,
    backend: String,
    git: PathBuf,
    just: PathBuf,
    timeout_seconds: u64,
    logs: PathBuf,
    #[serde(default)]
    skip_build: bool,
}

#[derive(Serialize)]
struct Build {
    label: String,
    engine: &'static str,
    #[serde(rename = "ref")]
    reference: String,
    commit: String,
    worktree: PathBuf,
    binary: PathBuf,
    binary_sha256: String,
    runtime_root: PathBuf,
    runtime: PathBuf,
    runtime_sha256: String,
    backend: String,
}

struct Commands {
    git: PathBuf,
    just: PathBuf,
    limits: Limits,
    interrupt: Interrupt,
}

impl Commands {
    fn execute(&self, spec: &ProcessSpec, files: OutputFiles) -> DynResult<String> {
        let bounded_output = files.stdout.is_none();
        let report = supervise(spec, &self.limits, &self.interrupt.cancellation(), files)?;
        self.interrupt.check()?;
        if !report.success() {
            return Err(format!("replay build command failed: {:?}", report.outcome).into());
        }
        if bounded_output
            && report.stdout.bytes_retained.len() >= self.limits.retained_bytes_per_stream
        {
            return Err("replay command output exceeds retained bound".into());
        }
        Ok(String::from_utf8(report.stdout.bytes_retained)?
            .trim()
            .into())
    }

    fn git(&self, cwd: &Path, args: &[&str]) -> DynResult<String> {
        self.execute(&self.spec(&self.git, cwd, args), OutputFiles::default())
    }

    fn spec(&self, executable: &Path, cwd: &Path, args: &[&str]) -> ProcessSpec {
        ProcessSpec {
            executable: executable.into(),
            cwd: cwd.into(),
            arguments: args
                .iter()
                .map(|arg| Value::Public((*arg).into()))
                .collect(),
            environment: std::env::vars_os()
                .filter(|(key, _)| {
                    [
                        "PATH",
                        "HOME",
                        "USERPROFILE",
                        "SYSTEMROOT",
                        "WINDIR",
                        "TMPDIR",
                        "TEMP",
                        "TMP",
                        "CARGO_HOME",
                        "RUSTUP_HOME",
                        "SCCACHE_DIR",
                        "SCCACHE_SERVER_UDS",
                        "MACOSX_DEPLOYMENT_TARGET",
                    ]
                    .iter()
                    .any(|name| key == name)
                })
                .map(|(key, value)| (key, Value::Public(value)))
                .collect(),
        }
    }
}

fn build(input: Input) -> DynResult<Build> {
    if input.label.is_empty()
        || !input
            .label
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || b"_.-".contains(&byte))
        || input.label.starts_with(['.', '-'])
        || input.label.ends_with(['.', '-'])
        || input.reference.trim().is_empty()
        || !(1..=86400).contains(&input.timeout_seconds)
        || ![
            "cpu",
            "metal",
            "cuda",
            "cuda-blackwell",
            "rocm",
            "hip",
            "vulkan",
        ]
        .contains(&input.backend.as_str())
    {
        return Err("invalid replay arm identity, backend or deadline".into());
    }
    if !input.git.is_absolute()
        || !input.just.is_absolute()
        || !input.logs.is_absolute()
        || !input.worktree_root.is_absolute()
    {
        return Err("replay executables, worktree root and log directory must be absolute".into());
    }
    let repo = input.repo.canonicalize()?;
    let commands = Commands {
        git: input.git.canonicalize()?,
        just: input.just.canonicalize()?,
        limits: Limits {
            execution: Duration::from_secs(input.timeout_seconds),
            graceful_shutdown: Duration::from_secs(5),
            forced_shutdown: Duration::from_secs(5),
            retained_bytes_per_stream: 65536,
            readiness: Readiness::None,
            completion: Completion::Exit,
        },
        interrupt: Interrupt::install()?,
    };
    let commit = commands.git(
        &repo,
        &[
            "rev-parse",
            "--verify",
            "--end-of-options",
            &format!("{}^{{commit}}", input.reference.trim()),
        ],
    )?;
    if commit.len() != 40 || !commit.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return Err("invalid resolved commit identity".into());
    }
    let worktree = input
        .worktree_root
        .join(format!("{}-{}", input.label, &commit[..10]));
    if worktree.try_exists()? {
        if commands.git(&worktree, &["rev-parse", "HEAD"])? != commit {
            return Err("existing replay worktree identity differs".into());
        }
    } else {
        std::fs::create_dir_all(&input.worktree_root)?;
        commands.git(
            &repo,
            &[
                "worktree",
                "add",
                "--detach",
                worktree.to_str().ok_or("non-Unicode worktree path")?,
                &commit,
            ],
        )?;
    }
    if !commands
        .git(
            &worktree,
            &["status", "--porcelain", "--untracked-files=no"],
        )?
        .is_empty()
    {
        return Err("replay worktree is dirty".into());
    }
    if !input.skip_build {
        std::fs::create_dir_all(&input.logs)?;
        for (name, args) in [
            ("host", vec!["release-host-build"]),
            (
                "runtime",
                vec!["release-runtime-build", input.backend.as_str()],
            ),
        ] {
            let files = OutputFiles {
                stdout: Some(input.logs.join(format!("build-{}-{name}.log", input.label))),
                stderr: Some(
                    input
                        .logs
                        .join(format!("build-{}-{name}.stderr.log", input.label)),
                ),
            };
            commands.execute(&commands.spec(&commands.just, &worktree, &args), files)?;
        }
    }
    let binary = worktree.join("target/release/mesh-llm");
    let runtime_root = worktree.join("dist/native-runtimes");
    let runtime = super::build_runtime::select(&runtime_root, &input.backend)?;
    if commands.git(&worktree, &["rev-parse", "HEAD"])? != commit {
        return Err("replay worktree moved during build".into());
    }
    let binary_sha256 =
        crate::product::digest::file_sha256(&binary).map_err(|error| error.error)?;
    let runtime_sha256 =
        crate::product::digest::tree_sha256(&runtime).map_err(|error| error.error)?;
    commands.interrupt.finish()?;
    Ok(Build {
        label: input.label,
        engine: "mesh",
        reference: input.reference,
        commit,
        worktree,
        binary,
        binary_sha256,
        runtime_root,
        runtime,
        runtime_sha256,
        backend: input.backend,
    })
}

pub(in crate::automation) fn run(args: &[String]) -> DynResult<()> {
    const GRAMMAR: Grammar = Grammar {
        usage: "cargo xtool automation replay-matrix build-arm --input PATH --output PATH",
        values: &["--input", "--output"],
        flags: &["--help"],
    };
    let parsed = match GRAMMAR.parse(args) {
        Ok(parsed) => parsed,
        Err(report) => return report.emit(),
    };
    if parsed.flag("--help") {
        return CheckReport::success(format!("{}\n", GRAMMAR.usage)).emit();
    }
    if !parsed.positionals.is_empty() {
        return GRAMMAR.error("unexpected positional arguments").emit();
    }
    let input: Input = serde_json::from_slice(&std::fs::read(
        parsed.last("--input").ok_or("missing --input")?,
    )?)?;
    crate::command::write_json_file(
        Path::new(parsed.last("--output").ok_or("missing --output")?),
        &build(input)?,
    )
}
