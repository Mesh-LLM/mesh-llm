use super::runtime_seed_io::save;
use crate::{
    command::DynResult,
    process::{self, Cancellation, Completion, Limits, OutputFiles, ProcessSpec, Readiness, Value},
};
use std::{
    collections::BTreeMap,
    ffi::OsString,
    fs,
    path::{Path, PathBuf},
    time::{Duration, Instant},
};

pub(super) type Environment = BTreeMap<OsString, OsString>;

pub(super) fn recorded_environment(environment: &Environment) -> DynResult<BTreeMap<&str, &str>> {
    environment
        .iter()
        .map(|(key, value)| {
            Ok((
                key.to_str().ok_or("non-UTF8 environment name")?,
                value.to_str().ok_or("non-UTF8 environment value")?,
            ))
        })
        .collect()
}

pub(super) struct Runner {
    pub root: PathBuf,
    pub environment: Environment,
    pub cancellation: Cancellation,
    pub deadline: Instant,
}

pub(super) fn environment(work: &Path, cache: &Path, target: &Path) -> Environment {
    let mut env: Environment = std::env::vars_os()
        .filter(|(key, _)| {
            [
                "PATH",
                "HOME",
                "USER",
                "LANG",
                "LC_ALL",
                "CARGO_HOME",
                "RUSTUP_HOME",
                "RUSTUP_TOOLCHAIN",
                "LD_LIBRARY_PATH",
                "LIBCLANG_PATH",
                "PROTOC",
                "SSL_CERT_FILE",
                "SSL_CERT_DIR",
                "MESH_LLM_AUTOMATION_BIN",
            ]
            .iter()
            .any(|name| key == name)
        })
        .collect();
    for (key, value) in [
        ("CARGO_INCREMENTAL", "0"),
        ("RUSTC_WRAPPER", "sccache"),
        ("MESH_LLM_REQUIRE_SCCACHE", "1"),
        ("MESH_LLM_SKIP_UI", "1"),
        ("SCCACHE_GHA_ENABLED", "false"),
        ("SCCACHE_MULTILEVEL_CHAIN", "disk"),
        ("SCCACHE_MULTILEVEL_WRITE_ERROR_POLICY", "all"),
        ("SCCACHE_CACHE_SIZE", "2G"),
        ("SCCACHE_IDLE_TIMEOUT", "0"),
        ("CARGO_PROFILE_DEV_DEBUG", "0"),
        ("CARGO_PROFILE_TEST_DEBUG", "0"),
        ("CARGO_PROFILE_RELEASE_DEBUG", "0"),
        ("CARGO_PROFILE_BENCH_DEBUG", "0"),
        ("GIT_CONFIG_NOSYSTEM", "1"),
    ] {
        env.insert(key.into(), value.into());
    }
    for (key, path) in [
        ("SCCACHE_DIR", cache.to_path_buf()),
        ("CARGO_TARGET_DIR", target.to_path_buf()),
        ("CARGO_BUILD_BUILD_DIR", target.join("build")),
        ("SCCACHE_SERVER_UDS", work.join("server.sock")),
        ("SCCACHE_CONF", work.join("sccache.toml")),
        ("SCCACHE_CACHED_CONF", work.join("cached-config")),
        ("TMPDIR", work.join("tmp")),
    ] {
        env.insert(key.into(), path.into_os_string());
    }
    env
}

impl Runner {
    pub fn command(&self, program: &str, args: &[&str], evidence: &Path) -> DynResult<Vec<u8>> {
        let result = self.execute(program, args, evidence);
        if let Err(error) = &result {
            save(
                &evidence.with_extension("failure.json"),
                &serde_json::json!({"program":program,"arguments":args,"error":error.to_string()}),
            )?;
        }
        result
    }

    fn execute(&self, program: &str, args: &[&str], evidence: &Path) -> DynResult<Vec<u8>> {
        let executable = executable(program, &self.environment)?;
        let spec = ProcessSpec {
            executable,
            cwd: self.root.clone(),
            arguments: args
                .iter()
                .map(|arg| Value::Public((*arg).into()))
                .collect(),
            environment: self
                .environment
                .iter()
                .map(|(key, value)| (key.clone(), Value::Public(value.clone())))
                .collect(),
        };
        let remaining = self
            .deadline
            .checked_duration_since(Instant::now())
            .ok_or("pair execution deadline expired")?;
        let limits = Limits {
            execution: remaining.min(Duration::from_secs(7200)),
            graceful_shutdown: Duration::from_secs(5),
            forced_shutdown: Duration::from_secs(5),
            retained_bytes_per_stream: 16 * 1024 * 1024,
            readiness: Readiness::None,
            completion: Completion::Exit,
        };
        let report = process::supervise(
            &spec,
            &limits,
            &self.cancellation,
            OutputFiles {
                stdout: Some(evidence.with_extension("stdout.log")),
                stderr: Some(evidence.with_extension("stderr.log")),
            },
        )?;
        save(
            &evidence.with_extension("process.json"),
            &serde_json::json!({
                "program":program,"arguments":args,"seconds":report.elapsed.as_secs_f64(),
                "outcome":format!("{:?}", report.outcome),"exit_code":report.status.and_then(|status| status.code()),
                "cleanup_complete":report.cleanup.complete,"cleanup_forced":report.cleanup.forced,
                "failure":report.failure.as_ref().map(ToString::to_string),
                "stdout_bytes":report.stdout.bytes_seen,"stderr_bytes":report.stderr.bytes_seen,
                "stdout_truncated":report.stdout.truncated,"stderr_truncated":report.stderr.truncated,
                "peak_resources":null,"resource_note":"supervisor has no per-tree peak resource accounting; host/cgroup observations retained separately"
            }),
        )?;
        if !report.success() {
            return Err(format!("{program} failed; see {}", evidence.display()).into());
        }
        if report.stdout.truncated
            || report.stderr.truncated
            || report.stdout.suppressed_lines != 0
            || report.stderr.suppressed_lines != 0
        {
            return Err("incomplete diagnostic output: inconclusive".into());
        }
        Ok(report.stdout.bytes_retained)
    }
}

fn executable(program: &str, environment: &Environment) -> DynResult<PathBuf> {
    let path = environment
        .get(&OsString::from("PATH"))
        .ok_or("missing PATH")?;
    for directory in std::env::split_paths(path) {
        let candidate = directory.join(program);
        if fs::metadata(&candidate).is_ok_and(|metadata| {
            #[cfg(unix)]
            {
                use std::os::unix::fs::PermissionsExt;
                metadata.is_file() && metadata.permissions().mode() & 0o111 != 0
            }
            #[cfg(not(unix))]
            {
                metadata.is_file()
            }
        }) {
            return Ok(if candidate.is_absolute() {
                candidate
            } else {
                std::env::current_dir()?.join(candidate)
            });
        }
    }
    Err(format!("missing executable: {program}").into())
}
