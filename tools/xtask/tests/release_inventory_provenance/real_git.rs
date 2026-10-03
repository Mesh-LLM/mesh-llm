use crate::{
    process::{
        self, Cancellation, Completion, Limits, ProcessSpec, RawCaptureOptions, Readiness, Value,
    },
    provenance::{self, Git, Output},
};
use std::{
    collections::BTreeMap,
    fs,
    num::NonZeroUsize,
    path::PathBuf,
    time::{Duration, Instant},
};
pub(super) struct Repository {
    _temp: tempfile::TempDir,
    pub(super) root: PathBuf,
    executable: PathBuf,
    deadline: Instant,
}
impl Repository {
    pub(super) fn new() -> Self {
        let temp = tempfile::tempdir().unwrap();
        let root = temp
            .path()
            .canonicalize()
            .unwrap()
            .join("release provenance fixture space");
        fs::create_dir(&root).unwrap();
        let name = if cfg!(windows) { "git.exe" } else { "git" };
        let executable = std::env::split_paths(&std::env::var_os("PATH").unwrap())
            .map(|p| p.join(name))
            .find(|p| p.is_file())
            .expect("native Git required for actual source proof")
            .canonicalize()
            .unwrap();
        let mut repo = Self {
            _temp: temp,
            root,
            executable,
            deadline: Instant::now() + Duration::from_secs(60),
        };
        repo.ok(&["init", "-b", "main"]);
        repo.ok(&["config", "user.name", "Finite Release Fixture"]);
        repo.ok(&["config", "user.email", "release-fixture@example.invalid"]);
        repo.commit("initial main source");
        repo.ok(&["update-ref", "refs/remotes/origin/main", "HEAD"]);
        repo.ok(&["tag", "v1.0.0"]);
        repo
    }
    pub(super) fn ok(&mut self, args: &[&str]) -> String {
        let out = self.read(args).unwrap();
        assert_eq!(out.code, 0, "{args:?}");
        String::from_utf8(out.stdout)
            .unwrap()
            .trim_end_matches(['\r', '\n'])
            .to_owned()
    }
    pub(super) fn commit(&mut self, subject: &str) -> String {
        self.ok(&[
            "-c",
            "core.hooksPath=",
            "commit",
            "--allow-empty",
            "--no-gpg-sign",
            "-m",
            subject,
        ]);
        self.ok(&["rev-parse", "HEAD"])
    }
    pub(super) fn main_commit(&mut self, subject: &str) -> String {
        let sha = self.commit(subject);
        self.ok(&["update-ref", "refs/remotes/origin/main", &sha]);
        sha
    }
    pub(super) fn prepared(&mut self, base: &str, tag: &str, annotated: bool) -> String {
        self.ok(&["checkout", "--detach", base]);
        let oid = self.commit(&format!("{tag}: prepare release source"));
        if annotated {
            self.ok(&[
                "-c",
                "tag.gpgSign=false",
                "tag",
                "-a",
                tag,
                "-m",
                "finite annotated release",
            ]);
        } else {
            self.ok(&["tag", tag]);
        }
        oid
    }
}
impl Git for Repository {
    fn read(&mut self, args: &[&str]) -> provenance::Result<Output> {
        let remaining = self
            .deadline
            .checked_duration_since(Instant::now())
            .ok_or_else(|| provenance::Error("finite fixture Git budget exhausted".into()))?;
        let mut environment = BTreeMap::from([
            ("GIT_CONFIG_NOSYSTEM".into(), Value::Public("1".into())),
            (
                "GIT_CONFIG_GLOBAL".into(),
                Value::Public(self.root.join("absent-global-config").into()),
            ),
            ("GIT_TERMINAL_PROMPT".into(), Value::Public("0".into())),
            (
                "GIT_AUTHOR_DATE".into(),
                Value::Public("2026-10-02T00:00:00+00:00".into()),
            ),
            (
                "GIT_COMMITTER_DATE".into(),
                Value::Public("2026-10-02T00:00:00+00:00".into()),
            ),
        ]);
        for key in ["PATH", "SystemRoot", "TEMP", "TMP", "HOME", "USERPROFILE"] {
            if let Some(value) = std::env::var_os(key) {
                environment.insert(key.into(), Value::Public(value));
            }
        }
        let report = process::supervise_raw(
            &ProcessSpec {
                executable: self.executable.clone(),
                cwd: self.root.clone(),
                environment,
                arguments: args.iter().map(|s| Value::Public((*s).into())).collect(),
            },
            &Limits {
                execution: remaining,
                graceful_shutdown: Duration::from_secs(1),
                forced_shutdown: Duration::from_secs(2),
                retained_bytes_per_stream: 65536,
                readiness: Readiness::None,
                completion: Completion::Exit,
            },
            &Cancellation::default(),
            RawCaptureOptions {
                stdout: NonZeroUsize::new(1024 * 1024),
                stderr: NonZeroUsize::new(65536),
            },
        )
        .map_err(|e| provenance::Error(e.to_string()))?;
        if report.process.failure.is_some()
            || !report.process.cleanup.complete
            || report.process.outcome != process::Outcome::Exited
        {
            return Err(provenance::Error(format!(
                "finite Git transport refused: {:?}",
                report.process
            )));
        }
        Ok(Output {
            code: report
                .process
                .status
                .and_then(|s| s.code())
                .ok_or_else(|| provenance::Error("Git exited without native status".into()))?,
            stdout: report.stdout.unwrap().as_bytes().to_vec(),
        })
    }
}
