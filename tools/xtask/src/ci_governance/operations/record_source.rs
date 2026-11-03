//! Resolve fixed record identities from immutable current-main Git blobs.
use super::intent::ReviewedIntent;
use super::require;
use super::state::{ExpectedState, unique_operation_ids};
use crate::ci_governance::activation::Activation;
use crate::ci_governance::policy::{Policy, valid_sha};
use crate::command::DynResult;
use crate::process::{self, Cancellation, Completion, Limits, ProcessSpec, Readiness, Value};
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeMap,
    path::{Path, PathBuf},
    time::{Duration, Instant},
};

const LIMIT: usize = 1024 * 1024;

pub(super) struct RecordSelector {
    pub id: String,
    pub raw_sha256: String,
}

impl RecordSelector {
    pub fn validate(&self) -> DynResult<()> {
        unique_operation_ids(std::slice::from_ref(&self.id))?;
        require(
            self.raw_sha256.len() == 64
                && self
                    .raw_sha256
                    .bytes()
                    .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte)),
            "record digest must be exactly 64 lowercase hexadecimal digits",
        )
    }
    fn path(&self) -> String {
        format!("ci/governance/records/{}.json", self.id)
    }
}

#[derive(Debug)]
pub(super) struct ResolvedIntent {
    pub protected_main_revision: String,
    pub record_sha256: String,
    pub intent: ReviewedIntent,
    pub policy: Policy,
    pub expected: ExpectedState,
}

pub(super) struct RecordSource {
    root: PathBuf,
    git: PathBuf,
    deadline: Instant,
    cancellation: Cancellation,
}

impl RecordSource {
    pub fn activation_at(&self, current_main: &str) -> DynResult<Activation> {
        require(
            valid_sha(current_main),
            "protected activation revision is invalid",
        )?;
        self.require_head(current_main)?;
        let bytes = self.blob(current_main, "ci/governance/activation.json")?;
        require(
            bytes.len() <= 64 * 1024,
            "protected activation exceeds 64 KiB",
        )?;
        parse(&bytes)
    }
    pub fn new(root: &Path, cancellation: Cancellation) -> DynResult<Self> {
        let root = std::path::absolute(root)?;
        require(
            root == root.canonicalize()?,
            "protected checkout root must be canonical without symlink aliases",
        )?;
        #[cfg(windows)]
        let executable = "git.exe";
        #[cfg(not(windows))]
        let executable = "git";
        let git = std::env::split_paths(&std::env::var_os("PATH").ok_or("Git PATH missing")?)
            .filter(|path| path.is_absolute())
            .map(|path| path.join(executable))
            .find(|path| path.is_file())
            .ok_or("Git executable unavailable")?;
        Ok(Self {
            root,
            git,
            deadline: Instant::now() + Duration::from_secs(30),
            cancellation,
        })
    }

    pub fn resolve(
        &self,
        selector: &RecordSelector,
        current_main: &str,
    ) -> DynResult<ResolvedIntent> {
        selector.validate()?;
        require(
            valid_sha(current_main),
            "current protected main SHA is invalid",
        )?;
        self.require_head(current_main)?;
        let bytes = self.blob(current_main, &selector.path())?;
        require(
            hex::encode(Sha256::digest(&bytes)) == selector.raw_sha256,
            "protected intent record byte digest mismatch",
        )?;
        let intent: ReviewedIntent = parse(&bytes)?;
        require(
            intent.operation_id == selector.id,
            "record identity does not match the fixed UUID selector",
        )?;
        let policy = Policy::from_bytes(&self.blob(current_main, "ci/governance/policy.json")?)?;
        let expected = parse(&self.blob(current_main, "ci/governance/expected-state.json")?)?;
        self.require_head(current_main)?;
        Ok(ResolvedIntent {
            protected_main_revision: current_main.into(),
            record_sha256: selector.raw_sha256.clone(),
            intent,
            policy,
            expected,
        })
    }

    fn require_head(&self, revision: &str) -> DynResult<()> {
        let bytes = self.git(&["rev-parse", "--verify", "HEAD^{commit}"])?;
        require(
            std::str::from_utf8(&bytes)?.trim_end_matches(['\r', '\n']) == revision,
            "checkout must be the exact current protected main; ancestry and other refs are insufficient",
        )
    }

    fn blob(&self, revision: &str, path: &str) -> DynResult<Vec<u8>> {
        self.regular_checkout_path(path)?;
        let tree = self.git(&["ls-tree", "-z", revision, "--", path])?;
        let entry = std::str::from_utf8(&tree)?;
        let (metadata, actual_path) = entry
            .strip_suffix('\0')
            .ok_or("missing protected tree entry")?
            .split_once('\t')
            .ok_or("malformed protected tree entry")?;
        let fields = metadata.split_whitespace().collect::<Vec<_>>();
        require(
            actual_path == path
                && fields.len() == 3
                && fields[0] == "100644"
                && fields[1] == "blob"
                && valid_sha(fields[2]),
            "protected inputs must be single non-executable regular Git blobs, not symlinks or submodules",
        )?;
        let size = self.git(&["cat-file", "-s", fields[2]])?;
        let size = std::str::from_utf8(&size)?.trim().parse::<usize>()?;
        require(size <= LIMIT, "protected governance input exceeds 1 MiB")?;
        let bytes = self.git(&["cat-file", "blob", fields[2]])?;
        require(
            bytes.len() == size,
            "protected blob read truncated or changed",
        )?;
        Ok(bytes)
    }

    fn regular_checkout_path(&self, path: &str) -> DynResult<()> {
        let mut current = self.root.clone();
        let components = Path::new(path).components().collect::<Vec<_>>();
        for (index, component) in components.iter().enumerate() {
            require(
                matches!(component, std::path::Component::Normal(_)),
                "protected input escaped fixed checkout path",
            )?;
            current.push(component.as_os_str());
            let metadata = std::fs::symlink_metadata(&current)?;
            #[cfg(windows)]
            {
                use std::os::windows::fs::MetadataExt;
                require(
                    metadata.file_attributes() & 0x400 == 0,
                    "protected input contains a Windows reparse point",
                )?;
            }
            require(
                !metadata.file_type().is_symlink()
                    && if index + 1 == components.len() {
                        metadata.is_file()
                    } else {
                        metadata.is_dir()
                    },
                "protected input contains a symlink or non-regular path component",
            )?;
        }
        Ok(())
    }

    fn git(&self, args: &[&str]) -> DynResult<Vec<u8>> {
        let execution = self
            .deadline
            .checked_duration_since(Instant::now())
            .ok_or("protected record read deadline expired")?;
        let mut environment = BTreeMap::new();
        for key in ["PATH", "SYSTEMROOT", "WINDIR", "TMPDIR", "TEMP", "TMP"] {
            if let Some(value) = std::env::var_os(key) {
                environment.insert(key.into(), Value::Public(value));
            }
        }
        for (key, value) in [
            ("GIT_CONFIG_NOSYSTEM", "1"),
            ("GIT_CONFIG_COUNT", "0"),
            ("GIT_TERMINAL_PROMPT", "0"),
            ("GIT_NO_LAZY_FETCH", "1"),
            ("GIT_NO_REPLACE_OBJECTS", "1"),
            ("GIT_OPTIONAL_LOCKS", "0"),
        ] {
            environment.insert(key.into(), Value::Public(value.into()));
        }
        #[cfg(windows)]
        let null = "NUL";
        #[cfg(not(windows))]
        let null = "/dev/null";
        environment.insert("GIT_CONFIG_GLOBAL".into(), Value::Public(null.into()));
        // ProcessSpec replaces the child environment. Never inherit injected
        // configuration parameters, including if the allowlist later expands.
        environment.remove(std::ffi::OsStr::new("GIT_CONFIG_PARAMETERS"));
        let mut arguments = vec![Value::Public("--no-replace-objects".into())];
        arguments.extend(args.iter().map(|arg| Value::Public((*arg).into())));
        let report = process::supervise_raw(
            &ProcessSpec {
                executable: self.git.clone(),
                arguments,
                cwd: self.root.clone(),
                environment,
            },
            &Limits {
                execution,
                graceful_shutdown: Duration::from_secs(1),
                forced_shutdown: Duration::from_secs(2),
                retained_bytes_per_stream: LIMIT,
                readiness: Readiness::None,
                completion: Completion::Exit,
            },
            &self.cancellation,
            process::RawCaptureOptions {
                stdout: std::num::NonZeroUsize::new(LIMIT),
                stderr: None,
            },
        )?;
        require(
            report.process.outcome == process::Outcome::Exited
                && report.process.success()
                && report.process.cleanup.complete
                && !report.process.stdout.truncated,
            "protected Git read failed its process/output/cleanup bounds",
        )?;
        Ok(report
            .stdout
            .ok_or("protected Git read lacks complete raw bytes")?
            .as_bytes()
            .to_vec())
    }
}

fn parse<T: serde::de::DeserializeOwned>(bytes: &[u8]) -> DynResult<T> {
    serde_json::from_slice(bytes).map_err(|_| "invalid protected governance input schema".into())
}

#[cfg(test)]
#[path = "record_source_tests.rs"]
mod tests;
