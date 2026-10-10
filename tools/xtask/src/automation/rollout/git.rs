use super::evidence::CatalogBinding;
use super::input::{CommitSha, Request};
use super::{Checked, Issue, reject};
use sha2::{Digest, Sha256};
use std::path::{Path, PathBuf};
use std::process::{Command, Output, Stdio};

pub(super) const CATALOGS: [&str; 2] = ["ci/ownership.yml", "ci/slices.yml"];
const SUPPORT: [&str; 5] = [
    ".github/actions/prepare-automation/action.yml",
    "just/ci.just",
    "tools/xtask/Cargo.toml",
    "tools/xtask/src/automation_bootstrap.rs",
    "tools/xtask/src/ci_plan/mod.rs",
];
const POLICY: [&str; 2] = [
    ".github/actions/select-ci-runners/action.yml",
    "ci/runner-images.json",
];

pub(super) struct Repository(PathBuf);

impl Repository {
    pub(super) fn open(path: &Path) -> Checked<Self> {
        let path = path
            .canonicalize()
            .map_err(|error| reject(Issue::Input, format!("repository: {error}")))?;
        let repository = Self(path);
        repository.success(&["rev-parse", "--git-dir"])?;
        Ok(repository)
    }

    fn command(&self, args: &[&str]) -> Checked<Output> {
        Command::new("git")
            .current_dir(&self.0)
            .env("GIT_MASTER", "1")
            .env("GIT_OPTIONAL_LOCKS", "0")
            .env("GIT_NO_LAZY_FETCH", "1")
            .env("GIT_NO_REPLACE_OBJECTS", "1")
            .env("GIT_TERMINAL_PROMPT", "0")
            .env_remove("GIT_DIR")
            .env_remove("GIT_WORK_TREE")
            .env_remove("GIT_INDEX_FILE")
            .env_remove("GIT_OBJECT_DIRECTORY")
            .env_remove("GIT_ALTERNATE_OBJECT_DIRECTORIES")
            .args(["--no-pager", "--no-replace-objects"])
            .args(args)
            .stdin(Stdio::null())
            .output()
            .map_err(|error| reject(Issue::Input, format!("local git: {error}")))
    }

    fn success(&self, args: &[&str]) -> Checked<Vec<u8>> {
        let output = self.command(args)?;
        if !output.status.success() {
            return Err(reject(
                Issue::Sequence,
                format!(
                    "local git {args:?} failed: {}",
                    String::from_utf8_lossy(&output.stderr)
                ),
            ));
        }
        Ok(output.stdout)
    }

    fn ancestor(&self, before: &CommitSha, after: &CommitSha) -> Checked<()> {
        let output = self.command(&[
            "merge-base",
            "--is-ancestor",
            before.as_str(),
            after.as_str(),
        ])?;
        match output.status.code() {
            Some(0) => Ok(()),
            Some(1) => Err(reject(
                Issue::Sequence,
                format!(
                    "{} is not an ancestor of {}",
                    before.as_str(),
                    after.as_str()
                ),
            )),
            _ => Err(reject(
                Issue::Sequence,
                "cannot establish local commit ancestry",
            )),
        }
    }

    pub(super) fn blob(&self, sha: &CommitSha, path: &str) -> Checked<Vec<u8>> {
        let listing = self.success(&["ls-tree", sha.as_str(), "--", path])?;
        let text = std::str::from_utf8(&listing)
            .map_err(|error| reject(Issue::Input, error.to_string()))?;
        let (metadata, name) = text
            .trim_end_matches('\n')
            .split_once('\t')
            .ok_or_else(|| reject(Issue::Evidence, format!("missing committed file: {path}")))?;
        let fields: Vec<_> = metadata.split_whitespace().collect();
        let [mode, "blob", object] = fields.as_slice() else {
            return Err(reject(
                Issue::Evidence,
                format!("not a committed regular blob: {path}"),
            ));
        };
        let valid_mode = if CATALOGS.contains(&path) {
            *mode == "100644"
        } else {
            matches!(*mode, "100644" | "100755")
        };
        if name != path || !valid_mode {
            return Err(reject(
                Issue::Evidence,
                format!("invalid committed mode or path: {path}"),
            ));
        }
        let size = self.success(&["cat-file", "-s", object])?;
        let size = std::str::from_utf8(&size)
            .map_err(|error| reject(Issue::Evidence, error.to_string()))?
            .trim()
            .parse::<u64>()
            .map_err(|error| reject(Issue::Evidence, error.to_string()))?;
        if size > 16 * 1024 * 1024 {
            return Err(reject(
                Issue::Evidence,
                format!("committed blob exceeds byte limit: {path}"),
            ));
        }
        self.success(&["cat-file", "blob", object])
    }

    pub(super) fn validate_sequence(&self, request: &Request) -> Checked<Vec<CatalogBinding>> {
        for sha in [
            &request.catalog_sha,
            &request.support_sha,
            &request.protected_sha,
            &request.source_sha,
        ] {
            let kind = self.success(&["cat-file", "-t", sha.as_str()])?;
            if kind != b"commit\n" {
                return Err(reject(Issue::Sequence, "revision is not a commit object"));
            }
        }
        self.ancestor(&request.catalog_sha, &request.support_sha)?;
        self.ancestor(&request.support_sha, &request.protected_sha)?;
        self.ancestor(&request.catalog_sha, &request.source_sha)?;
        for path in SUPPORT {
            for sha in [&request.support_sha, &request.protected_sha] {
                if self.blob(sha, path)?.is_empty() {
                    return Err(reject(
                        Issue::Bootstrap,
                        format!("empty protected support file: {path}"),
                    ));
                }
            }
        }
        self.validate_policy(request)?;
        let mut bindings = Vec::new();
        for path in CATALOGS {
            let protected = self.blob(&request.protected_sha, path)?;
            for sha in [
                &request.catalog_sha,
                &request.support_sha,
                &request.source_sha,
            ] {
                if self.blob(sha, path)? != protected {
                    return Err(reject(
                        Issue::Catalog,
                        format!("catalog bytes differ at {}: {path}", sha.as_str()),
                    ));
                }
            }
            bindings.push(CatalogBinding {
                path,
                sha256: hex::encode(Sha256::digest(protected)),
            });
        }
        Ok(bindings)
    }

    fn validate_policy(&self, request: &Request) -> Checked<()> {
        for path in POLICY {
            let baseline = self.blob(&request.catalog_sha, path)?;
            for sha in [
                &request.support_sha,
                &request.protected_sha,
                &request.source_sha,
            ] {
                if self.blob(sha, path)? != baseline {
                    return Err(reject(
                        Issue::Policy,
                        format!("provider policy change is outside migration scope: {path}"),
                    ));
                }
            }
        }
        Ok(())
    }
}
