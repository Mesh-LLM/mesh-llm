//! The claimed models the audit worker keeps open between audits.
//!
//! Opening a claimed model reads its GGUF header from Hugging Face by
//! range and loads a layer here, which takes seconds, so the worker keeps
//! the last [`KEPT_MODELS`] it audited against. A kept model holds a loaded
//! layer and open files, so one that no audit used for [`MODEL_IDLE`] is
//! released, and all of them when a job panicked or the audits end.
//!
//! The file to open is never a name out of a receipt: it is the one
//! `requested_model` built from the request's own model name. When the
//! request named no commit, the receipt's commit must be in the history of
//! the branch the request named, which is checked on every audit, before a
//! file at a new commit is opened.

use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use skippy_receipt_audit::claimed_file::{self, HfFile};
use skippy_receipt_audit::native_model::load_claimed_model;
use skippy_receipt_audit::turn_audit::ClaimedSetup;

/// Claimed models kept open between audits.
const KEPT_MODELS: usize = 2;
/// How long a claimed model stays open without an audit that uses it.
pub(super) const MODEL_IDLE: Duration = Duration::from_secs(10 * 60);

/// Opens the claimed model `file` names on Hugging Face (or `HF_ENDPOINT`)
/// and loads it here. An error is this node's: Hugging Face is out of
/// reach, the file is not a GGUF it reads, the native runtime is missing
/// or does not load the model.
/// How an audit worker opens a claimed model.
pub(super) type Opener = fn(&HfFile) -> Result<ClaimedSetup>;

pub(super) fn open_from_hub(file: &HfFile) -> Result<ClaimedSetup> {
    let claimed = claimed_file::open_remote(file, &claimed_file::default_cache_root()?)?;
    let setup = ClaimedSetup::from_file(claimed, &load_claimed_model)?;
    // a node that cannot load the model here cannot check it
    setup
        .model
        .detokenize(&[])
        .context("load the claimed model here")?;
    Ok(setup)
}

struct Kept {
    name: String,
    setup: ClaimedSetup,
    used: Instant,
}

/// Checks that the commit of a file is in the history of a branch.
type OnBranch = fn(&HfFile, &str) -> Result<()>;

/// The claimed models an audit worker holds, opened by `open`.
pub(super) struct ClaimedModels<O> {
    open: O,
    on_branch: OnBranch,
    kept: Vec<Kept>,
}

impl<O: Fn(&HfFile) -> Result<ClaimedSetup>> ClaimedModels<O> {
    /// Models `open` opens, whose commits Hugging Face places on a branch.
    pub(super) fn opened_by(open: O) -> Self {
        Self::checked_by(open, claimed_file::check_on_branch)
    }

    fn checked_by(open: O, on_branch: OnBranch) -> Self {
        Self {
            open,
            on_branch,
            kept: Vec::new(),
        }
    }

    /// The claimed model `file` names, opened now or kept from an earlier
    /// audit. `now` is when the audit uses it. With a `branch`, the file's
    /// commit must be in its history first ([`claimed_file::check_on_branch`]):
    /// a commit that is not is a [`claimed_file::NotOnBranch`] error, and
    /// nothing of the file is read.
    pub(super) fn get(
        &mut self,
        file: &HfFile,
        branch: Option<&str>,
        now: Instant,
    ) -> Result<&ClaimedSetup> {
        if let Some(branch) = branch {
            (self.on_branch)(file, branch)?;
        }
        let name = file.to_string();
        let at = match self.kept.iter().position(|kept| kept.name == name) {
            Some(at) => at,
            None => {
                let setup = (self.open)(file)?;
                if self.kept.len() >= KEPT_MODELS {
                    // the one used longest ago
                    let stalest = (0..self.kept.len()).min_by_key(|&at| self.kept[at].used);
                    self.kept.remove(stalest.unwrap_or(0));
                }
                self.kept.push(Kept {
                    name,
                    setup,
                    used: now,
                });
                self.kept.len() - 1
            }
        };
        self.kept[at].used = now;
        Ok(&self.kept[at].setup)
    }

    /// Releases the models no audit used for `idle` before `now`.
    pub(super) fn release_idle(&mut self, now: Instant, idle: Duration) {
        self.kept.retain(|kept| {
            let keep = now.saturating_duration_since(kept.used) < idle;
            if !keep {
                tracing::debug!(
                    "receipt audits: released the idle claimed model {}",
                    kept.name
                );
            }
            keep
        });
    }

    /// Releases every model.
    pub(super) fn release_all(&mut self) {
        self.kept.clear();
    }

    #[cfg(test)]
    pub(super) fn kept(&self) -> Vec<&str> {
        self.kept.iter().map(|kept| kept.name.as_str()).collect()
    }
}

#[cfg(test)]
mod tests {
    use std::cell::Cell;

    use super::*;
    use crate::network::receipt_audit::test_support::{hf_file, toy_setup};

    #[test]
    fn a_model_is_opened_once_and_the_stalest_gives_way() {
        let opened = Cell::new(0);
        let state = tempfile::tempdir().unwrap();
        let mut models = ClaimedModels::opened_by(|_: &HfFile| {
            opened.set(opened.get() + 1);
            Ok(toy_setup(state.path()))
        });
        let start = Instant::now();
        let at = |seconds: u64| start + Duration::from_secs(seconds);
        let (a, b, c) = (hf_file("a"), hf_file("b"), hf_file("c"));
        models.get(&a, None, at(0)).unwrap();
        models.get(&b, None, at(1)).unwrap();
        models.get(&a, None, at(2)).unwrap();
        assert_eq!(opened.get(), 2, "the kept model is not opened again");
        // a third model takes the place of the one used longest ago
        models.get(&c, None, at(3)).unwrap();
        assert_eq!(models.kept(), [a.to_string(), c.to_string()]);
        models.get(&b, None, at(4)).unwrap();
        assert_eq!(opened.get(), 4);
    }

    #[test]
    fn a_model_that_does_not_open_is_not_kept() {
        let mut models = ClaimedModels::opened_by(|file: &HfFile| {
            anyhow::bail!("Hugging Face did not answer for {file}")
        });
        let error = models
            .get(&hf_file("a"), None, Instant::now())
            .err()
            .unwrap();
        assert!(format!("{error:#}").contains("did not answer"));
        assert!(models.kept().is_empty());
    }

    /// A commit off the branch the request named opens nothing, kept or
    /// not; one on it opens once.
    #[test]
    fn a_commit_off_the_branch_opens_nothing() {
        let opened = Cell::new(0);
        let state = tempfile::tempdir().unwrap();
        let on_main = |file: &HfFile, branch: &str| -> Result<()> {
            if file.file == "a.gguf" && branch == "main" {
                return Ok(());
            }
            Err(anyhow::anyhow!("{file} is not on {branch}"))
        };
        let mut models = ClaimedModels::checked_by(
            |_: &HfFile| {
                opened.set(opened.get() + 1);
                Ok(toy_setup(state.path()))
            },
            on_main,
        );
        let now = Instant::now();
        let (a, b) = (hf_file("a"), hf_file("b"));
        models.get(&a, Some("main"), now).unwrap();
        assert!(models.get(&b, Some("main"), now).is_err());
        // a model kept is checked again for another branch
        assert!(models.get(&a, Some("dev"), now).is_err());
        // a commit the request named needs no branch
        models.get(&b, None, now).unwrap();
        assert_eq!(opened.get(), 2);
    }

    #[test]
    fn idle_models_are_released() {
        let state = tempfile::tempdir().unwrap();
        let mut models = ClaimedModels::opened_by(|_: &HfFile| Ok(toy_setup(state.path())));
        let start = Instant::now();
        let (a, b) = (hf_file("a"), hf_file("b"));
        models.get(&a, None, start).unwrap();
        models
            .get(&b, None, start + Duration::from_secs(400))
            .unwrap();
        models.release_idle(start + Duration::from_secs(599), MODEL_IDLE);
        assert_eq!(models.kept().len(), 2, "neither is idle for ten minutes");
        models.release_idle(start + Duration::from_secs(600), MODEL_IDLE);
        assert_eq!(models.kept(), [b.to_string()]);
        // an audit that uses a model keeps it
        models
            .get(&b, None, start + Duration::from_secs(900))
            .unwrap();
        models.release_idle(start + Duration::from_secs(1400), MODEL_IDLE);
        assert_eq!(models.kept().len(), 1);
        models.release_all();
        assert!(models.kept().is_empty());
    }
}
