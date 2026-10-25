//! A fresh blobless mirror of llama.cpp holding both pins with complete
//! history. Any shallow, missing or incomplete state fails closed.

use super::git::{self, PinGuardError};
use crate::command::unique_temp_dir;
use std::path::{Path, PathBuf};
use std::time::Duration;

const FETCH_TIMEOUT: Duration = Duration::from_secs(300);

/// A bare mirror in a private temporary directory, removed on drop.
pub(super) struct UpstreamMirror {
    checkout: PathBuf,
}

impl Drop for UpstreamMirror {
    fn drop(&mut self) {
        let _cleanup = std::fs::remove_dir_all(&self.checkout);
    }
}

impl UpstreamMirror {
    pub(super) fn repository(&self) -> PathBuf {
        self.checkout.join("upstream.git")
    }

    pub(super) fn fetch(
        url: &str,
        base_pin: &str,
        proposed_pin: &str,
    ) -> Result<Self, PinGuardError> {
        let mirror = Self {
            checkout: unique_temp_dir("mesh-llm-llama-pin"),
        };
        let upstream = mirror.repository();
        std::fs::create_dir_all(&upstream)
            .map_err(|error| PinGuardError(format!("cannot create llama.cpp mirror: {error}")))?;
        git::checked(&upstream, &["init", "--bare", "--quiet"])?;
        git::checked(&upstream, &["remote", "add", "origin", url])?;
        let fetch = [
            "-c",
            "protocol.version=2",
            "fetch",
            "--no-tags",
            "--filter=blob:none",
            "origin",
            base_pin,
            proposed_pin,
        ];
        let fetched = git::run(&upstream, &fetch, Some(FETCH_TIMEOUT))?;
        if !fetched.success() {
            return Err(PinGuardError(format!(
                "unable to fetch both llama.cpp upstream pins; the guard cannot prove ancestry and will fail closed: {}",
                fetched.detail()
            )));
        }
        unshallow(&upstream)?;
        for pin in [base_pin, proposed_pin] {
            let object = format!("{pin}^{{commit}}");
            if !git::run(&upstream, &["cat-file", "-e", &object], None)?.success() {
                return Err(PinGuardError(format!(
                    "llama.cpp upstream object {pin} is unavailable after fetch; cannot prove ancestry"
                )));
            }
        }
        let missing = git::run(
            &upstream,
            &["rev-list", "--missing=print", base_pin, proposed_pin],
            None,
        )?;
        if !missing.success() {
            return Err(PinGuardError(format!(
                "cannot walk llama.cpp upstream history: {}",
                missing.detail()
            )));
        }
        if missing.stdout.lines().any(|line| line.starts_with('?')) {
            return Err(PinGuardError(
                "llama.cpp upstream history is incomplete after fetch; cannot prove ancestry"
                    .to_owned(),
            ));
        }
        Ok(mirror)
    }
}

/// The fetch has no depth limit; this covers servers that still answer
/// shallow.
fn unshallow(upstream: &Path) -> Result<(), PinGuardError> {
    let is_shallow = |upstream: &Path| -> Result<bool, PinGuardError> {
        let probe = git::checked(upstream, &["rev-parse", "--is-shallow-repository"])?;
        Ok(probe.stdout.trim() == "true")
    };
    if !is_shallow(upstream)? {
        return Ok(());
    }
    let fetched = git::run(
        upstream,
        &["fetch", "--no-tags", "--unshallow", "origin"],
        Some(FETCH_TIMEOUT),
    )?;
    if !fetched.success() || is_shallow(upstream)? {
        return Err(PinGuardError(format!(
            "llama.cpp upstream history is shallow after fetch; cannot prove ancestry: {}",
            fetched.detail()
        )));
    }
    Ok(())
}
