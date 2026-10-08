use crate::automation::canary_receipts::pass_identity::PassId;
use crate::command::DynResult;
use serde::Deserialize;
use std::path::PathBuf;

#[derive(Clone, Copy, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub(super) enum Mode {
    #[serde(rename = "repair-build")]
    Repair,
    #[serde(rename = "verify-build")]
    Verify,
    #[serde(rename = "pinned-build")]
    Pinned,
}
impl Mode {
    pub(super) fn text(self) -> &'static str {
        match self {
            Self::Repair => "repair-build",
            Self::Verify => "verify-build",
            Self::Pinned => "pinned-build",
        }
    }
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Previous {
    pub(super) package: PathBuf,
    pub(super) identity: String,
    pub(super) candidate: String,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Input {
    pub(super) controller_root: PathBuf,
    pub(super) source_root: PathBuf,
    pub(super) controller_revision: String,
    pub(super) selected_revision: String,
    pub(super) mesh_source: String,
    pub(super) upstream_revision: String,
    pub(super) mode: Mode,
    pub(super) pass_id: String,
    pub(super) run_id: String,
    pub(super) run_attempt: String,
    pub(super) previous: Option<Previous>,
    pub(super) previous_feedback: Option<PathBuf>,
    pub(super) evidence: PathBuf,
    pub(super) export: PathBuf,
    pub(super) agent_timeout_seconds: u64,
    pub(super) verification_timeout_seconds: u64,
}

impl Input {
    pub(super) fn validate(&mut self) -> DynResult<()> {
        for revision in [
            &self.controller_revision,
            &self.selected_revision,
            &self.upstream_revision,
        ] {
            revision_valid(revision)?;
        }
        if !self.mesh_source.is_empty() {
            revision_valid(&self.mesh_source)?;
            if self.mesh_source != self.selected_revision
                || !matches!(self.mode, Mode::Pinned)
                || self.previous.is_some()
            {
                return Err(
                    "selected source requires unchanged pinned-build without previous candidate"
                        .into(),
                );
            }
        } else if self.selected_revision != self.controller_revision {
            return Err("ordinary canary source must match the frozen controller revision".into());
        }
        let pass = PassId::parse(&self.pass_id)?;
        crate::automation::canary_receipts::RunAttempt::try_from(self.run_attempt.clone())?;
        if self.run_id.is_empty() || !self.run_id.bytes().all(|byte| byte.is_ascii_digit()) {
            return Err("invalid canary workflow run identity".into());
        }
        if !(1..=41400).contains(&self.agent_timeout_seconds)
            || !(1..=43200).contains(&self.verification_timeout_seconds)
        {
            return Err("canary coding and verification budgets exceed approved ceilings".into());
        }
        let first = matches!(pass, PassId::Repair1);
        let valid = match self.mode {
            Mode::Repair if first => self.previous.is_none() && self.previous_feedback.is_none(),
            Mode::Repair => {
                pass.is_repair() && self.previous.is_some() && self.previous_feedback.is_some()
            }
            Mode::Verify => {
                !pass.is_repair() && self.previous.is_some() && self.previous_feedback.is_none()
            }
            Mode::Pinned => first && self.previous.is_none() && self.previous_feedback.is_none(),
        };
        if !valid {
            return Err(
                "canary mode/pass requires its exact candidate and feedback dependencies".into(),
            );
        }
        for checkout in [&mut self.controller_root, &mut self.source_root] {
            if !checkout.is_absolute() || !checkout.is_dir() {
                return Err("canary checkout must exist and be absolute".into());
            }
            *checkout = checkout.canonicalize()?;
        }
        for output in [&mut self.evidence, &mut self.export] {
            if !output.is_absolute() || output.exists() {
                return Err("canary evidence and export must be new absolute directories".into());
            }
            let parent = output
                .parent()
                .ok_or("canary output has no parent")?
                .canonicalize()?;
            let resolved = parent.join(output.file_name().ok_or("canary output has no basename")?);
            if resolved.starts_with(&self.source_root)
                || resolved.starts_with(&self.controller_root)
            {
                return Err("canary output must be outside both checkouts".into());
            }
            *output = resolved;
        }
        if self.evidence == self.export
            || self.evidence.starts_with(&self.export)
            || self.export.starts_with(&self.evidence)
        {
            return Err("canary evidence and export directories must be disjoint".into());
        }
        Ok(())
    }
}

pub(super) fn revision_valid(value: &str) -> DynResult<()> {
    if value.len() != 40
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        return Err("expected lowercase 40-hex canary revision".into());
    }
    Ok(())
}
