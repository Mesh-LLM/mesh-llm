//! Full quantization request; local paths carry supplied bytes, never cloud qualification.
use super::*;
use serde::{Deserialize, Serialize};
#[derive(Clone, Copy, Deserialize, Serialize)]
#[serde(rename_all = "kebab-case")]
pub(in crate::automation::hf_certify) enum Workflow {
    Quantization,
    QuantizationAndPackage,
}
impl Workflow {
    pub(super) fn allowance(&self) -> u64 {
        match self {
            Self::Quantization => 259200,
            Self::QuantizationAndPackage => 345600,
        }
    }
}
#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(in crate::automation::hf_certify) struct ResumeEntry {
    pub ordinal: u32,
    pub resume: window::contract::Resume,
}
#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(in crate::automation::hf_certify) struct Package {
    pub writer: admission::Artifact,
    pub writer_source: admission::Artifact,
    pub generation_defaults: admission::Artifact,
    pub target_repo: String,
    pub max_artifact_bytes: u64,
}
#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(in crate::automation::hf_certify) struct Input {
    pub schema_version: u32,
    pub workflow: Workflow,
    pub timeout_seconds: u64,
    pub window_template: window::contract::Input,
    pub resumes: Vec<ResumeEntry>,
    pub loader: admission::Artifact,
    pub package: Option<Package>,
}
fn repo(s: &str) -> bool {
    let pieces: Vec<_> = s.split('/').collect();
    pieces.len() == 2
        && pieces.iter().all(|s| {
            !s.is_empty()
                && s.len() <= 96
                && !matches!(*s, "." | "..")
                && s.bytes()
                    .all(|b| b.is_ascii_alphanumeric() || b"._-".contains(&b))
        })
}
impl Input {
    pub(in crate::automation::hf_certify) fn validate(&self) -> DynResult<()> {
        let w = &self.window_template;
        w.validate()?;
        if self.schema_version != 1
            || !(5..=self.workflow.allowance()).contains(&self.timeout_seconds)
            || w.ordinal != 1
            || w.resume.is_some()
            || self.resumes.len() > w.expected_splits as usize
            || matches!(self.workflow, Workflow::QuantizationAndPackage) != self.package.is_some()
        {
            return Err("quant job workflow/template/allowance correlation refused".into());
        }
        let mut ordinals = std::collections::BTreeSet::new();
        for r in &self.resumes {
            if !(1..=w.expected_splits).contains(&r.ordinal)
                || !ordinals.insert(r.ordinal)
                || !bootstrap::contract::hex(&r.resume.commit, 40)
            {
                return Err("quant job duplicate/out-of-roster resume refused".into());
            }
        }
        for r in &self.resumes {
            self.window(r.ordinal)?.validate()?;
        }
        let mut additional = vec![&self.loader];
        if let Some(p) = &self.package {
            if !repo(&p.target_repo)
                || p.target_repo == w.target_repo
                || !(1..=1024_u64.pow(4)).contains(&p.max_artifact_bytes)
            {
                return Err("quant package target/size authority refused".into());
            }
            additional.extend([&p.writer, &p.writer_source, &p.generation_defaults]);
        }
        for a in additional {
            if !a.path.is_absolute()
                || !bootstrap::contract::hex(&a.sha256, 64)
                || a.path.starts_with(&w.target_root)
                || a.path.starts_with(&w.work_root)
            {
                return Err("quant loader/package immutable pin refused".into());
            }
        }
        Ok(())
    }
    pub(super) fn window(&self, ordinal: u32) -> DynResult<window::contract::Input> {
        let mut value = serde_json::to_value(&self.window_template)?;
        value["ordinal"] = json!(ordinal);
        if let Some(r) = self.resumes.iter().find(|r| r.ordinal == ordinal) {
            value["resume"] = serde_json::to_value(&r.resume)?;
        }
        Ok(serde_json::from_value(value)?)
    }
}
