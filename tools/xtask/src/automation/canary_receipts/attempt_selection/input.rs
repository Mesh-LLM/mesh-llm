use super::*;
use serde::Deserialize;

#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub(super) enum JobResult {
    Success,
    Failure,
    Cancelled,
    Skipped,
}
impl JobResult {
    pub(super) fn parse(value: &str) -> DynResult<Self> {
        match value {
            "success" => Ok(Self::Success),
            "failure" => Ok(Self::Failure),
            "cancelled" => Ok(Self::Cancelled),
            "skipped" => Ok(Self::Skipped),
            _ => Err("unknown canary job result".into()),
        }
    }
}
#[derive(Debug, Deserialize)]
pub(super) struct Job {
    pub(super) result: JobResult,
    #[serde(default)]
    pub(super) outputs: JobOutputs,
}
impl Job {
    pub(super) fn validate(&self) -> DynResult<()> {
        if self.outputs.green == "true" && self.outputs.repairable == "true" {
            return Err("canary job cannot claim green and repair admission together".into());
        }
        if matches!(self.result, JobResult::Cancelled | JobResult::Skipped)
            && (self.outputs.green == "true" || self.outputs.repairable == "true")
        {
            return Err(
                "cancelled or skipped canary jobs cannot claim green or repair admission".into(),
            );
        }
        for value in [&self.outputs.green, &self.outputs.repairable] {
            if !matches!(value.as_str(), "" | "true" | "false") {
                return Err("canary job output boolean must be true, false or absent".into());
            }
        }
        Ok(())
    }
}
#[derive(Debug, Default, Deserialize)]
#[serde(default)]
pub(super) struct JobOutputs {
    pub(super) state: String,
    pub(super) green: String,
    pub(super) repairable: String,
    pub(super) package: String,
    pub(super) identity: String,
    pub(super) head: String,
    pub(super) branch: String,
    pub(super) feedback: String,
    pub(super) failure_class: String,
    pub(super) failure_stage: String,
}
#[derive(Debug, Deserialize)]
pub(super) struct Attempts {
    #[serde(default)]
    attempt_1: Option<Job>,
    #[serde(default)]
    attempt_2: Option<Job>,
    #[serde(default)]
    attempt_3: Option<Job>,
    #[serde(flatten)]
    other: BTreeMap<String, serde::de::IgnoredAny>,
}
impl Attempts {
    pub(super) fn validate_slots(&self) -> DynResult<()> {
        if self.other.keys().any(|key| key.starts_with("attempt_")) {
            return Err("canary selection supports only attempt_1, attempt_2 and attempt_3".into());
        }
        for job in [&self.attempt_1, &self.attempt_2, &self.attempt_3]
            .into_iter()
            .flatten()
        {
            job.validate()?;
        }
        Ok(())
    }
    pub(super) fn latest(&self) -> DynResult<&Job> {
        for job in [&self.attempt_3, &self.attempt_2, &self.attempt_1]
            .into_iter()
            .flatten()
        {
            if !job.outputs.state.is_empty() || job.result != JobResult::Skipped {
                return Ok(job);
            }
        }
        Err("no distributed canary attempt completed".into())
    }
}
pub(super) fn head(value: &str) -> DynResult<String> {
    if value.len() != 40
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        return Err("canary head must be a lowercase full commit SHA".into());
    }
    Ok(value.to_owned())
}
pub(super) fn digest(value: &str) -> DynResult<String> {
    crate::automation::canary_receipts::Digest::try_from(value.to_owned())
        .map_err(|error| format!("invalid canary identity: {error}"))?;
    Ok(value.to_owned())
}
