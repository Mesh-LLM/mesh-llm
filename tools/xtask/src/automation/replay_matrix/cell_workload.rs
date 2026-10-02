use super::recorded_requests::Trajectory;
use serde::Deserialize;

#[derive(Deserialize)]
pub(super) struct Workload {
    pub trajectories: Vec<Trajectory>,
    #[serde(default)]
    pub replay_mode: super::replay_profile::Mode,
    pub model: String,
    pub base_url: String,
    pub concurrency: usize,
    pub max_output_tokens: u64,
    pub request_timeout_seconds: u64,
    #[serde(default)]
    pub qualification_probe: bool,
    pub warmup_turns: Option<usize>,
    #[serde(default)]
    pub following_cells: Vec<CellJob>,
    pub runtime_context: Option<RuntimeContext>,
    pub model_pin: Option<ModelPin>,
    pub eligibility: Option<super::context_eligibility::Budget>,
    pub minimum_recurrent_restored_tokens: Option<u64>,
    pub measured_prefix: Option<super::measured_prefix::Qualification>,
}

impl Workload {
    pub(super) fn mode(&self) -> super::replay_profile::Mode {
        if self.warmup_turns.is_some() || self.qualification_probe {
            super::replay_profile::Mode::All
        } else {
            self.replay_mode
        }
    }

    pub(super) fn validate(&self) -> crate::command::DynResult<()> {
        if self.mode() != super::replay_profile::Mode::All
            && (self.eligibility.is_some()
                || self.measured_prefix.is_some()
                || self.minimum_recurrent_restored_tokens.is_some()
                || self.runtime_context.is_some()
                || self.model_pin.is_some())
        {
            return Err(
                "selected replay profiles cannot certify full-session context or recurrent state"
                    .into(),
            );
        }
        if self.trajectories.is_empty()
            || !(1..=256).contains(&self.concurrency)
            || !(1..=86400).contains(&self.request_timeout_seconds)
            || self.max_output_tokens == 0
            || self.warmup_turns == Some(0)
        {
            return Err("cell requires trajectories, positive output/warmup budgets, concurrency in 1..=256 and timeout in 1..=86400".into());
        }
        if self.warmup_turns.is_some()
            && (self.eligibility.is_some()
                || self.measured_prefix.is_some()
                || self.minimum_recurrent_restored_tokens.is_some())
        {
            return Err("warmup cannot certify full-session qualification".into());
        }
        let mut sessions = std::collections::BTreeSet::new();
        let selection = super::recorded_requests::Selection {
            model: &self.model,
            maximum_output_tokens: self.max_output_tokens,
            turn_limit: None,
            qualification_probe: self.qualification_probe,
        };
        let mut available = 0_usize;
        for trajectory in &self.trajectories {
            if !sessions.insert(&trajectory.session_id) {
                return Err("duplicate session IDs".into());
            }
            available = available
                .checked_add(super::recorded_requests::build(trajectory, &selection)?.len())
                .ok_or("recorded turn count overflow")?;
        }
        if self.warmup_turns.is_some_and(|turns| turns > available) {
            return Err("warmup cohort has insufficient recorded turns".into());
        }
        Ok(())
    }
}

#[derive(Deserialize)]
pub(super) struct ModelPin {
    pub sha256: String,
    pub minimum_context_tokens: u64,
    pub output: std::path::PathBuf,
}

#[derive(Deserialize)]
pub(super) struct RuntimeContext {
    pub required_tokens: u64,
    pub output: std::path::PathBuf,
}

#[derive(Deserialize)]
pub(super) struct CellJob {
    pub workload: Workload,
    pub requests_output: std::path::PathBuf,
    pub summary_output: std::path::PathBuf,
}
