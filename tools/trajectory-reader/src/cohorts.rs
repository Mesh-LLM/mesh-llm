//! Whole recorded trajectories, balanced across declared frameworks and cohorts.
use crate::DynResult;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::{BTreeMap, BTreeSet};

pub mod cli;
pub mod document;
pub mod input;
#[cfg(test)]
mod tests;

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct Selection {
    pub cohorts: Vec<String>,
    pub frameworks: Vec<String>,
    pub sources: Vec<String>,
    pub trajectories_per_framework: Option<usize>,
    pub sessions_per_cohort: Option<usize>,
    pub min_isl: u64,
    pub max_isl_exclusive: u64,
    pub min_turns: u64,
}
impl Selection {
    pub fn allocation(&self) -> DynResult<Vec<usize>> {
        for names in [&self.cohorts, &self.frameworks, &self.sources] {
            if names.is_empty()
                || names
                    .iter()
                    .any(|n| n.trim().is_empty() || n.contains(['\n', '\r']))
                || names.iter().collect::<BTreeSet<_>>().len() != names.len()
            {
                return Err(
                    "cohort, framework and source names must be nonempty and unique".into(),
                );
            }
        }
        if self.min_isl == 0 || self.min_turns == 0 || self.max_isl_exclusive <= self.min_isl {
            return Err("selection thresholds must be positive and ordered".into());
        }
        let counts = match (self.trajectories_per_framework, self.sessions_per_cohort) {
            (Some(n), None) if n > 0 => vec![n; self.frameworks.len()],
            (None, Some(n)) if n >= self.frameworks.len() => (0..self.frameworks.len())
                .map(|i| n / self.frameworks.len() + usize::from(i < n % self.frameworks.len()))
                .collect(),
            _ => return Err("provide exactly one positive count covering every framework".into()),
        };
        for count in &counts {
            count
                .checked_mul(self.cohorts.len())
                .ok_or("cohort count overflow")?;
        }
        Ok(counts)
    }
}
#[derive(Debug, Clone, Serialize)]
pub struct Message {
    pub role: String,
    pub content: String,
    pub tool_calls_json: Option<String>,
    pub tool_call_id: Option<String>,
}
#[derive(Debug, Clone, Serialize)]
pub struct RecordedTrajectory {
    pub session_id: String,
    pub source_dataset: String,
    pub agent_framework: String,
    pub recorded_model: Option<String>,
    pub n_turns: u64,
    pub max_isl: u64,
    pub total_tokens: u64,
    pub assistant_turns: usize,
    pub messages: Vec<Message>,
}
pub fn messages(body: &str) -> DynResult<Vec<Message>> {
    let values: Vec<Value> = serde_json::from_str(body)?;
    if values.is_empty() {
        return Err("trajectory has no messages".into());
    }
    let mut output = Vec::with_capacity(values.len());
    for value in values {
        let row = value.as_object().ok_or("message must be an object")?;
        let role = row
            .get("role")
            .and_then(Value::as_str)
            .ok_or("message role missing")?;
        if !["system", "developer", "user", "assistant", "tool"].contains(&role) {
            return Err("unsupported message role".into());
        }
        let text = |key| -> DynResult<Option<String>> {
            match row.get(key) {
                None | Some(Value::Null) => Ok(None),
                Some(Value::String(s)) => Ok(Some(s.clone())),
                _ => Err(format!("{key} must be text or null").into()),
            }
        };
        let tool_calls_json = text("tool_calls_json")?;
        if let Some(calls) = tool_calls_json.as_ref().filter(|s| !s.is_empty()) {
            let calls: Value = serde_json::from_str(calls)?;
            let calls = calls.as_array().ok_or("tool calls must be an array")?;
            if calls.iter().any(|call| !call.is_object()) {
                return Err("tool calls must contain objects".into());
            }
        }
        output.push(Message {
            role: role.into(),
            content: text("content")?.unwrap_or_default(),
            tool_calls_json,
            tool_call_id: text("tool_call_id")?,
        });
    }
    if !output.iter().any(|m| m.role == "assistant") {
        return Err("trajectory has no assistant turns".into());
    }
    Ok(output)
}
pub type Cohorts = BTreeMap<String, Vec<RecordedTrajectory>>;
