//! typed KV result and aggregate evidence contracts.
use super::{kv_cache::Metrics, transport::millis};
use serde::Serialize;
use std::{collections::BTreeMap, time::Instant};
#[derive(Serialize)]
pub(super) struct Row {
    pub model: String,
    pub attempt: u32,
    pub phase: &'static str,
    pub ok: bool,
    pub detail: String,
    pub elapsed_ms: u64,
    pub status_code: Option<u16>,
    pub prompt_tokens: Option<u64>,
    pub cached_tokens: Option<u64>,
}
impl Row {
    pub fn new(
        identity: (&str, u32, &'static str),
        started: Instant,
        status_code: Option<u16>,
        metrics: Option<Metrics>,
        result: Result<String, String>,
    ) -> Self {
        let (model, attempt, phase) = identity;
        let (ok, detail) = match result {
            Ok(detail) => (true, detail),
            Err(detail) => (false, detail),
        };
        Self {
            model: model.into(),
            attempt,
            phase,
            ok,
            detail,
            elapsed_ms: millis(started),
            status_code,
            prompt_tokens: metrics.map(|metrics| metrics.prompt_tokens),
            cached_tokens: metrics.map(|metrics| metrics.cached_tokens),
        }
    }
}
#[derive(Default, Serialize)]
pub(super) struct Counts {
    pub passed: usize,
    pub failed: usize,
    pub total: usize,
}
#[derive(Serialize)]
pub(super) struct Summary {
    pub ok: bool,
    pub total: usize,
    pub passed: usize,
    pub failed: usize,
    pub cancelled: bool,
    pub phases: BTreeMap<&'static str, Counts>,
}
pub(super) fn summarize(rows: &[Row], cancelled: bool) -> Summary {
    let mut phases = BTreeMap::<&'static str, Counts>::new();
    for row in rows {
        let counts = phases.entry(row.phase).or_default();
        counts.total += 1;
        if row.ok {
            counts.passed += 1;
        } else {
            counts.failed += 1;
        }
    }
    let passed = rows.iter().filter(|row| row.ok).count();
    Summary {
        ok: !cancelled && !rows.is_empty() && passed == rows.len(),
        total: rows.len(),
        passed,
        failed: rows.len() - passed,
        cancelled,
        phases,
    }
}
pub(super) fn markdown(summary: &Summary, rows: &[Row]) -> String {
    let status = if summary.ok { "PASS" } else { "FAIL" };
    let mut text = format!(
        "# KV Tool-Loop Stability Summary\n\nStatus: **{status}**\n\nCancelled: {}\n\n| Phase | Passed | Failed | Total |\n|---|---:|---:|---:|\n",
        summary.cancelled
    );
    for (phase, counts) in &summary.phases {
        text.push_str(&format!(
            "| {phase} | {} | {} | {} |\n",
            counts.passed, counts.failed, counts.total
        ));
    }
    text.push_str("\n| Result | Model | Attempt | Phase | Detail |\n|---|---|---:|---|---|\n");
    for row in rows {
        text.push_str(&format!(
            "| {} | {} | {} | {} | {} |\n",
            if row.ok { "PASS" } else { "FAIL" },
            cell(&row.model),
            row.attempt,
            row.phase,
            cell(&row.detail)
        ));
    }
    text
}
fn cell(value: &str) -> String {
    value.replace('|', "\\|").replace(['\r', '\n'], " ")
}
