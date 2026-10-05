//! Public and internal turn result types.

use crate::arbiter;
use crate::normalize::{self, WorkerOutput};
use crate::session::Session;
use crate::worker::WorkerRole;
use serde_json::Value;

// ─── Turn result ─────────────────────────────────────────────────────

/// Which gateway path produced this turn's response.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TurnKind {
    /// Fan-out path: arbiter decided from full worker outputs.
    Fanout,
    /// Fan-out path with early-exit consensus before all workers returned.
    EarlyExit,
    /// Tool-result turn: skipped fan-out, went straight to reducer.
    ToolResult,
    /// All workers failed and no reducer recovery happened.
    Failed,
}

impl TurnKind {
    /// Lowercase header-friendly label.
    pub fn label(&self) -> &'static str {
        match self {
            Self::Fanout => "fanout",
            Self::EarlyExit => "early-exit",
            Self::ToolResult => "tool-result",
            Self::Failed => "failed",
        }
    }
}

/// What the gateway returns for a single turn.
#[derive(Debug)]
pub struct TurnResult {
    /// OpenAI chat.completion response body.
    pub response_body: Value,
    /// Per-worker details for observability.
    pub worker_summaries: Vec<WorkerSummary>,
    /// Whether the reducer was invoked.
    pub reducer_used: bool,
    /// How many reducer candidates were spawned (0 if reducer didn't run,
    /// 1 on the happy reducer path, ≥2 if the hedge fired or a fast-fail
    /// cascaded to the next candidate).
    pub reducer_attempts: u32,
    /// Which gateway path produced this response.
    pub turn_kind: TurnKind,
    /// Wall-clock time for this turn.
    pub elapsed_ms: u64,
}

#[derive(Debug)]
pub struct WorkerSummary {
    pub model: String,
    pub role: WorkerRole,
    pub succeeded: bool,
    pub elapsed_ms: u64,
    pub output_kind: Option<normalize::OutputKind>,
    pub confidence: Option<f32>,
}

#[derive(Debug, Clone)]
pub(crate) struct ForcedToolChoice {
    pub(crate) name: String,
    pub(crate) fallback_arguments: Value,
}

pub(crate) struct DecisionResolution<'a> {
    pub(crate) session: &'a Session,
    pub(crate) decision: arbiter::Decision,
    pub(crate) outputs: &'a [WorkerOutput],
    pub(crate) has_tools: bool,
    pub(crate) selected_tool_names: &'a [String],
    pub(crate) forced_tool: Option<&'a ForcedToolChoice>,
    pub(crate) allowed_tools: &'a [String],
}
