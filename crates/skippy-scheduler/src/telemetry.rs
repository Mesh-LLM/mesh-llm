#[derive(Debug, Clone, Default, PartialEq)]
pub struct SchedulerMetrics {
    pub iterations: u64,
    pub admitted: u64,
    pub preempted: u64,
    pub finished: u64,
    pub failed: u64,
    pub cancelled: u64,
    pub rejected_overload: u64,
    pub prefill_tokens: u64,
    pub recompute_tokens: u64,
    pub decode_tokens: u64,
    pub prefix_hits: u64,
    pub prefix_misses: u64,
    pub active_sequences: usize,
    pub waiting_sequences: usize,
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct IterationTelemetry {
    pub iteration: u64,
    pub active_sequences: usize,
    pub waiting_sequences: usize,
    pub admitted: usize,
    pub preempted: usize,
    pub prefill_tokens: usize,
    pub recompute_tokens: usize,
    pub decode_tokens: usize,
    pub component_used_bytes: Vec<(String, u64)>,
    pub component_available_bytes: Vec<(String, u64)>,
    pub prefix_hits: u64,
    pub prefix_misses: u64,
    pub finished: u64,
    pub failed: u64,
    pub cancelled: u64,
    pub rejected_overload: u64,
}
