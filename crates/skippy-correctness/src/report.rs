use serde::Serialize;

pub use model_artifact::ModelIdentity;

#[derive(Debug, Serialize)]
pub struct BaselineReport {
    pub token_id: i32,
    pub predicted_token: i32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub second_predicted_token: Option<i32>,
}

#[derive(Debug, Serialize)]
pub struct BoundaryReport {
    pub producer_stage_index: i32,
    pub layer_start: i32,
    pub layer_end: i32,
    pub token_count: u32,
    pub payload_bytes: u64,
    pub wire_payload_bytes: usize,
}

#[derive(Debug, Serialize)]
pub struct SplitReport {
    pub token_id: i32,
    pub predicted_token: i32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub second_predicted_token: Option<i32>,
    pub native_mtp: NativeMtpSidebandReport,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub native_mtp_verification: Option<NativeMtpVerificationReport>,
    pub activation_width: i32,
    pub wire_dtype: String,
    pub boundary: BoundaryReport,
}

#[derive(Debug, Clone, Serialize)]
pub struct NativeMtpSidebandReport {
    pub sideband_present: bool,
    pub predicted_token_count: usize,
    pub authoritative_matches_reply: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub authoritative_token: Option<i32>,
    pub draft_token_count: usize,
    pub draft_tokens: Vec<i32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub proposal_compute_us: Option<i64>,
}

#[derive(Debug, Clone, Serialize)]
pub struct NativeMtpVerificationReport {
    pub drafted_tokens: u64,
    pub accepted_tokens: u64,
    pub rejected_tokens: u64,
    pub pending_tokens: u64,
    pub verification_count: u64,
    pub accept_rate: f64,
    pub byte_identical: bool,
    pub draft_tokens: Vec<i32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub second_target_token: Option<i32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub second_baseline_token: Option<i32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub proposal_compute_us: Option<i64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub verification_compute_us: Option<i64>,
}

#[derive(Debug, Serialize)]
pub struct SingleStepReport {
    pub mode: &'static str,
    pub status: &'static str,
    pub model_identity: ModelIdentity,
    pub matches: bool,
    pub native_mtp_draft_required: bool,
    pub baseline: BaselineReport,
    pub split: SplitReport,
    pub stage_models: Vec<StageModelReport>,
}

#[derive(Debug, Serialize)]
pub struct ChainStageReport {
    pub stage_index: u32,
    pub layer_start: u32,
    pub layer_end: u32,
    pub payload_bytes: Option<u64>,
    pub wire_payload_bytes: Option<usize>,
    pub forwarded_over_binary: bool,
    pub returned_predicted_token: bool,
}

#[derive(Debug, Serialize)]
pub struct ChainReport {
    pub mode: &'static str,
    pub status: &'static str,
    pub model_identity: ModelIdentity,
    pub matches: bool,
    pub native_mtp_draft_required: bool,
    pub baseline: BaselineReport,
    pub token_id: i32,
    pub predicted_token: i32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub second_predicted_token: Option<i32>,
    pub native_mtp: NativeMtpSidebandReport,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub native_mtp_verification: Option<NativeMtpVerificationReport>,
    pub activation_width: i32,
    pub wire_dtype: String,
    pub stages: Vec<ChainStageReport>,
    pub stage_models: Vec<StageModelReport>,
}

#[derive(Debug, Serialize, Clone)]
pub struct StageModelReport {
    pub stage_index: u32,
    pub layer_start: u32,
    pub layer_end: u32,
    pub load_mode: &'static str,
    pub model_path: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub package: Option<PackageStageReport>,
}

#[derive(Debug, Serialize, Clone)]
pub struct PackageStageReport {
    pub package_ref: String,
    pub materialized_path: String,
    pub manifest_sha256: String,
    pub selected_parts: Vec<PackagePartReport>,
}

#[derive(Debug, Serialize, Clone)]
pub struct PackagePartReport {
    pub role: String,
    pub layer_index: Option<u32>,
    pub path: String,
    pub sha256: String,
    pub artifact_bytes: u64,
}

#[derive(Debug, Serialize)]
pub struct SplitScanReport {
    pub mode: &'static str,
    pub status: &'static str,
    pub model_identity: ModelIdentity,
    pub baseline: BaselineReport,
    pub split_count: usize,
    pub mismatch_count: usize,
    pub results: Vec<SingleStepReport>,
}

#[derive(Debug, Serialize)]
pub struct DtypeMatrixReport {
    pub mode: &'static str,
    pub status: &'static str,
    pub model_identity: ModelIdentity,
    pub baseline: BaselineReport,
    pub dtype_count: usize,
    pub mismatch_count: usize,
    pub results: Vec<SingleStepReport>,
}

#[derive(Debug, Serialize)]
pub struct StateHandoffReport {
    pub mode: &'static str,
    pub status: &'static str,
    pub model_identity: ModelIdentity,
    pub matches: bool,
    pub predicted_token_matches: bool,
    pub roundtrip_state_matches: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub restored_output_matches: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub suffix_prefill_matches: Option<bool>,
    pub cache_hit_matches: bool,
    pub stage_index: u32,
    pub layer_start: u32,
    pub layer_end: u32,
    pub include_embeddings: bool,
    pub include_output: bool,
    pub handoff_transport: &'static str,
    pub state_payload_kind: &'static str,
    pub borrowed_resident_hits: bool,
    pub cached_decoded_result_hits: bool,
    pub source_predicted_token: i32,
    pub restored_predicted_token: i32,
    pub prompt_token_count: usize,
    pub benchmark_prompt_token_count: usize,
    pub benchmark_prompt_text: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub requested_prefix_token_count: Option<usize>,
    pub activation_width: i32,
    pub state_bytes: usize,
    pub state_bytes_per_prompt_token: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache_storage_bytes: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache_storage_bytes_per_prompt_token: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub resident_state_bytes: Option<usize>,
    pub roundtrip_state_bytes: usize,
    pub payload_digest: StatePayloadDigestReport,
    pub tokenize_ms: f64,
    pub source_prefill_ms: f64,
    pub source_export_ms: f64,
    pub source_decode_ms: f64,
    pub restore_import_ms: f64,
    pub restore_export_ms: f64,
    pub restore_decode_ms: f64,
    pub cache_hit_repeats: usize,
    pub recompute_total_ms: f64,
    pub cache_hit_total_ms: f64,
    pub cache_hit_speedup: f64,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub cache_hit_import_ms: Vec<f64>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub cache_hit_decode_ms: Vec<f64>,
    pub stage_models: Vec<StageModelReport>,
}

#[derive(Debug, Serialize, Clone)]
pub struct StatePayloadDigestReport {
    pub payload_kind: &'static str,
    pub payload_sha256: String,
    pub total_bytes: usize,
    pub kv_bytes: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub kv_sha256: Option<String>,
    pub recurrent_bytes: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub recurrent_sha256: Option<String>,
    pub block_size_bytes: usize,
    pub block_count: usize,
    pub unique_block_count: usize,
    pub duplicate_block_count: usize,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub blocks: Vec<StatePayloadBlockDigestReport>,
}

#[derive(Debug, Serialize, Clone)]
pub struct StatePayloadBlockDigestReport {
    pub component: &'static str,
    pub index: usize,
    pub offset: usize,
    pub bytes: usize,
    pub sha256: String,
}

#[derive(Debug, Serialize)]
pub struct NativeMtpOpenAiAbReport {
    pub mode: &'static str,
    pub status: &'static str,
    pub model_id: String,
    pub model_path: String,
    pub prompt: String,
    pub max_tokens: u32,
    pub split_layer: u32,
    pub layer_end: u32,
    pub activation_width: i32,
    pub activation_wire_dtype: String,
    pub exact_content_match: bool,
    pub batched_events_required: bool,
    pub batched_events_present: bool,
    pub matches: bool,
    pub baseline: NativeMtpOpenAiCaseReport,
    pub n1: NativeMtpOpenAiCaseReport,
    pub batched: NativeMtpOpenAiCaseReport,
}

#[derive(Debug, Serialize)]
pub struct NativeMtpOpenAiCaseReport {
    pub case: &'static str,
    pub native_mtp_enabled: bool,
    pub batched_verify_enabled: bool,
    pub http_status: u16,
    pub content: String,
    pub completion_tokens: Option<u64>,
    pub openai_bind_addr: String,
    pub stage0_bind_addr: String,
    pub stage0_endpoint_addr: String,
    pub stage1_bind_addr: String,
    pub stage1_endpoint_addr: String,
    pub stage0_config: String,
    pub stage1_config: String,
    pub topology_config: String,
    pub stage0_log: String,
    pub stage1_log: String,
    pub stage1_launch_mode: String,
    pub stage1_remote_config: Option<String>,
    pub stage1_remote_topology: Option<String>,
    pub stage1_remote_log: Option<String>,
    pub metrics: NativeMtpOpenAiMetricsReport,
}

#[derive(Debug, Serialize)]
pub struct GlmDsaStage0TraceReport {
    pub mode: &'static str,
    pub status: &'static str,
    pub run_id: String,
    pub model_id: String,
    pub model_path: String,
    pub case_root: String,
    pub stage_layer_end: u32,
    pub activation_width: i32,
    pub activation_wire_dtype: String,
    pub prefill_chunk_size: u32,
    pub max_new_tokens: u32,
    pub trace_filter: String,
    pub both_variants_completed: bool,
    pub fused_prefill_speedup_vs_direct: Option<f64>,
    pub fused_glm_dsa_op_speedup_vs_direct: Option<f64>,
    pub trace_parity: GlmDsaTraceParityReport,
    pub downstream_parity: GlmDsaDownstreamParityReport,
    pub semantic_parity: GlmDsaSemanticParityReport,
    pub variants: Vec<GlmDsaTraceVariantReport>,
}

#[derive(Debug, Serialize)]
pub struct GlmDsaTraceParityReport {
    pub required: bool,
    pub matched: bool,
    pub fused_trace_count: usize,
    pub direct_trace_count: usize,
    pub compared_trace_count: usize,
    pub mismatched_trace_count: usize,
    pub missing_in_fused_count: usize,
    pub missing_in_direct_count: usize,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub mismatches: Vec<GlmDsaTraceParityMismatchReport>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub missing_in_fused: Vec<GlmDsaTraceKeyReport>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub missing_in_direct: Vec<GlmDsaTraceKeyReport>,
}

#[derive(Debug, Clone, Serialize)]
pub struct GlmDsaTraceKeyReport {
    pub tokens: u64,
    pub name: String,
    pub occurrence: usize,
}

#[derive(Debug, Serialize)]
pub struct GlmDsaTraceParityMismatchReport {
    pub key: GlmDsaTraceKeyReport,
    pub reason: String,
    pub fused_stats: Option<String>,
    pub direct_stats: Option<String>,
    pub fused_type: String,
    pub direct_type: String,
    pub fused_shape: [i64; 4],
    pub direct_shape: [i64; 4],
}

#[derive(Debug, Serialize)]
pub struct GlmDsaTraceVariantReport {
    pub variant: &'static str,
    pub direct_sparse_attn: bool,
    pub fused_sparse_mask: bool,
    pub prompt_exit_code: Option<i32>,
    pub prompt_success: bool,
    pub stage_log: String,
    pub prompt_log: String,
    pub fake_downstream_message_count: usize,
    pub fake_downstream_prefill_message_count: usize,
    pub fake_downstream_decode_message_count: usize,
    pub fake_downstream_prefill_token_count: usize,
    pub fake_downstream_top_k_message_count: usize,
    pub fake_downstream_max_top_k_count: usize,
    pub fake_downstream_total_top_k_count: usize,
    pub fake_downstream_total_causal_visible_top_k_count: usize,
    pub fake_downstream_total_active_top_k_window_count: usize,
    pub fake_downstream_total_finite_top_k_count: usize,
    pub fake_downstream_total_padded_top_k_count: usize,
    pub fake_downstream_avg_top_k_per_token: Option<f64>,
    pub fake_downstream_avg_causal_visible_top_k_per_token: Option<f64>,
    pub fake_downstream_avg_active_top_k_window_per_token: Option<f64>,
    pub fake_downstream_avg_finite_top_k_per_token: Option<f64>,
    pub fake_downstream_max_top_k_per_token: Option<f64>,
    pub fake_downstream_top_k_padding_ratio: Option<f64>,
    pub fake_downstream_top_k_sideband_to_hidden_ratio: Option<f64>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub fake_downstream_messages: Vec<GlmDsaDownstreamMessageReport>,
    pub trace_line_count: usize,
    pub timing_line_count: usize,
    pub prompt_prefill_tok_s: Option<f64>,
    pub prompt_decode_tok_s: Option<f64>,
    pub avg_128_token_timing: Option<GlmDsaTimingReport>,
    pub max_128_token_timing: Option<GlmDsaTimingChunkReport>,
    pub last_128_token_timing: Option<GlmDsaTimingChunkReport>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub timing_chunks: Vec<GlmDsaTimingChunkReport>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub timing_group_chunks: Vec<GlmDsaTimingGroupChunkReport>,
}

#[derive(Debug, Clone, Serialize)]
pub struct GlmDsaDownstreamMessageReport {
    pub kind: String,
    pub pos_start: i32,
    pub token_count: i32,
    pub activation_bytes: usize,
    pub activation_sha256: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub activation_f32: Option<GlmDsaActivationStatsReport>,
    pub top_k_count: usize,
    pub top_k_sha256: String,
}

#[derive(Debug, Serialize)]
pub struct GlmDsaDownstreamParityReport {
    pub matched: bool,
    pub fused_message_count: usize,
    pub direct_message_count: usize,
    pub compared_message_count: usize,
    pub mismatched_message_count: usize,
    pub activation_mismatch_count: usize,
    pub top_k_mismatch_count: usize,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub messages: Vec<GlmDsaDownstreamComparisonReport>,
}

#[derive(Debug, Serialize)]
pub struct GlmDsaSemanticParityReport {
    pub matched: bool,
    pub activation_atol: f32,
    pub activation_relative_rmse_tolerance: f64,
    pub activation_within_tolerance: bool,
    pub activation_out_of_tolerance_count: usize,
    pub top_k_exact: bool,
    pub message_metadata_exact: bool,
    pub compared_message_count: usize,
}

#[derive(Debug, Serialize)]
pub struct GlmDsaDownstreamComparisonReport {
    pub index: usize,
    pub fused_kind: String,
    pub direct_kind: String,
    pub fused_pos_start: i32,
    pub direct_pos_start: i32,
    pub fused_token_count: i32,
    pub direct_token_count: i32,
    pub activation_sha256_equal: bool,
    pub top_k_sha256_equal: bool,
    pub top_k_count_equal: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_k_comparison: Option<GlmDsaTopKComparisonReport>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub activation_error: Option<GlmDsaActivationErrorReport>,
}

#[derive(Debug, Serialize)]
pub struct GlmDsaTopKComparisonReport {
    pub compared_count: usize,
    pub mismatch_count: usize,
    pub mismatch_ratio: f64,
    pub active_compared_count: usize,
    pub active_mismatch_count: usize,
    pub active_mismatch_ratio: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub first_mismatch_index: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub first_mismatch_fused: Option<i32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub first_mismatch_direct: Option<i32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub first_active_mismatch_index: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub first_active_mismatch_fused: Option<i32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub first_active_mismatch_direct: Option<i32>,
}

#[derive(Debug, Clone, Serialize)]
pub struct GlmDsaActivationErrorReport {
    pub count: usize,
    pub max_abs_error: f32,
    pub mean_abs_error: f64,
    pub rmse: f64,
    pub relative_rmse: Option<f64>,
    pub max_reference_abs: f32,
    pub non_finite_pair_count: usize,
}

#[derive(Debug, Clone, Serialize)]
pub struct GlmDsaActivationStatsReport {
    pub count: usize,
    pub sum: f64,
    pub mean: f64,
    pub max_abs: f32,
    pub non_finite_count: usize,
}

#[derive(Debug, Serialize)]
pub struct GlmDsaTimingReport {
    pub chunk_count: usize,
    pub total_us: f64,
    pub indexer_topk_us: f64,
    pub sparse_mask_us: f64,
    pub dsa_sparse_attn_us: f64,
    pub mla_attention_us: f64,
}

#[derive(Debug, Clone, Serialize)]
pub struct GlmDsaTimingChunkReport {
    pub index: usize,
    pub tokens: u32,
    pub total_us: f64,
    pub indexer_topk_us: f64,
    pub sparse_mask_us: f64,
    pub dsa_sparse_attn_us: f64,
    pub mla_attention_us: f64,
}

#[derive(Debug, Clone, Serialize)]
pub struct GlmDsaTimingGroupChunkReport {
    pub index: usize,
    pub tokens: u32,
    pub group: String,
    pub total_us: f64,
    pub indexer_topk_us: f64,
    pub sparse_mask_us: f64,
    pub dsa_sparse_attn_us: f64,
    pub mla_attention_us: f64,
}

#[derive(Debug, Default, Serialize)]
pub struct NativeMtpOpenAiMetricsReport {
    pub native_mtp_enabled: bool,
    pub drafted_tokens: u64,
    pub accepted_tokens: u64,
    pub rejected_tokens: u64,
    pub pending_tokens: u64,
    pub verification_count: u64,
    pub accept_rate: f64,
    pub proposal_compute_us: i64,
    pub verification_compute_us: i64,
    pub decode_token_events: u64,
    pub batched_verify_events: u64,
    pub batched_accepted_events: u64,
    pub batched_rejected_events: u64,
    pub batched_accepted_verify_elapsed_ms: f64,
    pub batched_accepted_verify_avg_ms: f64,
    pub batched_rejected_verify_elapsed_ms: f64,
    pub batched_rejected_verify_avg_ms: f64,
    pub batched_consumed_positions: u64,
    pub batched_committed_positions: u64,
    pub batched_trim_count: u64,
    pub batched_trim_elapsed_ms: f64,
    pub batched_trim_local_ms: f64,
    pub batched_trim_downstream_write_ms: f64,
    pub batched_trim_downstream_wait_ms: f64,
    pub fatal_error_events: u64,
}
