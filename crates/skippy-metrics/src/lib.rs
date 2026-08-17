pub mod attr {
    pub const RUN_ID: &str = "skippy.run_id";
    pub const NODE_ID: &str = "skippy.node_id";
    pub const CLUSTER_ID: &str = "skippy.cluster_id";
    pub const MODEL_ID: &str = "skippy.model_id";
    pub const TOPOLOGY_ID: &str = "skippy.topology_id";
    pub const REQUEST_ID: &str = "skippy.request_id";
    pub const SESSION_ID: &str = "skippy.session_id";
    pub const PROMPT_INDEX: &str = "skippy.prompt_index";
    pub const MESSAGE_KIND: &str = "skippy.message_kind";
    pub const TOKEN_COUNT: &str = "skippy.token_count";
    pub const CHECKPOINT_GENERATION: &str = "skippy.checkpoint_generation";
    pub const PROMPT_TOKEN_COUNT: &str = "skippy.prompt_token_count";
    pub const DECODE_STEP: &str = "skippy.decode_step";
    pub const STAGE_ID: &str = "skippy.stage_id";
    pub const STAGE_INDEX: &str = "skippy.stage_index";
    pub const LAYER_START: &str = "skippy.layer_start";
    pub const LAYER_END: &str = "skippy.layer_end";
    pub const LOAD_MODE: &str = "skippy.load_mode";
    pub const KV_PROACTIVE_EVICTION_STATUS: &str = "skippy.kv.proactive_eviction_status";
    pub const KV_PROACTIVE_EVICTION_ERROR_KIND: &str = "skippy.kv.proactive_eviction_error_kind";
    pub const KV_PROACTIVE_EVICTION_TARGET_TOKENS: &str =
        "skippy.kv.proactive_eviction_target_tokens";
    pub const KV_PROACTIVE_EVICTED_ENTRIES: &str = "skippy.kv.proactive_evicted_entries";
    pub const KV_PROACTIVE_EVICTED_TOKENS: &str = "skippy.kv.proactive_evicted_tokens";
    pub const VERIFY_WINDOW_DIRECT_RETURN_UPSTREAM_OPENED: &str =
        "llama_stage.verify_window.direct_return_upstream_opened";
    pub const VERIFY_WINDOW_DIRECT_RETURN_REVERSE_FALLBACK: &str =
        "llama_stage.verify_window.direct_return_reverse_fallback";
}

pub mod metric {
    pub const LLAMA_DECODE_SECONDS: &str = "skippy.llama_decode_seconds";
    pub const ACTIVATION_BYTES_SENT: &str = "skippy.activation_bytes_sent";
    pub const OTEL_QUEUE_DEPTH: &str = "skippy.otel_queue_depth";
    pub const OTEL_DROPPED_EVENTS: &str = "skippy.otel_dropped_events";
    pub const OTEL_EXPORT_ERRORS: &str = "skippy.otel_export_errors";
    pub const KV_LOCAL_PAGE_BYTES: &str = "skippy.kv.local_page_bytes";
    pub const KV_EVICTABLE_PAGE_BYTES: &str = "skippy.kv.evictable_page_bytes";
    pub const KV_READY_PAGES: &str = "skippy.kv.ready_pages";
    pub const KV_PINNED_PAGES: &str = "skippy.kv.pinned_pages";
    pub const KV_REMOTE_ONLY_PAGES: &str = "skippy.kv.remote_only_pages";
    pub const KV_PAGE_HITS: &str = "skippy.kv.page_hits";
    pub const KV_PAGE_MISSES: &str = "skippy.kv.page_misses";
    pub const KV_COMMITTED_PAGES: &str = "skippy.kv.committed_pages";
    pub const KV_IMPORTED_PAGES: &str = "skippy.kv.imported_pages";
    pub const KV_EVICTED_PAGES: &str = "skippy.kv.evicted_pages";
    pub const KV_LOOKUP_REQUESTS: &str = "skippy.kv.lookup_requests";
    pub const KV_LOOKUP_HITS: &str = "skippy.kv.lookup_hits";
    pub const KV_LOOKUP_MISSES: &str = "skippy.kv.lookup_misses";
    pub const KV_LOOKUP_MICROS: &str = "skippy.kv.lookup_micros";
    pub const KV_ATTACH_REQUESTS: &str = "skippy.kv.attach_requests";
    pub const KV_ATTACH_MICROS: &str = "skippy.kv.attach_micros";
    pub const KV_RESERVE_REQUESTS: &str = "skippy.kv.reserve_requests";
    pub const KV_RESERVE_MICROS: &str = "skippy.kv.reserve_micros";
    pub const KV_COMMIT_REQUESTS: &str = "skippy.kv.commit_requests";
    pub const KV_COMMIT_MICROS: &str = "skippy.kv.commit_micros";
    pub const KV_DROP_SESSION_REQUESTS: &str = "skippy.kv.drop_session_requests";
    pub const KV_DROP_SESSION_MICROS: &str = "skippy.kv.drop_session_micros";
    pub const KV_LOCAL_PROTOCOL_ERRORS: &str = "skippy.kv.local_protocol_errors";
    pub const KV_PEER_TRANSFER_ATTEMPTS: &str = "skippy.kv.peer_transfer_attempts";
    pub const KV_PEER_TRANSFER_ERRORS: &str = "skippy.kv.peer_transfer_errors";
}
