//! Which part of the system a span belongs to, exported as the span's
//! `service.name` (within the `skippy` service namespace). Trace viewers
//! colour a multi-service trace by service, so llama, Metal and each layer of
//! serving get their own colour while staying one trace.
//!
//! Classification is by span name. Serving names its spans `stage.*`, plus
//! `skippy.process` for the root and `native.*` for native ABI calls; the
//! native runtime names its own phases `skippy.*`, `llama.*` and `mtl.*`.

/// The `service.namespace` every component shares.
pub(super) const SERVICE_NAMESPACE: &str = "skippy";

/// The component, and so the exported `service.name`, of a span named `name`.
pub(super) fn component(name: &str) -> &'static str {
    let starts = |prefix: &str| name.starts_with(prefix);
    if name == "skippy.process" {
        return "skippy";
    }
    if starts("native.") {
        return "skippy-native";
    }
    if starts("llama.") {
        return "llama";
    }
    if starts("mtl.library_") || starts("mtl.pipeline_") {
        return "metal-compile";
    }
    if starts("mtl.") {
        return "metal-gpu";
    }
    if starts("skippy.") {
        return "skippy-runtime";
    }
    serving_component(name)
}

/// Serving's own `stage.*` spans, by the layer that emits them.
fn serving_component(name: &str) -> &'static str {
    let starts = |prefix: &str| name.starts_with(prefix);
    if is_kv(name) {
        "skippy-kv"
    } else if starts("stage.scheduler") || starts("stage.runtime_") {
        "skippy-scheduler"
    } else if starts("stage.embedded_runtime")
        || starts("stage.openai_server_")
        || starts("stage.openai_runtime_")
    {
        "skippy"
    } else if [
        "stage.openai_prefill",
        "stage.openai_media_prefill",
        "stage.openai_decode",
        "stage.openai_generation",
        "stage.openai_scheduler_generation",
        "stage.openai_linear_proposal",
    ]
    .iter()
    .any(|prefix| starts(prefix))
    {
        "skippy-decode"
    } else if starts("stage.openai_") {
        "skippy-http"
    } else if starts("stage.binary_") {
        "skippy-transport"
    } else {
        "skippy"
    }
}

/// KV cache and session-slot work: lookups, records, payloads, alignment.
fn is_kv(name: &str) -> bool {
    name.starts_with("stage.openai_session_")
        || name.split(['.', '_']).any(|segment| segment == "kv")
}

#[cfg(test)]
mod tests {
    use super::component;

    #[test]
    fn native_runtime_spans_are_told_apart_from_serving() {
        assert_eq!(component("skippy.process"), "skippy");
        assert_eq!(
            component("native.skippy_iteration_batch_sampled"),
            "skippy-native"
        );
        assert_eq!(component("skippy.sample"), "skippy-runtime");
        assert_eq!(component("llama.graph_compute"), "llama");
        assert_eq!(component("mtl.pipeline_compile"), "metal-compile");
        assert_eq!(component("mtl.library_compile"), "metal-compile");
        assert_eq!(component("mtl.command_buffer"), "metal-gpu");
        assert_eq!(component("mtl.execute"), "metal-gpu");
    }

    #[test]
    fn serving_spans_follow_their_layer() {
        let cases = [
            ("stage.embedded_runtime_load", "skippy"),
            ("stage.openai_server_setup", "skippy"),
            ("stage.openai_runtime_prewarm", "skippy"),
            ("stage.scheduler_iteration", "skippy-scheduler"),
            ("stage.scheduler", "skippy-scheduler"),
            ("stage.scheduler_coalesce", "skippy-scheduler"),
            ("stage.runtime_lock_wait", "skippy-scheduler"),
            ("stage.openai_http_request", "skippy-http"),
            ("stage.openai_chat_template", "skippy-http"),
            ("stage.openai_tokenize", "skippy-http"),
            ("stage.openai_generation_summary", "skippy-decode"),
            ("stage.openai_prefill", "skippy-decode"),
            ("stage.openai_decode_token", "skippy-decode"),
            ("stage.openai_kv_lookup_decision", "skippy-kv"),
            ("stage.binary_kv_record_decision", "skippy-kv"),
            ("stage.kv_payload_selected", "skippy-kv"),
            ("stage.openai_session_stop", "skippy-kv"),
            ("stage.binary_downstream_write", "skippy-transport"),
            ("stage.something_new", "skippy"),
        ];
        for (name, expected) in cases {
            assert_eq!(component(name), expected, "{name}");
        }
    }
}
