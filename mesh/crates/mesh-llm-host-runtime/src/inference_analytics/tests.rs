use super::*;
use mesh_llm_analytics::Value;

fn stats() -> GenerationStats {
    GenerationStats {
        model_id: "Qwen3-8B-Q4_K_M".to_owned(),
        split: true,
        downstream_lanes: Some(4),
        concurrency_limit: 2,
        context_size: 32_768,
        prompt_tokens: 3_000,
        completion_tokens: 200,
        cached_prompt_tokens: 2_000,
        prefill_tokens: 1_000,
        cache_hit_kind: "prefix",
        queue_wait_ms: 5.0,
        restore_ms: 10.0,
        prefill_ms: 500.0,
        decode_ms: 4_000.0,
        total_ms: 4_600.0,
        finish_reason: "stop",
        speculative_strategy: "dflash".to_owned(),
        speculation_allowed: true,
        draft_windows: 0,
        draft_proposed_tokens: 0,
        draft_accepted_tokens: 0,
        mtp_verifications: 50,
        mtp_proposed_tokens: 300,
        mtp_accepted_tokens: 150,
        activation_codec: Some("f16"),
        activation_prefill_bytes: 1 << 20,
        activation_decode_bytes: 1 << 16,
    }
}

#[test]
fn request_properties_derive_rates_from_the_measurement() {
    let properties = request_properties(&stats(), Label::redacted(), 3);

    assert_eq!(properties.get("weight"), Some(&Value::Int(3)));
    assert_eq!(properties.get("serving"), Some(&Value::Static("split")));
    assert_eq!(properties.get("context_tokens"), Some(&Value::Int(3_200)));
    assert_eq!(properties.get("prefill_tps"), Some(&Value::Float(2_000.0)));
    assert_eq!(properties.get("decode_tps"), Some(&Value::Float(50.0)));
    assert_eq!(properties.get("ttft_ms"), Some(&Value::Float(515.0)));
    assert_eq!(properties.get("spec_acceptance"), Some(&Value::Float(0.5)));
    assert_eq!(
        properties.get("spec_accepted_per_window"),
        Some(&Value::Float(3.0))
    );
    assert_eq!(properties.get("downstream_lanes"), Some(&Value::Int(4)));
}

#[test]
fn request_properties_omit_rates_with_no_denominator() {
    let mut idle = stats();
    idle.completion_tokens = 0;
    idle.decode_ms = 0.0;
    idle.mtp_proposed_tokens = 0;
    idle.mtp_accepted_tokens = 0;
    idle.mtp_verifications = 0;
    idle.downstream_lanes = None;

    let properties = request_properties(&idle, Label::redacted(), 1);

    assert!(properties.get("decode_tps").is_none());
    assert!(properties.get("spec_acceptance").is_none());
    assert!(properties.get("spec_accepted_per_window").is_none());
    assert!(properties.get("downstream_lanes").is_none());
}

#[test]
fn model_names_are_redacted_unless_loaded_from_a_reference() {
    assert_eq!(model_label("acme-merger-finetune").as_str(), "redacted");
    allow_model_name("Qwen3-Coder-30B-A3B-Q4_K_M");
    assert_eq!(
        model_label("Qwen3-Coder-30B-A3B-Q4_K_M").as_str(),
        "Qwen3-Coder-30B-A3B-Q4_K_M"
    );
}

#[test]
fn limiter_weights_the_next_event_by_what_it_dropped() {
    let limiter = RateLimiter::new(2);
    let start = Instant::now();

    assert_eq!(limiter.admit(start), Some(1));
    assert_eq!(limiter.admit(start), Some(1));
    assert_eq!(limiter.admit(start), None);
    assert_eq!(limiter.admit(start), None);
    // The next window's first event stands for itself and the two dropped.
    assert_eq!(limiter.admit(start + RateLimiter::WINDOW), Some(3));
}

fn path(path_type: &'static str, addr: &str) -> SelectedPathObservation {
    SelectedPathObservation {
        path_type,
        rtt_ms: Some(4),
        observed_direct_remote_addr: Some(addr.parse().unwrap()),
    }
}

#[test]
fn links_classify_by_path_and_address() {
    assert_eq!(
        LinkClass::of(Some(&path("direct", "192.168.1.20:4000"))),
        LinkClass::Lan
    );
    assert_eq!(
        LinkClass::of(Some(&path("direct", "[fd00::1]:4000"))),
        LinkClass::Lan
    );
    assert_eq!(
        LinkClass::of(Some(&path("direct", "203.0.113.9:4000"))),
        LinkClass::Wan
    );
    assert_eq!(
        LinkClass::of(Some(&path("relay", "203.0.113.9:4000"))),
        LinkClass::Relay
    );
    assert_eq!(LinkClass::of(None), LinkClass::Unknown);
}

#[test]
fn mesh_links_summarize_each_class() {
    let properties = mesh_links_properties(&[
        (LinkClass::Lan, Some(1)),
        (LinkClass::Lan, Some(3)),
        (LinkClass::Lan, Some(2)),
        (LinkClass::Wan, Some(80)),
        (LinkClass::Relay, None),
    ]);

    assert_eq!(properties.get("peers"), Some(&Value::Int(5)));
    assert_eq!(properties.get("lan_peers"), Some(&Value::Int(3)));
    assert_eq!(properties.get("lan_rtt_ms_median"), Some(&Value::Int(2)));
    assert_eq!(properties.get("lan_rtt_ms_max"), Some(&Value::Int(3)));
    assert_eq!(properties.get("wan_rtt_ms_median"), Some(&Value::Int(80)));
    assert_eq!(properties.get("relay_peers"), Some(&Value::Int(1)));
    assert!(properties.get("relay_rtt_ms_median").is_none());
    assert_eq!(properties.get("unknown_peers"), Some(&Value::Int(0)));
}
