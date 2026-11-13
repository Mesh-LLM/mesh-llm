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
        speculation_stand_down: None,
        draft_token_cap: Some(15),
        verify_window_max_tokens: 16,
        runahead_budget_tokens: 0,
        draft_windows: 0,
        draft_proposed_tokens: 0,
        draft_accepted_tokens: 0,
        mtp_verifications: 50,
        mtp_proposed_tokens: 300,
        mtp_accepted_tokens: 150,
        activation_codec: Some("f16"),
        activation_prefill_bytes: 1 << 20,
        activation_decode_bytes: 1 << 16,
        request_shape: RequestShape {
            chat: true,
            tools_offered: true,
            tool_results: true,
            system_prompt: true,
            user_turns: 3,
        },
    }
}

#[test]
fn request_properties_derive_rates_from_the_measurement() {
    let properties = request_properties(&stats(), &ModelIdentity::redacted(), 3);

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

    let properties = request_properties(&idle, &ModelIdentity::redacted(), 1);

    assert!(properties.get("decode_tps").is_none());
    assert!(properties.get("spec_acceptance").is_none());
    assert!(properties.get("spec_accepted_per_window").is_none());
    assert!(properties.get("downstream_lanes").is_none());
}

#[test]
fn request_properties_carry_the_workload_without_content() {
    let properties = request_properties(&stats(), &ModelIdentity::redacted(), 1);

    assert_eq!(
        properties.get("request_kind"),
        Some(&Value::Static("agentic"))
    );
    assert_eq!(properties.get("tools_offered"), Some(&Value::Bool(true)));
    assert_eq!(properties.get("tool_results"), Some(&Value::Bool(true)));
    assert_eq!(properties.get("system_prompt"), Some(&Value::Bool(true)));
    // Three user turns land in the 3-4 bucket, not as an exact count.
    assert_eq!(properties.get("user_turns"), Some(&Value::Static("3-4")));
    assert_eq!(
        properties.get("context_band"),
        Some(&Value::Static("2k-8k"))
    );
}

#[test]
fn request_kinds_split_agentic_chat_and_completion() {
    let chat = RequestShape {
        chat: true,
        ..RequestShape::default()
    };
    assert_eq!(request_kind(&chat), "chat");
    assert_eq!(request_kind(&RequestShape::default()), "completion");
    let tool_loop = RequestShape {
        tool_results: true,
        ..chat
    };
    assert_eq!(request_kind(&tool_loop), "agentic");
}

#[test]
fn context_bands_do_not_share_edges() {
    assert_eq!(context_band(0), "lt2k");
    assert_eq!(context_band(2_047), "lt2k");
    assert_eq!(context_band(2_048), "2k-8k");
    assert_eq!(context_band(32_768), "32k-128k");
    assert_eq!(context_band(131_072), "128k+");
}

#[test]
fn request_properties_carry_the_speculation_plan() {
    let mut stood_down = stats();
    stood_down.speculation_allowed = false;
    stood_down.speculation_stand_down =
        Some(skippy_serving::generation_stats::SpeculationStandDown::GateTrial);
    let properties = request_properties(&stood_down, &ModelIdentity::redacted(), 1);

    assert_eq!(
        properties.get("spec_stand_down"),
        Some(&Value::Static("gate_trial"))
    );
    assert_eq!(properties.get("spec_draft_cap"), Some(&Value::Int(15)));
    assert_eq!(properties.get("spec_window_max"), Some(&Value::Int(16)));
    assert_eq!(properties.get("spec_runahead_tokens"), Some(&Value::Int(0)));
    // Drafted and accepted counts are what make acceptance computable.
    assert_eq!(
        properties.get("spec_proposed_tokens"),
        Some(&Value::Int(300))
    );
    assert_eq!(
        properties.get("spec_accepted_tokens"),
        Some(&Value::Int(150))
    );

    let speculating = request_properties(&stats(), &ModelIdentity::redacted(), 1);
    assert!(speculating.get("spec_stand_down").is_none());
}

#[test]
fn model_names_are_redacted_unless_a_catalog_identity_was_registered() {
    let unknown = request_properties(&stats(), &model_identity::lookup("acme-merger-finetune"), 1);
    assert_eq!(unknown.get("model"), Some(&Value::Text(Label::redacted())));
    model_identity::register(
        "served-under-a-stem",
        ModelIdentity {
            model: Label::sanitize("Qwen3-Coder-30B-A3B-Q4_K_M").unwrap(),
            ..ModelIdentity::redacted()
        },
    );
    let named = request_properties(&stats(), &model_identity::lookup("served-under-a-stem"), 1);
    assert_eq!(
        named.get("model"),
        Some(&Value::Text(
            Label::sanitize("Qwen3-Coder-30B-A3B-Q4_K_M").unwrap()
        ))
    );
}

#[test]
fn route_outcomes_separate_disconnects_and_broken_streams() {
    assert_eq!(
        RouteOutcome::from(AttemptOutcome::Unavailable).as_str(),
        "unavailable"
    );
    assert_eq!(RouteOutcome::StreamFailed.as_str(), "stream_failed");
    assert_eq!(
        RouteOutcome::ClientDisconnected.as_str(),
        "client_disconnected"
    );
}

#[test]
fn a_peer_gone_by_the_end_of_the_attempt_is_departed_not_unknown() {
    assert_eq!(route_link(None), ("departed", None));
    // Still in the mesh with no selected path: genuinely unknown.
    assert_eq!(route_link(Some((None, Some(9)))), ("unknown", None));
    assert_eq!(
        route_link(Some((Some(path("direct", "192.168.1.20:4000")), Some(2)))),
        ("lan", Some(2))
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
fn a_relay_link_reports_its_own_round_trip_not_a_stale_direct_one() {
    let mut relay = path("relay", "203.0.113.9:4000");
    relay.rtt_ms = Some(140);
    assert_eq!(
        link_and_rtt(Some(relay), Some(9)),
        (LinkClass::Relay, Some(140))
    );
    relay.rtt_ms = None;
    assert_eq!(link_and_rtt(Some(relay), Some(9)), (LinkClass::Relay, None));
    // No selected path: a gossip RTT belongs to no link.
    assert_eq!(link_and_rtt(None, Some(9)), (LinkClass::Unknown, None));
    // A direct link prefers the best direct measurement.
    assert_eq!(
        link_and_rtt(Some(path("direct", "192.168.1.20:4000")), Some(2)),
        (LinkClass::Lan, Some(2))
    );
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
