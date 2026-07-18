use super::support::*;
use super::*;

#[test]
fn prefill_chunk_schedule_parses_and_repeats_last_size() {
    let schedule = PrefillChunkSchedule::parse(Some("128, 256,512"))
        .unwrap()
        .unwrap();
    assert_eq!(schedule.label(), "128,256,512");
    assert_eq!(schedule.chunk_size_for(0), 128);
    assert_eq!(schedule.chunk_size_for(1), 256);
    assert_eq!(schedule.chunk_size_for(2), 512);
    assert_eq!(schedule.chunk_size_for(3), 512);
}

#[test]
fn prefill_chunk_schedule_rejects_bad_sizes() {
    assert!(PrefillChunkSchedule::parse(Some("128,0")).is_err());
    assert!(PrefillChunkSchedule::parse(Some("128,,256")).is_err());
    assert!(PrefillChunkSchedule::parse(Some("abc")).is_err());
}

#[test]
fn prefill_chunk_policy_keeps_legacy_schedule_behavior() {
    let policy = PrefillChunkPolicy::parse(PrefillChunkPolicyArgs {
        policy: "fixed",
        schedule: Some("128,256,384"),
        fixed_chunk_size: 256,
        adaptive_start: 128,
        adaptive_step: 128,
        adaptive_max: 384,
        schedule_arg: "--prefill-chunk-schedule",
        policy_arg: "--prefill-chunk-policy",
    })
    .unwrap();
    let mut planner = policy.planner();
    assert_eq!(planner.chunk_size_for(0, 512), 128);
    assert_eq!(planner.chunk_size_for(1, 512), 256);
    assert_eq!(planner.chunk_size_for(2, 512), 384);
    assert_eq!(planner.chunk_size_for(3, 512), 384);
}

#[test]
fn prefill_adaptive_ramp_grows_when_downstream_wait_is_hidden() {
    let policy = PrefillChunkPolicy::parse(PrefillChunkPolicyArgs {
        policy: "adaptive-ramp",
        schedule: None,
        fixed_chunk_size: 256,
        adaptive_start: 128,
        adaptive_step: 128,
        adaptive_max: 384,
        schedule_arg: "--prefill-chunk-schedule",
        policy_arg: "--prefill-chunk-policy",
    })
    .unwrap();
    let mut planner = policy.planner();
    assert_eq!(planner.chunk_size_for(0, 512), 128);
    planner.observe(PrefillChunkObservation {
        compute_ms: 100.0,
        forward_write_ms: 5.0,
        downstream_wait_ms: 20.0,
    });
    assert_eq!(planner.chunk_size_for(1, 512), 256);
    planner.observe(PrefillChunkObservation {
        compute_ms: 100.0,
        forward_write_ms: 5.0,
        downstream_wait_ms: 20.0,
    });
    assert_eq!(planner.chunk_size_for(2, 512), 384);
}

#[test]
fn prefill_adaptive_ramp_can_advance_without_observations() {
    let policy = PrefillChunkPolicy::parse(PrefillChunkPolicyArgs {
        policy: "adaptive-ramp",
        schedule: None,
        fixed_chunk_size: 256,
        adaptive_start: 128,
        adaptive_step: 128,
        adaptive_max: 384,
        schedule_arg: "--prefill-chunk-schedule",
        policy_arg: "--prefill-chunk-policy",
    })
    .unwrap();
    let mut planner = policy.planner();
    assert_eq!(planner.chunk_size_for(0, 512), 128);
    planner.advance_without_observation();
    assert_eq!(planner.chunk_size_for(1, 512), 256);
    planner.advance_without_observation();
    assert_eq!(planner.chunk_size_for(2, 512), 384);
    planner.advance_without_observation();
    assert_eq!(planner.chunk_size_for(3, 512), 384);
}

#[test]
fn prefill_adaptive_ramp_backs_off_when_wait_is_exposed() {
    let policy = PrefillChunkPolicy::parse(PrefillChunkPolicyArgs {
        policy: "adaptive-ramp",
        schedule: None,
        fixed_chunk_size: 256,
        adaptive_start: 128,
        adaptive_step: 128,
        adaptive_max: 384,
        schedule_arg: "--prefill-chunk-schedule",
        policy_arg: "--prefill-chunk-policy",
    })
    .unwrap();
    let mut planner = policy.planner();
    planner.observe(PrefillChunkObservation {
        compute_ms: 100.0,
        forward_write_ms: 5.0,
        downstream_wait_ms: 10.0,
    });
    assert_eq!(planner.chunk_size_for(1, 512), 256);
    planner.observe(PrefillChunkObservation {
        compute_ms: 100.0,
        forward_write_ms: 5.0,
        downstream_wait_ms: 150.0,
    });
    assert_eq!(planner.chunk_size_for(2, 512), 128);
}

#[test]
fn prefill_adaptive_ramp_backs_off_when_write_is_exposed() {
    let policy = PrefillChunkPolicy::parse(PrefillChunkPolicyArgs {
        policy: "adaptive-ramp",
        schedule: None,
        fixed_chunk_size: 256,
        adaptive_start: 128,
        adaptive_step: 128,
        adaptive_max: 384,
        schedule_arg: "--prefill-chunk-schedule",
        policy_arg: "--prefill-chunk-policy",
    })
    .unwrap();
    let mut planner = policy.planner();
    planner.advance_without_observation();
    assert_eq!(planner.chunk_size_for(1, 512), 256);
    planner.observe(PrefillChunkObservation {
        compute_ms: 100.0,
        forward_write_ms: 90.0,
        downstream_wait_ms: 0.0,
    });
    assert_eq!(planner.chunk_size_for(2, 512), 128);
}

#[test]
fn prefill_adaptive_ramp_keeps_short_prompts_fixed() {
    let policy = PrefillChunkPolicy::parse(PrefillChunkPolicyArgs {
        policy: "adaptive-ramp",
        schedule: None,
        fixed_chunk_size: 256,
        adaptive_start: 128,
        adaptive_step: 128,
        adaptive_max: 384,
        schedule_arg: "--prefill-chunk-schedule",
        policy_arg: "--prefill-chunk-policy",
    })
    .unwrap();
    let mut planner = policy.planner();

    assert_eq!(planner.chunk_size_for(0, 256), 256);
    assert_eq!(planner.chunk_size_for(0, 257), 128);
}

#[test]
fn prefill_transport_ewma_seeds_adaptive_ramp() {
    let config = prefix_cache_test_config();
    let pool = PersistentStageLanePool {
        config: config.clone(),
        timeout_secs: 5,
        telemetry: Telemetry::new(None, 1, config, crate::telemetry::TelemetryLevel::Off),
        lanes: Mutex::new(Vec::new()),
        prefill_transport: Mutex::new(PrefillTransportEstimate::default()),
        next_lane_id: AtomicU64::new(0),
        capacity: 1,
    };
    let mut stats = StageReplyStats::default();
    stats.observe_prefill_edge_transport(1, 1_000, 0, 1_048_576);
    pool.observe_prefill_transport(&stats, 10.0, 1);

    let policy = PrefillChunkPolicy::parse(PrefillChunkPolicyArgs {
        policy: "adaptive-ramp",
        schedule: None,
        fixed_chunk_size: 256,
        adaptive_start: 128,
        adaptive_step: 128,
        adaptive_max: 384,
        schedule_arg: "--prefill-chunk-schedule",
        policy_arg: "--prefill-chunk-policy",
    })
    .unwrap();
    let mut planner = policy.planner();
    planner.observe(pool.prefill_transport_seed().unwrap());

    assert_eq!(planner.chunk_size_for(0, 512), 256);
}

#[test]
fn persistent_lane_ready_handshake_times_out_for_silent_downstream() {
    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let address = listener.local_addr().unwrap();
    let server = std::thread::spawn(move || {
        let (_stream, _) = listener.accept().unwrap();
        std::thread::sleep(Duration::from_millis(200));
    });
    let mut client = std::net::TcpStream::connect(address).unwrap();

    let error = receive_persistent_lane_ready(&mut client, Duration::from_millis(25)).unwrap_err();

    assert!(
        error
            .to_string()
            .contains("persistent downstream lane did not become ready")
    );
    server.join().unwrap();
}

#[test]
fn steady_state_lane_reconnect_deadline_is_much_shorter_than_warmup() {
    assert!(LANE_STEADY_CONNECT_TIMEOUT < LANE_READY_READ_TIMEOUT);
    assert!(LANE_STEADY_CONNECT_TIMEOUT <= Duration::from_secs(5));
}

#[test]
fn steady_state_ready_handshake_times_out_fast_for_silent_downstream() {
    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let address = listener.local_addr().unwrap();
    let server = std::thread::spawn(move || {
        let (_stream, _) = listener.accept().unwrap();
        std::thread::sleep(Duration::from_millis(200));
    });
    let mut client = std::net::TcpStream::connect(address).unwrap();

    let start = std::time::Instant::now();
    let error = receive_persistent_lane_ready(&mut client, Duration::from_millis(25)).unwrap_err();
    let elapsed = start.elapsed();

    assert!(
        error
            .to_string()
            .contains("persistent downstream lane did not become ready")
    );
    assert!(
        elapsed < Duration::from_secs(1),
        "handshake read must fail within the supplied deadline, took {elapsed:?}"
    );
    server.join().unwrap();
}
