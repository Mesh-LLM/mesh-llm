use super::super::{
    health_log::Observation,
    paired_execution::{self, Outcome},
    stream_metrics::Measurement,
    trial_cell::Trial,
};
use super::*;

fn inputs() -> (tempfile::TempDir, plan::Spec, [plan::Side; 2]) {
    let directory = tempfile::tempdir().unwrap();
    let spec = plan::Spec {
        seed: 42,
        pairs_primary: 1,
        pairs_scenario: 1,
        scenarios: vec!["fixture".into()],
    };
    let sides = plan::sides(
        directory.path().join("host"),
        None,
        &[plan::Mode::Production, plan::Mode::EventDisabled],
    )
    .unwrap();
    (directory, spec, sides)
}

fn measured() -> Measurement {
    Measurement {
        completion_tokens: Some(2),
        ttft_ms: Some(1.0),
        elapsed_ms: 10.0,
        decode_tok_s: Some(200.0),
        decode_only_tok_s: Some(2.0 / 0.009),
        malformed: false,
    }
}
fn successful() -> Outcome {
    Outcome {
        launched: true,
        measurement: Some(measured()),
        ..Outcome::default()
    }
}
fn environment(mode: plan::Mode) -> BTreeMap<String, trial_environment::Entry> {
    trial_environment::snapshot(&trial_environment::effective(BTreeMap::new(), mode))
}
fn manifest(
    spec: &plan::Spec,
    sides: &[plan::Side; 2],
    batch: &Batch,
    index: usize,
    environment: Option<&BTreeMap<String, trial_environment::Entry>>,
    inheritance: Option<&trial_profile::Inheritance>,
) -> DynResult<Value> {
    let host = Host::classify("Darwin".into(), "arm64".into());
    let context = Context {
        spec,
        model: "/fixture/model.gguf",
        source_model_sha256: &"a".repeat(64),
        attempt: 2,
        generated_at: "2026-10-05T00:00:00Z",
        host: &host,
        thermal_state: &json!({"available":false}),
    };
    let binary = Binary {
        path: sides[index].binary.clone(),
        sha256: "b".repeat(64),
        version: Some("fixture".into()),
    };
    build_normalized(
        &context,
        &sides[index],
        &binary,
        environment,
        inheritance,
        batch,
        index,
    )
}

#[test]
fn interrupted_before_any_launch_publishes_both_sides_without_invented_environment_or_health() {
    let (_directory, spec, sides) = inputs();
    let entries = plan::build(&spec, &sides).unwrap();
    let mut batch = paired_execution::run(&entries, &sides, |_, _| {
        Err("interrupted before launch".into())
    })
    .unwrap();
    batch.final_health = std::array::from_fn(|_| Observation {
        health: Some(serde_json::from_value(json!({"dropped_progress":99})).unwrap()),
        ingress_p99_us: Some(99.0),
    });
    for index in [0, 1] {
        let value = manifest(&spec, &sides, &batch, index, None, None).unwrap();
        assert_eq!(value["expected_dropped_progress"], 0);
        assert_eq!(value["expected_dropped_diagnostic"], 0);
        assert!(value["health"].is_null());
        assert!(value["callback_ingress_p99_us"].is_null());
        assert!(value["environment"].is_null());
        assert!(value["inherited_profile"].is_null());
        assert_eq!(value["executed_order"], json!([]));
        assert_eq!(value["execution_incomplete"], "interrupted before launch");
    }
}

#[test]
fn disabled_side_spawn_refusal_is_not_an_event_engine_trial_even_with_failed_record() {
    let (_directory, spec, sides) = inputs();
    let mut entries = plan::build(&spec, &sides).unwrap();
    entries[0].side_order_first = sides[1].side_id.clone();
    let batch = paired_execution::run(&entries, &sides, |_, _| {
        Ok(Outcome {
            error: Some("spawn refused".into()),
            ..Outcome::default()
        })
    })
    .unwrap();
    assert_eq!(batch.trials[1].len(), 1);
    assert!(!batch.trials[1][0].launched);
    for index in [0, 1] {
        let value = manifest(&spec, &sides, &batch, index, None, None).unwrap();
        assert_eq!(value["expected_dropped_progress"], 0);
        assert_eq!(value["expected_dropped_diagnostic"], 0);
        assert!(value["health"].is_null());
        assert!(value["callback_ingress_p99_us"].is_null());
    }
}

#[test]
fn launched_request_failure_still_has_single_trial_expectations_and_requires_real_profile() {
    let (_directory, spec, sides) = inputs();
    let entries = plan::build(&spec, &sides).unwrap();
    let batch = paired_execution::run(&entries, &sides, |_, _| {
        Ok(Outcome {
            launched: true,
            error: Some("request failed".into()),
            ..Outcome::default()
        })
    })
    .unwrap();
    assert!(manifest(&spec, &sides, &batch, 1, None, None).is_err());
    let env = environment(sides[1].mode);
    let inheritance = trial_profile::Inheritance::default();
    assert!(manifest(&spec, &sides, &batch, 1, Some(&env), None).is_err());
    let value = manifest(&spec, &sides, &batch, 1, Some(&env), Some(&inheritance)).unwrap();
    assert_eq!(value["expected_dropped_progress"], 1);
    assert_eq!(value["expected_dropped_diagnostic"], 0);
}

#[test]
fn supplied_full_health_dictionary_and_independent_null_p99_roundtrip_unchanged() {
    let (_directory, spec, sides) = inputs();
    let supplied = json!({"version":1,"reservation_exhausted":0,"cancelled_reservation_rejected":2,
        "terminal_delivery_failed":3,"dropped_progress":1,"coalesced_progress":4,"dropped_diagnostic":0,
        "replay_evicted":5,"subscriber_disconnected":6,"shutdown_degraded":7,"reducer_rejected":8,
        "state_transition_rejected":9,"state_degraded":true,"rebuild_required":true,"rebuild_generation":10});
    let batch = paired_execution::run(&plan::build(&spec, &sides).unwrap(), &sides, |_, _| {
        let mut outcome = successful();
        outcome.health = Observation {
            health: Some(serde_json::from_value(supplied.clone()).unwrap()),
            ingress_p99_us: None,
        };
        Ok(outcome)
    })
    .unwrap();
    for index in [0, 1] {
        let env = environment(sides[index].mode);
        let value = manifest(
            &spec,
            &sides,
            &batch,
            index,
            Some(&env),
            Some(&trial_profile::Inheritance::default()),
        )
        .unwrap();
        assert_eq!(value["health"], supplied);
        assert!(value["callback_ingress_p99_us"].is_null());
    }
}

#[test]
fn normalized_effective_profile_preserves_redaction_and_extra_device_provenance() {
    let (_directory, spec, sides) = inputs();
    let batch = paired_execution::run(&plan::build(&spec, &sides).unwrap(), &sides, |_, _| {
        Ok(successful())
    })
    .unwrap();
    let mut env = environment(sides[0].mode);
    env.insert(
        "MESH_LLM_CONFIG".into(),
        trial_environment::Entry {
            value: json!("<redacted:present>"),
            redacted: true,
        },
    );
    let inheritance = trial_profile::Inheritance {
        dropped_inherited_settings: vec!["MESH_LLM_AUTH_TOKEN".into()],
        device_and_backend_settings: [("CUDA_VISIBLE_DEVICES".into(), "2".into())]
            .into_iter()
            .collect(),
    };
    let value = manifest(&spec, &sides, &batch, 0, Some(&env), Some(&inheritance)).unwrap();
    assert_eq!(
        value["environment"]["MESH_LLM_CONFIG"]["value"],
        "<redacted:present>"
    );
    assert_eq!(
        value["inherited_profile"]["device_and_backend_settings"]["CUDA_VISIBLE_DEVICES"],
        "2"
    );
    assert_eq!(
        value["inherited_profile"]["dropped_inherited_settings"],
        json!(["MESH_LLM_AUTH_TOKEN"])
    );
}

#[cfg(unix)]
fn captured_trial(
    directory: &std::path::Path,
    message: Option<&str>,
    completeness: bool,
    ambiguous: bool,
) -> Trial {
    use super::super::trial_owner::Owner;
    use crate::process::retained::{ExpectedExit, Launch, MemberId, Report};
    use crate::process::{
        Cancellation, Completion, Limits, OutputFiles, ProcessSpec, Readiness, Value,
    };
    use std::time::Duration;
    let health = message.map(|message| {
        serde_json::to_string(&json!({"context":"event_system_health","message":message})).unwrap()
    });
    let stop = match health {
        Some(health) => {
            format!("health='{health}'; trap 'printf \"%s\\n\" \"$health\" >&2; exit 0' TERM")
        }
        None => "trap 'exit 0' TERM".into(),
    };
    let additional = if ambiguous {
        "printf '%s\\n' \"$health\";"
    } else {
        ""
    };
    let script = format!(
        "{stop}; {additional} printf '%s\\n' '{{\"event\":\"api_ready\",\"url\":\"http://127.0.0.1:12345/v1\"}}'; while :; do sleep 1; done"
    );
    let launch = |member, script: String, label: &str| Launch {
        member,
        spec: ProcessSpec {
            executable: "/bin/sh".into(),
            cwd: directory.into(),
            arguments: vec![Value::Public("-c".into()), Value::Public(script.into())],
            environment: Default::default(),
        },
        files: OutputFiles {
            stdout: Some(directory.join(format!("{label}.stdout.log"))),
            stderr: Some(directory.join(format!("{label}.stderr.log"))),
        },
        readiness_deadline: Duration::from_secs(5),
    };
    let mut owner = Owner {
        server: Some(launch(MemberId::Seed, script, "server")),
        worker: Some(launch(MemberId::WorkerOne, "exit 0".into(), "worker")),
        worker_policy: ExpectedExit::new(&[0, 1], Duration::from_secs(5)).unwrap(),
        stopping: false,
        setup_ms: None,
        stop_started: None,
        shutdown_ms: None,
        api_readiness: Some(("http://127.0.0.1:12345/v1".into(), 12345)),
        listener_ready: false,
        host_readiness_timeout: Duration::from_secs(2),
        host_started: None,
        health_streams: [None, None],
    };
    let limits = Limits {
        execution: Duration::from_secs(5),
        graceful_shutdown: Duration::from_millis(500),
        forced_shutdown: Duration::from_secs(1),
        retained_bytes_per_stream: 4096,
        readiness: Readiness::None,
        completion: Completion::Exit,
    };
    let report: Report<String> =
        crate::process::retained::run(&mut owner, &limits, &Cancellation::default()).unwrap();
    assert!(report.recovery_success());
    let server = report
        .members
        .iter()
        .find(|member| member.member == MemberId::Seed)
        .unwrap();
    assert!(!server.process.stdout.truncated && !server.process.stderr.truncated);
    assert_eq!(
        server.process.stdout.suppressed_lines + server.process.stderr.suppressed_lines,
        0
    );
    let mut trial = Trial {
        outcome: successful(),
        environment: environment(plan::Mode::Production),
        worker_status: Some(0),
        cleanup_complete: true,
        cleanup_forced: false,
        capture_complete: completeness,
        health_capture_complete: false,
        health_observation_error: if completeness {
            None
        } else {
            Some("capture incomplete".into())
        },
        inheritance: Default::default(),
    };
    trial.observe_final_health(directory);
    trial
}

#[cfg(unix)]
#[test]
fn actual_shutdown_capture_trial_batch_and_manifest_keep_each_sides_last_log() {
    let (directory, spec, sides) = inputs();
    let mut ordinal = 0;
    let batch = paired_execution::run(
        &plan::build(&spec, &sides).unwrap(),
        &sides,
        |side, entry| {
            ordinal += 1;
            let child = directory.path().join(format!("trial-{ordinal}"));
            std::fs::create_dir(&child)?;
            let final_trial = entry.scenario == "fixture";
            let disabled = side.mode == plan::Mode::EventDisabled;
            let message = if final_trial {
                if disabled {
                    "version=1 dropped_progress=1 dropped_diagnostic=0 ingress_p99_us=null"
                } else {
                    "version=1 dropped_progress=0 dropped_diagnostic=0 ingress_p99_us=12"
                }
            } else {
                "version=1 dropped_progress=99 dropped_diagnostic=99 ingress_p99_us=99"
            };
            let mut trial = captured_trial(&child, Some(message), true, false);
            trial.environment = environment(side.mode);
            Ok(trial.into_outcome())
        },
    )
    .unwrap();
    for index in [0, 1] {
        let env = environment(sides[index].mode);
        let value = manifest(
            &spec,
            &sides,
            &batch,
            index,
            Some(&env),
            Some(&trial_profile::Inheritance::default()),
        )
        .unwrap();
        assert_eq!(value["health"]["dropped_progress"], index);
        assert_eq!(
            value["callback_ingress_p99_us"],
            if index == 0 { json!(12.0) } else { Value::Null }
        );
        assert_eq!(value["trials"].as_array().unwrap().len(), 2);
    }
}

#[cfg(unix)]
#[test]
fn missing_final_trial_log_health_replaces_prior_and_preserves_throughput() {
    let (directory, spec, sides) = inputs();
    let mut ordinal = 0;
    let batch = paired_execution::run(&plan::build(&spec, &sides).unwrap(), &sides, |_, entry| {
        ordinal += 1;
        let child = directory.path().join(format!("trial-{ordinal}"));
        std::fs::create_dir(&child)?;
        let message = (entry.scenario == plan::PRIMARY)
            .then_some("version=1 dropped_progress=99 ingress_p99_us=99");
        Ok(captured_trial(&child, message, true, false).into_outcome())
    })
    .unwrap();
    for index in [0, 1] {
        let env = environment(sides[index].mode);
        let value = manifest(
            &spec,
            &sides,
            &batch,
            index,
            Some(&env),
            Some(&trial_profile::Inheritance::default()),
        )
        .unwrap();
        assert!(value["health"].is_null());
        assert!(value["callback_ingress_p99_us"].is_null());
        assert_eq!(value["trials"][1]["decode_tok_s"], 200.0);
        assert_eq!(value["trials"][1]["status"], "succeeded");
    }
}

#[test]
fn unavailable_health_diagnostic_survives_trial_to_manifest_without_erasing_metrics() {
    let (_directory, spec, sides) = inputs();
    let batch = paired_execution::run(&plan::build(&spec, &sides).unwrap(), &sides, |_, _| {
        let trial = Trial {
            outcome: successful(),
            environment: Default::default(),
            worker_status: Some(0),
            cleanup_complete: true,
            cleanup_forced: false,
            capture_complete: false,
            health_capture_complete: false,
            health_observation_error: Some("ambiguous final health across streams".into()),
            inheritance: Default::default(),
        };
        Ok(trial.into_outcome())
    })
    .unwrap();
    for index in [0, 1] {
        let env = environment(sides[index].mode);
        let value = manifest(
            &spec,
            &sides,
            &batch,
            index,
            Some(&env),
            Some(&trial_profile::Inheritance::default()),
        )
        .unwrap();
        assert!(value["health"].is_null());
        assert!(value["callback_ingress_p99_us"].is_null());
        assert_eq!(
            value["health_observation_error"],
            "ambiguous final health across streams"
        );
        assert_eq!(
            value["trials"][1]["health_observation_error"],
            value["health_observation_error"]
        );
        assert_eq!(value["trials"][1]["decode_tok_s"], 200.0);
        assert_eq!(value["trials"][1]["status"], "succeeded");
    }
}

#[cfg(unix)]
#[test]
fn actual_ambiguous_shutdown_capture_diagnostic_reaches_manifest_and_keeps_throughput() {
    let (directory, spec, sides) = inputs();
    let mut ordinal = 0;
    let batch = paired_execution::run(&plan::build(&spec, &sides).unwrap(), &sides, |_, _| {
        ordinal += 1;
        let child = directory.path().join(format!("trial-{ordinal}"));
        std::fs::create_dir(&child)?;
        Ok(captured_trial(
            &child,
            Some("version=1 dropped_progress=1 ingress_p99_us=4"),
            true,
            true,
        )
        .into_outcome())
    })
    .unwrap();
    for index in [0, 1] {
        let env = environment(sides[index].mode);
        let value = manifest(
            &spec,
            &sides,
            &batch,
            index,
            Some(&env),
            Some(&trial_profile::Inheritance::default()),
        )
        .unwrap();
        assert!(value["health"].is_null());
        assert!(value["callback_ingress_p99_us"].is_null());
        assert!(
            value["health_observation_error"]
                .as_str()
                .unwrap()
                .contains("ambiguous")
        );
        assert_eq!(value["trials"][1]["decode_tok_s"], 200.0);
        assert_eq!(value["trials"][1]["status"], "succeeded");
    }
}
