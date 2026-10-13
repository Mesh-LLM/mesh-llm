//! Finite cases retained from the retired manual replay contract.
use super::{pooled_rows, report_escape, stream_evidence::Stream};
use serde_json::{Value, json};
use std::time::Duration;

fn consume(stream: &mut Stream, event: Value) {
    let line = format!("data: {event}\n");
    stream
        .consume(line.as_bytes(), Duration::from_secs(1))
        .unwrap();
}
fn usage(stream: &mut Stream) {
    consume(
        stream,
        json!({"usage":{"prompt_tokens":10,"completion_tokens":8,"prompt_tokens_details":{"cached_tokens":0}}}),
    );
}
fn finish(mut stream: Stream, probe: bool) -> Result<super::stream_evidence::Evidence, String> {
    stream
        .consume(b"data: [DONE]\n", Duration::from_secs(2))
        .unwrap();
    stream.finish(Duration::from_secs(2), probe)
}

#[test]
fn hidden_generation_is_accepted_only_for_context_probes() {
    let mut probe = Stream::default();
    usage(&mut probe);
    assert_eq!(finish(probe, true).unwrap().prompt_tokens, 10);
    let mut measured = Stream::default();
    usage(&mut measured);
    assert!(
        finish(measured, false)
            .unwrap_err()
            .contains("without generated content")
    );
}

#[test]
fn valid_usage_survives_later_invalid_boolean_usage() {
    let mut stream = Stream::default();
    consume(
        &mut stream,
        json!({"choices":[{"delta":{"content":"done"}}]}),
    );
    usage(&mut stream);
    consume(
        &mut stream,
        json!({"usage":{"prompt_tokens":false,"completion_tokens":false,"prompt_tokens_details":{"cached_tokens":false}}}),
    );
    let evidence = finish(stream, false).unwrap();
    assert_eq!(
        (
            evidence.prompt_tokens,
            evidence.cached_tokens,
            evidence.completion_tokens
        ),
        (10, 0, 8)
    );
}

#[test]
fn terminal_finish_reason_survives_later_usage_only_events() {
    let mut stream = Stream::default();
    consume(
        &mut stream,
        json!({"choices":[{"delta":{"content":"done"},"finish_reason":"length"}]}),
    );
    usage(&mut stream);
    assert_eq!(
        finish(stream, false).unwrap().finish_reason.as_deref(),
        Some("length")
    );
}

#[test]
fn tool_output_identity_ignores_generated_ids_and_transport_chunking() {
    let digest = |pieces: &[&str], identifier: &str| {
        let mut stream = Stream::default();
        for (index, arguments) in pieces.iter().enumerate() {
            let mut delta = json!({"index":0,"id":identifier,"function":{"arguments":arguments}});
            if index == 0 {
                delta["type"] = json!("function");
                delta["function"]["name"] = json!("shell");
            }
            consume(
                &mut stream,
                json!({"choices":[{"delta":{"tool_calls":[delta]}}]}),
            );
        }
        usage(&mut stream);
        finish(stream, false).unwrap().content_sha256
    };
    assert_eq!(
        digest(&["{\"command\":", "\"ls\"}"], "random-a"),
        digest(&["{\"command\":\"ls\"}"], "random-b")
    );
    assert_ne!(
        digest(&["{\"command\":\"ls\"}"], "random-a"),
        digest(&["{\"command\":\"pwd\"}"], "random-a")
    );
}

fn cell() -> Value {
    json!({"concurrency":1,"trajectories":1,"requests":1,"successful_requests":1,
        "failed_request_ids":[],"successful_request_ids":["s:0"],"content_sha256_by_request":{"s:0":"same"},
        "prompt_tokens_min":40,"prompt_tokens_max":40,"completion_tokens":10,"prompt_tokens":40,"cached_tokens":30,
        "generation_seconds":1,"workload_window_seconds":1,"ttft_samples":[1,9],
        "decode_tokens_per_second":10,"agent_steps_per_second":1,"workload_output_tokens_per_second":10,
        "ttft_p50_seconds":5,"ttft_p95_seconds":9,"mean_in_flight":1})
}
fn arm(label: &str, cells: Vec<Value>) -> Value {
    json!({"label":label,"ref":label,"commit":label,"cells":cells})
}
fn pool(passes: Vec<Value>) -> Vec<Value> {
    let passes = serde_json::from_value::<Vec<pooled_rows::ArmPass>>(json!(passes)).unwrap();
    pooled_rows::pool(&passes).unwrap()
}

#[test]
fn legacy_rows_without_hashes_keep_comparability() {
    let mut legacy = cell();
    legacy
        .as_object_mut()
        .unwrap()
        .remove("successful_request_ids");
    legacy
        .as_object_mut()
        .unwrap()
        .remove("content_sha256_by_request");
    let rows = pool(vec![
        arm("base", vec![legacy.clone()]),
        arm("candidate", vec![legacy]),
    ]);
    assert!(
        rows.iter()
            .all(|row| row["delta_comparable"] == true && row["content_identity_known"] == false)
    );
}

#[test]
fn hashes_must_cover_every_successful_pass() {
    let mut partial = cell();
    partial["content_sha256_by_request"] = json!({});
    let rows = pool(vec![
        arm("base", vec![cell(), partial]),
        arm("candidate", vec![cell(), cell()]),
    ]);
    assert_eq!(rows[0]["content_identity_known"], false);
    assert_eq!(rows[1]["delta_comparable"], false);
    assert!(rows[1]["decode_tokens_per_second_delta_pct"].is_null());
}

#[test]
fn different_failed_requests_suppress_paired_deltas() {
    let failed = |identifier: &str| {
        let mut value = cell();
        value["requests"] = json!(2);
        value["failed_request_ids"] = json!([identifier]);
        value
    };
    let rows = pool(vec![
        arm("base", vec![failed("s:1")]),
        arm("candidate", vec![failed("s:2")]),
    ]);
    assert_eq!(rows[1]["delta_comparable"], false);
    assert!(rows[1]["decode_tokens_per_second_delta_pct"].is_null());
}

#[test]
fn baseline_is_first_input_arm_even_when_sorted_rows_have_other_order() {
    let scaled = |seconds: u64| {
        let mut value = cell();
        value["generation_seconds"] = json!(seconds);
        value["workload_window_seconds"] = json!(seconds);
        value
    };
    let rows = pool(vec![
        arm("z-base", vec![scaled(2)]),
        arm("a-second", vec![scaled(1)]),
        arm("m-third", vec![scaled(4)]),
    ]);
    assert_eq!(rows[0]["label"], "a-second");
    assert_eq!(rows[0]["decode_tokens_per_second_delta_pct"], 100.0);
    assert_eq!(rows[1]["decode_tokens_per_second_delta_pct"], -50.0);
    assert_eq!(rows[2]["decode_tokens_per_second_delta_pct"], 0.0);
}

#[test]
fn version_code_cells_escape_markup_and_table_delimiters() {
    assert_eq!(
        report_escape::code("`version`|<tag>\nnext"),
        "<code>&#96;version&#96;&#124;&lt;tag&gt; next</code>"
    );
}

fn external_input(path: &std::path::Path, digest: &str) -> super::run_workload::Input {
    serde_json::from_value(json!({"manifest":"/fixture/manifest.json","requirements":{"concurrency":[1],"minimum_worker_waves":1,"warmup_turns":1,"required_frameworks":[]},"builds":[],"engine_config":path,"engine_config_sha256":digest,"context_qualification":"captured","model":"opaque/model","passes":1,"max_output_tokens":1,"request_timeout_seconds":1,"startup_timeout_seconds":2,"timeout_seconds":3,"output":"/fixture/artifact"})).unwrap()
}

#[cfg(unix)]
#[test]
fn validated_external_snapshot_survives_path_mutation_but_new_admission_rejects_drift() {
    use sha2::{Digest, Sha256};
    use std::os::unix::fs::PermissionsExt;
    let temporary = tempfile::tempdir().unwrap();
    let executable = temporary.path().join("engine");
    std::fs::write(&executable, "#!/bin/sh\nprintf 'snapshot version\\n'\n").unwrap();
    std::fs::set_permissions(&executable, std::fs::Permissions::from_mode(0o700)).unwrap();
    let path = temporary.path().join("engines.json");
    let mut config = json!({"schema_version":1,"comparison":{"model":"opaque/model"},"arms":[{"label":"old","engine":"llama","executable":executable,"cwd":temporary.path(),"model":"opaque/model","context_size":32768,"max_concurrency":1}]});
    let bytes = serde_json::to_vec(&config).unwrap();
    let digest = hex::encode(Sha256::digest(&bytes));
    std::fs::write(&path, bytes).unwrap();
    let mut input = external_input(&path, &digest);
    super::run_arms::prepare_config(&mut input).unwrap();
    config["arms"][0]["label"] = json!("mutated");
    config["arms"][0]["executable"] = json!(temporary.path().join("must-not-launch"));
    std::fs::write(&path, serde_json::to_vec(&config).unwrap()).unwrap();
    let budget = super::run_budget::Budget::new(&input);
    let retained = super::run_arms::append(&mut input, &budget)
        .unwrap()
        .unwrap();
    assert_eq!(retained.sha256, digest);
    assert_eq!(retained.arms[0].label, "old");
    assert_eq!(input.builds[0].label(), "old");
    let mut new_admission = external_input(&path, &digest);
    let error = super::run_arms::prepare_config(&mut new_admission).unwrap_err();
    assert!(
        error
            .to_string()
            .contains("changed after manual plan validation")
    );
    assert!(new_admission.builds.is_empty());
    assert!(new_admission.validated_engine_config.is_none());
}

#[test]
fn manual_plan_total_session_counts_do_not_multiply_total_by_frameworks() {
    let arguments = [
        "--ref",
        "candidate=HEAD",
        "--model",
        "opaque/model",
        "--sessions-per-concurrency",
        "16",
        "--concurrency",
        "1",
        "--concurrency",
        "8",
    ]
    .map(str::to_owned);
    let parsed = super::manual_options::GRAMMAR.parse(&arguments).unwrap();
    let options = super::manual_options::parse(&parsed, false).unwrap();
    let commits = [("candidate".to_owned(), "a".repeat(40))]
        .into_iter()
        .collect();
    let plan = super::manual_replay::plan(&options, &commits).unwrap();
    assert_eq!(plan["selection"]["measured_unique_trajectory_count"], 32);
    assert_eq!(plan["selection"]["warmup_unique_trajectory_count"], 16);
    assert!(plan["workload"]["measured_requests_total"].is_null());
}

#[test]
fn generated_tool_schemas_are_stable_and_tool_calls_size_output_budget() {
    let trajectory: super::recorded_requests::Trajectory = serde_json::from_value(json!({"session_id":"s","source_dataset":"fixture","agent_framework":"goose","messages":[{"role":"user","content":"task"},{"role":"assistant","content":null,"tool_calls_json":serde_json::to_string(&json!([{"id":"call","type":"function","function":{"name":"zebra","arguments":"x".repeat(400)}}])).unwrap()},{"role":"tool","tool_call_id":"call","content":"observation"},{"role":"assistant","tool_calls_json":serde_json::to_string(&json!([{"function":{"name":"alpha","arguments":"{}"}}])).unwrap()}]})).unwrap();
    let selection = super::recorded_requests::Selection {
        model: "target",
        maximum_output_tokens: 2048,
        turn_limit: None,
        qualification_probe: false,
    };
    let turns = super::recorded_requests::build(&trajectory, &selection).unwrap();
    assert!(turns[0].body["max_tokens"].as_u64().unwrap() > 8);
    let tools = turns[0].body["tools"].as_array().unwrap();
    assert_eq!(tools[0]["function"]["name"], "alpha");
    assert_eq!(tools[1]["function"]["name"], "zebra");
    assert!(
        tools
            .iter()
            .all(|tool| tool["function"]["parameters"]["type"] == "object")
    );
    assert_eq!(turns[0].body["tools"], turns[1].body["tools"]);
}

#[cfg(unix)]
#[test]
fn bounded_version_probe_timeout_is_a_preflight_error() {
    use std::os::unix::fs::PermissionsExt;
    let temporary = tempfile::tempdir().unwrap();
    let executable = temporary.path().join("engine");
    std::fs::write(&executable, "#!/bin/sh\nwhile :; do :; done\n").unwrap();
    std::fs::set_permissions(&executable, std::fs::Permissions::from_mode(0o700)).unwrap();
    let config = temporary.path().join("engines.json");
    std::fs::write(&config, serde_json::to_vec(&json!({"schema_version":1,"comparison":{"model":"opaque"},"arms":[{"label":"timeout","engine":"llama","executable":executable,"cwd":temporary.path(),"model":"opaque","context_size":32768,"max_concurrency":1}]})).unwrap()).unwrap();
    let config = super::external_config::load(&config).unwrap();
    let arm = &config.arms[0];
    let started = std::time::Instant::now();
    let error = super::external_probe::verify_with_budget(arm, Duration::from_millis(30))
        .err()
        .expect("hung fixture must fail preflight");
    assert!(
        error.to_string().contains("version probe failed: Deadline"),
        "{error}"
    );
    assert!(started.elapsed() < Duration::from_secs(8));
}

#[test]
fn each_measured_cohort_requires_framework_coverage_even_when_union_is_complete() {
    let trajectory = |id: &str, framework: &str| json!({"session_id":id,"source_dataset":"fixture","agent_framework":framework,"recorded_model":null,"messages":[{"role":"user","content":"task"},{"role":"assistant","content":"answer"}]});
    let document = json!({"cohorts":{
        "warmup":[trajectory("w", "goose")],
        "1":[trajectory("a", "goose"), trajectory("b", "openhands")],
        "2":[trajectory("c", "goose"), trajectory("d", "openhands")]
    }});
    let requirements:super::manifest_preflight::Requirements=serde_json::from_value(json!({"concurrency":[1,2],"minimum_worker_waves":1,"warmup_turns":1,"required_frameworks":["goose","openhands"]})).unwrap();
    let manifest = serde_json::from_value(document.clone()).unwrap();
    assert!(super::manifest_preflight::validate(&manifest, &requirements).is_ok());
    for cohort in ["1", "2"] {
        let mut incomplete = document.clone();
        incomplete["cohorts"][cohort][1]["agent_framework"] = "goose".into();
        let manifest = serde_json::from_value(incomplete).unwrap();
        let error = super::manifest_preflight::validate(&manifest, &requirements).unwrap_err();
        assert_eq!(
            error.to_string(),
            format!("{cohort}: missing required framework")
        );
    }
}

#[test]
fn manifest_admission_covers_shape_warmup_waves_and_each_framework() {
    let trajectory = |id: &str| json!({"session_id":id,"source_dataset":"fixture","agent_framework":"goose","recorded_model":null,"messages":[{"role":"user","content":"task"},{"role":"assistant","content":"answer"}]});
    let document =
        json!({"cohorts":{"warmup":[trajectory("w")],"1":[trajectory("a"),trajectory("b")]}});
    let mut requirements:super::manifest_preflight::Requirements=serde_json::from_value(json!({"concurrency":[1],"minimum_worker_waves":2,"warmup_turns":1,"required_frameworks":["goose"]})).unwrap();
    let manifest: super::manifest_preflight::Manifest =
        serde_json::from_value(document.clone()).unwrap();
    assert!(super::manifest_preflight::validate(&manifest, &requirements).is_ok());
    assert!(serde_json::from_value::<super::manifest_preflight::Manifest>(json!({})).is_err());
    assert!(
        serde_json::from_value::<super::manifest_preflight::Manifest>(
            json!({"cohorts":{"warmup":[false],"1":[]}})
        )
        .is_err()
    );
    requirements.warmup_turns = 2;
    assert!(
        super::manifest_preflight::validate(&manifest, &requirements)
            .unwrap_err()
            .to_string()
            .contains("warmup")
    );
    requirements.warmup_turns = 1;
    requirements.required_frameworks.push("openhands".into());
    assert!(
        super::manifest_preflight::validate(&manifest, &requirements)
            .unwrap_err()
            .to_string()
            .contains("framework")
    );
    requirements.required_frameworks.pop();
    let mut insufficient = document;
    insufficient["cohorts"]["1"] = json!([trajectory("a")]);
    let manifest: super::manifest_preflight::Manifest =
        serde_json::from_value(insufficient).unwrap();
    assert!(super::manifest_preflight::validate(&manifest, &requirements).is_err());
    requirements.minimum_worker_waves = 1;
    assert!(super::manifest_preflight::validate(&manifest, &requirements).is_ok());
}

#[test]
fn pinned_relative_model_handoff_uses_caller_path_and_digest_changes_plan_identity() {
    let temporary = tempfile::tempdir().unwrap();
    let arguments = [
        "--ref",
        "candidate=HEAD",
        "--model",
        "relative-model.gguf",
        "--trajectory-manifest",
        "captured.json",
        "--output",
        "artifact",
        "--expected-model-sha256",
        &"a".repeat(64),
    ]
    .map(str::to_owned);
    let parsed = super::manual_options::GRAMMAR.parse(&arguments).unwrap();
    let options = super::manual_options::parse(&parsed, true).unwrap();
    let request = super::manual_replay::input(
        temporary.path(),
        &options,
        temporary.path(),
        std::path::Path::new("/fixture/git"),
        std::path::Path::new("/fixture/just"),
        None,
    )
    .unwrap();
    assert_eq!(
        request["model"],
        json!(std::env::current_dir().unwrap().join("relative-model.gguf"))
    );
    assert_ne!(
        request["model"],
        json!(temporary.path().join("relative-model.gguf"))
    );
    let input: super::run_workload::Input = serde_json::from_value(request).unwrap();
    let mut identity = serde_json::to_value(input).unwrap();
    let first = super::cohort_identity::digest(&identity).unwrap();
    identity["model_sha256"] = json!("b".repeat(64));
    assert_ne!(first, super::cohort_identity::digest(&identity).unwrap());
}
