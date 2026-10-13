use super::{
    manual_options::{self, GRAMMAR},
    manual_replay,
    recorded_requests::{Selection, Trajectory},
    replay_profile::{self, Mode},
};
use serde_json::json;
fn trajectory(id: &str, framework: &str, turns: usize) -> Trajectory {
    let mut messages = vec![json!({"role":"user","content":"task"})];
    for index in 0..turns {
        messages.push(json!({"role":"assistant","content":format!("recorded-{index}")}));
        messages.push(json!({"role":"tool","tool_call_id":format!("call-{index}"),"content":format!("observation-{index}")}));
    }
    serde_json::from_value(json!({"session_id":id,"source_dataset":"capture","agent_framework":framework,"recorded_model":null,"messages":messages,"tools":[{"type":"function","function":{"name":"search","parameters":{"type":"object","required":["query"]}}}]})).unwrap()
}
fn selection() -> Selection<'static> {
    Selection {
        model: "target",
        maximum_output_tokens: 2048,
        turn_limit: None,
        qualification_probe: false,
    }
}
fn args(values: &[&str]) -> Vec<String> {
    values.iter().map(|v| (*v).into()).collect()
}
fn parse(values: &[&str], running: bool) -> crate::command::DynResult<manual_options::Options> {
    manual_options::parse(&GRAMMAR.parse(&args(values)).unwrap(), running)
}
#[test]
fn checkpoint_stages_are_balanced_per_framework_and_independent_of_cohort_order() {
    let mut cohort = (0..4)
        .flat_map(|i| {
            [
                trajectory(&format!("g:{i}"), "goose", 10),
                trajectory(&format!("p:{i}"), "pi", 10),
            ]
        })
        .collect::<Vec<_>>();
    let first = replay_profile::build(&cohort, &selection(), Mode::Checkpoint).unwrap();
    for framework in ["goose", "pi"] {
        let mut indices = first
            .iter()
            .flatten()
            .filter(|turn| turn.agent_framework == framework)
            .map(|turn| turn.assistant_turn)
            .collect::<Vec<_>>();
        indices.sort_unstable();
        assert_eq!(indices, [0, 3, 6, 9]);
    }
    let identities = first
        .into_iter()
        .flatten()
        .map(|turn| (turn.session_id, turn.request_id))
        .collect::<std::collections::BTreeMap<_, _>>();
    cohort.reverse();
    let reversed = replay_profile::build(&cohort, &selection(), Mode::Checkpoint)
        .unwrap()
        .into_iter()
        .flatten()
        .map(|turn| (turn.session_id, turn.request_id))
        .collect::<std::collections::BTreeMap<_, _>>();
    assert_eq!(identities, reversed);
}
#[test]
fn selected_checkpoints_keep_original_ids_prefixes_tools_and_output_budgets() {
    let cohort = vec![trajectory("buzz", "goose", 4)];
    let all = replay_profile::build(&cohort, &selection(), Mode::All).unwrap();
    let final_turn = replay_profile::build(&cohort, &selection(), Mode::Final).unwrap();
    assert_eq!(all[0].len(), 4);
    assert_eq!(final_turn[0].len(), 1);
    assert_eq!(final_turn[0][0].request_id, "buzz:3");
    assert_eq!(final_turn[0][0].body, all[0][3].body);
    assert_eq!(
        final_turn[0][0].body["messages"][2]["content"],
        "observation-0"
    );
    assert_eq!(
        final_turn[0][0].body["messages"][3]["content"],
        "recorded-1"
    );
    assert_eq!(final_turn[0][0].body["tools"], json!(cohort[0].tools));
}
#[test]
fn single_turn_checkpoints_stay_valid_and_duplicate_sessions_fail() {
    let cohort = vec![trajectory("one", "goose", 1)];
    assert_eq!(
        replay_profile::expected(&cohort, Mode::Checkpoint).unwrap(),
        ["one:0"]
    );
    let duplicate = vec![trajectory("one", "goose", 1), trajectory("one", "pi", 1)];
    assert!(replay_profile::expected(&duplicate, Mode::All).is_err());
}
#[test]
fn checkpoint_completeness_rejects_missing_duplicate_unexpected_failed_and_wrong_stage() {
    let cohort = vec![trajectory("one", "goose", 4)];
    let valid = vec![json!({"session_id":"one","request_id":"one:3"})];
    assert_eq!(
        replay_profile::completeness(&cohort, &valid, Mode::Final).unwrap()["passed"],
        true
    );
    for records in [
        vec![],
        vec![valid[0].clone(), valid[0].clone()],
        vec![json!({"session_id":"extra","request_id":"extra:3"})],
        vec![json!({"session_id":"one","request_id":"one:3","error":"failure"})],
        vec![json!({"session_id":"one","request_id":"one:0"})],
    ] {
        assert_eq!(
            replay_profile::completeness(&cohort, &records, Mode::Final).unwrap()["passed"],
            false
        );
    }
}
#[test]
fn documented_default_and_explicit_modes_keep_manual_defaults() {
    let default = parse(
        &[
            "--ref",
            "stable=v0.75.1",
            "--ref",
            "main=origin/main",
            "--model",
            "hf://owner/model",
            "--trajectories-per-framework",
            "4",
        ],
        false,
    )
    .unwrap();
    assert_eq!(default.mode, Mode::Checkpoint);
    assert_eq!(default.concurrency, [1, 2, 4]);
    assert_eq!(default.sessions, Some(12));
    assert_eq!(default.passes, 1);
    assert_eq!(default.warmup, 4);
    let captured = parse(
        &[
            "--ref",
            "candidate=HEAD",
            "--model",
            "hf://owner/model",
            "--trajectory-manifest",
            "captured.json",
            "--replay-mode",
            "final",
            "--passes",
            "2",
            "--require-framework",
            "buzz",
            "--require-framework",
            "goose",
            "--prompt-token-range",
            "18000:22000",
            "--min-cache-pct",
            "70",
            "--require-output-match",
            "--max-ttft-regression-pct",
            "5",
            "--output",
            "artifact",
        ],
        true,
    )
    .unwrap();
    assert_eq!(captured.mode, Mode::Final);
    assert_eq!(captured.prompt_range, Some([18000, 22000]));
    assert!(captured.output_match);
}
#[test]
fn zero_minimum_cache_percentage_is_rejected_before_plan_creation() {
    assert!(
        parse(
            &[
                "--ref",
                "main=HEAD",
                "--model",
                "model.gguf",
                "--trajectory-manifest",
                "capture.json",
                "--min-cache-pct",
                "0",
            ],
            false,
        )
        .is_err()
    );
}

#[test]
fn invalid_manual_inputs_fail_before_tools_or_output() {
    for extra in [
        vec!["--concurrency", "4", "--concurrency", "4"],
        vec!["--sessions-per-concurrency", "1"],
        vec!["--replay-mode", "final", "--require-recurrent-restores"],
        vec!["--replay-mode", "unknown"],
        vec!["--min-cache-pct", "NaN"],
        vec!["--max-ttft-regression-pct", "-1"],
        vec!["--framework", "goose", "--framework", "goose"],
        vec!["--prompt-token-range", "22:18"],
    ] {
        let mut values = vec![
            "--ref",
            "main=HEAD",
            "--model",
            "model.gguf",
            "--trajectories-per-framework",
            "4",
        ];
        values.extend(extra);
        assert!(parse(&values, false).is_err());
    }
    assert!(
        parse(
            &[
                "--ref",
                "main=HEAD",
                "--model",
                "model.gguf",
                "--output",
                "artifact",
                "--trajectory-manifest",
                "a.json",
                "--dataset-file",
                "a.parquet"
            ],
            true
        )
        .is_err()
    );
}
#[test]
fn read_only_plan_has_abba_order_untuned_launch_and_distinct_commit_requirement() {
    let options = parse(
        &[
            "--ref",
            "stable=tag",
            "--ref",
            "main=HEAD",
            "--model",
            "hf://owner/model",
            "--passes",
            "2",
            "--trajectories-per-framework",
            "4",
        ],
        false,
    )
    .unwrap();
    let commits = std::collections::BTreeMap::from([
        ("stable".into(), "a".repeat(40)),
        ("main".into(), "b".repeat(40)),
    ]);
    let plan = manual_replay::plan(&options, &commits).unwrap();
    assert_eq!(
        plan["order"]
            .as_array()
            .unwrap()
            .iter()
            .map(|row| row["label"].as_str().unwrap())
            .collect::<Vec<_>>(),
        ["stable", "main", "main", "stable"]
    );
    assert_eq!(
        plan["server_command"],
        json!([
            "<release-binary>",
            "serve",
            "--model",
            "hf://owner/model",
            "--log-format",
            "json"
        ])
    );
    assert_eq!(
        plan["build_commands"],
        json!([
            ["just", "release-host-build"],
            ["just", "release-runtime-build", "metal"]
        ])
    );
    let commits = std::collections::BTreeMap::from([
        ("stable".into(), "a".repeat(40)),
        ("main".into(), "a".repeat(40)),
    ]);
    assert!(manual_replay::plan(&options, &commits).is_err());
}
#[test]
fn public_run_handoff_keeps_captured_profile_uri_gates_and_build_jobs() {
    let parent = tempfile::tempdir().unwrap();
    let options = parse(
        &[
            "--ref",
            "main=HEAD",
            "--model",
            "hf://owner/model",
            "--trajectory-manifest",
            "captured.json",
            "--replay-mode",
            "final",
            "--min-cache-pct",
            "70",
            "--output",
            "artifact",
        ],
        true,
    )
    .unwrap();
    let input = manual_replay::input(
        parent.path(),
        &options,
        parent.path(),
        std::path::Path::new("/fixture/git"),
        std::path::Path::new("/fixture/just"),
        None,
    )
    .unwrap();
    assert_eq!(input["replay_mode"], "final");
    assert_eq!(input["context_qualification"], "captured");
    assert_eq!(input["model_reference"], "hf://owner/model");
    assert_eq!(input["min_cache_pct"], 70.0);
    assert_eq!(input["build_jobs"][0]["ref"], "HEAD");
    assert_eq!(input["build_jobs"][0]["backend"], "metal");
    assert!(!parent.path().join("artifact").exists());
}
#[test]
fn selected_profile_summary_keeps_metrics_and_checks_selected_ids() {
    let cohort = vec![trajectory("s", "goose", 4)];
    let record = json!({"session_id":"s","request_id":"s:3","assistant_turn":3,"prompt_tokens":19000,"cached_tokens":15000,"completion_tokens":2,"requested_output_tokens":2,"generation_seconds":0.1,"elapsed_seconds":0.2,"started":0.0,"completed":0.2,"ttft_seconds":0.1,"cache_pct":78.947,"content_sha256":"fixture","finish_reason":"stop","decode_inter_token_seconds":[0.05]});
    let summary = replay_profile::summarize(&cohort, &[record], 1, Mode::Final).unwrap();
    assert_eq!(summary["acceptance"]["passed"], true);
    assert_eq!(summary["sessions"][0]["complete"], true);
    assert_eq!(
        summary["sessions"][0]["expected_request_ids"],
        json!(["s:3"])
    );
    assert_eq!(summary["measured_selected_requests"], 1);
    assert_eq!(summary["replay_mode"], "final");
    assert_eq!(summary["recorded_assistant_turns"], 4);
    assert_eq!(summary["prompt_tokens"], 19000.0);
}
#[test]
fn actual_local_http_final_exchange_sends_full_recorded_checkpoint_prefix() {
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    runtime.block_on(async {
 let listener=tokio::net::TcpListener::bind(("127.0.0.1",0)).await.unwrap();let base=format!("http://{}/v1",listener.local_addr().unwrap());
 let cohort=vec![trajectory("s","goose",4)];let mut turns=replay_profile::build(&cohort,&selection(),Mode::Final).unwrap();let turn=turns.pop().unwrap().pop().unwrap();assert_eq!(turn.request_id,"s:3");
 let server=async {let(mut socket,_)=listener.accept().await.unwrap();let mut bytes=Vec::new();let mut chunk=[0;4096];loop {let n=socket.read(&mut chunk).await.unwrap();assert!(n>0);bytes.extend_from_slice(&chunk[..n]);if let Some(end)=bytes.windows(4).position(|v|v==b"\r\n\r\n"){let headers=String::from_utf8_lossy(&bytes[..end]);let length=headers.lines().find_map(|line|line.to_ascii_lowercase().strip_prefix("content-length:").map(|v|v.trim().parse::<usize>().unwrap())).unwrap();if bytes.len()>=end+4+length{break;}}}
 let end=bytes.windows(4).position(|v|v==b"\r\n\r\n").unwrap();let body:serde_json::Value=serde_json::from_slice(&bytes[end+4..]).unwrap();assert_eq!(body["messages"].as_array().unwrap().len(),7);assert_eq!(body["messages"][6]["content"],"observation-2");assert_eq!(body["seed"],42);assert_eq!(body["temperature"],0);assert_eq!(body["tools"][0]["function"]["name"],"search");
 let sse="data: {\"choices\":[{\"delta\":{\"content\":\"answer\"}}]}\n\ndata: {\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}],\"usage\":{\"prompt_tokens\":19000,\"completion_tokens\":1,\"prompt_tokens_details\":{\"cached_tokens\":0}}}\n\ndata: [DONE]\n\n";socket.write_all(format!("HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{sse}",sse.len()).as_bytes()).await.unwrap();};
 let measured=super::trajectory_execution::exchange(&base,&turn);
 let ((),evidence)=tokio::time::timeout(std::time::Duration::from_secs(2),async{tokio::join!(server,measured)}).await.unwrap();let evidence=evidence.unwrap();assert_eq!(evidence.prompt_tokens,19000);assert_eq!(evidence.cached_tokens,0);
 });
}
#[test]
fn typed_run_and_arm_workload_propagate_mode_and_hf_home_but_warmup_remains_all() {
    let temporary = tempfile::tempdir().unwrap();
    let options = parse(
        &[
            "--ref",
            "main=HEAD",
            "--model",
            "hf://owner/model",
            "--trajectory-manifest",
            "capture.json",
            "--output",
            "artifact",
            "--replay-mode",
            "final",
            "--hf-home",
            "/fixture/hf",
        ],
        true,
    )
    .unwrap();
    let request = manual_replay::input(
        temporary.path(),
        &options,
        temporary.path(),
        std::path::Path::new("/fixture/git"),
        std::path::Path::new("/fixture/just"),
        None,
    )
    .unwrap();
    let input: super::run_workload::Input = serde_json::from_value(request).unwrap();
    assert_eq!(input.replay_mode, Mode::Final);
    assert_eq!(input.hf_home, Some(std::path::PathBuf::from("/fixture/hf")));
    let serialized = serde_json::to_value(input).unwrap();
    assert_eq!(serialized["replay_mode"], "final");
    assert_eq!(serialized["hf_home"], "/fixture/hf");
    let manifest:super::manifest_preflight::Manifest=serde_json::from_value(json!({"cohorts":{"warmup":[trajectory("w","goose",4).original],"1":[trajectory("s","goose",4).original]}})).unwrap();
    let input:super::arm_pass::Input=serde_json::from_value(json!({"manifest":"/fixture/manifest.json","requirements":{"concurrency":[1],"minimum_worker_waves":1,"warmup_turns":4,"required_frameworks":[]},"model":"hf://owner/model","label":"main","ref":"HEAD","commit":"a".repeat(40),"pass":1,"max_output_tokens":2048,"request_timeout_seconds":1,"startup_timeout_seconds":2,"timeout_seconds":3,"port":9337,"output":"/fixture/artifact","replay_mode":"final","hf_home":"/fixture/hf"})).unwrap();
    let (workload, levels) = super::arm_pass_workload::prepare(&input, manifest).unwrap();
    assert_eq!(levels, [1]);
    assert_eq!(workload["replay_mode"], "all");
    assert_eq!(
        workload["following_cells"][0]["workload"]["replay_mode"],
        "final"
    );
    assert_eq!(input.hf_home, Some(std::path::PathBuf::from("/fixture/hf")));
}
#[test]
fn selected_resume_verifies_raw_identity_and_recomputed_summary_without_weakening_all() {
    let temporary = tempfile::tempdir().unwrap();
    let cohort = vec![trajectory("s", "goose", 4)];
    let record = json!({"session_id":"s","request_id":"s:3","assistant_turn":3,"prompt_tokens":19000,"cached_tokens":15000,"completion_tokens":2,"requested_output_tokens":2,"generation_seconds":0.1,"elapsed_seconds":0.2,"started":0.0,"completed":0.2,"ttft_seconds":0.1,"cache_pct":78.947,"content_sha256":"fixture","finish_reason":"stop","decode_inter_token_seconds":[0.05]});
    let summary =
        replay_profile::summarize(&cohort, std::slice::from_ref(&record), 1, Mode::Final).unwrap();
    let manifest = temporary.path().join("manifest.json");
    std::fs::write(
        &manifest,
        serde_json::to_vec(&json!({"cohorts":{"1":[cohort[0].original]}})).unwrap(),
    )
    .unwrap();
    let raw = temporary.path().join("c-1-requests.jsonl");
    std::fs::write(
        &raw,
        format!("{}\n", serde_json::to_string(&record).unwrap()),
    )
    .unwrap();
    assert!(
        super::resume_profile::verify(
            temporary.path(),
            &manifest,
            std::slice::from_ref(&summary),
            Mode::Final
        )
        .is_ok()
    );
    let mut wrong = record.clone();
    wrong["request_id"] = "s:0".into();
    std::fs::write(&raw, format!("{wrong}\n")).unwrap();
    assert!(
        super::resume_profile::verify(
            temporary.path(),
            &manifest,
            std::slice::from_ref(&summary),
            Mode::Final
        )
        .is_err()
    );
    std::fs::write(&raw, format!("{record}\n{record}\n")).unwrap();
    assert!(
        super::resume_profile::verify(
            temporary.path(),
            &manifest,
            std::slice::from_ref(&summary),
            Mode::Final
        )
        .is_err()
    );
    std::fs::write(&raw, format!("{record}\n")).unwrap();
    let mut wrong_summary = summary.clone();
    wrong_summary["prompt_tokens"] = 19001.into();
    assert!(
        super::resume_profile::verify(temporary.path(), &manifest, &[wrong_summary], Mode::Final)
            .is_err()
    );
    assert!(
        super::resume_profile::verify(
            temporary.path(),
            std::path::Path::new("/no-new-all-requirement"),
            &[],
            Mode::All
        )
        .is_ok()
    );
}
