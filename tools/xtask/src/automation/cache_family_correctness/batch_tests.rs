use super::*;
fn request() -> Request {
    serde_json::from_value(json!({"schema_version":1,"prepared_input":"/prepared.json","prefix_tokens":null,"n_gpu_layers":null})).unwrap()
}
#[test]
fn batch_defaults_preserve_catalog_intersection_three_topologies_and_four_lanes() {
    let r = request();
    r.validate().unwrap();
    assert_eq!(r.runtime_lane_count, 4);
    assert_eq!(r.cache_hit_repeats, 3);
    assert_eq!(r.execution_seconds, 900);
    assert_eq!(
        r.topologies
            .iter()
            .map(|t| t.range(9).unwrap())
            .collect::<Vec<_>>(),
        vec![(0, 9, 0), (3, 6, 1), (6, 9, 2)]
    );
    let catalog: Value =
        serde_json::from_slice(include_bytes!("../cache_family_plan/catalog.json")).unwrap();
    let old = [
        "llama",
        "qwen3_dense",
        "gemma3",
        "gemma",
        "olmo",
        "glm4",
        "falcon_h1",
        "mistral3",
        "gpt2",
        "mpt",
        "olmo2",
        "olmoe",
        "phi3",
        "granite",
        "bloom",
        "gptneox",
        "baichuan",
        "exaone",
        "exaone4",
        "command_r",
        "cohere2",
        "falcon",
        "internlm2",
        "stablelm",
        "starcoder2",
        "qwen2moe",
        "qwen3moe",
        "jamba",
        "lfm2",
        "mamba",
        "mamba2",
        "qwen3next",
        "rwkv6",
        "rwkv7",
    ];
    let observed: Vec<_> = catalog
        .as_array()
        .unwrap()
        .iter()
        .filter_map(|c| c["key"].as_str())
        .filter(|k| old.contains(k))
        .collect();
    assert_eq!(observed, DEFAULT_CASES);
    let mut r = request();
    r.cases = vec!["invented".into()];
    assert!(r.validate().is_err());
    r.cases = vec!["llama".into(), "llama".into()];
    assert!(r.validate().is_err());
    r.cases.clear();
    r.topologies = vec![Topology::PackageStage1];
    assert!(r.validate().is_err());
}
#[test]
fn batch_table_preserves_report_fields_missing_and_failure_without_false_pass() {
    let case =
        json!({"family":"fixture|family","model_id":"fixture\nmodel","payload":"kv-recurrent"});
    let evidence = json!({"status":"pass","skippy":{"matches":true,"native_seq_remapped":true,"source_native_seq_id":3,"restore_native_seq_id":7,"prompt_token_count":32,"suffix_token_count":1,"suffix_prefill_matches":true,"state_bytes":900,"cache_storage_bytes":600,"payload_digest":{"recurrent_bytes":200,"kv_bytes":700},"cache_hit_repeats":3,"cache_hit_matches":true}});
    let pass = row(&case, json!("split-middle"), &evidence);
    assert_eq!(pass["recurrent_bytes"], 200);
    assert_eq!(pass["kv_bytes"], 700);
    assert_eq!(pass["suffix_tokens"], 1);
    assert_eq!(pass["promotion_decision"], "pass");
    let missing = row(&case, json!("all"), &json!({"status":"missing-model"}));
    assert!(missing["suffix_tokens"].is_null());
    assert_eq!(missing["cache_hit_repeats"], 0);
    let failure = row(
        &case,
        json!("one-stage"),
        &json!({"status":"failed-process"}),
    );
    assert_eq!(failure["promotion_decision"], "disabled-or-recompute");
    assert!(markdown(&[pass]).contains("fixture\\|family"));
    assert!(!markdown(&[missing]).contains("fixture\nmodel"));
}
#[test]
fn batch_terminal_refusal_retains_observations_and_disables_promotion() {
    for (cancel, expired, finish) in [
        (false, false, true),
        (true, false, true),
        (false, true, true),
        (false, false, false),
    ] {
        let mut rows = vec![
            json!({"status":"pass","matches":true,"state_bytes":3,"promotion_decision":"pass"}),
            json!({"status":"missing-model"}),
        ];
        assert_eq!(
            terminal(&mut rows, cancel, expired, finish),
            !cancel && !expired && finish
        );
        assert_eq!(rows[0]["state_bytes"], 3);
        assert_eq!(rows[0]["matches"], true);
        if cancel || expired || !finish {
            assert_eq!(rows[0]["status"], "failed-terminal");
            assert_eq!(rows[0]["promotion_decision"], "disabled-or-recompute");
        }
    }
    assert!(!terminal(
        &mut [json!({"status":"refused"})],
        false,
        false,
        true
    ));
}

#[test]
fn batch_markdown_retains_match_diagnostics_distinct_from_numeric_counts() {
    let rows = [
        json!({"family":"fixture","suffix_tokens":1,"cache_hit_repeats":3,"suffix_prefill_matches":false,"cache_hit_matches":true}),
        json!({"family":"missing","suffix_tokens":null,"cache_hit_repeats":0,"suffix_prefill_matches":null,"cache_hit_matches":false}),
    ];
    let text = markdown(&rows);
    let lines: Vec<_> = text
        .lines()
        .filter(|l| l.starts_with("| fixture") || l.starts_with("| missing"))
        .collect();
    let first: Vec<_> = lines[0].split('|').map(str::trim).collect();
    assert_eq!(first[10], "false");
    assert_eq!(first[13], "true");
    let second: Vec<_> = lines[1].split('|').map(str::trim).collect();
    assert_eq!(second[10], "null");
    assert_eq!(second[13], "false");
}
#[test]
fn batch_actual_final_files_downgrade_after_write_cancel_deadline_and_preserve_foreign_path() {
    for phase in 0..3 {
        for expired in [false, true] {
            let tmp = tempfile::tempdir().unwrap();
            let mut files = publication::Files::new(tmp.path());
            let mut rows = vec![
                json!({"status":"pass","matches":true,"state_bytes":7,"suffix_prefill_matches":true,"cache_hit_matches":true,"promotion_decision":"pass"}),
                json!({"status":"missing-model","promotion_decision":"disabled-or-recompute"}),
            ];
            let mut summary = json!({"completed":false});
            let refused = std::cell::Cell::new(false);
            let complete = publication::finish(
                &mut rows,
                &mut summary,
                &mut || (refused.get() && !expired, refused.get() && expired, true),
                &mut |index, bytes| {
                    files.write(index, bytes)?;
                    if index == phase {
                        refused.set(true);
                    }
                    Ok(())
                },
            )
            .unwrap();
            assert!(!complete);
            let observed: Value = serde_json::from_slice(
                &std::fs::read(tmp.path().join("cache-correctness-table.json")).unwrap(),
            )
            .unwrap();
            assert_eq!(observed[0]["status"], "failed-terminal");
            assert_eq!(observed[0]["state_bytes"], 7);
            assert_eq!(observed[1]["status"], "missing-model");
            assert_eq!(observed[0]["promotion_decision"], "disabled-or-recompute");
            let summary: Value = serde_json::from_slice(
                &std::fs::read(tmp.path().join("batch-summary.json")).unwrap(),
            )
            .unwrap();
            assert_eq!(summary["completed"], false);
            assert_eq!(summary["terminal_refusal"]["deadline_expired"], expired);
        }
    }
    let tmp = tempfile::tempdir().unwrap();
    std::fs::write(tmp.path().join("batch-summary.json"), b"foreign").unwrap();
    let mut files = publication::Files::new(tmp.path());
    assert!(files.write(2, b"replacement").is_err());
    assert_eq!(
        std::fs::read(tmp.path().join("batch-summary.json")).unwrap(),
        b"foreign"
    );
}

#[test]
fn batch_final_publication_success_and_prior_failure_keep_actual_rows() {
    for prior_failed in [false, true] {
        let tmp = tempfile::tempdir().unwrap();
        let mut files = publication::Files::new(tmp.path());
        let status = if prior_failed {
            "failed-process"
        } else {
            "pass"
        };
        let mut rows = vec![
            json!({"status":status,"state_bytes":9,"promotion_decision":if prior_failed{"disabled-or-recompute"}else{"pass"}}),
        ];
        let mut summary = json!({"completed":false});
        assert_eq!(
            publication::finish(
                &mut rows,
                &mut summary,
                &mut || (false, false, true),
                &mut |i, bytes| files.write(i, bytes)
            )
            .unwrap(),
            !prior_failed
        );
        let summary: Value =
            serde_json::from_slice(&std::fs::read(tmp.path().join("batch-summary.json")).unwrap())
                .unwrap();
        assert_eq!(summary["completed"], !prior_failed);
        assert_eq!(rows[0]["status"], status);
        assert_eq!(rows[0]["state_bytes"], 9);
    }
}

#[cfg(unix)]
#[test]
fn batch_registered_final_publication_actual_sigterm_and_sigint_are_downgraded() {
    use crate::process::{
        self, Completion, Limits, OutputFiles, ProcessSpec, Readiness, Value as Arg,
    };
    for signal in [libc::SIGTERM, libc::SIGINT] {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path().canonicalize().unwrap();
        let result=process::supervise(&ProcessSpec{executable:std::env::current_exe().unwrap(),arguments:["--ignored","--exact","automation::cache_family_correctness::batch::tests::batch_registered_signal_publication_worker","--nocapture"].map(|v|Arg::Public(v.into())).to_vec(),cwd:root.clone(),environment:[("GATE_SIGNAL_ROOT".into(),Arg::Public(root.clone().into_os_string())),("GATE_SIGNAL".into(),Arg::Public(signal.to_string().into()))].into_iter().collect()},&Limits{execution:Duration::from_secs(10),graceful_shutdown:Duration::from_secs(1),forced_shutdown:Duration::from_secs(1),retained_bytes_per_stream:65536,readiness:Readiness::None,completion:Completion::Exit},&Cancellation::default(),OutputFiles::default()).unwrap();
        assert!(result.success());
        assert_eq!(result.outcome, process::Outcome::Exited);
        assert!(
            result.failure.is_none()
                && result.cleanup.complete
                && !result.cleanup.forced
                && result.cleanup.failure.is_none()
                && !result.cleanup.graceful_signal_failed
        );
        assert!(
            [&result.stdout, &result.stderr]
                .iter()
                .all(|s| s.line_capture_complete && !s.truncated && s.oversized_lines == 0)
        );
        let summary: Value =
            serde_json::from_slice(&std::fs::read(root.join("batch-summary.json")).unwrap())
                .unwrap();
        assert_eq!(summary["completed"], false);
        assert_eq!(summary["terminal_refusal"]["cancelled"], true);
        let table: Value = serde_json::from_slice(
            &std::fs::read(root.join("cache-correctness-table.json")).unwrap(),
        )
        .unwrap();
        assert_eq!(table[0]["status"], "failed-terminal");
        assert_eq!(table[0]["state_bytes"], 12);
        assert_eq!(table[0]["promotion_decision"], "disabled-or-recompute");
    }
}
#[cfg(unix)]
#[test]
#[ignore = "owned subprocess actual signal during final receipt write"]
fn batch_registered_signal_publication_worker() {
    let root = PathBuf::from(std::env::var_os("GATE_SIGNAL_ROOT").unwrap());
    let signal: libc::c_int = std::env::var("GATE_SIGNAL").unwrap().parse().unwrap();
    assert!([libc::SIGTERM, libc::SIGINT].contains(&signal));
    let interrupt = crate::automation::command_interrupt::Interrupt::install().unwrap();
    let mut rows = vec![json!({"status":"pass","state_bytes":12,"promotion_decision":"pass"})];
    let mut summary = json!({"completed":false});
    let mut files = publication::Files::new(&root);
    let mut sent = false;
    let result = publication::finish_owned(
        &mut rows,
        &mut summary,
        interrupt,
        Instant::now() + Duration::from_secs(5),
        &mut |i, bytes| {
            files.write(i, bytes)?;
            if i == 2 && !sent {
                sent = true;
                assert_eq!(unsafe { libc::raise(signal) }, 0);
            }
            Ok(())
        },
    );
    assert!(sent);
    assert!(result.is_err());
}
