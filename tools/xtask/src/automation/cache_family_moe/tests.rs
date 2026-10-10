use super::evidence;
use serde_json::json;
#[test]
fn moe_tensor_selection_uses_actual_layer_range_and_all_original_name_markers() {
    let names = [
        "_exps",
        "_shexp",
        ".expert",
        "_expert",
        "ffn_gate_inp",
        "expert_gate",
        "exp_probs",
    ];
    let mut tensors: Vec<_> = names
        .into_iter()
        .enumerate()
        .map(|(i, n)| json!({"name":format!("blk.10.{n}.{i}"),"layer_index":10,"byte_size":2}))
        .collect();
    tensors.push(json!({"name":"blk.3.up_exps.weight","layer_index":3,"byte_size":100}));
    tensors.push(json!({"name":"blk.10.dense.weight","layer_index":10,"byte_size":100}));
    let result = evidence::experts(
        &json!({"tensor_count":tensors.len(),"tensors":tensors}),
        9,
        18,
    )
    .unwrap();
    assert_eq!(result["expert_tensor_count"], 7);
    assert_eq!(result["expert_tensor_bytes"], 14);
    assert_eq!(result["expert_layers"], json!([10]));
}
#[test]
fn moe_tensor_projection_refuses_duplicate_corrupt_count_and_byte_overflow() {
    let t = json!({"name":"blk.1.ffn_up_exps","layer_index":1,"byte_size":u64::MAX});
    assert!(evidence::experts(&json!({"tensor_count":2,"tensors":[t.clone(),t]}), 0, 2).is_err());
    assert!(evidence::experts(&json!({"tensor_count":1,"tensors":[]}), 0, 2).is_err());
    assert!(evidence::experts(&json!({"tensor_count":2,"tensors":[{"name":"blk.1.ffn_up_exps","layer_index":1,"byte_size":u64::MAX},{"name":"blk.1.ffn_down_exps","layer_index":1,"byte_size":1}]}),0,2).is_err());
}
#[test]
fn moe_report_preserves_absent_sequence_observation_and_resident_zero_payload() {
    let report = json!({"status":"pass","model_identity":{"model_id":"provided"},"state_payload_kind":"resident-kv","layer_start":0,"layer_end":6,"suffix_prefill_matches":true,"cache_hit_matches":true,"state_bytes":0,"resident_state_bytes":1024});
    let expert = json!({"expert_tensor_count":2});
    let row = evidence::row(
        &json!({"family":"provided"}),
        &json!("one-stage"),
        &report,
        &expert,
    );
    assert_eq!(row["status"], "pass");
    assert!(row["native_seq_remapped"].is_null());
    assert_eq!(row["serialized_payload_bytes"], 0);
    assert_eq!(row["resident_state_bytes"], 1024);
    let mut bad = report;
    bad["suffix_prefill_matches"] = json!(false);
    assert_eq!(
        evidence::row(&json!({}), &json!("one-stage"), &bad, &expert)["status"],
        "fail"
    );
}

#[test]
fn moe_terminal_refusal_preserves_observed_rows_and_prior_reason() {
    let observed = json!({"status":"completed","cases":[{"observation":{"rows":[{"status":"pass","expert":{"expert_tensor_count":2}}]}}],"reason":"prior observation"});
    for (cancelled, expired, finish_ok) in [
        (false, false, true),
        (true, false, true),
        (false, true, true),
        (false, false, false),
    ] {
        let mut receipt = observed.clone();
        super::finalize(&mut receipt, cancelled, expired, finish_ok);
        assert_eq!(receipt["cases"], observed["cases"]);
        assert_eq!(receipt["reason"], observed["reason"]);
        assert_eq!(
            receipt["status"],
            if cancelled || expired || !finish_ok {
                "incomplete"
            } else {
                "completed"
            }
        );
        if cancelled || expired || !finish_ok {
            assert_eq!(receipt["terminal_refusal"]["cancelled"], cancelled);
            assert_eq!(receipt["terminal_refusal"]["deadline_expired"], expired);
            assert_eq!(
                receipt["terminal_refusal"]["interrupt_finish_failed"],
                !finish_ok
            );
        }
    }
    let mut prior = json!({"status":"failed","reason":"inspector refused","cases":[]});
    super::finalize(&mut prior, false, false, true);
    assert_eq!(prior["status"], "failed");
}

#[test]
fn moe_admission_preserves_supplied_artifact_and_pinned_toolkit_profile() {
    let directory = tempfile::tempdir().unwrap();
    let root = directory.path().canonicalize().unwrap();
    let profile = json!({"case_key":"minimax_m27","artifact":{"kind":"complete-shards"},"toolkit_directories":{"CUDA_PATH":{"path":root.join("toolkit"),"sha256":"a".repeat(64)}},"settings":{"CUDA_VISIBLE_DEVICES":"2"}});
    let input: super::contract::Input = serde_json::from_value(json!({"schema_version":1,"inspector":root.join("inspector"),"inspector_sha256":"b".repeat(64),"inspector_source_commit":"c".repeat(40),"execution_seconds":30,"cases":[{"layer_end":62,"correctness":profile}]})).unwrap();
    input.validate().unwrap();
    let admission = input.admission(&input.cases[0]).unwrap();
    assert_eq!(admission["artifact"], profile["artifact"]);
    assert_eq!(
        admission["toolkit_directories"],
        profile["toolkit_directories"]
    );
    assert_eq!(admission["environment"], profile["settings"]);
}

#[test]
fn moe_table_keeps_original_model_range_hit_expert_and_storage_observations() {
    let row = json!({"family":"provided","model_id":"model|ref","topology":"split-middle","layer_start":9,"layer_end":18,"status":"pass","native_seq_remapped":null,"suffix_prefill_matches":true,"cache_hit_matches":true,"expert":{"expert_layers":[10],"expert_tensor_count":2,"expert_tensor_bytes":100},"resident_state_bytes":512,"cache_storage_bytes":768,"serialized_payload_bytes":0});
    let receipt = json!({"status":"completed","cases":[{"observation":{"rows":[row]}}]});
    let markdown = super::render::markdown(&receipt);
    assert!(markdown.contains("Model ref | Topology | Layers"));
    assert!(markdown.contains("Cache storage bytes | Serialized bytes"));
    assert!(markdown.contains("model\\|ref | split-middle | 9..18 | pass | n/a | true | true | [10] | 2 | 100 | 512 | 768 | 0"));
}

fn publication_receipt() -> serde_json::Value {
    json!({"status":"completed","reason":"prior observation","cases":[{"observation":{"rows":[{"status":"pass","expert":{"expert_tensor_count":2}}]}}]})
}
fn assert_publication(root: &std::path::Path, receipt: &serde_json::Value, complete: bool) {
    let aggregate: serde_json::Value =
        serde_json::from_slice(&std::fs::read(root.join("moe-expert-smoke.json")).unwrap())
            .unwrap();
    let table: serde_json::Value =
        serde_json::from_slice(&std::fs::read(root.join("moe-expert-smoke-table.json")).unwrap())
            .unwrap();
    assert_eq!(aggregate["cases"], receipt["cases"]);
    assert_eq!(aggregate["reason"], "prior observation");
    assert_eq!(
        aggregate["status"],
        if complete { "completed" } else { "incomplete" }
    );
    assert_eq!(table["status"], aggregate["status"]);
    assert_eq!(
        table["rows"][0],
        receipt["cases"][0]["observation"]["rows"][0]
    );
    for name in ["moe-expert-smoke.md", "moe-expert-smoke-table.md"] {
        let markdown = std::fs::read_to_string(root.join(name)).unwrap();
        assert!(markdown.contains(if complete {
            "Status: completed."
        } else {
            "Status: incomplete."
        }));
    }
}
#[test]
fn moe_publication_late_cancel_deadline_finish_and_noclobber_preserve_observations() {
    use std::{
        cell::Cell,
        time::{Duration, Instant},
    };
    for mode in ["success", "cancel", "deadline", "finish", "existing"] {
        let directory = tempfile::tempdir().unwrap();
        let root = directory.path();
        let cancel = crate::process::Cancellation::default();
        let deadline = Cell::new(Instant::now() + Duration::from_secs(30));
        let mut receipt = publication_receipt();
        if mode == "existing" {
            std::fs::write(root.join("moe-expert-smoke-table.md"), b"foreign evidence").unwrap();
        }
        let mut reports = super::publication::publish(
            root,
            &mut receipt,
            &mut || (cancel.is_cancelled(), Instant::now() >= deadline.get()),
            &mut |index| {
                if index == 3 {
                    match mode {
                        "cancel" => cancel.cancel(),
                        "deadline" => deadline.set(Instant::now()),
                        _ => {}
                    }
                }
            },
        )
        .unwrap();
        let result = super::publication::finish(
            &mut reports,
            &mut receipt,
            cancel.is_cancelled(),
            Instant::now() >= deadline.get(),
            mode != "finish",
        );
        assert_eq!(result.is_ok(), mode == "success");
        if mode == "existing" {
            assert_eq!(
                std::fs::read(root.join("moe-expert-smoke-table.md")).unwrap(),
                b"foreign evidence"
            );
            let aggregate: serde_json::Value =
                serde_json::from_slice(&std::fs::read(root.join("moe-expert-smoke.json")).unwrap())
                    .unwrap();
            assert_eq!(aggregate["status"], "incomplete");
            assert_eq!(aggregate["publication_refused"], true);
            assert_eq!(aggregate["cases"], receipt["cases"]);
        } else {
            assert_publication(root, &receipt, mode == "success");
        }
    }
}
#[test]
#[cfg(unix)]
fn moe_publication_actual_signals_stay_owned_through_report_writes() {
    use crate::process::{
        self, Cancellation, Completion, Limits, OutputFiles, ProcessSpec, Readiness, Value as Arg,
    };
    use std::time::Duration;
    for signal in [libc::SIGINT, libc::SIGTERM] {
        let directory = tempfile::tempdir().unwrap();
        let root = directory.path().canonicalize().unwrap();
        let result = process::supervise(
            &ProcessSpec {
                executable: std::env::current_exe().unwrap(),
                arguments: [
                    "--ignored",
                    "--exact",
                    "automation::cache_family_moe::tests::moe_publication_signal_subprocess",
                    "--nocapture",
                ]
                .into_iter()
                .map(|s| Arg::Public(s.into()))
                .collect(),
                cwd: root.clone(),
                environment: std::collections::BTreeMap::from([
                    (
                        "MOE_PUBLICATION_ROOT".into(),
                        Arg::Public(root.clone().into_os_string()),
                    ),
                    (
                        "MOE_PUBLICATION_SIGNAL".into(),
                        Arg::Public(signal.to_string().into()),
                    ),
                ]),
            },
            &Limits {
                execution: Duration::from_secs(10),
                graceful_shutdown: Duration::from_secs(1),
                forced_shutdown: Duration::from_secs(1),
                retained_bytes_per_stream: 65536,
                readiness: Readiness::None,
                completion: Completion::Exit,
            },
            &Cancellation::default(),
            OutputFiles::default(),
        )
        .unwrap();
        assert!(result.success(), "{result:?}");
        assert!(
            result.failure.is_none()
                && result.cleanup.complete
                && !result.cleanup.forced
                && !result.cleanup.graceful_signal_failed
                && result.cleanup.failure.is_none()
        );
        assert!(
            [&result.stdout, &result.stderr]
                .iter()
                .all(|s| s.line_capture_complete && !s.truncated && s.oversized_lines == 0)
        );
        assert_publication(&root, &publication_receipt(), false);
        assert_eq!(
            std::fs::read_to_string(root.join("signal-observed")).unwrap(),
            signal.to_string()
        );
    }
}
#[test]
#[cfg(unix)]
#[ignore = "isolated real signal publication callback; selected through owning outer test"]
fn moe_publication_signal_subprocess() {
    let root = std::path::PathBuf::from(std::env::var_os("MOE_PUBLICATION_ROOT").unwrap());
    let signal = std::env::var("MOE_PUBLICATION_SIGNAL")
        .unwrap()
        .parse::<i32>()
        .unwrap();
    assert!([libc::SIGINT, libc::SIGTERM].contains(&signal));
    let interrupt = crate::automation::command_interrupt::Interrupt::install().unwrap();
    let cancel = interrupt.cancellation();
    let mut receipt = publication_receipt();
    let mut reports = super::publication::publish(
        &root,
        &mut receipt,
        &mut || (cancel.is_cancelled(), false),
        &mut |index| {
            if index == 3 {
                // SAFETY: an isolated helper owns the installed fixed SIGINT/SIGTERM handlers.
                assert_eq!(unsafe { libc::raise(signal) }, 0);
                assert!(cancel.is_cancelled());
                std::fs::write(root.join("signal-observed"), signal.to_string()).unwrap();
            }
        },
    )
    .unwrap();
    let finish = interrupt.finish();
    assert!(finish.is_err());
    assert!(
        super::publication::finish(
            &mut reports,
            &mut receipt,
            cancel.is_cancelled(),
            false,
            finish.is_ok()
        )
        .is_err()
    );
    assert_publication(&root, &receipt, false);
}
