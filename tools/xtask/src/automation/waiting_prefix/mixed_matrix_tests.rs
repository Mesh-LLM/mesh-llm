use super::*;
fn input() -> Input {
    let arm = |version: &str| {
        serde_json::from_value(json!({"schema_version":1,"round":1,"version":version,"binary":"/fixture/binary","binary_sha256":"a".repeat(64),"commit":"b".repeat(40),"native_build":"/fixture/native","native_build_sha256":"c".repeat(64),"native_profile":"standalone-static-skippy-server","model":"/fixture/model","model_sha256":"d".repeat(64),"model_id":"fixture","ctx_size":32768,"split_layer":14,"layer_end":27,"n_gpu_layers":999,"adaptive_target_ms":100.0,"stage_ports":[9000,9001],"openai_port":9002})).unwrap()
    };
    Input{schema_version:1,old:arm("old"),new:arm("new"),shape:serde_json::from_value(json!({"rounds":2,"anchors":1,"prefills":1,"anchor_prompt_blocks":8,"prefill_prompt_blocks":256,"anchor_output_tokens":128,"prefill_output_tokens":8,"prefill_delay_ms":50.0,"prefill_stagger_ms":5.0,"lanes":2,"n_batch":1024,"n_ubatch":256,"prefill_adaptive_start":256,"prefill_adaptive_step":256,"prefill_adaptive_max":256,"adaptive_target_new_only":true})).unwrap(),split:true,manifest:Some(serde_json::from_value(json!({"metadata":{"revision":"pinned"},"prompts":[{"family":"trace1","source_id":"r1","prompt":"first trace"},{"family":"trace2","source_id":"r2","prompt":"second trace"}]})).unwrap()),request_timeout_secs:2.0,worker_timeout_secs:5,suppressed_token_ids:vec![7],timeout_secs:120,cell_timeout_secs:20,startup_timeout_secs:2}
}
fn cell(input: &mixed_cell::Input) -> Value {
    input.validate().unwrap();
    let rows=input.worker.requests.iter().map(|r|json!({"request_index":r.request_index,"role":r.role,"prompt_sha256":identity::digest(r.prompt.prompt.as_bytes()),"prompt_provenance":r.prompt.provenance,"content_sha256":"a".repeat(64),"ttft_ms":1.0,"content_gaps_ms":[0.5],"completion_tokens":2,"error":null})).collect::<Vec<_>>();
    let worker = json!({"requests":rows,"makespan_ms":10.0,"error":null});
    let summary = summary::requests(&worker).unwrap();
    json!({"requests":worker,"summary":summary,"fixture_scope":"injected-orchestration-only"})
}
#[test]
fn mixed_matrix_injected_dispatch_alternates_exact_roles_trace_rounds_and_old_new_profile() {
    let input = input();
    let cancel = Cancellation::default();
    let mut calls = Vec::new();
    let output = execute_with(
        &input,
        Path::new("/fixture/output"),
        Instant::now() + Duration::from_secs(120),
        &cancel,
        |cell_input, _, _, _| {
            calls.push((cell_input.arm.round, cell_input.arm.version));
            assert_eq!(cell_input.shape.n_batch, 1024);
            assert!(cell_input.split);
            assert_eq!(cell_input.worker.requests[0].role, Role::Anchor);
            assert_eq!(cell_input.worker.requests[0].delay_ms, 0.0);
            assert_eq!(cell_input.worker.requests[0].output_tokens, 128);
            assert_eq!(cell_input.worker.requests[1].role, Role::Prefill);
            assert_eq!(cell_input.worker.requests[1].delay_ms, 50.0);
            assert_eq!(cell_input.worker.requests[1].output_tokens, 8);
            assert_eq!(
                cell_input.worker.requests[1].prompt.provenance["source_id"],
                if cell_input.arm.round == 1 {
                    "r1"
                } else {
                    "r2"
                }
            );
            assert_eq!(cell_input.worker.manifest_metadata["revision"], "pinned");
            Ok(cell(cell_input))
        },
    );
    assert_eq!(
        calls,
        vec![
            (1, Version::Old),
            (1, Version::New),
            (2, Version::New),
            (2, Version::Old)
        ]
    );
    assert!(output["error"].is_null());
    assert_eq!(output["cells"].as_array().unwrap().len(), 4);
    assert_eq!(output["comparison"]["qualified"], false);
}
#[test]
fn mixed_matrix_injected_failure_cancel_deadline_and_profile_mismatch_are_closed() {
    for mode in ["failure", "cancel", "deadline"] {
        let input = input();
        let cancel = Cancellation::default();
        let mut count = 0;
        let until = Instant::now()
            + if mode == "deadline" {
                Duration::ZERO
            } else {
                Duration::from_secs(120)
            };
        let output = execute_with(
            &input,
            Path::new("/fixture/output"),
            until,
            &cancel,
            |cell_input, _, _, _| {
                count += 1;
                if count == 2 {
                    return Err("injected later-arm refusal".into());
                }
                if mode == "cancel" {
                    cancel.cancel();
                }
                Ok(cell(cell_input))
            },
        );
        assert!(!output["error"].is_null());
        assert_eq!(
            output["cells"].as_array().unwrap().len(),
            if mode == "deadline" { 0 } else { 1 }
        );
        assert_eq!(
            count,
            if mode == "deadline" {
                0
            } else if mode == "failure" {
                2
            } else {
                1
            }
        );
    }
    let mut changed = input();
    changed.new.ctx_size = 65536;
    assert!(changed.validate().is_err());
    let mut changed = input();
    changed.manifest.as_mut().unwrap().prompts.pop();
    assert!(changed.validate().is_err());
}

fn observed() -> Value {
    json!({"cells":[{"round":1,"version":"old","requests":[{"completion_tokens":2}]}],"comparison":{"qualified":false,"output_parity":{"exact_matches":1,"comparable_requests":1}},"error":null})
}
#[test]
fn mixed_terminal_success_preserves_unqualified_comparison_and_original_default_workload() {
    let mut output = observed();
    let comparison = output["comparison"].clone();
    finalize(
        &mut output,
        Ok(()),
        &Cancellation::default(),
        Instant::now() + Duration::from_secs(30),
    )
    .unwrap();
    assert_eq!(output["status"], "mixed_matrix_completed");
    assert_eq!(output["orchestration_completed"], true);
    assert_eq!(output["comparison"], comparison);
    assert_eq!(output["comparison"]["qualified"], false);
    assert!(output["terminal_error"].is_null());
    assert!(rendered(&output).contains("qualification remains"));
    let mut given = input();
    given.shape = serde_json::from_value(json!({"rounds":8,"anchors":4,"prefills":8,"anchor_prompt_blocks":8,"prefill_prompt_blocks":256,"anchor_output_tokens":128,"prefill_output_tokens":8,"prefill_delay_ms":100.0,"prefill_stagger_ms":5.0,"lanes":12,"n_batch":1024,"n_ubatch":256,"prefill_adaptive_start":256,"prefill_adaptive_step":256,"prefill_adaptive_max":256,"adaptive_target_new_only":false})).unwrap();
    given.manifest = None;
    let projected = worker(&given, &given.old, 1).unwrap();
    projected.validate().unwrap();
    assert_eq!(projected.requests.len(), 12);
    assert_eq!(
        projected
            .requests
            .iter()
            .filter(|r| r.role == Role::Anchor)
            .count(),
        4
    );
    assert_eq!(
        projected
            .requests
            .iter()
            .filter(|r| r.role == Role::Prefill)
            .count(),
        8
    );
    assert!(projected.requests[4].prompt.prompt.len() > 16 * 1024);
    assert_eq!(projected.requests[4].delay_ms, 100.0);
    assert_eq!(projected.requests[11].delay_ms, 135.0);
}
#[test]
fn mixed_terminal_cancel_retains_cells_and_comparison_without_completion() {
    let mut output = observed();
    let prior = output.clone();
    let cancel = Cancellation::default();
    cancel.cancel();
    assert!(
        finalize(
            &mut output,
            Ok(()),
            &cancel,
            Instant::now() + Duration::from_secs(30)
        )
        .is_err()
    );
    assert_eq!(output["cells"], prior["cells"]);
    assert_eq!(output["comparison"], prior["comparison"]);
    assert_eq!(output["orchestration_completed"], false);
    assert_eq!(output["status"], "mixed_matrix_failed");
    assert_eq!(output["terminal_error"], "mixed terminal cancellation");
    assert!(rendered(&output).contains("FAILED"));
}
#[test]
fn mixed_terminal_deadline_retains_observations_without_completion() {
    let mut output = observed();
    let prior = output.clone();
    assert!(
        finalize(
            &mut output,
            Ok(()),
            &Cancellation::default(),
            Instant::now()
        )
        .is_err()
    );
    assert_eq!(output["cells"], prior["cells"]);
    assert_eq!(output["comparison"], prior["comparison"]);
    assert_eq!(output["orchestration_completed"], false);
    assert_eq!(
        output["terminal_error"],
        "mixed terminal deadline exhausted"
    );
}
#[test]
fn mixed_terminal_finish_failure_preserves_prior_error_and_observations() {
    let mut output = observed();
    output["error"] = json!("prior parity refusal");
    let prior = output.clone();
    assert!(
        finalize(
            &mut output,
            Err("inert signal restoration refusal".into()),
            &Cancellation::default(),
            Instant::now() + Duration::from_secs(30)
        )
        .is_err()
    );
    assert_eq!(output["error"], "prior parity refusal");
    assert_eq!(output["cells"], prior["cells"]);
    assert_eq!(output["comparison"], prior["comparison"]);
    assert_eq!(
        output["terminal_error"],
        "mixed interrupt finalization failed"
    );
    assert_eq!(output["orchestration_completed"], false);
}
