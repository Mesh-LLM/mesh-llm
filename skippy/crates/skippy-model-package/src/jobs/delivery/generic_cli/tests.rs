use super::*;
fn fixture() -> Input {
    let (v, mounts, plan) = super::super::generic::facade_fixture();
    Input {
        schema_version: 1,
        namespace: "fixture".into(),
        worker_input: v,
        mounts,
        cpu_plan: plan,
    }
}
#[test]
fn generic_facade_actual_prepare_dispatch_has_no_secret_or_submission_authority() {
    let temp = tempfile::tempdir().unwrap();
    let root = temp.path().canonicalize().unwrap();
    let input = fixture();
    let source = root.join("input.json");
    std::fs::write(&source, serde_json::to_vec(&input).unwrap()).unwrap();
    let out = root.join("prepared");
    assert!(
        run_args(
            [
                "model-package-generic-jobs",
                "prepare",
                "--input",
                source.to_str().unwrap(),
                "--output-directory",
                out.to_str().unwrap()
            ]
            .map(Into::into)
        )
        .unwrap()
    );
    let result: Value =
        serde_json::from_slice(&std::fs::read(out.join("result.json")).unwrap()).unwrap();
    assert_eq!(result["status"], "PREPARED");
    assert_eq!(result["submitted"], false);
    assert_eq!(result["conversion_admitted"], false);
    let declaration = std::fs::read_to_string(out.join("declaration.json")).unwrap();
    assert!(!declaration.contains("worker_input") && !declaration.contains("MESH_HF_JOB_INPUT"));
    assert_eq!(
        serde_json::from_str::<Value>(&declaration).unwrap()["timeout_seconds"],
        259200
    );
    temp.close().unwrap();
}
#[test]
fn generic_facade_rehydrated_acknowledgment_requires_original_request_and_declaration() {
    let input = fixture();
    let prepared = prepare(&input).unwrap();
    let mut declaration: super::super::DeliveryDeclaration =
        serde_json::from_value(serde_json::to_value(prepared.declaration()).unwrap()).unwrap();
    declaration.submitted = true;
    let mut ack = Acknowledgment {
        schema_version: 1,
        namespace: input.namespace.clone(),
        input_sha256: hash(&input).unwrap(),
        submitted: SubmittedConversionDelivery {
            native: super::super::SubmittedCertificationDelivery {
                declaration,
                job_id: "job-1".into(),
                stage: crate::jobs::JobStage::Pending,
            },
            expected_status: prepared.expected_status().into(),
        },
    };
    correlated(&input, &ack).unwrap();
    ack.submitted.native.declaration.evidence_path = "foreign.json".into();
    assert!(correlated(&input, &ack).is_err());
    ack.submitted.native.declaration.evidence_path =
        input.worker_input["receipt_export"]["path_in_repo"]
            .as_str()
            .unwrap()
            .into();
    ack.input_sha256 = "0".repeat(64);
    assert!(correlated(&input, &ack).is_err());
}

#[test]
fn generic_facade_closed_cli_missing_fields_refuse_without_global_required_assertion() {
    assert!(run_args(["model-package-generic-jobs", "prepare"].map(Into::into)).is_err());
    assert!(
        run_args(
            [
                "model-package-generic-jobs",
                "submit",
                "--confirm-submission"
            ]
            .map(Into::into)
        )
        .is_err()
    );
}

#[test]
fn generic_facade_terminal_gate_preserves_ack_rows_and_refuses_late_cancel_deadline_or_prior_error()
{
    for mode in ["success", "prior", "cancel", "deadline", "late-cancel"] {
        let temp = tempfile::tempdir().unwrap();
        let root = temp.path().canonicalize().unwrap();
        write(&root,"observations.json",&json!({"job_id":"job-1","rows":[{"measured":true}],"error":if mode=="prior"{Some("original")}else{None}})).unwrap();
        let calls = std::cell::Cell::new(0);
        let until = if mode == "deadline" {
            Instant::now()
        } else {
            Instant::now() + Duration::from_secs(10)
        };
        let admitted=final_result(&root,json!({"rows":[{"measured":true}],"submission_acknowledged":true,"conversion_admitted":true,"error":if mode=="prior"{Some("original")}else{None}}),(mode!="prior","CONVERSION_ADMITTED","OBSERVATIONS_ONLY"),until,||{calls.set(calls.get()+1);mode=="cancel" || (mode=="late-cancel"&&calls.get()>1)}).unwrap();
        assert_eq!(admitted, mode == "success");
        let result: Value =
            serde_json::from_slice(&std::fs::read(root.join("result.json")).unwrap()).unwrap();
        assert_eq!(result["operation_completed"], mode == "success");
        assert_eq!(result["conversion_admitted"], mode == "success");
        assert_eq!(result["rows"][0]["measured"], true);
        assert_eq!(result["submission_acknowledged"], true);
        if mode == "prior" {
            assert_eq!(result["error"], "original");
        }
        assert!(root.join("observations.json").is_file());
        temp.close().unwrap();
    }
}
