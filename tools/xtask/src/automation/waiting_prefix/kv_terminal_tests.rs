use super::*;
use std::time::Duration;
fn observations(worker: bool) -> Value {
    let mut value = json!({"schema_version":1,"requests":[{"cohort":"fill","prompt_tokens":10,"cached_tokens":0,"content_sha256":"observed","error":null}],"error":null});
    if worker {
        value["request_sha256"] = json!("input-bound");
        value["manifest_sha256"] = json!("manifest-bound");
        value["baseline_sha256"] = json!("observed");
    } else {
        value["sessions"] = json!([{"members":[{"cleanup_complete":true,"forced":false}]}]);
        value["cohorts"] = json!([{"cohort":"fill","requests":1,"failed":0}]);
    }
    value
}
fn retained(before: &Value, after: &Value) {
    for (key, value) in before.as_object().unwrap() {
        if key != "error" {
            assert_eq!(value, &after[key], "{key}");
        }
    }
}
#[test]
fn restart_terminal_success_preserves_run_and_worker_observations() {
    for worker in [false, true] {
        let mut value = observations(worker);
        let before = value.clone();
        finalize(
            &mut value,
            Ok(()),
            &Cancellation::default(),
            Instant::now() + Duration::from_secs(60),
        )
        .unwrap();
        assert_eq!(value["terminal_complete"], true);
        assert!(value["terminal_error"].is_null());
        retained(&before, &value);
    }
}
#[test]
fn restart_terminal_cancel_refuses_run_and_worker_without_erasing_rows() {
    let cancel = Cancellation::default();
    cancel.cancel();
    for worker in [false, true] {
        let mut value = observations(worker);
        let before = value.clone();
        assert!(
            finalize(
                &mut value,
                Ok(()),
                &cancel,
                Instant::now() + Duration::from_secs(60)
            )
            .is_err()
        );
        assert_eq!(value["terminal_complete"], false);
        assert_eq!(value["error"], "restart terminal cancellation");
        retained(&before, &value);
    }
}
#[test]
fn restart_terminal_deadline_refuses_run_and_worker_without_erasing_rows() {
    for worker in [false, true] {
        let mut value = observations(worker);
        let before = value.clone();
        assert!(finalize(&mut value, Ok(()), &Cancellation::default(), Instant::now()).is_err());
        assert_eq!(value["terminal_complete"], false);
        assert_eq!(value["error"], "restart terminal deadline expired");
        retained(&before, &value);
    }
}
#[test]
fn restart_terminal_finish_failure_preserves_prior_error_without_private_details() {
    for worker in [false, true] {
        for prior in [None, Some("earlier session shutdown failed")] {
            let mut value = observations(worker);
            if let Some(error) = prior {
                value["error"] = json!(error);
            }
            let before = value.clone();
            assert!(
                finalize(
                    &mut value,
                    Err("private finalizer detail".into()),
                    &Cancellation::default(),
                    Instant::now() + Duration::from_secs(60)
                )
                .is_err()
            );
            assert_eq!(value["terminal_complete"], false);
            assert_eq!(
                value["terminal_error"],
                "restart interrupt finalization failed"
            );
            assert_eq!(
                value["error"],
                prior.unwrap_or("restart interrupt finalization failed")
            );
            retained(&before, &value);
            assert!(!value.to_string().contains("private finalizer detail"));
        }
    }
}
#[test]
fn restart_original_default_workload_is_admitted_without_server_tuning() {
    let root = std::env::current_dir().unwrap().canonicalize().unwrap();
    let input = super::super::kv_command::Input {
        schema_version: 1,
        binary: root.join("not-launched-host"),
        model: root.join("not-opened-model.gguf"),
        turns: 4,
        turn_target_tokens: 4750,
        system_tokens: 500,
        restore_repeats: 3,
        max_output_tokens: 256,
        request_timeout_secs: 900.0,
        ready_timeout_secs: 900,
        worker_timeout_secs: 7200,
        timeout_secs: 30000,
        serve_extra_args: vec![],
    };
    input.validate().unwrap();
    let manifest = super::super::kv_manifest::build(
        input.turns,
        input.turn_target_tokens,
        input.system_tokens,
    )
    .unwrap();
    manifest.validate().unwrap();
    assert_eq!(manifest.settings["approx_total_prompt_tokens"], 19500);
    assert_eq!(manifest.turns.len(), 4);
    let last = manifest.messages(3).unwrap();
    assert_eq!(last.len(), 8);
    assert_eq!(last.last().unwrap()["role"], "user");
    assert_eq!(
        super::super::kv_command::server_args(&input.model, &[])
            .unwrap()
            .len(),
        5
    );
}
