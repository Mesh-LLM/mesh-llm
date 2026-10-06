use super::*;
use serde_json::{Value, json};
fn request() -> Value {
    json!({"schema_version":1,"hardware":[{"name":"cpu-upgrade","cpu":"8 vCPU","ram":"32 GB","unitCostUSD":0.01,"unit_label":"minute"}],"requested_flavor":"auto","requested_timeout_seconds":30,"model_size_bytes":1024,"max_cost_usd":2.0})
}
fn invoke(temp: &std::path::Path, value: &Value, extra: &[&str]) -> Result<bool> {
    let input = temp.join("input.json");
    std::fs::write(&input, serde_json::to_vec(value)?)?;
    let mut args: Vec<std::ffi::OsString> = ["model-package-generic-jobs", "plan", "--input"]
        .map(Into::into)
        .into();
    args.push(input.into_os_string());
    args.push("--output-directory".into());
    args.push(temp.join("output").into_os_string());
    args.extend(extra.iter().map(|value| std::ffi::OsString::from(*value)));
    super::super::run_args(args)
}
#[test]
fn offline_plan_actual_facade_emits_existing_planner_and_typed_digest() {
    let temp = tempfile::tempdir().unwrap();
    assert!(invoke(temp.path(), &request(), &[]).unwrap());
    let output = temp.path().join("output");
    let bytes = std::fs::read(output.join("cpu-plan.json")).unwrap();
    let plan: CpuJobPlan = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(plan.flavor, "cpu-upgrade");
    assert_eq!(plan.timeout_seconds, 7200);
    assert!(plan.auto_selected_hardware && plan.timeout_bumped_to_minimum);
    assert_eq!(plan.max_cost_usd, 1.2);
    let resources: Value =
        serde_json::from_slice(&std::fs::read(output.join("bootstrap-resources.json")).unwrap())
            .unwrap();
    assert_eq!(resources["timeout_seconds"], 7200);
    assert_eq!(resources["max_cost_usd"], 2.0);
    assert_eq!(
        resources["cpu_plan_receipt_sha256"],
        super::super::super::admission::digest(&serde_json::to_vec(&plan).unwrap())
    );
    assert_ne!(
        resources["cpu_plan_receipt_sha256"],
        super::super::super::admission::digest(&bytes)
    );
    let result: Value =
        serde_json::from_slice(&std::fs::read(output.join("result.json")).unwrap()).unwrap();
    assert_eq!(result["status"], "PLANNED_OFFLINE");
    assert_eq!(result["submitted"], false);
    assert_eq!(result["hardware_observed"], false);
    assert!(!output.join("submitted.json").exists());
    assert!(invoke(temp.path(), &request(), &[]).is_err());
    temp.close().unwrap();
}
#[test]
fn offline_plan_refuses_cost_hardware_bounds_and_credentials_before_output() {
    for (pointer, value) in [
        ("/max_cost_usd", json!(0.5)),
        ("/requested_timeout_seconds", json!(259201)),
        ("/model_size_bytes", json!(0)),
        ("/requested_flavor", json!("missing")),
        ("/hardware/0/unitCostUSD", json!(-0.1)),
        ("/hardware/0/accelerator", json!("GPU")),
    ] {
        let temp = tempfile::tempdir().unwrap();
        let mut input = request();
        // accelerator is absent in the source fixture; insert through the owning object.
        if pointer.ends_with("accelerator") {
            input["hardware"][0]["accelerator"] = value;
        } else {
            *input.pointer_mut(pointer).unwrap() = value;
        }
        assert!(invoke(temp.path(), &input, &[]).is_err());
        assert!(!temp.path().join("output").exists());
        temp.close().unwrap();
    }
    let temp = tempfile::tempdir().unwrap();
    assert!(invoke(temp.path(), &request(), &["--confirm-submission"]).is_err());
    assert!(!temp.path().join("output").exists());
    temp.close().unwrap();
}
#[test]
fn offline_plan_preserves_original_72h_explicit_hardware_selection() {
    let temp = tempfile::tempdir().unwrap();
    let mut input = request();
    input["requested_timeout_seconds"] = json!(259200);
    input["requested_flavor"] = json!("cpu-upgrade");
    input["max_cost_usd"] = json!(50.0);
    assert!(invoke(temp.path(), &input, &[]).unwrap());
    let plan: CpuJobPlan =
        serde_json::from_slice(&std::fs::read(temp.path().join("output/cpu-plan.json")).unwrap())
            .unwrap();
    assert_eq!(plan.timeout_seconds, 259200);
    assert!(!plan.auto_selected_hardware && !plan.timeout_bumped_to_minimum);
    assert_eq!(plan.requested_timeout_seconds, 259200);
    temp.close().unwrap();
}
