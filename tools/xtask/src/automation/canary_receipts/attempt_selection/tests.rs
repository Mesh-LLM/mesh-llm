use super::*;
use serde_json::{Value, json};

fn green(result: &str, head: char) -> Value {
    json!({"result":result,"outputs":{
        "green":"true","repairable":"false","head":head.to_string().repeat(40),
        "package":"producer-package","identity":"b".repeat(64),"branch":"llama-canary/fixture"
    }})
}
fn resume() -> Value {
    json!({"result":"failure","outputs":{
        "repairable":"true","head":"a".repeat(40),"package":"candidate-package",
        "identity":"b".repeat(64),"feedback":"candidate-feedback",
        "failure_class":"candidate","failure_stage":"family-certification"
    }})
}
fn job(value: &Value) -> Job {
    decode(&value.to_string()).unwrap()
}
fn attempt(
    changed: bool,
    repair: Value,
    verification: Value,
) -> DynResult<BTreeMap<&'static str, String>> {
    let repair = job(&repair);
    let verification = job(&verification);
    repair.validate()?;
    verification.validate()?;
    Ok(select_attempt(changed, &repair, &verification)?.outputs())
}
fn final_outputs(
    certify: bool,
    changed: bool,
    selected: &str,
    preflight: JobResult,
    value: Value,
) -> DynResult<BTreeMap<&'static str, String>> {
    let attempts = decode(&value.to_string())?;
    Ok(select_final(certify, changed, selected, preflight, &attempts)?.outputs())
}
fn selected(result: &str, state: &str) -> Value {
    let mut value = green(result, 'a');
    value["outputs"]["state"] = json!(state);
    value
}
#[test]
fn reconciled_green_survives_earlier_failed_nested_jobs_and_uses_verifier_identity() {
    let mut verified = green("failure", 'a');
    verified["outputs"]["package"] = json!("independent-package");
    verified["outputs"]["identity"] = json!("c".repeat(64));
    let result = attempt(true, green("failure", 'a'), verified).unwrap();
    assert_eq!(result["state"], "green");
    assert_eq!(result["package"], "independent-package");
    assert_eq!(result["identity"], "c".repeat(64));
    assert_eq!(
        attempt(true, green("success", 'a'), green("success", 'c')).unwrap()["failure_stage"],
        "independent-verification-identity"
    );
}
#[test]
fn candidate_failure_can_resume_from_repair_or_verification_but_other_classes_cannot() {
    let empty = json!({"result":"skipped","outputs":{}});
    for (repair, verifier) in [(resume(), empty.clone()), (green("success", 'a'), resume())] {
        let result = attempt(true, repair, verifier).unwrap();
        assert_eq!(result["state"], "repairable");
        assert_eq!(result["resume_feedback"], "candidate-feedback");
    }
    for class in ["infrastructure", "contract", "environment"] {
        let mut value = resume();
        value["outputs"]["failure_class"] = json!(class);
        let result = attempt(true, value, empty.clone()).unwrap();
        assert_eq!(result["state"], "failed");
        assert_eq!(result["repairable"], "false");
        assert!(!result.contains_key("resume_feedback"));
    }
    assert_eq!(
        attempt(false, resume(), empty.clone()).unwrap()["state"],
        "failed"
    );
    assert_eq!(
        attempt(false, green("failure", 'a'), empty).unwrap()["state"],
        "green"
    );
}
#[test]
fn incomplete_or_injected_handoffs_and_changed_verification_identity_cannot_resume() {
    let empty = json!({"result":"skipped","outputs":{}});
    for key in ["package", "identity", "head", "feedback"] {
        let mut value = resume();
        value["outputs"][key] = json!("");
        assert!(attempt(true, value, empty.clone()).is_err(), "{key}");
    }
    for value in ["x\npublish=true", "x\ry", "x\0y", "x\u{1f}y"] {
        let mut candidate = resume();
        candidate["outputs"]["feedback"] = json!(value);
        assert!(attempt(true, candidate, empty.clone()).is_err());
    }
    let mut verifier = resume();
    verifier["outputs"]["head"] = json!("c".repeat(40));
    assert_eq!(
        attempt(true, green("success", 'a'), verifier).unwrap()["failure_stage"],
        "independent-verification-identity"
    );
}
#[test]
fn exact_boolean_and_typed_identity_admission_never_coerces_json_values() {
    for value in ["", "TRUE", "1", "true\n"] {
        assert!(boolean(value).is_err());
    }
    for value in [json!(true), json!(0), json!(null), json!("TRUE")] {
        let mut candidate = green("success", 'a');
        candidate["outputs"]["green"] = value;
        if let Ok(job) = decode::<Job>(&candidate.to_string()) {
            assert!(job.validate().is_err());
        }
    }
    assert!(decode::<Job>(&" ".repeat(INPUT_LIMIT + 1)).is_err());
    for head in ["a", &"A".repeat(40), &"g".repeat(40)] {
        assert!(input::head(head).is_err());
    }
    assert!(input::digest(&"B".repeat(64)).is_err());
}
#[test]
fn final_selects_latest_numeric_slot_and_never_falls_back_from_later_failure() {
    let value = json!({"attempt_1":selected("success","repairable"),"attempt_2":selected("success","green"),"attempt_3":{"result":"skipped","outputs":{}},"resolve":{"outputs":{"state":"green"}}});
    let result = final_outputs(true, true, "", JobResult::Success, value.clone()).unwrap();
    assert_eq!(result["publish"], "true");
    for later in [
        selected("success", "failed"),
        selected("failure", "green"),
        json!({"result":"cancelled","outputs":{}}),
        json!({"result":"failure","outputs":{}}),
        json!({"result":"success","outputs":{}}),
    ] {
        let mut changed = value.clone();
        changed["attempt_3"] = later;
        assert!(final_outputs(true, true, "", JobResult::Success, changed).is_err());
    }
    let mut unknown = value;
    unknown["attempt_99"] = selected("success", "green");
    assert!(final_outputs(true, true, "", JobResult::Success, unknown).is_err());
}
#[test]
fn preflight_noop_historical_unchanged_and_exhaustion_preserve_publication_boundary() {
    assert_eq!(
        final_outputs(false, false, "", JobResult::Skipped, json!({})).unwrap()["state"],
        "noop"
    );
    let valid = json!({"attempt_1": selected("success", "green")});
    assert!(final_outputs(true, true, "", JobResult::Failure, valid.clone()).is_err());
    assert_eq!(
        final_outputs(true, false, "", JobResult::Success, valid.clone()).unwrap()["publish"],
        "false"
    );
    assert_eq!(
        final_outputs(
            true,
            false,
            &"a".repeat(40),
            JobResult::Success,
            valid.clone()
        )
        .unwrap()["publish"],
        "false"
    );
    for (changed, source) in [(true, "a".repeat(40)), (false, "c".repeat(40))] {
        assert!(final_outputs(true, changed, &source, JobResult::Success, valid.clone()).is_err());
    }
    let mut exhausted = valid;
    for slot in ["attempt_1", "attempt_2", "attempt_3"] {
        exhausted[slot] = selected("success", "repairable");
    }
    assert!(final_outputs(true, true, "", JobResult::Success, exhausted).is_err());
}

#[test]
fn cancelled_skipped_jobs_and_contradictory_final_claims_never_admit_progress() {
    let empty = json!({"result":"skipped","outputs":{}});
    for status in ["cancelled", "skipped"] {
        assert!(attempt(true, green(status, 'a'), green("success", 'a')).is_err());
        let mut value = resume();
        value["result"] = json!(status);
        assert!(attempt(true, value, empty.clone()).is_err());
    }
    for (key, value) in [
        ("green", "false"),
        ("repairable", "true"),
        ("state", "unknown"),
    ] {
        let mut input = json!({"attempt_1":selected("success", "green")});
        input["attempt_1"]["outputs"][key] = json!(value);
        assert!(final_outputs(true, true, "", JobResult::Success, input).is_err());
    }
    for value in [" ", &"x".repeat(LINE_LIMIT + 1)] {
        assert!(safe_line(value, "package").is_err());
    }
}

#[test]
fn contradictory_producer_progress_flags_are_rejected_before_normalization() {
    for verifier in [false, true] {
        let mut repair = green("failure", 'a');
        let mut verification = green("failure", 'a');
        if verifier {
            verification["outputs"]["repairable"] = json!("true");
        } else {
            repair["outputs"]["repairable"] = json!("true");
        }
        assert!(attempt(true, repair, verification).is_err());
    }
}
