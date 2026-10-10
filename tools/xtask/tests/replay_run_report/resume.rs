use std::path::Path;

pub(super) fn assert_resume_integrity(input: &Path, run: &serde_json::Value, passed: bool) {
    let mut request: serde_json::Value =
        serde_json::from_slice(&std::fs::read(input).unwrap()).unwrap();
    let output = std::path::PathBuf::from(request["output"].as_str().unwrap());
    if request.get("build_jobs").is_some() {
        request.as_object_mut().unwrap().remove("build_jobs");
        request["builds"] = run["builds"].clone();
    }
    request["resume"] = true.into();
    std::fs::write(input, serde_json::to_vec(&request).unwrap()).unwrap();
    let resume = || {
        std::process::Command::new(env!("CARGO_BIN_EXE_xtask"))
            .args(["automation", "replay-matrix", "execute-run", "--input"])
            .arg(input)
            .output()
            .unwrap()
    };
    let resumed = resume();
    assert!(
        resumed.status.success() == passed,
        "{}",
        String::from_utf8_lossy(&resumed.stderr)
    );
    let resumed_run: serde_json::Value =
        serde_json::from_slice(&std::fs::read(output.join("run.json")).unwrap()).unwrap();
    assert_eq!(resumed_run["results"], run["results"]);
    assert_eq!(resumed_run["order"], run["order"]);
    let original_model = request["model"].clone();
    let copy = input.parent().unwrap().join("same-bytes-model.gguf");
    std::fs::copy(original_model.as_str().unwrap(), &copy).unwrap();
    request["model"] = serde_json::to_value(&copy).unwrap();
    std::fs::write(input, serde_json::to_vec(&request).unwrap()).unwrap();
    let rejected = resume();
    assert!(!rejected.status.success());
    assert!(String::from_utf8_lossy(&rejected.stderr).contains("plan_sha256 differs"));
    request["model"] = original_model;
    let original_reference = request.get("model_reference").cloned();
    request["model_reference"] = "foreign/model@revision/file.gguf".into();
    std::fs::write(input, serde_json::to_vec(&request).unwrap()).unwrap();
    assert!(!resume().status.success());
    match original_reference {
        Some(reference) => {
            request["model_reference"] = reference;
        }
        None => {
            request.as_object_mut().unwrap().remove("model_reference");
        }
    }
    request["max_output_tokens"] = 1024.into();
    std::fs::write(input, serde_json::to_vec(&request).unwrap()).unwrap();
    assert!(!resume().status.success());
    let retained: serde_json::Value =
        serde_json::from_slice(&std::fs::read(output.join("run.json")).unwrap()).unwrap();
    assert_eq!(retained, resumed_run);
    request["max_output_tokens"] = 2048.into();
    std::fs::write(input, serde_json::to_vec(&request).unwrap()).unwrap();
    let mut altered_order = resumed_run.clone();
    altered_order["order"][0]["label"] = "foreign".into();
    std::fs::write(
        output.join("run.json"),
        serde_json::to_vec(&altered_order).unwrap(),
    )
    .unwrap();
    let rejected = resume();
    assert!(!rejected.status.success());
    assert!(String::from_utf8_lossy(&rejected.stderr).contains("retained arm order differs"));
    let mut altered = resumed_run.clone();
    altered["inputs"]["kind"] = "foreign".into();
    std::fs::write(
        output.join("run.json"),
        serde_json::to_vec(&altered).unwrap(),
    )
    .unwrap();
    let rejected = resume();
    assert!(!rejected.status.success());
    assert!(String::from_utf8_lossy(&rejected.stderr).contains("input provenance differs"));
    let mut altered = resumed_run.clone();
    altered["results"][0]["passed"] = false.into();
    altered["results"][0]["acceptance_failed"] = false.into();
    std::fs::write(
        output.join("run.json"),
        serde_json::to_vec(&altered).unwrap(),
    )
    .unwrap();
    let rejected = resume();
    assert!(!rejected.status.success());
    assert!(String::from_utf8_lossy(&rejected.stderr).contains("failed infrastructure"));
    let mut altered = resumed_run.clone();
    altered["results"][0]["cells"][0]["completion_tokens"] = 999.into();
    std::fs::write(
        output.join("run.json"),
        serde_json::to_vec(&altered).unwrap(),
    )
    .unwrap();
    let rejected = resume();
    assert!(!rejected.status.success());
    assert!(String::from_utf8_lossy(&rejected.stderr).contains("retained cell summary"));
    let mut altered = resumed_run.clone();
    altered["results"][0]["commit"] = "foreign".into();
    std::fs::write(
        output.join("run.json"),
        serde_json::to_vec(&altered).unwrap(),
    )
    .unwrap();
    let rejected = resume();
    assert!(!rejected.status.success());
    assert!(String::from_utf8_lossy(&rejected.stderr).contains("retained arm provenance differs"));
    std::fs::write(
        output.join("run.json"),
        serde_json::to_vec(&resumed_run).unwrap(),
    )
    .unwrap();
    std::fs::write(
        output.join("data/pass-1/baseline/c-1-requests.jsonl"),
        b"changed evidence\n",
    )
    .unwrap();
    let rejected = resume();
    assert!(!rejected.status.success());
    assert!(
        String::from_utf8_lossy(&rejected.stderr).contains("retained pass artifact bytes changed")
    );
}
