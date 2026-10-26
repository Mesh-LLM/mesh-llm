use super::*;
use contract::{Case, digest, tree};
use std::collections::BTreeMap;
pub(super) fn tokenizer() -> Vec<u8> {
    serde_json::to_vec_pretty(&json!({"version":"1.0","truncation":null,"padding":null,"added_tokens":[{"id":0,"content":"[UNK]","single_word":false,"lstrip":false,"rstrip":false,"normalized":false,"special":true},{"id":1,"content":"[BOS]","single_word":false,"lstrip":false,"rstrip":false,"normalized":false,"special":true}],"normalizer":{"type":"Lowercase"},"pre_tokenizer":{"type":"Whitespace"},"post_processor":null,"decoder":null,"model":{"type":"WordLevel","vocab":{"[UNK]":0,"[BOS]":1,"hello":2,"world":3},"unk_token":"[UNK]"}})).unwrap()
}
pub(super) fn cases() -> Vec<Case> {
    vec![
        Case {
            text: "HELLO world".into(),
            add_special_tokens: false,
            decode_ids: vec![2, 3],
            skip_special_tokens: false,
            expected_ids: vec![2, 3],
            expected_decoded_sha256: digest(b"hello world"),
        },
        Case {
            text: "[BOS] hello".into(),
            add_special_tokens: true,
            decode_ids: vec![1, 2],
            skip_special_tokens: true,
            expected_ids: vec![1, 2],
            expected_decoded_sha256: digest(b"hello"),
        },
    ]
}
pub(super) fn expected() -> String {
    let serialized = tokenizers::Tokenizer::from_bytes(tokenizer())
        .unwrap()
        .to_string(false)
        .unwrap();
    let configuration=serde_json::to_vec_pretty(&json!({"tokenizer_class":"PreTrainedTokenizerFast","chat_template":"{{ messages }}","bos_token":"[BOS]"})).unwrap();
    tree(&BTreeMap::from([
        ("tokenizer.json".into(), digest(serialized.as_bytes())),
        ("tokenizer_config.json".into(), digest(&configuration)),
        ("chat_template.jinja".into(), digest(b"{{ messages }}")),
    ]))
    .unwrap()
}
#[test]
fn family_selection_default_filter_unknown_duplicate_and_complete_roster_are_admitted_explicitly() {
    let models = ["llama32-dense","deepseek-v2-moe","falcon-h1-recurrent","granite-h1-hybrid"].map(|k|json!({"key":k,"vllm_hf_config":{"repo":"owner/repo","revision":"a".repeat(40),"sha256":"b".repeat(64)}}));
    let exports = [
        "llama32-dense",
        "deepseek-v2-moe",
        "falcon-h1-recurrent",
        "granite-h1-hybrid",
    ]
    .map(|k| (k, "c".repeat(64)))
    .into_iter()
    .collect::<BTreeMap<_, _>>();
    let config = json!({"models":models});
    let mut request:Request=serde_json::from_value(json!({"schema_version":1,"config":"/unused","config_sha256":"a".repeat(64),"model_keys":[],"output_directory":"/unused-out","timeout_seconds":30,"maximum_bytes":1024,"credential_file":null,"export_sha256":exports,"semantic_cases":{"llama32-dense":cases(),"deepseek-v2-moe":cases(),"falcon-h1-recurrent":cases()}})).unwrap();
    assert_eq!(select(&config, &request).unwrap().len(), 4);
    request.model_keys = vec!["deepseek-v2-moe".into()];
    assert_eq!(
        select(&config, &request).unwrap()[0]["key"],
        "deepseek-v2-moe"
    );
    request.model_keys.push("deepseek-v2-moe".into());
    assert!(select(&config, &request).is_err());
    request.model_keys.clear();
    request.skip_dataset = true;
    request.skip_tokenizers = true;
    request.skip_vllm_configs = true;
    request.export_sha256.clear();
    request.semantic_cases.clear();
    assert_eq!(select(&config, &request).unwrap().len(), 4);
    request.model_keys = vec!["missing".into()];
    assert!(select(&config, &request).is_err());
}
#[cfg(unix)]
#[test]
fn native_export_preserves_known_source_ids_decoder_special_tokens_and_template_with_derived_lineage()
 {
    let temp = tempfile::tempdir().unwrap();
    let root = temp.path().canonicalize().unwrap();
    let source = root.join("source");
    std::fs::create_dir(&source).unwrap();
    std::fs::write(source.join("tokenizer.json"), tokenizer()).unwrap();
    std::fs::write(source.join("tokenizer_config.json"),serde_json::to_vec(&json!({"auto_map":{"AutoTokenizer":"not-executed.py"},"bos_token":"[BOS]","chat_template":"{{ messages }}"})).unwrap()).unwrap();
    let output = root.join("export");
    let receipt = export::fast(
        &source,
        &output,
        &expected(),
        &cases(),
        Instant::now() + Duration::from_secs(5),
    )
    .unwrap();
    assert_eq!(receipt["cases"].as_array().unwrap().len(), 2);
    assert_eq!(receipt["tree_sha256"], expected());
    assert_ne!(
        std::fs::read(output.join("tokenizer.json")).unwrap(),
        tokenizer()
    );
    let config: Value =
        serde_json::from_slice(&std::fs::read(output.join("tokenizer_config.json")).unwrap())
            .unwrap();
    assert_eq!(config["tokenizer_class"], "PreTrainedTokenizerFast");
    assert!(config.get("auto_map").is_none());
    assert_eq!(
        std::fs::read(output.join("chat_template.jinja")).unwrap(),
        b"{{ messages }}"
    );
    assert_eq!(
        std::fs::read(source.join("tokenizer.json")).unwrap(),
        tokenizer()
    );
    temp.close().unwrap();
}
#[cfg(unix)]
#[test]
fn export_refuses_wrong_known_ids_pin_duplicate_json_and_expired_budget_before_publication() {
    let temp = tempfile::tempdir().unwrap();
    let source = temp.path().canonicalize().unwrap();
    std::fs::write(source.join("tokenizer.json"), tokenizer()).unwrap();
    std::fs::write(
        source.join("tokenizer_config.json"),
        serde_json::to_vec(&json!({"bos_token":"[BOS]","chat_template":"{{ messages }}"})).unwrap(),
    )
    .unwrap();
    let output = source.parent().unwrap().join(format!(
        "{}-export",
        source.file_name().unwrap().to_str().unwrap()
    ));
    let mut cases = cases();
    cases[0].expected_ids = vec![0];
    assert!(
        export::fast(
            &source,
            &output,
            &expected(),
            &cases,
            Instant::now() + Duration::from_secs(5)
        )
        .is_err()
    );
    assert!(!output.exists());
    assert!(
        export::fast(
            &source,
            &output,
            &"a".repeat(64),
            &self::cases(),
            Instant::now() + Duration::from_secs(5)
        )
        .is_err()
    );
    assert!(
        export::fast(
            &source,
            &output,
            &expected(),
            &self::cases(),
            Instant::now()
        )
        .is_err()
    );
    std::fs::write(
        source.join("tokenizer_config.json"),
        br#"{"chat_template":"a","chat_template":"b"}"#,
    )
    .unwrap();
    assert!(
        export::fast(
            &source,
            &output,
            &expected(),
            &self::cases(),
            Instant::now() + Duration::from_secs(5)
        )
        .is_err()
    );
    assert!(!output.exists());
    temp.close().unwrap();
}
#[cfg(unix)]
#[test]
fn granite_full_snapshot_flatten_preserves_weight_bytes_omits_all_readme_and_refuses_collision() {
    let temp = tempfile::tempdir().unwrap();
    let root = temp.path().canonicalize().unwrap();
    let source = root.join("source");
    std::fs::create_dir(&source).unwrap();
    std::fs::create_dir(source.join("nested")).unwrap();
    for (name, bytes) in [
        ("weight.safetensors", b"tiny-real-fixture".as_slice()),
        ("nested/config.json", b"{}"),
        ("README.md", b"omit"),
        ("nested/README.md", b"omit too"),
    ] {
        std::fs::write(source.join(name), bytes).unwrap();
    }
    let rows = local::pins(&source, Instant::now() + Duration::from_secs(5)).unwrap();
    let pin = tree(&BTreeMap::from([
        ("weight.safetensors".into(), digest(b"tiny-real-fixture")),
        ("config.json".into(), digest(b"{}")),
    ]))
    .unwrap();
    let output = root.join("flat");
    export::granite(
        &source,
        &output,
        &rows,
        &pin,
        Instant::now() + Duration::from_secs(5),
    )
    .unwrap();
    assert_eq!(
        std::fs::read(output.join("weight.safetensors")).unwrap(),
        b"tiny-real-fixture"
    );
    assert!(!output.join("README.md").exists());
    std::fs::write(source.join("config.json"), b"collision").unwrap();
    let rows = local::pins(&source, Instant::now() + Duration::from_secs(5)).unwrap();
    assert!(
        export::granite(
            &source,
            &root.join("refused"),
            &rows,
            &pin,
            Instant::now() + Duration::from_secs(5)
        )
        .is_err()
    );
    assert!(!root.join("refused").exists());
    temp.close().unwrap();
}
#[cfg(unix)]
#[test]
fn actual_owned_custody_phase_retains_before_receipt_and_refuses_post_dataset_drift_or_false_success()
 {
    let temp = tempfile::tempdir().unwrap();
    let root = temp.path().canonicalize().unwrap();
    let output = root.join("owned");
    std::fs::create_dir(&output).unwrap();
    let key = "llama32-dense";
    for group in ["models", "vllm-configs", "tokenizers"] {
        std::fs::create_dir_all(output.join(group).join(key)).unwrap();
    }
    std::fs::create_dir(output.join("thoughtworks")).unwrap();
    std::fs::write(
        output.join("models").join(key).join("fixture.gguf"),
        b"tiny",
    )
    .unwrap();
    std::fs::write(
        output.join("vllm-configs").join(key).join("config.json"),
        b"{}",
    )
    .unwrap();
    std::fs::write(
        output.join("tokenizers").join(key).join("tokenizer.json"),
        b"{}",
    )
    .unwrap();
    std::fs::write(
        output.join("thoughtworks/sessions.parquet"),
        b"original finite bytes",
    )
    .unwrap();
    let tree = tree(&BTreeMap::from([("tokenizer.json".into(), digest(b"{}"))])).unwrap();
    let config = root.join("config.json");
    let value = json!({"models":[{"key":key,"filename":"fixture.gguf","sha256":digest(b"tiny"),"vllm_hf_config":{"repo":"owner/repo","revision":"a".repeat(40),"sha256":digest(b"{}")}}],"thoughtworks":{"dataset":{"filename":"sessions.parquet","sha256":digest(b"original finite bytes")}}});
    let config_bytes = serde_json::to_vec(&value).unwrap();
    std::fs::write(&config, &config_bytes).unwrap();
    let request = json!({"schema_version":1,"config":config,"config_sha256":digest(&config_bytes),"model_keys":[],"output_directory":output,"timeout_seconds":5,"maximum_bytes":1024,"credential_file":null,"export_sha256":{(key):tree},"semantic_cases":{(key):cases()}});
    let bytes = serde_json::to_vec(&request).unwrap();
    let input = root.join("request.json");
    std::fs::write(&input, &bytes).unwrap();
    let final_report = json!({"request_transport_sha256":digest(&bytes),"status":"ACQUIRED_EXPORTED","error":null,"families":[{"key":key}]});
    std::fs::write(
        output.join("acquisition.json"),
        serde_json::to_vec(&final_report).unwrap(),
    )
    .unwrap();
    std::fs::create_dir_all(output.join("sources").join(key)).unwrap();
    let mut acquired: Value =
        serde_json::from_slice(&std::fs::read(output.join("acquisition.json")).unwrap()).unwrap();
    acquired["families"][0]["source"] = json!({"files":{}});
    std::fs::write(
        output.join("acquisition.json"),
        serde_json::to_vec(&acquired).unwrap(),
    )
    .unwrap();
    verify::run(&input, "before-manifest").unwrap();
    assert!(output.join("custody-before-manifest.json").is_file());
    std::fs::write(
        output.join("thoughtworks/sessions.parquet"),
        b"changed finite bytes",
    )
    .unwrap();
    assert!(verify::run(&input, "after-manifest").is_err());
    assert!(!output.join("custody-after-manifest.json").exists());
    assert!(output.join("custody-before-manifest.json").is_file());
    std::fs::write(
        output.join("thoughtworks/sessions.parquet"),
        b"original finite bytes",
    )
    .unwrap();
    let mut contradictory = final_report;
    contradictory["error"] = json!("prior refusal");
    std::fs::write(
        output.join("acquisition.json"),
        serde_json::to_vec(&contradictory).unwrap(),
    )
    .unwrap();
    assert!(verify::run(&input, "after-manifest").is_err());
    temp.close().unwrap();
}
