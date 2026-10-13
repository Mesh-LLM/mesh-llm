#![cfg(unix)]
use serde_json::{Value, json};
use sha2::Digest as _;
use std::{
    collections::BTreeMap,
    fs,
    path::Path,
    process::{Command, Stdio},
    time::{Duration, Instant},
};
fn hash(b: &[u8]) -> String {
    sha2::Sha256::digest(b)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}
fn tree(rows: BTreeMap<String, String>) -> String {
    let mut h = sha2::Sha256::new();
    for (name, pin) in rows {
        h.update((name.len() as u64).to_be_bytes());
        h.update(name.as_bytes());
        for bytes in pin.as_bytes().as_chunks::<2>().0 {
            h.update([u8::from_str_radix(std::str::from_utf8(bytes).unwrap(), 16).unwrap()]);
        }
    }
    h.finalize().iter().map(|b| format!("{b:02x}")).collect()
}
fn run(input: &Path, root: &Path, label: &str) -> bool {
    let stdout = root.join(format!("{label}.stdout"));
    let stderr = root.join(format!("{label}.stderr"));
    let mut child = Command::new(env!("CARGO_BIN_EXE_model-package-competitive-inputs"))
        .env_clear()
        .args(["export-local", "--input"])
        .arg(input)
        .stdout(Stdio::from(fs::File::create(&stdout).unwrap()))
        .stderr(Stdio::from(fs::File::create(&stderr).unwrap()))
        .spawn()
        .unwrap();
    let until = Instant::now() + Duration::from_secs(10);
    let (status, timeout) = loop {
        if let Some(s) = child.try_wait().unwrap() {
            break (s, false);
        }
        if Instant::now() >= until {
            child.kill().unwrap();
            break (child.wait().unwrap(), true);
        }
        std::thread::sleep(Duration::from_millis(5));
    };
    assert!(
        !timeout,
        "owned native helper exceeded local fixture budget"
    );
    assert!(fs::metadata(stdout).unwrap().len() < 65536);
    assert!(fs::metadata(stderr).unwrap().len() < 65536);
    status.success()
}
fn fixture(root: &Path) -> (Value, std::path::PathBuf) {
    let source = root.join("source");
    fs::create_dir(&source).unwrap();
    let original=serde_json::to_vec_pretty(&json!({"version":"1.0","truncation":null,"padding":null,"added_tokens":[],"normalizer":null,"pre_tokenizer":{"type":"Whitespace"},"post_processor":null,"decoder":null,"model":{"type":"WordLevel","vocab":{"[UNK]":0,"hello":1,"world":2},"unk_token":"[UNK]"}})).unwrap();
    let config = serde_json::to_vec(&json!({"chat_template":"{{ messages }}"})).unwrap();
    fs::write(source.join("tokenizer.json"), &original).unwrap();
    fs::write(source.join("tokenizer_config.json"), &config).unwrap();
    let source_pin = tree(BTreeMap::from([
        ("tokenizer.json".into(), hash(&original)),
        ("tokenizer_config.json".into(), hash(&config)),
    ]));
    let derived = tokenizers::Tokenizer::from_bytes(&original)
        .unwrap()
        .to_string(false)
        .unwrap();
    let config = serde_json::to_vec_pretty(
        &json!({"chat_template":"{{ messages }}","tokenizer_class":"PreTrainedTokenizerFast"}),
    )
    .unwrap();
    let expected = tree(BTreeMap::from([
        ("tokenizer.json".into(), hash(derived.as_bytes())),
        ("tokenizer_config.json".into(), hash(&config)),
        ("chat_template.jinja".into(), hash(b"{{ messages }}")),
    ]));
    (
        json!({"source_directory":source,"source_sha256":source_pin,"output_directory":root.join("output"),"export_sha256":expected,"timeout_seconds":5,"cases":[{"text":"hello world","add_special_tokens":false,"decode_ids":[1,2],"skip_special_tokens":false,"expected_ids":[1,2],"expected_decoded_sha256":hash(b"hello world")}]}),
        source,
    )
}
#[test]
fn actual_native_export_cli_known_tokens_lineage_and_wrong_reference_refusal() {
    let temp = tempfile::tempdir().unwrap();
    let root = temp.path().canonicalize().unwrap();
    let (mut input, source) = fixture(&root);
    let path = root.join("request.json");
    fs::write(&path, serde_json::to_vec(&input).unwrap()).unwrap();
    let before = fs::read(source.join("tokenizer.json")).unwrap();
    assert!(run(&path, &root, "good"));
    let report: Value =
        serde_json::from_slice(&fs::read(root.join("output/export.json")).unwrap()).unwrap();
    assert_eq!(report["status"], "EXPORTED_SINGLE");
    assert_eq!(report["acquisition_performed"], false);
    assert_eq!(report["export"]["cases"][0]["matched"], true);
    assert_eq!(report["request_sha256"], hash(&fs::read(&path).unwrap()));
    assert_eq!(fs::read(source.join("tokenizer.json")).unwrap(), before);
    input["output_directory"] = json!(root.join("wrong"));
    input["cases"][0]["expected_ids"] = json!([0]);
    fs::write(&path, serde_json::to_vec(&input).unwrap()).unwrap();
    assert!(!run(&path, &root, "bad"));
    assert!(!root.join("wrong").exists());
    temp.close().unwrap();
}
#[test]
fn actual_native_export_cli_nested_output_existing_output_and_writerless_fifo_refuse_without_source_mutation()
 {
    let temp = tempfile::tempdir().unwrap();
    let root = temp.path().canonicalize().unwrap();
    let (mut input, source) = fixture(&root);
    let path = root.join("request.json");
    let before = fs::read(source.join("tokenizer.json")).unwrap();
    input["output_directory"] = json!(source.join("nested"));
    fs::write(&path, serde_json::to_vec(&input).unwrap()).unwrap();
    assert!(!run(&path, &root, "overlap"));
    assert!(!source.join("nested").exists());
    let foreign = root.join("foreign");
    fs::create_dir(&foreign).unwrap();
    fs::write(foreign.join("sentinel"), b"preserve").unwrap();
    input["output_directory"] = json!(foreign);
    fs::write(&path, serde_json::to_vec(&input).unwrap()).unwrap();
    assert!(!run(&path, &root, "foreign"));
    assert_eq!(
        fs::read(root.join("foreign/sentinel")).unwrap(),
        b"preserve"
    );
    let fifo = root.join("writerless");
    let name = std::ffi::CString::new(fifo.as_os_str().as_encoded_bytes()).unwrap(); // SAFETY: owned valid C path; no file descriptor or foreign process is touched.
    assert_eq!(unsafe { libc::mkfifo(name.as_ptr(), 0o600) }, 0);
    assert!(!run(&fifo, &root, "fifo"));
    assert_eq!(fs::read(source.join("tokenizer.json")).unwrap(), before);
    temp.close().unwrap();
}
