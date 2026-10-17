//! OCR's smoke input must be the independently reviewed transcription fixture.
use super::{Fixture, fs, root};

#[test]
fn ocr_smoke_selects_the_manifest_reviewed_text_fixture() {
    let root = root();
    let source =
        fs::read_to_string(root.join("skippy/scripts/skippy-workload-certify.sh")).unwrap();
    let selector = source
        .split_once("MEDIA_PATH=\"\"\n")
        .unwrap()
        .1
        .split_once("\nTEST_FILTER=")
        .unwrap()
        .0;
    let fixture = Fixture::new();
    let mut command = fixture.command();
    command
        .args(["-c", &format!("{selector}\nprintf '%s' \"$MEDIA_PATH\"")])
        .env("ROOT", &root)
        .env("MODEL_CLASS", "ocr");
    let receipt = fixture.run(command);
    assert!(receipt.status.success(), "{}", receipt.stderr);
    let manifest: serde_json::Value = serde_json::from_slice(
        &fs::read(root.join("ci/llama-canary/family-certified.json")).unwrap(),
    )
    .unwrap();
    let row = manifest["models"]
        .as_array()
        .unwrap()
        .iter()
        .find(|row| row["class"] == "ocr")
        .unwrap();
    let reviewed = root.join(row["evidence"]["fixture"].as_str().unwrap());
    assert!(reviewed.is_file());
    assert_eq!(receipt.stdout, reviewed.to_str().unwrap());
}
