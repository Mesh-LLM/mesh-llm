use super::super::{Error, validate_tts_metrics, write_evidence};
use super::fixture::{Fixture, METRICS};
use std::fs;

#[test]
fn excessive_json_nesting_rejects_without_overwriting_evidence() {
    let mut fixture = Fixture::new();
    fixture.select("speech_synthesis", "llama-tts");
    let nested = format!("{}0{}", "[".repeat(500), "]".repeat(500));
    fixture.tts_result(&format!(
        "{},\"extension\":{nested}}}",
        METRICS.strip_suffix('}').unwrap()
    ));
    fs::write(&fixture.write.output, b"RETAINED").unwrap();
    let result = write_evidence(&fixture.write);
    assert!(matches!(result, Err(Error::Json(_))));
    assert_eq!(fs::read(&fixture.write.output).unwrap(), b"RETAINED");
}

#[test]
fn oversized_pcm_integer_is_rejected() {
    let raw = METRICS.replace("\"channels\":1", "\"channels\":18446744073709551616");
    assert!(validate_tts_metrics(raw.as_bytes()).is_err());
}
