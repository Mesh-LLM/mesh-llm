use super::super::{validate_tts_metrics, verify_evidence, write_evidence};
use super::fixture::{Fixture, METRICS};
use std::fs;

#[test]
fn verifier_uses_last_duplicate_identity_value() {
    for (suffix, accepted) in [("pass", true), ("fail", false)] {
        let fixture = Fixture::new();
        let body = fixture.body().to_string();
        let body = format!(
            "{},\"status\":\"{suffix}\"}}",
            body.strip_suffix('}').unwrap()
        );
        fs::write(&fixture.write.output, body).unwrap();

        let result = verify_evidence(&fixture.verify());

        assert_eq!(result.is_ok(), accepted, "{suffix}: {result:?}");
    }
}

#[test]
fn tts_uses_last_duplicate_metric_value() {
    for (suffix, accepted) in [("1", true), ("true", false), ("0", false)] {
        let body = format!(
            "{},\"channels\":{suffix}}}",
            METRICS.strip_suffix('}').unwrap()
        );

        let result = validate_tts_metrics(body.as_bytes());

        assert_eq!(result.is_ok(), accepted, "{suffix}: {result:?}");
    }
}

#[test]
fn verifier_ignores_nonfinite_extensions_but_not_typed_identity_fields() {
    for (field, accepted) in [("extension", false), ("model_sha256", false)] {
        let fixture = Fixture::new();
        let body = fixture.body().to_string();
        let body = format!("{},\"{field}\":NaN}}", body.strip_suffix('}').unwrap());
        fs::write(&fixture.write.output, body).unwrap();

        let result = verify_evidence(&fixture.verify());

        assert_eq!(result.is_ok(), accepted, "{field}: {result:?}");
    }
}

#[test]
fn writer_matches_source_derived_sorted_ascii_evidence_bytes() {
    let fixture = Fixture::new();

    write_evidence(&fixture.write).unwrap();

    assert_eq!(
        fs::read(&fixture.write.output).unwrap(),
        include_bytes!("embedding-evidence.json")
    );
}

#[test]
fn verifier_accepts_an_empty_comparison_payload_after_exact_prefix() {
    let fixture = Fixture::new();
    let mut body = fixture.body();
    body["comparison"] = "embedding local-monolithic oracle passed: ".into();
    fixture.save(&body);

    assert!(verify_evidence(&fixture.verify()).is_ok());
}

#[test]
fn writer_hashes_supplied_projector_for_optional_classes() {
    let mut fixture = Fixture::new();
    fixture.projector();
    let expected = fixture.body();

    write_evidence(&fixture.write).unwrap();

    let actual: serde_json::Value =
        serde_json::from_slice(&fs::read(&fixture.write.output).unwrap()).unwrap();
    assert_eq!(actual, expected);
}

#[test]
fn writer_rejects_invalid_utf8_comparison_log() {
    let fixture = Fixture::new();
    fs::write(
        &fixture.write.comparison_log,
        b"\xff\nembedding local-monolithic oracle passed: fixture",
    )
    .unwrap();

    let result = write_evidence(&fixture.write);

    assert!(result.is_err());
    assert!(!fixture.write.output.exists());
}
