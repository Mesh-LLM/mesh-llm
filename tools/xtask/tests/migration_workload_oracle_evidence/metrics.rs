use super::super::{validate_tts_metrics, verify_evidence, write_evidence};
use super::fixture::{Fixture, METRICS};
use std::fs;

pub(super) fn metric_cases() -> Vec<(String, bool)> {
    let mut cases = vec![(METRICS.to_owned(), true)];
    cases.extend(["null", "{}", "[]", "true", "1", "\"metrics\""].map(|body| (body.into(), false)));
    let baseline: serde_json::Value = serde_json::from_str(METRICS).unwrap();
    for field in [
        "sample_rate_hz",
        "channels",
        "sample_count",
        "relative_rms_error",
        "waveform_cosine",
    ] {
        let mut missing = baseline.clone();
        missing.as_object_mut().unwrap().remove(field);
        cases.push((missing.to_string(), false));
    }
    for field in ["sample_rate_hz", "channels", "sample_count"] {
        for value in [
            "0", "-1", "true", "false", "1.0", "1.5", "1e0", "\"1\"", "null", "[]", "{}",
        ] {
            cases.push((replace(field, value), false));
        }
        cases.push((
            replace(
                field,
                "18446744073709551616000000000000000000000000000000000000000000000000000",
            ),
            false,
        ));
    }
    for (field, values) in [
        (
            "relative_rms_error",
            vec![
                "-0.001",
                "0.020001",
                "0.020000000000000004",
                "NaN",
                "Infinity",
                "-Infinity",
                "true",
                "\"0\"",
                "1e999",
                "10000000000000000000000000000000000000000000000000000000000000000000000",
            ],
        ),
        (
            "waveform_cosine",
            vec![
                "0.99949",
                "0.999499",
                "1.0001",
                "1.0000000000000002",
                "NaN",
                "Infinity",
                "-Infinity",
                "false",
                "\"1\"",
                "null",
            ],
        ),
    ] {
        cases.extend(
            values
                .into_iter()
                .map(|value| (replace(field, value), false)),
        );
    }
    cases.push((replace("relative_rms_error", "0"), true));
    cases.push((replace("relative_rms_error", "-0.0"), true));
    cases.push((replace("waveform_cosine", "1"), true));
    cases.push((replace("waveform_cosine", "1.0"), true));
    cases
}

fn replace(field: &str, value: &str) -> String {
    let mut body: serde_json::Value = serde_json::from_str(METRICS).unwrap();
    body.as_object_mut().unwrap().remove(field);
    let encoded = body.to_string();
    format!(
        "{},\"{field}\":{value}}}",
        encoded.strip_suffix('}').unwrap()
    )
}

#[test]
fn pcm_acceptance_when_metrics_are_at_or_outside_source_bounds() {
    for (body, accepted) in metric_cases() {
        let result = validate_tts_metrics(body.as_bytes());

        assert_eq!(result.is_ok(), accepted, "{body}: {result:?}");
    }
}

#[test]
fn writer_and_verifier_agree_when_metrics_are_missing_invalid_or_passing() {
    for (metrics, accepted) in metric_cases() {
        let mut fixture = Fixture::new();
        fixture.select("speech_synthesis", "llama-tts");
        fixture.projector();
        fixture.tts_result(&metrics);
        let body = fixture.body().to_string();
        let evidence = format!(
            "{},\"metrics\":{metrics}}}",
            body.strip_suffix('}').unwrap()
        );

        let written = write_evidence(&fixture.write);
        fs::write(&fixture.write.output, evidence).unwrap();
        let verified = verify_evidence(&fixture.verify());

        assert_eq!(written.is_ok(), accepted, "writer {metrics}: {written:?}");
        assert_eq!(
            verified.is_ok(),
            accepted,
            "verifier {metrics}: {verified:?}"
        );
    }
}

#[test]
fn extension_metrics_survive_when_not_used_for_pcm_acceptance() {
    let fixture = Fixture::new();
    let mut request = fixture.write.clone();
    request.model_class = "speech_synthesis".into();
    fs::write(
        &request.comparison_log,
        "speech_synthesis local-monolithic oracle passed: PCM\n",
    )
    .unwrap();
    let metrics = format!(
        "{},\"extension\":{{\"finite\":0.5,\"large\":18446744073709551615}}}}",
        METRICS.strip_suffix('}').unwrap()
    );
    fixture.tts_result(&metrics);

    write_evidence(&request).unwrap();

    let evidence = fs::read_to_string(&request.output).unwrap();
    assert!(evidence.contains("\"finite\": 0.5"));
    assert!(evidence.contains("\"large\": 18446744073709551615"));
}
