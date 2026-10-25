use super::super::{verify_evidence, write_evidence};
use super::fixture::{Fixture, METRICS};
use std::fs;

#[test]
fn surrogate_in_identity_must_not_equal_a_replacement_character() {
    let mut fixture = Fixture::new();
    fixture.write.model_id = "\u{fffd}".into();
    let mut body = fixture.body();
    body.as_object_mut().unwrap().remove("model_id");
    let body = body.to_string();
    let body = format!(
        r#"{},"model_id":"\ud800"}}"#,
        body.strip_suffix('}').unwrap()
    );
    fs::write(&fixture.write.output, body).unwrap();

    let result = verify_evidence(&fixture.verify());

    assert!(result.is_err());
}

#[test]
fn surrogate_extension_is_preserved_when_writing_tts_metrics() {
    let mut fixture = Fixture::new();
    fixture.select("speech_synthesis", "llama-tts");
    fixture.tts_result(&format!(
        r#"{},"extension":"\ud800"}}"#,
        METRICS.strip_suffix('}').unwrap()
    ));

    assert!(write_evidence(&fixture.write).is_err());
}

#[test]
fn distinct_surrogate_and_replacement_keys_survive_in_nested_metrics() {
    let mut fixture = Fixture::new();
    fixture.select("speech_synthesis", "llama-tts");
    fixture.tts_result(&format!(
        r#"{},"extension":{{"\ud800":1,"\ufffd":2}}}}"#,
        METRICS.strip_suffix('}').unwrap()
    ));

    assert!(write_evidence(&fixture.write).is_err());
}

#[test]
fn valid_del_identity_uses_python_ascii_output_bytes() {
    let mut fixture = Fixture::new();
    fixture.write.model_id = "\u{7f}".into();
    let expected = include_str!("embedding-evidence.json")
        .replace(r#""model_id": "fixture""#, r#""model_id": "\u007f""#);

    write_evidence(&fixture.write).unwrap();

    assert_eq!(
        fs::read(&fixture.write.output).unwrap(),
        expected.as_bytes()
    );
}

#[test]
fn valid_replacement_and_supplementary_identities_still_match() {
    for (expected, encoded) in [
        ("\u{fffd}", r#""\ufffd""#),
        ("\u{10000}", r#""\ud800\udc00""#),
    ] {
        let mut fixture = Fixture::new();
        fixture.write.model_id = expected.into();
        let body = fixture.body().to_string().replace(
            r#""model_id":"fixture""#,
            &format!("\"model_id\":{encoded}"),
        );
        fs::write(&fixture.write.output, body).unwrap();

        let result = verify_evidence(&fixture.verify());

        assert!(result.is_ok(), "{encoded}: {result:?}");
    }
}

#[test]
fn nested_metric_keys_sort_by_codepoint_and_use_last_decoded_value() {
    let mut fixture = Fixture::new();
    fixture.select("speech_synthesis", "llama-tts");
    fixture.tts_result(&format!(
        r#"{},"extension":{{"\ud800\udc00":1,"a":2,"\u0061":3,"\ufffd":4,"\u007f":"\u007f","\\ud800":5,"\u0000exact-number":6}}}}"#,
        METRICS.strip_suffix('}').unwrap()
    ));

    write_evidence(&fixture.write).unwrap();

    let written = fs::read_to_string(&fixture.write.output).unwrap();
    assert!(written.contains(concat!(
        "\"extension\": {\n",
        "      \"\\u0000exact-number\": 6,\n",
        "      \"\\\\ud800\": 5,\n",
        "      \"a\": 3,\n",
        "      \"\\u007f\": \"\\u007f\",\n",
        "      \"\\ufffd\": 4,\n",
        "      \"\\ud800\\udc00\": 1\n    }"
    )));
}
