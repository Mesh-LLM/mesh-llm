use super::*;

fn report(body: &str, pass: Pass) -> Result<Summary, Error> {
    let root = tempfile::tempdir().unwrap();
    let path = root.path().join("report.json");
    std::fs::write(&path, body).unwrap();
    load(&path, pass)
}

#[test]
fn inherited_wins_when_first_report_also_contains_refusals() {
    let body = r#"{"builders":[{"file":"refused","verdict":"error"},{"file":"inherited","verdict":"already_transformed"}]}"#;
    let result = report(body, Pass::First);
    assert!(matches!(result, Err(Error::Inherited(files)) if files == "inherited"));
}

#[test]
fn second_accepts_unsupported_and_error_when_no_transformable_record_remains() {
    let body = r#"{"builders":[{"file":"a","verdict":"unsupported_shape"},{"file":"b","verdict":"error"}]}"#;
    let result = report(body, Pass::Second);
    assert_eq!(result.unwrap(), Summary::new());
}

#[test]
fn second_rejects_transformable_when_other_records_are_allowed() {
    let body = r#"{"builders":[{"file":"a","verdict":"unsupported_shape"},{"file":"b","verdict":"transformable"}]}"#;
    let result = report(body, Pass::Second);
    assert!(matches!(result, Err(Error::Remaining(files)) if files == "b"));
}

#[test]
fn first_reports_reason_when_decoder_is_refused() {
    let body = r#"{"builders":[{"file":"a","verdict":"unsupported_shape","unsupported_reason":"shape"},{"file":"b","verdict":"error"}]}"#;
    let result = report(body, Pass::First);
    assert!(matches!(result, Err(Error::Refused(files)) if files == "a: shape, b: error"));
}

#[test]
fn both_require_records_when_builders_are_empty() {
    for pass in [Pass::First, Pass::Second] {
        let result = report(r#"{"builders":[]}"#, pass);
        assert!(matches!(result, Err(Error::Empty)));
    }
}

#[test]
fn consumed_types_and_duplicates_reject_when_producer_is_malformed() {
    for body in [
        r#"{"builders":[null]}"#,
        r#"{"builders":[],"builders":[]}"#,
        r#"{"builders":[{"file":12}]}"#,
        r#"{"builders":[{"verdict":"future"}]}"#,
        r#"{"builders":[{}],"summary":{"error":true}}"#,
        r#"{"builders":[{}],"summary":{"error":9223372036854775808}}"#,
    ] {
        let result = report(body, Pass::Second);
        assert!(matches!(result, Err(Error::Contract(_))), "{body}");
    }
}

#[test]
fn counters_remain_exact_when_larger_than_float_integer_precision() {
    let body = r#"{"builders":[{"verdict":"supported_auxiliary"}],"summary":{"supported_auxiliary":9007199254740993,"error":0}}"#;
    let result = report(body, Pass::First);
    assert_eq!(
        result.unwrap(),
        Summary::from([("error", 0), ("supported_auxiliary", 9007199254740993)])
    );
}
