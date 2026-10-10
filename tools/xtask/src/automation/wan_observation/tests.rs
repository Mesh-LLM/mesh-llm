use super::*;
#[test]
fn rtt_reduction_preserves_half_rtt_and_three_decimal_units() {
    for (input, expected) in [
        ("12.3456", "6.173\n"),
        ("0", "0.000\n"),
        (" 2.5 ", "1.250\n"),
    ] {
        assert_eq!(delay(input).unwrap(), expected);
    }
    for input in ["NaN", "inf", "-1", "text", "1e400"] {
        assert!(delay(input).is_err());
    }
    assert!(delay(&"1".repeat(129)).is_err());
}
#[test]
fn sender_precedence_fallback_and_optional_measurement_are_explicit() {
    for (json, expected) in [
        (
            r#"{"end":{"sum_sent":{"bits_per_second":2500000},"sum":{"bits_per_second":9000000}}}"#,
            "2\n",
        ),
        (
            r#"{"end":{"sum_sent":{},"sum":{"bits_per_second":1500000}}}"#,
            "2\n",
        ),
        (r#"{"end":{"sum":{"bits_per_second":1}}}"#, "1\n"),
        (
            r#"{"end":{"sum_sent":{"bits_per_second":0},"sum":{"bits_per_second":9000000}}}"#,
            "",
        ),
        (
            r#"{"end":{"sum_sent":{"bytes":1},"sum":{"bits_per_second":9000000}}}"#,
            "",
        ),
        (
            r#"{"end":{"sum_sent":{"bits_per_second":null},"sum":{"bits_per_second":9000000}}}"#,
            "",
        ),
        (r#"{"end":null}"#, ""),
        ("{}", ""),
    ] {
        assert_eq!(bandwidth(json.as_bytes()).unwrap(), expected);
    }
}
#[test]
fn malformed_nonfinite_and_oversize_bandwidth_never_become_measurements() {
    for json in [
        "{",
        "[]",
        r#"{"end":{"sum_sent":false}}"#,
        r#"{"end":{"sum":{"bits_per_second":"secret"}}}"#,
        r#"{"end":{"sum":{"bits_per_second":-1}}}"#,
        r#"{"end":{"sum":{"bits_per_second":1e400}}}"#,
        r#"{"end":{"sum":{"bits_per_second":1e100}}}"#,
    ] {
        assert!(bandwidth(json.as_bytes()).is_err());
    }
    assert!(bandwidth(&vec![b' '; JSON_LIMIT + 1]).is_err());
}
