use super::*;
fn family() -> Family {
    "dense".to_owned().try_into().unwrap()
}
#[test]
fn candidate_rows_do_not_override_environment_failure_or_malformed_streams() {
    for (bytes, expected) in [
        (br#"{"family":"dense","exit_code":1}"#.as_slice(), FailureClass::Candidate),
        (br#"{"family":"battery","outcomes":[{"name":"environment-preflight","status":"fail","exit_code":0}]} {"family":"dense"}"#, FailureClass::Infrastructure),
        (br#"{"family":"battery","outcomes":[{"name":"environment-preflight","status":"pass","exit_code":false}]} {"family":"dense"}"#, FailureClass::Infrastructure),
        (br#"{"family":"battery","outcomes":[]}"#, FailureClass::Infrastructure),
        (br#"{"family":"dense"} {"#, FailureClass::Contract),
        (br#"[]"#, FailureClass::Contract),
        (br#"{"family":"battery","outcomes":[{"name":"environment-preflight","status":"pass","exit_code":0.0}]} {"family":"dense"}"#, FailureClass::Infrastructure),
        (br#"{"family":"battery","outcomes":false}"#, FailureClass::Contract),
    ] {
        assert_eq!(classify_results(bytes, &family()), expected);
    }
}
#[test]
fn classification_admits_json_whitespace_and_refuses_control_separators() {
    for separator in [" ", "\t", "\n", "\r\n"] {
        let stream = format!("{{\"family\":\"dense\"}}{separator}{{}}");
        assert_eq!(
            classify_results(stream.as_bytes(), &family()),
            FailureClass::Candidate
        );
    }
    for separator in ['\x1c', '\x1d', '\x1e', '\x0b', '\x0c'] {
        let stream = format!("{{\"family\":\"dense\"}}{separator}{{}}");
        assert_eq!(
            classify_results(stream.as_bytes(), &family()),
            FailureClass::Contract
        );
    }
}
