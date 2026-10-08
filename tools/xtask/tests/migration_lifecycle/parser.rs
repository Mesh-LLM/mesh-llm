use super::matches;
use crate::process::{LineEnding, ObservedLine, Stream};

fn classify(bytes: &[u8], ending: LineEnding) -> bool {
    matches(ObservedLine {
        stream: Stream::Stdout,
        bytes,
        ending,
    })
}

macro_rules! predicate_case {
    ($name:ident, $input:expr, $expected:expr) => {
        #[test]
        fn $name() {
            let input = $input;

            let matched = classify(input, LineEnding::Lf);

            assert_eq!(matched, $expected);
        }
    };
}

#[path = "stringification.rs"]
mod stringification;

#[path = "approved_readiness.rs"]
mod approved_readiness;

predicate_case!(
    migration_lifecycle_reordered_fields,
    br#"{"extra":42,"role":"client","event":"passive_mode","status":"ready"}"#,
    true
);
predicate_case!(
    migration_lifecycle_mixed_case_message,
    br#"{"message":"cLiEnT ReAdY"}"#,
    true
);
predicate_case!(
    migration_lifecycle_array_message,
    br#"{"message":["Client ready"]}"#,
    false
);
predicate_case!(
    migration_lifecycle_object_key_message,
    br#"{"message":{"Client ready":false}}"#,
    false
);
predicate_case!(
    migration_lifecycle_nested_message,
    br#"{"message":{"note":["Client ready"]}}"#,
    false
);
predicate_case!(
    migration_lifecycle_split_strings,
    br#"{"message":["Client","ready"]}"#,
    false
);
predicate_case!(
    migration_lifecycle_split_key_value,
    br#"{"message":{"client":"ready"}}"#,
    false
);
predicate_case!(
    migration_lifecycle_wrong_type_event,
    br#"{"event":7,"message":"Client ready"}"#,
    true
);
predicate_case!(
    migration_lifecycle_wrong_type_message,
    br#"{"message":false,"role":"client","status":"ready","event":"passive_mode"}"#,
    true
);
predicate_case!(migration_lifecycle_plain_text, b"Client ready", false);
predicate_case!(
    migration_lifecycle_top_level_string,
    br#""Client ready""#,
    false
);
predicate_case!(
    migration_lifecycle_top_level_array,
    br#"[{"message":"Client ready"}]"#,
    false
);
predicate_case!(
    migration_lifecycle_detail_only,
    br#"{"detail":"Client ready"}"#,
    false
);
predicate_case!(
    migration_lifecycle_structured_case,
    br#"{"event":"PASSIVE_MODE","status":"ready","role":"client"}"#,
    false
);
predicate_case!(
    migration_lifecycle_wrong_role,
    br#"{"event":"passive_mode","status":"ready","role":"server"}"#,
    false
);
predicate_case!(
    migration_lifecycle_wrong_status,
    br#"{"event":"passive_mode","status":"starting","role":"client"}"#,
    false
);
predicate_case!(
    migration_lifecycle_wrong_event,
    br#"{"event":"other","status":"ready","role":"client"}"#,
    false
);
predicate_case!(
    migration_lifecycle_duplicate_last_negative,
    br#"{"message":"Client ready","message":"no"}"#,
    false
);
predicate_case!(
    migration_lifecycle_duplicate_last_positive,
    br#"{"message":"no","message":"Client ready"}"#,
    true
);
predicate_case!(
    migration_lifecycle_nested_duplicate_last_negative,
    br#"{"message":{"note":"Client ready","note":"no"}}"#,
    false
);
predicate_case!(
    migration_lifecycle_structured_duplicate_last_negative,
    br#"{"event":"passive_mode","status":"ready","role":"client","role":null}"#,
    false
);
predicate_case!(
    migration_lifecycle_escaped_phrase,
    br#"{"message":"Clie\u006et\u0020ready"}"#,
    true
);
predicate_case!(
    migration_lifecycle_lossy_utf8,
    b"{\"other\":\"\xff\",\"message\":\"Client ready\"}",
    true
);
predicate_case!(
    migration_lifecycle_secret_key,
    br#"{"event":"passive_mode","status":"ready","role":"client","tokens":0}"#,
    true
);
predicate_case!(
    migration_lifecycle_malformed_before_redaction,
    br#"{"message":"Client ready a"b"}"#,
    false
);
predicate_case!(
    migration_lifecycle_null_message,
    br#"{"message":null}"#,
    false
);
predicate_case!(
    migration_lifecycle_boolean_message,
    br#"{"message":true}"#,
    false
);
predicate_case!(
    migration_lifecycle_numeric_message,
    br#"{"message":123}"#,
    false
);
predicate_case!(
    migration_lifecycle_nan_grammar_blocker,
    br#"{"message":"Client ready","other":NaN}"#,
    false
);
predicate_case!(
    migration_lifecycle_infinity_grammar_blocker,
    br#"{"message":"Client ready","other":Infinity}"#,
    false
);
predicate_case!(
    migration_lifecycle_overflow_grammar_blocker,
    br#"{"message":"Client ready","other":1e9999}"#,
    false
);
predicate_case!(
    migration_lifecycle_surrogate_grammar_blocker,
    br#"{"message":"Client ready","other":"\ud800"}"#,
    false
);
predicate_case!(
    migration_lifecycle_trailing_json,
    br#"{"message":"Client ready"}{}"#,
    false
);
predicate_case!(
    migration_lifecycle_crlf,
    b"{\"message\":\"Client ready\"}\r",
    true
);

#[test]
fn migration_lifecycle_eof_is_not_a_record() {
    let input = br#"{"message":"Client ready"}"#;

    let matched = classify(input, LineEnding::Eof);

    assert!(!matched);
}

#[test]
fn migration_lifecycle_depth_at_limit_is_supported() {
    let input = format!(
        "{{\"message\":\"Client ready\",\"other\":{}0{}}}",
        "[".repeat(63),
        "]".repeat(63)
    );

    let matched = classify(input.as_bytes(), LineEnding::Lf);

    assert!(matched);
}

#[test]
fn migration_lifecycle_depth_above_limit_is_a_parity_blocker() {
    let input = format!(
        "{{\"message\":\"Client ready\",\"other\":{}0{}}}",
        "[".repeat(64),
        "]".repeat(64)
    );

    let matched = classify(input.as_bytes(), LineEnding::Lf);

    assert!(!matched);
}

#[test]
fn migration_lifecycle_braces_inside_strings_do_not_count_as_depth() {
    let input = format!(
        "{{\"message\":\"Client ready\",\"other\":\"{}\\\"{}\"}}",
        "[".repeat(80),
        "{".repeat(80)
    );

    let matched = classify(input.as_bytes(), LineEnding::Lf);

    assert!(matched);
}
