use super::tests::{matrix, policy};
use super::*;
use crate::automation::codepoint_json::parser;

const MODEL: &str = r#"{"family":"dense","repo":"org/model","revision":"pin","file":"one.gguf","sha256":"digest","class":"dense","native_context_tokens":131072}"#;

#[test]
fn present_nonstring_families_are_nonmatches() {
    let policy = policy();
    let models = matrix(&format!(
        r#"[{{"family":null}},{{"family":7}},{{"family":true}},{MODEL}]"#
    ));
    let result = select(models.get("models"), &policy, &"dense".into());
    assert!(result.is_ok());
}

#[test]
fn absent_models_retains_missing_key_classification() {
    let policy = policy();
    let result = select(None, &policy, &"dense".into());
    assert_eq!(result.err(), Some(SelectionError::MissingField("models")));
}

#[test]
fn finite_native_context_is_below_minimum_beyond_float_range() {
    let minimum = PositiveInteger::from_decimal(&format!("1{}", "0".repeat(400)))
        .expect("positive large minimum");
    let native = Value::Float(f64::MAX);
    let result = native_context_below(Some(&native), &minimum);
    assert_eq!(result, Ok(true));
}

#[test]
fn missing_native_context_defaults_to_zero() {
    let policy = policy();
    let result = native_context_below(None, &policy.minimum_context_tokens);
    assert_eq!(result, Ok(true));
}

#[test]
fn selection_rejects_float_when_validated_minimum_is_adjacent_large_integer() {
    let raw = include_str!("../../../tests/fixtures/migration/optional_replay/valid.json").replace(
        "\"minimum_context_tokens\": 131072",
        "\"minimum_context_tokens\": 9007199254740993",
    );
    let replay = parser::parse(raw.as_bytes()).expect("valid fixture");
    let policy = super::super::input::validate(replay.get("replay").expect("replay"))
        .unwrap_or_else(|_| panic!("large minimum is policy-valid"));
    let models = matrix(&format!(
        "[{}]",
        MODEL.replace("131072", "9007199254740992.0")
    ));
    let result = select(models.get("models"), &policy, &"dense".into());
    assert_eq!(result.err(), Some(SelectionError::NativeContext));
}

#[test]
fn escaped_duplicate_family_key_uses_last_decoded_value() {
    let policy = policy();
    let models = matrix(&format!(
        "[{}]",
        MODEL.replace(
            r#""family":"dense""#,
            r#""family":"other","fam\u0069ly":"dense""#
        )
    ));
    let result = select(models.get("models"), &policy, &"dense".into());
    assert!(result.is_ok());
}

#[test]
fn rejects_missing_and_duplicated_requested_family() {
    let policy = policy();
    let models = matrix(r#"[{"family":"dense"},{"family":"dense"}]"#);
    assert!(matches!(
        select(models.get("models"), &policy, &"missing".into()),
        Err(SelectionError::FamilyCardinality)
    ));
    assert!(matches!(
        select(models.get("models"), &policy, &"dense".into()),
        Err(SelectionError::FamilyCardinality)
    ));
}

#[test]
fn checks_context_after_cardinality_without_rejecting_ignored_duplicates() {
    let policy = policy();
    let models = matrix(
        r#"[{"family":"other"},{"family":"other"},{"family":"dense","native_context_tokens":131071}]"#,
    );
    assert!(matches!(
        select(models.get("models"), &policy, &"dense".into()),
        Err(SelectionError::NativeContext)
    ));
}

#[cfg(unix)]
#[test]
fn unencodable_surrogate_is_retained_until_encoding_boundary() {
    assert!(parser::parse(br#""\ud800""#).is_err());
}

#[cfg(unix)]
#[test]
fn embedded_nul_is_retained_until_encoding_boundary() {
    let args = [Argument::Text("a\0b".into())];
    let result = encoding::unix_utf8_argv(&args);
    assert_eq!(
        result,
        Err(encoding::EncodingError::EmbeddedNul { index: 0 })
    );
}
