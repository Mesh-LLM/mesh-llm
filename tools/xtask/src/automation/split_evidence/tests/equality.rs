use crate::automation::{codepoint_json::parser, split_evidence::verify};

#[test]
fn integer_comparison_retains_exact_values_across_internal_representations() {
    use crate::automation::codepoint_json::value::Value;

    for integer in [0, 1, 9007199254740993_i128, i128::MAX] {
        let number = Value::Int(integer);
        let decimal = Value::BigInt(integer.to_string());
        assert!(verify::equal(&number, &decimal));
        assert!(verify::equal(&decimal, &number));
        let different = Value::BigInt((num_bigint::BigInt::from(integer) + 1_u8).to_string());
        assert!(!verify::equal(&number, &different));
    }
    let integer = Value::BigInt("340282366920938463463374607431768211456".into());
    let rounded = Value::Float(3.402823669209385e38);
    assert!(!verify::equal(&integer, &rounded));
    assert!(!verify::equal(&rounded, &integer));
}

#[test]
fn evidence_equality_preserves_scalar_types() {
    for (left, right, expected) in [
        ("true", "true", true),
        ("false", "false", true),
        ("true", "false", false),
        ("true", "1", false),
        ("false", "0", false),
        ("true", "1.0", false),
        ("false", "-0.0", false),
        ("true", "2", false),
        ("false", "-1", false),
        ("12", "12.0", false),
        ("-12", "-12.0", false),
        ("12", "12.5", false),
        ("0", "5e-324", false),
        ("0", "-0.0", false),
        ("0.0", "-0.0", true),
        ("1", "\"1\"", false),
    ] {
        let left = parser::parse(left.as_bytes()).unwrap();
        let right = parser::parse(right.as_bytes()).unwrap();

        let results = (verify::equal(&left, &right), verify::equal(&right, &left));

        assert_eq!(results, (expected, expected));
    }
}

#[test]
fn numeric_equality_when_binary_float_rounding_would_hide_integer_mismatch() {
    for (left, right, expected) in [
        ("9007199254740992", "9007199254740992.0", false),
        ("9007199254740993", "9007199254740992.0", false),
        ("9007199254740993", "9007199254740993.0", false),
        ("9007199254740994", "9007199254740994.0", false),
        ("-9007199254740993", "-9007199254740992.0", false),
    ] {
        let left = parser::parse(left.as_bytes()).unwrap();
        let right = parser::parse(right.as_bytes()).unwrap();

        let results = (verify::equal(&left, &right), verify::equal(&right, &left));

        assert_eq!(results, (expected, expected));
    }
}

#[test]
fn recursive_equality_when_keys_are_decoded_and_lists_remain_ordered() {
    for (left, right, expected) in [
        (
            r#"{"a":[true,{"b":12}],"z":null}"#,
            r#"{"z":null,"\u0061":[1.0,{"b":12.0}]}"#,
            false,
        ),
        (r#"{"a":[1,2]}"#, r#"{"a":[2,1]}"#, false),
        (r#"{"a":[1,{"b":2}]}"#, r#"{"a":[true,{"b":3}]}"#, false),
        (r#"{"a":1}"#, r#"{"a":1,"b":2}"#, false),
        (r#"{"a":1,"\u0061":2}"#, r#"{"a":2.0}"#, false),
        ("[]", "{}", false),
        ("null", "false", false),
    ] {
        let left = parser::parse(left.as_bytes()).unwrap();
        let right = parser::parse(right.as_bytes()).unwrap();

        let result = verify::equal(&left, &right);

        assert_eq!(result, expected);
    }
}

#[test]
fn verification_rejects_nonfinite_replacements_for_canonical_integers() {
    for nonfinite in ["NaN", "Infinity", "-Infinity", "1e400", "-1e400"] {
        assert!(parser::parse(format!("{{\"schema_version\":{nonfinite}}}").as_bytes()).is_err());
    }
}

#[test]
fn verification_rejects_boolean_and_float_replacements_for_evidence_integers() {
    let root = tempfile::tempdir().unwrap();
    let mut request = super::request(root.path());
    let path = root.path().join("split-evidence.json");
    let original = std::fs::read_to_string(super::fixture().join("expected-ready.json")).unwrap();
    request.mode = crate::automation::split_evidence::args::Mode::Verify(path.clone());
    for (integer, replacement) in [
        ("\"schema_version\": 1", "\"schema_version\": true"),
        ("\"schema_version\": 1", "\"schema_version\": 1.0"),
        ("\"stage_index\": 0", "\"stage_index\": false"),
        ("\"stage_index\": 1", "\"stage_index\": true"),
        ("\"layer_start\": 0", "\"layer_start\": -0.0"),
        ("\"layer_end\": 24", "\"layer_end\": 24.0"),
    ] {
        let modified = original.replace(integer, replacement);
        assert_ne!(modified, original, "fixture must exercise {integer}");
        std::fs::write(&path, &modified).unwrap();
        assert!(super::execute(&request).is_err(), "accepted {replacement}");
        assert_eq!(std::fs::read_to_string(&path).unwrap(), modified);
    }
    std::fs::write(&path, &original).unwrap();
    assert!(super::execute(&request).is_ok());
    assert_eq!(std::fs::read_to_string(path).unwrap(), original);
}

#[test]
fn irrelevant_nonfinite_extensions_remain_accepted_by_reconciliation() {
    let root = tempfile::tempdir().unwrap();
    let request = super::request(root.path());
    std::fs::write(
        &request.paths[2],
        b"{\"data\":[{\"id\":\"model-a\"}],\"extensions\":[NaN,Infinity,-Infinity,{\"huge\":1e400}]}",
    )
    .unwrap();

    let result = super::execute(&request);

    assert!(result.is_err());
}
