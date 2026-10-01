use super::*;
use crate::automation::codepoint_json::{parser, strings::JsonString};
use crate::automation::replay_matrix::input;

pub(super) fn policy() -> Parameters {
    let raw = include_bytes!("../../../tests/fixtures/migration/optional_replay/valid.json");
    let matrix = parser::parse(raw).expect("valid fixture JSON");
    input::validate(matrix.get("replay").expect("fixture replay"))
        .unwrap_or_else(|_| panic!("valid replay policy"))
}

pub(super) fn matrix(models: &str) -> Value {
    parser::parse(format!(r#"{{"models":{models}}}"#).as_bytes()).expect("valid models")
}

const MODEL: &str = r#"{"family":"dense","repo":"org/model","revision":"pin","file":"one.gguf","sha256":"digest","class":"dense","native_context_tokens":131072}"#;

macro_rules! row_failure {
    ($name:ident, $rows:expr, $error:expr) => {
        #[test]
        fn $name() {
            let policy = policy();
            let models = matrix(&$rows);
            let result = select(models.get("models"), &policy, &"dense".into());
            assert_eq!(result.err(), Some($error));
        }
    };
}

row_failure!(
    missing_family_before_match_is_not_skipped,
    format!("[{{}},{MODEL}]"),
    SelectionError::RowMissingFamily { index: 0 }
);
row_failure!(
    null_row_after_match_is_not_skipped,
    format!("[{MODEL},null]"),
    SelectionError::RowType {
        index: 1,
        kind: "NoneType"
    }
);
row_failure!(
    malformed_tail_precedes_duplicate_cardinality,
    format!("[{MODEL},{MODEL},{{}}]"),
    SelectionError::RowMissingFamily { index: 2 }
);
row_failure!(
    first_row_failure_precedes_later_missing_family,
    format!("[7,{{}},{MODEL}]"),
    SelectionError::RowType {
        index: 0,
        kind: "int"
    }
);
row_failure!(
    null_models_is_an_iteration_error,
    "null",
    SelectionError::ModelsNotIterable("NoneType")
);
row_failure!(
    empty_object_models_has_no_matches,
    "{}",
    SelectionError::FamilyCardinality
);
row_failure!(
    object_models_iterates_string_keys,
    r#"{"dense":null}"#,
    SelectionError::RowType {
        index: 0,
        kind: "str"
    }
);
row_failure!(
    empty_string_models_has_no_matches,
    r#""""#,
    SelectionError::FamilyCardinality
);
row_failure!(
    string_models_iterates_characters,
    r#""dense""#,
    SelectionError::RowType {
        index: 0,
        kind: "str"
    }
);

macro_rules! context_case {
    ($name:ident, $minimum:expr, $native:expr, $expected:expr) => {
        #[test]
        fn $name() {
            let minimum = PositiveInteger::from_decimal($minimum).expect("positive minimum");
            let native = parser::parse($native.as_bytes()).expect("decoded context");
            let result = native_context_below(Some(&native), &minimum);
            assert_eq!(result, $expected);
        }
    };
}
context_case!(
    float_equality_meets_context,
    "131072",
    "131072.0",
    Ok(false)
);
context_case!(
    fraction_below_context_rejects,
    "131072",
    "131071.999",
    Ok(true)
);
context_case!(
    fraction_above_context_meets,
    "131072",
    "131072.75",
    Ok(false)
);
#[test]
fn nonfinite_context_is_rejected_at_json_boundary() {
    for token in ["NaN", "Infinity", "-Infinity"] {
        assert!(parser::parse(token.as_bytes()).is_err());
    }
}
context_case!(
    boolean_native_context_is_numeric,
    "131072",
    "true",
    Ok(true)
);
context_case!(negative_native_context_rejects, "131072", "-1.0", Ok(true));
context_case!(
    subnormal_native_context_rejects,
    "131072",
    "5e-324",
    Ok(true)
);
context_case!(
    native_float_does_not_round_integer_minimum,
    "9007199254740993",
    "9007199254740992.0",
    Ok(true)
);
context_case!(
    native_float_above_adjacent_integer_meets,
    "9007199254740993",
    "9007199254740994.0",
    Ok(false)
);
context_case!(
    float_at_two_to_128_does_not_round_minimum,
    "340282366920938463463374607431768211457",
    "340282366920938463463374607431768211456.0",
    Ok(true)
);
context_case!(
    bigint_context_is_exact,
    "131072",
    "340282366920938463463374607431768211456",
    Ok(false)
);
context_case!(
    string_context_retains_type_failure,
    "131072",
    r#""131072""#,
    Err(SelectionError::NativeContextType("str"))
);
context_case!(
    null_context_retains_type_failure,
    "131072",
    "null",
    Err(SelectionError::NativeContextType("NoneType"))
);

#[test]
fn selects_unique_family_even_when_other_rows_share_identity() {
    let policy = policy();
    let models = matrix(
        r#"[{"family":"else"},{"family":"else"},{"family":"dense","repo":"org/model","revision":"pin","file":"one.gguf","sha256":"hash","class":"dense","native_context_tokens":131072}]"#,
    );
    let selected = select(models.get("models"), &policy, &"dense".into()).expect("selected family");
    assert_eq!(
        selected.reference,
        ReplayString::from("org/model@pin/one.gguf")
    );
}

#[test]
fn container_unicode_key_matches_python_repr_in_model_argument() {
    let policy = policy();
    let models = matrix(
        r#"[{"family":"dense","repo":{"é\u200b":"漢"},"revision":"pin","file":"one.gguf","sha256":"digest","class":"dense","native_context_tokens":131072}]"#,
    );
    let model = select(models.get("models"), &policy, &"dense".into()).expect("selection");
    assert_eq!(
        model.reference,
        ReplayString::from(r#"{"\u00e9\u200b": "\u6f22"}@pin/one.gguf"#)
    );
}

#[test]
fn container_unicode_value_and_controls_match_python_repr_in_model_argument() {
    let policy = policy();
    let models = matrix(
        r#"[{"family":"dense","repo":["é","漢","\u200b","\u0001","\t","\n","\u0085","\u00a0"],"revision":"pin","file":"one.gguf","sha256":"digest","class":"dense","native_context_tokens":131072}]"#,
    );
    let model = select(models.get("models"), &policy, &"dense".into()).expect("selection");
    assert_eq!(
        model.reference,
        ReplayString::from(
            r#"["\u00e9", "\u6f22", "\u200b", "\u0001", "\t", "\n", "\u0085", "\u00a0"]@pin/one.gguf"#
        )
    );
}

#[cfg(unix)]
#[test]
fn unix_utf8_adapter_preserves_unicode_surrogateescape_and_os_bytes() {
    use std::os::unix::ffi::{OsStrExt, OsStringExt};
    let policy = policy();
    let models = matrix(
        r#"[{"family":"dense","repo":"caf\u00e9/model","revision":"pin","file":"\ud83d\ude00","sha256":"digest","class":"dense","native_context_tokens":131072}]"#,
    );
    let model =
        select(models.get("models"), &policy, &JsonString::from("dense")).expect("selection");
    let raw = OsString::from_vec(b"space '\t\n\xff".to_vec());
    let refs = [raw.clone(), raw.clone()];
    let invocation = ReplayInvocation {
        python: OsStr::new("python3"),
        script: Path::new("/checkout/evals/agentic-replay.py"),
        dataset: Path::new(&raw),
        output: Path::new(&raw),
        worktree_root: Some(&raw),
        refs: &refs,
    };
    let args = encoding::unix_utf8_argv(&argv(&model, &policy, &invocation))
        .expect("UTF-8 filesystem encoding");
    assert_eq!(
        args[4].as_bytes(),
        b"caf\xc3\xa9/model@pin/\xf0\x9f\x98\x80"
    );
    for index in [12, 14, 16, 18, 20] {
        assert_eq!(args[index].as_bytes(), b"space '\t\n\xff");
    }
}
