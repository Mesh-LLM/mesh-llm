use super::{admit, exact_decimal};

#[test]
fn exact_sampling_uses_decimal_value_without_float_rounding() {
    for token in ["0", "0.0", "-0.0", "0e10", "0e-9999"] {
        assert!(exact_decimal(token, "0"), "{token}");
    }
    for token in ["42", "42.0", "4.2e1", "420e-1", "0.42e2"] {
        assert!(exact_decimal(token, "42"), "{token}");
    }
    for token in ["false", "true", "\"0\"", "1e-9999", "-1e-9999"] {
        assert!(!exact_decimal(token, "0"), "{token}");
    }
    for token in [
        "42.000000000000000000001",
        "41.999999999999999999999",
        "-42",
        "42e999999999999999999999",
    ] {
        assert!(!exact_decimal(token, "42"), "{token}");
    }
}

#[test]
fn sampling_path_decodes_keys_and_uses_the_last_duplicate_field() {
    for raw in [
        r#"{"ignored":{"temperature":1e-9999,"seed":false},"replay":{"temperature":0,"seed":42}}"#,
        r#"{"replay":{"temperature":false,"\u0074emperature":0,"seed":42}}"#,
        r#"{"replay":{"temperature":false,"seed":false},"\u0072eplay":{"temperature":0,"seed":42}}"#,
    ] {
        assert!(admit(raw.as_bytes()).unwrap(), "{raw}");
    }
    for raw in [
        r#"{"replay":{"temperature":0,"\u0074emperature":false,"seed":42}}"#,
        r#"{"replay":{"temperature":0,"seed":42,"\u0073eed":42.000000000000000000001}}"#,
        r#"{"replay":{"temperature":0,"seed":42},"\u0072eplay":{"temperature":1e-9999,"seed":42}}"#,
    ] {
        assert!(!admit(raw.as_bytes()).unwrap(), "{raw}");
    }
}
