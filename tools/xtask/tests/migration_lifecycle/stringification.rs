use super::classify;
use crate::process::LineEnding;

predicate_case!(
    migration_lifecycle_repr_array_form_feed_suffix,
    br#"{"message":["\fLIENT ready"]}"#,
    false
);
predicate_case!(
    migration_lifecycle_repr_object_key_control_suffix,
    br#"{"message":{"\u001cLIENT ready":false}}"#,
    false
);
predicate_case!(
    migration_lifecycle_repr_top_level_string_stays_unquoted,
    br#"{"message":"\fLIENT ready"}"#,
    false
);
predicate_case!(
    migration_lifecycle_repr_nested_value_suffix,
    br#"{"message":{"note":[{"value":"\u008cLIENT ready"}]}}"#,
    false
);
predicate_case!(
    migration_lifecycle_repr_format_control_suffix,
    br#"{"message":["\u200cLIENT ready"]}"#,
    false
);
predicate_case!(
    migration_lifecycle_repr_private_use_suffix,
    br#"{"message":["\ue00cLIENT ready"]}"#,
    false
);
predicate_case!(
    migration_lifecycle_repr_unassigned_suffix,
    br#"{"message":["\u2fdcLIENT ready"]}"#,
    false
);
predicate_case!(
    migration_lifecycle_repr_astral_private_use_suffix,
    br#"{"message":["\udb80\udc0cLIENT ready"]}"#,
    false
);
predicate_case!(
    migration_lifecycle_repr_noncharacter_suffix,
    br#"{"message":["\ufddcLIENT ready"]}"#,
    false
);
predicate_case!(
    migration_lifecycle_repr_printable_hex_suffix_is_not_an_escape,
    br#"{"message":["\u00acLIENT ready"]}"#,
    false
);
predicate_case!(
    migration_lifecycle_repr_top_level_unicode_stays_unquoted,
    br#"{"message":"\u200cLIENT ready"}"#,
    false
);
predicate_case!(
    migration_lifecycle_repr_backslash_literal_is_not_decoded_twice,
    br#"{"message":["\\u200cLIENT ready"]}"#,
    false
);
predicate_case!(
    migration_lifecycle_repr_escape_then_backslash_blocks_match,
    br#"{"message":["\f\\LIENT ready"]}"#,
    false
);
predicate_case!(
    migration_lifecycle_repr_single_quote_blocks_escape_suffix,
    br#"{"message":["\f'LIENT ready"]}"#,
    false
);
predicate_case!(
    migration_lifecycle_repr_double_quote_blocks_escape_suffix,
    br#"{"message":["\f\"LIENT ready"]}"#,
    false
);
predicate_case!(
    migration_lifecycle_repr_both_quotes_before_suffix,
    br#"{"message":["'\"\fLIENT ready"]}"#,
    false
);
predicate_case!(
    migration_lifecycle_repr_literal_escape_does_not_make_space,
    br#"{"message":["client\\x20ready"]}"#,
    false
);
predicate_case!(
    migration_lifecycle_repr_control_does_not_make_space,
    br#"{"message":["client\tready"]}"#,
    false
);
predicate_case!(
    migration_lifecycle_repr_separator_does_not_make_ascii_space,
    br#"{"message":["client\u2028ready"]}"#,
    false
);
predicate_case!(
    migration_lifecycle_repr_c_inside_escape_is_not_its_suffix,
    br#"{"message":["\ue0c0LIENT ready"]}"#,
    false
);
predicate_case!(
    migration_lifecycle_repr_control_escape_cannot_cross_elements,
    br#"{"message":["\f","LIENT ready"]}"#,
    false
);
predicate_case!(
    migration_lifecycle_repr_control_escape_cannot_cross_key_value,
    br#"{"message":{"\u001c":"LIENT ready"}}"#,
    false
);
predicate_case!(
    migration_lifecycle_repr_other_fields_are_not_message,
    br#"{"detail":["\fLIENT ready"]}"#,
    false
);
predicate_case!(
    migration_lifecycle_repr_structured_arm_survives_negative_message,
    br#"{"message":["\f\\LIENT ready"],"event":"passive_mode","status":"ready","role":"client"}"#,
    true
);
predicate_case!(
    migration_lifecycle_repr_wrong_structured_type_and_nonstring_message_reject,
    br#"{"message":["\fLIENT ready"],"event":false}"#,
    false
);
predicate_case!(
    migration_lifecycle_repr_scalars_do_not_contribute_phrase,
    br#"{"message":[null,true,false,0,-42,0.5,1e30,{},[]]}"#,
    false
);
predicate_case!(
    migration_lifecycle_repr_lowercase_is_not_casefold,
    br#"{"message":["\fL\u0130ENT ready"]}"#,
    false
);

#[test]
fn migration_lifecycle_repr_printable_unicode_is_not_an_escape_suffix() {
    for character in [
        '\u{00ec}',
        '\u{00ac}',
        '\u{030c}',
        '\u{180c}',
        '\u{201c}',
        '\u{215c}',
        '\u{fe0c}',
        '\u{1f60c}',
        '\u{e010c}',
    ] {
        let input = format!("{{\"message\":[\"{character}LIENT ready\"]}}");

        let matched = classify(input.as_bytes(), LineEnding::Lf);

        assert!(!matched, "printable U+{:04X}", u32::from(character));
    }
}

#[test]
fn migration_lifecycle_repr_controls_never_supply_a_message_match() {
    for codepoint in (0_u8..=31).chain(127..=159) {
        let input = format!("{{\"message\":[\"\\u{codepoint:04x}LIENT ready\"]}}");
        let matched = classify(input.as_bytes(), LineEnding::Lf);

        assert!(!matched, "control U+{codepoint:04X}");
    }
}

#[test]
fn migration_lifecycle_repr_escape_suffix_at_depth_limit_is_ineligible() {
    let input = format!(
        "{{\"message\":{}\"\\fLIENT ready\"{}}}",
        "[".repeat(63),
        "]".repeat(63)
    );

    let matched = classify(input.as_bytes(), LineEnding::Lf);

    assert!(!matched);
}
