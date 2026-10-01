use super::classify;
use crate::process::LineEnding;

predicate_case!(
    migration_lifecycle_approved_structured_when_message_absent,
    br#"{"event":"passive_mode","status":"ready","role":"client"}"#,
    true
);
predicate_case!(
    migration_lifecycle_approved_structured_when_message_array,
    br#"{"event":"passive_mode","status":"ready","role":"client","message":["starting"]}"#,
    true
);
predicate_case!(
    migration_lifecycle_approved_structured_when_message_object,
    br#"{"event":"passive_mode","status":"ready","role":"client","message":{"detail":"starting"}}"#,
    true
);
predicate_case!(
    migration_lifecycle_approved_structured_when_message_null,
    br#"{"event":"passive_mode","status":"ready","role":"client","message":null}"#,
    true
);
predicate_case!(
    migration_lifecycle_approved_structured_when_message_number,
    br#"{"event":"passive_mode","status":"ready","role":"client","message":42}"#,
    true
);
predicate_case!(
    migration_lifecycle_approved_structured_when_message_boolean,
    br#"{"event":"passive_mode","status":"ready","role":"client","message":false}"#,
    true
);
predicate_case!(
    migration_lifecycle_approved_string_when_structured_fields_invalid,
    br#"{"event":[],"status":{},"role":7,"message":"prefix cLiEnT ReAdY suffix"}"#,
    true
);
predicate_case!(
    migration_lifecycle_approved_string_when_phrase_absent,
    br#"{"message":"client starting"}"#,
    false
);
predicate_case!(
    migration_lifecycle_approved_array_is_not_coerced,
    br#"{"message":["Client ready"]}"#,
    false
);
predicate_case!(
    migration_lifecycle_approved_object_value_is_not_coerced,
    br#"{"message":{"detail":"Client ready"}}"#,
    false
);
predicate_case!(
    migration_lifecycle_approved_object_key_is_not_coerced,
    br#"{"message":{"Client ready":false}}"#,
    false
);
predicate_case!(
    migration_lifecycle_approved_escape_suffix_array_is_not_coerced,
    br#"{"message":["\fLIENT ready"]}"#,
    false
);
predicate_case!(
    migration_lifecycle_approved_escape_suffix_key_is_not_coerced,
    br#"{"message":{"\u001cLIENT ready":false}}"#,
    false
);
predicate_case!(
    migration_lifecycle_approved_unicode16_counterexample_is_not_coerced,
    br#"{"message":["\u0c5cLIENT ready"]}"#,
    false
);
predicate_case!(
    migration_lifecycle_approved_unicode16_counterexample_key_is_not_coerced,
    br#"{"message":{"\u0c5cLIENT ready":true}}"#,
    false
);
predicate_case!(
    migration_lifecycle_approved_direct_escape_suffix_stays_negative,
    br#"{"message":"\fLIENT ready"}"#,
    false
);
predicate_case!(
    migration_lifecycle_approved_direct_literal_backslash_stays_positive,
    br#"{"message":"\\u200cLIENT ready"}"#,
    true
);
predicate_case!(
    migration_lifecycle_approved_direct_unicode_is_not_ascii_restricted,
    br#"{"message":"\u0c5c Client ready \ud83d\ude0c"}"#,
    true
);
predicate_case!(
    migration_lifecycle_approved_direct_lowercase_does_not_casefold,
    br#"{"message":"CL\u0130ENT READY"}"#,
    false
);
predicate_case!(
    migration_lifecycle_approved_direct_space_is_not_normalized,
    br#"{"message":"client\u00a0ready"}"#,
    false
);
predicate_case!(
    migration_lifecycle_approved_duplicate_last_nonstring_rejects,
    br#"{"message":"Client ready","message":["Client ready"]}"#,
    false
);
predicate_case!(
    migration_lifecycle_approved_duplicate_last_string_accepts,
    br#"{"message":["starting"],"message":"Client ready"}"#,
    true
);
