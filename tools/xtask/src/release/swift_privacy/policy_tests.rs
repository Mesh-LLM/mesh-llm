use super::{PrivacyError, policy};
use plist::{Dictionary, Value};

const TEMPLATE: &[u8] = include_bytes!("fixtures/PrivacyInfo.xcprivacy");

fn manifest() -> Dictionary {
    Value::from_reader(std::io::Cursor::new(TEMPLATE))
        .unwrap()
        .into_dictionary()
        .unwrap()
}

fn xml(manifest: Dictionary) -> Vec<u8> {
    let mut bytes = Vec::new();
    Value::Dictionary(manifest)
        .to_writer_xml(&mut bytes)
        .unwrap();
    bytes
}

fn entries(manifest: &mut Dictionary) -> &mut Vec<Value> {
    manifest
        .get_mut("NSPrivacyAccessedAPITypes")
        .unwrap()
        .as_array_mut()
        .unwrap()
}

fn entry(category: &str, reasons: &[&str]) -> Value {
    let mut entry = Dictionary::new();
    entry.insert("NSPrivacyAccessedAPIType".into(), Value::from(category));
    entry.insert(
        "NSPrivacyAccessedAPITypeReasons".into(),
        Value::Array(reasons.iter().map(|reason| Value::from(*reason)).collect()),
    );
    Value::Dictionary(entry)
}

#[test]
fn accepts_actual_template_when_xml_is_unchanged() {
    let bytes = TEMPLATE;
    let result = policy::validate(bytes);
    assert!(result.is_ok(), "{result:?}");
}

#[test]
fn accepts_policy_when_encoded_as_binary() {
    let mut bytes = Vec::new();
    Value::Dictionary(manifest())
        .to_writer_binary(&mut bytes)
        .unwrap();
    let result = policy::validate(&bytes);
    assert!(result.is_ok(), "{result:?}");
}

#[test]
fn rejects_tracking_when_boolean_is_not_false() {
    for tracking in [Value::from(true), Value::from(0), Value::from("false")] {
        let mut fixture = manifest();
        fixture.insert("NSPrivacyTracking".into(), tracking);
        let result = policy::validate(&xml(fixture));
        assert!(matches!(result, Err(PrivacyError::Tracking)));
    }
}

#[test]
fn rejects_policy_when_required_top_level_field_is_missing() {
    for (field, expected) in [
        ("NSPrivacyTracking", "tracking"),
        ("NSPrivacyCollectedDataTypes", "collected"),
        ("NSPrivacyTrackingDomains", "domains"),
        ("NSPrivacyAccessedAPITypes", "reasons"),
    ] {
        let mut fixture = manifest();
        fixture.remove(field);
        let result = policy::validate(&xml(fixture));
        match expected {
            "tracking" => assert!(matches!(result, Err(PrivacyError::Tracking))),
            "collected" => assert!(matches!(result, Err(PrivacyError::CollectedData))),
            "domains" => assert!(matches!(result, Err(PrivacyError::TrackingDomains))),
            "reasons" => assert!(matches!(
                result,
                Err(PrivacyError::Reasons {
                    category: "NSPrivacyAccessedAPICategoryFileTimestamp",
                    ..
                })
            )),
            _ => unreachable!(),
        }
    }
}

#[test]
fn rejects_empty_list_fields_when_nonempty_or_wrong_shape() {
    for field in ["NSPrivacyCollectedDataTypes", "NSPrivacyTrackingDomains"] {
        for value in [
            Value::Array(vec![Value::from("data")]),
            Value::Dictionary(Dictionary::new()),
        ] {
            let mut fixture = manifest();
            fixture.insert(field.into(), value);
            let result = policy::validate(&xml(fixture));
            assert!(matches!(
                result,
                Err(PrivacyError::CollectedData | PrivacyError::TrackingDomains)
            ));
        }
    }
}

#[test]
fn rejects_entry_when_category_or_reasons_are_missing() {
    for field in [
        "NSPrivacyAccessedAPIType",
        "NSPrivacyAccessedAPITypeReasons",
    ] {
        let mut fixture = manifest();
        entries(&mut fixture)[0]
            .as_dictionary_mut()
            .unwrap()
            .remove(field);
        let result = policy::validate(&xml(fixture));
        assert!(matches!(result, Err(PrivacyError::InvalidEntry(0))));
    }
}

#[test]
fn rejects_category_when_required_entry_is_absent() {
    for (index, (category, _)) in policy::EXPECTED.iter().enumerate() {
        let mut fixture = manifest();
        entries(&mut fixture).remove(index);
        let result = policy::validate(&xml(fixture));
        assert!(
            matches!(result, Err(PrivacyError::Reasons { category: found, .. }) if found == *category)
        );
    }
}

#[test]
fn rejects_reasons_when_set_has_missing_or_extra_reason() {
    for (index, (category, _)) in policy::EXPECTED.iter().enumerate() {
        let mut fixture = manifest();
        entries(&mut fixture)[index] = entry(category, &["wrong"]);
        let result = policy::validate(&xml(fixture));
        assert!(
            matches!(result, Err(PrivacyError::Reasons { category: found, .. }) if found == *category)
        );
    }
}

#[test]
fn accepts_reasons_when_duplicated_and_entries_reordered() {
    let mut fixture = manifest();
    entries(&mut fixture)[0] = entry(policy::EXPECTED[0].0, &["3B52.1", "C617.1", "C617.1"]);
    entries(&mut fixture).reverse();
    let result = policy::validate(&xml(fixture));
    assert!(result.is_ok(), "{result:?}");
}

#[test]
fn accepts_legacy_reason_dictionary_when_keys_match() {
    let mut fixture = manifest();
    let reasons: Dictionary = [
        (String::from("C617.1"), Value::from(1)),
        (String::from("3B52.1"), Value::from(false)),
    ]
    .into_iter()
    .collect();
    entries(&mut fixture)[0]
        .as_dictionary_mut()
        .unwrap()
        .insert(
            "NSPrivacyAccessedAPITypeReasons".into(),
            Value::Dictionary(reasons),
        );
    fixture.insert("UnknownFutureField".into(), Value::from("ignored"));
    let result = policy::validate(&xml(fixture));
    assert!(result.is_ok(), "{result:?}");
}

#[test]
fn rejects_duplicate_category_before_reason_mismatch() {
    let mut fixture = manifest();
    entries(&mut fixture).push(entry(policy::EXPECTED[0].0, &["wrong"]));
    let result = policy::validate(&xml(fixture));
    assert!(
        matches!(result, Err(PrivacyError::DuplicateCategory(category)) if category == policy::EXPECTED[0].0)
    );
}

#[test]
fn orders_unexpected_categories_when_required_sets_match() {
    let mut fixture = manifest();
    entries(&mut fixture).extend([entry("Zulu", &["Z"]), entry("Alpha", &["A"])]);
    let result = policy::validate(&xml(fixture));
    assert!(
        matches!(result, Err(PrivacyError::UnexpectedCategories(categories)) if categories == ["Alpha", "Zulu"])
    );
}

#[test]
fn reports_required_reason_mismatch_before_unexpected_category() {
    let mut fixture = manifest();
    entries(&mut fixture).push(entry("Alpha", &["A"]));
    entries(&mut fixture).remove(0);
    let result = policy::validate(&xml(fixture));
    assert!(matches!(
        result,
        Err(PrivacyError::Reasons {
            category: "NSPrivacyAccessedAPICategoryFileTimestamp",
            ..
        })
    ));
}

#[test]
fn reports_field_errors_in_legacy_order_when_multiple_fields_fail() {
    for (removed, expected) in [(0, "tracking"), (1, "collected"), (2, "domains")] {
        let mut fixture = manifest();
        for field in [
            "NSPrivacyTracking",
            "NSPrivacyCollectedDataTypes",
            "NSPrivacyTrackingDomains",
        ]
        .into_iter()
        .skip(removed)
        {
            fixture.remove(field);
        }
        let result = policy::validate(&xml(fixture));
        match expected {
            "tracking" => assert!(matches!(result, Err(PrivacyError::Tracking))),
            "collected" => assert!(matches!(result, Err(PrivacyError::CollectedData))),
            "domains" => assert!(matches!(result, Err(PrivacyError::TrackingDomains))),
            _ => unreachable!(),
        }
    }
}

#[test]
fn rejects_malformed_input_with_typed_error() {
    let bytes = b"not a plist";
    let result = policy::validate(bytes);
    assert!(matches!(result, Err(PrivacyError::Plist(_))));
}
