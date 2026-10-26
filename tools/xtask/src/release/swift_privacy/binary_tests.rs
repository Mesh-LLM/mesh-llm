use super::{PrivacyError, policy};
use plist::Value;

#[test]
fn rejects_numeric_tracking_when_binary_plist_contains_zero() {
    let mut fixture = Value::from_reader(std::io::Cursor::new(include_bytes!(
        "fixtures/PrivacyInfo.xcprivacy"
    )))
    .unwrap();
    fixture
        .as_dictionary_mut()
        .unwrap()
        .insert("NSPrivacyTracking".into(), Value::from(0));
    let mut bytes = Vec::new();
    fixture.to_writer_binary(&mut bytes).unwrap();
    let result = policy::validate(&bytes);
    assert!(matches!(result, Err(PrivacyError::Tracking)));
}

#[test]
fn rejects_partial_and_extra_reasons_when_timestamp_set_differs() {
    for reasons in [vec!["C617.1"], vec!["C617.1", "3B52.1", "extra"]] {
        let mut fixture = Value::from_reader(std::io::Cursor::new(include_bytes!(
            "fixtures/PrivacyInfo.xcprivacy"
        )))
        .unwrap();
        fixture
            .as_dictionary_mut()
            .unwrap()
            .get_mut("NSPrivacyAccessedAPITypes")
            .unwrap()
            .as_array_mut()
            .unwrap()[0]
            .as_dictionary_mut()
            .unwrap()
            .insert(
                "NSPrivacyAccessedAPITypeReasons".into(),
                Value::Array(reasons.iter().map(|reason| Value::from(*reason)).collect()),
            );
        let mut bytes = Vec::new();
        fixture.to_writer_binary(&mut bytes).unwrap();
        let result = policy::validate(&bytes);
        assert!(matches!(
            result,
            Err(PrivacyError::Reasons {
                category: "NSPrivacyAccessedAPICategoryFileTimestamp",
                ..
            })
        ));
    }
}
