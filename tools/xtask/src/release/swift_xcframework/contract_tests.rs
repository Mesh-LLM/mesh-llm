use super::{
    Mode,
    contract::Component,
    fixtures::{Fixture, entry, set},
};
use plist::Value;

#[test]
fn full_matrix_when_xml_and_binary_plists() {
    for binary in [false, true] {
        let fixture = Fixture::full();
        fixture.write(binary);
        let result = fixture.declarations(Some(Mode::Full));
        assert_eq!(result.unwrap(), 4);
    }
}

#[test]
fn host_only_when_exact_arm64_declaration() {
    let fixture = Fixture::host();
    fixture.write(false);
    let result = fixture.declarations(Some(Mode::HostOnly));
    assert_eq!(result.unwrap(), 1);
}

#[test]
fn architecture_contract_when_simulator_includes_unsupported_x86_64() {
    let mut fixture = Fixture::full();
    fixture.entries[1] = entry("ios", "simulator", &["arm64", "x86_64"]);
    fixture.write(false);
    let result = fixture.declarations(Some(Mode::Full));
    assert!(
        result
            .unwrap_err()
            .to_string()
            .contains("unexpected architecture contract")
    );
}

#[test]
fn macos_required_when_mode_is_absent() {
    let fixture = Fixture::new(vec![entry("ios", "", &["arm64"])]);
    fixture.write(true);
    let result = fixture.declarations(None);
    assert!(
        result
            .unwrap_err()
            .to_string()
            .contains("does not contain a macOS")
    );
}

#[test]
fn duplicate_matrix_when_same_platform_variant_appears_twice() {
    let fixture = Fixture::new(vec![
        entry("macos", "", &["arm64"]),
        entry("macos", "", &["x86_64"]),
    ]);
    fixture.write(false);
    let result = fixture.declarations(None);
    assert!(
        result
            .unwrap_err()
            .to_string()
            .contains("duplicate platform slice")
    );
}

#[test]
fn duplicate_architectures_when_declaration_repeats_name() {
    let fixture = Fixture::new(vec![entry("macos", "", &["arm64", "arm64"])]);
    fixture.write(true);
    let result = fixture.declarations(None);
    assert!(
        result
            .unwrap_err()
            .to_string()
            .contains("duplicate architectures")
    );
}

#[test]
fn invalid_platform_and_variant_when_values_have_wrong_types() {
    for (field, value) in [
        ("SupportedPlatform", Value::Boolean(true)),
        ("SupportedPlatform", Value::String(String::new())),
        ("SupportedPlatformVariant", Value::Boolean(false)),
    ] {
        let mut fixture = Fixture::host();
        set(&mut fixture.entries[0], field, value);
        fixture.write(false);
        let result = fixture.declarations(None);
        assert!(result.unwrap_err().to_string().contains(field));
    }
}

#[test]
fn invalid_architectures_when_absent_empty_or_nonstring() {
    for value in [
        Value::Boolean(true),
        Value::Array(vec![]),
        Value::Array(vec![Value::Boolean(true)]),
        Value::Array(vec![Value::String(String::new())]),
    ] {
        let mut fixture = Fixture::host();
        set(&mut fixture.entries[0], "SupportedArchitectures", value);
        fixture.write(false);
        let result = fixture.declarations(None);
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("SupportedArchitectures")
        );
    }
}

#[test]
fn invalid_components_when_absolute_parent_or_multicomponent() {
    for value in [
        "", ".", "./", "..", "../", "/slice", "//slice", "a/b", "a/../b",
    ] {
        let input = Value::String(value.into());
        let result = Component::parse(Some(&input), "LibraryIdentifier");
        assert!(result.is_err(), "{value:?}");
    }
}

#[test]
fn posix_components_when_redundant_dot_slash_or_literal_backslash() {
    for value in [
        "slice",
        "./slice",
        "slice/",
        "slice//",
        "slice/./",
        "././slice",
        "a\\b",
        "C:drive",
        "token",
        "a\0b",
    ] {
        let input = Value::String(value.into());
        let result = Component::parse(Some(&input), "LibraryIdentifier");
        assert_eq!(result.unwrap().as_str(), value);
    }
}

#[test]
fn platform_order_when_mode_independent_has_arbitrary_slices() {
    let fixture = Fixture::new(vec![
        entry("watchos", "future", &["custom"]),
        entry("macos", "", &["arm64"]),
    ]);
    fixture.write(true);
    let document = super::input::Document::read(&fixture.root).unwrap();
    let result = document.entries(None).unwrap();
    assert_eq!(
        result
            .iter()
            .map(|entry| entry.key.platform.as_str())
            .collect::<Vec<_>>(),
        ["watchos", "macos"]
    );
}

#[test]
fn matrix_precedes_invalid_declaration_when_mode_is_full() {
    let mut fixture = Fixture::host();
    set(
        &mut fixture.entries[0],
        "SupportedArchitectures",
        Value::Boolean(false),
    );
    fixture.write(false);
    let result = fixture.declarations(Some(Mode::Full));
    assert!(
        result
            .unwrap_err()
            .to_string()
            .contains("unexpected platform matrix")
    );
}

#[test]
fn finalization_preserves_primary_when_unregister_also_fails() {
    let primary = super::Error::Contract("preceding-input-error".into());
    let result = super::error::finalize::<()>(
        Err(primary),
        Err(crate::command_interrupt::Reason::ScopeBusy),
    );
    match result.unwrap_err() {
        super::Error::Finalization { primary, reason } => {
            assert_eq!(primary.to_string(), "preceding-input-error");
            assert!(matches!(
                reason,
                crate::command_interrupt::Reason::ScopeBusy
            ));
        }
        other => panic!("unexpected {other:?}"),
    }
}

#[test]
fn finalization_fails_success_when_interrupted() {
    let result = super::error::finalize(Ok(4), Err(crate::command_interrupt::Reason::Interrupted));
    assert!(matches!(
        result,
        Err(super::Error::Interrupt(
            crate::command_interrupt::Reason::Interrupted
        ))
    ));
}
