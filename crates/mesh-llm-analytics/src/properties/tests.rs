use super::*;
use crate::event::Value;

#[test]
fn classifies_build_channels() {
    assert_eq!(BuildChannel::classify("0.76.0"), BuildChannel::Release);
    assert_eq!(
        BuildChannel::classify("0.76.0-rc8"),
        BuildChannel::Prerelease
    );
    assert_eq!(
        BuildChannel::classify("0.76.0+gABCDEF"),
        BuildChannel::Development
    );
    assert_eq!(
        BuildChannel::classify("0.76.0+gABCDEF.dirty"),
        BuildChannel::Development
    );
}

#[test]
fn base_properties_describe_the_build_and_platform() {
    let properties = base_properties();
    let keys: Vec<_> = properties.entries().map(|(key, _)| key).collect();
    for expected in [
        "mesh_llm_version",
        "build_channel",
        "os",
        "arch",
        "$lib",
        "$lib_version",
    ] {
        assert!(keys.contains(&expected), "missing {expected}");
    }
}

#[test]
fn base_properties_carry_no_free_text() {
    // Every base value is either a compile-time constant or a sanitized
    // label; nothing here can carry an arbitrary string from the environment.
    for (key, value) in base_properties().entries() {
        match value {
            Value::Static(_) => {}
            Value::Text(label) => assert!(
                crate::Label::sanitize(label.as_str()).is_some() || label.as_str() == "redacted",
                "{key} holds an unsanitized value",
            ),
            other => panic!("{key} holds an unexpected value: {other:?}"),
        }
    }
}

#[test]
fn platform_strings_are_from_the_fixed_set() {
    assert!(["macos", "linux", "windows", "other"].contains(&os_family()));
    assert!(["aarch64", "x86_64", "other"].contains(&architecture()));
}
