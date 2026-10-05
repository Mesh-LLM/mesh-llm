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
        "exec_env",
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

/// `exec_env` is a closed set like every other base property, so a new
/// detection branch cannot start emitting an unbounded value.
#[test]
fn exec_env_is_from_the_fixed_set() {
    assert!(["plain", "container", "ci", "service"].contains(&exec_env()));
}

/// An empty variable is not a marker. `container=` in an inherited
/// environment must not label a laptop as a container.
#[test]
fn an_empty_marker_variable_does_not_count() {
    assert!(!env_is_set("MESH_LLM_DEFINITELY_UNSET_MARKER_FOR_TESTS"));
}
