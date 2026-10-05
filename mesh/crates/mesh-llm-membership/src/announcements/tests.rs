use super::*;

#[test]
fn version_allowed_for_rebroadcast_handles_floor() {
    // At or above the floor — allowed.
    assert!(version_allowed_for_rebroadcast(Some("0.60.0")));
    assert!(version_allowed_for_rebroadcast(Some("0.60.2")));
    assert!(version_allowed_for_rebroadcast(Some("0.64.0")));
    assert!(version_allowed_for_rebroadcast(Some("0.65.1")));
    assert!(version_allowed_for_rebroadcast(Some("1.0.0")));
    // Below the floor — refused.
    assert!(!version_allowed_for_rebroadcast(Some("0.57.0")));
    assert!(!version_allowed_for_rebroadcast(Some("0.55.1")));
    assert!(!version_allowed_for_rebroadcast(Some("0.58.0")));
    assert!(!version_allowed_for_rebroadcast(Some("0.59.99")));
}

#[test]
fn version_allowed_for_rebroadcast_handles_metadata_and_prerelease() {
    // Build metadata is stripped.
    assert!(version_allowed_for_rebroadcast(Some(
        "0.65.1+skippy.20260504.kv.2"
    )));
    assert!(!version_allowed_for_rebroadcast(Some("0.57.0+anything")));
    // Pre-release tags are stripped — 0.63.0-rc5 still passes.
    assert!(version_allowed_for_rebroadcast(Some("0.63.0-rc5")));
    assert!(!version_allowed_for_rebroadcast(Some("0.58.0-beta")));
}

#[test]
fn version_allowed_for_rebroadcast_is_conservative_on_unknown() {
    // Unparseable / missing / empty — preserved (don't drop legacy nodes
    // that never advertised a version).
    assert!(version_allowed_for_rebroadcast(None));
    assert!(version_allowed_for_rebroadcast(Some("")));
    assert!(version_allowed_for_rebroadcast(Some("   ")));
    assert!(version_allowed_for_rebroadcast(Some("garbage")));
    assert!(version_allowed_for_rebroadcast(Some("0")));
    assert!(version_allowed_for_rebroadcast(Some("0.x")));
}
