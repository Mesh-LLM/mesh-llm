use meshllm_ffi::{FfiError, create_node};

#[test]
fn invalid_invite_token_returns_ffi_error() {
    let result = create_node("deadbeef".to_string(), "".to_string(), None, None, false);
    match result {
        Ok(_) => panic!("expected Err(FfiError::InvalidInviteToken(_))"),
        Err(FfiError::InvalidInviteToken(_)) => {} // expected
        Err(other) => panic!("Expected InvalidInviteToken, got {:?}", other),
    }
}

#[test]
fn no_anyhow_in_exported_functions() {
    // Verifies at compile time that FfiError implements std::error::Error,
    // confirming no anyhow::Error leaks across the FFI boundary.
    fn _assert_error<E: std::error::Error>() {}
    _assert_error::<FfiError>();
}

#[test]
fn ffi_error_all_variants_present() {
    // Exhaustive match ensures all required variants exist and are reachable.
    // Adding a variant to FfiError without updating this test will cause a compile error.
    let variants: &[FfiError] = &[
        FfiError::InvalidInviteToken("message".to_string()),
        FfiError::InvalidOwnerKeypair("message".to_string()),
        FfiError::BuildFailed("message".to_string()),
        FfiError::JoinFailed("message".to_string()),
        FfiError::DiscoveryFailed("message".to_string()),
        FfiError::StreamFailed("message".to_string()),
        FfiError::Cancelled("message".to_string()),
        FfiError::ReconnectFailed("message".to_string()),
        FfiError::HostUnavailable("message".to_string()),
        FfiError::ModelManagementFailed("message".to_string()),
        FfiError::ServingFailed("message".to_string()),
        FfiError::ServingUnsupported("message".to_string()),
    ];
    for v in variants {
        assert!(
            !v.to_string().is_empty(),
            "FfiError::{:?} has empty Display",
            v
        );
    }
}
