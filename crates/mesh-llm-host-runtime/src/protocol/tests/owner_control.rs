use super::*;

#[test]
fn owner_control_handshake_empty_owner_id_uses_handshake_error() {
    let mut handshake = make_valid_owner_control_handshake();
    handshake
        .ownership
        .as_mut()
        .expect("test handshake must include ownership")
        .owner_id = "   ".to_string();

    let err = handshake
        .validate_frame()
        .expect_err("handshake with blank owner_id must be rejected");
    assert!(matches!(err, ControlFrameError::MissingControlOwnerId));
    assert_eq!(err.to_string(), "owner control handshake missing owner_id");
}

#[test]
fn owner_control_error_rejects_invalid_error_code() {
    for code in [OwnerControlErrorCode::Unspecified as i32, 9999] {
        let err = OwnerControlError {
            code,
            message: "invalid".to_string(),
            request_id: Some(1),
            current_revision: None,
        }
        .validate_frame()
        .expect_err("invalid owner-control error code must be rejected");
        assert!(matches!(
            err,
            ControlFrameError::InvalidOwnerControlErrorCode { got } if got == code
        ));
        assert_eq!(
            err.to_string(),
            format!("invalid owner control error code: {code}")
        );
    }
}
