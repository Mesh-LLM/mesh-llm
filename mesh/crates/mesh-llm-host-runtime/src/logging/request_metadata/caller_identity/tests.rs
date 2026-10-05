use super::*;

#[test]
fn local_http_caller_identity_keeps_ipv4_and_ipv6_socket_addresses() {
    for caller_addr in [
        "127.0.0.1:40123",
        "192.0.2.10:40123",
        "[2001:db8::10]:40123",
    ] {
        let metadata = RequestSummaryMetadata::default().with_caller_identity(
            None,
            Some(caller_addr),
            Some(CallerPathType::LocalHttp),
        );

        assert_eq!(metadata.caller_addr(), Some(caller_addr));
        assert_eq!(metadata.caller_path_type(), Some("local_http"));
    }
}

#[test]
fn caller_path_vocabulary_rejects_stage_transport_as_a_request_caller() {
    let parsed = serde_json::from_str::<CallerPathType>("\"remote_quic_stage\"");

    assert!(parsed.is_err());
}

#[test]
fn authenticated_endpoint_only_caller_is_preserved_without_address_or_path() {
    let endpoint_id = "ab".repeat(32);

    let metadata =
        RequestSummaryMetadata::default().with_caller_identity(Some(&endpoint_id), None, None);

    assert_eq!(metadata.caller_endpoint_id(), Some(endpoint_id.as_str()));
    assert_eq!(metadata.caller_addr(), None);
    assert_eq!(metadata.caller_path_type(), None);
    assert!(metadata.has_authenticated_remote_caller());
}

#[test]
fn endpoint_only_caller_rejects_partial_or_unauthenticated_tuples() {
    let endpoint_id = "cd".repeat(32);
    for metadata in [
        RequestSummaryMetadata::default().with_caller_identity(
            Some(&endpoint_id),
            Some("192.0.2.80:11204"),
            None,
        ),
        RequestSummaryMetadata::default().with_caller_identity(
            Some("not-an-authenticated-endpoint"),
            None,
            None,
        ),
        RequestSummaryMetadata::default().with_caller_identity(None, None, None),
        RequestSummaryMetadata::default().with_caller_identity(
            Some(&endpoint_id),
            Some("127.0.0.1:40123"),
            Some(CallerPathType::LocalHttp),
        ),
    ] {
        assert!(!metadata.has_authenticated_remote_caller());
        assert_eq!(metadata.caller_endpoint_id(), None);
    }
}

#[test]
fn first_authenticated_endpoint_only_caller_cannot_be_overwritten() {
    let first_endpoint_id = "de".repeat(32);
    let later_endpoint_id = "ef".repeat(32);
    let mut metadata = RequestSummaryMetadata::default().with_caller_identity(
        Some(&first_endpoint_id),
        None,
        None,
    );

    let changed = metadata.merge_authenticated_remote_caller(
        RequestSummaryMetadata::default().with_caller_identity(
            Some(&later_endpoint_id),
            Some("192.0.2.81:11204"),
            Some(CallerPathType::RemoteQuicHttp),
        ),
    );

    assert!(!changed);
    assert_eq!(
        metadata.caller_endpoint_id(),
        Some(first_endpoint_id.as_str())
    );
    assert_eq!(metadata.caller_addr(), None);
    assert_eq!(metadata.caller_path_type(), None);
}
