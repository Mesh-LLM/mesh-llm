use super::*;
use std::time::Duration;
use tokio::io::AsyncWriteExt;

#[test]
fn tunneled_request_metadata_uses_authenticated_peer_and_selected_direct_path() {
    let remote = EndpointId::from(iroh::SecretKey::from_bytes(&[0x61; 32]).public());
    let direct_addr = "192.0.2.61:11204".parse().expect("direct address");
    let metadata = remote_tunnel_request_metadata(
        remote,
        Some(crate::mesh::SelectedPathObservation {
            path_type: "direct",
            rtt_ms: Some(9),
            observed_direct_remote_addr: Some(direct_addr),
        }),
    );

    assert_eq!(
        metadata.caller_endpoint_id(),
        Some(hex::encode(remote.as_bytes()).as_str())
    );
    assert_eq!(
        metadata.caller_addr(),
        Some(direct_addr.to_string().as_str())
    );
    assert_eq!(metadata.caller_path_type(), Some("remote_quic_http"));
}

#[test]
fn tunneled_request_metadata_omits_address_for_relay_path() {
    let remote = EndpointId::from(iroh::SecretKey::from_bytes(&[0x62; 32]).public());
    let metadata = remote_tunnel_request_metadata(
        remote,
        Some(crate::mesh::SelectedPathObservation {
            path_type: "relay",
            rtt_ms: Some(27),
            observed_direct_remote_addr: Some(
                "203.0.113.62:443".parse().expect("relay-shaped address"),
            ),
        }),
    );

    assert_eq!(
        metadata.caller_endpoint_id(),
        Some(hex::encode(remote.as_bytes()).as_str())
    );
    assert!(metadata.caller_addr().is_none());
    assert_eq!(metadata.caller_path_type(), Some("relay"));
}

#[test]
fn tunneled_request_metadata_preserves_endpoint_when_path_is_missing() {
    let remote = EndpointId::from(iroh::SecretKey::from_bytes(&[0x63; 32]).public());
    let metadata = remote_tunnel_request_metadata(remote, None);

    assert_eq!(
        metadata.caller_endpoint_id(),
        Some(hex::encode(remote.as_bytes()).as_str())
    );
    assert!(metadata.caller_addr().is_none());
    assert!(metadata.caller_path_type().is_none());
    assert!(metadata.has_authenticated_remote_caller());
}

#[test]
fn tunneled_request_metadata_preserves_endpoint_for_unrecognized_path() {
    let remote = EndpointId::from(iroh::SecretKey::from_bytes(&[0x64; 32]).public());
    let metadata = remote_tunnel_request_metadata(
        remote,
        Some(crate::mesh::SelectedPathObservation {
            path_type: "unknown",
            rtt_ms: None,
            observed_direct_remote_addr: Some(
                "198.51.100.64:443"
                    .parse()
                    .expect("unrecognized path address"),
            ),
        }),
    );

    assert_eq!(
        metadata.caller_endpoint_id(),
        Some(hex::encode(remote.as_bytes()).as_str())
    );
    assert!(metadata.caller_addr().is_none());
    assert!(metadata.caller_path_type().is_none());
    assert!(metadata.has_authenticated_remote_caller());
}

#[tokio::test]
async fn tunnel_prefetch_forwards_a_complete_bounded_header_prefix() {
    let (mut writer, mut reader) = tokio::io::duplex(4096);
    let request = b"POST /v1/chat/completions HTTP/1.1\r\nx-request-id: 550e8400-e29b-41d4-a716-446655440000\r\n\r\n";
    writer.write_all(request).await.unwrap();

    let prefix = read_tunneled_http_header_prefix(&mut reader).await.unwrap();

    assert_eq!(prefix, request);
    assert!(prefix.len() <= crate::network::openai::request_parse::MAX_HEADER_BYTES);
}

#[tokio::test(start_paused = true)]
async fn tunnel_prefetch_completes_on_lf_only_header_terminator() {
    let (mut writer, mut reader) = tokio::io::duplex(4096);
    let request = b"GET /v1/models HTTP/1.1\nHost: localhost\n\n";
    writer.write_all(request).await.unwrap();

    let prefix = tokio::time::timeout(
        Duration::from_secs(1),
        read_tunneled_http_header_prefix(&mut reader),
    )
    .await
    .expect("LF-only header pre-read must complete while the stream remains open")
    .unwrap();

    assert_eq!(prefix, request);
}

#[tokio::test(start_paused = true)]
async fn tunnel_prefetch_times_out_when_peer_is_silent() {
    let (_writer, mut reader) = tokio::io::duplex(4096);

    let result = tokio::time::timeout(
        Duration::from_secs(6),
        read_tunneled_http_header_prefix(&mut reader),
    )
    .await
    .expect("header pre-read must finish within the regression guard");

    let error = result.expect_err("a silent peer must exceed the header deadline");
    assert!(
        error
            .to_string()
            .contains("timed out reading tunneled HTTP header prefix")
    );
}

#[tokio::test(start_paused = true)]
async fn tunnel_prefetch_slow_partial_progress_does_not_reset_deadline() {
    let (mut writer, mut reader) = tokio::io::duplex(4096);
    let writer_task = tokio::spawn(async move {
        for fragment in [
            b"POST /v1/responses HTTP/1.1\r\n".as_slice(),
            b"Host: localhost\r\n".as_slice(),
            b"Content-Length: 2\r\n".as_slice(),
            b"\r\n{}".as_slice(),
        ] {
            writer.write_all(fragment).await.unwrap();
            tokio::time::sleep(Duration::from_secs(2)).await;
        }
    });

    let result = tokio::time::timeout(
        Duration::from_secs(7),
        read_tunneled_http_header_prefix(&mut reader),
    )
    .await
    .expect("header pre-read must finish within the regression guard");

    let error = result.expect_err("partial progress must not reset the header deadline");
    assert!(
        error
            .to_string()
            .contains("timed out reading tunneled HTTP header prefix")
    );
    writer_task.await.unwrap();
}

#[tokio::test(start_paused = true)]
async fn tunnel_prefetch_preserves_fragmented_header_and_body_overread() {
    let (mut writer, mut reader) = tokio::io::duplex(4096);
    let fragments: [&[u8]; 3] = [
        b"POST /v1/responses HTTP/1.1\r\n",
        b"Host: localhost\r\nContent-Length: 2\r\n",
        b"\r\n{}",
    ];
    let writer_task = tokio::spawn(async move {
        for fragment in fragments {
            writer.write_all(fragment).await.unwrap();
            tokio::time::sleep(Duration::from_secs(1)).await;
        }
    });

    let prefix = read_tunneled_http_header_prefix(&mut reader)
        .await
        .expect("fragmented header must complete before the deadline");

    assert_eq!(
        prefix,
        b"POST /v1/responses HTTP/1.1\r\nHost: localhost\r\nContent-Length: 2\r\n\r\n{}"
    );
    writer_task.await.unwrap();
}

#[test]
fn tunnel_prefix_without_request_id_gets_one_canonical_id() {
    let original = b"POST /v1/responses HTTP/1.1\r\nHost: localhost\r\nContent-Length: 2\r\n\r\n{}";

    let (rewritten, request_id) =
        crate::network::openai::request_parse::ensure_canonical_request_id_in_header_prefix(
            original.to_vec(),
        );

    let request_id = request_id.expect("generated canonical request ID");
    assert_eq!(
        remote_tunnel_request_ids(&rewritten),
        (Some(request_id), None)
    );
    assert_eq!(
        rewritten
            .windows(b"x-request-id:".len())
            .filter(|window| window.eq_ignore_ascii_case(b"x-request-id:"))
            .count(),
        1
    );
    assert!(rewritten.ends_with(b"\r\n\r\n{}"));
}

#[test]
fn lf_only_tunnel_prefix_inserts_request_id_on_its_own_header_line() {
    let original = b"GET /v1/models HTTP/1.1\nHost: localhost\n\n";

    let (rewritten, request_id) =
        crate::network::openai::request_parse::ensure_canonical_request_id_in_header_prefix(
            original.to_vec(),
        );

    let request_id = request_id.expect("generated canonical request ID");
    let mut headers = [httparse::EMPTY_HEADER; 8];
    let mut request = httparse::Request::new(&mut headers);
    assert_eq!(
        request
            .parse(&rewritten)
            .expect("rewritten request must parse"),
        httparse::Status::Complete(rewritten.len())
    );
    assert_eq!(
        request
            .headers
            .iter()
            .filter(|header| header.name.eq_ignore_ascii_case("x-request-id"))
            .count(),
        1
    );
    let request_id_line = format!("\nx-request-id: {}\n\n", request_id.as_uuid());
    assert!(
        rewritten
            .windows(request_id_line.len())
            .any(|window| window == request_id_line.as_bytes())
    );
    assert!(!rewritten.contains(&b'\r'));
}

#[test]
fn tunnel_prefix_preserves_existing_canonical_request_id_byte_for_byte() {
    let request_id = mesh_llm_events::logging::identifiers::RequestId::new();
    let original = format!(
        "POST /v1/responses HTTP/1.1\r\nHost: localhost\r\nx-request-id: {}\r\n\r\n",
        request_id.as_uuid()
    )
    .into_bytes();

    let (rewritten, selected) =
        crate::network::openai::request_parse::ensure_canonical_request_id_in_header_prefix(
            original.clone(),
        );

    assert_eq!(selected, Some(request_id));
    assert_eq!(rewritten, original);
}

#[test]
fn duplicate_or_malformed_tunnel_request_id_fails_open_without_rewrite() {
    let first = mesh_llm_events::logging::identifiers::RequestId::new();
    let second = mesh_llm_events::logging::identifiers::RequestId::new();
    let duplicate = format!(
        "GET /v1/models HTTP/1.1\r\nx-request-id: {}\r\nx-request-id: {}\r\n\r\n",
        first.as_uuid(),
        second.as_uuid()
    )
    .into_bytes();
    let malformed = b"GET /v1/models HTTP/1.1\r\nx-request-id: client-controlled\r\n\r\n".to_vec();

    for original in [duplicate, malformed] {
        let (rewritten, selected) =
            crate::network::openai::request_parse::ensure_canonical_request_id_in_header_prefix(
                original.clone(),
            );
        assert_eq!(selected, None);
        assert_eq!(rewritten, original);
        assert_eq!(remote_tunnel_request_ids(&rewritten), (None, None));
    }
}

#[test]
fn full_bounded_tunnel_header_without_request_id_fails_open() {
    let start = b"GET /v1/models HTTP/1.1\r\nx-padding: ";
    let ending = b"\r\n\r\n";
    let padding_length =
        crate::network::openai::request_parse::MAX_HEADER_BYTES - start.len() - ending.len();
    let mut original = start.to_vec();
    original.extend(std::iter::repeat_n(b'x', padding_length));
    original.extend_from_slice(ending);

    let (rewritten, selected) =
        crate::network::openai::request_parse::ensure_canonical_request_id_in_header_prefix(
            original.clone(),
        );

    assert_eq!(
        original.len(),
        crate::network::openai::request_parse::MAX_HEADER_BYTES
    );
    assert_eq!(selected, None);
    assert_eq!(rewritten, original);
    assert_eq!(remote_tunnel_request_ids(&rewritten), (None, None));
}

#[test]
fn markerless_canonical_request_id_allows_attribution_without_suppression() {
    let request_id = mesh_llm_events::logging::identifiers::RequestId::new();
    let prefix = format!(
        "POST /v1/responses HTTP/1.1\r\nx-request-id: {}\r\n\r\n",
        request_id.as_uuid()
    );

    assert_eq!(
        remote_tunnel_request_ids(prefix.as_bytes()),
        (Some(request_id), None)
    );
}

#[test]
fn near_limit_markerless_headers_keep_canonical_attribution() {
    let request_id = mesh_llm_events::logging::identifiers::RequestId::new();
    let start = format!(
        "POST /v1/responses HTTP/1.1\r\nx-request-id: {}\r\nx-padding: ",
        request_id.as_uuid()
    );
    let ending = "\r\n\r\n";
    let padding_length =
        crate::network::openai::request_parse::MAX_HEADER_BYTES - start.len() - ending.len();
    let prefix = format!("{start}{}{ending}", "x".repeat(padding_length));

    assert_eq!(
        prefix.len(),
        crate::network::openai::request_parse::MAX_HEADER_BYTES
    );
    assert_eq!(
        remote_tunnel_request_ids(prefix.as_bytes()),
        (Some(request_id), None)
    );
}

#[test]
fn matching_private_marker_enables_single_parent_suppression() {
    let request_id = mesh_llm_events::logging::identifiers::RequestId::new();
    let prefix = format!(
        "POST /v1/responses HTTP/1.1\r\nx-request-id: {0}\r\n{1}: {0}\r\n\r\n",
        request_id.as_uuid(),
        crate::network::openai::request_parse::RAW_LIFECYCLE_OWNER_HEADER,
    );

    assert_eq!(
        remote_tunnel_request_ids(prefix.as_bytes()),
        (Some(request_id), Some(request_id))
    );
}

#[test]
fn malformed_incomplete_and_maximal_header_prefixes_fail_open() {
    let request_id = mesh_llm_events::logging::identifiers::RequestId::new();
    let malformed = b"POST /v1/responses HTTP/1.1\r\nx-request-id: not-a-uuid\r\n\r\n";
    let incomplete = format!(
        "POST /v1/responses HTTP/1.1\r\nx-request-id: {}\r\n",
        request_id.as_uuid()
    );
    let maximal = vec![b'x'; crate::network::openai::request_parse::MAX_HEADER_BYTES];

    for prefix in [
        malformed.as_slice(),
        incomplete.as_bytes(),
        maximal.as_slice(),
    ] {
        assert_eq!(remote_tunnel_request_ids(prefix), (None, None));
    }
}
