//! Selected-route evidence must commit the forwarded entity, excluding chunk framing.

use super::*;
use std::borrow::Cow;

const ENTITY: &[u8] = br#"{"model":"m","messages":[]}"#;
// Independently calculated with `printf '%s' ... | shasum -a 256`.
const ENTITY_SHA256: &str = "0bfcf1c873fe23e87366969117efdc24b95f341eb2f4abe10ae01e7a1f4994c6";

async fn read_request(
    path: &str,
    wire_body: &[u8],
    chunked: bool,
) -> transport::BufferedHttpRequest {
    let framing = if chunked {
        "Transfer-Encoding: chunked\r\nTrailer: X-Receipt\r\n".to_owned()
    } else {
        format!("Content-Length: {}\r\n", wire_body.len())
    };
    let mut wire = format!(
        "POST {path} HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\n{framing}\r\n"
    )
    .into_bytes();
    wire.extend_from_slice(wire_body);
    let (mut reader, mut writer) = tokio::io::duplex(wire.len() + 1);
    writer.write_all(&wire).await.unwrap();
    transport::read_http_request(&mut reader).await.unwrap()
}

fn event(request: &transport::BufferedHttpRequest) -> serde_json::Value {
    selected_route_event(
        request,
        Some("m"),
        "external",
        SelectedRouteTarget::Url("https://example.com"),
        1,
    )
    .unwrap()
    .unwrap()
}

fn assert_entity(event: &serde_json::Value, entity: &[u8], expected_sha256: &str) {
    assert_eq!(
        event["effective_request_wire_digest"]["sha256"],
        expected_sha256
    );
    assert_eq!(
        event["effective_request_wire_digest"]["byte_count"],
        entity.len()
    );
    // The side-stream worker decodes this exact field; it must receive the entity too.
    assert_eq!(
        hex::decode(event["body_hex"].as_str().unwrap()).unwrap(),
        entity
    );
    assert_eq!(
        event["body"],
        serde_json::from_slice::<serde_json::Value>(entity).unwrap()
    );
}

#[tokio::test]
async fn selected_chunked_entity_ignores_segmentation_extensions_and_trailers() {
    let mut one_chunk = format!("{:x};layout=one\r\n", ENTITY.len()).into_bytes();
    one_chunk.extend_from_slice(ENTITY);
    one_chunk.extend_from_slice(b"\r\n0\r\nX-Receipt: first\r\n\r\n");
    let mut byte_chunks = Vec::new();
    for byte in ENTITY {
        byte_chunks.extend_from_slice(b"1;layout=byte\r\n");
        byte_chunks.push(*byte);
        byte_chunks.extend_from_slice(b"\r\n");
    }
    byte_chunks.extend_from_slice(b"0;last=yes\r\nX-Receipt: different\r\n\r\n");
    for path in ["/v1/chat/completions", "/v1/completions"] {
        for chunks in [&one_chunk, &byte_chunks] {
            let request = read_request(path, chunks, true).await;
            let forwarding = request.raw.clone();
            assert!(forwarding.ends_with(chunks));
            assert_entity(&event(&request), ENTITY, ENTITY_SHA256);
            assert_eq!(
                request.raw, forwarding,
                "observation must not rewrite forwarding"
            );
            assert_eq!(request.body_bytes.as_deref(), Some(ENTITY));
        }
    }
}

#[tokio::test]
async fn selected_entity_tracks_model_rewrite_without_reusing_original_bytes() {
    let mut request = read_request("/v1/chat/completions", ENTITY, false).await;
    assert!(matches!(
        request.effective_http_entity().unwrap(),
        Cow::Borrowed(_)
    ));
    super::super::request_parse::rewrite_model_field(&mut request, "new");
    let transformed = br#"{"messages":[],"model":"new"}"#;
    assert_entity(
        &event(&request),
        transformed,
        "3c437116811c2cd6bb0a02f4fc8550c23c6e0fa2f2320ada0bd36b999f3821cd",
    );
    assert!(request.raw.ends_with(transformed));
}

#[tokio::test]
async fn prepared_unchunked_entity_can_exceed_original_ingress_limit_without_copying() {
    let mut request = read_request("/v1/chat/completions", ENTITY, false).await;
    let expanded_len = super::super::request_parse::MAX_BODY_BYTES + 1;
    request.raw =
        format!("POST /v1/chat/completions HTTP/1.1\r\nContent-Length: {expanded_len}\r\n\r\n")
            .into_bytes();
    let entity_start = request.raw.len();
    request.raw.resize(entity_start + expanded_len, b'x');
    let Cow::Borrowed(entity) = request.effective_http_entity().unwrap() else {
        panic!("prepared unchunked evidence must borrow existing storage");
    };
    assert_eq!(entity.len(), expanded_len);
    assert_eq!(entity.as_ptr(), request.raw[entity_start..].as_ptr());
    assert_eq!(entity.last(), Some(&b'x'));
}

#[tokio::test]
async fn incomplete_effective_chunked_entity_cannot_silently_skip_admission() {
    let mut request = read_request("/v1/chat/completions", ENTITY, false).await;
    request.raw =
        b"POST /v1/chat/completions HTTP/1.1\r\nTransfer-Encoding: chunked\r\n\r\n1\r\nx".to_vec();
    assert!(
        selected_route_event(
            &request,
            Some("m"),
            "external",
            SelectedRouteTarget::MeshLabel("local"),
            1,
        )
        .is_err()
    );
}

#[tokio::test]
async fn accepted_header_limit_allows_host_added_correlation_headers() {
    let limit = super::super::request_parse::MAX_HEADERS;
    let mut wire = format!(
        "POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: {}\r\n",
        ENTITY.len()
    ).into_bytes();
    for index in 0..limit - 3 {
        wire.extend_from_slice(format!("X-Fixture-{index}: accepted\r\n").as_bytes());
    }
    wire.extend_from_slice(b"\r\n");
    wire.extend_from_slice(ENTITY);
    let (mut reader, mut writer) = tokio::io::duplex(wire.len() + 1);
    writer.write_all(&wire).await.unwrap();
    let request = transport::read_http_request(&mut reader).await.unwrap();
    assert!(
        request
            .raw
            .windows(13)
            .any(|bytes| bytes == b"x-request-id:")
    );
    assert_entity(&event(&request), ENTITY, ENTITY_SHA256);
}
