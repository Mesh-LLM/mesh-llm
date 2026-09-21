//! Multipart routing and byte-preserving request regression tests.

use super::super::audio_multipart::{multipart_boundary, multipart_part_is_model};
use super::*;

/// Routing and rewriting must not choose different multipart boundaries.
#[tokio::test]
async fn duplicate_content_type_is_rejected_before_body_routing() {
    for second in ["multipart/form-data; boundary=other", "application/json"] {
        let request = format!(
            "POST /v1/audio/transcriptions HTTP/1.1\r\nContent-Type: multipart/form-data; boundary=mesh\r\ncOnTeNt-TyPe: {second}\r\nContent-Length: 0\r\n\r\n"
        );
        let (mut client, mut server) = tokio::io::duplex(request.len() + 1);
        client.write_all(request.as_bytes()).await.unwrap();
        client.shutdown().await.unwrap();
        let error = read_http_request_with_plugin_manager_with_context(&mut server, None)
            .await
            .unwrap_err();
        assert!(error.to_string().contains("duplicate Content-Type"));
    }
}

#[tokio::test]
/// Model routing must never decode or rewrite the uploaded audio bytes.
async fn multipart_model_is_parsed_and_rewritten_without_touching_file_bytes() {
    const BOUNDARY: &str = "mesh-audio-boundary";
    // A boundary prefix inside binary content is not a multipart delimiter.
    let file_bytes = [
        0_u8, 255, 13, 10, b'-', b'-', b'm', b'e', b's', b'h', b'-', b'a', b'u', b'd', b'i', b'o',
        b'-', b'b', b'o', b'u', b'n', b'd', b'a', b'r', b'y', b'X', 13, 10, 1, 2, 3, 128,
    ];
    let mut body = format!(
        "--{BOUNDARY}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"voice; name=model\"\r\nContent-Type: audio/wav\r\n\r\n"
    )
    .into_bytes();
    body.extend_from_slice(&file_bytes);
    body.extend_from_slice(
        format!(
            "\r\n--{BOUNDARY}\r\nContent-Disposition: form-data; name=\"model\"\r\n\r\nauto\r\n--{BOUNDARY}--\r\n"
        )
        .as_bytes(),
    );
    let headers = format!(
        "POST /v1/audio/transcriptions HTTP/1.1\r\nHost: localhost\r\nContent-Type: multipart/form-data; boundary=\"{BOUNDARY}\"\r\nContent-Length: {}\r\n\r\n",
        body.len()
    )
    .into_bytes();

    let mut request = read_request_from_parts(vec![headers, body]).await;
    assert_eq!(request.model_name.as_deref(), Some("auto"));

    rewrite_model_field(&mut request, "whisper-local");

    assert_eq!(request.model_name.as_deref(), Some("whisper-local"));
    assert!(
        request
            .raw
            .windows(file_bytes.len())
            .any(|window| window == file_bytes)
    );
    let header_end = request
        .raw
        .windows(4)
        .position(|window| window == b"\r\n\r\n")
        .unwrap()
        + 4;
    let content_type = format!("multipart/form-data; boundary={BOUNDARY}");
    assert_eq!(
        multipart_model_field(&content_type, &request.raw[header_end..])
            .unwrap()
            .as_deref(),
        Some("whisper-local")
    );
    let declared = std::str::from_utf8(&request.raw[..header_end])
        .unwrap()
        .lines()
        .find_map(|line| {
            line.split_once(':')
                .filter(|(name, _)| name.eq_ignore_ascii_case("content-length"))
                .and_then(|(_, value)| value.trim().parse::<usize>().ok())
        })
        .unwrap();
    assert_eq!(declared, request.raw.len() - header_end);
    assert_eq!(declared, request.body_len_bytes);
}

#[test]
/// Reject ambiguous framing and unbounded model identifiers before route selection.
fn multipart_parser_rejects_invalid_boundaries_and_oversized_model_values() {
    assert!(multipart_boundary("multipart/form-data; boundary=bad space").is_none());
    assert!(multipart_boundary("application/json; boundary=mesh").is_none());

    let boundary = "mesh";
    let body = format!(
        "--{boundary}\r\nContent-Disposition: form-data; name=\"model\"\r\n\r\n{}\r\n--{boundary}--\r\n",
        "x".repeat(257)
    );
    let error = multipart_model_field(
        &format!("multipart/form-data; boundary={boundary}"),
        body.as_bytes(),
    )
    .unwrap_err();
    assert!(error.to_string().contains("256-byte limit"));
}

/// The identifier ceiling counts UTF-8 bytes; absent and blank fields stay optional.
#[test]
fn multipart_model_limit_preserves_valid_and_missing_values() {
    let content_type = "multipart/form-data; boundary=mesh";
    for value in ["x".repeat(256), "é".repeat(128), " auto ".to_string()] {
        let body = format!(
            "--mesh\r\nContent-Disposition: form-data; name=\"model\"\r\n\r\n{value}\r\n--mesh--\r\n"
        );
        assert_eq!(
            multipart_model_field(content_type, body.as_bytes()).unwrap(),
            Some(value.trim().to_string())
        );
    }
    for body in [
        b"--mesh--\r\n".as_slice(),
        b"--mesh\r\nContent-Disposition: form-data; name=\"model\"\r\n\r\n \t\r\n--mesh--\r\n",
        b"--mesh\r\nContent-Disposition: form-data; name=\"file\"; filename=\"audio.wav\"\r\n\r\nWAVE\r\n--mesh--\r\n",
    ] {
        assert_eq!(multipart_model_field(content_type, body).unwrap(), None);
    }
}

/// Both audio endpoints reject overlong explicit destinations instead of routing automatically.
#[tokio::test]
async fn oversized_multipart_model_is_rejected_before_audio_routing() {
    for path in ["/v1/audio/transcriptions", "/v1/audio/translations"] {
        for value in ["x".repeat(257), "é".repeat(129)] {
            let body = format!(
                "--mesh\r\nContent-Disposition: form-data; name=\"model\"\r\n\r\n{value}\r\n--mesh--\r\n"
            );
            let request = format!(
                "POST {path} HTTP/1.1\r\nContent-Type: multipart/form-data; boundary=mesh\r\nContent-Length: {}\r\n\r\n{body}",
                body.len()
            );
            let (mut client, mut server) = tokio::io::duplex(request.len() + 1);
            client.write_all(request.as_bytes()).await.unwrap();
            client.shutdown().await.unwrap();
            let error = read_http_request_with_plugin_manager_with_context(&mut server, None)
                .await
                .unwrap_err();
            assert_eq!(error.context().unwrap().client_path, path);
            assert!(error.to_string().contains("256-byte limit"));
        }
    }
}

#[tokio::test]
/// Two model fields cannot disagree about the destination of one upload.
async fn duplicate_multipart_model_is_rejected_before_audio_routing() {
    let boundary = "mesh-audio-boundary";
    let body = format!(
        "--{boundary}\r\nContent-Disposition: form-data; name=\"model\"\r\n\r\nauto\r\n\
         --{boundary}\r\nContent-Disposition: form-data; name=\"model\"\r\n\r\nother-model\r\n\
         --{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"voice.wav\"\r\n\r\nWAVE\r\n\
         --{boundary}--\r\n"
    );
    let content_type = format!("multipart/form-data; boundary={boundary}");
    assert!(
        multipart_model_field(&content_type, body.as_bytes())
            .unwrap_err()
            .to_string()
            .contains("duplicate multipart model field")
    );

    for path in ["/v1/audio/transcriptions", "/v1/audio/translations"] {
        let request = format!(
            "POST {path} HTTP/1.1\r\nHost: localhost\r\nContent-Type: {content_type}\r\nContent-Length: {}\r\n\r\n{body}",
            body.len()
        );
        let (mut client, mut server) = tokio::io::duplex(request.len() + 1);
        client.write_all(request.as_bytes()).await.unwrap();
        client.shutdown().await.unwrap();
        let error = read_http_request_with_plugin_manager_with_context(&mut server, None)
            .await
            .unwrap_err();
        assert_eq!(error.context().unwrap().client_path, path);
        assert!(
            error
                .to_string()
                .contains("duplicate multipart model field")
        );
    }
}

#[test]
/// A boundary-like sequence inside payload data is not a valid multipart start.
fn multipart_model_scanner_rejects_non_initial_boundary() {
    let body = b"binary--mesh\r\nContent-Disposition: form-data; name=\"model\"\r\n\r\nauto\r\n--mesh--\r\n";
    let error = multipart_model_field("multipart/form-data; boundary=mesh", body).unwrap_err();
    assert!(
        error
            .to_string()
            .contains("must start with its declared boundary")
    );
}

#[test]
/// Quoted filenames cannot impersonate the disposition's model-field parameter.
fn multipart_disposition_ignores_name_like_text_inside_quoted_filename() {
    assert!(
        !multipart_part_is_model(
            "Content-Disposition: form-data; name=\"file\"; filename=\"voice; name=model\""
        )
        .unwrap()
    );
    assert!(multipart_part_is_model("Content-Disposition: form-data; name=\"model\"").unwrap());
    assert!(
        multipart_part_is_model("Content-Disposition: form-data; name=\"file\"; name=\"model\"")
            .is_err()
    );
}
