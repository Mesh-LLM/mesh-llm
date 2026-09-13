use super::*;
use tokio::io::AsyncWriteExt;

async fn read_request(raw: Vec<u8>) -> BufferedHttpRequest {
    let (mut writer, mut reader) = tokio::io::duplex(4096);
    let write = tokio::spawn(async move {
        writer.write_all(&raw).await.unwrap();
    });
    let request = crate::network::openai::request_parse::read_http_request(&mut reader)
        .await
        .expect("valid request");
    write.await.unwrap();
    request
}

fn multipart(model: &str) -> Vec<u8> {
    [
        b"--audio\r\nContent-Disposition: form-data; name=\"file\"; filename=\"audio.wav\"\r\nContent-Type: audio/wav\r\n\r\nRIFF\0{\xff}\x80\r\n--audio\r\nContent-Disposition: form-data; name=\"model\"\r\n\r\n".as_slice(),
        model.as_bytes(), b"\r\n--audio--\r\n",
    ].concat()
}

fn raw_request(path: &str, content_type: &str, body: &[u8], chunked: bool) -> Vec<u8> {
    let framing = if chunked {
        "Transfer-Encoding: chunked\r\nTrailer: X-Checksum".into()
    } else {
        format!("Content-Length: {}", body.len())
    };
    let mut raw = format!("POST {path} HTTP/1.1\r\nHost: localhost\r\nContent-Type: {content_type}\r\n{framing}\r\n\r\n").into_bytes();
    if chunked {
        for chunk in body.chunks(7) {
            raw.extend_from_slice(format!("{:x};test=1\r\n", chunk.len()).as_bytes());
            raw.extend_from_slice(chunk);
            raw.extend_from_slice(b"\r\n");
        }
        raw.extend_from_slice(b"0\r\nX-Checksum: fixture\r\n\r\n");
    } else {
        raw.extend_from_slice(body);
    }
    raw
}

#[tokio::test]
async fn automatic_audio_rewrite_preserves_binary_bytes_and_replaces_chunk_framing() {
    for path in ["/v1/audio/transcriptions", "/v1/audio/translations?trace=1"] {
        for chunked in [false, true] {
            let raw = raw_request(
                path,
                "multipart/form-data; boundary=audio",
                &multipart("auto"),
                chunked,
            );
            let mut request = read_request(raw).await;
            assert_eq!(request.model_name.as_deref(), Some("auto"));
            // Host ingress injects first; the passive path injects after rewriting.
            let original = request.raw.clone();
            inject_mesh_hooks_flag(&mut request.raw, true);
            assert_eq!(
                request.raw, original,
                "hooks must not modify audio or framing"
            );
            rewrite_model_field(&mut request, "audio-model");
            inject_mesh_hooks_flag(&mut request.raw, true);
            let headers = body_headers(&request.raw).unwrap();
            assert!(!headers.chunked);
            let body = &request.raw[headers.end..];
            assert_eq!(body, multipart("audio-model"));
            assert_eq!(request.body_bytes.as_deref(), Some(body));
            assert_eq!(request.body_len_bytes, body.len());
            assert!(request.body_json.is_none());
            let header_text = std::str::from_utf8(&request.raw[..headers.end])
                .unwrap()
                .to_lowercase();
            assert_eq!(header_text.matches("content-length:").count(), 1);
            assert!(header_text.contains(&format!("content-length: {}\r\n", body.len())));
            assert!(!header_text.contains("trailer:"));
        }
    }
}

#[test]
fn hook_injection_rejects_non_json_and_sets_one_valid_flag() {
    for (content_type, body) in [
        ("application/octet-stream", b"{\"binary\":true}".as_slice()),
        ("application/json", b"prefix{\"invalid\":true}".as_slice()),
        ("application/json", b"[{}]".as_slice()),
    ] {
        let mut raw = raw_request("/v1/chat/completions", content_type, body, false);
        let before = raw.clone();
        inject_mesh_hooks_flag(&mut raw, true);
        assert_eq!(raw, before);
    }
    for body in [
        b"{}".as_slice(),
        b"{\"mesh_hooks\":false,\"model\":\"auto\"}".as_slice(),
    ] {
        for chunked in [false, true] {
            let mut raw = raw_request(
                "/v1/chat/completions",
                "application/json; charset=utf-8",
                body,
                chunked,
            );
            inject_mesh_hooks_flag(&mut raw, true);
            let headers = body_headers(&raw).unwrap();
            assert!(!headers.chunked);
            let json: serde_json::Value = serde_json::from_slice(&raw[headers.end..]).unwrap();
            assert_eq!(json["mesh_hooks"], true);
            assert_eq!(
                String::from_utf8_lossy(&raw[headers.end..])
                    .matches("mesh_hooks")
                    .count(),
                1
            );
        }
    }
}
