use super::*;
use tokio::io::{AsyncReadExt, AsyncWriteExt};

#[tokio::test]
async fn multipart_upload_preserves_binary_and_provider_error_response() {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let document = b"%PDF-1.7\0\xff\r\nexact binary bytes".to_vec();
    let expected = document.clone();
    let server = tokio::spawn(async move {
        let (mut stream, _) = listener.accept().await.unwrap();
        let (headers, body) = read_upload(&mut stream).await;
        assert!(headers.starts_with("POST /v1/documents/ocr HTTP/1.1\r\n"));
        assert!(
            headers
                .to_ascii_lowercase()
                .contains("authorization: bearer secret-key\r\n")
        );
        let content_type = headers
            .lines()
            .find(|line| line.to_ascii_lowercase().starts_with("content-type:"))
            .unwrap();
        assert!(content_type.contains("multipart/form-data; boundary="));
        let part_headers = String::from_utf8_lossy(&body);
        assert!(part_headers.contains("name=\"file\"; filename=\"scan.pdf\""));
        assert!(part_headers.contains("Content-Type: application/pdf\r\n"));
        assert!(
            body.windows(expected.len())
                .any(|window| window == expected)
        );
        for field in [
            "name=\"model\"\r\n\r\norg/ocr@revision#profile\r\n",
            "name=\"response_format\"\r\n\r\nmarkdown\r\n",
            "name=\"max_pages\"\r\n\r\n3\r\n",
        ] {
            assert!(part_headers.contains(field), "missing field: {field}");
        }
        let body = "# Provider rejected page 2\n";
        stream
            .write_all(
                format!(
                    "HTTP/1.1 422 Unprocessable Entity\r\nContent-Type: text/markdown\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                    body.len()
                )
                .as_bytes(),
            )
            .await
            .unwrap();
    });
    let client = OpenAiClient::new(format!("http://{address}/v1")).with_api_key("secret-key");
    let response = client
        .document_ocr_raw(
            document,
            DocumentOcrOptions {
                model: "org/ocr@revision#profile".into(),
                filename: "scan.pdf".into(),
                content_type: "application/pdf".into(),
                response_format: Some("markdown".into()),
                max_pages: Some(3),
            },
        )
        .await
        .unwrap();
    assert_eq!(response.status_code, 422);
    assert_eq!(response.content_type.as_deref(), Some("text/markdown"));
    assert_eq!(response.body, "# Provider rejected page 2\n");
    let error = response.error_for_status().err().unwrap();
    assert!(error.to_string().contains("422"));
    assert!(error.to_string().contains("Provider rejected page 2"));
    server.await.unwrap();
}

#[test]
fn raw_response_status_check_preserves_success_body() {
    let response = RawOpenAiResponse {
        status_code: 201,
        content_type: Some("application/json".into()),
        body: "{\"pages\":[]}".into(),
    }
    .error_for_status()
    .unwrap();
    assert_eq!(response.body, "{\"pages\":[]}");
}

#[tokio::test]
async fn checked_ocr_reports_provider_status_and_body() {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let server = tokio::spawn(async move {
        let (mut stream, _) = listener.accept().await.unwrap();
        let _ = read_upload(&mut stream).await;
        let body = r#"{"error":{"message":"OCR provider unavailable"}}"#;
        stream.write_all(format!(
            "HTTP/1.1 503 Service Unavailable\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
            body.len(),
        ).as_bytes()).await.unwrap();
    });
    let error = OpenAiClient::new(format!("http://{address}/v1"))
        .document_ocr(vec![1, 2, 3], DocumentOcrOptions::new("ocr"))
        .await
        .err()
        .unwrap();
    assert!(error.to_string().contains("503"));
    assert!(error.to_string().contains("OCR provider unavailable"));
    server.await.unwrap();
}

async fn read_upload(stream: &mut tokio::net::TcpStream) -> (String, Vec<u8>) {
    let mut bytes = Vec::new();
    let header_end = loop {
        let mut chunk = [0_u8; 1024];
        let count = tokio::time::timeout(Duration::from_secs(5), stream.read(&mut chunk))
            .await
            .unwrap()
            .unwrap();
        assert!(count > 0);
        bytes.extend_from_slice(&chunk[..count]);
        if let Some(end) = bytes.windows(4).position(|window| window == b"\r\n\r\n") {
            break end + 4;
        }
        assert!(bytes.len() < 64 * 1024);
    };
    let headers = String::from_utf8(bytes[..header_end].to_vec()).unwrap();
    let length = headers
        .lines()
        .find_map(|line| {
            line.to_ascii_lowercase()
                .strip_prefix("content-length:")
                .map(|value| value.trim().parse::<usize>().unwrap())
        })
        .unwrap();
    assert!(length < 16 * 1024);
    while bytes.len() < header_end + length {
        let mut chunk = [0_u8; 1024];
        let count = tokio::time::timeout(Duration::from_secs(5), stream.read(&mut chunk))
            .await
            .unwrap()
            .unwrap();
        assert!(count > 0);
        bytes.extend_from_slice(&chunk[..count]);
    }
    (headers, bytes[header_end..header_end + length].to_vec())
}

#[test]
fn options_defaults_and_validation_match_upload_contract() {
    let mut options: DocumentOcrOptions = serde_json::from_str(r#"{"model":"ocr"}"#).unwrap();
    assert_eq!(options.filename, "document");
    assert_eq!(options.content_type, "application/octet-stream");
    assert!(options.validate(MAX_DOCUMENT_BYTES).is_ok());
    assert!(options.validate(0).is_err());
    assert!(options.validate(MAX_DOCUMENT_BYTES + 1).is_err());
    options.max_pages = Some(0);
    assert!(options.validate(1).is_err());
    options.max_pages = None;
    options.response_format = Some("html".into());
    assert!(options.validate(1).is_err());
    options.response_format = None;
    options.model = "  ".into();
    assert!(options.validate(1).is_err());
}

#[tokio::test]
async fn invalid_options_and_mime_fail_before_network_io() {
    let client = OpenAiClient::new("http://127.0.0.1:1/v1");
    let error = client
        .document_ocr(Vec::new(), DocumentOcrOptions::new("ocr"))
        .await
        .err()
        .unwrap();
    assert!(error.to_string().contains("empty"));
    let mut options = DocumentOcrOptions::new("ocr");
    options.content_type = "invalid\r\nheader".into();
    assert!(client.document_ocr(vec![1], options).await.is_err());
}
