//! Exercise real HTTP chunk boundaries inside streamed JSON image content.
use super::*;
use base64::Engine;
use serde_json::{Value, json};
use std::io::Cursor;
use tokio::net::{TcpListener, TcpStream};

fn image_text() -> String {
    let mut state = 0x1234_5678_u32;
    let pixels = image::RgbImage::from_fn(384, 384, |_, _| {
        image::Rgb(std::array::from_fn(|_| {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            state as u8
        }))
    });
    let mut png = Cursor::new(Vec::new());
    image::DynamicImage::ImageRgb8(pixels)
        .write_to(&mut png, image::ImageFormat::Png)
        .unwrap();
    let encoded = base64::engine::general_purpose::STANDARD.encode(png.into_inner());
    assert!(encoded.len() > 500_000);
    format!(
        "Here is the generated image: 猫.\n\n![Generated image](data:image/png;base64,{encoded})"
    )
}

fn chunked_chat_body(text: &str) -> Vec<u8> {
    let mut sse = String::new();
    // Keep chunks on character boundaries; this fixture's only Unicode is in
    // its short caption. The later HTTP split deliberately separates it.
    let mut start = 0;
    while start < text.len() {
        let end = (start + 8192).min(text.len());
        assert!(text.is_char_boundary(end));
        let event = json!({"id":"chat1","object":"chat.completion.chunk","model":"noise","choices":[
            {"index":0,"delta":{"content":&text[start..end]},"finish_reason":null}
        ]});
        sse.push_str(&format!("data: {event}\n\n"));
        start = end;
    }
    sse.push_str("data: {\"id\":\"chat1\",\"model\":\"noise\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}]}\n\ndata: [DONE]\n\n");
    let first = sse.find('猫').unwrap() + 1;
    let mut wire = Vec::new();
    for chunk in
        std::iter::once(&sse.as_bytes()[..first]).chain(sse.as_bytes()[first..].chunks(4093))
    {
        wire.extend_from_slice(format!("{:x}\r\n", chunk.len()).as_bytes());
        wire.extend_from_slice(chunk);
        wire.extend_from_slice(b"\r\n");
    }
    wire.extend_from_slice(b"0\r\n\r\n");
    wire
}

async fn assert_chunked_image_preserved(responses: bool) {
    let expected = image_text();
    let upstream = chunked_chat_body(&expected);
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let mut client = TcpStream::connect(listener.local_addr().unwrap())
        .await
        .unwrap();
    let (socket, _) = listener.accept().await.unwrap();
    let task = tokio::spawn(async move {
        let header = b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nTransfer-Encoding: chunked\r\n\r\n";
        let probe = ResponseProbe {
            buffered: header.to_vec(),
            header_end: header.len(),
            status_code: 200,
            retryable_context_overflow: false,
        };
        let mut downstream: ClientStream = socket.into();
        let mut upstream = Cursor::new(upstream);
        if responses {
            relay_translated_responses_stream(
                &mut downstream,
                &mut upstream,
                probe,
                ResponseRetryPolicy::next_target_available(false),
                None,
                OpenAiRouteObserver::default(),
                RequestId::new(),
            )
            .await
        } else {
            relay_normalized_chat_completion_stream(
                &mut downstream,
                &mut upstream,
                probe,
                ResponseRetryPolicy::next_target_available(false),
                None,
                OpenAiRouteObserver::default(),
                RequestId::new(),
            )
            .await
        }
    });
    let mut wire = Vec::new();
    client.read_to_end(&mut wire).await.unwrap();
    let result = task.await.unwrap();
    assert!(
        matches!(result, Ok(RouteAttemptResult::Delivered { .. })),
        "relay failed: {result:?}"
    );
    let head = wire
        .windows(4)
        .position(|window| window == b"\r\n\r\n")
        .unwrap()
        + 4;
    let mut reader = super::super::body_reader::BodyReader::new(
        Cursor::new(Vec::new()),
        wire[head..].to_vec(),
        true,
        None,
    );
    let mut body = Vec::new();
    while let Some(bytes) = reader.next().await.unwrap() {
        body.extend(bytes);
    }
    let body = std::str::from_utf8(&body).unwrap();
    let image_count = crate::inference::image_safety::validated_image_count_for_test(
        "text/event-stream",
        body.as_bytes(),
    )
    .expect("the actual safety scanner must accept the relayed image stream");
    assert_eq!(image_count, 1);
    let mut actual = String::new();
    for frame in body.split("\n\n") {
        let Some(data) = frame.lines().find_map(|line| line.strip_prefix("data: ")) else {
            continue;
        };
        if data == "[DONE]" {
            continue;
        }
        let event: Value = serde_json::from_str(data).expect("relay must emit valid JSON frames");
        let text = if responses {
            (event["type"] == "response.output_text.delta")
                .then(|| event["delta"].as_str())
                .flatten()
        } else {
            event
                .pointer("/choices/0/delta/content")
                .and_then(Value::as_str)
        };
        if let Some(text) = text {
            actual.push_str(text);
        }
    }
    assert!(
        actual == expected,
        "HTTP chunk framing or split UTF-8 changed the image text"
    );
    assert!(body.contains("data: [DONE]\n\n"));
}

#[tokio::test]
async fn real_chunked_chat_image_preserves_every_delta_and_utf8() {
    assert_chunked_image_preserved(false).await;
}

#[tokio::test]
async fn real_chunked_chat_to_responses_image_preserves_every_delta_and_utf8() {
    assert_chunked_image_preserved(true).await;
}
