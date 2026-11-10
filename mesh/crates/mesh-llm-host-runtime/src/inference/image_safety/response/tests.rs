use super::*;
use std::collections::BTreeSet;

fn response(headers: &str, body: &[u8]) -> Vec<u8> {
    let mut wire =
        format!("HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\n{headers}\r\n").into_bytes();
    wire.extend_from_slice(body);
    wire
}

#[test]
fn exact_content_length_retains_every_entity_byte() {
    let body = b"data: {\"choices\":[]}\n\ndata: [DONE]\n\n";
    let wire = response(&format!("Content-Length: {}\r\n", body.len()), body);
    let declared_length = body.len();
    let (content_type, decoded) = entity(&wire).unwrap();
    assert_eq!(content_type, "text/event-stream");
    assert_eq!(decoded, body);
    for body in [
        &body[..body.len() - 1],
        &[body.as_slice(), b"extra"].concat(),
    ] {
        assert!(
            entity(&response(
                &format!("Content-Length: {declared_length}\r\n"),
                body
            ))
            .is_err()
        );
    }
}

#[test]
fn rejects_redirects_even_when_the_body_is_a_valid_image_free_error() {
    let body = br#"{"error":{"message":"moved"}}"#;
    for code in 300..=399 {
        let mut wire = format!(
            "HTTP/1.1 {code} Redirect\r\nLocation: https://example.invalid/image.png\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n",
            body.len()
        )
        .into_bytes();
        wire.extend_from_slice(body);
        assert!(entity(&wire).is_err(), "redirect {code} was admitted");
    }
}

#[test]
fn dechunking_preserves_split_sse_and_accepts_bounded_trailers() {
    let body = b"data: {\"choices\":[]}\n\ndata: [DONE]\n\n";
    let split = 13;
    let mut chunks = format!("{:x};part=first;quoted=\"a;b\\\"c\"\r\n", split).into_bytes();
    chunks.extend_from_slice(&body[..split]);
    chunks.extend_from_slice(format!("\r\n{:x}\r\n", body.len() - split).as_bytes());
    chunks.extend_from_slice(&body[split..]);
    chunks.extend_from_slice(b"\r\n0\r\nX-Receipt: benign\r\n\r\n");
    let wire = response(
        "Transfer-Encoding: chunked\r\nTrailer: X-Receipt\r\n",
        &chunks,
    );
    let (_, decoded) = entity(&wire).unwrap();
    assert_eq!(decoded, body);
    assert!(
        artifacts::images_from_response("text/event-stream", &decoded)
            .unwrap()
            .is_empty()
    );
}

#[test]
fn rejects_ambiguous_duplicate_and_unsupported_framing() {
    for headers in [
        "Content-Length: 0\r\ncontent-length: 0\r\n",
        "Content-Length: 0, 0\r\n",
        "Content-Length: +0\r\n",
        "Content-Length: 0\r\nTransfer-Encoding: chunked\r\n",
        "Transfer-Encoding: chunked\r\ntransfer-encoding: chunked\r\n",
        "Transfer-Encoding: gzip, chunked\r\n",
        "Transfer-Encoding: identity\r\n",
        "Content-Length: 0\r\nContent-Encoding: gzip\r\n",
        "Content-Length: 0\r\nContent-Encoding: identity\r\nContent-Encoding: identity\r\n",
        "Content-Length: 0\r\nContent-Type: application/json\r\n",
        "Content-Length: 0\r\nConnection: content-type\r\n",
        "Trailer: X-Receipt\r\nContent-Length: 0\r\n",
        "Transfer-Encoding: chunked\r\nTrailer: Content-Type\r\n",
    ] {
        assert!(
            entity(&response(headers, b"")).is_err(),
            "ambiguous framing was admitted"
        );
    }
}

#[test]
fn rejects_unframed_persistent_response_but_accepts_explicit_close() {
    assert!(entity(&response("", b"hello")).is_err());
    let (_, body) = entity(&response("Connection: close\r\n", b"hello")).unwrap();
    assert_eq!(body, b"hello");
    let wire = b"HTTP/1.0 200 OK\r\nContent-Type: text/plain\r\n\r\nhello";
    assert_eq!(entity(wire).unwrap().1, b"hello");
}

#[test]
fn rejects_incomplete_chunking_missing_terminators_and_trailing_entities() {
    for body in [
        b"1\r\nx\r\n".as_slice(),
        b"1\r\nx\r\n0\r\n",
        b"2\r\nx\r\n0\r\n\r\n",
        b"1\r\nx!\n0\r\n\r\n",
        b"0\r\n\r\nextra",
        b"1\nx\n0\n\n",
        b"+1\r\nx\r\n0\r\n\r\n",
        b"xyz\r\nx\r\n0\r\n\r\n",
        b"0\r\nX-Missing-Value\r\n\r\n",
    ] {
        assert!(entity(&response("Transfer-Encoding: chunked\r\n", body)).is_err());
    }
}

#[test]
fn trailers_cannot_change_framing_or_content_interpretation() {
    for trailer in [
        "Content-Type: image/png",
        "Content-Length: 0",
        "Transfer-Encoding: chunked",
        "Content-Encoding: gzip",
        "Connection: close",
        "Host: example.invalid",
        "X-Receipt: one\r\nx-receipt: two",
        "bad name: value",
        "X-Receipt: bad\u{7f}",
    ] {
        let body = format!("0\r\n{trailer}\r\n\r\n");
        assert!(entity(&response("Transfer-Encoding: chunked\r\n", body.as_bytes())).is_err());
    }
    assert!(
        entity(&response(
            "Transfer-Encoding: chunked\r\nTrailer: X-Receipt\r\n",
            b"0\r\nX-Other: value\r\n\r\n"
        ))
        .is_err()
    );
}

#[test]
fn bounds_decoded_chunks_headers_trailers_and_total_wire() {
    assert!(chunked_body(b"4\r\n", 3, &BTreeSet::new()).is_err());
    let huge_chunk = format!("{:x}\r\n", usize::MAX);
    assert!(chunked_body(huge_chunk.as_bytes(), 3, &BTreeSet::new()).is_err());
    let long_extension = format!("1;{}\r\nx\r\n0\r\n\r\n", "x".repeat(MAX_CHUNK_LINE_BYTES));
    assert!(chunked_body(long_extension.as_bytes(), 3, &BTreeSet::new()).is_err());
    let long_trailer = format!(
        "0\r\nX-Receipt: {}\r\n\r\n",
        "x".repeat(MAX_TRAILER_LINE_BYTES)
    );
    assert!(chunked_body(long_trailer.as_bytes(), 3, &BTreeSet::new()).is_err());
    let headers = format!(
        "X-Long: {}\r\nContent-Length: 0\r\n",
        "x".repeat(MAX_HEADER_BYTES)
    );
    assert!(entity(&response(&headers, b"")).is_err());
    assert!(entity(&vec![b'x'; artifacts::MAX_RESPONSE_BYTES + 1]).is_err());
}

#[test]
fn chunk_extensions_have_strict_token_and_quoted_string_syntax() {
    for line in [
        b"f;foo=bar;quoted=\"a;b\"".as_slice(),
        b"f ; foo = bar",
        b"f;flag",
    ] {
        assert_eq!(chunk_size(line).unwrap(), 15);
    }
    for line in [
        b"f;".as_slice(),
        b"f;=value",
        b"f;name=",
        b"f;name=\"unterminated",
        b"f;name=\"bad\x01\"",
        b"f\n;foo=bar",
    ] {
        assert!(chunk_size(line).is_err());
    }
}

#[test]
fn rejects_noncanonical_header_line_endings_and_provisional_response() {
    for wire in [
        b"HTTP/1.1 200 OK\nContent-Type: text/plain\nContent-Length: 0\n\n".as_slice(),
        b"HTTP/1.1 100 Continue\r\nContent-Type: text/plain\r\nContent-Length: 0\r\n\r\n",
        b"HTTP/1.1 200 OK\r\nContent-Type: text/plain\r\nContent-Length: 0\r\n",
    ] {
        assert!(entity(wire).is_err());
    }
}
