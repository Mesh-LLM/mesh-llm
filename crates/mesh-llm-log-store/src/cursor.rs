//! Versioned opaque cursors for keyset pagination.

use crate::error::LogStoreError;

/// Cursor version tag (must be bumped when encoding changes).
const CURSOR_VERSION: u8 = 1;

/// Encode a cursor string from `(occurred_at, id)` position.
pub fn encode_cursor(occurred_at: &str, id: &str) -> String {
    let payload = format!("{}|{}", occurred_at, id);
    // Version byte + base64 of the payload.
    let encoded = data_encoding::BASE64.encode(payload.as_bytes());
    format!("v{CURSOR_VERSION}:{encoded}")
}

/// Decode a cursor string into `(occurred_at, id)`.
pub fn decode_cursor(cursor: &str) -> Result<(String, String), LogStoreError> {
    let parts: Vec<&str> = cursor.splitn(2, ':').collect();
    if parts.len() != 2 {
        return Err(LogStoreError::CursorMalformed(format!(
            "expected version:data, got: {}",
            cursor
        )));
    }

    let (version_str, encoded) = (parts[0], parts[1]);

    match version_str {
        "v1" => {} // supported
        other => {
            return Err(LogStoreError::CursorMalformed(format!(
                "unknown cursor version: {}",
                other
            )));
        }
    }

    let decoded_bytes = data_encoding::BASE64
        .decode(encoded.as_bytes())
        .map_err(|e| LogStoreError::CursorMalformed(format!("base64 decode failed: {}", e)))?;

    let payload_str = String::from_utf8(decoded_bytes).map_err(|_| {
        LogStoreError::CursorMalformed("cursor payload is not valid UTF-8".to_string())
    })?;

    let pipe_parts: Vec<&str> = payload_str.splitn(2, '|').collect();
    if pipe_parts.len() != 2 {
        return Err(LogStoreError::CursorMalformed(format!(
            "expected occurred_at|id, got: {}",
            payload_str
        )));
    }

    Ok((pipe_parts[0].to_string(), pipe_parts[1].to_string()))
}
