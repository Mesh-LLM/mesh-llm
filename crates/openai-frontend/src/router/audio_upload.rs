//! Multipart audio ingestion with a file limit enforced during streaming.

use super::multipart_error;
use crate::{AudioTranscriptionRequest, OpenAiError, OpenAiResult};
use axum::extract::{Multipart, multipart::Field};

/// Decode a bounded audio upload, rejecting duplicate recognized fields consistently.
pub(super) async fn parse_audio_multipart(
    mut multipart: Multipart,
) -> OpenAiResult<AudioTranscriptionRequest> {
    let mut model = None;
    let mut file = None;
    let mut filename = None;
    let mut language = None;
    let mut prompt = None;
    let mut response_format = None;
    let mut temperature = None;
    let mut seen_fields = std::collections::HashSet::new();

    while let Some(field) = multipart
        .next_field()
        .await
        .map_err(|error| multipart_error(error, "multipart body"))?
    {
        let name = field.name().unwrap_or_default().to_string();
        if matches!(
            name.as_str(),
            "model" | "file" | "language" | "prompt" | "response_format" | "temperature"
        ) && !seen_fields.insert(name.clone())
        {
            return Err(OpenAiError::invalid_request(format!(
                "duplicate multipart {name} field"
            )));
        }
        if name == "file" {
            filename = field.file_name().map(str::to_owned);
            file = Some(read_audio_file(field, AudioTranscriptionRequest::MAX_FILE_BYTES).await?);
            continue;
        }
        let value = field
            .text()
            .await
            .map_err(|error| multipart_error(error, "multipart text field"))?;
        match name.as_str() {
            "model" => model = Some(value),
            "language" => language = Some(value),
            "prompt" => prompt = Some(value),
            "response_format" => response_format = Some(value),
            "temperature" => {
                temperature =
                    Some(value.parse::<f32>().map_err(|_| {
                        OpenAiError::invalid_request("temperature must be a number")
                    })?);
            }
            _ => {}
        }
    }

    Ok(AudioTranscriptionRequest {
        model: model.ok_or_else(|| OpenAiError::invalid_request("model field is required"))?,
        file: file.ok_or_else(|| OpenAiError::invalid_request("file field is required"))?,
        filename,
        language,
        prompt,
        response_format: response_format.unwrap_or_else(|| "json".to_string()),
        temperature,
    })
}

/// Stop consuming the upload before copying a chunk that exceeds the file budget.
async fn read_audio_file(mut field: Field<'_>, limit: usize) -> OpenAiResult<Vec<u8>> {
    let mut file = Vec::new();
    while let Some(chunk) = field
        .chunk()
        .await
        .map_err(|error| multipart_error(error, "audio file field"))?
    {
        if chunk.len() > limit - file.len() {
            return Err(OpenAiError::payload_too_large(format!(
                "audio file exceeds the {limit} byte limit"
            )));
        }
        file.extend_from_slice(&chunk);
    }
    Ok(file)
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::{
        body::{Body, Bytes},
        extract::FromRequest,
        http::Request,
    };
    use futures_util::{StreamExt, stream};
    use std::{
        convert::Infallible,
        sync::{
            Arc,
            atomic::{AtomicUsize, Ordering},
        },
    };

    /// Exact-limit files survive; an over-limit stream is rejected before it is exhausted.
    #[tokio::test]
    async fn file_limit_is_applied_while_streaming() {
        for (chunks, limit, accepted) in [(2, 8, true), (100, 8, false)] {
            let reads = Arc::new(AtomicUsize::new(0));
            let seen = reads.clone();
            let payload =
                stream::iter((0..chunks).map(|_| Ok::<_, Infallible>(Bytes::from_static(b"data"))))
                    .then(|chunk| async {
                        // Model separately arriving network chunks. A fully ready
                        // in-memory stream is eagerly buffered by the multipart parser.
                        tokio::task::yield_now().await;
                        chunk
                    })
                    .inspect(move |_| {
                        seen.fetch_add(1, Ordering::SeqCst);
                    });
            let body = stream::once(async {
                Ok::<_, Infallible>(Bytes::from_static(
                    b"--mesh\r\nContent-Disposition: form-data; name=\"file\"\r\n\r\n",
                ))
            })
            .chain(payload)
            .chain(stream::once(async {
                Ok(Bytes::from_static(b"\r\n--mesh--\r\n"))
            }));
            let request = Request::builder()
                .header("content-type", "multipart/form-data; boundary=mesh")
                .body(Body::from_stream(body))
                .unwrap();
            let mut multipart = Multipart::from_request(request, &()).await.unwrap();
            let field = multipart.next_field().await.unwrap().unwrap();
            let result = read_audio_file(field, limit).await;
            if accepted {
                assert_eq!(result.unwrap(), b"datadata");
            } else {
                assert!(result.unwrap_err().to_string().contains("exceeds"));
                assert!(reads.load(Ordering::SeqCst) < chunks);
            }
        }
    }
}
