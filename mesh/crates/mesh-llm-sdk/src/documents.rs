//! Binary document OCR requests through the Mesh inference API.

use crate::embedded_node::{OpenAiClient, RawOpenAiResponse};
use anyhow::{Result, ensure};
use reqwest::multipart::{Form, Part};
use serde::{Deserialize, Serialize};
use std::time::Duration;

/// Largest document accepted by the Mesh OCR upload contract (64 MiB).
pub const MAX_DOCUMENT_BYTES: usize = 64 * 1024 * 1024;

/// Fields sent alongside a document's unchanged bytes in a multipart upload.
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct DocumentOcrOptions {
    /// Exact advertised OCR serving ID, including its revision/profile if any.
    pub model: String,
    /// File name reported to the provider; defaults to `document`.
    #[serde(default = "default_filename")]
    pub filename: String,
    /// MIME type reported to the provider; defaults to `application/octet-stream`.
    #[serde(default = "default_content_type")]
    pub content_type: String,
    /// Requested provider output: `json`, `text`, or `markdown`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub response_format: Option<String>,
    /// Optional positive limit on pages processed by the provider.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_pages: Option<u32>,
}

impl DocumentOcrOptions {
    /// Select an advertised OCR model and use the default upload metadata.
    pub fn new(model: impl Into<String>) -> Self {
        Self {
            model: model.into(),
            filename: default_filename(),
            content_type: default_content_type(),
            response_format: None,
            max_pages: None,
        }
    }

    fn validate(&self, file_bytes: usize) -> Result<()> {
        ensure!(file_bytes > 0, "OCR document must not be empty");
        ensure!(
            file_bytes <= MAX_DOCUMENT_BYTES,
            "OCR document exceeds 64 MiB"
        );
        ensure!(!self.model.trim().is_empty(), "OCR model must not be empty");
        ensure!(
            self.max_pages.is_none_or(|pages| pages > 0),
            "OCR max_pages must be positive"
        );
        ensure!(
            self.response_format
                .as_deref()
                .is_none_or(|format| matches!(format, "json" | "text" | "markdown")),
            "OCR response_format must be json, text, or markdown"
        );
        Ok(())
    }
}

impl OpenAiClient {
    /// Upload a PDF or image to the selected mesh OCR provider.
    ///
    /// The response preserves successful JSON, text or Markdown output. Provider
    /// HTTP errors include their status/body in the returned error. Use
    /// [`Self::document_ocr_raw`] when the application needs the response for a
    /// non-success status.
    ///
    /// # Errors
    /// Fails for invalid upload options, an empty or oversized file, invalid MIME
    /// syntax, transport failures, provider HTTP errors, or an unreadable body.
    pub async fn document_ocr(
        &self,
        file: Vec<u8>,
        options: DocumentOcrOptions,
    ) -> Result<RawOpenAiResponse> {
        self.document_ocr_raw(file, options)
            .await?
            .error_for_status()
    }

    /// Upload document bytes and preserve every provider HTTP response intact.
    ///
    /// # Errors
    /// Fails for invalid upload options, an empty or oversized file, invalid MIME
    /// syntax, transport failures, or an unreadable response body. HTTP errors
    /// themselves remain available through the response status and body.
    pub async fn document_ocr_raw(
        &self,
        file: Vec<u8>,
        options: DocumentOcrOptions,
    ) -> Result<RawOpenAiResponse> {
        options.validate(file.len())?;
        let part = Part::bytes(file)
            .file_name(options.filename)
            .mime_str(&options.content_type)?;
        let mut form = Form::new().text("model", options.model).part("file", part);
        if let Some(format) = options.response_format {
            form = form.text("response_format", format);
        }
        if let Some(pages) = options.max_pages {
            form = form.text("max_pages", pages.to_string());
        }
        let response = self
            .document_request(form)
            .timeout(Duration::from_secs(120))
            .send()
            .await?;
        let status_code = response.status().as_u16();
        let content_type = response
            .headers()
            .get(reqwest::header::CONTENT_TYPE)
            .and_then(|value| value.to_str().ok())
            .map(ToString::to_string);
        let body = response.text().await?;
        Ok(RawOpenAiResponse {
            status_code,
            content_type,
            body,
        })
    }
}

fn default_filename() -> String {
    "document".into()
}

fn default_content_type() -> String {
    "application/octet-stream".into()
}

#[cfg(test)]
mod tests;
