//! Hands PDF attachments on locally received chat requests to a
//! `document-ingest.v1` plugin before routing.
//!
//! The host only finds PDF parts, stages their bytes in the plugin's own
//! staging directory and applies the plugin's answer: each PDF part becomes the
//! text part the plugin returns, and the request may be routed to the model the
//! plugin names. Extraction, the inline/pointer/RLM decision and any document
//! store belong to the plugin. Requests from mesh peers are never touched, and
//! a raw PDF is never forwarded once a provider is installed: any failure ends
//! the request.

use std::path::{Path, PathBuf};
use std::time::Duration;

use base64::Engine;
use serde::Deserialize;
use serde_json::{Value, json};

use crate::plugin;

pub(crate) const DOCUMENT_INGEST_CAPABILITY: &str = "document-ingest.v1";
const STAGING_DIR_OPERATION: &str = "staging_dir";
const PREPARE_DOCUMENTS_OPERATION: &str = "prepare_documents";
/// Per-request override of the plugin's mode decision: `inline`, `pointer`
/// or `rlm`. Passed to the plugin unread.
pub(crate) const DOCUMENT_MODE_HEADER: &str = "x-mesh-documents";
const MAX_REQUEST_PDFS: usize = 8;
const PDF_MAGIC: &[u8] = b"%PDF-";
/// Base64 of `%PDF-`, the start of every inline PDF data URL.
const PDF_MAGIC_BASE64: &str = "JVBERi0";
/// The plugin enforces this deadline itself, aborting extraction; the host
/// waits a little longer so the plugin's own deadline error reaches the caller.
const PREPARE_DEADLINE: Duration = Duration::from_secs(120);
const PREPARE_HOST_GRACE: Duration = Duration::from_secs(5);
const STAGED_NAME_PREFIX: &str = "mesh-doc-";

/// Where a request came from, as far as document handling needs to know.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct RequestOrigin {
    /// Received on this node's own API listener, not tunnelled from a peer.
    pub(crate) local_listener: bool,
    /// The client connected from a loopback address.
    pub(crate) loopback: bool,
}

impl RequestOrigin {
    pub(crate) fn local(peer: Option<std::net::SocketAddr>) -> Self {
        Self {
            local_listener: true,
            loopback: peer.is_some_and(|addr| addr.ip().is_loopback()),
        }
    }
}

/// A document failure the caller can act on, with its HTTP status.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct DocumentIngestError {
    pub(crate) status: u16,
    pub(crate) code: String,
    pub(crate) message: String,
}

impl DocumentIngestError {
    fn new(status: u16, code: &str, message: impl Into<String>) -> Self {
        Self {
            status,
            code: code.to_string(),
            message: message.into(),
        }
    }

    fn plugin_failure(message: impl Into<String>) -> Self {
        Self::new(502, "document_plugin_failed", message)
    }
}

impl std::fmt::Display for DocumentIngestError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(&self.message)
    }
}

impl std::error::Error for DocumentIngestError {}

/// Whether a chat body may carry an inline PDF and so must be parsed.
pub(super) fn body_may_contain_pdf(body_text: &str) -> bool {
    body_text.contains("application/pdf") || body_text.contains(PDF_MAGIC_BASE64)
}

/// What the plugin changed, for the reader's routing fields.
#[derive(Debug, Default, PartialEq, Eq)]
pub(super) struct DocumentRewrite {
    pub(super) documents: usize,
    pub(super) route_model: Option<String>,
}

struct PdfPart {
    message_index: usize,
    block_index: usize,
    bytes: Vec<u8>,
    filename: Option<String>,
}

/// Replace every PDF part in `body` with the plugin's text, in place.
///
/// Returns `None` when there is nothing to do: no provider, a peer request,
/// or no PDF parts.
pub(super) async fn prepare_request_documents(
    body: &mut Value,
    plugin_manager: &plugin::PluginManager,
    origin: RequestOrigin,
    mode_header: Option<&str>,
) -> Result<Option<DocumentRewrite>, DocumentIngestError> {
    if !origin.local_listener {
        return Ok(None);
    }
    let parts = find_pdf_parts(body)?;
    if parts.is_empty() {
        return Ok(None);
    }
    let Some(provider) = plugin_manager
        .available_provider_for_capability(DOCUMENT_INGEST_CAPABILITY)
        .await
        .ok()
        .flatten()
    else {
        return Ok(None);
    };
    if parts.len() > MAX_REQUEST_PDFS {
        return Err(DocumentIngestError::new(
            413,
            "document_too_large",
            format!("at most {MAX_REQUEST_PDFS} PDF attachments per request"),
        ));
    }
    let staging_dir = staging_dir(plugin_manager, &provider.plugin_name).await?;
    let mut staged = StagedFiles::default();
    let mut documents = Vec::with_capacity(parts.len());
    for (index, part) in parts.iter().enumerate() {
        let name = staged
            .stage(&staging_dir, &part.bytes)
            .await
            .map_err(|error| {
                DocumentIngestError::new(
                    503,
                    "document_staging_failed",
                    format!("could not stage PDF attachment: {error}"),
                )
            })?;
        documents.push(json!({
            "index": index,
            "staged": name,
            "filename": part.filename,
        }));
    }
    let request = json!({
        "documents": documents,
        "caller": {
            "loopback": origin.loopback,
            "has_tools": request_has_tools(body),
            "mode_header": mode_header,
        },
        "model": body.get("model").and_then(Value::as_str),
        "deadline_ms": PREPARE_DEADLINE.as_millis() as u64,
    });
    let response = invoke_prepare(plugin_manager, &provider.plugin_name, &request).await;
    drop(staged);
    let response = response?;
    apply_prepared_parts(body, &parts, response)
}

async fn staging_dir(
    plugin_manager: &plugin::PluginManager,
    plugin_name: &str,
) -> Result<PathBuf, DocumentIngestError> {
    #[derive(Deserialize)]
    struct StagingDirResponse {
        dir: PathBuf,
    }
    let result = plugin_manager
        .invoke_operation(plugin_name, STAGING_DIR_OPERATION, "{}")
        .await
        .map_err(|error| DocumentIngestError::plugin_failure(error.to_string()))?;
    if result.is_error {
        return Err(DocumentIngestError::plugin_failure(result.content_json));
    }
    let response: StagingDirResponse = serde_json::from_str(&result.content_json)
        .map_err(|error| DocumentIngestError::plugin_failure(format!("staging_dir: {error}")))?;
    let metadata = std::fs::symlink_metadata(&response.dir).ok();
    if !response.dir.is_absolute() || !metadata.is_some_and(|metadata| metadata.is_dir()) {
        return Err(DocumentIngestError::plugin_failure(
            "staging_dir is not an existing absolute directory",
        ));
    }
    Ok(response.dir)
}

#[derive(Deserialize)]
struct PrepareResponse {
    #[serde(default)]
    parts: Vec<PreparedPart>,
    #[serde(default)]
    route_model: Option<String>,
    #[serde(default)]
    error: Option<PrepareError>,
}

#[derive(Deserialize)]
struct PreparedPart {
    index: usize,
    text: String,
}

#[derive(Deserialize)]
struct PrepareError {
    status: u16,
    code: String,
    message: String,
}

async fn invoke_prepare(
    plugin_manager: &plugin::PluginManager,
    plugin_name: &str,
    request: &Value,
) -> Result<PrepareResponse, DocumentIngestError> {
    let input = request.to_string();
    let call = plugin_manager.invoke_operation_without_timeout(
        plugin_name,
        PREPARE_DOCUMENTS_OPERATION,
        &input,
    );
    let result = tokio::time::timeout(PREPARE_DEADLINE + PREPARE_HOST_GRACE, call)
        .await
        .map_err(|_| {
            DocumentIngestError::new(504, "document_timeout", "document extraction timed out")
        })?
        .map_err(|error| DocumentIngestError::plugin_failure(error.to_string()))?;
    if result.is_error {
        return Err(DocumentIngestError::plugin_failure(result.content_json));
    }
    let response: PrepareResponse =
        serde_json::from_str(&result.content_json).map_err(|error| {
            DocumentIngestError::plugin_failure(format!("prepare_documents: {error}"))
        })?;
    if let Some(error) = response.error {
        let status = if (400..600).contains(&error.status) {
            error.status
        } else {
            502
        };
        return Err(DocumentIngestError {
            status,
            code: error.code,
            message: error.message,
        });
    }
    Ok(response)
}

fn apply_prepared_parts(
    body: &mut Value,
    parts: &[PdfPart],
    response: PrepareResponse,
) -> Result<Option<DocumentRewrite>, DocumentIngestError> {
    let mut texts: Vec<Option<String>> = vec![None; parts.len()];
    for part in response.parts {
        let slot = texts.get_mut(part.index).ok_or_else(|| {
            DocumentIngestError::plugin_failure("prepare_documents returned an unknown index")
        })?;
        if slot.replace(part.text).is_some() {
            return Err(DocumentIngestError::plugin_failure(
                "prepare_documents returned an index twice",
            ));
        }
    }
    for (part, text) in parts.iter().zip(texts) {
        let text = text.ok_or_else(|| {
            DocumentIngestError::plugin_failure("prepare_documents left a document unanswered")
        })?;
        let block = body
            .get_mut("messages")
            .and_then(|messages| messages.get_mut(part.message_index))
            .and_then(|message| message.get_mut("content"))
            .and_then(|content| content.get_mut(part.block_index))
            .ok_or_else(|| DocumentIngestError::plugin_failure("PDF part moved"))?;
        *block = json!({"type": "text", "text": text});
    }
    let route_model = response
        .route_model
        .map(|model| model.trim().to_string())
        .filter(|model| !model.is_empty());
    if let (Some(model), Some(object)) = (route_model.as_ref(), body.as_object_mut()) {
        object.insert("model".into(), Value::String(model.clone()));
    }
    Ok(Some(DocumentRewrite {
        documents: parts.len(),
        route_model,
    }))
}

fn request_has_tools(body: &Value) -> bool {
    body.get("tools")
        .and_then(Value::as_array)
        .is_some_and(|tools| !tools.is_empty())
}

fn find_pdf_parts(body: &Value) -> Result<Vec<PdfPart>, DocumentIngestError> {
    let Some(messages) = body.get("messages").and_then(Value::as_array) else {
        return Ok(Vec::new());
    };
    let mut parts = Vec::new();
    for (message_index, message) in messages.iter().enumerate() {
        let Some(blocks) = message.get("content").and_then(Value::as_array) else {
            continue;
        };
        for (block_index, block) in blocks.iter().enumerate() {
            if let Some((bytes, filename)) = pdf_block(block)? {
                parts.push(PdfPart {
                    message_index,
                    block_index,
                    bytes,
                    filename,
                });
            }
        }
    }
    Ok(parts)
}

/// The decoded bytes and filename of a `file`/`input_file` part carrying an
/// inline PDF. Remote URLs and other file types are left alone.
type InlinePdf = (Vec<u8>, Option<String>);

fn pdf_block(block: &Value) -> Result<Option<InlinePdf>, DocumentIngestError> {
    let Some(container) = ["file", "input_file"]
        .into_iter()
        .find_map(|key| block.get(key))
        .filter(|container| container.is_object())
    else {
        return Ok(None);
    };
    let Some(data_url) = ["file_data", "url"]
        .into_iter()
        .find_map(|key| container.get(key).and_then(Value::as_str))
        .filter(|value| value.starts_with("data:"))
    else {
        return Ok(None);
    };
    let Some((header, payload)) = data_url["data:".len()..].split_once(',') else {
        return Ok(None);
    };
    let declared_pdf = header
        .split(';')
        .next()
        .is_some_and(|mime| mime.eq_ignore_ascii_case("application/pdf"))
        || container
            .get("mime_type")
            .and_then(Value::as_str)
            .is_some_and(|mime| mime.eq_ignore_ascii_case("application/pdf"));
    let base64 = header.split(';').any(|param| param == "base64");
    if !base64 || !(declared_pdf || payload.starts_with(PDF_MAGIC_BASE64)) {
        return Ok(None);
    }
    let bytes = base64::engine::general_purpose::STANDARD
        .decode(payload.trim())
        .map_err(|_| DocumentIngestError::new(400, "invalid_pdf", "PDF data is not base64"))?;
    if !bytes.starts_with(PDF_MAGIC) {
        return Err(DocumentIngestError::new(
            422,
            "invalid_pdf",
            "attachment is declared as a PDF but is not one",
        ));
    }
    let filename = ["filename", "file_name"]
        .into_iter()
        .find_map(|key| container.get(key).and_then(Value::as_str))
        .map(str::to_string);
    Ok(Some((bytes, filename)))
}

/// Staged PDFs, removed on every exit including errors and cancellation.
#[derive(Default)]
struct StagedFiles(Vec<PathBuf>);

impl StagedFiles {
    async fn stage(&mut self, dir: &Path, bytes: &[u8]) -> std::io::Result<String> {
        use tokio::io::AsyncWriteExt;
        let name = format!(
            "{STAGED_NAME_PREFIX}{}.pdf",
            hex::encode(rand::random::<[u8; 16]>())
        );
        let path = dir.join(&name);
        let mut options = tokio::fs::OpenOptions::new();
        options.write(true).create_new(true);
        #[cfg(unix)]
        options.mode(0o600);
        let mut file = options.open(&path).await?;
        self.0.push(path);
        file.write_all(bytes).await?;
        file.flush().await?;
        Ok(name)
    }
}

impl Drop for StagedFiles {
    fn drop(&mut self) {
        for path in &self.0 {
            let _ = std::fs::remove_file(path);
        }
    }
}

#[cfg(test)]
#[path = "document_ingest_tests.rs"]
pub(crate) mod tests;
