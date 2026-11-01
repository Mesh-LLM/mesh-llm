//! Binary document OCR bridge; native inference returns provider responses intact.

use crate::errors::FfiError;
use crate::handles::MeshNodeHandle;
use crate::request_types::OpenAiResponseNative;
use crate::runtime_blocking::block_on;

#[uniffi::export]
impl MeshNodeHandle {
    /// Upload unchanged document bytes with the same options as the Rust API.
    pub fn document_ocr(
        &self,
        file: Vec<u8>,
        options_json: String,
    ) -> Result<OpenAiResponseNative, FfiError> {
        let options = serde_json::from_str(&options_json)
            .map_err(|error| FfiError::OpenAiRequestFailed(error.to_string()))?;
        let client = crate::node::openai_client(self)?;
        let response = block_on(client.document_ocr_raw(file, options))
            .map_err(|error| FfiError::OpenAiRequestFailed(error.to_string()))?;
        Ok(OpenAiResponseNative {
            status_code: response.status_code,
            content_type: response.content_type,
            body: response.body,
        })
    }
}
