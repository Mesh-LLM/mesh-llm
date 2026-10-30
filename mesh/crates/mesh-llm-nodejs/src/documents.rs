//! Binary document OCR requests through the Rust SDK.

use napi::bindgen_prelude::*;
use napi_derive::napi;
use serde_json::json;

use super::{Node, to_napi_error};

#[napi]
impl Node {
    #[napi(js_name = "documentOcrJson")]
    pub async fn document_ocr_json(&self, file: Buffer, options_json: String) -> Result<String> {
        if file.is_empty() || file.len() > mesh_llm_sdk::documents::MAX_DOCUMENT_BYTES {
            return Err(Error::from_reason(
                "OCR document must contain 1 byte to 64 MiB",
            ));
        }
        let file = file.to_vec();
        let options = serde_json::from_str(&options_json).map_err(to_napi_error)?;
        let response = self
            .openai_client()
            .await?
            .document_ocr_raw(file, options)
            .await
            .map_err(to_napi_error)?;
        Ok(json!({
            "statusCode": response.status_code,
            "contentType": response.content_type,
            "body": response.body,
        })
        .to_string())
    }
}
