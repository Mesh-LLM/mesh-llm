//! Translation of one `/v1/responses` input content item to its Chat
//! Completions content part.

use serde_json::{Map, Value};

use crate::errors::InferenceError;

fn object_or_url_container(
    value: Option<&Value>,
    fallback_url: Option<&str>,
) -> Option<Map<String, Value>> {
    match value {
        Some(Value::Object(map)) => Some(map.clone()),
        Some(Value::String(url)) => Some(Map::from_iter([(
            "url".to_string(),
            Value::String(url.clone()),
        )])),
        _ => fallback_url
            .map(|url| Map::from_iter([("url".to_string(), Value::String(url.to_string()))])),
    }
}

pub(super) fn translate_responses_content_item(item: &Value) -> Result<Value, InferenceError> {
    let Some(object) = item.as_object() else {
        return Ok(serde_json::json!({
            "type": "text",
            "text": item.as_str().unwrap_or_default(),
        }));
    };
    let item_type = object.get("type").and_then(Value::as_str).unwrap_or("text");

    match item_type {
        "input_text" | "output_text" | "text" => {
            let text = object
                .get("text")
                .and_then(Value::as_str)
                .unwrap_or_default();
            Ok(serde_json::json!({"type": "text", "text": text}))
        }
        "input_image" | "image_url" | "image" => {
            let container = object_or_url_container(
                object.get("image_url").or_else(|| object.get("image")),
                object.get("url").and_then(Value::as_str),
            )
            .ok_or_else(|| {
                InferenceError::invalid_request(
                    "responses input_image block is missing image_url/url",
                )
            })?;
            Ok(serde_json::json!({"type": "image_url", "image_url": container}))
        }
        "input_audio" | "audio" | "audio_url" => {
            let mut container = object_or_url_container(
                object
                    .get("input_audio")
                    .or_else(|| object.get("audio_url")),
                object.get("url").and_then(Value::as_str),
            )
            .unwrap_or_default();
            for key in [
                "data",
                "format",
                "mime_type",
                "mesh_token",
                "blob_token",
                "token",
            ] {
                if let Some(value) = object.get(key) {
                    container
                        .entry(key.to_string())
                        .or_insert_with(|| value.clone());
                }
            }
            if container.is_empty() {
                return Err(InferenceError::invalid_request(
                    "responses input_audio block is missing input_audio/audio_url/url",
                ));
            }
            Ok(serde_json::json!({"type": "input_audio", "input_audio": container}))
        }
        "input_file" | "file" => {
            // OpenAI's own Responses form is flat: `file_data`, `file_id` or
            // `file_url` beside `type`; older clients nest a container.
            let mut container = object_or_url_container(
                object.get("input_file").or_else(|| object.get("file")),
                object.get("url").and_then(Value::as_str),
            )
            .unwrap_or_default();
            for key in [
                "file_data",
                "file_id",
                "file_url",
                "mime_type",
                "file_name",
                "filename",
                "mesh_token",
                "blob_token",
                "token",
            ] {
                if let Some(value) = object.get(key) {
                    container
                        .entry(key.to_string())
                        .or_insert_with(|| value.clone());
                }
            }
            if ![
                "url",
                "file_data",
                "file_id",
                "file_url",
                "mesh_token",
                "blob_token",
                "token",
            ]
            .iter()
            .any(|key| container.contains_key(*key))
            {
                return Err(InferenceError::invalid_request(
                    "responses input_file block is missing file_data/file_id/file_url/url",
                ));
            }
            Ok(serde_json::json!({"type": "input_file", "input_file": container}))
        }
        other => Err(InferenceError::unsupported(format!(
            "unsupported /v1/responses content block type '{other}'"
        ))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn flat_openai_input_files_pass_through() {
        for (item, key, value) in [
            (
                json!({"type": "input_file", "filename": "a.pdf", "file_data": "data:application/pdf;base64,JVBERi0="}),
                "file_data",
                "data:application/pdf;base64,JVBERi0=",
            ),
            (
                json!({"type": "input_file", "file_id": "file-1"}),
                "file_id",
                "file-1",
            ),
            (
                json!({"type": "input_file", "file_url": "https://example.com/a.pdf"}),
                "file_url",
                "https://example.com/a.pdf",
            ),
        ] {
            let part = translate_responses_content_item(&item).unwrap();
            assert_eq!(part["type"], "input_file");
            assert_eq!(part["input_file"][key], value, "{item}");
        }
        let part = translate_responses_content_item(
            &json!({"type": "input_file", "filename": "a.pdf", "file_data": "x"}),
        )
        .unwrap();
        assert_eq!(part["input_file"]["filename"], "a.pdf");
    }

    #[test]
    fn nested_input_files_keep_their_container() {
        let part = translate_responses_content_item(&json!({
            "type": "input_file",
            "input_file": {"url": "data:text/plain;base64,aGVsbG8=", "file_name": "hello.txt"},
        }))
        .unwrap();
        assert_eq!(part["input_file"]["url"], "data:text/plain;base64,aGVsbG8=");
        assert_eq!(part["input_file"]["file_name"], "hello.txt");
    }

    #[test]
    fn input_files_without_content_are_rejected() {
        let error =
            translate_responses_content_item(&json!({"type": "input_file", "filename": "a.pdf"}))
                .unwrap_err();
        assert!(error.to_string().contains("missing file_data"), "{error}");
    }
}
