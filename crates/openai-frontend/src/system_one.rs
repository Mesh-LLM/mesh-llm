use std::collections::BTreeMap;

use crate::{OpenAiError, OpenAiResult};
use serde::{Deserialize, Serialize};
use serde_json::Value;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SystemOneRequest {
    pub state: Value,
    pub model: String,
    pub questions: BTreeMap<String, SystemOneQuestion>,
    #[serde(default)]
    pub images: Option<Vec<Value>>,
    #[serde(default)]
    pub steps: Option<u8>,
    #[serde(default)]
    pub samples: Option<u8>,
    #[serde(default)]
    pub think: Option<u32>,
    #[serde(default)]
    pub sequential: Option<bool>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum SystemOneQuestion {
    Noul {
        #[serde(default)]
        instructions: Option<Value>,
        #[serde(default)]
        criteria: Option<SystemOneNoulCriteria>,
    },
    Choice {
        #[serde(default)]
        instructions: Option<Value>,
        criteria: BTreeMap<String, Value>,
    },
    Score {
        #[serde(default)]
        instructions: Option<Value>,
        criteria: Vec<Value>,
    },
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct SystemOneNoulCriteria {
    #[serde(default)]
    pub r#true: Option<Value>,
    #[serde(default)]
    pub r#false: Option<Value>,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct SystemOneResponse {
    pub model: String,
    pub answers: BTreeMap<String, SystemOneAnswer>,
    pub usage: SystemOneUsage,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum SystemOneAnswer {
    Noul {
        noul: f32,
    },
    Choice {
        choice: String,
        probabilities: BTreeMap<String, f32>,
        confidence: f32,
    },
    Score {
        score: f32,
        legend: BTreeMap<String, Value>,
        probabilities: BTreeMap<String, f32>,
        confidence: f32,
    },
}

#[derive(Debug, Clone, Copy, Serialize, PartialEq, Eq)]
pub struct SystemOneUsage {
    pub input_tokens: u32,
    pub output_tokens: u32,
}

/// Maximum number of images on one System One read (upstream OpenJEV contract).
pub const SYSTEM_ONE_MAX_IMAGES: usize = 8;
/// Maximum decoded size per image, 5 MiB (upstream OpenJEV contract).
pub const SYSTEM_ONE_MAX_IMAGE_BYTES: usize = 5 * 1024 * 1024;

/// Image content types accepted on a System One read.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SystemOneImageContentType {
    Jpeg,
    Png,
    Webp,
    Gif,
}

impl SystemOneImageContentType {
    pub fn mime(self) -> &'static str {
        match self {
            Self::Jpeg => "image/jpeg",
            Self::Png => "image/png",
            Self::Webp => "image/webp",
            Self::Gif => "image/gif",
        }
    }

    fn from_mime(mime: &str) -> Option<Self> {
        match mime {
            "image/jpeg" | "image/jpg" => Some(Self::Jpeg),
            "image/png" => Some(Self::Png),
            "image/webp" => Some(Self::Webp),
            "image/gif" => Some(Self::Gif),
            _ => None,
        }
    }
}

/// One decoded inline image from a System One read request.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SystemOneImage {
    pub content_type: SystemOneImageContentType,
    pub bytes: Vec<u8>,
}

fn decode_system_one_base64(payload: &str) -> OpenAiResult<Vec<u8>> {
    use base64::Engine;
    base64::engine::general_purpose::STANDARD
        .decode(payload.trim())
        .or_else(|_| {
            base64::engine::general_purpose::URL_SAFE_NO_PAD.decode(payload.trim().as_bytes())
        })
        .map_err(|error| OpenAiError::invalid_request(format!("invalid image base64: {error}")))
}

/// Parse and validate the request's inline images. Accepts `data:image/...;base64,...`
/// URLs or `{content_type, base64}` objects, mirroring the upstream OpenJEV contract.
pub fn parse_system_one_images(images: Option<&Vec<Value>>) -> OpenAiResult<Vec<SystemOneImage>> {
    let raw = match images {
        None => return Ok(Vec::new()),
        Some(images) => images,
    };
    if raw.len() > SYSTEM_ONE_MAX_IMAGES {
        return Err(OpenAiError::invalid_request(format!(
            "System One accepts at most {SYSTEM_ONE_MAX_IMAGES} images; got {}",
            raw.len()
        )));
    }
    let mut parsed = Vec::with_capacity(raw.len());
    for (index, value) in raw.iter().enumerate() {
        let (mime, payload) = match value {
            Value::String(data_url) => {
                let (prefix, payload) = data_url.split_once(',').ok_or_else(|| {
                    OpenAiError::invalid_request(format!(
                        "image {index} must be a data:image/...;base64,... URL or a {{content_type, base64}} object"
                    ))
                })?;
                let mime = prefix
                    .strip_prefix("data:")
                    .and_then(|prefix| prefix.strip_suffix(";base64"))
                    .ok_or_else(|| {
                        OpenAiError::invalid_request(format!(
                            "image {index} must be a data:image/...;base64,... URL"
                        ))
                    })?;
                (mime.to_owned(), payload.to_owned())
            }
            Value::Object(object) => {
                let mime = object
                    .get("content_type")
                    .and_then(Value::as_str)
                    .ok_or_else(|| {
                        OpenAiError::invalid_request(format!(
                            "image {index} object requires a string content_type"
                        ))
                    })?
                    .to_owned();
                let payload = object
                    .get("base64")
                    .and_then(Value::as_str)
                    .ok_or_else(|| {
                        OpenAiError::invalid_request(format!(
                            "image {index} object requires a string base64 payload"
                        ))
                    })?
                    .to_owned();
                (mime, payload)
            }
            _ => {
                return Err(OpenAiError::invalid_request(format!(
                    "image {index} must be a data URL string or a {{content_type, base64}} object"
                )));
            }
        };
        let content_type = SystemOneImageContentType::from_mime(&mime).ok_or_else(|| {
            OpenAiError::invalid_request(format!(
                "image {index} has unsupported content type {mime:?}; expected JPEG, PNG, WebP, or GIF"
            ))
        })?;
        let bytes = decode_system_one_base64(&payload)?;
        if bytes.len() > SYSTEM_ONE_MAX_IMAGE_BYTES {
            return Err(OpenAiError::invalid_request(format!(
                "image {index} is {} bytes; the limit is {SYSTEM_ONE_MAX_IMAGE_BYTES} bytes",
                bytes.len()
            )));
        }
        if bytes.is_empty() {
            return Err(OpenAiError::invalid_request(format!(
                "image {index} is empty"
            )));
        }
        parsed.push(SystemOneImage {
            content_type,
            bytes,
        });
    }
    Ok(parsed)
}
