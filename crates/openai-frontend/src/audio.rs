use serde::{Deserialize, Serialize};

use crate::{OpenAiError, OpenAiResult};

const MAX_AUDIO_BYTES: usize = 64 * 1024 * 1024;

#[derive(Debug, Clone, Copy, Default, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum AudioFormat {
    #[default]
    Mp3,
    Opus,
    Aac,
    Flac,
    Wav,
    Pcm,
}

impl AudioFormat {
    /// Return the media type corresponding to the requested audio wire encoding.
    pub const fn content_type(self) -> &'static str {
        match self {
            Self::Mp3 => "audio/mpeg",
            Self::Opus => "audio/opus",
            Self::Aac => "audio/aac",
            Self::Flac => "audio/flac",
            Self::Wav => "audio/wav",
            Self::Pcm => "audio/pcm",
        }
    }
}

#[derive(Debug, Clone, Deserialize, PartialEq)]
pub struct AudioSpeechRequest {
    pub model: String,
    pub input: String,
    pub voice: String,
    #[serde(default)]
    pub response_format: AudioFormat,
    #[serde(default = "default_speed")]
    pub speed: f32,
}

/// Omitted speed retains the model's unscaled playback rate.
fn default_speed() -> f32 {
    1.0
}

impl AudioSpeechRequest {
    /// Validate required inputs and the finite OpenAI playback-speed range.
    pub fn validate(&self) -> OpenAiResult<()> {
        if self.model.trim().is_empty() || self.input.is_empty() || self.voice.trim().is_empty() {
            return Err(OpenAiError::invalid_request(
                "model, input, and voice must not be empty",
            ));
        }
        if !(0.25..=4.0).contains(&self.speed) || !self.speed.is_finite() {
            return Err(OpenAiError::invalid_request(
                "speed must be a finite value between 0.25 and 4.0",
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AudioResponse {
    pub bytes: Vec<u8>,
    pub content_type: String,
}

impl AudioResponse {
    /// Reject empty backend output before constructing a binary HTTP response.
    pub fn new(bytes: Vec<u8>, content_type: impl Into<String>) -> OpenAiResult<Self> {
        if bytes.is_empty() {
            return Err(OpenAiError::backend(
                "audio backend returned an empty payload",
            ));
        }
        Ok(Self {
            bytes,
            content_type: content_type.into(),
        })
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct AudioTranscriptionRequest {
    pub model: String,
    pub file: Vec<u8>,
    pub filename: Option<String>,
    pub language: Option<String>,
    pub prompt: Option<String>,
    pub response_format: String,
    pub temperature: Option<f32>,
}

impl AudioTranscriptionRequest {
    pub const MAX_FILE_BYTES: usize = MAX_AUDIO_BYTES;

    /// Enforce upload size, output format, and the finite `[0, 1]` temperature range.
    pub fn validate(&self) -> OpenAiResult<()> {
        if self.model.trim().is_empty() {
            return Err(OpenAiError::invalid_request("model must not be empty"));
        }
        if self.file.is_empty() {
            return Err(OpenAiError::invalid_request("audio file must not be empty"));
        }
        if self.file.len() > Self::MAX_FILE_BYTES {
            return Err(OpenAiError::payload_too_large(format!(
                "audio file exceeds the {} byte limit",
                Self::MAX_FILE_BYTES
            )));
        }
        if !matches!(self.response_format.as_str(), "json" | "text") {
            return Err(OpenAiError::unsupported(
                "response_format must be 'json' or 'text'",
            ));
        }
        if self
            .temperature
            .is_some_and(|value| !value.is_finite() || !(0.0..=1.0).contains(&value))
        {
            return Err(OpenAiError::invalid_request(
                "temperature must be a finite value between 0.0 and 1.0",
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct AudioTranscriptionResponse {
    pub text: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn transcription(temperature: Option<f32>) -> AudioTranscriptionRequest {
        AudioTranscriptionRequest {
            model: "fixture".to_string(),
            file: vec![1],
            filename: None,
            language: None,
            prompt: None,
            response_format: "json".to_string(),
            temperature,
        }
    }

    #[test]
    fn transcription_temperature_is_bounded_to_openai_range() {
        for temperature in [None, Some(0.0), Some(0.5), Some(1.0)] {
            assert!(transcription(temperature).validate().is_ok());
        }
        for temperature in [Some(-0.1), Some(1.1), Some(f32::NAN), Some(f32::INFINITY)] {
            assert!(transcription(temperature).validate().is_err());
        }
    }
}
