use serde::{Deserialize, Serialize};

use crate::{InferenceError, InferenceResult};

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
    pub fn validate(&self) -> InferenceResult<()> {
        if self.model.trim().is_empty() || self.input.is_empty() || self.voice.trim().is_empty() {
            return Err(InferenceError::invalid_request(
                "model, input, and voice must not be empty",
            ));
        }
        if !(0.25..=4.0).contains(&self.speed) || !self.speed.is_finite() {
            return Err(InferenceError::invalid_request(
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
    pub fn new(bytes: Vec<u8>, content_type: impl Into<String>) -> InferenceResult<Self> {
        if bytes.is_empty() {
            return Err(InferenceError::backend(
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

    /// Whether the requested format needs per-segment timing
    /// (`verbose_json`, `srt`, `vtt`), which only some models report.
    pub fn needs_segment_timing(&self) -> bool {
        matches!(
            self.response_format.as_str(),
            "verbose_json" | "srt" | "vtt"
        )
    }

    /// Enforce upload size, output format, and the finite `[0, 1]` temperature range.
    pub fn validate(&self) -> InferenceResult<()> {
        if self.model.trim().is_empty() {
            return Err(InferenceError::invalid_request("model must not be empty"));
        }
        if self.file.is_empty() {
            return Err(InferenceError::invalid_request(
                "audio file must not be empty",
            ));
        }
        if self.file.len() > Self::MAX_FILE_BYTES {
            return Err(InferenceError::payload_too_large(format!(
                "audio file exceeds the {} byte limit",
                Self::MAX_FILE_BYTES
            )));
        }
        if !matches!(
            self.response_format.as_str(),
            "json" | "text" | "verbose_json" | "srt" | "vtt"
        ) {
            return Err(InferenceError::unsupported(
                "response_format must be 'json', 'text', 'verbose_json', 'srt', or 'vtt'",
            ));
        }
        if self
            .temperature
            .is_some_and(|value| !value.is_finite() || !(0.0..=1.0).contains(&value))
        {
            return Err(InferenceError::invalid_request(
                "temperature must be a finite value between 0.0 and 1.0",
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct AudioTranscriptionResponse {
    pub text: String,
    /// Segment timing from models that report it; `verbose_json`, `srt`, and
    /// `vtt` need it.
    #[serde(skip)]
    pub timing: Option<AudioTranscriptionTiming>,
}

impl AudioTranscriptionResponse {
    /// A transcript without segment timing.
    pub fn text(text: impl Into<String>) -> Self {
        Self {
            text: text.into(),
            timing: None,
        }
    }
}

/// Language, length, and timed segments of a transcription.
#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct AudioTranscriptionTiming {
    /// ISO 639-1 code of the spoken language; empty when unknown.
    pub language: String,
    /// Seconds of audio transcribed.
    pub duration: f64,
    pub segments: Vec<AudioTranscriptionSegment>,
}

/// One timed segment, in the `verbose_json` shape.
#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct AudioTranscriptionSegment {
    pub id: usize,
    /// Seconds from the start of the audio.
    pub start: f64,
    pub end: f64,
    pub text: String,
    pub no_speech_prob: f32,
}

/// Renders `verbose_json`: the OpenAI field subset these models report.
pub(crate) fn verbose_json(
    response: &AudioTranscriptionResponse,
    timing: &AudioTranscriptionTiming,
    translate: bool,
) -> serde_json::Value {
    serde_json::json!({
        "task": if translate { "translate" } else { "transcribe" },
        "language": timing.language,
        "duration": timing.duration,
        "text": response.text,
        "segments": timing.segments,
    })
}

/// Renders SubRip subtitles: numbered cues with `HH:MM:SS,mmm` bounds.
pub(crate) fn srt(timing: &AudioTranscriptionTiming) -> String {
    let mut out = String::new();
    for (index, segment) in timing.segments.iter().enumerate() {
        out.push_str(&format!(
            "{}\n{} --> {}\n{}\n\n",
            index + 1,
            subtitle_time(segment.start, ','),
            subtitle_time(cue_end(segment), ','),
            segment.text.trim()
        ));
    }
    out
}

/// Renders WebVTT subtitles with `HH:MM:SS.mmm` cue bounds.
pub(crate) fn vtt(timing: &AudioTranscriptionTiming) -> String {
    let mut out = String::from("WEBVTT\n\n");
    for segment in &timing.segments {
        out.push_str(&format!(
            "{} --> {}\n{}\n\n",
            subtitle_time(segment.start, '.'),
            subtitle_time(cue_end(segment), '.'),
            segment.text.trim()
        ));
    }
    out
}

/// A cue's end, never before its start: Whisper can emit zero-length or
/// rounded segments, and players reject an end earlier than the start.
fn cue_end(segment: &AudioTranscriptionSegment) -> f64 {
    segment.end.max(segment.start)
}

fn subtitle_time(seconds: f64, separator: char) -> String {
    let millis = (seconds.max(0.0) * 1000.0).round() as u64;
    format!(
        "{:02}:{:02}:{:02}{separator}{:03}",
        millis / 3_600_000,
        millis / 60_000 % 60,
        millis / 1000 % 60,
        millis % 1000
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a valid transcription fixture for multipart parameter tests.
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

    fn timing() -> AudioTranscriptionTiming {
        AudioTranscriptionTiming {
            language: "en".into(),
            duration: 3_725.5,
            segments: vec![
                AudioTranscriptionSegment {
                    id: 0,
                    start: 0.0,
                    end: 1.25,
                    text: " Hello.".into(),
                    no_speech_prob: 0.01,
                },
                AudioTranscriptionSegment {
                    id: 1,
                    start: 3_723.0,
                    end: 3_725.5,
                    text: " Bye. ".into(),
                    no_speech_prob: 0.02,
                },
            ],
        }
    }

    #[test]
    fn subtitles_number_cues_and_format_hours() {
        assert_eq!(
            srt(&timing()),
            "1\n00:00:00,000 --> 00:00:01,250\nHello.\n\n2\n01:02:03,000 --> 01:02:05,500\nBye.\n\n"
        );
        assert_eq!(
            vtt(&timing()),
            "WEBVTT\n\n00:00:00.000 --> 00:00:01.250\nHello.\n\n01:02:03.000 --> 01:02:05.500\nBye.\n\n"
        );
    }

    #[test]
    fn subtitle_cues_never_end_before_they_start() {
        let mut timing = timing();
        timing.segments[0].end = -1.0;
        assert!(srt(&timing).starts_with("1\n00:00:00,000 --> 00:00:00,000\n"));
    }

    #[test]
    fn verbose_json_names_the_task_and_carries_segments() {
        let response = AudioTranscriptionResponse {
            text: "Hello. Bye.".into(),
            timing: Some(timing()),
        };
        let body = verbose_json(&response, response.timing.as_ref().unwrap(), true);
        assert_eq!(body["task"], "translate");
        assert_eq!(body["language"], "en");
        assert_eq!(body["segments"][1]["start"], 3_723.0);
        assert_eq!(body["text"], "Hello. Bye.");
    }

    #[test]
    fn every_openai_response_format_is_accepted() {
        for format in ["json", "text", "verbose_json", "srt", "vtt"] {
            let mut request = transcription(None);
            request.response_format = format.into();
            assert!(request.validate().is_ok(), "{format}");
        }
        let mut request = transcription(None);
        request.response_format = "xml".into();
        assert!(request.validate().is_err());
    }

    #[test]
    /// Reject non-finite and out-of-range transcription temperatures.
    fn transcription_temperature_is_bounded_to_openai_range() {
        for temperature in [None, Some(0.0), Some(0.5), Some(1.0)] {
            assert!(transcription(temperature).validate().is_ok());
        }
        for temperature in [Some(-0.1), Some(1.1), Some(f32::NAN), Some(f32::INFINITY)] {
            assert!(transcription(temperature).validate().is_err());
        }
    }
}
