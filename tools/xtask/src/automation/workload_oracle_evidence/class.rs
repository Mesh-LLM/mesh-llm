use super::Error;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum WorkloadClass {
    Embedding,
    Rerank,
    EncoderDecoder,
    Ocr,
    SpeechSynthesis,
    SpeechRecognition,
}

impl WorkloadClass {
    pub(crate) fn parse(value: &str) -> Result<Self, Error> {
        match value {
            "embedding" => Ok(Self::Embedding),
            "rerank" => Ok(Self::Rerank),
            "encoder_decoder" => Ok(Self::EncoderDecoder),
            "ocr" => Ok(Self::Ocr),
            "speech_synthesis" => Ok(Self::SpeechSynthesis),
            "speech_recognition" => Ok(Self::SpeechRecognition),
            _ => Err(Error::UnknownClass(value.to_owned())),
        }
    }

    pub(crate) const fn name(self) -> &'static str {
        match self {
            Self::Embedding => "embedding",
            Self::Rerank => "rerank",
            Self::EncoderDecoder => "encoder_decoder",
            Self::Ocr => "ocr",
            Self::SpeechSynthesis => "speech_synthesis",
            Self::SpeechRecognition => "speech_recognition",
        }
    }

    pub(super) const fn executable(self) -> &'static str {
        match self {
            Self::Embedding | Self::Rerank | Self::Ocr | Self::SpeechRecognition => "llama-server",
            Self::EncoderDecoder => "llama-completion",
            Self::SpeechSynthesis => "llama-tts",
        }
    }

    pub(super) const fn requires_projector(self) -> bool {
        match self {
            Self::Ocr | Self::SpeechSynthesis | Self::SpeechRecognition => true,
            Self::Embedding | Self::Rerank | Self::EncoderDecoder => false,
        }
    }
}
