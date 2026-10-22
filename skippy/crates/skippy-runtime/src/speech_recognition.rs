use std::ffi::{CStr, CString, c_void};
use std::path::Path;
use std::ptr;

use anyhow::{Context, Result, anyhow};

use crate::error::ensure_ok;
use crate::logging::write_native_log_note;
use crate::path_cstring::path_to_cstring;

/// Sample rate of the PCM [`decode_audio`] returns and transcription expects.
pub const SPEECH_SAMPLE_RATE: u32 = skippy_ffi::ASR_SAMPLE_RATE;
/// Longest audio decoded or transcribed in one call: 30 minutes.
pub const MAX_SPEECH_SAMPLES: usize = skippy_ffi::ASR_MAX_SAMPLES;

/// Shape and weight memory of a loaded speech-recognition model.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SpeechRecognitionModelInfo {
    /// False for English-only (`.en`) Whisper models.
    pub multilingual: bool,
    pub vocab_size: u32,
    pub audio_layer_count: u32,
    pub text_layer_count: u32,
    pub mel_count: u32,
    pub weights_bytes: u64,
    pub on_accelerator: bool,
}

/// Options for one transcription.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct TranscriptionOptions {
    /// ISO 639-1 code such as `"en"`; `None` detects the language.
    pub language: Option<String>,
    /// Translate the speech into English instead of transcribing it.
    pub translate: bool,
    /// Text that conditions the first window, such as expected vocabulary.
    pub prompt: Option<String>,
    /// `0.0` decodes greedily; above it samples, with fallback on low confidence.
    pub temperature: f32,
    /// CPU threads for the parts that run on the CPU; `0` picks a default.
    /// Values above the available parallelism are clamped to it.
    pub threads: usize,
}

/// One timed segment of a transcription.
#[derive(Debug, Clone, PartialEq)]
pub struct TranscriptSegment {
    pub start_ms: i64,
    pub end_ms: i64,
    pub text: String,
    /// Probability that the segment holds no speech.
    pub no_speech_prob: f32,
}

/// The segments and detected language of one transcription.
#[derive(Debug, Clone, PartialEq)]
pub struct Transcription {
    pub segments: Vec<TranscriptSegment>,
    /// ISO 639-1 code of the spoken language; empty when it is unknown.
    pub language: String,
    pub duration_seconds: f64,
}

impl Transcription {
    /// The segments joined into one transcript with surrounding space trimmed.
    pub fn text(&self) -> String {
        self.segments
            .iter()
            .map(|segment| segment.text.as_str())
            .collect::<String>()
            .trim()
            .to_string()
    }
}

/// A loaded Whisper model.
///
/// Speech recognition does not load through `skippy_model_open`: it owns its
/// whisper.cpp model file and runs on the runtime's ggml backends.
pub struct SpeechRecognitionModel {
    raw: *mut skippy_ffi::AsrModel,
    info: SpeechRecognitionModelInfo,
}

// The native model is immutable after loading; every transcription allocates
// its own decoder state.
unsafe impl Send for SpeechRecognitionModel {}
unsafe impl Sync for SpeechRecognitionModel {}

impl SpeechRecognitionModel {
    /// Opens a whisper.cpp model file. `device` is `None` for the CPU backend,
    /// `"auto"` for the first GPU device, or a ggml backend device name such
    /// as `"MTL0"`.
    pub fn open(path: impl AsRef<Path>, device: Option<&str>) -> Result<Self> {
        ensure_speech_recognition_supported()?;
        let path = path.as_ref();
        write_native_log_note(format!(
            "skippy_asr_model_open begin path={}",
            path.display()
        ));
        let c_path = path_to_cstring(path, "speech model path")?;
        let c_device = device
            .map(|device| CString::new(device).context("speech device name contains a NUL byte"))
            .transpose()?;
        let mut raw = ptr::null_mut();
        let mut error = ptr::null_mut();
        let status = unsafe {
            skippy_ffi::skippy_asr_model_open(
                c_path.as_ptr(),
                c_device
                    .as_ref()
                    .map_or(ptr::null(), |device| device.as_ptr()),
                &mut raw,
                &mut error,
            )
        };
        ensure_ok(status, error)
            .with_context(|| format!("open speech model {}", path.display()))?;
        if raw.is_null() {
            return Err(anyhow!("skippy_asr_model_open returned a null handle"));
        }
        let mut model = Self {
            raw,
            info: SpeechRecognitionModelInfo {
                multilingual: false,
                vocab_size: 0,
                audio_layer_count: 0,
                text_layer_count: 0,
                mel_count: 0,
                weights_bytes: 0,
                on_accelerator: false,
            },
        };
        let mut info = skippy_ffi::AsrInfoV1::default();
        let mut error = ptr::null_mut();
        let status = unsafe { skippy_ffi::skippy_asr_model_info_v1(raw, &mut info, &mut error) };
        ensure_ok(status, error).context("read speech model info")?;
        model.info = SpeechRecognitionModelInfo {
            multilingual: info.multilingual,
            vocab_size: u32::try_from(info.n_vocab).unwrap_or(0),
            audio_layer_count: u32::try_from(info.n_audio_layer).unwrap_or(0),
            text_layer_count: u32::try_from(info.n_text_layer).unwrap_or(0),
            mel_count: u32::try_from(info.n_mels).unwrap_or(0),
            weights_bytes: info.weights_bytes,
            on_accelerator: info.on_accelerator,
        };
        write_native_log_note(format!(
            "skippy_asr_model_open done path={} multilingual={} on_accelerator={}",
            path.display(),
            model.info.multilingual,
            model.info.on_accelerator
        ));
        Ok(model)
    }

    pub fn info(&self) -> &SpeechRecognitionModelInfo {
        &self.info
    }

    /// Transcribes 16 kHz mono PCM, at most [`MAX_SPEECH_SAMPLES`].
    /// `cancelled` is polled between decoder steps; once it returns true the
    /// call fails instead of finishing. Backends poll it on the calling
    /// thread today; the `Sync` bound keeps it sound if one ever polls from
    /// a worker.
    pub fn transcribe<F>(
        &self,
        samples: &[f32],
        options: &TranscriptionOptions,
        cancelled: F,
    ) -> Result<Transcription>
    where
        F: Fn() -> bool + Sync,
    {
        if samples.is_empty() {
            return Err(anyhow!("transcription needs at least one audio sample"));
        }
        if samples.len() > MAX_SPEECH_SAMPLES {
            return Err(anyhow!("transcription takes at most 30 minutes of audio"));
        }
        let language = options
            .language
            .as_deref()
            .map(|language| CString::new(language).context("language contains a NUL byte"))
            .transpose()?;
        let prompt = options
            .prompt
            .as_deref()
            .map(|prompt| CString::new(prompt).context("prompt contains a NUL byte"))
            .transpose()?;
        let params = skippy_ffi::AsrTranscribeParamsV1 {
            language: language
                .as_ref()
                .map_or(ptr::null(), |value| value.as_ptr()),
            translate: options.translate,
            prompt: prompt.as_ref().map_or(ptr::null(), |value| value.as_ptr()),
            temperature: options.temperature.max(0.0),
            n_threads: clamp_threads(options.threads),
            ..Default::default()
        };
        let mut raw = ptr::null_mut();
        let mut error = ptr::null_mut();
        let status = unsafe {
            skippy_ffi::skippy_asr_transcribe(
                self.raw,
                samples.as_ptr(),
                samples.len(),
                &params,
                Some(poll_cancellation::<F>),
                &cancelled as *const F as *mut c_void,
                &mut raw,
                &mut error,
            )
        };
        ensure_ok(status, error).context("transcribe audio")?;
        if raw.is_null() {
            return Err(anyhow!("skippy_asr_transcribe returned a null result"));
        }
        let result = NativeResult(raw);
        result.read()
    }
}

impl Drop for SpeechRecognitionModel {
    fn drop(&mut self) {
        if !self.raw.is_null() {
            unsafe { skippy_ffi::skippy_asr_model_free(self.raw) };
        }
    }
}

/// Decodes a WAV, MP3, or FLAC upload to 16 kHz mono PCM. Audio longer than
/// [`MAX_SPEECH_SAMPLES`] is refused while decoding.
pub fn decode_audio(bytes: &[u8]) -> Result<Vec<f32>> {
    ensure_speech_recognition_supported()?;
    let mut samples = ptr::null_mut();
    let mut count = 0usize;
    let mut error = ptr::null_mut();
    let status = unsafe {
        skippy_ffi::skippy_asr_decode_audio(
            bytes.as_ptr(),
            bytes.len(),
            &mut samples,
            &mut count,
            &mut error,
        )
    };
    ensure_ok(status, error).context("decode audio upload")?;
    if samples.is_null() {
        return Err(anyhow!("skippy_asr_decode_audio returned no samples"));
    }
    let decoded = unsafe { std::slice::from_raw_parts(samples, count) }.to_vec();
    unsafe { skippy_ffi::skippy_asr_samples_free(samples) };
    Ok(decoded)
}

/// Whether the loaded native runtime was built with speech recognition.
pub fn speech_recognition_supported() -> bool {
    skippy_ffi::try_abi_features()
        .is_some_and(|features| features & skippy_ffi::FEATURE_SPEECH_RECOGNITION != 0)
}

fn ensure_speech_recognition_supported() -> Result<()> {
    if !speech_recognition_supported() {
        return Err(anyhow!(
            "native runtime was built without speech recognition"
        ));
    }
    Ok(())
}

/// Bridges the native abort poll to the caller's cancellation closure.
/// `user_data` is the `&F` that [`SpeechRecognitionModel::transcribe`]
/// passes and keeps alive for the whole native call. A panicking closure
/// cannot unwind into C, so it is treated as a request to cancel.
unsafe extern "C" fn poll_cancellation<F: Fn() -> bool>(user_data: *mut c_void) -> bool {
    let cancelled = unsafe { &*(user_data as *const F) };
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(cancelled)).unwrap_or(true)
}

/// Explicit thread counts are clamped to the available parallelism; zero
/// leaves the choice to the runtime.
fn clamp_threads(requested: usize) -> i32 {
    if requested == 0 {
        return 0;
    }
    let available = std::thread::available_parallelism().map_or(1, usize::from);
    i32::try_from(requested.min(available)).unwrap_or(i32::MAX)
}

/// Owns a native result until its segments are copied out.
struct NativeResult(*mut skippy_ffi::AsrResult);

impl NativeResult {
    fn read(&self) -> Result<Transcription> {
        let count = unsafe { skippy_ffi::skippy_asr_result_segment_count(self.0) };
        let mut segments = Vec::with_capacity(count);
        for index in 0..count {
            let mut segment = skippy_ffi::AsrSegmentV1::default();
            let mut error = ptr::null_mut();
            let status = unsafe {
                skippy_ffi::skippy_asr_result_segment_v1(self.0, index, &mut segment, &mut error)
            };
            ensure_ok(status, error).with_context(|| format!("read transcript segment {index}"))?;
            segments.push(TranscriptSegment {
                start_ms: segment.start_ms,
                end_ms: segment.end_ms,
                text: native_string(segment.text),
                no_speech_prob: segment.no_speech_prob,
            });
        }
        Ok(Transcription {
            segments,
            language: native_string(unsafe { skippy_ffi::skippy_asr_result_language(self.0) }),
            duration_seconds: unsafe { skippy_ffi::skippy_asr_result_duration_seconds(self.0) },
        })
    }
}

impl Drop for NativeResult {
    fn drop(&mut self) {
        unsafe { skippy_ffi::skippy_asr_result_free(self.0) };
    }
}

fn native_string(value: *const std::ffi::c_char) -> String {
    if value.is_null() {
        return String::new();
    }
    unsafe { CStr::from_ptr(value) }
        .to_string_lossy()
        .into_owned()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn transcript_text_joins_segments_and_trims() {
        let transcription = Transcription {
            segments: vec![
                TranscriptSegment {
                    start_ms: 0,
                    end_ms: 1_000,
                    text: " Hello".into(),
                    no_speech_prob: 0.0,
                },
                TranscriptSegment {
                    start_ms: 1_000,
                    end_ms: 2_000,
                    text: " world. ".into(),
                    no_speech_prob: 0.0,
                },
            ],
            language: "en".into(),
            duration_seconds: 2.0,
        };
        assert_eq!(transcription.text(), "Hello world.");
    }

    #[test]
    fn explicit_thread_counts_never_exceed_the_hardware() {
        let available = std::thread::available_parallelism().map_or(1, usize::from);
        assert_eq!(clamp_threads(0), 0);
        assert_eq!(clamp_threads(1), 1);
        assert_eq!(clamp_threads(usize::MAX), available as i32);
    }

    /// The model and clip named by `SKIPPY_ASR_TEST_MODEL` and
    /// `SKIPPY_ASR_TEST_AUDIO` (WAV/MP3/FLAC).
    fn real_model_and_clip() -> (SpeechRecognitionModel, Vec<f32>) {
        let model = std::env::var_os("SKIPPY_ASR_TEST_MODEL").expect("set SKIPPY_ASR_TEST_MODEL");
        let audio = std::env::var_os("SKIPPY_ASR_TEST_AUDIO").expect("set SKIPPY_ASR_TEST_AUDIO");
        let device = std::env::var("SKIPPY_ASR_TEST_DEVICE").ok();
        let model = SpeechRecognitionModel::open(model, device.as_deref()).unwrap();
        let samples = decode_audio(&std::fs::read(audio).unwrap()).unwrap();
        (model, samples)
    }

    /// One model serves concurrent transcriptions, each with its own decoder
    /// state, and they agree with each other.
    #[test]
    #[ignore = "needs SKIPPY_ASR_TEST_MODEL and SKIPPY_ASR_TEST_AUDIO"]
    fn concurrent_transcriptions_share_one_model() {
        let (model, samples) = real_model_and_clip();
        let texts = std::thread::scope(|scope| {
            let workers = (0..2)
                .map(|_| {
                    scope.spawn(|| {
                        model
                            .transcribe(&samples, &TranscriptionOptions::default(), || false)
                            .unwrap()
                            .text()
                    })
                })
                .collect::<Vec<_>>();
            workers
                .into_iter()
                .map(|worker| worker.join().unwrap())
                .collect::<Vec<_>>()
        });
        assert!(!texts[0].is_empty());
        assert_eq!(texts[0], texts[1]);
    }

    #[test]
    #[ignore = "needs SKIPPY_ASR_TEST_MODEL and SKIPPY_ASR_TEST_AUDIO"]
    fn transcribes_a_real_clip() {
        let (model, samples) = real_model_and_clip();
        let transcription = model
            .transcribe(&samples, &TranscriptionOptions::default(), || false)
            .unwrap();
        assert!(!transcription.text().is_empty());
        assert!(transcription.duration_seconds > 0.0);
        let cancelled = model.transcribe(&samples, &TranscriptionOptions::default(), || true);
        assert!(cancelled.is_err());
    }
}
