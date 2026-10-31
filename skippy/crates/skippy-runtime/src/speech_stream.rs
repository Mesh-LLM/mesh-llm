//! Live transcription over a growing audio stream.
//!
//! Whisper transcribes fixed windows, not streams. [`SpeechStream`] turns it
//! into a stream with the LocalAgreement policy: it re-transcribes the audio of
//! the current turn as more arrives, and confirms a word only once two
//! consecutive passes agree on it. Confirmed words never change, so they can be
//! sent as append-only deltas. Once the buffer grows long it is cut at the end
//! of the last fully confirmed segment, with the confirmed text carried forward
//! as the prompt, so each pass stays inside Whisper's 30 s window.
//!
//! Turns come from an energy voice-activity detector, or from the caller when
//! turn detection is off.

mod resample;

use std::collections::VecDeque;

use anyhow::Result;

pub use resample::PcmResampler;

use crate::logging::write_native_log_note;
use crate::speech_recognition::{SPEECH_SAMPLE_RATE, Transcription, TranscriptionOptions};

const SAMPLES_PER_MS: u64 = SPEECH_SAMPLE_RATE as u64 / 1000;
// Millisecond arithmetic below assumes a whole number of samples per ms.
const _: () = assert!(SPEECH_SAMPLE_RATE.is_multiple_of(1000));
/// Voice-activity frames are 20 ms.
const FRAME_SAMPLES: usize = 320;
/// Consecutive speech frames that open a turn, so a click does not.
const SPEECH_START_FRAMES: usize = 3;
/// Segments Whisper itself judges this likely to be silence are dropped:
/// on silence it tends to invent stock phrases.
const MAX_NO_SPEECH_PROB: f32 = 0.6;
/// Confirmed text kept as the prompt once its audio is trimmed away.
const PROMPT_TAIL_CHARS: usize = 200;
/// Audio a turn needs before any of its words are confirmed. With less,
/// Whisper may miss or mishear the first word in two passes alike, and a word
/// confirmed wrongly stays wrong; a pass a second becomes the default's two.
const MIN_CONFIRM_MS: u64 = 2_000;

/// Server-side turn detection, in the shape of OpenAI's `server_vad`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TurnDetection {
    /// 0.0 to 1.0; higher needs louder audio to count as speech.
    pub threshold: f32,
    /// Audio kept from before speech starts, so its first sound is not cut.
    pub prefix_padding_ms: u32,
    /// Silence that ends a turn.
    pub silence_duration_ms: u32,
}

impl Default for TurnDetection {
    fn default() -> Self {
        Self {
            threshold: 0.5,
            prefix_padding_ms: 300,
            silence_duration_ms: 500,
        }
    }
}

impl TurnDetection {
    /// Frame RMS above which a frame counts as speech: about -46 dBFS at
    /// threshold 0, -34 dBFS at the default, and -28 dBFS at 1.
    fn rms_threshold(&self) -> f32 {
        0.005 + self.threshold.clamp(0.0, 1.0) * 0.03
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct SpeechStreamConfig {
    /// `None` leaves turns to the caller's [`SpeechStream::commit`].
    pub turn_detection: Option<TurnDetection>,
    /// ISO 639-1 code; `None` detects the language.
    pub language: Option<String>,
    /// Text that conditions every pass, such as expected vocabulary.
    pub prompt: Option<String>,
    /// New audio between passes; shorter gives faster deltas for more work.
    pub step_ms: u32,
    /// Buffer length past which a pass trims confirmed audio.
    pub trim_after_ms: u32,
    /// Buffer length past which unconfirmed words are confirmed anyway, to
    /// stay inside Whisper's 30 s window.
    pub max_buffer_ms: u32,
    /// Report the words a pass heard but has not confirmed, so a caller can
    /// show speech a pass sooner than the deltas do.
    pub tentative: bool,
}

impl Default for SpeechStreamConfig {
    fn default() -> Self {
        Self {
            turn_detection: Some(TurnDetection::default()),
            language: None,
            prompt: None,
            step_ms: 1_000,
            trim_after_ms: 15_000,
            max_buffer_ms: 28_000,
            tentative: false,
        }
    }
}

/// What a stream reports; `turn` numbers each utterance from 0.
#[derive(Debug, Clone, PartialEq)]
pub enum SpeechStreamEvent {
    SpeechStarted {
        turn: u64,
        audio_start_ms: u64,
    },
    SpeechStopped {
        turn: u64,
        audio_end_ms: u64,
    },
    /// Newly confirmed text, to append to the turn's transcript.
    Delta {
        turn: u64,
        text: String,
    },
    /// The turn's whole transcript; no more deltas follow for it.
    Completed {
        turn: u64,
        transcript: String,
    },
    /// What the latest pass heard after the confirmed text, when
    /// [`SpeechStreamConfig::tentative`] is set. It may change: each replaces
    /// the last, an empty one clears it, and `Completed` supersedes it.
    Tentative {
        turn: u64,
        text: String,
    },
}

/// Transcribes 16 kHz mono PCM; a [`crate::SpeechRecognitionModel`] in
/// production.
pub type Transcribe<'a> = dyn FnMut(&[f32], &TranscriptionOptions) -> Result<Transcription> + 'a;

/// A live transcription session over 16 kHz mono PCM. See the module docs.
#[derive(Debug)]
pub struct SpeechStream {
    config: SpeechStreamConfig,
    /// Samples received so far, which places every event on the audio clock.
    received: u64,
    turn: u64,
    in_turn: bool,
    /// The current turn's audio not yet trimmed, starting at `buffer_start`.
    buffer: Vec<f32>,
    buffer_start: u64,
    buffer_has_speech: bool,
    /// Audio held before a turn starts, up to the prefix padding.
    preroll: VecDeque<f32>,
    /// Audio short of a whole voice-activity frame.
    partial_frame: Vec<f32>,
    speech_frames: usize,
    silence_samples: u64,
    /// Every word confirmed in this turn.
    confirmed: Vec<String>,
    /// How many of `confirmed` lie in `buffer`, rather than trimmed audio.
    confirmed_in_buffer: usize,
    /// The previous pass's words after the confirmed ones.
    pending: Vec<String>,
    /// The tentative text last reported, so an unchanged one isn't repeated.
    reported_tentative: String,
    last_pass_at: u64,
}

impl SpeechStream {
    pub fn new(config: SpeechStreamConfig) -> Self {
        Self {
            config,
            received: 0,
            turn: 0,
            in_turn: false,
            buffer: Vec::new(),
            buffer_start: 0,
            buffer_has_speech: false,
            preroll: VecDeque::new(),
            partial_frame: Vec::new(),
            speech_frames: 0,
            silence_samples: 0,
            confirmed: Vec::new(),
            confirmed_in_buffer: 0,
            pending: Vec::new(),
            reported_tentative: String::new(),
            last_pass_at: 0,
        }
    }

    pub fn config(&self) -> &SpeechStreamConfig {
        &self.config
    }

    /// Replaces the configuration immediately, including mid-turn: a new
    /// `turn_detection`, `step_ms`, or buffer bound governs the very next
    /// frame and pass. Language and prompt reach the next pass.
    pub fn set_config(&mut self, config: SpeechStreamConfig) {
        if config.tentative != self.config.tentative {
            // A consumer that stopped asking may have cleared its display;
            // the next pass must be free to repeat the last text it saw.
            self.reported_tentative.clear();
        }
        self.config = config;
    }

    /// Whether a turn is open with audio or confirmed words that its
    /// `Completed` has not reported yet.
    pub fn has_pending_audio(&self) -> bool {
        self.in_turn && (!self.buffer.is_empty() || !self.confirmed.is_empty())
    }

    /// The turn the next events belong to.
    pub fn current_turn(&self) -> u64 {
        self.turn
    }

    /// Adds 16 kHz mono audio, ending turns on silence and running a pass
    /// once `step_ms` of new audio has arrived.
    pub fn push(
        &mut self,
        samples: &[f32],
        transcribe: &mut Transcribe<'_>,
    ) -> Result<Vec<SpeechStreamEvent>> {
        let mut events = Vec::new();
        self.partial_frame.extend_from_slice(samples);
        let whole = self.partial_frame.len() / FRAME_SAMPLES * FRAME_SAMPLES;
        let frames = self.partial_frame.drain(..whole).collect::<Vec<_>>();
        for frame in frames.chunks(FRAME_SAMPLES) {
            self.received += frame.len() as u64;
            self.push_frame(frame, transcribe, &mut events)?;
        }
        if self.in_turn
            && self.received - self.last_pass_at >= u64::from(self.config.step_ms) * SAMPLES_PER_MS
        {
            self.run_pass(transcribe, &mut events)?;
        }
        Ok(events)
    }

    /// Ends the current turn now, confirming everything the final pass hears.
    /// Returns no events when no audio is pending.
    pub fn commit(&mut self, transcribe: &mut Transcribe<'_>) -> Result<Vec<SpeechStreamEvent>> {
        let mut events = Vec::new();
        if !self.partial_frame.is_empty() {
            let rest = std::mem::take(&mut self.partial_frame);
            self.received += rest.len() as u64;
            if self.in_turn {
                self.buffer.extend_from_slice(&rest);
            }
        }
        if self.has_pending_audio() {
            self.finish_turn(transcribe, &mut events)?;
        }
        Ok(events)
    }

    /// Drops the current turn's audio and unconfirmed words without a
    /// transcript. Tentative text already reported for that turn is replaced
    /// by an empty event, so a consumer stops showing the discarded words.
    pub fn clear(&mut self) -> Vec<SpeechStreamEvent> {
        let mut events = Vec::new();
        if self.config.tentative && !self.reported_tentative.is_empty() {
            events.push(SpeechStreamEvent::Tentative {
                turn: self.turn,
                text: String::new(),
            });
        }
        self.reset_turn();
        self.preroll.clear();
        self.partial_frame.clear();
        events
    }

    fn push_frame(
        &mut self,
        frame: &[f32],
        transcribe: &mut Transcribe<'_>,
        events: &mut Vec<SpeechStreamEvent>,
    ) -> Result<()> {
        // With turn detection off the caller owns turn boundaries, but the
        // default detector still decides whether a buffer holds speech worth
        // transcribing; silence is never sent to Whisper.
        let speech = rms(frame)
            > self
                .config
                .turn_detection
                .unwrap_or_default()
                .rms_threshold();
        if !self.in_turn {
            match self.config.turn_detection {
                None => self.start_turn(frame, None, speech),
                Some(detection) => {
                    self.preroll.extend(frame);
                    let keep = (u64::from(detection.prefix_padding_ms) * SAMPLES_PER_MS) as usize
                        + SPEECH_START_FRAMES * FRAME_SAMPLES;
                    while self.preroll.len() > keep {
                        self.preroll.pop_front();
                    }
                    self.speech_frames = if speech { self.speech_frames + 1 } else { 0 };
                    if self.speech_frames >= SPEECH_START_FRAMES {
                        let preroll = self.preroll.drain(..).collect::<Vec<_>>();
                        let start = self.received - preroll.len() as u64;
                        self.start_turn(&preroll, Some(start), true);
                        events.push(SpeechStreamEvent::SpeechStarted {
                            turn: self.turn,
                            audio_start_ms: start / SAMPLES_PER_MS,
                        });
                    }
                }
            }
            return Ok(());
        }
        self.buffer.extend_from_slice(frame);
        if speech {
            self.buffer_has_speech = true;
            self.silence_samples = 0;
        } else {
            self.silence_samples += frame.len() as u64;
        }
        if let Some(detection) = self.config.turn_detection
            && self.silence_samples >= u64::from(detection.silence_duration_ms) * SAMPLES_PER_MS
        {
            events.push(SpeechStreamEvent::SpeechStopped {
                turn: self.turn,
                audio_end_ms: (self.received - self.silence_samples) / SAMPLES_PER_MS,
            });
            self.finish_turn(transcribe, events)?;
        }
        Ok(())
    }

    fn start_turn(&mut self, audio: &[f32], start: Option<u64>, speech: bool) {
        self.in_turn = true;
        self.buffer = audio.to_vec();
        self.buffer_start = start.unwrap_or(self.received - audio.len() as u64);
        self.buffer_has_speech = speech;
        self.silence_samples = 0;
        self.speech_frames = 0;
        self.last_pass_at = self.buffer_start;
    }

    fn run_pass(
        &mut self,
        transcribe: &mut Transcribe<'_>,
        events: &mut Vec<SpeechStreamEvent>,
    ) -> Result<()> {
        self.last_pass_at = self.received;
        if !self.buffer_has_speech {
            // Silence is never transcribed, but with turn detection off it
            // still accumulates; keep it bounded like any other buffer.
            let max = u64::from(self.config.max_buffer_ms) * SAMPLES_PER_MS;
            if self.buffer.len() as u64 > max {
                self.buffer_start += self.buffer.len() as u64;
                self.buffer.clear();
            }
            return Ok(());
        }
        let segments = self.transcribe_buffer(transcribe)?;
        let words = segments.iter().flatten().cloned().collect::<Vec<_>>();
        if words.len() < self.confirmed_in_buffer {
            // Whisper now hears fewer words than were confirmed from this
            // audio (merged words, or a segment dropped as silence). The
            // confirmed words stand; realign so the next words can still be
            // confirmed rather than waiting for the count to catch up.
            write_native_log_note(format!(
                "speech stream desync: pass heard {} words, {} already confirmed; realigning",
                words.len(),
                self.confirmed_in_buffer
            ));
            self.confirmed_in_buffer = self
                .confirmed
                .last()
                .and_then(|last| words.iter().rposition(|word| same_word(word, last)))
                .map_or(words.len(), |position| position + 1);
            self.pending.clear();
        }
        let tail = &words[self.confirmed_in_buffer..];
        // Only a turn's start is held back; later words have the earlier ones
        // as context.
        let heard_ms = self.buffer.len() as u64 / SAMPLES_PER_MS;
        let agreed = if self.confirmed.is_empty() && heard_ms < MIN_CONFIRM_MS {
            0
        } else {
            tail.iter()
                .zip(&self.pending)
                .take_while(|(now, before)| same_word(now, before))
                .count()
        };
        let (newly_confirmed, pending) = tail.split_at(agreed);
        self.pending = pending.to_vec();
        self.confirm(newly_confirmed.to_vec(), events);

        let buffer_ms = self.buffer.len() as u64 / SAMPLES_PER_MS;
        if buffer_ms > u64::from(self.config.max_buffer_ms) {
            // No confirmed boundary arrived in time: confirm what is heard
            // and start a fresh window.
            let pending = std::mem::take(&mut self.pending);
            self.confirm(pending, events);
            self.buffer_start += self.buffer.len() as u64;
            self.buffer.clear();
            self.buffer_has_speech = false;
            self.confirmed_in_buffer = 0;
        } else if buffer_ms > u64::from(self.config.trim_after_ms) {
            self.trim_confirmed(&segments);
        }
        self.report_tentative(events);
        Ok(())
    }

    fn report_tentative(&mut self, events: &mut Vec<SpeechStreamEvent>) {
        if !self.config.tentative {
            return;
        }
        let mut text = self.pending.join(" ");
        // Like a delta, it reads on from the confirmed text.
        if !text.is_empty() && !self.confirmed.is_empty() {
            text.insert(0, ' ');
        }
        if text != self.reported_tentative {
            self.reported_tentative.clone_from(&text);
            events.push(SpeechStreamEvent::Tentative {
                turn: self.turn,
                text,
            });
        }
    }

    /// Cuts the buffer at the end of the last segment whose words are all
    /// confirmed.
    fn trim_confirmed(&mut self, segments: &[SegmentWords]) {
        let mut words_before = 0;
        let mut cut = None;
        for segment in segments {
            words_before += segment.words.len();
            if words_before > self.confirmed_in_buffer {
                break;
            }
            cut = Some((segment.end_ms, words_before));
        }
        let Some((end_ms, words)) = cut else {
            return;
        };
        let samples = ((end_ms.max(0) as u64) * SAMPLES_PER_MS).min(self.buffer.len() as u64);
        if samples == 0 {
            return;
        }
        self.buffer.drain(..samples as usize);
        self.buffer_start += samples;
        self.confirmed_in_buffer -= words;
    }

    fn finish_turn(
        &mut self,
        transcribe: &mut Transcribe<'_>,
        events: &mut Vec<SpeechStreamEvent>,
    ) -> Result<()> {
        if self.buffer_has_speech {
            let words = self
                .transcribe_buffer(transcribe)?
                .into_iter()
                .flatten()
                .collect::<Vec<_>>();
            let rest = words[self.confirmed_in_buffer.min(words.len())..].to_vec();
            self.confirm(rest, events);
        }
        events.push(SpeechStreamEvent::Completed {
            turn: self.turn,
            transcript: self.confirmed.join(" "),
        });
        self.turn += 1;
        self.reset_turn();
        Ok(())
    }

    fn reset_turn(&mut self) {
        self.in_turn = false;
        self.buffer.clear();
        self.buffer_has_speech = false;
        self.confirmed.clear();
        self.confirmed_in_buffer = 0;
        self.pending.clear();
        self.reported_tentative.clear();
        self.speech_frames = 0;
        self.silence_samples = 0;
    }

    fn confirm(&mut self, words: Vec<String>, events: &mut Vec<SpeechStreamEvent>) {
        if words.is_empty() {
            return;
        }
        let mut text = words.join(" ");
        if !self.confirmed.is_empty() {
            text.insert(0, ' ');
        }
        self.confirmed_in_buffer += words.len();
        self.confirmed.extend(words);
        events.push(SpeechStreamEvent::Delta {
            turn: self.turn,
            text,
        });
    }

    fn transcribe_buffer(&self, transcribe: &mut Transcribe<'_>) -> Result<Vec<SegmentWords>> {
        let options = TranscriptionOptions {
            language: self.config.language.clone(),
            prompt: self.prompt(),
            ..Default::default()
        };
        let transcription = transcribe(&self.buffer, &options)?;
        Ok(transcription
            .segments
            .into_iter()
            .filter(|segment| segment.no_speech_prob <= MAX_NO_SPEECH_PROB)
            .map(|segment| SegmentWords {
                end_ms: segment.end_ms,
                words: segment
                    .text
                    .split_whitespace()
                    .map(str::to_string)
                    .collect(),
            })
            .filter(|segment| !segment.words.is_empty())
            .collect())
    }

    /// The configured prompt, then the tail of the text confirmed from audio
    /// that has been trimmed away.
    fn prompt(&self) -> Option<String> {
        let trimmed = &self.confirmed[..self.confirmed.len() - self.confirmed_in_buffer];
        let carried = trimmed.join(" ");
        let carried = match carried.char_indices().rev().nth(PROMPT_TAIL_CHARS) {
            Some((index, _)) => carried[index..].trim_start().to_string(),
            None => carried,
        };
        match (self.config.prompt.as_deref(), carried.is_empty()) {
            (Some(prompt), false) => Some(format!("{prompt} {carried}")),
            (Some(prompt), true) => Some(prompt.to_string()),
            (None, false) => Some(carried),
            (None, true) => None,
        }
    }
}

#[derive(Debug, Clone)]
struct SegmentWords {
    end_ms: i64,
    words: Vec<String>,
}

impl IntoIterator for SegmentWords {
    type Item = String;
    type IntoIter = std::vec::IntoIter<String>;

    fn into_iter(self) -> Self::IntoIter {
        self.words.into_iter()
    }
}

impl<'a> IntoIterator for &'a SegmentWords {
    type Item = &'a String;
    type IntoIter = std::slice::Iter<'a, String>;

    fn into_iter(self) -> Self::IntoIter {
        self.words.iter()
    }
}

/// Words match across passes ignoring case and punctuation, which Whisper
/// revises as more context arrives.
fn same_word(a: &str, b: &str) -> bool {
    let normalize = |word: &str| {
        word.chars()
            .filter(|c| c.is_alphanumeric())
            .flat_map(char::to_lowercase)
            .collect::<String>()
    };
    let (a_norm, b_norm) = (normalize(a), normalize(b));
    if a_norm.is_empty() && b_norm.is_empty() {
        a == b
    } else {
        a_norm == b_norm
    }
}

fn rms(frame: &[f32]) -> f32 {
    (frame.iter().map(|s| s * s).sum::<f32>() / frame.len().max(1) as f32).sqrt()
}

#[cfg(test)]
mod tests;
