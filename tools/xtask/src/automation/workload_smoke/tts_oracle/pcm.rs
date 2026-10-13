//! PCM16 admission and complete-waveform comparison; no duration-only substitute.
use crate::{automation::canary_receipts::Digest, command::DynResult};
use serde::Serialize;
use std::path::Path;
#[derive(Serialize)]
pub(super) struct Metrics {
    pub sample_rate_hz: u32,
    pub channels: u16,
    pub sample_count: usize,
    pub relative_rms_error: f64,
    pub waveform_cosine: f64,
}
pub(super) struct Pcm {
    rate: u32,
    channels: u16,
    samples: Vec<i16>,
    pub digest: Digest,
}
fn u16_at(bytes: &[u8], at: usize) -> DynResult<u16> {
    Ok(u16::from_le_bytes(
        bytes
            .get(at..at + 2)
            .ok_or("truncated PCM16 format")?
            .try_into()?,
    ))
}
fn u32_at(bytes: &[u8], at: usize) -> DynResult<u32> {
    Ok(u32::from_le_bytes(
        bytes
            .get(at..at + 4)
            .ok_or("truncated WAV chunk")?
            .try_into()?,
    ))
}
pub(super) fn read(path: &Path) -> DynResult<Pcm> {
    decode(&super::regular_input::read(
        path,
        64 * 1024 * 1024,
        "TTS WAV",
    )?)
}
pub(super) fn decode(bytes: &[u8]) -> DynResult<Pcm> {
    if bytes.get(..4) != Some(b"RIFF") || bytes.get(8..12) != Some(b"WAVE") {
        return Err("not a RIFF WAVE file".into());
    }
    let limit = usize::try_from(u32_at(bytes, 4)?)?
        .checked_add(8)
        .ok_or("RIFF size overflow")?;
    if limit > bytes.len() || limit < 12 {
        return Err("truncated RIFF WAVE file".into());
    }
    let mut at = 12;
    let mut format = None;
    let mut data = None;
    while at < limit {
        let size = usize::try_from(u32_at(bytes, at + 4)?)?;
        let start = at.checked_add(8).ok_or("WAV offset overflow")?;
        let end = start.checked_add(size).ok_or("WAV chunk overflow")?;
        let chunk = bytes
            .get(start..end)
            .filter(|_| end <= limit)
            .ok_or("truncated WAV data")?;
        match bytes.get(at..at + 4).ok_or("truncated WAV header")? {
            b"fmt " => {
                if format.is_some() {
                    return Err("ambiguous WAV format chunks".into());
                }
                format = Some(chunk);
            }
            b"data" => {
                if data.is_some() {
                    return Err("ambiguous WAV data chunks".into());
                }
                data = Some(chunk);
            }
            _ => {}
        }
        at = end
            .checked_add(size % 2)
            .filter(|next| *next <= limit)
            .ok_or("truncated WAV padding")?;
    }
    let format = format.ok_or("missing WAV format")?;
    let data = data.ok_or("missing WAV data")?;
    let channels = u16_at(format, 2)?;
    let rate = u32_at(format, 4)?;
    if u16_at(format, 0)? != 1
        || u16_at(format, 14)? != 16
        || channels == 0
        || rate == 0
        || data.is_empty()
        || !data.len().is_multiple_of(usize::from(channels) * 2)
    {
        return Err("TTS requires nonempty uncompressed PCM16 frames".into());
    }
    Ok(Pcm {
        rate,
        channels,
        samples: data
            .as_chunks::<2>()
            .0
            .iter()
            .map(|sample| i16::from_le_bytes(*sample))
            .collect(),
        digest: Digest::of_bytes(bytes),
    })
}
pub(super) fn compare(a: &Pcm, b: &Pcm) -> DynResult<Metrics> {
    if a.rate != b.rate || a.channels != b.channels {
        return Err("TTS sample rate or channel count differs from monolithic oracle".into());
    }
    if a.samples.len() != b.samples.len() {
        return Err("TTS sample count differs from monolithic oracle".into());
    }
    let mut ae = 0.0;
    let mut be = 0.0;
    let mut delta = 0.0;
    let mut dot = 0.0;
    for (a, b) in a.samples.iter().zip(&b.samples) {
        let a = f64::from(*a);
        let b = f64::from(*b);
        ae += a * a;
        be += b * b;
        delta += (a - b).powi(2);
        dot += a * b;
    }
    let count = u32::try_from(a.samples.len())?;
    if ae <= f64::from(count) || be <= f64::from(count) {
        return Err("TTS candidate or monolithic oracle is silent".into());
    }
    let metrics = Metrics {
        sample_rate_hz: a.rate,
        channels: a.channels,
        sample_count: a.samples.len() / usize::from(a.channels),
        relative_rms_error: (delta / be).sqrt(),
        waveform_cosine: (dot / (ae * be).sqrt()).clamp(-1.0, 1.0),
    };
    crate::automation::workload_oracle_evidence::validate_tts_metrics(&serde_json::to_vec(
        &metrics,
    )?)?;
    Ok(metrics)
}
