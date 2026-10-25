use crate::command::DynResult;
use serde::Deserialize;
#[derive(Deserialize)]
struct Usage {
    prompt_tokens: Option<u64>,
    completion_tokens: Option<u64>,
}
#[derive(Deserialize)]
struct Rerank {
    results: Vec<Rank>,
    usage: Usage,
}
#[derive(Deserialize)]
struct Rank {
    index: usize,
    relevance_score: f64,
    document: String,
}
pub(super) fn rerank(bytes: &[u8]) -> DynResult<()> {
    let response: Rerank = serde_json::from_slice(bytes)?;
    if response.results.len() != 2 || response.usage.prompt_tokens.unwrap_or(0) == 0 {
        return Err("rerank cardinality or usage failed".into());
    }
    let mut scores = [None; 2];
    for row in response.results {
        if row.index > 1 || scores[row.index].is_some() || !row.relevance_score.is_finite() {
            return Err("rerank indexes or scores failed".into());
        }
        let _document = row.document;
        scores[row.index] = Some(row.relevance_score);
    }
    if scores[0] <= scores[1] {
        return Err("rerank semantic ordering failed".into());
    }
    Ok(())
}
#[derive(Deserialize)]
struct Completion {
    choices: Vec<Text>,
    usage: Usage,
}
#[derive(Deserialize)]
struct Text {
    text: String,
}
pub(super) fn completion(bytes: &[u8]) -> DynResult<()> {
    let response: Completion = serde_json::from_slice(bytes)?;
    if response.choices.len() != 1
        || !response.choices[0].text.to_lowercase().contains("haus")
        || response.usage.completion_tokens.unwrap_or(0) == 0
    {
        return Err("translation anchor or usage failed".into());
    }
    Ok(())
}
#[derive(Deserialize)]
struct Chat {
    choices: Vec<Choice>,
}
#[derive(Deserialize)]
struct Choice {
    message: Transcription,
}
#[derive(Deserialize)]
struct Transcription {
    #[serde(alias = "content")]
    text: String,
}
pub(super) fn ocr(bytes: &[u8]) -> DynResult<()> {
    let response: Chat = serde_json::from_slice(bytes)?;
    if response.choices.len() != 1 || response.choices[0].message.text.trim().is_empty() {
        return Err("OCR transcription is empty".into());
    }
    Ok(())
}
pub(super) fn transcription(bytes: &[u8]) -> DynResult<()> {
    let response: Transcription = serde_json::from_slice(bytes)?;
    if response.text.trim().is_empty() {
        return Err("transcription is empty".into());
    }
    Ok(())
}
pub(super) fn audio(bytes: &[u8]) -> DynResult<()> {
    if bytes.get(..4) != Some(b"RIFF") || bytes.get(8..12) != Some(b"WAVE") {
        return Err("speech output is not WAV".into());
    }
    let mut position = 12_usize;
    let mut format = None;
    let mut samples = None;
    while position
        .checked_add(8)
        .is_some_and(|end| end <= bytes.len())
    {
        let length = usize::try_from(u32::from_le_bytes(
            bytes[position + 4..position + 8].try_into()?,
        ))?;
        let start = position + 8;
        let end = start.checked_add(length).ok_or("WAV length overflow")?;
        let chunk = bytes.get(start..end).ok_or("truncated WAV chunk")?;
        match &bytes[position..position + 4] {
            b"fmt " if chunk.len() >= 16 => {
                format = Some((
                    u16::from_le_bytes(chunk[0..2].try_into()?),
                    u16::from_le_bytes(chunk[2..4].try_into()?),
                    u32::from_le_bytes(chunk[4..8].try_into()?),
                    u16::from_le_bytes(chunk[14..16].try_into()?),
                ))
            }
            b"data" => samples = Some(chunk),
            _ => {}
        }
        position = end.checked_add(length % 2).ok_or("WAV length overflow")?;
    }
    let (kind, channels, rate, bits) = format.ok_or("missing WAV format")?;
    let samples = samples.ok_or("missing WAV samples")?;
    if kind != 1 || channels == 0 || bits != 16 || rate == 0 || !samples.len().is_multiple_of(2) {
        return Err("unsupported WAV samples".into());
    }
    let frames = samples.len() / 2 / usize::from(channels);
    if frames < usize::try_from(rate / 10)? || samples.is_empty() {
        return Err("too little audio".into());
    }
    let energy = samples
        .as_chunks::<2>()
        .0
        .iter()
        .map(|chunk| f64::from(i16::from_le_bytes([chunk[0], chunk[1]])).powi(2))
        .sum::<f64>();
    let count = u32::try_from(samples.len() / 2)?;
    if (energy / f64::from(count)).sqrt() < 1.0 {
        return Err("silent audio".into());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn rerank_rejects_duplicates_wrong_order_and_missing_usage() {
        for body in [
            r#"{"results":[{"index":0,"relevance_score":1,"document":"a"},{"index":0,"relevance_score":2,"document":"b"}],"usage":{"prompt_tokens":1}}"#,
            r#"{"results":[{"index":0,"relevance_score":1,"document":"a"},{"index":1,"relevance_score":2,"document":"b"}],"usage":{"prompt_tokens":1}}"#,
            r#"{"results":[{"index":0,"relevance_score":2,"document":"a"},{"index":1,"relevance_score":1,"document":"b"}],"usage":{}}"#,
        ] {
            assert!(rerank(body.as_bytes()).is_err());
        }
    }
    #[test]
    fn translation_and_transcription_require_output() {
        assert!(
            completion(br#"{"choices":[{"text":"hello"}],"usage":{"completion_tokens":1}}"#)
                .is_err()
        );
        assert!(ocr(br#"{"choices":[{"message":{"content":" "}}]}"#).is_err());
        assert!(transcription(br#"{"text":""}"#).is_err());
    }
    #[test]
    fn audio_rejects_empty_and_truncated_frames() {
        assert!(audio(b"").is_err());
        assert!(audio(b"RIFF\0\0\0\0WAVEdata\xff\xff\xff\xff").is_err());
    }
}
