use super::*;

fn phrase_occurrences(text: &str, phrase: &str) -> usize {
    let words = text
        .split(|character: char| !character.is_alphanumeric())
        .filter(|word| !word.is_empty())
        .map(str::to_lowercase)
        .collect::<Vec<_>>();
    let phrase = phrase
        .split_whitespace()
        .map(str::to_string)
        .collect::<Vec<_>>();
    words
        .windows(phrase.len())
        .filter(|window| *window == phrase.as_slice())
        .count()
}

#[test]
#[ignore = "requires SKIPPY_ASR_TEST_MODEL and SKIPPY_ASR_TEST_AUDIO (11-second JFK fixture)"]
fn streams_four_real_repetitions_without_replaying_trimmed_speech() {
    stream_four_repetitions(400);
}

#[test]
#[ignore = "requires SKIPPY_ASR_TEST_MODEL and SKIPPY_ASR_TEST_AUDIO (11-second JFK fixture)"]
fn streams_four_real_repetitions_at_default_cadence() {
    stream_four_repetitions(SpeechStreamConfig::default().step_ms);
}

fn stream_four_repetitions(step_ms: u32) {
    crate::test_support::require_native_runtime();
    let model = std::env::var_os("SKIPPY_ASR_TEST_MODEL").expect("set SKIPPY_ASR_TEST_MODEL");
    let audio = std::env::var_os("SKIPPY_ASR_TEST_AUDIO").expect("set SKIPPY_ASR_TEST_AUDIO");
    let device = std::env::var("SKIPPY_ASR_TEST_DEVICE").ok();
    let model = crate::SpeechRecognitionModel::open(model, device.as_deref()).unwrap();
    let clip = crate::decode_audio(&std::fs::read(audio).unwrap()).unwrap();
    assert_eq!(
        clip.len(),
        11 * 16_000,
        "use the pinned 11-second JFK fixture"
    );
    let samples = clip.repeat(4);
    let options = TranscriptionOptions {
        language: Some("en".into()),
        ..Default::default()
    };
    let offline = model
        .transcribe(&samples, &options, || false)
        .unwrap()
        .text();
    assert_eq!(
        offline.to_lowercase().matches("fellow americans").count(),
        4
    );
    let pass = std::cell::Cell::new(0);
    let mut transcribe = |s: &[f32], o: &TranscriptionOptions| {
        let result = model.transcribe(s, o, || false)?;
        pass.set(pass.get() + 1);
        eprintln!(
            "asr_pass={} samples={} prompt={:?} segments={:?}",
            pass.get(),
            s.len(),
            o.prompt,
            result.segments
        );
        Ok(result)
    };
    let mut stream = SpeechStream::new(SpeechStreamConfig {
        language: Some("en".into()),
        step_ms,
        tentative: true,
        ..manual()
    });
    let mut events = Vec::new();
    for chunk in samples.chunks(1_600) {
        let before = (pass.get(), stream.buffer_start);
        let batch = stream.push(chunk, &mut transcribe).unwrap();
        if pass.get() != before.0 || stream.buffer_start != before.1 || !batch.is_empty() {
            eprintln!(
                "stream_pass={} received={} window_start={} window_samples={} canonical_words={} hypothesis_words={} events={batch:?}",
                pass.get(),
                stream.received,
                stream.buffer_start,
                stream.buffer.len(),
                stream.confirmed_in_buffer,
                stream.confirmed_prefix.observed_len()
            );
        }
        events.extend(batch);
    }
    let final_events = stream.commit(&mut transcribe).unwrap();
    eprintln!("final_events={final_events:?}");
    events.extend(final_events);
    let Some(SpeechStreamEvent::Completed { transcript, .. }) = events.last() else {
        panic!("the manual turn must finish");
    };
    eprintln!("offline transcript: {offline}");
    eprintln!("live transcript: {transcript}");
    assert_eq!(deltas(&events).concat(), *transcript);
    for phrase in [
        "fellow americans",
        "can do for you",
        "what you can do",
        "do for your country",
        "for your country",
    ] {
        assert_eq!(phrase_occurrences(&offline, phrase), 4);
        assert_eq!(
            phrase_occurrences(transcript, phrase),
            4,
            "rolling clause count for {phrase:?}; offline: {offline}; live: {transcript}"
        );
    }
}
