//! Streams a WAV clip through a running node's live transcription when
//! `MESH_LLM_LIVE_API` (an API base URL such as `http://127.0.0.1:9337/v1`)
//! and `MESH_LLM_LIVE_WAV` (16-bit mono WAV) are set; otherwise does nothing.

#![cfg(feature = "serving")]

use std::time::Duration;

use mesh_llm_sdk::live_transcription::{
    LiveTranscription, LiveTranscriptionEvent, LiveTranscriptionOptions,
};

#[tokio::test]
async fn streams_a_clip_and_reads_the_transcript() {
    let (Ok(api), Ok(wav)) = (
        std::env::var("MESH_LLM_LIVE_API"),
        std::env::var("MESH_LLM_LIVE_WAV"),
    ) else {
        return;
    };
    let bytes = std::fs::read(wav).unwrap();
    // A canonical 44-byte header: the sample rate sits at byte 24.
    let rate = u32::from_le_bytes(bytes[24..28].try_into().unwrap());
    let samples = bytes[44..]
        .as_chunks::<2>()
        .0
        .iter()
        .map(|pair| i16::from_le_bytes(*pair))
        .collect::<Vec<_>>();

    let live = LiveTranscription::start(
        &api,
        LiveTranscriptionOptions {
            sample_rate: rate,
            ..Default::default()
        },
    )
    .await
    .unwrap();
    let (audio, mut events) = live.split();
    let sender = tokio::spawn(async move {
        // Real-time pace, 100 ms at a time, then a second of silence.
        let frame = rate as usize / 10;
        for chunk in samples.chunks(frame) {
            audio.send_audio(chunk).await.unwrap();
            tokio::time::sleep(Duration::from_millis(100)).await;
        }
        audio.send_audio(&vec![0; rate as usize]).await.unwrap();
        audio
    });

    let mut deltas = 0;
    let mut transcript = String::new();
    let collect = async {
        while let Some(event) = events.next_event().await {
            match event {
                LiveTranscriptionEvent::Delta { .. } => deltas += 1,
                LiveTranscriptionEvent::Completed {
                    transcript: text, ..
                } => {
                    transcript.push_str(&text);
                    transcript.push(' ');
                    if transcript.to_lowercase().contains("your country.")
                        && transcript.matches(' ').count() > 15
                    {
                        break;
                    }
                }
                LiveTranscriptionEvent::Error { message } => panic!("{message}"),
                _ => {}
            }
        }
    };
    tokio::time::timeout(Duration::from_secs(60), collect)
        .await
        .expect("transcript arrives");
    sender.await.unwrap().close().await;
    let words = transcript.to_lowercase();
    assert!(deltas >= 4, "words arrive as deltas: {deltas}");
    assert!(words.contains("ask not"), "{transcript}");
    assert!(
        words.contains("what you can do for your country"),
        "{transcript}"
    );
}
