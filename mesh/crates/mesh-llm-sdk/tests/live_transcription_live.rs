//! Streams a WAV clip through a running node's live transcription. Needs a
//! node serving a speech model, so it is ignored by default:
//!
//! ```sh
//! MESH_LLM_LIVE_API=http://127.0.0.1:9337/v1 MESH_LLM_LIVE_WAV=speech.wav \
//!   cargo test -p mesh-llm-sdk --test live_transcription_live -- --ignored
//! ```
//!
//! The WAV must be 16-bit mono with a 44-byte header, and hold more than one
//! sentence of speech.

#![cfg(feature = "serving")]

use std::collections::HashMap;
use std::time::Duration;

use mesh_llm_sdk::live_transcription::{
    LiveTranscription, LiveTranscriptionEvent, LiveTranscriptionOptions,
};

#[tokio::test]
#[ignore = "needs a node serving a speech model; see the module docs"]
async fn streams_a_clip_and_reads_the_transcript() {
    let api = std::env::var("MESH_LLM_LIVE_API").expect("MESH_LLM_LIVE_API");
    let wav = std::env::var("MESH_LLM_LIVE_WAV").expect("MESH_LLM_LIVE_WAV");
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

    // Deltas per turn, and each turn's final transcript.
    let mut deltas = HashMap::<String, String>::new();
    let mut completed = Vec::new();
    let collect = async {
        while let Some(event) = events.next_event().await {
            match event {
                LiveTranscriptionEvent::Delta { item_id, text } => {
                    deltas.entry(item_id).or_default().push_str(&text);
                }
                LiveTranscriptionEvent::Completed {
                    item_id,
                    transcript,
                } => completed.push((item_id, transcript)),
                LiveTranscriptionEvent::Error { message } => panic!("{message}"),
                _ => {}
            }
        }
    };
    // The sender closes the session after the trailing silence; give the
    // last turn time to finish first.
    let finish = async {
        let audio = sender.await.unwrap();
        tokio::time::sleep(Duration::from_secs(10)).await;
        audio.close().await;
    };
    tokio::time::timeout(Duration::from_secs(120), async {
        tokio::join!(collect, finish)
    })
    .await
    .expect("the session finishes");

    assert!(!completed.is_empty(), "no turn completed");
    for (item_id, transcript) in &completed {
        assert!(!transcript.trim().is_empty(), "turn {item_id} is empty");
        let streamed = deltas.get(item_id).map(String::as_str).unwrap_or_default();
        assert_eq!(
            streamed.split_whitespace().collect::<Vec<_>>(),
            transcript.split_whitespace().collect::<Vec<_>>(),
            "turn {item_id}: deltas join to the transcript"
        );
    }
    let streamed_words = deltas
        .values()
        .map(|text| text.split_whitespace().count())
        .sum::<usize>();
    assert!(
        streamed_words >= 4,
        "words arrive as deltas: {streamed_words}"
    );
}
