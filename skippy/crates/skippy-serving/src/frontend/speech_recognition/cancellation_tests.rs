use super::*;

#[tokio::test]
async fn cancellation_before_audio_releases_the_backend_even_if_input_is_retained() {
    let backend = SpeechRecognitionBackend::new("speech", Arc::new(tests::FakeModel::default()), 1);
    let context = InferenceRequestContext::new();
    let mut session = backend
        .realtime_transcription(RealtimeTranscriptionConfig::new("speech"), context.clone())
        .await
        .unwrap();
    context.cancel();
    tokio::time::timeout(Duration::from_secs(5), session.input.closed())
        .await
        .unwrap();
    assert!(session.output.recv().await.is_none());
}

#[tokio::test]
async fn cancellation_unblocks_a_full_output_queue() {
    let (output, mut received) = mpsc::channel(1);
    output.send(RealtimeOutput::CommitEmpty).await.unwrap();
    let cancellation = CancellationToken::new();
    let live = LiveSession {
        model: Arc::new(tests::FakeModel::default()),
        lanes: Arc::new(Semaphore::new(1)),
        runtime: tokio::runtime::Handle::current(),
        cancellation: cancellation.clone(),
        idle_timeout: Duration::from_secs(30),
        output,
    };
    let task = task::spawn_blocking(move || live.send(RealtimeOutput::CommitEmpty));
    cancellation.cancel();
    tokio::time::timeout(Duration::from_secs(5), task)
        .await
        .unwrap()
        .unwrap();
    assert_eq!(received.recv().await, Some(RealtimeOutput::CommitEmpty));
    assert!(received.recv().await.is_none());
}
