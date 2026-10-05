use super::*;

#[tokio::test]
async fn queue_bounds_slow_consumers_and_cancellation() {
    let (queue, mut receiver) = ConnectionQueue::new(1);
    queue.try_send("first".into()).expect("first fits");
    assert_eq!(
        queue.try_send("second".into()),
        Err(QueueError::SlowConsumer)
    );
    assert_eq!(receiver.recv().await.as_deref(), Some("first"));
    queue.cancel();
    assert_eq!(
        queue.try_send("after-cancel".into()),
        Err(QueueError::Cancelled)
    );
    assert!(receiver.recv().await.is_none());
}

#[tokio::test]
async fn queue_times_out_a_slow_socket_writer_without_unbounded_growth() {
    let (queue, _receiver) = ConnectionQueue::new(1);
    queue.try_send("first".into()).unwrap();
    assert_eq!(
        queue
            .send_with_timeout("second".into(), std::time::Duration::from_millis(5))
            .await,
        Err(QueueError::SlowConsumer)
    );
}
