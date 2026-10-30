use super::*;
use crate::binary_transport::binary_messaging::idle::IdleDeadlines;
use skippy_protocol::binary::write_stage_message;
use std::io::{Read, Write};
use std::net::TcpListener;

fn connected_pair() -> (TcpStream, TcpStream) {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let peer = TcpStream::connect(listener.local_addr().unwrap()).unwrap();
    let (upstream, _) = listener.accept().unwrap();
    peer.set_read_timeout(Some(Duration::from_secs(2))).unwrap();
    (peer, upstream)
}

fn control() -> Arc<ConnectionWorkerControl> {
    Arc::new(ConnectionWorkerControl {
        idle_deadlines: IdleDeadlines {
            ordinary: Duration::from_millis(30),
            persistent_lane: Duration::from_millis(300),
            frame_decode: Duration::from_millis(50),
        },
        ..ConnectionWorkerControl::default()
    })
}

fn reader(upstream: &TcpStream, control: Arc<ConnectionWorkerControl>) -> InboundMessageReader {
    spawn_message_reader(
        upstream,
        0,
        StageActivationCodec::default(),
        StageActivationCodecPolicy::default(),
        1,
        Arc::new(StaleDiscardRegistry::default()),
        control,
    )
    .unwrap()
}

#[test]
fn ordinary_idle_after_first_message_closes_cleanly() {
    let (mut peer, upstream) = connected_pair();
    let control = control();
    let reader = reader(&upstream, control.clone());
    write_stage_message(&mut peer, &StageWireMessage::stop_with_identity(7, 11)).unwrap();
    assert_eq!(reader.next(None, 0, 0).unwrap().unwrap().request_id, 7);
    let (done, finished) = mpsc::sync_channel(1);
    let worker = thread::spawn(move || {
        let result = reader.next(None, 0, 1);
        drop(reader);
        drop(upstream);
        done.send(result).unwrap();
    });

    let result = finished.recv_timeout(Duration::from_secs(1));
    if result.is_err() {
        control.shutdown();
        peer.shutdown(Shutdown::Both).unwrap();
    }
    worker.join().unwrap();
    assert!(
        result
            .expect("idle deadline must release the worker")
            .unwrap()
            .is_none()
    );
    assert_eq!(peer.read(&mut [0]).unwrap(), 0);
}

#[test]
fn marked_lane_outlives_ordinary_idle_then_closes() {
    let (mut peer, mut upstream) = connected_pair();
    let control = control();
    crate::binary_transport::connection_mode::mark_persistent_lane(&mut peer).unwrap();
    write_stage_message(&mut peer, &StageWireMessage::stop_with_identity(7, 11)).unwrap();
    super::super::first_message::read_first_message(
        &control,
        &mut upstream,
        &skippy_protocol::StageConfig::default(),
        0,
        Duration::from_secs(1),
    )
    .unwrap()
    .unwrap();
    let reader = reader(&upstream, control.clone());
    let (done, finished) = mpsc::sync_channel(1);
    let worker = thread::spawn(move || {
        let result = reader.next(None, 0, 1);
        drop(reader);
        drop(upstream);
        done.send(result).unwrap();
    });
    assert!(matches!(
        finished.recv_timeout(Duration::from_millis(150)),
        Err(mpsc::RecvTimeoutError::Timeout)
    ));
    let result = finished.recv_timeout(Duration::from_secs(2));
    if result.is_err() {
        control.shutdown();
        peer.shutdown(Shutdown::Both).unwrap();
    }
    worker.join().unwrap();
    assert!(result.unwrap().unwrap().is_none());
    assert_eq!(peer.read(&mut [0]).unwrap(), 0);
}

#[test]
fn marked_lane_first_message_wait_expires_cleanly() {
    let (mut peer, mut upstream) = connected_pair();
    let control = control();
    crate::binary_transport::connection_mode::mark_persistent_lane(&mut peer).unwrap();
    let task_control = control.clone();
    let (done, finished) = mpsc::sync_channel(1);
    let worker = thread::spawn(move || {
        let result = super::super::first_message_prefix(
            &task_control,
            &mut upstream,
            Duration::from_secs(1),
        );
        drop(upstream);
        done.send(result).unwrap();
    });
    assert!(matches!(
        finished.recv_timeout(Duration::from_millis(150)),
        Err(mpsc::RecvTimeoutError::Timeout)
    ));
    let result = finished.recv_timeout(Duration::from_secs(2));
    if result.is_err() {
        control.shutdown();
        peer.shutdown(Shutdown::Both).unwrap();
    }
    worker.join().unwrap();
    assert!(result.unwrap().unwrap().is_none());
    assert_eq!(peer.read(&mut [0]).unwrap(), 0);
}

#[test]
fn active_peer_resets_idle_deadline_between_messages() {
    let (mut peer, upstream) = connected_pair();
    let control = control();
    let reader = reader(&upstream, control);
    let started = std::time::Instant::now();
    let mut count = 0;
    while started.elapsed() < Duration::from_millis(600) {
        write_stage_message(&mut peer, &StageWireMessage::stop_with_identity(count, 11)).unwrap();
        assert_eq!(reader.next(None, 0, 1).unwrap().unwrap().request_id, count);
        count += 1;
    }
    assert!(count > 1);
}

#[test]
fn shutdown_interrupts_ordinary_and_marked_idle_readers() {
    for persistent in [false, true] {
        let (_peer, upstream) = connected_pair();
        let control = Arc::new(ConnectionWorkerControl {
            idle_deadlines: IdleDeadlines {
                ordinary: Duration::from_secs(60),
                persistent_lane: Duration::from_secs(60),
                frame_decode: Duration::from_secs(60),
            },
            ..ConnectionWorkerControl::default()
        });
        control.persistent_lane.store(persistent, Ordering::Release);
        control.track(&upstream).unwrap();
        let reader = reader(&upstream, control.clone());
        let (done, finished) = mpsc::sync_channel(1);
        let worker = thread::spawn(move || {
            let result = reader.next(None, 0, 0);
            drop(reader);
            done.send(result).unwrap();
        });
        control.shutdown();
        assert!(
            finished
                .recv_timeout(Duration::from_secs(2))
                .unwrap()
                .unwrap()
                .is_none()
        );
        worker.join().unwrap();
    }
}

#[test]
fn request_processing_longer_than_idle_deadline_keeps_the_connection() {
    let (mut peer, upstream) = connected_pair();
    let control = control();
    let reader = reader(&upstream, control);
    write_stage_message(&mut peer, &StageWireMessage::stop_with_identity(7, 11)).unwrap();
    assert_eq!(reader.next(None, 0, 0).unwrap().unwrap().request_id, 7);
    // The executor holds request 7 well past the 30 ms ordinary idle deadline;
    // time spent processing is the behavior under test.
    thread::sleep(Duration::from_millis(200));
    write_stage_message(&mut peer, &StageWireMessage::stop_with_identity(8, 11)).unwrap();
    assert_eq!(
        reader
            .next(None, 0, 1)
            .unwrap()
            .expect("an in-flight request must not let the idle deadline close the connection")
            .request_id,
        8
    );
}

fn encoded_stop(request_id: u64) -> Vec<u8> {
    let mut bytes = Vec::new();
    write_stage_message(
        &mut bytes,
        &StageWireMessage::stop_with_identity(request_id, 11),
    )
    .unwrap();
    bytes
}

#[test]
fn stalled_later_frame_is_dropped_at_the_decode_deadline() {
    let (mut peer, upstream) = connected_pair();
    let control = Arc::new(ConnectionWorkerControl {
        idle_deadlines: IdleDeadlines {
            ordinary: Duration::from_secs(60),
            persistent_lane: Duration::from_secs(60),
            frame_decode: Duration::from_millis(50),
        },
        ..ConnectionWorkerControl::default()
    });
    let reader = reader(&upstream, control.clone());
    write_stage_message(&mut peer, &StageWireMessage::stop_with_identity(7, 11)).unwrap();
    assert_eq!(reader.next(None, 0, 0).unwrap().unwrap().request_id, 7);
    let frame = encoded_stop(8);
    peer.write_all(&frame[..frame.len() / 2]).unwrap();
    let (done, finished) = mpsc::sync_channel(1);
    let worker = thread::spawn(move || {
        let result = reader.next(None, 0, 1);
        drop(reader);
        done.send(result.map(|message| message.map(|message| message.request_id)))
            .unwrap();
    });

    let result = finished.recv_timeout(Duration::from_secs(2));
    if result.is_err() {
        control.shutdown();
        peer.shutdown(Shutdown::Both).unwrap();
    }
    worker.join().unwrap();
    let error = result
        .expect("a stalled later frame must release its worker at the decode deadline")
        .expect_err("a half-sent frame must not decode");
    assert!(
        error.chain().any(
            |cause| cause.downcast_ref::<io::Error>().is_some_and(|io| matches!(
                io.kind(),
                io::ErrorKind::TimedOut | io::ErrorKind::WouldBlock
            ))
        ),
        "{error:#}"
    );
}

#[test]
fn slow_but_progressing_later_frame_is_not_cut_off() {
    let (mut peer, upstream) = connected_pair();
    let control = Arc::new(ConnectionWorkerControl {
        idle_deadlines: IdleDeadlines {
            ordinary: Duration::from_secs(60),
            persistent_lane: Duration::from_secs(60),
            frame_decode: Duration::from_millis(400),
        },
        ..ConnectionWorkerControl::default()
    });
    let reader = reader(&upstream, control);
    write_stage_message(&mut peer, &StageWireMessage::stop_with_identity(7, 11)).unwrap();
    assert_eq!(reader.next(None, 0, 0).unwrap().unwrap().request_id, 7);
    let frame = encoded_stop(8);
    let (head, tail) = frame.split_at(frame.len() / 2);
    peer.write_all(head).unwrap();
    // The rest arrives after a pause well inside the decode deadline; the pacing
    // of the peer's writes is the behavior under test.
    thread::sleep(Duration::from_millis(100));
    peer.write_all(tail).unwrap();
    assert_eq!(reader.next(None, 0, 1).unwrap().unwrap().request_id, 8);
}
