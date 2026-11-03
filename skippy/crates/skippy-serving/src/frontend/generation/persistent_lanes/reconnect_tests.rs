use super::*;
use crate::frontend::InferenceCacheHints;
use skippy_protocol::binary::{
    StageWireMessage, read_stage_message, send_ready, write_stage_message,
};
use std::{io::Read, net::TcpListener, sync::mpsc, thread};

#[test]
fn checkout_replaces_a_remotely_closed_pooled_lane() {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let config = StageConfig {
        bind_addr: "127.0.0.1:0".into(),
        downstream: Some(skippy_protocol::PeerConfig {
            persistent_lane_marker: true,
            stage_id: "stage-1".into(),
            stage_index: 1,
            endpoint: listener.local_addr().unwrap().to_string(),
        }),
        ..StageConfig::default()
    };
    let (closed, closure) = mpsc::sync_channel(1);
    let server = thread::spawn(move || {
        let (mut stream, _) = listener.accept().unwrap();
        stream
            .set_read_timeout(Some(Duration::from_secs(2)))
            .unwrap();
        send_ready(&mut stream).unwrap();
        let mut marker = [0; 4];
        stream.read_exact(&mut marker).unwrap();
        assert_eq!(&marker, b"SPLN");
        drop(stream);
        closed.send(()).unwrap();

        let (mut replacement, _) = listener.accept().unwrap();
        replacement
            .set_read_timeout(Some(Duration::from_secs(2)))
            .unwrap();
        send_ready(&mut replacement).unwrap();
        replacement.read_exact(&mut marker).unwrap();
        assert_eq!(&marker, b"SPLN");
        read_stage_message(&mut replacement, 0).unwrap()
    });
    let telemetry = Telemetry::new(
        None,
        0,
        config.clone(),
        crate::telemetry::TelemetryLevel::Off,
    );
    let pool = PersistentStageLanePool::new(&config, 1, 2, telemetry)
        .unwrap()
        .unwrap();
    closure.recv_timeout(Duration::from_secs(2)).unwrap();
    // Observe FIN on a clone without consuming data or removing the pooled lane.
    let (old_id, socket) = {
        let lanes = pool.lanes.lock().unwrap();
        (lanes[0].id, lanes[0].stream.try_clone().unwrap())
    };
    socket
        .set_read_timeout(Some(Duration::from_secs(2)))
        .unwrap();
    assert_eq!(socket.peek(&mut [0]).unwrap(), 0);

    let ids =
        InferenceGenerationIds::new_with_trust(InferenceCacheHints::default(), None, false, None);
    let mut replacement = pool.checkout(&ids).unwrap();
    assert_ne!(replacement.id, old_id);
    write_stage_message(
        &mut replacement.stream,
        &StageWireMessage::stop_with_identity(23, 29),
    )
    .unwrap();
    let decoded = server.join().unwrap();
    assert_eq!((decoded.request_id, decoded.session_id), (23, 29));
}
