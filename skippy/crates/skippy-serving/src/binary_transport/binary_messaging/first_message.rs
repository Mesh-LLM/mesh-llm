//! Decode the first frame before handing the connection to its session worker.

use std::{
    io::{self, Read},
    net::TcpStream,
    time::Duration,
};

use skippy_protocol::{
    MAX_STAGE_FRAME_BYTES, StageConfig,
    binary::{StageWireMessage, read_stage_message_for_codec_policy_with_frame_budget},
};

use super::{ConnectionWorkerControl, first_message_prefix};

pub(super) fn read_first_message(
    control: &ConnectionWorkerControl,
    upstream: &mut TcpStream,
    config: &StageConfig,
    input_activation_width: i32,
    deadline: Duration,
) -> io::Result<Option<StageWireMessage>> {
    let Some(prefix) = first_message_prefix(control, upstream, deadline)? else {
        return Ok(None);
    };
    upstream.set_read_timeout((!prefix.is_empty()).then_some(deadline))?;
    let result = match read_stage_message_for_codec_policy_with_frame_budget(
        io::Cursor::new(prefix).chain(&mut *upstream),
        input_activation_width,
        config.activation_codec,
        config.activation_codec_policy,
        MAX_STAGE_FRAME_BYTES,
    ) {
        Ok(message) => Ok(Some(message)),
        Err(error) if error.kind() == io::ErrorKind::UnexpectedEof => Ok(None),
        Err(error) => Err(error),
    };
    upstream.set_read_timeout(None)?;
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{io::Write, net::TcpListener, sync::mpsc, thread};

    #[test]
    fn ordinary_partial_first_frame_is_dropped_at_deadline() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let mut peer = TcpStream::connect(listener.local_addr().unwrap()).unwrap();
        let (mut server, _) = listener.accept().unwrap();
        let cleanup = server.try_clone().unwrap();
        let (done, finished) = mpsc::sync_channel(1);
        let worker = thread::spawn(move || {
            let result = read_first_message(
                &ConnectionWorkerControl::default(),
                &mut server,
                &StageConfig::default(),
                0,
                Duration::from_millis(50),
            );
            done.send(result).unwrap();
        });

        peer.write_all(&1_i32.to_le_bytes()).unwrap();
        let result = finished.recv_timeout(Duration::from_secs(2));
        cleanup.shutdown(std::net::Shutdown::Both).unwrap();
        worker.join().unwrap();

        let error = result
            .expect("partial first frame must release its worker at the deadline")
            .unwrap_err();
        assert!(matches!(
            error.kind(),
            io::ErrorKind::WouldBlock | io::ErrorKind::TimedOut
        ));
    }

    #[test]
    fn complete_first_frame_clears_the_decode_timeout() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let mut peer = TcpStream::connect(listener.local_addr().unwrap()).unwrap();
        let (mut server, _) = listener.accept().unwrap();
        let message = StageWireMessage::stop_with_identity(7, 11);
        skippy_protocol::binary::write_stage_message(&mut peer, &message).unwrap();

        let decoded = read_first_message(
            &ConnectionWorkerControl::default(),
            &mut server,
            &StageConfig::default(),
            0,
            Duration::from_millis(50),
        )
        .unwrap()
        .unwrap();

        assert_eq!(decoded.request_id, 7);
        assert_eq!(decoded.session_id, 11);
        assert_eq!(server.read_timeout().unwrap(), None);
    }

    #[test]
    fn marked_first_frame_decodes_without_leaking_a_timeout() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let mut peer = TcpStream::connect(listener.local_addr().unwrap()).unwrap();
        let (mut server, _) = listener.accept().unwrap();
        crate::binary_transport::connection_mode::mark_persistent_lane(&mut peer).unwrap();
        skippy_protocol::binary::write_stage_message(
            &mut peer,
            &StageWireMessage::stop_with_identity(17, 19),
        )
        .unwrap();

        let decoded = read_first_message(
            &ConnectionWorkerControl::default(),
            &mut server,
            &StageConfig::default(),
            0,
            Duration::from_millis(50),
        )
        .unwrap()
        .unwrap();

        assert_eq!(decoded.request_id, 17);
        assert_eq!(decoded.session_id, 19);
        assert_eq!(server.read_timeout().unwrap(), None);
    }
}
