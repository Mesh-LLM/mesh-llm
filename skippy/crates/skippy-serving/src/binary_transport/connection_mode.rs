//! Explicit persistent-lane marker, separate from the ready handshake.

use std::io::{self, Read, Write};
use std::net::TcpStream;
use std::time::Duration;

const PERSISTENT_LANE: [u8; 4] = *b"SPLN";

pub(crate) fn mark_persistent_lane(stream: &mut TcpStream) -> io::Result<()> {
    stream.write_all(&PERSISTENT_LANE)?;
    stream.flush()
}

/// Read one opening word under the ordinary peer deadline. Only the explicit
/// marker grants an idle persistent lane; ready-handshake completion does not.
pub(super) fn opening_word(
    stream: &mut TcpStream,
    deadline: Duration,
) -> io::Result<Option<[u8; 4]>> {
    stream.set_read_timeout(Some(deadline))?;
    let mut word = [0; 4];
    let result = stream.read_exact(&mut word);
    stream.set_read_timeout(None)?;
    result?;
    Ok((word != PERSISTENT_LANE).then_some(word))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::net::TcpListener;

    #[test]
    fn downstream_marker_requires_explicit_opt_in() {
        use crate::binary_transport::stage_execution::{
            prefix_cache_test_config, take_ready_downstream,
        };
        use skippy_protocol::binary::{
            StageWireMessage, read_stage_message, send_ready, write_stage_message,
        };
        use std::sync::{Arc, Mutex, atomic::AtomicBool};

        for enabled in [false, true] {
            let listener = TcpListener::bind("127.0.0.1:0").unwrap();
            let mut config = prefix_cache_test_config();
            config.downstream.as_mut().unwrap().endpoint =
                listener.local_addr().unwrap().to_string();
            let mut value = serde_json::to_value(config).unwrap();
            if enabled {
                value["downstream"]["persistent_lane_marker"] = true.into();
            }
            let config = serde_json::from_value(value).unwrap();
            let receiver = std::thread::spawn(move || {
                let (mut server, _) = listener.accept().unwrap();
                server
                    .set_read_timeout(Some(Duration::from_secs(2)))
                    .unwrap();
                send_ready(&mut server).unwrap();
                if enabled {
                    let mut marker = [0; 4];
                    server.read_exact(&mut marker).unwrap();
                    assert_eq!(marker, PERSISTENT_LANE);
                }
                read_stage_message(&mut server, 0)
            });

            let mut downstream = take_ready_downstream(
                &config,
                &Arc::new(Mutex::new(None)),
                2,
                &AtomicBool::new(false),
            )
            .unwrap()
            .unwrap();
            write_stage_message(
                &mut downstream,
                &StageWireMessage::stop_with_identity(7, 11),
            )
            .unwrap();
            let decoded = receiver
                .join()
                .unwrap()
                .expect("legacy peer must decode its first frame without marker bytes");
            assert_eq!(decoded.request_id, 7);
            assert_eq!(decoded.session_id, 11);
        }
    }

    #[test]
    fn only_explicit_marker_selects_persistent_mode() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let mut peer = TcpStream::connect(listener.local_addr().unwrap()).unwrap();
        let (mut server, _) = listener.accept().unwrap();
        mark_persistent_lane(&mut peer).unwrap();
        assert_eq!(
            opening_word(&mut server, Duration::from_secs(1)).unwrap(),
            None
        );
        assert_eq!(server.read_timeout().unwrap(), None);
        let ordinary = 1_i32.to_le_bytes();
        peer.write_all(&ordinary).unwrap();
        assert_eq!(
            opening_word(&mut server, Duration::from_secs(1)).unwrap(),
            Some(ordinary)
        );
    }
}
