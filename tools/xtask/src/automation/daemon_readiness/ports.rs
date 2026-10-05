use super::Error;
use std::net::{Ipv4Addr, TcpListener, UdpSocket};

#[derive(Clone, Copy, Debug)]
pub(crate) struct Ports {
    pub(crate) api: u16,
    pub(crate) console: u16,
    pub(crate) quic: u16,
}

pub(crate) struct Reservation {
    _api: TcpListener,
    _console: TcpListener,
    _quic: UdpSocket,
    pub(crate) ports: Ports,
}

impl Reservation {
    pub(crate) fn acquire() -> Result<Self, Error> {
        Self::acquire_at((0, 0))
    }
    pub(crate) fn acquire_at(endpoints: (u16, u16)) -> Result<Self, Error> {
        let api = TcpListener::bind((Ipv4Addr::LOCALHOST, endpoints.0))
            .map_err(|error| Error::io("reserve API port", error))?;
        let console = TcpListener::bind((Ipv4Addr::LOCALHOST, endpoints.1))
            .map_err(|error| Error::io("reserve console port", error))?;
        let api_port = api
            .local_addr()
            .map_err(|error| Error::io("API address", error))?
            .port();
        let console_port = console
            .local_addr()
            .map_err(|error| Error::io("console address", error))?
            .port();
        let quic = distinct_quic(api_port, console_port, || {
            UdpSocket::bind((Ipv4Addr::LOCALHOST, 0))
        })?;
        let ports = Ports {
            api: api_port,
            console: console_port,
            quic: quic
                .local_addr()
                .map_err(|error| Error::io("QUIC address", error))?
                .port(),
        };
        if ports.api == ports.console || ports.api == ports.quic || ports.console == ports.quic {
            return Err(Error::Invalid("reserved ports must be distinct"));
        }
        Ok(Self {
            _api: api,
            _console: console,
            _quic: quic,
            ports,
        })
    }
}

fn distinct_quic(
    api: u16,
    console: u16,
    mut bind: impl FnMut() -> std::io::Result<UdpSocket>,
) -> Result<UdpSocket, Error> {
    // TCP and UDP have separate port namespaces. Hold any colliding UDP
    // sockets so the allocator cannot choose either of those numbers again.
    // With two excluded numbers, the third exclusive UDP bind must differ.
    let mut excluded = Vec::with_capacity(2);
    for _ in 0..3 {
        let socket = bind().map_err(|error| Error::io("reserve QUIC port", error))?;
        let port = socket
            .local_addr()
            .map_err(|error| Error::io("QUIC address", error))?
            .port();
        if port != api && port != console {
            return Ok(socket);
        }
        excluded.push(socket);
    }
    Err(Error::Invalid("could not reserve a distinct QUIC port"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn udp_reservation_retains_collisions_until_a_distinct_number_is_selected() {
        let sockets: [UdpSocket; 3] =
            std::array::from_fn(|_| UdpSocket::bind((Ipv4Addr::LOCALHOST, 0)).unwrap());
        let [api_port, console_port, expected] = sockets
            .each_ref()
            .map(|socket| socket.local_addr().unwrap().port());
        let mut candidates = sockets.into_iter();
        let mut calls = 0;
        let quic = distinct_quic(api_port, console_port, || {
            if calls > 0 {
                assert!(UdpSocket::bind((Ipv4Addr::LOCALHOST, api_port)).is_err());
            }
            if calls > 1 {
                assert!(UdpSocket::bind((Ipv4Addr::LOCALHOST, console_port)).is_err());
            }
            calls += 1;
            Ok(candidates.next().unwrap())
        })
        .unwrap();
        let selected = quic.local_addr().unwrap().port();
        assert_eq!(selected, expected);
        assert_ne!(selected, api_port);
        assert_ne!(selected, console_port);
        assert!(UdpSocket::bind((Ipv4Addr::LOCALHOST, selected)).is_err());
        drop(quic);
        assert!(UdpSocket::bind((Ipv4Addr::LOCALHOST, selected)).is_ok());
    }
}
