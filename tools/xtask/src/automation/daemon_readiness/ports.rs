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
        let bind = || TcpListener::bind((Ipv4Addr::LOCALHOST, 0));
        let api = bind().map_err(|error| Error::io("reserve API port", error))?;
        let console = bind().map_err(|error| Error::io("reserve console port", error))?;
        let quic = UdpSocket::bind((Ipv4Addr::LOCALHOST, 0))
            .map_err(|error| Error::io("reserve QUIC port", error))?;
        let ports = Ports {
            api: api
                .local_addr()
                .map_err(|error| Error::io("API address", error))?
                .port(),
            console: console
                .local_addr()
                .map_err(|error| Error::io("console address", error))?
                .port(),
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
