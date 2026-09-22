use super::ActivationAgreement;
use std::io::{self, Read, Write};
use std::sync::Arc;

/// Context travels with the stream, not a global socket/stage lookup.
pub trait StageMessageContext {
    fn activation_agreement(&self) -> Option<&ActivationAgreement>;
}
impl<T: StageMessageContext + ?Sized> StageMessageContext for &T {
    fn activation_agreement(&self) -> Option<&ActivationAgreement> {
        (**self).activation_agreement()
    }
}
impl<T: StageMessageContext + ?Sized> StageMessageContext for &mut T {
    fn activation_agreement(&self) -> Option<&ActivationAgreement> {
        (**self).activation_agreement()
    }
}
// Unadmitted I/O can carry control messages, never activation frames.
impl StageMessageContext for Vec<u8> {
    fn activation_agreement(&self) -> Option<&ActivationAgreement> {
        None
    }
}
impl StageMessageContext for [u8] {
    fn activation_agreement(&self) -> Option<&ActivationAgreement> {
        None
    }
}
impl<T> StageMessageContext for io::Cursor<T> {
    fn activation_agreement(&self) -> Option<&ActivationAgreement> {
        None
    }
}
impl StageMessageContext for std::net::TcpStream {
    fn activation_agreement(&self) -> Option<&ActivationAgreement> {
        None
    }
}

#[derive(Debug)]
pub struct StageMessageIo<T> {
    io: T,
    agreement: Option<Arc<ActivationAgreement>>,
}
impl<T> StageMessageIo<T> {
    pub fn new(io: T) -> Self {
        Self {
            io,
            agreement: None,
        }
    }
    pub fn establish(&mut self, agreement: ActivationAgreement) -> io::Result<()> {
        if self.agreement.is_some() {
            return Err(super::invalid_data(
                "cannot replace an established activation agreement",
            ));
        }
        agreement.validate()?;
        self.agreement = Some(Arc::new(agreement));
        Ok(())
    }
    pub fn into_inner(self) -> T {
        self.io
    }
}
impl<T> StageMessageContext for StageMessageIo<T> {
    fn activation_agreement(&self) -> Option<&ActivationAgreement> {
        self.agreement.as_deref()
    }
}
impl<T: Read> Read for StageMessageIo<T> {
    fn read(&mut self, bytes: &mut [u8]) -> io::Result<usize> {
        self.io.read(bytes)
    }
}
impl<T: Write> Write for StageMessageIo<T> {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        self.io.write(bytes)
    }
    fn flush(&mut self) -> io::Result<()> {
        self.io.flush()
    }
}
impl<T> std::ops::Deref for StageMessageIo<T> {
    type Target = T;
    fn deref(&self) -> &T {
        &self.io
    }
}
impl StageMessageIo<std::net::TcpStream> {
    pub fn connect(address: impl std::net::ToSocketAddrs) -> io::Result<Self> {
        std::net::TcpStream::connect(address).map(Self::new)
    }
    pub fn connect_timeout(
        address: &std::net::SocketAddr,
        timeout: std::time::Duration,
    ) -> io::Result<Self> {
        std::net::TcpStream::connect_timeout(address, timeout).map(Self::new)
    }
    pub fn try_clone(&self) -> io::Result<Self> {
        Ok(Self {
            io: self.io.try_clone()?,
            agreement: self.agreement.clone(),
        })
    }
}
impl Read for &StageMessageIo<std::net::TcpStream> {
    fn read(&mut self, bytes: &mut [u8]) -> io::Result<usize> {
        (&self.io).read(bytes)
    }
}
impl Write for &StageMessageIo<std::net::TcpStream> {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        (&self.io).write(bytes)
    }
    fn flush(&mut self) -> io::Result<()> {
        (&self.io).flush()
    }
}
