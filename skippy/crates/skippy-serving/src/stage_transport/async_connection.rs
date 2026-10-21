use std::{future::Future, io, net::Shutdown, sync::Mutex, time::Duration};

use futures_util::FutureExt;
use tokio::{
    io::{
        AsyncBufReadExt, AsyncRead, AsyncReadExt, AsyncWrite, AsyncWriteExt, BufReader, ReadHalf,
        WriteHalf,
    },
    runtime::Handle,
    sync::watch,
};

trait Duplex: AsyncRead + AsyncWrite + Unpin + Send {}
impl<T: AsyncRead + AsyncWrite + Unpin + Send> Duplex for T {}
type Stream = Box<dyn Duplex>;

#[derive(Clone, Copy, Default)]
struct Settings {
    read_timeout: Option<Duration>,
    write_timeout: Option<Duration>,
    nonblocking: bool,
}

pub(super) struct AsyncConnection {
    read: Mutex<BufReader<ReadHalf<Stream>>>,
    write: Mutex<WriteHalf<Stream>>,
    settings: Mutex<Settings>,
    closed: watch::Sender<u8>,
    runtime: Handle,
}

impl std::fmt::Debug for AsyncConnection {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("AsyncConnection")
            .finish_non_exhaustive()
    }
}

impl AsyncConnection {
    pub(super) fn new<S>(stream: S, runtime: Handle) -> Self
    where
        S: AsyncRead + AsyncWrite + Unpin + Send + 'static,
    {
        let (read, write) = tokio::io::split(Box::new(stream) as Stream);
        Self {
            read: Mutex::new(BufReader::new(read)),
            write: Mutex::new(write),
            settings: Mutex::new(Settings::default()),
            closed: watch::channel(0).0,
            runtime,
        }
    }

    fn run<T>(
        &self,
        direction: u8,
        operation: impl Future<Output = io::Result<T>>,
    ) -> io::Result<T> {
        let settings = *self.settings.lock().map_err(poisoned)?;
        let timeout = if direction == 1 {
            settings.read_timeout
        } else {
            settings.write_timeout
        };
        let mut closed = self.closed.subscribe();
        self.runtime.block_on(async {
            if *closed.borrow() & direction != 0 {
                return Err(io::ErrorKind::ConnectionAborted.into());
            }
            let operation = async {
                if settings.nonblocking {
                    operation
                        .now_or_never()
                        .unwrap_or_else(|| Err(io::ErrorKind::WouldBlock.into()))
                } else if let Some(timeout) = timeout {
                    tokio::time::timeout(timeout, operation)
                        .await
                        .map_err(|_| io::Error::from(io::ErrorKind::TimedOut))?
                } else {
                    operation.await
                }
            };
            tokio::pin!(operation);
            loop {
                tokio::select! {
                    biased;
                    result = closed.changed() => {
                        if result.is_err() || *closed.borrow() & direction != 0 {
                            return Err(io::ErrorKind::ConnectionAborted.into());
                        }
                    }
                    result = &mut operation => return result,
                }
            }
        })
    }

    pub(super) fn read(&self, buffer: &mut [u8]) -> io::Result<usize> {
        let mut reader = self.read.lock().map_err(poisoned)?;
        self.run(1, reader.read(buffer))
    }

    pub(super) fn peek(&self, buffer: &mut [u8]) -> io::Result<usize> {
        if buffer.is_empty() {
            return Ok(0);
        }
        let mut reader = self.read.lock().map_err(poisoned)?;
        self.run(1, async {
            let bytes = reader.fill_buf().await?;
            let count = bytes.len().min(buffer.len());
            buffer[..count].copy_from_slice(&bytes[..count]);
            Ok(count)
        })
    }

    pub(super) fn write(&self, buffer: &[u8]) -> io::Result<usize> {
        let mut writer = self.write.lock().map_err(poisoned)?;
        self.run(2, writer.write(buffer))
    }

    pub(super) fn flush(&self) -> io::Result<()> {
        let mut writer = self.write.lock().map_err(poisoned)?;
        self.run(2, writer.flush())
    }

    pub(super) fn shutdown(&self, direction: Shutdown) -> io::Result<()> {
        let mask = match direction {
            Shutdown::Read => 1,
            Shutdown::Write => 2,
            Shutdown::Both => 3,
        };
        // Wake blocked operations before acquiring either I/O lock.
        self.closed.send_modify(|closed| *closed |= mask);
        if mask & 2 != 0 {
            let mut writer = self.write.lock().map_err(poisoned)?;
            self.runtime.block_on(async {
                tokio::time::timeout(Duration::from_secs(1), writer.shutdown())
                    .await
                    .map_err(|_| io::Error::from(io::ErrorKind::TimedOut))?
            })?;
        }
        Ok(())
    }

    pub(super) fn set_read_timeout(&self, timeout: Option<Duration>) -> io::Result<()> {
        validate_timeout(timeout)?;
        self.settings.lock().map_err(poisoned)?.read_timeout = timeout;
        Ok(())
    }

    pub(super) fn read_timeout(&self) -> io::Result<Option<Duration>> {
        Ok(self.settings.lock().map_err(poisoned)?.read_timeout)
    }

    pub(super) fn set_write_timeout(&self, timeout: Option<Duration>) -> io::Result<()> {
        validate_timeout(timeout)?;
        self.settings.lock().map_err(poisoned)?.write_timeout = timeout;
        Ok(())
    }

    pub(super) fn write_timeout(&self) -> io::Result<Option<Duration>> {
        Ok(self.settings.lock().map_err(poisoned)?.write_timeout)
    }

    pub(super) fn set_nonblocking(&self, enabled: bool) -> io::Result<()> {
        self.settings.lock().map_err(poisoned)?.nonblocking = enabled;
        Ok(())
    }
}

fn validate_timeout(timeout: Option<Duration>) -> io::Result<()> {
    if timeout.is_some_and(|timeout| timeout.is_zero()) {
        return Err(io::ErrorKind::InvalidInput.into());
    }
    Ok(())
}

fn poisoned<T>(_: std::sync::PoisonError<T>) -> io::Error {
    io::Error::other("stage connection lock poisoned")
}
