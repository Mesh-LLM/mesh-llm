//! Platform TLS is a transport adapter. Both the transfer and file reader retain
//! joined ownership. Dropping the async future cancels and joins its transfer.
use super::{Failure, Reply, failure};
use crate::process::Cancellation;
use hyper::Method;
use std::{
    thread::JoinHandle,
    time::{Duration, Instant},
};

#[path = "https_command.rs"]
mod command;
#[path = "https_response.rs"]
mod response;
#[path = "https_transfer.rs"]
mod transfer;
pub(super) use command::Curl;

pub(super) struct Request {
    pub endpoint: String,
    pub method: Method,
    pub body: Option<Vec<u8>>,
    pub stream: bool,
    pub timeout: Duration,
    pub token: &'static str,
    pub started: Instant,
}

impl Curl {
    pub(super) async fn exchange(
        &self,
        request: Request,
        cancellation: Cancellation,
    ) -> Result<Reply, Failure> {
        let local = Cancellation::default();
        let stopped = local.clone();
        let curl = self.clone();
        let (sent, received) = tokio::sync::oneshot::channel();
        let handle = std::thread::Builder::new()
            .name("stability-https".into())
            .spawn(move || {
                let result = transfer::run(&curl, request, &cancellation, &stopped);
                let _ = sent.send(());
                result
            })
            .map_err(|_| failure("stability HTTPS worker unavailable", None))?;
        let mut owner = Pending {
            local,
            handle: Some(handle),
        };
        let _ = received.await;
        owner.finish()
    }
}

struct Pending {
    local: Cancellation,
    handle: Option<JoinHandle<Result<Reply, Failure>>>,
}
impl Pending {
    fn finish(&mut self) -> Result<Reply, Failure> {
        self.handle
            .take()
            .ok_or_else(|| failure("HTTPS worker ownership lost", None))?
            .join()
            .map_err(|_| failure("stability HTTPS worker failed", None))?
    }
}
impl Drop for Pending {
    fn drop(&mut self) {
        self.local.cancel();
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
    }
}

#[cfg(test)]
#[path = "https_ownership_tests.rs"]
mod tests;
