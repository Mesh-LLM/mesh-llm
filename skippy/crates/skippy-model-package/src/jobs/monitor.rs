//! One bounded Jobs observation owner. Stopping observation never cancels a remote job.
use super::{HfJobsClient, JobStage};
use anyhow::Result;
use futures::{
    Future, FutureExt as _, StreamExt as _,
    future::{Either, select},
};
use serde::Serialize;
use std::{
    pin::Pin,
    time::{Duration, Instant},
};

#[derive(Clone, Copy, Debug)]
pub struct MonitorLimits {
    pub poll_interval: Duration,
    pub max_polls: u32,
    pub max_reconnects: u32,
    pub max_log_lines: usize,
    pub max_log_bytes: usize,
}
impl Default for MonitorLimits {
    fn default() -> Self {
        Self {
            poll_interval: Duration::from_secs(3),
            max_polls: 28800,
            max_reconnects: 1024,
            max_log_lines: 100000,
            max_log_bytes: 32 * 1024 * 1024,
        }
    }
}
impl MonitorLimits {
    fn valid(self) -> bool {
        !self.poll_interval.is_zero()
            && self.poll_interval <= Duration::from_secs(60)
            && (1..=28800).contains(&self.max_polls)
            && (1..=1024).contains(&self.max_reconnects)
            && (1..=100000).contains(&self.max_log_lines)
            && (1..=256 * 1024 * 1024).contains(&self.max_log_bytes)
    }
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum MonitorEnd {
    Completed,
    TerminalFailure,
    Deadline,
    Cancelled,
    TransportFailure,
    CorrelationFailure,
    LimitFailure,
    ObserverFailure,
    InvalidLimits,
}
#[derive(Debug, Serialize)]
pub struct MonitorReceipt {
    pub end: MonitorEnd,
    pub last_stage: Option<JobStage>,
    pub polls: u32,
    pub reconnects: u32,
    pub stream_failures: u32,
    pub log_lines: usize,
    pub log_bytes: usize,
    pub remote_cancel_requested: bool,
    pub remote_cancel_confirmed: bool,
}
impl Default for MonitorReceipt {
    fn default() -> Self {
        Self {
            end: MonitorEnd::TransportFailure,
            last_stage: None,
            polls: 0,
            reconnects: 0,
            stream_failures: 0,
            log_lines: 0,
            log_bytes: 0,
            remote_cancel_requested: false,
            remote_cancel_confirmed: false,
        }
    }
}
enum Wait<T> {
    Ready(T),
    Cancelled,
    Deadline,
}
async fn wait<T, W: Future<Output = T>, C: Future<Output = ()>>(
    work: W,
    mut cancel: Pin<&mut C>,
    deadline: Instant,
) -> Wait<T> {
    if Instant::now() >= deadline {
        return Wait::Deadline;
    }
    if cancel.as_mut().now_or_never().is_some() {
        return Wait::Cancelled;
    }
    match tokio::time::timeout_at(deadline.into(), select(Box::pin(work), cancel)).await {
        Err(_) => Wait::Deadline,
        Ok(Either::Right(_)) => Wait::Cancelled,
        Ok(Either::Left((value, cancel))) => {
            if cancel.now_or_never().is_some() {
                Wait::Cancelled
            } else if Instant::now() < deadline {
                Wait::Ready(value)
            } else {
                Wait::Deadline
            }
        }
    }
}
impl MonitorReceipt {
    fn accept<T>(&mut self, value: Wait<T>) -> Option<T> {
        match value {
            Wait::Ready(value) => Some(value),
            Wait::Cancelled => {
                self.end = MonitorEnd::Cancelled;
                None
            }
            Wait::Deadline => {
                self.end = MonitorEnd::Deadline;
                None
            }
        }
    }
    fn log(
        &mut self,
        text: &str,
        limits: MonitorLimits,
        observer: &mut impl FnMut(&str) -> Result<()>,
    ) -> bool {
        if self.log_lines >= limits.max_log_lines
            || text.len() > limits.max_log_bytes.saturating_sub(self.log_bytes)
        {
            self.end = MonitorEnd::LimitFailure;
            return false;
        }
        self.log_lines += 1;
        self.log_bytes += text.len();
        if observer(text).is_err() {
            self.end = MonitorEnd::ObserverFailure;
            return false;
        }
        true
    }
}
impl HfJobsClient {
    /// Observe under a shared absolute deadline. Cancellation drops the currently owned
    /// HTTP/sleep future, retaining counts/stage; it never silently cancels the remote job.
    /// Log callbacks are synchronous caller code and must return promptly.
    pub async fn monitor_until<C: Future<Output = ()>>(
        &self,
        namespace: &str,
        job_id: &str,
        deadline: Instant,
        cancellation: C,
        limits: MonitorLimits,
        mut observer: impl FnMut(&str) -> Result<()>,
    ) -> MonitorReceipt {
        let mut receipt = MonitorReceipt::default();
        if !limits.valid() {
            receipt.end = MonitorEnd::InvalidLimits;
            return receipt;
        }
        let mut cancel = Box::pin(cancellation);
        loop {
            if receipt.polls >= limits.max_polls {
                receipt.end = MonitorEnd::LimitFailure;
                break;
            }
            let value = wait(
                self.inspect_until(namespace, job_id, deadline),
                cancel.as_mut(),
                deadline,
            )
            .await;
            let Some(value) = receipt.accept(value) else {
                break;
            };
            receipt.polls += 1;
            let Ok(info) = value else {
                receipt.end = MonitorEnd::TransportFailure;
                break;
            };
            if info.id != job_id {
                receipt.end = MonitorEnd::CorrelationFailure;
                break;
            }
            receipt.last_stage = Some(info.status.stage);
            if info.status.stage.is_terminal() {
                receipt.end = if info.status.stage.is_success() {
                    MonitorEnd::Completed
                } else {
                    MonitorEnd::TerminalFailure
                };
                break;
            }
            if info.status.stage == JobStage::Running {
                if receipt.reconnects >= limits.max_reconnects {
                    receipt.end = MonitorEnd::LimitFailure;
                    break;
                }
                receipt.reconnects += 1;
                let value = wait(
                    self.stream_logs_until(namespace, job_id, deadline),
                    cancel.as_mut(),
                    deadline,
                )
                .await;
                let Some(value) = receipt.accept(value) else {
                    break;
                };
                let Ok(stream) = value else {
                    receipt.end = MonitorEnd::TransportFailure;
                    break;
                };
                let mut stream = Box::pin(stream);
                loop {
                    let value = wait(stream.next(), cancel.as_mut(), deadline).await;
                    let Some(value) = receipt.accept(value) else {
                        return receipt;
                    };
                    match value {
                        None => break,
                        Some(Err(_)) => {
                            receipt.stream_failures += 1;
                            break;
                        }
                        Some(Ok(text)) if receipt.log(&text, limits, &mut observer) => (),
                        Some(Ok(_)) => return receipt,
                    }
                }
            }
            let value = wait(
                tokio::time::sleep(limits.poll_interval),
                cancel.as_mut(),
                deadline,
            )
            .await;
            if receipt.accept(value).is_none() {
                break;
            }
        }
        receipt
    }
}
