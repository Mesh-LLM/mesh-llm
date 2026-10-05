//! OpenCode request recording delegates HTTP(S) to the existing Curl component.
mod forwarding;
mod listener;
mod request_projection;
use crate::{command::DynResult, process::Cancellation};
use std::{path::Path, time::Duration};

pub(crate) fn run(args: &[String]) -> DynResult<()> {
    let [upstream, capture, ready, seconds] = args else {
        return Err("usage: automation agent-recording-proxy UPSTREAM_API_BASE CAPTURE_JSONL READY_FILE LIFETIME_SECONDS".into());
    };
    let seconds: u64 = seconds.parse()?;
    if !(30..=3600).contains(&seconds) {
        return Err("recording proxy lifetime must be between 30 and 3600 seconds".into());
    }
    let endpoint = request_projection::Endpoint::parse(upstream)?;
    let interrupt = crate::automation::command_interrupt::Interrupt::install()?;
    let cancellation: Cancellation = interrupt.cancellation();
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()?;
    let result = runtime.block_on(listener::serve(
        endpoint,
        Path::new(capture),
        Path::new(ready),
        Duration::from_secs(seconds),
        cancellation,
    ));
    drop(runtime);
    interrupt.finish()?;
    result.map_err(Into::into)
}

#[cfg(test)]
mod tests;
