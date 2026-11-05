//! How the scheduler worker waits for commands: idle until work arrives,
//! briefly while a cache operation defers, while a direct batch coalesces,
//! and between iterations. Each wait is a debug span under `stage.scheduler`,
//! so time the runtime sits unused between native calls has a name.

use super::*;

impl SchedulerWorker {
    /// Block for the next command while no work is queued; false once the
    /// worker should stop.
    pub(super) fn wait_idle(&mut self) -> bool {
        let span = ActiveSpan::start(
            self.telemetry.as_ref(),
            "stage.scheduler_idle",
            SpanLevel::Debug,
        );
        let command = self.commands.recv();
        span.end();
        match command {
            Ok(SchedulerCommand::Shutdown) | Err(_) => false,
            Ok(command) => {
                self.handle_command(command);
                true
            }
        }
    }

    /// Wait up to `timeout` for one command, as the span `name`; false once
    /// the worker should stop.
    pub(super) fn wait_for_scheduler_command(
        &mut self,
        name: &'static str,
        timeout: Duration,
    ) -> bool {
        let span = ActiveSpan::start(self.telemetry.as_ref(), name, SpanLevel::Debug);
        let command = self.commands.recv_timeout(timeout);
        span.end();
        match command {
            Ok(SchedulerCommand::Shutdown) => {
                self.fail_all(InferenceError::backend("iteration scheduler stopped"));
                false
            }
            Ok(command) => {
                self.handle_command(command);
                true
            }
            Err(std_mpsc::RecvTimeoutError::Timeout) => true,
            Err(std_mpsc::RecvTimeoutError::Disconnected) => false,
        }
    }

    /// Hold a direct batch open for up to the coalesce window while more
    /// active sessions may still join it; false once the worker should stop.
    pub(super) fn coalesce_direct_iterations(&mut self) -> bool {
        let target = direct_coalesce_target(
            self.active_runtime_sessions,
            self.direct_iterations.len(),
            self.direct_group_batch_size,
        );
        let queued_before = self.direct_iterations.len();
        if target <= queued_before {
            return true;
        }

        let mut span = ActiveSpan::start(
            self.telemetry.as_ref(),
            "stage.scheduler_coalesce",
            SpanLevel::Debug,
        );
        let deadline = Instant::now() + DIRECT_ITERATION_COALESCE_WINDOW;
        let mut timed_out = false;
        let mut running = true;
        while running && self.direct_iterations.len() < target {
            let Some(remaining) = deadline.checked_duration_since(Instant::now()) else {
                timed_out = true;
                break;
            };
            match self.commands.recv_timeout(remaining) {
                Ok(SchedulerCommand::Shutdown) => {
                    self.fail_all(InferenceError::backend("iteration scheduler stopped"));
                    running = false;
                }
                Ok(command) => self.handle_command(command),
                Err(std_mpsc::RecvTimeoutError::Timeout) => timed_out = true,
                Err(std_mpsc::RecvTimeoutError::Disconnected) => running = false,
            }
            if timed_out {
                break;
            }
        }
        span.extend_attributes([
            (
                "skippy.scheduler.coalesce_target".to_string(),
                json!(target),
            ),
            (
                "skippy.scheduler.coalesce_queued_before".to_string(),
                json!(queued_before),
            ),
            (
                "skippy.scheduler.coalesce_queued_after".to_string(),
                json!(self.direct_iterations.len()),
            ),
            (
                "skippy.scheduler.coalesce_timed_out".to_string(),
                json!(timed_out),
            ),
        ]);
        span.end();
        running
    }

    /// The configured pause between iterations, open to commands; false once
    /// the worker should stop.
    pub(super) fn wait_iteration_interval(&mut self) -> bool {
        if self.iteration_interval.is_zero() {
            return true;
        }
        self.wait_for_scheduler_command("stage.scheduler_interval", self.iteration_interval)
    }
}
