use super::{
    CANCELLATION_POLL_INTERVAL, ScheduledGenerationStats, SchedulerCommand, SchedulerEvent,
};
use crate::frontend::generation::TokenControl;
use skippy_inference_api::{InferenceError, InferenceResult};
use std::sync::mpsc as std_mpsc;
use std::time::Instant;

pub(super) fn drive_generation_events(
    commands: &std_mpsc::SyncSender<SchedulerCommand>,
    id: &str,
    cancellation: Option<&skippy_inference_api::CancellationToken>,
    events: std_mpsc::Receiver<SchedulerEvent>,
    on_token: &mut impl FnMut(i32) -> InferenceResult<TokenControl>,
    started: Instant,
    resumed: bool,
) -> InferenceResult<ScheduledGenerationStats> {
    let mut first_token_at = None;
    let mut stopped = false;
    loop {
        if cancellation.is_some_and(skippy_inference_api::CancellationToken::is_cancelled) {
            let _ = commands.try_send(SchedulerCommand::Cancel(id.to_string()));
            return Err(InferenceError::backend("request cancelled"));
        }
        match events.recv_timeout(CANCELLATION_POLL_INTERVAL) {
            Ok(SchedulerEvent::Token { token, ack }) => {
                // The worker can enqueue predictions before it observes our stop.
                // Drain them without exposing any token after the terminal decision.
                if stopped {
                    let _ = ack.send(TokenControl::Stop);
                    continue;
                }
                first_token_at.get_or_insert_with(Instant::now);
                match on_token(token) {
                    Ok(control) => {
                        stopped = control == TokenControl::Stop;
                        let _ = ack.send(control);
                    }
                    Err(error) => {
                        let _ = ack.send(TokenControl::Stop);
                        let _ = commands.try_send(SchedulerCommand::Cancel(id.to_string()));
                        return Err(error);
                    }
                }
            }
            Ok(SchedulerEvent::Complete) => {
                let elapsed_ms = started.elapsed().as_secs_f64() * 1_000.0;
                let prompt_ms = if resumed {
                    0.0
                } else {
                    first_token_at
                        .map(|first| first.duration_since(started).as_secs_f64() * 1_000.0)
                        .unwrap_or(elapsed_ms)
                };
                return Ok(ScheduledGenerationStats {
                    prompt_ms,
                    predicted_ms: (elapsed_ms - prompt_ms).max(0.0),
                });
            }
            Ok(SchedulerEvent::Error(error)) => return Err(error),
            Err(std_mpsc::RecvTimeoutError::Timeout) => {}
            Err(std_mpsc::RecvTimeoutError::Disconnected) => {
                return Err(InferenceError::backend(
                    "iteration scheduler stopped before generation completed",
                ));
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn queued_predictions_after_stop_never_reach_the_consumer() {
        let (commands, _receiver) = std_mpsc::sync_channel(1);
        let (reply, events) = std_mpsc::channel();
        let mut controls = Vec::new();
        for token in [42, 0, 10, 11] {
            let (ack, control) = std_mpsc::sync_channel(1);
            reply.send(SchedulerEvent::Token { token, ack }).unwrap();
            controls.push(control);
        }
        reply.send(SchedulerEvent::Complete).unwrap();
        let mut observed = Vec::new();
        drive_generation_events(
            &commands,
            "request",
            None,
            events,
            &mut |token| {
                observed.push(token);
                Ok(if token == 0 {
                    TokenControl::Stop
                } else {
                    TokenControl::Continue
                })
            },
            Instant::now(),
            false,
        )
        .unwrap();
        assert_eq!(observed, [42, 0]);
        let controls: Vec<_> = controls
            .iter()
            .map(|control| control.recv().unwrap())
            .collect();
        assert!(
            controls
                == [
                    TokenControl::Continue,
                    TokenControl::Stop,
                    TokenControl::Stop,
                    TokenControl::Stop
                ],
            "every prediction after the terminal token must acknowledge stop"
        );
    }

    #[test]
    fn stop_still_waits_for_worker_completion() {
        let (commands, _receiver) = std_mpsc::sync_channel(1);
        let (reply, events) = std_mpsc::channel();
        let (ack, _control) = std_mpsc::sync_channel(1);
        reply.send(SchedulerEvent::Token { token: 0, ack }).unwrap();
        reply
            .send(SchedulerEvent::Error(InferenceError::backend(
                "worker failed",
            )))
            .unwrap();
        let result = drive_generation_events(
            &commands,
            "request",
            None,
            events,
            &mut |_| Ok(TokenControl::Stop),
            Instant::now(),
            false,
        );
        assert!(result.is_err(), "stop must not conceal a worker failure");
    }
}
