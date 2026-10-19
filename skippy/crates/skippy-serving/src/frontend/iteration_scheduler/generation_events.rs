use super::*;

impl IterationScheduler {
    pub(super) fn drive_generation_events(
        &self,
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
                let _ = self
                    .shared
                    .commands
                    .try_send(SchedulerCommand::Cancel(id.to_string()));
                return Err(InferenceError::backend("request cancelled"));
            }
            match events.recv_timeout(CANCELLATION_POLL_INTERVAL) {
                Ok(SchedulerEvent::Token { token, ack }) => {
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
                            let _ = self
                                .shared
                                .commands
                                .try_send(SchedulerCommand::Cancel(id.to_string()));
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
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn queued_tokens_are_not_emitted_after_consumer_stops() {
        let (commands, _commands_rx) = std_mpsc::sync_channel(8);
        let scheduler = IterationScheduler {
            shared: Arc::new(IterationSchedulerShared {
                prefill_admission: Default::default(),
                service_estimator: Arc::default(),
                commands,
                max_direct_iteration_tokens: MAX_NATIVE_ITERATION_TOKENS,
                owner_count: AtomicUsize::new(1),
                worker: Mutex::new(None),
            }),
        };
        let (reply, events) = std_mpsc::channel();
        let mut controls = Vec::new();
        for token in [1, 2, 3] {
            let (ack, control) = std_mpsc::sync_channel(1);
            reply.send(SchedulerEvent::Token { token, ack }).unwrap();
            controls.push(control);
        }
        reply.send(SchedulerEvent::Complete).unwrap();
        let mut emitted = Vec::new();
        scheduler
            .drive_generation_events(
                "stopped",
                None,
                events,
                &mut |token| {
                    emitted.push(token);
                    Ok(TokenControl::Stop)
                },
                Instant::now(),
                true,
            )
            .unwrap();
        assert_eq!(emitted, [1]);
        for control in controls {
            assert!(matches!(control.try_recv().unwrap(), TokenControl::Stop));
        }
    }
}
