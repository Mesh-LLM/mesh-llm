//! Native iteration execution, session progress, and completion accounting.
use super::*;

impl RuntimeState {
    pub(crate) fn take_completed_execution(
        &mut self,
    ) -> Option<super::execution_accounting::CompletedExecution> {
        self.completed_execution.take()
    }

    pub fn iteration_batch_sampled(
        &mut self,
        requests: &[RuntimeIterationBatchRequest<'_>],
    ) -> Result<IterationBatchOutput> {
        self.completed_execution = None;
        if requests.is_empty() {
            return Ok(IterationBatchOutput {
                request_outputs: Vec::new(),
                samples: Vec::new(),
            });
        }
        let mut unique = std::collections::BTreeSet::new();
        for request in requests {
            if !unique.insert(request.session_id) {
                bail!(
                    "iteration contains duplicate session {}",
                    request.session_id
                );
            }
        }
        let new_session_count = unique
            .iter()
            .filter(|session_id| !self.sessions.contains_key(**session_id))
            .count();
        let available_lanes = (self.lane_count as usize).saturating_sub(self.sessions.len());
        ensure_iteration_session_capacity(new_session_count, available_lanes)?;
        for request in requests {
            self.session(request.session_id)?;
        }

        let mut lane_sessions = Vec::with_capacity(requests.len());
        for request in requests {
            let lane_session = self.sessions.remove(request.session_id).ok_or_else(|| {
                anyhow::anyhow!(
                    "session {} was not active after admission",
                    request.session_id
                )
            })?;
            lane_sessions.push((request.session_id.to_string(), lane_session));
        }

        let result = {
            let mut iteration_requests = lane_sessions
                .iter_mut()
                .zip(requests.iter())
                .map(|((_, lane_session), request)| IterationBatchRequest {
                    session: &mut lane_session.session,
                    token_ids: request.token_ids,
                    positions: request.positions,
                    sampling: request.sampling,
                    input: request.input,
                    sample_last: request.sample_last,
                    phase: request.phase,
                })
                .collect::<Vec<_>>();
            let before = self.model.execution_epoch();
            let started = Instant::now();
            let result = StageSession::iteration_batch_sampled(&mut iteration_requests);
            let elapsed = started.elapsed();
            let after = self.model.execution_epoch();
            let mut work = super::execution_accounting::CompletedExecution {
                elapsed,
                ..Default::default()
            };
            for request in requests {
                match request.phase {
                    IterationBatchPhase::Prefill => {
                        work.prompt_tokens += request.token_ids.len() as u64
                    }
                    IterationBatchPhase::Decode => {
                        work.decode_tokens += request.token_ids.len() as u64
                    }
                }
            }
            self.completed_execution =
                self.execution_accounting
                    .finish(before, after, work, result.is_ok());
            result
        };

        for (session_id, lane_session) in lane_sessions {
            self.sessions.insert(session_id, lane_session);
        }
        match &result {
            Ok(_) => {
                for request in requests {
                    self.add_session_tokens(request.session_id, request.token_ids.len() as u64);
                }
            }
            Err(error) => {
                // A one-at-a-time batch reports how far it got. Those sessions
                // have already advanced natively and cannot be rolled back, so
                // their tracked positions have to move with them.
                if let Some(partial) = error.downcast_ref::<skippy_runtime::PartialBatchExecution>()
                {
                    for request in requests.iter().take(partial.executed()) {
                        self.add_session_tokens(request.session_id, request.token_ids.len() as u64);
                    }
                }
            }
        }
        result
    }
}

fn ensure_iteration_session_capacity(
    new_session_count: usize,
    available_lanes: usize,
) -> Result<()> {
    if new_session_count > available_lanes {
        bail!(
            "iteration requires {new_session_count} new sessions but only {available_lanes} execution lanes are available"
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn rejects_batch_before_partial_session_admission() {
        assert!(ensure_iteration_session_capacity(3, 2).is_err());
        assert!(ensure_iteration_session_capacity(2, 2).is_ok());
    }
}
