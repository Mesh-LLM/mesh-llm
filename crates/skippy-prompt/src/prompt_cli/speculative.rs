#[derive(Default)]
struct SpeculativeStats {
    windows: usize,
    draft_tokens: usize,
    accepted_tokens: usize,
    rejected_tokens: usize,
    full_accept_windows: usize,
    accepted_stop_windows: usize,
    rejected_windows: usize,
    early_reject_windows: usize,
    tail_reject_windows: usize,
    early_reject_stop_windows: usize,
    first_reject_position_sum: usize,
    primary_verify_requests: usize,
    primary_verify_tokens: usize,
    primary_verify_elapsed_ms: f64,
    primary_verify_write_ms: f64,
    primary_verify_wait_ms: f64,
    primary_verify_compute_us: i64,
    primary_verify_forward_write_us: i64,
    primary_verify_downstream_wait_us: i64,
    primary_verify_total_us: i64,
    primary_verify_stage_count: i64,
    adaptive_window_start: usize,
    adaptive_window_final: usize,
    adaptive_window_max: usize,
    adaptive_window_min: usize,
    adaptive_window_max_seen: usize,
    adaptive_window_sum: usize,
    adaptive_window_grows: usize,
    adaptive_window_shrinks: usize,
    adaptive_window_enabled: bool,
}

impl SpeculativeStats {
    fn observe_primary_verify(&mut self, reply: &VerifyWindowReply, token_count: usize) {
        self.primary_verify_requests += 1;
        self.primary_verify_tokens += token_count;
        self.primary_verify_elapsed_ms += reply.elapsed_ms;
        self.primary_verify_write_ms += reply.write_ms;
        self.primary_verify_wait_ms += reply.wait_ms;
        self.primary_verify_compute_us += reply.stats.verify_window_compute_us;
        self.primary_verify_forward_write_us += reply.stats.verify_window_forward_write_us;
        self.primary_verify_downstream_wait_us += reply.stats.verify_window_downstream_wait_us;
        self.primary_verify_total_us += reply.stats.verify_window_total_us;
        self.primary_verify_stage_count += reply.stats.verify_window_stage_count;
    }

    fn observe_verify_decision(
        &mut self,
        decision: VerifyWindowDecision,
        adaptive_window: &mut usize,
        adaptive_enabled: bool,
        max_speculative_window: usize,
    ) {
        self.accepted_tokens += decision.accepted_before_reject;
        if decision.rejected() {
            self.rejected_tokens += 1;
        }

        match decision.kind {
            VerifyWindowDecisionKind::FullAccept => {
                self.full_accept_windows += 1;
                self.grow_adaptive_window(
                    adaptive_window,
                    adaptive_enabled,
                    max_speculative_window,
                );
            }
            VerifyWindowDecisionKind::AcceptedStop => {
                self.accepted_stop_windows += 1;
            }
            VerifyWindowDecisionKind::TailReject => {
                self.observe_reject(decision);
                self.tail_reject_windows += 1;
                self.grow_adaptive_window(
                    adaptive_window,
                    adaptive_enabled,
                    max_speculative_window,
                );
            }
            VerifyWindowDecisionKind::EarlyReject => {
                self.observe_reject(decision);
                self.early_reject_windows += 1;
                self.shrink_adaptive_window(adaptive_window, adaptive_enabled, decision);
            }
            VerifyWindowDecisionKind::EarlyRejectStop => {
                self.observe_reject(decision);
                self.early_reject_windows += 1;
                self.early_reject_stop_windows += 1;
            }
        }
    }

    fn observe_reject(&mut self, decision: VerifyWindowDecision) {
        if decision.rejected() {
            self.rejected_windows += 1;
            self.first_reject_position_sum += decision.commit_count;
        }
    }

    fn grow_adaptive_window(
        &mut self,
        adaptive_window: &mut usize,
        adaptive_enabled: bool,
        max_speculative_window: usize,
    ) {
        if adaptive_enabled && *adaptive_window < max_speculative_window {
            *adaptive_window += 1;
            self.adaptive_window_grows += 1;
        }
    }

    fn shrink_adaptive_window(
        &mut self,
        adaptive_window: &mut usize,
        adaptive_enabled: bool,
        decision: VerifyWindowDecision,
    ) {
        if !adaptive_enabled {
            return;
        }
        if !decision.rejected() {
            return;
        }
        let next_window = (*adaptive_window)
            .saturating_sub(1)
            .max(decision.commit_count)
            .max(1);
        if next_window < *adaptive_window {
            *adaptive_window = next_window;
            self.adaptive_window_shrinks += 1;
        }
    }
}

fn verify_inputs_for_proposals(current: i32, proposals: &[i32]) -> Vec<i32> {
    let mut tokens = Vec::with_capacity(proposals.len());
    if proposals.is_empty() {
        return tokens;
    }
    tokens.push(current);
    tokens.extend(proposals.iter().take(proposals.len().saturating_sub(1)));
    tokens
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum VerifyWindowDecisionKind {
    FullAccept,
    AcceptedStop,
    TailReject,
    EarlyReject,
    EarlyRejectStop,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct VerifyWindowDecision {
    kind: VerifyWindowDecisionKind,
    accepted_before_reject: usize,
    commit_count: usize,
}

impl VerifyWindowDecision {
    fn rejected(self) -> bool {
        matches!(
            self.kind,
            VerifyWindowDecisionKind::TailReject
                | VerifyWindowDecisionKind::EarlyReject
                | VerifyWindowDecisionKind::EarlyRejectStop
        )
    }

    #[cfg(test)]
    fn tail_reject(self) -> bool {
        self.kind == VerifyWindowDecisionKind::TailReject
    }
}

fn classify_verify_window<F>(
    draft_tokens: &[i32],
    predicted_tokens: &[i32],
    generated_len: usize,
    max_new_tokens: usize,
    mut token_is_eog: F,
) -> Result<VerifyWindowDecision>
where
    F: FnMut(i32) -> Result<bool>,
{
    if predicted_tokens.len() < draft_tokens.len() {
        bail!(
            "verify window returned too few tokens: got {} expected {}",
            predicted_tokens.len(),
            draft_tokens.len()
        );
    }

    let mut accepted_before_reject = 0usize;
    let mut commit_count = 0usize;
    for (draft_token, predicted) in draft_tokens.iter().zip(predicted_tokens.iter()) {
        commit_count += 1;
        let accepted = *predicted == *draft_token;
        let reached_eog = token_is_eog(*predicted)?;
        let reached_limit = generated_len + commit_count >= max_new_tokens;
        if accepted {
            accepted_before_reject += 1;
            if (reached_eog || reached_limit) && commit_count < draft_tokens.len() {
                return Ok(VerifyWindowDecision {
                    kind: VerifyWindowDecisionKind::AcceptedStop,
                    accepted_before_reject,
                    commit_count,
                });
            }
            continue;
        }

        let commit_count = accepted_before_reject + 1;
        let kind = if commit_count == draft_tokens.len() {
            VerifyWindowDecisionKind::TailReject
        } else if reached_eog || reached_limit {
            VerifyWindowDecisionKind::EarlyRejectStop
        } else {
            VerifyWindowDecisionKind::EarlyReject
        };
        return Ok(VerifyWindowDecision {
            kind,
            accepted_before_reject,
            commit_count,
        });
    }

    Ok(VerifyWindowDecision {
        kind: VerifyWindowDecisionKind::FullAccept,
        accepted_before_reject,
        commit_count,
    })
}
