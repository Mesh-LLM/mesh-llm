use crate::frontend::generation::GENERATION_RETRY_AFTER_SECS;
use axum::http::StatusCode;
use openai_frontend::OpenAiError;
use openai_frontend::OpenAiErrorKind;
use openai_frontend::OpenAiResult;
use std::sync::Arc;
use std::sync::Condvar;
use std::sync::Mutex;
use std::time::Duration;
use std::time::Instant;

const DECODE_BATCH_HEADROOM_TOKENS: usize = 512;

#[derive(Debug)]
pub(super) struct GenerationTokenBudget {
    capacity_tokens: usize,
    state: Mutex<GenerationTokenBudgetState>,
    released: Condvar,
}

#[derive(Debug, Default)]
struct GenerationTokenBudgetState {
    active_tokens: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct GenerationTokenBudgetRequest {
    prompt_tokens: usize,
    max_tokens: u32,
}

#[derive(Debug)]
pub(super) struct GenerationTokenReservation {
    budget: Arc<GenerationTokenBudget>,
    tokens: usize,
    active_tokens_after_reservation: usize,
}

impl GenerationTokenBudget {
    pub(super) fn new(ctx_size: usize) -> Self {
        Self {
            capacity_tokens: ctx_size.max(1),
            state: Mutex::new(GenerationTokenBudgetState::default()),
            released: Condvar::new(),
        }
    }

    pub(super) fn reserve(
        self: &Arc<Self>,
        request: GenerationTokenBudgetRequest,
        admission_timeout: Duration,
    ) -> OpenAiResult<GenerationTokenReservation> {
        let tokens = request.reservation_tokens(self.capacity_tokens);
        let deadline = Instant::now() + admission_timeout;
        let mut state = self
            .state
            .lock()
            .map_err(|_| OpenAiError::backend("generation token budget lock poisoned"))?;
        loop {
            if state.active_tokens.saturating_add(tokens) <= self.capacity_tokens {
                state.active_tokens = state.active_tokens.saturating_add(tokens);
                return Ok(GenerationTokenReservation {
                    budget: self.clone(),
                    tokens,
                    active_tokens_after_reservation: state.active_tokens,
                });
            }

            let now = Instant::now();
            if now >= deadline {
                return Err(generation_token_budget_timeout_error(
                    admission_timeout,
                    tokens,
                    state.active_tokens,
                    self.capacity_tokens,
                ));
            }

            let wait_for = deadline.saturating_duration_since(now);
            let (next_state, wait_result) = self
                .released
                .wait_timeout(state, wait_for)
                .map_err(|_| OpenAiError::backend("generation token budget lock poisoned"))?;
            state = next_state;
            if wait_result.timed_out() {
                return Err(generation_token_budget_timeout_error(
                    admission_timeout,
                    tokens,
                    state.active_tokens,
                    self.capacity_tokens,
                ));
            }
        }
    }

    pub(super) fn capacity_tokens(&self) -> usize {
        self.capacity_tokens
    }

    #[cfg(test)]
    fn active_tokens(&self) -> usize {
        self.state
            .lock()
            .expect("generation token budget lock")
            .active_tokens
    }
}

impl GenerationTokenBudgetRequest {
    pub(super) fn new(prompt_tokens: usize, max_tokens: u32) -> Self {
        Self {
            prompt_tokens,
            max_tokens,
        }
    }

    fn reservation_tokens(self, capacity_tokens: usize) -> usize {
        let max_tokens = usize::try_from(self.max_tokens).unwrap_or(usize::MAX);
        let decode_headroom = max_tokens.min(DECODE_BATCH_HEADROOM_TOKENS);
        self.prompt_tokens
            .saturating_add(decode_headroom)
            .min(capacity_tokens.max(1))
    }
}

impl GenerationTokenReservation {
    pub(super) fn tokens(&self) -> usize {
        self.tokens
    }

    pub(super) fn active_tokens_after_reservation(&self) -> usize {
        self.active_tokens_after_reservation
    }
}

impl Drop for GenerationTokenReservation {
    fn drop(&mut self) {
        if self.tokens == 0 {
            return;
        }
        if let Ok(mut state) = self.budget.state.lock() {
            state.active_tokens = state.active_tokens.saturating_sub(self.tokens);
            self.budget.released.notify_one();
        }
    }
}

fn generation_token_budget_timeout_error(
    timeout: Duration,
    requested_tokens: usize,
    active_tokens: usize,
    capacity_tokens: usize,
) -> OpenAiError {
    OpenAiError::from_kind(
        StatusCode::TOO_MANY_REQUESTS,
        OpenAiErrorKind::RateLimit,
        format!(
            "timed out waiting for KV token budget after {} seconds \
             (requested_tokens={requested_tokens}, active_tokens={active_tokens}, \
             capacity_tokens={capacity_tokens})",
            timeout.as_secs()
        ),
    )
    .with_retry_after_secs(GENERATION_RETRY_AFTER_SECS)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{
        sync::mpsc,
        thread,
        time::{Duration, Instant},
    };

    #[test]
    fn generation_token_budget_reserves_and_releases_tokens() {
        let budget = Arc::new(GenerationTokenBudget::new(1_024));
        let reservation = budget
            .reserve(
                GenerationTokenBudgetRequest::new(400, 128),
                Duration::from_millis(10),
            )
            .unwrap();

        assert_eq!(reservation.tokens(), 528);
        assert_eq!(reservation.active_tokens_after_reservation(), 528);
        assert_eq!(budget.active_tokens(), 528);

        drop(reservation);
        assert_eq!(budget.active_tokens(), 0);
    }

    #[test]
    fn generation_token_budget_clamps_decode_headroom() {
        let budget = Arc::new(GenerationTokenBudget::new(8_192));
        let reservation = budget
            .reserve(
                GenerationTokenBudgetRequest::new(4_506, 4_096),
                Duration::from_millis(10),
            )
            .unwrap();

        assert_eq!(reservation.tokens(), 5_018);
        assert_eq!(reservation.active_tokens_after_reservation(), 5_018);
    }

    #[test]
    fn generation_token_budget_waits_for_tokens_to_release() {
        let budget = Arc::new(GenerationTokenBudget::new(1_000));
        let first = budget
            .reserve(
                GenerationTokenBudgetRequest::new(800, 0),
                Duration::from_millis(10),
            )
            .unwrap();
        let waiter = budget.clone();
        let (tx, rx) = mpsc::channel();
        let handle = thread::spawn(move || {
            let reservation = waiter
                .reserve(
                    GenerationTokenBudgetRequest::new(300, 0),
                    Duration::from_secs(1),
                )
                .unwrap();
            tx.send(reservation.active_tokens_after_reservation())
                .unwrap();
        });

        assert!(rx.recv_timeout(Duration::from_millis(20)).is_err());
        drop(first);

        assert_eq!(rx.recv_timeout(Duration::from_secs(1)).unwrap(), 300);
        handle.join().unwrap();
        assert_eq!(budget.active_tokens(), 0);
    }

    #[test]
    fn generation_token_budget_times_out_when_tokens_do_not_fit() {
        let budget = Arc::new(GenerationTokenBudget::new(1_000));
        let _first = budget
            .reserve(
                GenerationTokenBudgetRequest::new(800, 0),
                Duration::from_millis(10),
            )
            .unwrap();

        let started = Instant::now();
        let error = budget
            .reserve(
                GenerationTokenBudgetRequest::new(300, 0),
                Duration::from_millis(15),
            )
            .unwrap_err();

        assert!(started.elapsed() >= Duration::from_millis(10));
        assert_eq!(error.status(), StatusCode::TOO_MANY_REQUESTS);
        assert_eq!(
            error.body().error.code.as_deref(),
            Some("rate_limit_exceeded")
        );
        assert!(error.to_string().contains("KV token budget"));
        assert_eq!(budget.active_tokens(), 800);
    }
}
