use std::future::Future;
use std::time::Duration;

use reqwest::{Error as ReqwestError, Response, StatusCode};
use tokio_retry::strategy::{ExponentialBackoff, jitter};
use tracing::{debug, error};

pub(crate) const DEFAULT_MAX_ATTEMPTS: usize = 5;
pub(crate) const DEFAULT_BASE_DELAY: Duration = Duration::from_millis(100);

#[derive(Debug, Clone, Copy)]
pub(crate) struct RetryConfig {
    pub max_attempts: usize,
    pub base_delay: Duration,
}

impl Default for RetryConfig {
    fn default() -> Self {
        Self {
            max_attempts: DEFAULT_MAX_ATTEMPTS,
            base_delay: DEFAULT_BASE_DELAY,
        }
    }
}

fn is_transient_status(status: StatusCode) -> bool {
    matches!(
        status,
        StatusCode::REQUEST_TIMEOUT
            | StatusCode::TOO_MANY_REQUESTS
            | StatusCode::INTERNAL_SERVER_ERROR
            | StatusCode::BAD_GATEWAY
            | StatusCode::SERVICE_UNAVAILABLE
            | StatusCode::GATEWAY_TIMEOUT
    )
}

fn is_transient_reqwest_error(err: &ReqwestError) -> bool {
    // The wasm reqwest backend exposes no `is_connect`/`is_body`/`is_decode`/
    // `is_redirect` and no `hyper::Error` source, so wasm retries only on
    // timeout — strictly less generous than native.
    if err.is_timeout() {
        return true;
    }
    if err.is_builder() || err.is_status() {
        return false;
    }
    #[cfg(target_family = "wasm")]
    {
        false
    }
    #[cfg(not(target_family = "wasm"))]
    {
        if err.is_connect() {
            return true;
        }
        if err.is_body() || err.is_decode() || err.is_redirect() {
            return false;
        }
        if err.is_request()
            && let Some(hyper_err) = find_source::<hyper::Error>(err)
            && (hyper_err.is_incomplete_message()
                || hyper_err.is_canceled()
                || find_source::<std::io::Error>(hyper_err).is_some())
        {
            return true;
        }
        false
    }
}

fn find_source<T: std::error::Error + 'static>(err: &dyn std::error::Error) -> Option<&T> {
    let mut source = err.source();
    while let Some(e) = source {
        if let Some(t) = e.downcast_ref::<T>() {
            return Some(t);
        }
        source = e.source();
    }
    None
}

fn err_url(err: &ReqwestError) -> String {
    err.url()
        .map(|u| u.to_string())
        .unwrap_or_else(|| "<unknown url>".to_string())
}

fn is_transient(result: &Result<Response, ReqwestError>) -> bool {
    match result {
        Ok(resp) => is_transient_status(resp.status()),
        Err(e) => is_transient_reqwest_error(e),
    }
}

fn log_attempt(attempt: usize, transient: bool, result: &Result<Response, ReqwestError>) {
    match (transient, result) {
        (true, Ok(resp)) => {
            debug!(attempt, url = %resp.url(), status = %resp.status(), "retrying request");
        }
        (true, Err(e)) => {
            debug!(attempt, url = %err_url(e), error = %e, "retrying request");
        }
        (false, Ok(resp)) => {
            debug!(attempt, url = %resp.url(), status = %resp.status(), "request succeeded");
        }
        (false, Err(_)) => {}
    }
}

fn log_exhausted(max_attempts: usize, result: &Result<Response, ReqwestError>) {
    let url = match result {
        Ok(resp) => resp.url().to_string(),
        Err(e) => err_url(e),
    };
    error!(url = %url, max_attempts, "retry exhausted");
}

/// Delay iterator used between retry attempts.
///
/// Yields at most `config.max_attempts` durations. With `base_delay = B` and `max_attempts = N`
/// the pre-jitter schedule is `2B, 4B, 8B, ..., 2^N * B`; `jitter` multiplies each by a random
/// factor in `[0, 1)`, so the total sleep budget is bounded above by `B * (2^(N+1) - 2)`.
fn delay_strategy(config: &RetryConfig) -> impl Iterator<Item = Duration> {
    let base_ms = config.base_delay.as_millis().min(u64::MAX as u128) as u64;
    ExponentialBackoff::from_millis(2)
        .factor(base_ms)
        .map(jitter)
        .take(config.max_attempts)
}

/// Retry the provided async request factory using the given config.
/// On each attempt the closure is invoked to build a fresh `send()` future.
/// Returns the final `Response` (including non-retryable error statuses) or
/// a final transport error.
pub(crate) async fn retry<F, Fut>(config: &RetryConfig, mut f: F) -> Result<Response, ReqwestError>
where
    F: FnMut() -> Fut,
    Fut: Future<Output = Result<Response, ReqwestError>>,
{
    let mut delays = delay_strategy(config);

    let mut attempt = 0usize;
    loop {
        let result = f().await;
        let transient = is_transient(&result);
        log_attempt(attempt, transient, &result);

        if !transient {
            return result;
        }

        match delays.next() {
            Some(delay) => tokio::time::sleep(delay).await,
            None => {
                log_exhausted(config.max_attempts, &result);
                return result;
            }
        }

        attempt += 1;
    }
}

#[cfg(test)]
mod tests {
    use reqwest::StatusCode;

    use super::*;

    #[test]
    fn status_classification() {
        for s in [
            StatusCode::REQUEST_TIMEOUT,
            StatusCode::TOO_MANY_REQUESTS,
            StatusCode::INTERNAL_SERVER_ERROR,
            StatusCode::BAD_GATEWAY,
            StatusCode::SERVICE_UNAVAILABLE,
            StatusCode::GATEWAY_TIMEOUT,
        ] {
            assert!(is_transient_status(s), "{s} should be transient");
        }

        for s in [
            StatusCode::OK,
            StatusCode::CREATED,
            StatusCode::MOVED_PERMANENTLY,
            StatusCode::BAD_REQUEST,
            StatusCode::UNAUTHORIZED,
            StatusCode::FORBIDDEN,
            StatusCode::NOT_FOUND,
            StatusCode::CONFLICT,
            StatusCode::NOT_IMPLEMENTED,
        ] {
            assert!(!is_transient_status(s), "{s} should be fatal");
        }
    }

    #[tokio::test]
    async fn connect_error_is_transient() {
        let err = reqwest::Client::new()
            .get("http://127.0.0.1:1")
            .send()
            .await
            .unwrap_err();
        assert!(err.is_connect(), "expected connect error, got {err:?}");
        assert!(is_transient_reqwest_error(&err));
    }

    #[tokio::test]
    async fn builder_error_is_fatal() {
        let err = reqwest::Client::new()
            .get("not-a-url")
            .send()
            .await
            .unwrap_err();
        assert!(err.is_builder(), "expected builder error, got {err:?}");
        assert!(!is_transient_reqwest_error(&err));
    }

    /// Regression: confirm the backoff schedule is `base_delay * 2^n`, not `base_delay^n`.
    /// With max_attempts=4 and base_delay=10ms, the pre-jitter schedule is
    /// 20 + 40 + 80 + 160 = 300ms; jitter only shortens delays, so the total must stay
    /// well under that. A buggy `ExponentialBackoff::from_millis(10)` would yield
    /// 10 + 100 + 1000 + 10000 = 11110ms.
    #[test]
    fn retry_delay_budget_is_bounded() {
        let config = RetryConfig {
            max_attempts: 4,
            base_delay: Duration::from_millis(10),
        };
        let total: Duration = delay_strategy(&config).sum();
        assert!(
            total < Duration::from_millis(500),
            "total sleep budget {total:?} exceeds 500ms"
        );
    }
}
