use mesh_llm_events::logging::identifiers::AttemptId;
use mesh_llm_events::logging::proxy::ProxyRecord;

use super::RawMeshRequestLifecycle;

/// One transport attempt that is already owned by a raw mesh request parent.
///
/// The attempt identifier is created by the canonical lifecycle recorder and
/// is reused for durable proxy metadata; this type cannot create a second
/// terminal owner.
pub(crate) struct RawMeshProxyAttempt {
    attempt_id: AttemptId,
    started_at: String,
}

/// Sanitized terminal metadata for one durable proxy attempt record.
pub(crate) struct ProxyAttemptFinish {
    pub(crate) target: &'static str,
    pub(crate) provider: Option<&'static str>,
    pub(crate) engine: Option<&'static str>,
    pub(crate) status_code: Option<u16>,
    pub(crate) lifecycle_error: Option<&'static str>,
    pub(crate) error: Option<&'static str>,
}

impl RawMeshRequestLifecycle {
    /// Start one bounded transport attempt beneath this raw mesh parent.
    pub(crate) fn start_attempt(&self) -> AttemptId {
        self.service.start_attempt(self.request_id, &self.guard)
    }

    /// Start one lifecycle attempt and retain only the bounded metadata needed
    /// to persist its transport result later.
    pub(crate) fn start_proxy_attempt(&self) -> RawMeshProxyAttempt {
        RawMeshProxyAttempt {
            attempt_id: self.start_attempt(),
            started_at: self.service.proxy_record_timestamp(),
        }
    }

    /// Complete one previously started raw mesh transport attempt.
    pub(crate) fn complete_attempt(&self, attempt_id: AttemptId, status_code: u16) {
        self.service
            .complete_attempt(self.request_id, attempt_id, Some(status_code));
    }

    /// Fail one previously started raw mesh transport attempt with a bounded
    /// static outcome label. The parent remains active for later targets.
    pub(crate) fn fail_attempt(&self, attempt_id: AttemptId, label: &'static str) {
        self.service
            .fail_attempt(self.request_id, attempt_id, label.to_owned());
    }

    /// Finish the existing lifecycle attempt and enqueue one metadata-only
    /// durable proxy record. Persistence is deliberately fail-open and this
    /// method never terminalizes the parent request.
    pub(crate) fn finish_proxy_attempt(
        &self,
        attempt: RawMeshProxyAttempt,
        finish: ProxyAttemptFinish,
    ) {
        let ProxyAttemptFinish {
            target,
            provider,
            engine,
            status_code,
            lifecycle_error,
            error,
        } = finish;
        let completed_at = self.service.proxy_record_timestamp();
        let mut record = ProxyRecord::new(
            attempt.attempt_id,
            self.request_id,
            target.to_owned(),
            attempt.started_at,
        );
        record.provider = provider.map(str::to_owned);
        record.engine = engine.map(str::to_owned);

        if let Some(status_code) = status_code {
            self.complete_attempt(attempt.attempt_id, status_code);
            record.complete(status_code, completed_at);
            record.error = error.map(str::to_owned);
        } else {
            let lifecycle_error = lifecycle_error.unwrap_or("retryable_unavailable");
            let error = error.unwrap_or("unavailable");
            self.fail_attempt(attempt.attempt_id, lifecycle_error);
            record.fail(error.to_owned(), completed_at);
        }

        let _ = self.service.enqueue_proxy_record(record);
    }
}
