//! Proxy transport attempt records.

use serde::{Deserialize, Serialize};

use super::identifiers::{AttemptId, RequestId};

/// One proxy/transport attempt for a request.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ProxyRecord {
    pub attempt_id: AttemptId,
    pub request_id: RequestId,
    pub target: String,
    pub provider: Option<String>,
    pub engine: Option<String>,
    pub started_at: String,
    pub completed_at: Option<String>,
    pub status_code: Option<u16>,
    pub error: Option<String>,
}

impl ProxyRecord {
    /// Create a new attempt record.
    pub fn new(
        attempt_id: AttemptId,
        request_id: RequestId,
        target: String,
        started_at: String,
    ) -> Self {
        Self {
            attempt_id,
            request_id,
            target,
            provider: None,
            engine: None,
            started_at,
            completed_at: None,
            status_code: None,
            error: None,
        }
    }

    /// Record a successful completion.
    pub fn complete(&mut self, status_code: u16, completed_at: String) {
        self.status_code = Some(status_code);
        self.completed_at = Some(completed_at);
    }

    /// Record a failure.
    pub fn fail(&mut self, error: String, completed_at: String) {
        self.error = Some(error);
        self.completed_at = Some(completed_at);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_record() {
        let record = ProxyRecord::new(
            AttemptId::new(),
            RequestId::new(),
            "http://localhost:9337".into(),
            "2025-01-01T00:00:00Z".into(),
        );
        assert!(record.completed_at.is_none());
        assert!(record.status_code.is_none());
        assert!(record.error.is_none());
        assert!(record.provider.is_none());
        assert!(record.engine.is_none());
    }

    #[test]
    fn test_complete() {
        let mut record = ProxyRecord::new(
            AttemptId::new(),
            RequestId::new(),
            "http://localhost:9337".into(),
            "2025-01-01T00:00:00Z".into(),
        );
        record.complete(200, "2025-01-01T00:00:01Z".into());
        assert_eq!(record.status_code, Some(200));
        assert_eq!(record.completed_at.as_deref(), Some("2025-01-01T00:00:01Z"));
        assert!(record.error.is_none());
    }

    #[test]
    fn test_fail() {
        let mut record = ProxyRecord::new(
            AttemptId::new(),
            RequestId::new(),
            "http://localhost:9337".into(),
            "2025-01-01T00:00:00Z".into(),
        );
        record.fail("connection refused".into(), "2025-01-01T00:00:01Z".into());
        assert_eq!(record.error.as_deref(), Some("connection refused"));
        assert_eq!(record.completed_at.as_deref(), Some("2025-01-01T00:00:01Z"));
        assert!(record.status_code.is_none());
    }

    #[test]
    fn test_serde_roundtrip() {
        let mut record = ProxyRecord::new(
            AttemptId::new(),
            RequestId::new(),
            "http://localhost:9337".into(),
            "2025-01-01T00:00:00Z".into(),
        );
        record.complete(200, "2025-01-01T00:00:01Z".into());
        let json = serde_json::to_string(&record).unwrap();
        let parsed: ProxyRecord = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, record);
    }
}
