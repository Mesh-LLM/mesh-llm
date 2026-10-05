//! Recursion-safe persistence for trusted-local operator audit records.

use std::{cell::Cell, sync::Arc};

use mesh_llm_events::logging::identifiers::EventId;
use mesh_llm_log_store::{LogStore, LogStoreError};

thread_local! {
    static OPERATOR_AUDIT_ACTIVE: Cell<bool> = const { Cell::new(false) };
}

/// Owns the failure boundary for maintenance-route audit writes. It never
/// routes an audit failure back into the canonical service, which would risk
/// self-amplifying records when the store is unavailable.
pub(super) struct OperatorAuditWriter;

impl OperatorAuditWriter {
    pub(super) fn new() -> Self {
        Self
    }

    pub(super) fn write(
        &self,
        store: Arc<LogStore>,
        action: &'static str,
        reason: String,
        result: &'static str,
    ) -> Result<(), LogStoreError> {
        let _scope = OperatorAuditScope::enter()?;
        let entry_id = EventId::new().as_uuid().to_string();
        let occurred_at = store.now();
        let detail = serde_json::json!({
            "actor": "trusted_local_operator",
            "source": "logs_api",
            "result": result,
            "reason": reason,
        })
        .to_string();
        store.insert_audit_entry(
            &entry_id,
            None,
            &occurred_at,
            "logs_api",
            action,
            Some(&detail),
        )
    }
}

struct OperatorAuditScope;

impl OperatorAuditScope {
    fn enter() -> Result<Self, LogStoreError> {
        OPERATOR_AUDIT_ACTIVE.with(|active| {
            if active.replace(true) {
                Err(LogStoreError::QueryFailed(
                    "operator audit recursion was suppressed".to_string(),
                ))
            } else {
                Ok(Self)
            }
        })
    }
}

impl Drop for OperatorAuditScope {
    fn drop(&mut self) {
        OPERATOR_AUDIT_ACTIVE.with(|active| active.set(false));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn operator_audit_is_recursion_safe_and_uses_stable_fields() {
        let root = tempfile::tempdir().expect("temporary logging root");
        let store = Arc::new(
            LogStore::open(root.path(), Arc::new(mesh_llm_log_store::RealClock))
                .expect("open log store"),
        );
        let writer = Arc::new(OperatorAuditWriter::new());

        let scope = OperatorAuditScope::enter().expect("enter nested audit scope");
        assert!(
            writer
                .write(
                    Arc::clone(&store),
                    "log_export",
                    "nested".to_string(),
                    "succeeded"
                )
                .is_err()
        );
        drop(scope);
        let audit_count: i64 = store
            .conn()
            .query_row("SELECT COUNT(*) FROM audit_entries", [], |row| row.get(0))
            .expect("count audits");
        assert_eq!(
            audit_count, 0,
            "a blocked nested audit must not amplify itself"
        );

        writer
            .write(
                Arc::clone(&store),
                "log_export",
                "operator copy".to_string(),
                "succeeded",
            )
            .expect("top-level audit");
        let detail: String = store
            .conn()
            .query_row("SELECT detail_json FROM audit_entries LIMIT 1", [], |row| {
                row.get(0)
            })
            .expect("read audit detail");
        assert!(detail.contains("trusted_local_operator"));
        assert!(detail.contains("logs_api"));
        assert!(detail.contains("succeeded"));
    }

    #[test]
    fn concurrent_top_level_audits_are_not_suppressed() {
        use std::sync::{Arc, Barrier};

        let root = tempfile::tempdir().expect("temporary logging root");
        let store = Arc::new(
            LogStore::open(root.path(), Arc::new(mesh_llm_log_store::RealClock))
                .expect("open log store"),
        );
        let writer = Arc::new(OperatorAuditWriter::new());
        let barrier = Arc::new(Barrier::new(2));
        let handles = (0..2)
            .map(|index| {
                let store = Arc::clone(&store);
                let writer = Arc::clone(&writer);
                let barrier = Arc::clone(&barrier);
                std::thread::spawn(move || {
                    barrier.wait();
                    writer.write(
                        store,
                        "log_export",
                        format!("operator-{index}"),
                        "succeeded",
                    )
                })
            })
            .collect::<Vec<_>>();
        for handle in handles {
            handle
                .join()
                .expect("audit worker panicked")
                .expect("audit write");
        }
        let audit_count: i64 = store
            .conn()
            .query_row("SELECT COUNT(*) FROM audit_entries", [], |row| row.get(0))
            .expect("count audits");
        assert_eq!(audit_count, 2);
    }
}
