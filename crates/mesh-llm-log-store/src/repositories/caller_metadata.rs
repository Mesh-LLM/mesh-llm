use std::net::SocketAddr;

use crate::error::LogStoreError;
use crate::store::LogStore;
use crate::timestamps::canonical_persisted_timestamp;

#[derive(Default)]
pub(super) struct CallerMetadata<'a> {
    pub(super) endpoint_id: Option<&'a str>,
    pub(super) addr: Option<String>,
    pub(super) path_type: Option<&'static str>,
}

pub(super) fn canonical_caller_metadata<'a>(
    endpoint_id: Option<&'a str>,
    addr: Option<&str>,
    path_type: Option<&str>,
) -> CallerMetadata<'a> {
    match path_type {
        Some("local_http") => addr
            .and_then(|value| value.parse::<SocketAddr>().ok())
            .map(|value| CallerMetadata {
                endpoint_id: None,
                addr: Some(value.to_string()),
                path_type: Some("local_http"),
            })
            .unwrap_or_default(),
        Some("remote_quic_http") if endpoint_id.is_some_and(authenticated_endpoint_id) => {
            CallerMetadata {
                endpoint_id,
                addr: addr
                    .and_then(|value| value.parse::<SocketAddr>().ok())
                    .map(|value| value.to_string()),
                path_type: Some("remote_quic_http"),
            }
        }
        Some("relay") if endpoint_id.is_some_and(authenticated_endpoint_id) => CallerMetadata {
            endpoint_id,
            addr: None,
            path_type: Some("relay"),
        },
        None if endpoint_id.is_some_and(authenticated_endpoint_id) && addr.is_none() => {
            CallerMetadata {
                endpoint_id,
                addr: None,
                path_type: None,
            }
        }
        Some("remote_quic_http" | "relay") | Some(_) | None => CallerMetadata::default(),
    }
}

fn authenticated_endpoint_id(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

impl LogStore {
    /// Insert a summary when absent and otherwise fill only metadata fields
    /// that have not yet been recorded.
    #[allow(clippy::too_many_arguments)]
    pub fn upsert_summary_metadata(
        &self,
        request_id: &str,
        model: Option<&str>,
        route: Option<&str>,
        provider: Option<&str>,
        engine: Option<&str>,
        occurred_at: &str,
    ) -> Result<(), LogStoreError> {
        self.upsert_summary_metadata_with_caller(
            request_id,
            model,
            route,
            provider,
            engine,
            None,
            None,
            None,
            occurred_at,
        )
    }

    /// Insert a summary when absent and otherwise fill only metadata fields
    /// that have not yet been recorded. Lifecycle state, timestamps, and
    /// identity fields remain untouched.
    #[allow(clippy::too_many_arguments)]
    pub fn upsert_summary_metadata_with_caller(
        &self,
        request_id: &str,
        model: Option<&str>,
        route: Option<&str>,
        provider: Option<&str>,
        engine: Option<&str>,
        caller_endpoint_id: Option<&str>,
        caller_addr: Option<&str>,
        caller_path_type: Option<&str>,
        occurred_at: &str,
    ) -> Result<(), LogStoreError> {
        let occurred_at = canonical_persisted_timestamp(occurred_at)?;
        let caller = canonical_caller_metadata(caller_endpoint_id, caller_addr, caller_path_type);
        self.conn()
            .execute(
                "INSERT INTO summaries \
                 (request_id, created_at, model, route, provider, engine, caller_endpoint_id, caller_addr, caller_path_type) \
                 VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?) \
                 ON CONFLICT(request_id) DO UPDATE SET \
                    model = COALESCE(summaries.model, excluded.model), \
                    route = COALESCE(summaries.route, excluded.route), \
                    provider = COALESCE(summaries.provider, excluded.provider), \
                    engine = COALESCE(summaries.engine, excluded.engine), \
                    caller_endpoint_id = CASE \
                        WHEN summaries.caller_path_type IN ('remote_quic_http', 'relay') \
                             OR (summaries.caller_endpoint_id IS NOT NULL AND summaries.caller_addr IS NULL AND summaries.caller_path_type IS NULL) \
                            THEN summaries.caller_endpoint_id \
                        WHEN (excluded.caller_path_type IN ('remote_quic_http', 'relay') \
                              OR (excluded.caller_endpoint_id IS NOT NULL AND excluded.caller_addr IS NULL AND excluded.caller_path_type IS NULL)) \
                             AND (summaries.caller_path_type = 'local_http' \
                                  OR (summaries.caller_endpoint_id IS NULL AND summaries.caller_addr IS NULL AND summaries.caller_path_type IS NULL)) \
                            THEN excluded.caller_endpoint_id \
                        WHEN summaries.caller_endpoint_id IS NOT NULL OR summaries.caller_addr IS NOT NULL OR summaries.caller_path_type IS NOT NULL \
                            THEN summaries.caller_endpoint_id \
                        ELSE excluded.caller_endpoint_id END, \
                    caller_addr = CASE \
                        WHEN summaries.caller_path_type IN ('remote_quic_http', 'relay') \
                             OR (summaries.caller_endpoint_id IS NOT NULL AND summaries.caller_addr IS NULL AND summaries.caller_path_type IS NULL) \
                            THEN summaries.caller_addr \
                        WHEN (excluded.caller_path_type IN ('remote_quic_http', 'relay') \
                              OR (excluded.caller_endpoint_id IS NOT NULL AND excluded.caller_addr IS NULL AND excluded.caller_path_type IS NULL)) \
                             AND (summaries.caller_path_type = 'local_http' \
                                  OR (summaries.caller_endpoint_id IS NULL AND summaries.caller_addr IS NULL AND summaries.caller_path_type IS NULL)) \
                            THEN excluded.caller_addr \
                        WHEN summaries.caller_endpoint_id IS NOT NULL OR summaries.caller_addr IS NOT NULL OR summaries.caller_path_type IS NOT NULL \
                            THEN summaries.caller_addr \
                        ELSE excluded.caller_addr END, \
                    caller_path_type = CASE \
                        WHEN summaries.caller_path_type IN ('remote_quic_http', 'relay') \
                             OR (summaries.caller_endpoint_id IS NOT NULL AND summaries.caller_addr IS NULL AND summaries.caller_path_type IS NULL) \
                            THEN summaries.caller_path_type \
                        WHEN (excluded.caller_path_type IN ('remote_quic_http', 'relay') \
                              OR (excluded.caller_endpoint_id IS NOT NULL AND excluded.caller_addr IS NULL AND excluded.caller_path_type IS NULL)) \
                             AND (summaries.caller_path_type = 'local_http' \
                                  OR (summaries.caller_endpoint_id IS NULL AND summaries.caller_addr IS NULL AND summaries.caller_path_type IS NULL)) \
                            THEN excluded.caller_path_type \
                        WHEN summaries.caller_endpoint_id IS NOT NULL OR summaries.caller_addr IS NOT NULL OR summaries.caller_path_type IS NOT NULL \
                            THEN summaries.caller_path_type \
                        ELSE excluded.caller_path_type END",
                rusqlite::params![
                    request_id,
                    occurred_at,
                    model,
                    route,
                    provider,
                    engine,
                    caller.endpoint_id,
                    caller.addr,
                    caller.path_type
                ],
            )
            .map(|_| ())
            .map_err(|error| LogStoreError::InsertFailed(error.to_string()))
    }
}
