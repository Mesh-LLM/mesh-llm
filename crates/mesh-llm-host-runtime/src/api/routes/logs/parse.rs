use mesh_llm_events::logging::timestamp::canonical_logging_timestamp;
use mesh_llm_log_store::{
    AuditEntryFilters, AuditEntrySeverity, AuditEntrySource, PageQuery, ProxyQuery, QuerySort,
    RequestOutcome, RequestQuery,
};
use serde::Deserialize;

use super::LogsError;

const DEFAULT_LIMIT: usize = 50;
pub(super) const MAX_LIMIT: usize = 100;
const MAX_FILTER_LENGTH: usize = 128;
pub(super) const MAX_EXPORT_ROWS: usize = 50;
const MAX_AUDIT_REASON_LENGTH: usize = 256;
const MAX_WEBHOOK_DELIVERY_ID_LENGTH: usize = 128;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum SourceFilter {
    Active,
    Durable,
}

pub(super) struct RequestListQuery {
    pub(super) store: RequestQuery,
    pub(super) source: Option<SourceFilter>,
    pub(super) cursor_boundary: Option<(String, String)>,
}

pub(super) fn request_query(path: &str) -> Result<RequestListQuery, LogsError> {
    let pairs = pairs(path)?;
    let mut query = RequestQuery {
        limit: DEFAULT_LIMIT,
        cursor: None,
        from: None,
        to: None,
        route: None,
        exclude_route: None,
        exclude_route_prefix: None,
        model: None,
        provider: None,
        engine: None,
        status_code: None,
        outcome: None,
        sort: QuerySort::Descending,
    };
    let mut source = None;
    for (key, value) in pairs {
        match key.as_str() {
            "cursor" => query.cursor = Some(nonempty(value)?),
            "limit" => query.limit = limit(&value)?,
            "from" => query.from = Some(timestamp(&value)?),
            "to" => query.to = Some(timestamp(&value)?),
            "route" => query.route = Some(filter(value)?),
            "exclude_route" => query.exclude_route = Some(filter(value)?),
            "exclude_route_prefix" => query.exclude_route_prefix = Some(filter(value)?),
            "model" => query.model = Some(filter(value)?),
            "provider" => query.provider = Some(filter(value)?),
            "engine" => query.engine = Some(filter(value)?),
            "status" => query.status_code = Some(status(&value)?),
            "outcome" => query.outcome = Some(outcome(&value)?),
            "source" => source = Some(source_filter(&value)?),
            "sort" => query.sort = sort(&value)?,
            _ => return Err(LogsError::InvalidQuery("unknown request filter")),
        }
    }
    if query.from > query.to && query.to.is_some() {
        return Err(LogsError::InvalidQuery("from must not be after to"));
    }
    let cursor_boundary = query
        .cursor
        .as_deref()
        .map(mesh_llm_log_store::decode_cursor)
        .transpose()
        .map_err(|_| LogsError::InvalidCursor)?
        .map(|(timestamp, request_id)| {
            canonical_logging_timestamp(&timestamp)
                .map(|timestamp| (timestamp, request_id))
                .map_err(|_| LogsError::InvalidCursor)
        })
        .transpose()?;
    Ok(RequestListQuery {
        store: query,
        source,
        cursor_boundary,
    })
}

#[derive(Clone, Debug)]
pub(super) struct ExportRequest {
    pub(super) query: RequestQuery,
    pub(super) reason: String,
    pub(super) include_artifacts: bool,
}

pub(super) struct CleanupRunRequest {
    pub(super) operation_id: mesh_llm_log_store::MaintenanceOperationId,
    pub(super) reason: mesh_llm_log_store::MaintenanceReason,
}

/// Parsed contract for `POST /api/logs/requests/{requestId}/delete`.
///
/// The request ID remains in the path while the immutable operation ID and
/// operator reason stay in the strict JSON body.  This keeps retries
/// idempotent without placing operator text in a URL.
pub(super) struct DeleteRequest {
    pub(super) operation_id: mesh_llm_log_store::MaintenanceOperationId,
    pub(super) request_id: String,
    pub(super) reason: mesh_llm_log_store::MaintenanceReason,
}

/// Parsed contract for `POST /api/logs/webhooks/{deliveryId}/retry`.
///
/// The delivery identifier stays in the path because it is the idempotency
/// key. The required operator reason remains in a strict JSON body, avoiding
/// operator-provided text in URLs and access logs.
pub(super) struct WebhookRetryRequest {
    pub(super) delivery_id: String,
    pub(super) reason: String,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct CleanupPreviewBody {
    operation_id: String,
    cutoff_before: String,
    request_limit: usize,
    #[serde(default)]
    source: Option<String>,
    #[serde(default)]
    from: Option<String>,
    #[serde(default)]
    to: Option<String>,
    #[serde(default)]
    route: Option<String>,
    #[serde(default)]
    exclude_route: Option<String>,
    #[serde(default)]
    model: Option<String>,
    #[serde(default)]
    provider: Option<String>,
    #[serde(default)]
    engine: Option<String>,
    #[serde(default)]
    outcome: Option<String>,
    reason: String,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct CleanupRunBody {
    operation_id: String,
    reason: String,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct DeleteBody {
    operation_id: String,
    reason: String,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct WebhookRetryBody {
    reason: String,
}

pub(super) fn delete_request(
    request_id: &str,
    path: &str,
    body: &str,
) -> Result<DeleteRequest, LogsError> {
    cleanup_path(path)?;
    let request_id = id(request_id)?;
    let body = serde_json::from_str::<DeleteBody>(body).map_err(|_| LogsError::InvalidJsonBody)?;
    Ok(DeleteRequest {
        operation_id: mesh_llm_log_store::MaintenanceOperationId::try_from(
            body.operation_id.as_str(),
        )?,
        request_id,
        reason: maintenance_reason(body.reason)?,
    })
}

pub(super) fn webhook_retry_request(
    delivery_id: &str,
    path: &str,
    body: &str,
) -> Result<WebhookRetryRequest, LogsError> {
    if path.contains('?') {
        return Err(LogsError::InvalidQuery(
            "webhook retry does not accept query parameters",
        ));
    }
    let body =
        serde_json::from_str::<WebhookRetryBody>(body).map_err(|_| LogsError::InvalidJsonBody)?;
    Ok(WebhookRetryRequest {
        delivery_id: webhook_delivery_id(delivery_id)?,
        reason: audit_reason(body.reason)?,
    })
}

pub(super) fn cleanup_preview_request(
    path: &str,
    body: &str,
) -> Result<mesh_llm_log_store::CleanupPreviewRequest, LogsError> {
    cleanup_path(path)?;
    let body =
        serde_json::from_str::<CleanupPreviewBody>(body).map_err(|_| LogsError::InvalidJsonBody)?;
    let cutoff = timestamp(&body.cutoff_before)?;
    let operation_id =
        mesh_llm_log_store::MaintenanceOperationId::try_from(body.operation_id.as_str())?;
    let reason = maintenance_reason(body.reason)?;
    match body.source.as_deref() {
        None | Some("durable") => {}
        Some(_) => {
            return Err(LogsError::InvalidQuery("cleanup source must be durable"));
        }
    }
    let filters = mesh_llm_log_store::CleanupFilters::new(
        body.from
            .as_deref()
            .map(timestamp)
            .transpose()?
            .map(|value| mesh_llm_log_store::MaintenanceTimestamp::try_from(value.as_str()))
            .transpose()?,
        body.to
            .as_deref()
            .map(timestamp)
            .transpose()?
            .map(|value| mesh_llm_log_store::MaintenanceTimestamp::try_from(value.as_str()))
            .transpose()?,
        body.route,
        body.model,
        body.provider,
        body.engine,
        body.outcome
            .as_deref()
            .map(mesh_llm_log_store::CleanupOutcome::try_from)
            .transpose()?,
    )?
    .with_exclude_route(body.exclude_route)?;
    let scope = mesh_llm_log_store::CleanupScope::new(
        mesh_llm_log_store::MaintenanceTimestamp::try_from(cutoff.as_str())?,
        body.request_limit,
    )?
    .with_filters(filters);
    Ok(mesh_llm_log_store::CleanupPreviewRequest {
        operation_id,
        scope,
        reason,
    })
}

pub(super) fn cleanup_run_request(path: &str, body: &str) -> Result<CleanupRunRequest, LogsError> {
    cleanup_path(path)?;
    let body =
        serde_json::from_str::<CleanupRunBody>(body).map_err(|_| LogsError::InvalidJsonBody)?;
    Ok(CleanupRunRequest {
        operation_id: mesh_llm_log_store::MaintenanceOperationId::try_from(
            body.operation_id.as_str(),
        )?,
        reason: maintenance_reason(body.reason)?,
    })
}

fn cleanup_path(path: &str) -> Result<(), LogsError> {
    if path.contains('?') {
        Err(LogsError::InvalidQuery(
            "cleanup does not accept query parameters",
        ))
    } else {
        Ok(())
    }
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
struct ExportBody {
    reason: String,
    #[serde(default)]
    include_artifacts: bool,
}

/// Parse the export contract before a route acquires logging state. Exports
/// are durable-only so every emitted event/artifact has a stable owner.
pub(super) fn export_request(path: &str, body: &str) -> Result<ExportRequest, LogsError> {
    let parsed = request_query(path)?;
    if parsed.source.is_some() {
        return Err(LogsError::InvalidQuery(
            "export does not support active-source selection",
        ));
    }
    if parsed.store.limit > MAX_EXPORT_ROWS {
        return Err(LogsError::InvalidQuery(
            "export limit must be between 1 and 50",
        ));
    }
    let body = serde_json::from_str::<ExportBody>(body).map_err(|_| LogsError::InvalidJsonBody)?;
    Ok(ExportRequest {
        query: parsed.store,
        reason: audit_reason(body.reason)?,
        include_artifacts: body.include_artifacts,
    })
}

pub(super) fn page_query(path: &str) -> Result<PageQuery, LogsError> {
    let mut query = PageQuery {
        limit: DEFAULT_LIMIT,
        cursor: None,
        sort: QuerySort::Ascending,
    };
    for (key, value) in pairs(path)? {
        match key.as_str() {
            "cursor" => query.cursor = Some(nonempty(value)?),
            "limit" => query.limit = limit(&value)?,
            "sort" => query.sort = sort(&value)?,
            _ => return Err(LogsError::InvalidQuery("unknown page filter")),
        }
    }
    if let Some(cursor) = query.cursor.as_deref() {
        mesh_llm_log_store::decode_cursor(cursor).map_err(|_| LogsError::InvalidCursor)?;
    }
    Ok(query)
}

pub(super) fn proxy_query(path: &str) -> Result<ProxyQuery, LogsError> {
    let mut query = ProxyQuery {
        page: PageQuery {
            limit: DEFAULT_LIMIT,
            cursor: None,
            sort: QuerySort::Descending,
        },
        request_id: None,
        provider: None,
        engine: None,
        status_code: None,
    };
    for (key, value) in pairs(path)? {
        match key.as_str() {
            "cursor" => query.page.cursor = Some(nonempty(value)?),
            "limit" => query.page.limit = limit(&value)?,
            "sort" => query.page.sort = sort(&value)?,
            "request_id" => query.request_id = Some(id(&value)?),
            "provider" => query.provider = Some(filter(value)?),
            "engine" => query.engine = Some(filter(value)?),
            "status" => query.status_code = Some(status(&value)?),
            _ => return Err(LogsError::InvalidQuery("unknown proxy filter")),
        }
    }
    if let Some(cursor) = query.page.cursor.as_deref() {
        mesh_llm_log_store::decode_cursor(cursor).map_err(|_| LogsError::InvalidCursor)?;
    }
    Ok(query)
}

pub(super) struct AuditListQuery {
    pub(super) limit: usize,
    pub(super) cursor: Option<String>,
    pub(super) filters: AuditEntryFilters,
}

pub(super) fn audit_query(path: &str) -> Result<AuditListQuery, LogsError> {
    let mut parsed_limit = DEFAULT_LIMIT;
    let mut cursor = None;
    let mut filters = AuditEntryFilters::default();
    for (key, value) in pairs(path)? {
        match key.as_str() {
            "limit" => parsed_limit = limit(&value)?,
            "cursor" => cursor = Some(nonempty(value)?),
            "source" => filters.source = Some(audit_source(&value)?),
            "severity" => filters.severity = Some(audit_severity(&value)?),
            "from" => filters.from = Some(timestamp(&value)?),
            "to" => filters.to = Some(timestamp(&value)?),
            _ => return Err(LogsError::InvalidQuery("unknown audit filter")),
        }
    }
    if filters.from > filters.to && filters.to.is_some() {
        return Err(LogsError::InvalidQuery("from must not be after to"));
    }
    if let Some(ref c) = cursor {
        mesh_llm_log_store::decode_cursor(c).map_err(|_| LogsError::InvalidCursor)?;
    }
    Ok(AuditListQuery {
        limit: parsed_limit,
        cursor,
        filters,
    })
}

pub(super) fn audit_source(value: &str) -> Result<AuditEntrySource, LogsError> {
    match value {
        "logging_service" => Ok(AuditEntrySource::LoggingService),
        "runtime" => Ok(AuditEntrySource::Runtime),
        "mesh" => Ok(AuditEntrySource::Mesh),
        "cli" => Ok(AuditEntrySource::Cli),
        "logs_api" => Ok(AuditEntrySource::LogsApi),
        _ => Err(LogsError::InvalidQuery("source is invalid")),
    }
}

pub(super) fn audit_severity(value: &str) -> Result<AuditEntrySeverity, LogsError> {
    match value {
        "info" => Ok(AuditEntrySeverity::Info),
        "warning" => Ok(AuditEntrySeverity::Warning),
        "error" => Ok(AuditEntrySeverity::Error),
        _ => Err(LogsError::InvalidQuery("severity is invalid")),
    }
}

pub(super) fn id(value: &str) -> Result<String, LogsError> {
    uuid::Uuid::parse_str(value)
        .map(|id| id.to_string())
        .map_err(|_| LogsError::InvalidId)
}

fn webhook_delivery_id(value: &str) -> Result<String, LogsError> {
    if value.is_empty()
        || value.len() > MAX_WEBHOOK_DELIVERY_ID_LENGTH
        || value.chars().any(char::is_control)
        || value.contains('/')
    {
        Err(LogsError::InvalidWebhookDeliveryId)
    } else {
        Ok(value.to_owned())
    }
}

fn pairs(path: &str) -> Result<Vec<(String, String)>, LogsError> {
    let Some(raw) = path.split_once('?').map(|(_, query)| query) else {
        return Ok(Vec::new());
    };
    valid_percent_encoding(raw)?;
    let pairs = url::form_urlencoded::parse(raw.as_bytes())
        .map(|(key, value)| (key.into_owned(), value.into_owned()))
        .collect::<Vec<_>>();
    let mut keys = std::collections::HashSet::with_capacity(pairs.len());
    if pairs.iter().any(|(key, _)| !keys.insert(key.clone())) {
        return Err(LogsError::InvalidQuery("duplicate query parameter"));
    }
    Ok(pairs)
}

fn valid_percent_encoding(value: &str) -> Result<(), LogsError> {
    let bytes = value.as_bytes();
    let mut index = 0;
    while index < bytes.len() {
        if bytes[index] == b'%' {
            let valid = bytes.get(index + 1).is_some_and(u8::is_ascii_hexdigit)
                && bytes.get(index + 2).is_some_and(u8::is_ascii_hexdigit);
            if !valid {
                return Err(LogsError::InvalidQuery("query encoding is malformed"));
            }
            index += 3;
        } else {
            index += 1;
        }
    }
    Ok(())
}

fn nonempty(value: String) -> Result<String, LogsError> {
    if value.is_empty() {
        Err(LogsError::InvalidQuery("query value must not be empty"))
    } else {
        Ok(value)
    }
}

fn audit_reason(value: String) -> Result<String, LogsError> {
    let value = value.trim();
    if value.is_empty()
        || value.len() > MAX_AUDIT_REASON_LENGTH
        || value.chars().any(char::is_control)
    {
        return Err(LogsError::InvalidQuery(
            "audit reason must be non-empty and at most 256 characters",
        ));
    }
    let path_shaped = value.starts_with('/')
        || value.starts_with("~/")
        || value.as_bytes().get(1) == Some(&b':')
        || value.contains('\\');
    Ok(if path_shaped {
        "[REDACTED]".to_string()
    } else {
        crate::logging::policy::apply_redaction(value).0
    })
}

fn maintenance_reason(value: String) -> Result<mesh_llm_log_store::MaintenanceReason, LogsError> {
    let reason = audit_reason(value)?;
    mesh_llm_log_store::MaintenanceReason::try_from(reason.as_str()).map_err(Into::into)
}

fn filter(value: String) -> Result<String, LogsError> {
    if value.is_empty() || value.len() > MAX_FILTER_LENGTH || value.chars().any(char::is_control) {
        Err(LogsError::InvalidQuery("filter value is invalid"))
    } else {
        Ok(value)
    }
}

fn limit(value: &str) -> Result<usize, LogsError> {
    let value = value
        .parse::<usize>()
        .map_err(|_| LogsError::InvalidQuery("limit must be an integer"))?;
    if (1..=MAX_LIMIT).contains(&value) {
        Ok(value)
    } else {
        Err(LogsError::InvalidQuery("limit must be between 1 and 100"))
    }
}

fn timestamp(value: &str) -> Result<String, LogsError> {
    canonical_logging_timestamp(value)
        .map_err(|_| LogsError::InvalidQuery("time filter must be RFC 3339"))
}

fn status(value: &str) -> Result<u16, LogsError> {
    let status = value
        .parse::<u16>()
        .map_err(|_| LogsError::InvalidQuery("status must be an HTTP status code"))?;
    if (100..=599).contains(&status) {
        Ok(status)
    } else {
        Err(LogsError::InvalidQuery(
            "status must be between 100 and 599",
        ))
    }
}

fn outcome(value: &str) -> Result<RequestOutcome, LogsError> {
    match value {
        "active" => Ok(RequestOutcome::Active),
        "completed" => Ok(RequestOutcome::Completed),
        "failed" => Ok(RequestOutcome::Failed),
        "rejected" => Ok(RequestOutcome::Rejected),
        "cancelled" => Ok(RequestOutcome::Cancelled),
        "dropped" => Ok(RequestOutcome::Dropped),
        _ => Err(LogsError::InvalidQuery("outcome is invalid")),
    }
}

fn source_filter(value: &str) -> Result<SourceFilter, LogsError> {
    match value {
        "active" => Ok(SourceFilter::Active),
        "durable" => Ok(SourceFilter::Durable),
        _ => Err(LogsError::InvalidQuery("source is invalid")),
    }
}

fn sort(value: &str) -> Result<QuerySort, LogsError> {
    match value {
        "asc" => Ok(QuerySort::Ascending),
        "desc" => Ok(QuerySort::Descending),
        _ => Err(LogsError::InvalidQuery("sort must be asc or desc")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_forged_or_ambiguous_query_input() {
        assert!(matches!(
            request_query("/api/logs/requests?limit=101"),
            Err(LogsError::InvalidQuery(_))
        ));
        assert!(matches!(
            request_query("/api/logs/requests?limit=1&limit=2"),
            Err(LogsError::InvalidQuery(_))
        ));
        assert!(matches!(
            request_query("/api/logs/requests?cursor=not-a-cursor"),
            Err(LogsError::InvalidCursor)
        ));
        assert!(matches!(
            request_query("/api/logs/requests?model=%zz"),
            Err(LogsError::InvalidQuery(_))
        ));
    }

    #[test]
    fn normalizes_request_filters_and_uuid() {
        let request_id = uuid::Uuid::new_v4();
        let query = request_query(
            "/api/logs/requests?from=2026-01-01T01%3A00%3A00%2B01%3A00&to=2026-01-01T01%3A00%3A00Z&outcome=active&source=active&sort=asc",
        )
        .expect("parse request query");
        assert_eq!(
            query.store.from.as_deref(),
            Some("2026-01-01T00:00:00.000000000Z")
        );
        assert_eq!(query.source, Some(SourceFilter::Active));
        assert_eq!(
            id(&request_id.to_string()).expect("uuid"),
            request_id.to_string()
        );
    }

    #[test]
    fn parses_and_validates_request_route_exclusions() {
        // Given / When
        let query = request_query(
            "/api/logs/requests?exclude_route=models&exclude_route_prefix=management_",
        )
        .expect("parse route exclusions");

        // Then
        assert_eq!(query.store.exclude_route.as_deref(), Some("models"));
        assert_eq!(
            query.store.exclude_route_prefix.as_deref(),
            Some("management_")
        );
        for path in [
            "/api/logs/requests?exclude_route=",
            "/api/logs/requests?exclude_route_prefix=%00",
        ] {
            assert!(matches!(
                request_query(path),
                Err(LogsError::InvalidQuery(_))
            ));
        }
    }

    #[test]
    fn export_requires_a_bounded_reason_and_durable_scope() {
        assert!(matches!(
            export_request("/api/logs/requests/export", r#"{"reason":""}"#),
            Err(LogsError::InvalidQuery(_))
        ));
        assert!(matches!(
            export_request(
                "/api/logs/requests/export?source=active",
                r#"{"reason":"operator copy"}"#
            ),
            Err(LogsError::InvalidQuery(_))
        ));
        assert!(matches!(
            export_request(
                "/api/logs/requests/export?limit=51",
                r#"{"reason":"operator copy"}"#
            ),
            Err(LogsError::InvalidQuery(_))
        ));
    }

    #[test]
    fn cleanup_parsing_normalizes_and_rejects_unbounded_input_before_store_access() {
        let operation_id = uuid::Uuid::new_v4();
        let body = format!(
            r#"{{"operationId":"{operation_id}","cutoffBefore":"2026-08-03T01:00:00+01:00","requestLimit":1,"source":"durable","from":"2026-08-01T01:00:00+01:00","to":"2026-08-03T00:00:00Z","route":"route-a","excludeRoute":"models","model":"Qwen/Qwen3","provider":"mesh","engine":"skippy","outcome":"completed","reason":"operator cleanup"}}"#
        );
        let preview = cleanup_preview_request("/api/logs/cleanup/preview", &body)
            .expect("bounded cleanup preview");
        assert_eq!(preview.operation_id.to_string(), operation_id.to_string());
        assert_eq!(
            preview.scope.cutoff_before().as_str(),
            "2026-08-03T00:00:00.000000000Z"
        );
        assert_eq!(preview.scope.request_limit(), 1);
        assert_eq!(
            preview.scope.filters().from(),
            Some("2026-08-01T00:00:00.000000000Z")
        );
        assert_eq!(
            preview.scope.filters().to(),
            Some("2026-08-03T00:00:00.000000000Z")
        );
        assert_eq!(preview.scope.filters().model(), Some("Qwen/Qwen3"));
        assert_eq!(preview.scope.filters().exclude_route(), Some("models"));
        assert_eq!(
            preview.scope.filters().outcome(),
            Some(mesh_llm_log_store::CleanupOutcome::Completed)
        );
        assert!(matches!(
            cleanup_preview_request(
                "/api/logs/cleanup/preview",
                r#"{"operationId":"not-a-uuid","cutoffBefore":"2026-08-03T00:00:00Z","requestLimit":1,"reason":"operator cleanup"}"#
            ),
            Err(LogsError::InvalidQuery(_))
        ));
        for body in [
            r#"{"operationId":"00000000-0000-4000-8000-000000000031","cutoffBefore":"2026-08-03T00:00:00Z","requestLimit":1,"source":"active","reason":"operator cleanup"}"#,
            r#"{"operationId":"00000000-0000-4000-8000-000000000031","cutoffBefore":"2026-08-03T00:00:00Z","requestLimit":1,"outcome":"active","reason":"operator cleanup"}"#,
            r#"{"operationId":"00000000-0000-4000-8000-000000000031","cutoffBefore":"2026-08-03T00:00:00Z","requestLimit":1,"model":"/private/model?token=secret","reason":"operator cleanup"}"#,
            r#"{"operationId":"00000000-0000-4000-8000-000000000031","cutoffBefore":"2026-08-03T00:00:00Z","requestLimit":1,"from":"2026-08-04T00:00:00Z","to":"2026-08-03T00:00:00Z","reason":"operator cleanup"}"#,
        ] {
            assert!(matches!(
                cleanup_preview_request("/api/logs/cleanup/preview", body),
                Err(LogsError::InvalidQuery(_))
            ));
        }
        assert!(matches!(
            cleanup_preview_request(
                "/api/logs/cleanup/preview?limit=1",
                r#"{"operationId":"00000000-0000-4000-8000-000000000001","cutoffBefore":"2026-08-03T00:00:00Z","requestLimit":1,"reason":"operator cleanup"}"#
            ),
            Err(LogsError::InvalidQuery(_))
        ));
        assert!(matches!(
            cleanup_run_request(
                "/api/logs/cleanup/run",
                r#"{"operationId":"00000000-0000-4000-8000-000000000001","reason":""}"#
            ),
            Err(LogsError::InvalidQuery(_))
        ));
    }

    #[test]
    fn audit_query_parses_valid_limit_source_severity_bounds_and_cursor() {
        let query = audit_query(
            "/api/logs/audit?limit=10&source=mesh&severity=warning&from=2026-01-01T00%3A00%3A00Z&to=2026-01-02T00%3A00%3A00Z",
        )
        .expect("parse audit query");
        assert_eq!(query.limit, 10);
        assert_eq!(query.filters.source, Some(AuditEntrySource::Mesh));
        assert_eq!(query.filters.severity, Some(AuditEntrySeverity::Warning));
        assert_eq!(
            query.filters.from.as_deref(),
            Some("2026-01-01T00:00:00.000000000Z")
        );
        assert_eq!(
            query.filters.to.as_deref(),
            Some("2026-01-02T00:00:00.000000000Z")
        );
        assert!(query.cursor.is_none());

        let query =
            audit_query("/api/logs/audit?source=logs_api").expect("parse logs API audit source");
        assert_eq!(query.filters.source, Some(AuditEntrySource::LogsApi));

        let query = audit_query("/api/logs/audit").expect("parse bare audit query");
        assert_eq!(query.limit, DEFAULT_LIMIT);
        assert!(query.filters.source.is_none());
        assert!(query.filters.severity.is_none());
        assert!(query.filters.from.is_none());
        assert!(query.filters.to.is_none());
        assert!(query.cursor.is_none());
    }

    #[test]
    fn audit_query_validates_cursor_normalization() {
        let valid_cursor = mesh_llm_log_store::encode_cursor("2026-01-01T00:00:00Z", "some-uuid");
        let query = audit_query(&format!("/api/logs/audit?cursor={valid_cursor}"))
            .expect("parse with valid cursor");
        assert_eq!(query.cursor.as_deref(), Some(valid_cursor.as_str()));
    }

    #[test]
    fn audit_query_rejects_invalid_limit() {
        assert!(matches!(
            audit_query("/api/logs/audit?limit=0"),
            Err(LogsError::InvalidQuery(_))
        ));
        assert!(matches!(
            audit_query("/api/logs/audit?limit=101"),
            Err(LogsError::InvalidQuery(_))
        ));
        assert!(matches!(
            audit_query("/api/logs/audit?limit=abc"),
            Err(LogsError::InvalidQuery(_))
        ));
    }

    #[test]
    fn audit_query_rejects_invalid_source() {
        assert!(matches!(
            audit_query("/api/logs/audit?source=bogus"),
            Err(LogsError::InvalidQuery(_))
        ));
    }

    #[test]
    fn audit_query_rejects_invalid_severity() {
        assert!(matches!(
            audit_query("/api/logs/audit?severity=bogus"),
            Err(LogsError::InvalidQuery(_))
        ));
    }

    #[test]
    fn audit_query_rejects_malformed_cursor() {
        assert!(matches!(
            audit_query("/api/logs/audit?cursor=garbage"),
            Err(LogsError::InvalidCursor)
        ));
    }

    #[test]
    fn audit_query_rejects_duplicate_keys() {
        assert!(matches!(
            audit_query("/api/logs/audit?limit=1&limit=2"),
            Err(LogsError::InvalidQuery(_))
        ));
    }

    #[test]
    fn audit_query_rejects_unknown_keys() {
        assert!(matches!(
            audit_query("/api/logs/audit?unknown=1"),
            Err(LogsError::InvalidQuery(_))
        ));
    }

    #[test]
    fn audit_query_rejects_invalid_or_reversed_time_bounds() {
        for path in [
            "/api/logs/audit?from=not-a-time",
            "/api/logs/audit?to=not-a-time",
            "/api/logs/audit?from=2026-01-02T00%3A00%3A00Z&to=2026-01-01T00%3A00%3A00Z",
        ] {
            assert!(matches!(audit_query(path), Err(LogsError::InvalidQuery(_))));
        }
    }
}
