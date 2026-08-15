use std::collections::HashSet;

use mesh_llm_events::logging::envelope::CanonicalEnvelope;
use mesh_llm_events::logging::replay::ReplayChannel;
use mesh_llm_events::logging::timestamp::canonical_logging_timestamp;

use super::super::LogsError;
use crate::logging::RequestSummaryEventSnapshots;

const MAX_FILTERS: usize = 16;
const MAX_QUERY_PAIRS: usize = 32;

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(super) struct Cursor {
    requests: u64,
    operations: u64,
    system: u64,
}

impl Cursor {
    pub(super) const fn from_sequences(requests: u64, operations: u64, system: u64) -> Self {
        Self {
            requests,
            operations,
            system,
        }
    }

    pub(super) fn parse(value: &str) -> Result<Self, LogsError> {
        let mut values = value
            .strip_prefix("v1:")
            .ok_or(LogsError::InvalidCursor)?
            .split('.');
        let cursor = Self {
            requests: parse_sequence(values.next())?,
            operations: parse_sequence(values.next())?,
            system: parse_sequence(values.next())?,
        };
        if values.next().is_some() {
            return Err(LogsError::InvalidCursor);
        }
        Ok(cursor)
    }

    pub(super) fn sequence(self, channel: ReplayChannel) -> u64 {
        match channel {
            ReplayChannel::Requests => self.requests,
            ReplayChannel::Operations => self.operations,
            ReplayChannel::System => self.system,
        }
    }

    pub(super) fn merge_max(self, other: Self) -> Self {
        Self {
            requests: self.requests.max(other.requests),
            operations: self.operations.max(other.operations),
            system: self.system.max(other.system),
        }
    }

    pub(super) fn event_id(self) -> String {
        format!("v1:{}.{}.{}", self.requests, self.operations, self.system)
    }

    pub(super) fn advance(&mut self, channel: ReplayChannel, sequence: u64) {
        let slot = match channel {
            ReplayChannel::Requests => &mut self.requests,
            ReplayChannel::Operations => &mut self.operations,
            ReplayChannel::System => &mut self.system,
        };
        *slot = (*slot).max(sequence);
    }
}

/// Audit-mode cursor: `a1:<seq>`.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(super) struct AuditCursor(pub(super) u64);

impl AuditCursor {
    pub(super) fn parse(value: &str) -> Result<Self, LogsError> {
        let seq = value
            .strip_prefix("a1:")
            .ok_or(LogsError::InvalidCursor)?
            .parse()
            .map_err(|_| LogsError::InvalidCursor)?;
        Ok(Self(seq))
    }

    pub(super) fn sequence(self) -> u64 {
        self.0
    }

    pub(super) fn event_id(self) -> String {
        format!("a1:{}", self.0)
    }

    pub(super) fn advance(&mut self, sequence: u64) {
        self.0 = self.0.max(sequence);
    }
}

/// Audit subscription filters.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub(in crate::api::routes::logs) struct AuditSelection {
    pub(super) cursor: AuditCursor,
    pub(super) source: Option<String>,
    pub(super) severity: Option<String>,
}

impl AuditSelection {
    pub(super) fn durable_filters(
        &self,
    ) -> Result<mesh_llm_log_store::AuditEntryFilters, LogsError> {
        Ok(mesh_llm_log_store::AuditEntryFilters {
            source: self
                .source
                .as_deref()
                .map(super::super::parse::audit_source)
                .transpose()?,
            severity: self
                .severity
                .as_deref()
                .map(super::super::parse::audit_severity)
                .transpose()?,
            from: None,
            to: None,
        })
    }
}

fn parse_sequence(value: Option<&str>) -> Result<u64, LogsError> {
    value
        .ok_or(LogsError::InvalidCursor)?
        .parse()
        .map_err(|_| LogsError::InvalidCursor)
}

#[derive(Clone, Debug)]
pub(in crate::api::routes::logs) struct Subscription {
    pub(super) channels: Vec<ReplayChannel>,
    pub(super) filters: LedgerFilters,
    pub(super) cursor: Cursor,
    pub(in crate::api::routes::logs) audit: Option<AuditSelection>,
}

impl Subscription {
    pub(in crate::api::routes::logs) fn is_audit(&self) -> bool {
        self.audit.is_some()
    }
}

pub(in crate::api::routes::logs) fn parse_subscription(
    path: &str,
    raw_request: &[u8],
) -> Result<Subscription, LogsError> {
    require_sse_headers(raw_request)?;
    let parsed = parse_query(path)?;
    let reconnect = unique_header(raw_request, "last-event-id")?
        .map(str::trim)
        .filter(|value| !value.is_empty());
    let cursor = if parsed.audit.is_some() {
        Cursor::default()
    } else {
        let header_cursor = reconnect
            .map(Cursor::parse)
            .transpose()?
            .unwrap_or_default();
        parsed.cursor.merge_max(header_cursor)
    };
    let audit = if let Some(sel) = parsed.audit {
        let header_cursor = reconnect
            .map(AuditCursor::parse)
            .transpose()?
            .unwrap_or_default();
        let mut merged = sel;
        if header_cursor.sequence() > merged.cursor.sequence() {
            merged.cursor = header_cursor;
        }
        Some(merged)
    } else {
        None
    };
    Ok(Subscription {
        channels: parsed.channels,
        filters: parsed.filters,
        cursor,
        audit,
    })
}

struct ParsedQuery {
    channels: Vec<ReplayChannel>,
    filters: LedgerFilters,
    cursor: Cursor,
    audit: Option<AuditSelection>,
}

/// The event stream accepts the bounded replay-matchable ledger vocabulary as
/// repeated query parameters, as well as the original `filter=field:value`
/// form. A repeated field is an OR selection; distinct fields compose with
/// AND.
#[derive(Clone, Debug, Default)]
pub(super) struct LedgerFilters {
    pub(super) request_ids: HashSet<String>,
    pub(super) routes: HashSet<String>,
    pub(super) models: HashSet<String>,
    pub(super) providers: HashSet<String>,
    pub(super) engines: HashSet<String>,
    pub(super) outcomes: HashSet<String>,
    pub(super) from: Option<String>,
    pub(super) to: Option<String>,
}

fn parse_query(path: &str) -> Result<ParsedQuery, LogsError> {
    let raw = path.split_once('?').map_or("", |(_, query)| query);
    validate_percent_encoding(raw)?;
    let pairs = url::form_urlencoded::parse(raw.as_bytes()).collect::<Vec<_>>();
    if pairs.len() > MAX_QUERY_PAIRS {
        return Err(LogsError::InvalidQuery("too many query parameters"));
    }

    // Determine the subscription mode before interpreting any other field so
    // validation is independent of query-parameter order. `audit` is a mode
    // selector, not a repeatable filter.
    let audit_values = pairs
        .iter()
        .filter(|(key, _)| key.as_ref() == "audit")
        .map(|(_, value)| value.as_ref())
        .collect::<Vec<_>>();
    if audit_values.len() > 1 {
        return Err(LogsError::InvalidQuery("duplicate audit parameter"));
    }
    let audit_mode = match audit_values.first().copied() {
        Some("1" | "true") => true,
        Some(_) => {
            return Err(LogsError::InvalidQuery("audit parameter must be 1 or true"));
        }
        None => false,
    };

    let mut channels = Vec::new();
    let mut filters = LedgerFilters::default();
    let mut cursor = None;
    let mut audit_selection = AuditSelection::default();
    let mut audit_cursor_set = false;

    for (key, value) in pairs {
        match key.as_ref() {
            "audit" => {
                // Validated in the mode-selection pass above.
            }
            "channel" => {
                if audit_mode {
                    return Err(LogsError::InvalidQuery(
                        "channel is not supported in audit mode",
                    ));
                }
                let channel = parse_channel(&value)?;
                if channels.contains(&channel) {
                    return Err(LogsError::InvalidQuery("duplicate channel"));
                }
                channels.push(channel);
            }
            "filter" => {
                if audit_mode {
                    return Err(LogsError::InvalidQuery(
                        "filter is not supported in audit mode",
                    ));
                }
                filters.insert_encoded(&value)?;
            }
            "from" | "to" | "route" | "model" | "provider" | "engine" | "outcome"
            | "request_id" => {
                if audit_mode {
                    return Err(LogsError::InvalidQuery(
                        "lifecycle filters are not supported in audit mode",
                    ));
                }
                filters.insert(key.as_ref(), &value)?;
            }
            "source" => {
                if audit_mode {
                    let source = parse_audit_source(&value)?;
                    if audit_selection.source.is_some() {
                        return Err(LogsError::InvalidQuery("duplicate audit source"));
                    }
                    audit_selection.source = Some(source);
                } else {
                    return Err(LogsError::InvalidQuery(
                        "filter is unsupported for event streams",
                    ));
                }
            }
            "severity" => {
                if audit_mode {
                    let severity = parse_audit_severity(&value)?;
                    if audit_selection.severity.is_some() {
                        return Err(LogsError::InvalidQuery("duplicate audit severity"));
                    }
                    audit_selection.severity = Some(severity);
                } else {
                    return Err(LogsError::InvalidQuery("unknown event stream parameter"));
                }
            }
            "cursor" if cursor.is_none() && !audit_mode => {
                cursor = Some(Cursor::parse(nonempty(&value)?)?);
            }
            "cursor" if audit_mode && !audit_cursor_set => {
                audit_selection.cursor = AuditCursor::parse(nonempty(&value)?)?;
                audit_cursor_set = true;
            }
            "cursor" => return Err(LogsError::InvalidQuery("duplicate cursor")),
            _ => return Err(LogsError::InvalidQuery("unknown event stream parameter")),
        }
    }

    if !audit_mode && channels.is_empty() {
        return Err(LogsError::InvalidQuery("at least one channel is required"));
    }
    if !audit_mode {
        filters.validate_time_range()?;
    }

    Ok(ParsedQuery {
        channels,
        filters,
        cursor: cursor.unwrap_or_default(),
        audit: if audit_mode {
            Some(audit_selection)
        } else {
            None
        },
    })
}

fn parse_audit_source(value: &str) -> Result<String, LogsError> {
    match value {
        "logging_service" | "runtime" | "mesh" | "cli" | "logs_api" => Ok(value.to_owned()),
        _ => Err(LogsError::InvalidQuery("audit source is invalid")),
    }
}

fn parse_audit_severity(value: &str) -> Result<String, LogsError> {
    match value {
        "info" | "warning" | "error" => Ok(value.to_owned()),
        _ => Err(LogsError::InvalidQuery("audit severity is invalid")),
    }
}

impl LedgerFilters {
    fn insert_encoded(&mut self, value: &str) -> Result<(), LogsError> {
        let (key, value) = value
            .split_once(':')
            .ok_or(LogsError::InvalidQuery("filter is invalid"))?;
        self.insert(key, value)
    }

    fn insert(&mut self, key: &str, value: &str) -> Result<(), LogsError> {
        if self.len() >= MAX_FILTERS {
            return Err(LogsError::InvalidQuery("too many filters"));
        }
        match key {
            "request_id" => insert_filter(&mut self.request_ids, parse_request_id(value)?),
            "route" => insert_filter(&mut self.routes, filter_value(value)?),
            "model" => insert_filter(&mut self.models, filter_value(value)?),
            "provider" => insert_filter(&mut self.providers, filter_value(value)?),
            "engine" => insert_filter(&mut self.engines, filter_value(value)?),
            "outcome" => insert_filter(&mut self.outcomes, parse_outcome(value)?),
            "from" => insert_time_filter(&mut self.from, value),
            "to" => insert_time_filter(&mut self.to, value),
            "source" => {
                return Err(LogsError::InvalidQuery(
                    "filter is unsupported for event streams",
                ));
            }
            _ => return Err(LogsError::InvalidQuery("unknown event stream filter")),
        }?;
        Ok(())
    }

    fn len(&self) -> usize {
        self.request_ids.len()
            + self.routes.len()
            + self.models.len()
            + self.providers.len()
            + self.engines.len()
            + self.outcomes.len()
            + usize::from(self.from.is_some())
            + usize::from(self.to.is_some())
    }

    fn validate_time_range(&self) -> Result<(), LogsError> {
        if self.from > self.to && self.to.is_some() {
            Err(LogsError::InvalidQuery("from must not be after to"))
        } else {
            Ok(())
        }
    }
}

impl LedgerFilters {
    pub(super) fn matches(
        &self,
        envelope: &CanonicalEnvelope,
        summary_snapshots: Option<&RequestSummaryEventSnapshots>,
    ) -> bool {
        if !self.request_ids.is_empty()
            && !self
                .request_ids
                .contains(&envelope.request_id.as_uuid().to_string())
        {
            return false;
        }
        !self.has_membership_filters()
            || summary_snapshots.is_some_and(|snapshots| {
                snapshots
                    .iter()
                    .any(|snapshot| self.matches_snapshot(snapshot))
            })
    }

    fn has_membership_filters(&self) -> bool {
        self.from.is_some()
            || self.to.is_some()
            || !self.routes.is_empty()
            || !self.models.is_empty()
            || !self.providers.is_empty()
            || !self.engines.is_empty()
            || !self.outcomes.is_empty()
    }

    fn matches_snapshot(&self, snapshot: &crate::logging::RequestSummarySnapshot) -> bool {
        if self
            .from
            .as_ref()
            .is_some_and(|from| snapshot.created_at() < from.as_str())
            || self
                .to
                .as_ref()
                .is_some_and(|to| snapshot.created_at() > to.as_str())
        {
            return false;
        }
        let metadata = snapshot.metadata();
        matches_metadata(&self.routes, metadata.route())
            && matches_metadata(&self.models, metadata.model())
            && matches_metadata(&self.providers, metadata.provider())
            && matches_metadata(&self.engines, metadata.engine())
            && matches_outcome(&self.outcomes, snapshot.state())
    }
}

fn matches_metadata(filters: &HashSet<String>, value: Option<&str>) -> bool {
    filters.is_empty() || value.is_some_and(|value| filters.contains(value))
}

fn matches_outcome(filters: &HashSet<String>, state: &str) -> bool {
    filters.is_empty() || filters.contains(state)
}

fn insert_filter(filters: &mut HashSet<String>, value: String) -> Result<(), LogsError> {
    if filters.insert(value) {
        Ok(())
    } else {
        Err(LogsError::InvalidQuery("duplicate filter"))
    }
}

fn insert_time_filter(slot: &mut Option<String>, value: &str) -> Result<(), LogsError> {
    if slot.is_some() {
        return Err(LogsError::InvalidQuery("duplicate time filter"));
    }
    *slot = Some(timestamp(value)?);
    Ok(())
}

fn parse_channel(value: &str) -> Result<ReplayChannel, LogsError> {
    match value {
        "requests" => Ok(ReplayChannel::Requests),
        "operations" => Ok(ReplayChannel::Operations),
        "system" => Ok(ReplayChannel::System),
        _ => Err(LogsError::InvalidQuery("channel is invalid")),
    }
}

fn parse_request_id(value: &str) -> Result<String, LogsError> {
    uuid::Uuid::parse_str(value)
        .map(|id| id.to_string())
        .map_err(|_| LogsError::InvalidQuery("filter request ID is invalid"))
}

fn filter_value(value: &str) -> Result<String, LogsError> {
    if value.is_empty() || value.len() > 128 || value.chars().any(char::is_control) {
        Err(LogsError::InvalidQuery("filter value is invalid"))
    } else {
        Ok(value.to_owned())
    }
}

fn timestamp(value: &str) -> Result<String, LogsError> {
    canonical_logging_timestamp(value)
        .map_err(|_| LogsError::InvalidQuery("time filter must be RFC 3339"))
}

fn parse_outcome(value: &str) -> Result<String, LogsError> {
    match value {
        "active" | "completed" | "failed" | "rejected" | "cancelled" | "dropped" => {
            Ok(value.to_owned())
        }
        _ => Err(LogsError::InvalidQuery("outcome is invalid")),
    }
}

fn require_sse_headers(raw_request: &[u8]) -> Result<(), LogsError> {
    let Some(accept) = unique_header(raw_request, "accept")? else {
        return Err(LogsError::NotAcceptable);
    };
    let accepts_sse = accept.split(',').any(|range| {
        range
            .split(';')
            .next()
            .is_some_and(|media| media.trim().eq_ignore_ascii_case("text/event-stream"))
    });
    if !accepts_sse {
        return Err(LogsError::NotAcceptable);
    }
    unique_header(raw_request, "host")?
        .filter(|value| !value.trim().is_empty())
        .ok_or(LogsError::InvalidHeaders)?;
    Ok(())
}

fn unique_header<'a>(raw_request: &'a [u8], name: &str) -> Result<Option<&'a str>, LogsError> {
    let mut headers = [httparse::EMPTY_HEADER; 64];
    let mut request = httparse::Request::new(&mut headers);
    if !matches!(
        request.parse(raw_request),
        Ok(httparse::Status::Complete(_))
    ) {
        return Err(LogsError::InvalidHeaders);
    }
    let mut values = request
        .headers
        .iter()
        .filter(|header| header.name.eq_ignore_ascii_case(name));
    let value = values.next();
    if values.next().is_some() {
        return Err(LogsError::InvalidHeaders);
    }
    value
        .map(|header| std::str::from_utf8(header.value).map_err(|_| LogsError::InvalidHeaders))
        .transpose()
}

fn nonempty(value: &str) -> Result<&str, LogsError> {
    if value.is_empty() {
        Err(LogsError::InvalidQuery("query value must not be empty"))
    } else {
        Ok(value)
    }
}

fn validate_percent_encoding(value: &str) -> Result<(), LogsError> {
    let bytes = value.as_bytes();
    let mut index = 0;
    while index < bytes.len() {
        if bytes[index] == b'%' {
            if !bytes.get(index + 1).is_some_and(u8::is_ascii_hexdigit)
                || !bytes.get(index + 2).is_some_and(u8::is_ascii_hexdigit)
            {
                return Err(LogsError::InvalidQuery("query encoding is malformed"));
            }
            index += 3;
        } else {
            index += 1;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    const REQUEST: &[u8] =
        b"GET /api/logs/events HTTP/1.1\r\nHost: localhost\r\nAccept: text/event-stream\r\n\r\n";

    #[test]
    fn repeated_ledger_filters_decode_and_normalize() {
        let parsed = parse_subscription(
            "/api/logs/events?channel=requests&channel=operations&model=Qwen%2FQwen3&filter=model%3AQwen%2FQwen2.5&provider=mesh&filter=engine%3Askippy&filter=request_id%3A00000000-0000-4000-8000-000000000001&request_id=00000000-0000-4000-8000-000000000002&from=2026-08-03T01%3A00%3A00%2B01%3A00&to=2026-08-03T01%3A00%3A00Z&outcome=completed&cursor=v1%3A2.3.4",
            REQUEST,
        )
        .expect("valid repeated SSE selection");
        assert_eq!(
            parsed.channels,
            vec![ReplayChannel::Requests, ReplayChannel::Operations]
        );
        assert_eq!(parsed.filters.models.len(), 2);
        assert_eq!(parsed.filters.request_ids.len(), 2);
        assert_eq!(
            parsed.filters.from.as_deref(),
            Some("2026-08-03T00:00:00.000000000Z")
        );
        assert_eq!(
            parsed.filters.to.as_deref(),
            Some("2026-08-03T01:00:00.000000000Z")
        );
        assert_eq!(parsed.cursor.sequence(ReplayChannel::Requests), 2);
    }

    #[test]
    fn ledger_time_bounds_compare_instants_within_the_boundary_second() {
        fn snapshot(created_at: &str) -> RequestSummaryEventSnapshots {
            RequestSummaryEventSnapshots::current(&crate::logging::RequestSummaryEntry {
                request_id: uuid::Uuid::new_v4().to_string(),
                state: "active".to_string(),
                created_at: created_at.to_string(),
                terminal_at: None,
                metadata: crate::logging::RequestSummaryMetadata::default(),
            })
        }

        let request_id = mesh_llm_events::logging::identifiers::RequestId::new();
        let envelope = CanonicalEnvelope::new(
            mesh_llm_events::logging::identifiers::EventId::new(),
            request_id,
            ReplayChannel::Requests,
            1,
            "2026-08-03T00:00:00.123Z".to_string(),
            mesh_llm_events::logging::events::LifecycleEvent::Admitted {
                model: None,
                method: None,
            },
        );
        let at_boundary = snapshot("2026-08-03T00:00:00Z");
        let after_boundary = snapshot("2026-08-03T00:00:00.123Z");

        let from = parse_subscription(
            "/api/logs/events?channel=requests&from=2026-08-03T00%3A00%3A00Z",
            REQUEST,
        )
        .expect("lower-bound subscription parses");
        assert!(from.filters.matches(&envelope, Some(&at_boundary)));
        assert!(from.filters.matches(&envelope, Some(&after_boundary)));

        let to = parse_subscription(
            "/api/logs/events?channel=requests&to=2026-08-03T00%3A00%3A00Z",
            REQUEST,
        )
        .expect("upper-bound subscription parses");
        assert!(to.filters.matches(&envelope, Some(&at_boundary)));
        assert!(!to.filters.matches(&envelope, Some(&after_boundary)));
    }

    #[test]
    fn last_event_id_merges_with_explicit_cursor_per_channel() {
        let request = b"GET /api/logs/events HTTP/1.1\r\nHost: localhost\r\nAccept: text/event-stream\r\nLast-Event-ID: v1:1.5.0\r\n\r\n";
        let parsed = parse_subscription(
            "/api/logs/events?channel=requests&channel=operations&cursor=v1%3A2.0.3",
            request,
        )
        .expect("cursor sources compose");
        assert_eq!(parsed.cursor.sequence(ReplayChannel::Requests), 2);
        assert_eq!(parsed.cursor.sequence(ReplayChannel::Operations), 5);
        assert_eq!(parsed.cursor.sequence(ReplayChannel::System), 3);
    }

    #[test]
    fn malformed_query_filter_and_headers_are_rejected() {
        for path in [
            "/api/logs/events?channel=requests&filter=not-a-filter",
            "/api/logs/events?channel=requests&filter=request_id%ZZbad",
            "/api/logs/events?channel=requests&filter=unknown%3Avalue",
            "/api/logs/events?channel=requests&filter=outcome%3Aunknown",
            "/api/logs/events?channel=requests&filter=from%3Anot-a-time",
            "/api/logs/events?channel=requests&from=2026-08-04T00%3A00%3A00Z&to=2026-08-03T00%3A00%3A00Z",
            "/api/logs/events?channel=requests&filter=route%3A%00",
            "/api/logs/events?channel=requests&source=active",
            "/api/logs/events?channel=unknown",
            "/api/logs/events?channel=requests&cursor=not-a-cursor",
        ] {
            assert!(
                parse_subscription(path, REQUEST).is_err(),
                "must reject {path}"
            );
        }
        let wrong_accept =
            b"GET /api/logs/events HTTP/1.1\r\nHost: localhost\r\nAccept: application/json\r\n\r\n";
        assert!(matches!(
            parse_subscription("/api/logs/events?channel=requests", wrong_accept),
            Err(LogsError::NotAcceptable)
        ));
    }

    #[test]
    fn malformed_sse_headers_keep_the_header_error_variant() {
        let raw = b"GET /api/logs/events HTTP/1.1\r\nAccept: text/event-stream\r\n\r\n";
        assert_eq!(require_sse_headers(raw), Err(LogsError::InvalidHeaders));
    }

    #[test]
    fn route_is_supported_while_source_remains_unsupported() {
        let subscription = parse_subscription(
            "/api/logs/events?channel=requests&route=chat_completions",
            REQUEST,
        )
        .expect("route filter parses");
        assert!(subscription.filters.routes.contains("chat_completions"));
        assert!(matches!(
            parse_subscription(
                "/api/logs/events?channel=requests&filter=source%3Aactive",
                REQUEST
            ),
            Err(LogsError::InvalidQuery(
                "filter is unsupported for event streams"
            ))
        ));
    }

    #[test]
    fn audit_mode_parses_without_channel() {
        let subscription =
            parse_subscription("/api/logs/events?audit=1", REQUEST).expect("audit mode parses");
        assert!(subscription.is_audit());
        assert!(subscription.channels.is_empty());
    }

    #[test]
    fn audit_mode_accepts_source_and_severity() {
        let subscription = parse_subscription(
            "/api/logs/events?audit=1&source=mesh&severity=warning",
            REQUEST,
        )
        .expect("audit with filters parses");
        let sel = subscription.audit.as_ref().unwrap();
        assert_eq!(sel.source.as_deref(), Some("mesh"));
        assert_eq!(sel.severity.as_deref(), Some("warning"));
    }

    #[test]
    fn audit_mode_parses_audit_cursor() {
        let subscription = parse_subscription("/api/logs/events?audit=1&cursor=a1:42", REQUEST)
            .expect("audit cursor parses");
        let sel = subscription.audit.as_ref().unwrap();
        assert_eq!(sel.cursor.sequence(), 42);
    }

    #[test]
    fn audit_mode_rejects_lifecycle_filters() {
        for path in [
            "/api/logs/events?audit=1&channel=requests",
            "/api/logs/events?audit=1&route=chat",
            "/api/logs/events?audit=1&model=Qwen",
            "/api/logs/events?audit=1&provider=mesh",
            "/api/logs/events?audit=1&engine=skippy",
            "/api/logs/events?audit=1&outcome=completed",
            "/api/logs/events?audit=1&request_id=00000000-0000-4000-8000-000000000001",
            "/api/logs/events?audit=1&from=2026-01-01T00:00:00Z",
            "/api/logs/events?audit=1&to=2026-01-01T00:00:00Z",
            "/api/logs/events?audit=1&filter=source:active",
        ] {
            assert!(
                parse_subscription(path, REQUEST).is_err(),
                "audit mode must reject lifecycle param: {path}"
            );
        }
    }

    #[test]
    fn audit_mode_validation_is_independent_of_parameter_order() {
        for path in [
            "/api/logs/events?channel=requests&audit=1",
            "/api/logs/events?cursor=v1:1.2.3&audit=1",
            "/api/logs/events?source=runtime&audit=1&channel=requests",
        ] {
            assert!(
                parse_subscription(path, REQUEST).is_err(),
                "audit mode must reject lifecycle fields regardless of order: {path}"
            );
        }

        let subscription = parse_subscription(
            "/api/logs/events?cursor=a1:7&source=runtime&severity=warning&audit=true",
            REQUEST,
        )
        .expect("audit fields before the mode selector must parse");
        let audit = subscription.audit.expect("audit selection");
        assert_eq!(audit.cursor, AuditCursor(7));
        assert_eq!(audit.source.as_deref(), Some("runtime"));
        assert_eq!(audit.severity.as_deref(), Some("warning"));
    }

    #[test]
    fn duplicate_audit_mode_selector_is_rejected() {
        assert!(matches!(
            parse_subscription("/api/logs/events?audit=1&audit=true", REQUEST),
            Err(LogsError::InvalidQuery("duplicate audit parameter"))
        ));
    }

    #[test]
    fn audit_mode_rejects_invalid_source_and_severity() {
        for path in [
            "/api/logs/events?audit=1&source=bogus",
            "/api/logs/events?audit=1&severity=bogus",
        ] {
            assert!(
                parse_subscription(path, REQUEST).is_err(),
                "audit mode must reject invalid vocab: {path}"
            );
        }
    }

    #[test]
    fn audit_mode_rejects_non_boolean_audit_param() {
        assert!(matches!(
            parse_subscription("/api/logs/events?audit=yes", REQUEST),
            Err(LogsError::InvalidQuery(_))
        ));
    }

    #[test]
    fn audit_last_event_id_merges_with_query_cursor() {
        let request = b"GET /api/logs/events HTTP/1.1\r\nHost: localhost\r\nAccept: text/event-stream\r\nLast-Event-ID: a1:100\r\n\r\n";
        let subscription = parse_subscription("/api/logs/events?audit=1&cursor=a1:50", request)
            .expect("audit merge parses");
        let sel = subscription.audit.as_ref().unwrap();
        assert_eq!(sel.cursor.sequence(), 100);
    }

    #[test]
    fn audit_cursor_rejects_malformed() {
        for path in [
            "/api/logs/events?audit=1&cursor=garbage",
            "/api/logs/events?audit=1&cursor=v1:1.2.3",
            "/api/logs/events?audit=1&cursor=a1:",
        ] {
            assert!(
                parse_subscription(path, REQUEST).is_err(),
                "malformed audit cursor must reject: {path}"
            );
        }
    }

    #[test]
    fn non_audit_mode_still_rejects_source() {
        assert!(matches!(
            parse_subscription("/api/logs/events?channel=requests&source=active", REQUEST),
            Err(LogsError::InvalidQuery(
                "filter is unsupported for event streams"
            ))
        ));
    }
}
