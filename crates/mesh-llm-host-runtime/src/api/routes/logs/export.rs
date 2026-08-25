//! Bounded trusted-local durable log export.
//!
//! This route deliberately exports the privacy-safe DTOs used by the read
//! API, rather than raw SQLite rows or artifact paths. Artifact bytes are an
//! explicit opt-in and remain unavailable unless redacted capture is active.

use std::time::{Duration, Instant};

use mesh_llm_log_store::{ArtifactRecord, EventRecord, LogStoreError, RequestRecordWithCaller};
use serde::Serialize;
use tokio::net::TcpStream;

use super::dto::{ArtifactDto, EventDto, RequestDto};
use super::{LoggingQueryFacade, LoggingRuntimeState, LogsError, run_blocking};

const EXPORT_TIME_CAP: Duration = Duration::from_secs(2);

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct ExportDto {
    items: Vec<ExportItemDto>,
    next_cursor: Option<String>,
    truncated: bool,
    retry_required: bool,
    artifact_content_included: bool,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct ExportItemDto {
    summary: RequestDto,
    events: Vec<EventDto>,
    artifacts: Vec<ArtifactDto>,
    child_incomplete: bool,
}

pub(super) async fn handle(
    stream: &mut TcpStream,
    state: &LoggingRuntimeState,
    path: &str,
    body: &str,
) -> Result<(), LogsError> {
    let request = super::parse::export_request(path, body)?;
    let facade = super::query_facade(state)?;
    if request.include_artifacts && !facade.artifact_export_enabled() {
        return Err(LogsError::ArtifactExportForbidden);
    }
    let deadline = Instant::now() + EXPORT_TIME_CAP;
    let byte_limit = facade.export_limit_bytes();
    let reason = request.reason.clone();
    let export = tokio::time::timeout(
        EXPORT_TIME_CAP,
        run_blocking(move || {
            let selection = build_export(
                &facade,
                request.query,
                request.include_artifacts,
                byte_limit,
                deadline,
            );
            let result = match &selection {
                Ok(value) if value.truncated => "partial",
                Ok(_) => "succeeded",
                Err(_) => "failed",
            };
            // Audit persistence is deliberately best-effort: an unavailable audit
            // table must not turn a successful export into a failure, or conceal
            // the original store/timeout error from the caller.
            let _ = facade.write_operator_audit("log_export", reason, result);
            selection
        }),
    )
    .await
    .map_err(|_| LogsError::ExportTimedOut)??;
    crate::api::http::respond_json(stream, 200, &export)
        .await
        .map_err(|_| LogsError::StoreUnavailable)
}

fn build_export(
    facade: &LoggingQueryFacade,
    query: mesh_llm_log_store::RequestQuery,
    include_artifacts: bool,
    byte_limit: usize,
    deadline: Instant,
) -> Result<ExportDto, LogsError> {
    ensure_before(deadline)?;
    let page = facade.requests(&query)?;
    ensure_before(deadline)?;
    let request_ids = page
        .items
        .iter()
        .map(|record| record.request.request_id.clone())
        .collect::<Vec<_>>();
    // Each item may consume at most the complete export row budget. Fetch one
    // extra child per owner so a partial child window remains detectable while
    // this export page stays a fixed two-query fan-in instead of N+by-2.
    let child_window = super::parse::MAX_EXPORT_ROWS + 1;
    let mut children = facade.export_children(&request_ids, child_window)?;
    ensure_before(deadline)?;

    let mut export = ExportDto {
        items: Vec::new(),
        next_cursor: None,
        truncated: false,
        retry_required: false,
        artifact_content_included: false,
    };
    let mut remaining_rows = super::parse::MAX_EXPORT_ROWS;
    let mut item_lengths = Vec::new();
    let page_has_more = page.next_cursor.is_some();
    let page_len = page.items.len();
    let item_options = ExportItemOptions {
        facade,
        include_artifacts,
        byte_limit,
        deadline,
    };
    for (index, record) in page.items.into_iter().enumerate() {
        if remaining_rows == 0 {
            export.truncated = true;
            set_resume_cursor(&mut export);
            break;
        }
        let request_id = record.request.request_id.clone();
        let built = export_item(
            ExportItemInput {
                record,
                event_records: children
                    .events_by_request
                    .remove(&request_id)
                    .unwrap_or_default(),
                artifact_records: children
                    .artifacts_by_request
                    .remove(&request_id)
                    .unwrap_or_default(),
                child_row_limit: remaining_rows.saturating_sub(1),
            },
            &item_options,
        )?;
        let request_has_later = index + 1 < page_len || page_has_more;
        let used_rows = 1 + built.item.events.len() + built.item.artifacts.len();
        export.items.push(built.item);
        item_lengths.push(serialized_len(
            export.items.last().expect("export item was just pushed"),
        )?);
        // Calculate the final page semantics before the byte check. A page
        // that contains every selected summary and every child is complete,
        // even when it has more than one item. Previously `fully_exported`
        // was cleared by the first item with a later sibling and never
        // restored, so an exact final multi-item page lied with
        // `truncated: true, nextCursor: null`.
        export.truncated = built.child_truncated || request_has_later;
        export.retry_required |= built.child_truncated;
        export.next_cursor =
            (!built.child_truncated && request_has_later).then(|| cursor_for_last(&export));
        if estimated_export_len(&export, &item_lengths)? > byte_limit {
            export.items.pop();
            item_lengths.pop();
            export.truncated = true;
            set_resume_cursor(&mut export);
            if export.next_cursor.is_none() {
                export.retry_required = true;
            }
            break;
        }
        for (artifact_index, content) in built.content_artifacts {
            let item = export
                .items
                .last_mut()
                .expect("export item was just pushed");
            let metadata = std::mem::replace(&mut item.artifacts[artifact_index], content);
            let item_length = serialized_len(item)?;
            *item_lengths
                .last_mut()
                .expect("export item size was recorded") = item_length;
            if estimated_export_len(&export, &item_lengths)? > byte_limit {
                let item = export
                    .items
                    .last_mut()
                    .expect("export item was just pushed");
                item.artifacts[artifact_index] = metadata;
                *item_lengths
                    .last_mut()
                    .expect("export item size was recorded") = serialized_len(item)?;
                export.truncated = true;
                export.retry_required = true;
            } else {
                export.artifact_content_included = true;
            }
        }
        if built.child_truncated {
            // A request cursor would advance past partial child history. Keep
            // the partial item visible, require an explicit retry, and never
            // claim that the summary page can safely advance.
            export
                .items
                .last_mut()
                .expect("export item was just pushed")
                .child_incomplete = true;
            export.next_cursor = None;
            export.truncated = true;
            export.retry_required = true;
            break;
        }
        remaining_rows = remaining_rows.saturating_sub(used_rows);
    }
    ensure_final_size(&export, byte_limit)?;
    Ok(export)
}

fn cursor_for_last(export: &ExportDto) -> String {
    let item = export.items.last().expect("cursor requires an export item");
    mesh_llm_log_store::encode_cursor(item.summary.created_at(), item.summary.request_id())
}

fn set_resume_cursor(export: &mut ExportDto) {
    export.next_cursor = (!export.items.is_empty()).then(|| cursor_for_last(export));
}

struct ExportItemInput {
    record: RequestRecordWithCaller,
    event_records: Vec<EventRecord>,
    artifact_records: Vec<ArtifactRecord>,
    child_row_limit: usize,
}

struct ExportItemOptions<'a> {
    facade: &'a LoggingQueryFacade,
    include_artifacts: bool,
    byte_limit: usize,
    deadline: Instant,
}

fn export_item(
    input: ExportItemInput,
    options: &ExportItemOptions<'_>,
) -> Result<ExportItemBuild, LogsError> {
    let ExportItemInput {
        record,
        event_records,
        artifact_records,
        child_row_limit,
    } = input;
    let mut remaining_rows = child_row_limit;
    let event_limit = remaining_rows.min(super::parse::MAX_EXPORT_ROWS);
    let mut truncated = event_records.len() > event_limit;
    let events = event_records
        .into_iter()
        .take(event_limit)
        .map(EventDto::try_from)
        .collect::<Result<Vec<_>, _>>()?;
    remaining_rows = remaining_rows.saturating_sub(events.len());

    let mut artifacts = Vec::new();
    let mut content_artifacts = Vec::new();
    ensure_before(options.deadline)?;
    let artifact_limit = remaining_rows.min(super::parse::MAX_EXPORT_ROWS);
    truncated |= artifact_records.len() > artifact_limit;
    for record in artifact_records.into_iter().take(artifact_limit) {
        ensure_before(options.deadline)?;
        let content_omitted = options.include_artifacts
            && super::dto::artifact_state(&record) == "available"
            && (record.bytes.is_negative()
                || usize::try_from(record.bytes).unwrap_or(usize::MAX) > options.byte_limit / 2);
        let (metadata, content) = export_artifact(
            options.facade,
            record,
            options.include_artifacts,
            options.byte_limit,
            options.deadline,
        )?;
        truncated |= content_omitted;
        let artifact_index = artifacts.len();
        artifacts.push(metadata);
        if let Some(content) = content {
            content_artifacts.push((artifact_index, content));
        }
    }

    Ok(ExportItemBuild {
        item: ExportItemDto {
            summary: RequestDto::durable(record),
            events,
            artifacts,
            child_incomplete: false,
        },
        child_truncated: truncated,
        content_artifacts,
    })
}

struct ExportItemBuild {
    item: ExportItemDto,
    child_truncated: bool,
    content_artifacts: Vec<(usize, ArtifactDto)>,
}

fn export_artifact(
    facade: &LoggingQueryFacade,
    record: ArtifactRecord,
    include_content: bool,
    byte_limit: usize,
    deadline: Instant,
) -> Result<(ArtifactDto, Option<ArtifactDto>), LogsError> {
    // Base64 grows the captured content by roughly a third. Reserve half the
    // response for the summary/event envelope and avoid reading a file that
    // could never fit this bounded response; its pointer metadata is still
    // useful and remains retryable through a narrower export.
    let content_budget = byte_limit / 2;
    if record.bytes.is_negative()
        || usize::try_from(record.bytes).unwrap_or(usize::MAX) > content_budget
    {
        return Ok((ArtifactDto::metadata(record), None));
    }
    if !include_content || super::dto::artifact_state(&record) != "available" {
        return Ok((ArtifactDto::metadata(record), None));
    }
    ensure_before(deadline)?;
    match facade.read_artifact(&record.artifact_id) {
        Ok(content) if content.redacted => {
            let metadata = ArtifactDto::metadata(record.clone());
            Ok((metadata, Some(ArtifactDto::content(record, content))))
        }
        Ok(_) => Ok((ArtifactDto::metadata(record), None)),
        Err(LogStoreError::ArtifactMissing { .. }) => Ok((
            ArtifactDto::metadata(ArtifactRecord {
                missing: true,
                ..record
            }),
            None,
        )),
        Err(LogStoreError::ArtifactCorrupt { .. }) => Ok((
            ArtifactDto::metadata(ArtifactRecord {
                corrupt: true,
                ..record
            }),
            None,
        )),
        Err(error) => Err(error.into()),
    }
}

fn ensure_before(deadline: Instant) -> Result<(), LogsError> {
    if Instant::now() >= deadline {
        Err(LogsError::ExportTimedOut)
    } else {
        Ok(())
    }
}

fn serialized_len(value: &impl Serialize) -> Result<usize, LogsError> {
    serde_json::to_vec(value)
        .map(|bytes| bytes.len())
        .map_err(|_| LogsError::StoreUnavailable)
}

/// Compute the exact JSON envelope size from cached per-item serializations.
/// The public response has no whitespace, and replacing `items:[]` with the
/// comma-joined serialized items is the only variable-sized portion. This
/// keeps the byte-limit check linear instead of reserializing every preceding
/// item after each append or artifact-content attempt.
fn estimated_export_len(export: &ExportDto, item_lengths: &[usize]) -> Result<usize, LogsError> {
    debug_assert_eq!(export.items.len(), item_lengths.len());
    let empty_items = serde_json::json!({
        "items": [],
        "nextCursor": export.next_cursor,
        "truncated": export.truncated,
        "retryRequired": export.retry_required,
        "artifactContentIncluded": export.artifact_content_included,
    });
    let empty_len = serde_json::to_vec(&empty_items)
        .map(|bytes| bytes.len())
        .map_err(|_| LogsError::StoreUnavailable)?;
    // The empty array contributes exactly two bytes. Nonempty items add one
    // comma between adjacent serialized items.
    Ok(empty_len
        .saturating_sub(2)
        .saturating_add(item_lengths.iter().sum::<usize>())
        .saturating_add(item_lengths.len().saturating_sub(1)))
}

fn ensure_final_size(export: &ExportDto, byte_limit: usize) -> Result<(), LogsError> {
    if serialized_len(export)? > byte_limit {
        return Err(LogsError::StoreUnavailable);
    }
    Ok(())
}
