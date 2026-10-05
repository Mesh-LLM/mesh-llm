use super::super::{ArtifactRecord, EventRecord, ProxyRecord};

pub(super) fn event_record(row: &rusqlite::Row<'_>) -> rusqlite::Result<EventRecord> {
    Ok(EventRecord {
        event_id: row.get(0)?,
        request_id: row.get(1)?,
        occurred_at: row.get(2)?,
        payload_json: row.get(3)?,
    })
}

pub(super) fn artifact_record(row: &rusqlite::Row<'_>) -> rusqlite::Result<ArtifactRecord> {
    Ok(ArtifactRecord {
        artifact_id: row.get(0)?,
        request_id: row.get(1)?,
        occurred_at: row.get(2)?,
        kind: row.get(3)?,
        media_kind: row.get(4)?,
        checksum: row.get(5)?,
        bytes: row.get(6)?,
        version: row.get(7)?,
        redacted: row.get::<_, i32>(8)? != 0,
        truncated: row.get::<_, i32>(9)? != 0,
        missing: row.get::<_, i32>(10)? != 0,
        corrupt: row.get::<_, i32>(11)? != 0,
        unavailable_reason: row.get(12)?,
    })
}

pub(super) fn proxy_record(row: &rusqlite::Row<'_>) -> rusqlite::Result<ProxyRecord> {
    Ok(ProxyRecord {
        attempt_id: row.get(0)?,
        request_id: row.get(1)?,
        occurred_at: row.get(2)?,
        target: row.get(3)?,
        provider: row.get(4)?,
        engine: row.get(5)?,
        started_at: row.get(6)?,
        completed_at: row.get(7)?,
        status_code: row.get(8)?,
    })
}
