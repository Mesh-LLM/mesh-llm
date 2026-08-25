use super::super::{RequestRecord, RequestRecordWithCaller};

pub(super) const REQUEST_COLUMNS: &str = "request_id, state, created_at, terminal_at, route, model, provider, engine, status_code, \
     caller_endpoint_id, caller_addr, caller_path_type";

pub(super) fn request_record_with_caller(
    row: &rusqlite::Row<'_>,
) -> rusqlite::Result<RequestRecordWithCaller> {
    Ok(RequestRecordWithCaller {
        request: RequestRecord {
            request_id: row.get(0)?,
            outcome: row.get(1)?,
            created_at: row.get(2)?,
            terminal_at: row.get(3)?,
            route: row.get(4)?,
            model: row.get(5)?,
            provider: row.get(6)?,
            engine: row.get(7)?,
            status_code: row.get(8)?,
        },
        caller_endpoint_id: row.get(9)?,
        caller_addr: row.get(10)?,
        caller_path_type: row.get(11)?,
    })
}
