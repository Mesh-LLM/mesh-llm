use crate::repositories::AuditEntryRow;

#[non_exhaustive]
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AuditEntryDetail {
    pub entry: AuditEntryRow,
    pub remote_addr: Option<String>,
    pub path_type: Option<String>,
    pub command_summary: Option<String>,
}

impl From<AuditEntryDetail> for AuditEntryRow {
    fn from(detail: AuditEntryDetail) -> Self {
        detail.entry
    }
}
