#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RequestRecord {
    pub request_id: String,
    pub outcome: String,
    pub created_at: String,
    pub terminal_at: Option<String>,
    pub route: Option<String>,
    pub model: Option<String>,
    pub provider: Option<String>,
    pub engine: Option<String>,
    pub status_code: Option<i64>,
}

#[non_exhaustive]
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RequestRecordWithCaller {
    pub request: RequestRecord,
    pub caller_endpoint_id: Option<String>,
    pub caller_addr: Option<String>,
    pub caller_path_type: Option<String>,
}

impl From<RequestRecordWithCaller> for RequestRecord {
    fn from(detailed: RequestRecordWithCaller) -> Self {
        detailed.request
    }
}
