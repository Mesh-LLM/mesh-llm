use std::net::SocketAddr;

use super::RequestSummaryMetadata;

#[derive(Clone, Copy, Debug, Eq, PartialEq, serde::Deserialize, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum CallerPathType {
    LocalHttp,
    RemoteQuicHttp,
    Relay,
}

impl CallerPathType {
    pub(crate) const fn as_str(self) -> &'static str {
        match self {
            Self::LocalHttp => "local_http",
            Self::RemoteQuicHttp => "remote_quic_http",
            Self::Relay => "relay",
        }
    }
}

impl RequestSummaryMetadata {
    pub(crate) fn with_caller_identity(
        mut self,
        endpoint_id: Option<&str>,
        addr: Option<&str>,
        path_type: Option<CallerPathType>,
    ) -> Self {
        let endpoint_id = endpoint_id
            .filter(|value| authenticated_endpoint_id(value))
            .map(str::to_owned);
        let (caller_endpoint_id, caller_addr, caller_path_type) = match path_type {
            Some(CallerPathType::LocalHttp) if endpoint_id.is_none() => {
                match bounded_caller_addr(addr) {
                    Some(addr) => (None, Some(addr), path_type),
                    None => (None, None, None),
                }
            }
            Some(CallerPathType::RemoteQuicHttp) if endpoint_id.is_some() => {
                (endpoint_id, bounded_caller_addr(addr), path_type)
            }
            Some(CallerPathType::Relay) if endpoint_id.is_some() => (endpoint_id, None, path_type),
            None if endpoint_id.is_some() && addr.is_none() => (endpoint_id, None, None),
            Some(
                CallerPathType::LocalHttp | CallerPathType::RemoteQuicHttp | CallerPathType::Relay,
            )
            | None => (None, None, None),
        };
        self.caller_endpoint_id = caller_endpoint_id;
        self.caller_addr = caller_addr;
        self.caller_path_type = caller_path_type;
        self
    }

    pub(crate) fn caller_endpoint_id(&self) -> Option<&str> {
        self.caller_endpoint_id.as_deref()
    }

    pub(crate) fn caller_addr(&self) -> Option<&str> {
        self.caller_addr.as_deref()
    }

    pub(crate) fn caller_path_type(&self) -> Option<&'static str> {
        self.caller_path_type.map(CallerPathType::as_str)
    }

    pub(crate) fn merge_authenticated_remote_caller(&mut self, update: Self) -> bool {
        if !update.has_authenticated_remote_caller() || self.has_authenticated_remote_caller() {
            return false;
        }

        let changed = self.caller_endpoint_id != update.caller_endpoint_id
            || self.caller_addr != update.caller_addr
            || self.caller_path_type != update.caller_path_type;
        self.caller_endpoint_id = update.caller_endpoint_id;
        self.caller_addr = update.caller_addr;
        self.caller_path_type = update.caller_path_type;
        changed
    }

    pub(super) fn merge_missing_caller(
        &mut self,
        caller_endpoint_id: Option<String>,
        caller_addr: Option<String>,
        caller_path_type: Option<CallerPathType>,
    ) -> bool {
        let current_is_empty = self.caller_endpoint_id.is_none()
            && self.caller_addr.is_none()
            && self.caller_path_type.is_none();
        let update_is_empty =
            caller_endpoint_id.is_none() && caller_addr.is_none() && caller_path_type.is_none();
        if !current_is_empty || update_is_empty {
            return false;
        }
        self.caller_endpoint_id = caller_endpoint_id;
        self.caller_addr = caller_addr;
        self.caller_path_type = caller_path_type;
        true
    }

    pub(crate) fn has_authenticated_remote_caller(&self) -> bool {
        self.caller_endpoint_id
            .as_deref()
            .is_some_and(authenticated_endpoint_id)
            && match self.caller_path_type {
                Some(CallerPathType::LocalHttp) => false,
                Some(CallerPathType::RemoteQuicHttp) => true,
                Some(CallerPathType::Relay) | None => self.caller_addr.is_none(),
            }
    }
}

fn bounded_caller_addr(value: Option<&str>) -> Option<String> {
    value?
        .parse::<SocketAddr>()
        .ok()
        .map(|addr| addr.to_string())
}

pub(crate) fn authenticated_endpoint_id(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

#[cfg(test)]
mod tests;
