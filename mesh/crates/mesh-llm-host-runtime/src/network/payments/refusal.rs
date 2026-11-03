//! Bounded pre-admission reasons carried in the existing Error message.
use serde::Serialize;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum Refusal {
    ProtocolVersion,
    DebtBlocked,
    AdmissionRefused,
}

impl Refusal {
    pub(crate) fn message(self) -> &'static str {
        match self {
            Self::ProtocolVersion => {
                "payment_refused:protocol_version; this seller requires token-metered payment protocol v3; upgrade mesh-llm"
            }
            Self::DebtBlocked => "payment_refused:debt_blocked",
            Self::AdmissionRefused => "payment_refused:admission_refused",
        }
    }

    pub(crate) fn classify(message: &str) -> Self {
        match message.split(';').next().unwrap_or(message) {
            "payment_refused:protocol_version" => Self::ProtocolVersion,
            "payment_refused:debt_blocked" => Self::DebtBlocked,
            _ => match message {
                "this seller requires token-metered payment protocol v3; upgrade mesh-llm"
                | "this seller requires token-metered payment protocol v2; upgrade mesh-llm" => {
                    Self::ProtocolVersion
                }
                _ => Self::AdmissionRefused,
            },
        }
    }
}

impl std::fmt::Display for Refusal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.message())
    }
}

impl std::error::Error for Refusal {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn refusal_codes_are_bounded_and_require_explicit_evidence() {
        for reason in [
            Refusal::ProtocolVersion,
            Refusal::DebtBlocked,
            Refusal::AdmissionRefused,
        ] {
            assert_eq!(Refusal::classify(reason.message()), reason);
        }
        assert_eq!(
            Refusal::classify(
                "this seller requires token-metered payment protocol v3; upgrade mesh-llm"
            ),
            Refusal::ProtocolVersion
        );
        for message in [
            "payment_refused:unknown",
            "payment_refused:protocol_version_suffix",
            "unpaid final invoices",
            "HTTP/1.1 404 Not Found",
        ] {
            assert_eq!(Refusal::classify(message), Refusal::AdmissionRefused);
        }
    }
}
