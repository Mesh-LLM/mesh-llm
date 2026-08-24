/// Error type returned when bus enqueue fails (shouldn't happen with drop-oldest, but kept for API completeness).
#[derive(Clone, Debug)]
pub enum BusEnqueueError {
    /// The sink is unavailable and the error-audit fallback also failed.
    SinkUnavailable(String),
}

impl std::fmt::Display for BusEnqueueError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::SinkUnavailable(msg) => write!(f, "sink unavailable: {}", msg),
        }
    }
}

impl std::error::Error for BusEnqueueError {}
