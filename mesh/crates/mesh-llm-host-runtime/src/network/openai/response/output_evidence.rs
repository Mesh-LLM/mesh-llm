//! Best-effort client-side sanity bound for billed output tokens.
//!
//! Heuristic: visible generated tokens usually deliver at least one byte each,
//! and HTTP, SSE and JSON framing add more, so delivered bytes plus slack is a
//! generous bound. Hidden or special tokens (stop tokens, undelivered
//! reasoning) are exceptions; the slack absorbs a few, not a hidden-reasoning
//! bill. It catches "tiny response, enormous bill", nothing finer.

/// Allowance for special/stop tokens that produce no visible bytes.
const SLACK_TOKENS: u64 = 256;

#[derive(Default)]
pub(crate) struct OutputBound {
    bytes: u64,
}

impl OutputBound {
    pub(crate) fn observe(&mut self, bytes: &[u8]) {
        self.bytes = self.bytes.saturating_add(bytes.len() as u64);
    }

    pub(crate) fn bytes(&self) -> u64 {
        self.bytes
    }

    pub(crate) fn permits(&self, billed_tokens: u64) -> bool {
        billed_tokens <= self.bytes.saturating_add(SLACK_TOKENS)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plausible_bills_pass_and_inflated_bills_are_refused() {
        let mut bound = OutputBound::default();
        assert!(bound.permits(SLACK_TOKENS));
        assert!(!bound.permits(SLACK_TOKENS + 1));
        bound.observe(&[b'x'; 1000]);
        assert!(bound.permits(1000 + SLACK_TOKENS));
        assert!(!bound.permits(100_000));
    }
}
