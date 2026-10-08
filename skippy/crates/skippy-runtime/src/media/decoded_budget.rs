//! Per-request bound on decoded media.
//!
//! Each media part is decoded before the prompt is evaluated, and every
//! decoded part stays in memory until then. Each part is bounded by what its
//! header may declare (see `declared_size`), but a request with many parts
//! could still hold several of the largest allowed parts at once. The budget
//! fails the request as soon as the parts decoded so far exceed it.

use super::MediaRejected;

/// Decoded bytes all media parts of one request may hold at once. Four
/// 8-megapixel RGB images use about 96 MiB; one hour of 16 kHz audio about
/// 230 MiB.
pub(super) const MAX_DECODED_MEDIA_BYTES_PER_REQUEST: usize = 512 * 1024 * 1024;

#[derive(Debug, Default)]
pub(super) struct DecodedMediaBudget {
    used: usize,
}

impl DecodedMediaBudget {
    /// Records one more decoded part and fails once the request's parts
    /// together exceed [`MAX_DECODED_MEDIA_BYTES_PER_REQUEST`].
    pub(super) fn charge(&mut self, decoded_bytes: usize) -> Result<(), MediaRejected> {
        self.used = self.used.saturating_add(decoded_bytes);
        if self.used > MAX_DECODED_MEDIA_BYTES_PER_REQUEST {
            return Err(MediaRejected::new(format!(
                "media in one request decodes to more than {} MiB",
                MAX_DECODED_MEDIA_BYTES_PER_REQUEST / (1024 * 1024)
            )));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn allows_parts_up_to_the_request_budget() {
        let mut budget = DecodedMediaBudget::default();
        budget
            .charge(MAX_DECODED_MEDIA_BYTES_PER_REQUEST / 2)
            .unwrap();
        budget
            .charge(MAX_DECODED_MEDIA_BYTES_PER_REQUEST / 2)
            .unwrap();
    }

    #[test]
    fn rejects_the_part_that_crosses_the_request_budget() {
        let mut budget = DecodedMediaBudget::default();
        budget.charge(MAX_DECODED_MEDIA_BYTES_PER_REQUEST).unwrap();
        assert!(budget.charge(1).is_err());

        let mut overflowing = DecodedMediaBudget::default();
        overflowing.charge(usize::MAX).unwrap_err();
    }
}
