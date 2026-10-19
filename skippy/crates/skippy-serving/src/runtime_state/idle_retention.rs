//! Idle lane eviction uses explicit recency, never the swap-remove vector order.
use std::time::Instant;

pub(super) fn select_idle_victim(
    lanes: impl Iterator<Item = (usize, bool, Instant)>,
) -> Option<usize> {
    lanes
        .min_by_key(|(index, empty, parked)| (!empty, *parked, *index))
        .map(|(index, _, _)| index)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[test]
    fn empty_lane_wins_otherwise_oldest_cache_survives_vector_reordering() {
        let old = Instant::now();
        let recent = old + Duration::from_secs(1);
        assert_eq!(
            select_idle_victim([(0, false, old), (1, true, recent)].into_iter()),
            Some(1)
        );
        assert_eq!(
            select_idle_victim([(0, false, recent), (1, false, old)].into_iter()),
            Some(1)
        );
        assert_eq!(
            select_idle_victim([(0, false, old), (1, false, recent)].into_iter()),
            Some(0)
        );
        assert_eq!(select_idle_victim(std::iter::empty()), None);
    }
}
