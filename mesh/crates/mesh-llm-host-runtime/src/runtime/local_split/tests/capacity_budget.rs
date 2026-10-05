use super::*;

#[test]
fn split_replan_capacity_override_prefers_unpinned_budget() {
    assert_eq!(
        split_replan_local_capacity_override(Some(6_000_000_000), None),
        Some(6_000_000_000)
    );
    assert_eq!(
        split_replan_local_capacity_override(Some(6_000_000_000), Some(24_000_000_000)),
        Some(6_000_000_000)
    );
    assert_eq!(
        split_replan_local_capacity_override(None, Some(24_000_000_000)),
        Some(24_000_000_000)
    );
    assert_eq!(
        split_replan_local_capacity_override(Some(0), Some(24_000_000_000)),
        Some(24_000_000_000),
        "a zero optional budget must not mask the pinned-device fallback"
    );
    assert_eq!(split_replan_local_capacity_override(Some(0), None), None);
}
