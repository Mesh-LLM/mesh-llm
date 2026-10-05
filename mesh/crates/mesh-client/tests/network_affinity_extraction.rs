use mesh_client::network::affinity::AffinityRouter;

#[test]
fn affinity_tracks_sticky_and_session_fallbacks() {
    let router = AffinityRouter::new();
    router.record_sticky_route();
    router.record_session_route();

    let stats = router.stats_snapshot();
    assert_eq!(stats.sticky_routes, 1);
    assert_eq!(stats.session_routes, 1);
}

#[test]
fn affinity_stats_snapshot_initial() {
    let router = AffinityRouter::new();
    let stats = router.stats_snapshot();
    assert_eq!(stats.prefix_entries, 0);
    assert_eq!(stats.learned, 0);
    assert_eq!(stats.prefix_hits, 0);
}
