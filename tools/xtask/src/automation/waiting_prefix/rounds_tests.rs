use super::*;
use serde_json::json;

#[test]
fn alternating_launch_order_preserves_both_versions_in_every_round() {
    assert_eq!(
        schedule(3).unwrap(),
        vec![
            (1, Version::Old),
            (1, Version::New),
            (2, Version::New),
            (2, Version::Old),
            (3, Version::Old),
            (3, Version::New)
        ]
    );
    assert!(schedule(0).is_err());
    assert!(schedule(1001).is_err());
}

fn cell(round: u64, version: Version) -> Cell {
    serde_json::from_value(json!({"round":round,"version":version,"summary":{
        "requests":1,"successful":1,"cache_hits":0,"cache_misses":1,"usage_cached_requests":0,
        "capacity_rejections":0,"makespan_ms":1.0,"output_tokens_per_second":1.0,
        "family_switches":0,"family_service_order":["family-0"]}}))
    .unwrap()
}

#[test]
fn comparison_refuses_missing_duplicate_extra_or_incomplete_cells() {
    let mut cells = vec![cell(1, Version::Old), cell(1, Version::New)];
    complete(&cells, 1, 1).unwrap();
    assert!(complete(&cells, 2, 1).is_err());
    assert!(complete(&cells, 1, 0).is_err());
    assert!(complete(&cells, 1, 2).is_err());
    cells.push(cell(1, Version::Old));
    assert!(complete(&cells, 1, 1).is_err());
    cells.pop();
    cells.push(cell(2, Version::New));
    assert!(complete(&cells, 1, 1).is_err());
    cells.pop();
    cells[0].summary.successful = 0;
    assert!(complete(&cells, 1, 1).is_err());
}
