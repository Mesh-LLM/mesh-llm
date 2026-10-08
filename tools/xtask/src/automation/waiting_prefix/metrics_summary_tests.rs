use super::*;

fn timing(id: &str, ttft: f64) -> Timing {
    Timing {
        request_id: id.into(),
        server_ttft_ms: ttft,
        server_request_latency_ms: ttft + 10.0,
    }
}

#[test]
fn summarizes_all_measured_requests_with_cell_local_request_ids() {
    let summary = summarize(
        Version::Old,
        &[
            vec![timing("1", 1.0), timing("2", 3.0)],
            vec![timing("1", 100.0)],
        ],
    )
    .unwrap();
    assert_eq!(summary.measured_requests, 3);
    assert_eq!(summary.server_ttft_ms_p50, 3.0);
    assert_eq!(summary.server_ttft_ms_p95, 100.0);
    assert_eq!(summary.server_request_latency_ms_p50, 13.0);
}

#[test]
fn refuses_missing_duplicate_and_invalid_measurements() {
    assert!(summarize(Version::Old, &[]).is_err());
    assert!(summarize(Version::Old, &[vec![]]).is_err());
    assert!(summarize(Version::Old, &[vec![timing("1", 1.0), timing("1", 2.0)]]).is_err());
    for value in [f64::NAN, f64::INFINITY, -1.0] {
        assert!(summarize(Version::Old, &[vec![timing("1", value)]]).is_err());
    }
    let mut invalid = timing("1", 3.0);
    invalid.server_request_latency_ms = 2.0;
    assert!(summarize(Version::Old, &[vec![invalid]]).is_err());
}

#[test]
fn report_labels_server_boundary_and_refuses_incomplete_pairs() {
    let old = summarize(Version::Old, &[vec![timing("1", 1.0)]]).unwrap();
    let new = summarize(Version::New, &[vec![timing("1", 2.0)]]).unwrap();
    assert!(render(&[]).is_err());
    let text = render(&[old, new]).unwrap();
    assert!(text.contains("Server TTFT p50 ms | 1.0 | 2.0"));
    assert!(text.contains("excluding cache seeding"));
    assert!(text.contains("Hardware acceptance uses the client measurements"));
}
