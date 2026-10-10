use super::super::statistics::Status;
use super::*;

#[test]
fn stream_matches_independent_sha256_known_vector() {
    let mut stream = Stream::new(42, "__primary__", "decode_tok_s").unwrap();
    let first = (0..4).map(|_| stream.next().unwrap()).collect::<Vec<_>>();
    assert_eq!(
        first,
        [
            0xe487_5daa_f61d_ffcc,
            0xd66f_8ef6_e4e7_1dad,
            0x2ef6_b0d5_5c77_7f9c,
            0x9187_53d0_24ee_8de7
        ]
    );
}

#[test]
fn root_group_metric_and_length_boundaries_select_distinct_streams() {
    let first = |root, group, metric| Stream::new(root, group, metric).unwrap().next().unwrap();
    let base = first(42, "ab", "c");
    for other in [
        first(43, "ab", "c"),
        first(42, "a", "bc"),
        first(42, "ab", "d"),
        first(42, "é", "c"),
    ] {
        assert_ne!(base, other);
    }
    assert_eq!(base, first(42, "ab", "c"));
}

#[test]
fn bootstrap_is_reproducible_for_complete_ten_thousand_resamples() {
    let values = [-0.02, 0.01, 0.03, 0.04, 0.02];
    let run = |root| {
        serde_json::to_value(bootstrap(&values, 10_000, root, "case", "ttft_ms").unwrap()).unwrap()
    };
    assert_eq!(run(42), run(42));
    assert!(run(43)["ci_low"].as_f64().unwrap().is_finite());
    let result = bootstrap(&values, 10_000, 42, "case", "ttft_ms").unwrap();
    assert!((result.mean_relative_degradation - 0.016).abs() < 1e-12);
    assert!(result.ci_low <= result.ci_high);
}

#[test]
fn constant_paired_threshold_preserves_zero_variance_without_roundoff_failure() {
    let result = bootstrap(&[0.03; 20], 10_000, 42, "__primary__", "ttft_ms").unwrap();
    assert_eq!((result.ci_low, result.ci_high), (0.03, 0.03));
    assert_eq!(result.minimal_detectable_degradation, 0.0);
    assert_eq!(result.status(0.03, 0.03).unwrap(), Status::Pass);
    let singleton = bootstrap(&[0.05], 10_000, 42, "case", "ttft_ms").unwrap();
    assert_eq!(singleton.status(0.03, 0.03).unwrap(), Status::Fail);
    // A singleton is valid numerical input, but cannot satisfy the CLI's pair minimum.
}

#[test]
fn invalid_population_resamples_and_work_budgets_refuse_before_allocation() {
    for values in [vec![], vec![f64::NAN], vec![f64::INFINITY]] {
        assert!(bootstrap(&values, 10_000, 42, "case", "ttft_ms").is_err());
    }
    for resamples in [0, 1, MAX_RESAMPLES + 1, usize::MAX] {
        assert!(bootstrap(&[0.01, 0.02], resamples, 42, "case", "ttft_ms").is_err());
    }
    assert!(admit(&vec![0.01; 101], MAX_RESAMPLES).is_err());
    assert!(bootstrap(&[f64::MAX, -f64::MAX], 2, 42, "case", "ttft_ms").is_err());
}
