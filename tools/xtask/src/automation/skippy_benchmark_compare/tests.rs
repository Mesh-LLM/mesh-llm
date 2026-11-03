use super::*;
use serde_json::{Value, json};

fn sample(name: &str, time: f64, unit: &str) -> Value {
    json!({"name": name, "real_time": time, "time_unit": unit})
}

fn report(samples: Vec<Value>) -> Vec<u8> {
    serde_json::to_vec(&json!({"benchmarks": samples})).unwrap()
}

#[test]
fn command_dispatch_preserves_the_report_path() {
    let args = vec![
        "automation".into(),
        "skippy-benchmark-compare".into(),
        "report with spaces.json".into(),
    ];
    let parsed = crate::cli::Cli::parse(&args).unwrap();
    match parsed.command {
        crate::cli::CliCommand::SkippyBenchmarkCompare(rest) => {
            assert_eq!(rest, &["report with spaces.json"]);
        }
        _ => panic!("comparison command was not selected"),
    }
}

#[test]
fn correctness_rejections_on_both_sides_fail_instead_of_skipping() {
    let bytes = report(vec![
        json!({"name": "Embed/FFI/128", "error_occurred": true, "error_message": "wrong payload"}),
        json!({"name": "Embed/API/128", "error_occurred": true, "error_message": "wrong payload"}),
    ]);
    let error = compare(&bytes).unwrap_err().to_string();
    assert!(error.contains("wrong payload"));
    assert!(error.contains("rejected"));
}

#[test]
fn one_correctness_rejection_invalidates_otherwise_valid_measurements() {
    for side in ["FFI", "API"] {
        let bytes = report(vec![
            sample("Embed/FFI/128", 10.0, "ns"),
            sample("Embed/API/128", 5.0, "ns"),
            json!({"name": format!("Embed/{side}/128"), "error_occurred": true}),
        ]);
        assert!(compare(&bytes).is_err());
    }
}

#[test]
fn a_missing_counterpart_is_an_error_even_beside_a_complete_pair() {
    for side in ["FFI", "API"] {
        let bytes = report(vec![
            sample("Normalize/FFI", 10.0, "ns"),
            sample("Normalize/API", 5.0, "ns"),
            sample(&format!("Embed/{side}/128"), 5.0, "ns"),
        ]);
        assert!(
            compare(&bytes)
                .unwrap_err()
                .to_string()
                .contains("counterpart")
        );
    }
}

#[test]
fn medians_use_repetitions_and_convert_all_supported_units() {
    let bytes = report(vec![
        sample("Embed/FFI/128/real_time", 0.003, "ms"),
        sample("Embed/FFI/128/repeats:3", 1.0, "us"),
        sample("Embed/FFI/128/iterations:1", 2_000.0, "ns"),
        sample("Embed/API/128/cpu_time", 0.000001, "s"),
        // Aggregates must not count as extra repetitions.
        json!({"name": "Embed/FFI/128_mean", "run_type": "aggregate"}),
    ]);
    let text = compare(&bytes).unwrap();
    assert!(text.contains("2.000 us"));
    assert!(text.contains("1.000 us"));
    assert!(text.contains("2.000x new faster (+100.0%)"));
    assert!(text.contains("geomean"));
}

#[test]
fn even_sample_median_and_geomean_are_reported() {
    let bytes = report(vec![
        sample("Normalize/FFI", 2.0, "ns"),
        sample("Normalize/FFI", 6.0, "ns"),
        sample("Normalize/API", 2.0, "ns"),
        sample("BatchBuild/FFI", 8.0, "ns"),
        sample("BatchBuild/API", 1.0, "ns"),
    ]);
    let text = compare(&bytes).unwrap();
    assert!(text.contains("4.0 ns"));
    assert!(
        text.lines()
            .find(|line| line.starts_with("geomean"))
            .unwrap()
            .contains("4.000x")
    );
}

#[test]
fn minimum_aggregates_are_reported_separately_from_repetition_medians() {
    let bytes = report(vec![
        sample("Embed/FFI/128/real_time", 10.0, "ns"),
        sample("Embed/API/128/real_time", 5.0, "ns"),
        json!({"name":"Embed/FFI/128_min", "run_type":"aggregate", "aggregate_name":"min", "real_time":8.0, "time_unit":"ns"}),
        json!({"name":"Embed/API/128_min", "run_type":"aggregate", "aggregate_name":"min", "real_time":2.0, "time_unit":"ns"}),
    ]);
    let text = compare(&bytes).unwrap();
    let median = text
        .lines()
        .find(|line| line.starts_with("Embed/128 "))
        .unwrap();
    let minimum = text
        .lines()
        .find(|line| line.starts_with("Embed/128_min "))
        .unwrap();
    assert!(median.contains("2.000x"));
    assert!(minimum.contains("4.000x"));
}

#[test]
fn repetition_minima_are_derived_without_an_explicit_aggregate() {
    let text = compare(&report(vec![
        sample("Normalize/FFI/real_time", 100.0, "ns"),
        sample("Normalize/FFI/real_time", 4.0, "ns"),
        sample("Normalize/API/real_time", 10.0, "ns"),
        sample("Normalize/API/real_time", 2.0, "ns"),
    ]))
    .unwrap();
    let median = text
        .lines()
        .find(|line| line.starts_with("Normalize "))
        .unwrap();
    let minimum = text
        .lines()
        .find(|line| line.starts_with("Normalize_min "))
        .unwrap();
    assert!(median.contains("52.0 ns") && median.contains("6.0 ns"));
    assert!(median.contains("8.667x"));
    assert!(minimum.contains("4.0 ns") && minimum.contains("2.0 ns"));
    assert!(minimum.contains("2.000x"));
}

#[test]
fn explicit_minima_take_precedence_without_duplicate_derived_rows() {
    let text = compare(&report(vec![
        sample("Normalize/FFI", 100.0, "ns"),
        sample("Normalize/FFI", 4.0, "ns"),
        sample("Normalize/API", 10.0, "ns"),
        sample("Normalize/API", 2.0, "ns"),
        json!({"name":"Normalize/FFI_min", "run_type":"aggregate", "aggregate_name":"min", "real_time":1.0, "time_unit":"ns"}),
        json!({"name":"Normalize/API_min", "run_type":"aggregate", "aggregate_name":"min", "real_time":0.25, "time_unit":"ns"}),
    ]))
    .unwrap();
    let minima = text
        .lines()
        .filter(|line| line.starts_with("Normalize_min "))
        .collect::<Vec<_>>();
    assert_eq!(minima.len(), 1);
    assert!(minima[0].contains("4.000x"));
}

#[test]
fn derived_and_explicit_minima_do_not_change_the_median_geomean() {
    let samples = vec![
        sample("Normalize/FFI", 100.0, "ns"),
        sample("Normalize/FFI", 4.0, "ns"),
        sample("Normalize/API", 10.0, "ns"),
        sample("Normalize/API", 2.0, "ns"),
        sample("BatchBuild/FFI", 8.0, "ns"),
        sample("BatchBuild/API", 4.0, "ns"),
    ];
    let derived = compare(&report(samples.clone())).unwrap();
    let mut explicit_samples = samples;
    explicit_samples.extend([
        json!({"name":"Normalize/FFI_min", "run_type":"aggregate", "aggregate_name":"min", "real_time":1.0, "time_unit":"ns"}),
        json!({"name":"Normalize/API_min", "run_type":"aggregate", "aggregate_name":"min", "real_time":0.25, "time_unit":"ns"}),
    ]);
    let explicit = compare(&report(explicit_samples)).unwrap();
    let derived_geomean = derived
        .lines()
        .find(|line| line.starts_with("geomean"))
        .unwrap();
    let explicit_geomean = explicit
        .lines()
        .find(|line| line.starts_with("geomean"))
        .unwrap();
    assert!(derived_geomean.contains("4.163x"));
    assert_eq!(derived_geomean, explicit_geomean);
}

#[test]
fn a_single_sample_pair_does_not_invent_a_repetition_minimum() {
    let text = compare(&report(vec![
        sample("Normalize/FFI", 10.0, "ns"),
        sample("Normalize/API", 2.0, "ns"),
    ]))
    .unwrap();
    assert!(!text.contains("Normalize_min"));
}

#[test]
fn an_explicit_minimum_missing_one_arm_is_not_silently_filled_from_repetitions() {
    let bytes = report(vec![
        sample("Normalize/FFI", 100.0, "ns"),
        sample("Normalize/FFI", 4.0, "ns"),
        sample("Normalize/API", 10.0, "ns"),
        sample("Normalize/API", 2.0, "ns"),
        json!({"name":"Normalize/FFI_min", "run_type":"aggregate", "aggregate_name":"min", "real_time":1.0, "time_unit":"ns"}),
    ]);
    assert!(
        compare(&bytes)
            .unwrap_err()
            .to_string()
            .contains("counterpart")
    );
}

#[test]
fn invalid_derived_minimum_ratios_fail_even_when_medians_are_valid() {
    for (ffi, api) in [
        ([1e-300, 1e100], [1e100, 1e100]),
        ([1e100, 1e100], [1e-300, 1e100]),
    ] {
        let bytes = report(vec![
            sample("Normalize/FFI", ffi[0], "ns"),
            sample("Normalize/FFI", ffi[1], "ns"),
            sample("Normalize/API", api[0], "ns"),
            sample("Normalize/API", api[1], "ns"),
        ]);
        assert!(
            compare(&bytes)
                .unwrap_err()
                .to_string()
                .contains("invalid timing ratio")
        );
    }
}

#[test]
fn minimum_only_reports_support_names_without_arguments() {
    let bytes = report(vec![
        json!({"name":"Normalize/FFI_min", "run_type":"aggregate", "aggregate_name":"min", "real_time":8.0, "time_unit":"ns"}),
        json!({"name":"Normalize/API_min", "run_type":"aggregate", "aggregate_name":"min", "real_time":2.0, "time_unit":"ns"}),
    ]);
    assert!(compare(&bytes).unwrap().contains("Normalize_min"));
}

#[test]
fn minimum_correctness_rejections_cannot_be_hidden_by_valid_repetitions() {
    assert!(compare(&report(vec![
        sample("Normalize/FFI", 10.0, "ns"),
        sample("Normalize/API", 5.0, "ns"),
        json!({"name":"Normalize/API_min", "error_occurred":true,"run_type":"aggregate","aggregate_name":"min"}),
    ])).is_err());
}

#[test]
fn numeric_arguments_sort_numerically_within_the_family() {
    let bytes = report(
        [128, 2, 16]
            .into_iter()
            .flat_map(|argument| {
                [
                    sample(&format!("Embed/FFI/{argument}"), 10.0, "ns"),
                    sample(&format!("Embed/API/{argument}"), 5.0, "ns"),
                ]
            })
            .collect(),
    );
    let text = compare(&bytes).unwrap();
    assert!(text.find("Embed/2 ").unwrap() < text.find("Embed/16 ").unwrap());
    assert!(text.find("Embed/16 ").unwrap() < text.find("Embed/128 ").unwrap());
}

#[test]
fn malformed_and_empty_reports_are_rejected() {
    for bytes in [
        b"".as_slice(),
        b"{}",
        b"{\"benchmarks\":[]}",
        b"{\"benchmarks\":null}",
    ] {
        assert!(compare(bytes).is_err());
    }
    assert!(compare(&report(vec![sample("Unpaired", 1.0, "ns")])).is_err());
    assert!(
        compare(&report(vec![
            json!({"name":"Embed/FFI_mean", "run_type":"aggregate"})
        ]))
        .is_err()
    );
}

#[test]
fn invalid_timing_samples_fail_admission() {
    for bad in [
        sample("Embed/FFI", 0.0, "ns"),
        sample("Embed/FFI", -1.0, "us"),
        sample("Embed/FFI", 1.0, "minutes"),
        sample("Embed/FFI", f64::MAX, "s"),
        json!({"name":"Embed/FFI", "real_time":null, "time_unit":"ns"}),
        json!({"name":"Embed/FFI", "real_time":"NaN", "time_unit":"ns"}),
    ] {
        assert!(compare(&report(vec![bad, sample("Embed/API", 1.0, "ns")])).is_err());
    }
}

#[test]
fn error_rows_are_fatal_even_if_they_are_aggregates_or_unpaired() {
    for name in ["Embed/API_mean", "Unpaired"] {
        assert!(
            compare(&report(vec![
                sample("Normalize/FFI", 10.0, "ns"),
                sample("Normalize/API", 5.0, "ns"),
                json!({"name":name,"error_occurred":true,"run_type":"aggregate"}),
            ]))
            .is_err()
        );
    }
}
