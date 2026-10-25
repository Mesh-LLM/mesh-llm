use super::*;

#[test]
fn fixed_complete_degradation_fails_before_power_or_multiplicity_wording() {
    let evidence = summarize(0.05, &vec![0.05; 10_000]).unwrap();
    assert_eq!(evidence.ci_low, 0.05);
    assert_eq!(evidence.ci_high, 0.05);
    assert_eq!(evidence.minimal_detectable_degradation, 0.0);
    assert_eq!(evidence.raw_p_value, 0.0);
    assert_eq!(evidence.status(0.03, 0.03).unwrap(), Status::Fail);
    let _wording = holm(&[evidence.raw_p_value, 0.5, 1.0]).unwrap();
    assert_eq!(evidence.status(0.03, 0.03).unwrap(), Status::Fail);
}

#[test]
fn exact_threshold_and_neutral_measurements_do_not_claim_statistical_proof() {
    let threshold = summarize(0.03, &vec![0.03; 10_000]).unwrap();
    assert_eq!(threshold.status(0.03, 0.0).unwrap(), Status::Pass);
    let neutral = summarize(0.0, &vec![0.0; 10_000]).unwrap();
    assert_eq!(neutral.raw_p_value, 1.0);
    assert_eq!(neutral.status(0.03, 0.03).unwrap(), Status::Pass);
}

#[test]
fn wide_uncertainty_blocks_even_when_mean_is_neutral() {
    let evidence = summarize(0.0, &[-1.0, 1.0]).unwrap();
    assert_eq!((evidence.ci_low, evidence.ci_high), (-1.0, 1.0));
    assert!(
        (evidence.minimal_detectable_degradation - POWER_QUANTILES * 2.0_f64.sqrt()).abs() < 1e-12
    );
    assert_eq!(evidence.status(0.03, 0.03).unwrap(), Status::Underpowered);
}

#[test]
fn resample_percentile_indices_use_even_ties_and_sample_variance() {
    let values = (0..21).map(f64::from).collect::<Vec<_>>();
    let evidence = summarize(10.0, &values).unwrap();
    assert_eq!((evidence.ci_low, evidence.ci_high), (0.0, 20.0));
    assert!((standard_error(&values).unwrap() - 38.5_f64.sqrt()).abs() < 1e-12);
    assert!((evidence.raw_p_value - 2.0 / 21.0).abs() < 1e-12);
}

#[test]
fn incomplete_nonfinite_and_overflowing_evidence_is_refused() {
    for values in [
        vec![],
        vec![0.0],
        vec![0.0, f64::NAN],
        vec![f64::NEG_INFINITY, 0.0],
        vec![f64::MAX, -f64::MAX],
    ] {
        assert!(summarize(0.0, &values).is_err());
    }
    assert!(summarize(f64::INFINITY, &[0.0, 0.0]).is_err());
    let evidence = summarize(0.0, &[0.0, 0.0]).unwrap();
    for limit in [f64::NAN, f64::INFINITY, -0.01] {
        assert!(evidence.status(limit, 0.03).is_err());
        assert!(evidence.status(0.03, limit).is_err());
    }
}

#[test]
fn holm_preserves_original_order_and_monotonic_rank_without_relaxing_probability() {
    let inputs = [0.5, 0.01, 0.02, 0.9];
    assert_eq!(holm(&inputs).unwrap(), [1.0, 0.04, 0.06, 1.0]);
    assert_eq!(holm(&[0.0, 0.0, 1.0]).unwrap(), [0.0, 0.0, 1.0]);
    assert_eq!(holm(&[]).unwrap(), Vec::<f64>::new());
    for input in [f64::NAN, f64::INFINITY, -0.01, 1.01] {
        assert!(holm(&[input]).is_err());
    }
}
