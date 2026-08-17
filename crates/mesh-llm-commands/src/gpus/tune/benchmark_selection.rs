use super::*;

pub(crate) struct BenchmarkSelection {
    pub(crate) recommended: Option<TuneBenchmarkTrial>,
    pub(crate) raw_best: Option<TuneBenchmarkTrial>,
    pub(crate) pareto_frontier: Vec<TuneBenchmarkTrial>,
    pub(crate) reason: Option<String>,
}

pub(crate) fn select_benchmark_trials(
    trials: &[TuneBenchmarkTrial],
    throughput_tolerance_pct: f64,
) -> BenchmarkSelection {
    let successes = successful_trials(trials);
    let Some(raw_best_ref) = successes
        .iter()
        .copied()
        .max_by(|left, right| compare_raw_best(left, right))
    else {
        return BenchmarkSelection {
            recommended: None,
            raw_best: None,
            pareto_frontier: Vec::new(),
            reason: None,
        };
    };
    let recommended = select_recommended_trial(&successes, raw_best_ref, throughput_tolerance_pct);
    let reason = recommended
        .as_ref()
        .map(|trial| selection_reason(trial, raw_best_ref, throughput_tolerance_pct));

    BenchmarkSelection {
        recommended,
        raw_best: Some(raw_best_ref.clone()),
        pareto_frontier: pareto_frontier(&successes),
        reason,
    }
}

fn successful_trials(trials: &[TuneBenchmarkTrial]) -> Vec<&TuneBenchmarkTrial> {
    trials
        .iter()
        .filter(|trial| matches!(trial.status, TuneBenchmarkTrialStatus::Succeeded))
        .filter(|trial| trial.decode_tok_s.is_some())
        .collect()
}

fn select_recommended_trial(
    successes: &[&TuneBenchmarkTrial],
    raw_best: &TuneBenchmarkTrial,
    throughput_tolerance_pct: f64,
) -> Option<TuneBenchmarkTrial> {
    let threshold = throughput_threshold(raw_best.decode_tok_s?, throughput_tolerance_pct);
    successes
        .iter()
        .copied()
        .filter(|trial| trial.decode_tok_s.is_some_and(|rate| rate >= threshold))
        .max_by(|left, right| compare_recommendation(left, right))
        .cloned()
}

fn throughput_threshold(raw_best: f64, throughput_tolerance_pct: f64) -> f64 {
    let tolerated_fraction = (throughput_tolerance_pct / 100.0).clamp(0.0, 1.0);
    raw_best * (1.0 - tolerated_fraction)
}

fn pareto_frontier(successes: &[&TuneBenchmarkTrial]) -> Vec<TuneBenchmarkTrial> {
    let mut frontier = successes
        .iter()
        .copied()
        .filter(|candidate| {
            !successes.iter().copied().any(|other| {
                !std::ptr::eq(*candidate, other) && dominates_for_frontier(other, candidate)
            })
        })
        .cloned()
        .collect::<Vec<_>>();
    frontier.sort_by(|left, right| compare_frontier_order(left, right).reverse());
    frontier
}

fn dominates_for_frontier(left: &TuneBenchmarkTrial, right: &TuneBenchmarkTrial) -> bool {
    let Some(left_rate) = left.decode_tok_s else {
        return false;
    };
    let Some(right_rate) = right.decode_tok_s else {
        return false;
    };
    let left_ctx = left.candidate.ctx_size;
    let right_ctx = right.candidate.ctx_size;
    left_rate >= right_rate
        && left_ctx >= right_ctx
        && (left_rate > right_rate || left_ctx > right_ctx)
}

fn compare_raw_best(left: &TuneBenchmarkTrial, right: &TuneBenchmarkTrial) -> std::cmp::Ordering {
    compare_decode_tok_s(left, right)
        .then_with(|| left.candidate.ctx_size.cmp(&right.candidate.ctx_size))
        .then_with(|| compare_lower_optional_f64(request_ms(left), request_ms(right)))
        .then_with(|| compare_lower_optional_f64(readiness_ms(left), readiness_ms(right)))
}

fn compare_recommendation(
    left: &TuneBenchmarkTrial,
    right: &TuneBenchmarkTrial,
) -> std::cmp::Ordering {
    left.candidate
        .ctx_size
        .cmp(&right.candidate.ctx_size)
        .then_with(|| compare_decode_tok_s(left, right))
        .then_with(|| compare_lower_optional_f64(request_ms(left), request_ms(right)))
        .then_with(|| compare_lower_optional_f64(readiness_ms(left), readiness_ms(right)))
        .then_with(|| compare_lower_optional_f64(total_ms(left), total_ms(right)))
        .then_with(|| (!left.candidate.mlock).cmp(&(!right.candidate.mlock)))
        .then_with(|| {
            mmap_preference(left.candidate.mmap).cmp(&mmap_preference(right.candidate.mmap))
        })
}

fn compare_frontier_order(
    left: &TuneBenchmarkTrial,
    right: &TuneBenchmarkTrial,
) -> std::cmp::Ordering {
    left.candidate
        .ctx_size
        .cmp(&right.candidate.ctx_size)
        .then_with(|| compare_decode_tok_s(left, right))
}

fn compare_decode_tok_s(
    left: &TuneBenchmarkTrial,
    right: &TuneBenchmarkTrial,
) -> std::cmp::Ordering {
    left.decode_tok_s
        .unwrap_or(f64::NEG_INFINITY)
        .partial_cmp(&right.decode_tok_s.unwrap_or(f64::NEG_INFINITY))
        .unwrap_or(std::cmp::Ordering::Equal)
}

fn compare_lower_optional_f64(left: Option<f64>, right: Option<f64>) -> std::cmp::Ordering {
    match (left, right) {
        (Some(left), Some(right)) => right
            .partial_cmp(&left)
            .unwrap_or(std::cmp::Ordering::Equal),
        (Some(_), None) => std::cmp::Ordering::Greater,
        (None, Some(_)) => std::cmp::Ordering::Less,
        (None, None) => std::cmp::Ordering::Equal,
    }
}

fn request_ms(trial: &TuneBenchmarkTrial) -> Option<f64> {
    trial
        .timings
        .as_ref()
        .and_then(|timings| timings.request_ms)
}

fn readiness_ms(trial: &TuneBenchmarkTrial) -> Option<f64> {
    trial.timings.as_ref().map(|timings| timings.readiness_ms)
}

fn total_ms(trial: &TuneBenchmarkTrial) -> Option<f64> {
    trial.timings.as_ref().map(|timings| timings.total_ms)
}

fn mmap_preference(value: TuneBoolOrAutoValue) -> u8 {
    match value {
        TuneBoolOrAutoValue::Auto => 2,
        TuneBoolOrAutoValue::Disabled => 1,
        TuneBoolOrAutoValue::Enabled => 0,
    }
}

fn selection_reason(
    recommended: &TuneBenchmarkTrial,
    raw_best: &TuneBenchmarkTrial,
    throughput_tolerance_pct: f64,
) -> String {
    let recommended_rate = recommended.decode_tok_s.unwrap_or_default();
    let raw_rate = raw_best.decode_tok_s.unwrap_or_default();
    if recommended.candidate == raw_best.candidate {
        return format!("selected raw throughput winner at {recommended_rate:.2} tok/s");
    }
    let threshold = throughput_threshold(raw_rate, throughput_tolerance_pct);
    format!(
        "selected largest ctx_size within {:.2}% of raw best throughput \
         (recommended {:.2} tok/s, raw best {:.2} tok/s, minimum {:.2} tok/s)",
        throughput_tolerance_pct, recommended_rate, raw_rate, threshold,
    )
}
