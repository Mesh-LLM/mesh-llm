//! Numeric helpers of `collect-ci-metrics.py`: `summarize` with linear
//! `percentile`, CPython's compensated float `sum`, `round(x, n)`, and the
//! `(p95 or -1, mean or -1)` descending sort key.

use crate::ci_operations::ci_metrics_value::{Value, object};
use std::cmp::Ordering;

/// `summarize(values)`: `None` samples dropped, then sorted.
#[derive(Clone)]
pub(crate) struct Summary {
    pub(crate) count: usize,
    /// `min, mean, p50, p90, p95, max`, already rounded; absent when empty.
    pub(crate) stats: Option<[f64; 6]>,
}

impl Summary {
    pub(crate) fn mean(&self) -> Option<f64> {
        self.stats.map(|stats| stats[1])
    }

    pub(crate) fn p95(&self) -> Option<f64> {
        self.stats.map(|stats| stats[4])
    }

    pub(crate) fn to_value(&self) -> Value {
        let field = |index: usize| Value::opt_float(self.stats.map(|stats| stats[index]));
        object([
            ("count", Value::count(self.count)),
            ("min", field(0)),
            ("mean", field(1)),
            ("p50", field(2)),
            ("p90", field(3)),
            ("p95", field(4)),
            ("max", field(5)),
        ])
    }

    /// `(p95 or -1, mean or -1)`: zero and `None` both become -1.
    pub(crate) fn sort_key(&self) -> (f64, f64) {
        let truthy = |value: Option<f64>| value.filter(|number| *number != 0.0).unwrap_or(-1.0);
        (truthy(self.p95()), truthy(self.mean()))
    }
}

/// Python's `sorted(..., key=sort_key, reverse=True)`: stable descending.
pub(crate) fn sort_descending<T>(items: &mut [T], key: impl Fn(&T) -> (f64, f64)) {
    items.sort_by(|a, b| {
        let (a, b) = (key(a), key(b));
        (b.0, b.1)
            .partial_cmp(&(a.0, a.1))
            .unwrap_or(Ordering::Equal)
    });
}

/// `sorted` of floats (stable; `-0.0 == 0.0`).
pub(crate) fn sorted(mut values: Vec<f64>) -> Vec<f64> {
    values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
    values
}

/// CPython 3.13 `sum()` over floats (Neumaier compensation).
pub(crate) fn float_sum(values: impl IntoIterator<Item = f64>) -> f64 {
    let mut total = 0.0_f64;
    let mut compensation = 0.0_f64;
    for value in values {
        let next = total + value;
        if total.abs() >= value.abs() {
            compensation += (total - next) + value;
        } else {
            compensation += (value - next) + total;
        }
        total = next;
    }
    if compensation != 0.0 && compensation.is_finite() {
        total += compensation;
    }
    total
}

/// Python `round(value, digits)`: correctly rounded, ties to even.
pub(crate) fn round(value: f64, digits: usize) -> f64 {
    if !value.is_finite() {
        return value;
    }
    format!("{value:.digits$}").parse().unwrap_or(value)
}

fn percentile(values: &[f64], quantile: f64) -> f64 {
    let position = (values.len() - 1) as f64 * quantile;
    let (low, high) = (position.floor(), position.ceil());
    let value = |index: f64| values[index as usize];
    if low == high {
        return value(low);
    }
    value(low) * (high - position) + value(high) * (position - low)
}

pub(crate) fn summarize(values: impl IntoIterator<Item = Option<f64>>) -> Summary {
    let samples = sorted(values.into_iter().flatten().collect());
    if samples.is_empty() {
        return Summary {
            count: 0,
            stats: None,
        };
    }
    let mean = float_sum(samples.iter().copied()) / samples.len() as f64;
    let stats = [
        samples[0],
        mean,
        percentile(&samples, 0.50),
        percentile(&samples, 0.90),
        percentile(&samples, 0.95),
        samples[samples.len() - 1],
    ]
    .map(|value| round(value, 3));
    Summary {
        count: samples.len(),
        stats: Some(stats),
    }
}

/// A `collections.Counter` of strings in first-increment order.
#[derive(Default)]
pub(crate) struct Counter(pub(crate) Vec<(String, usize)>);

impl Counter {
    pub(crate) fn add(&mut self, key: &str) {
        match self.0.iter_mut().find(|(seen, _)| seen == key) {
            Some(entry) => entry.1 += 1,
            None => self.0.push((key.to_owned(), 1)),
        }
    }

    pub(crate) fn get(&self, key: &str) -> usize {
        self.0
            .iter()
            .find(|(seen, _)| seen == key)
            .map_or(0, |(_, count)| *count)
    }

    /// `dict(sorted(counter.items()))`.
    pub(crate) fn sorted_value(&self) -> Value {
        let mut entries = self.0.clone();
        entries.sort_by(|(a, _), (b, _)| a.cmp(b));
        Value::Object(
            entries
                .into_iter()
                .map(|(key, count)| (key, Value::count(count)))
                .collect(),
        )
    }

    /// `counter.most_common(limit)`: count descending, ties by insertion.
    pub(crate) fn most_common(&self, limit: usize) -> Vec<(String, usize)> {
        let mut entries = self.0.clone();
        entries.sort_by(|(_, a), (_, b)| b.cmp(a));
        entries.truncate(limit);
        entries
    }
}

#[cfg(test)]
mod tests {
    use super::{float_sum, round, summarize};

    #[test]
    fn migration_ci_operations_metrics_numbers_follow_cpython() {
        let summary = summarize([10.0, 20.0, 30.0, 40.0].map(Some));
        assert_eq!(summary.stats.map(|stats| stats[2]), Some(25.0));
        assert_eq!(summary.p95(), Some(38.5));
        assert_eq!(round(0.0625, 3), 0.062);
        assert_eq!(round(1234.5675, 3), 1234.568);
        assert_eq!(float_sum([0.1, 0.2, 0.3]), 0.6);
        assert_eq!(summarize([None]).count, 0);
    }
}
