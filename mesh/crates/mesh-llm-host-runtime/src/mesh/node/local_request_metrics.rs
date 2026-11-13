//! Rolling window of requests this node accepted and completed, for the
//! local dashboard.

use super::*;

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct LocalRequestMetricsSnapshot {
    pub accepted_request_counts: Vec<u64>,
    pub latency_samples_ms: Vec<u64>,
}

#[derive(Default)]
pub(crate) struct LocalRequestMetricsSampler {
    pub(crate) inner: std::sync::Mutex<LocalRequestMetricsWindow>,
}

#[derive(Default)]
pub(crate) struct LocalRequestMetricsWindow {
    pub(crate) accepted_by_second: VecDeque<(u64, u64)>,
    pub(crate) completed_latencies_ms: VecDeque<(u64, u64)>,
}

impl LocalRequestMetricsSampler {
    pub(crate) fn record_request_accepted(&self) {
        let now_sec = now_secs();
        let mut guard = self
            .inner
            .lock()
            .expect("pretty request metrics mutex poisoned");
        guard.prune(now_sec);
        if let Some((second, count)) = guard.accepted_by_second.back_mut()
            && *second == now_sec
        {
            *count += 1;
            return;
        }
        guard.accepted_by_second.push_back((now_sec, 1));
    }

    pub(crate) fn record_request_completed(&self, started_at: std::time::Instant) {
        let now_sec = now_secs();
        let latency_ms = u64::try_from(started_at.elapsed().as_millis()).unwrap_or(u64::MAX);
        let mut guard = self
            .inner
            .lock()
            .expect("pretty request metrics mutex poisoned");
        guard.prune(now_sec);
        guard
            .completed_latencies_ms
            .push_back((now_sec, latency_ms));
    }

    pub(crate) fn snapshot(&self) -> LocalRequestMetricsSnapshot {
        let now_sec = now_secs();
        let window_start = now_sec.saturating_sub(PRETTY_LOCAL_REQUEST_WINDOW_SECS - 1);
        let mut guard = self
            .inner
            .lock()
            .expect("pretty request metrics mutex poisoned");
        guard.prune(now_sec);

        let accepted_by_second = guard
            .accepted_by_second
            .iter()
            .copied()
            .collect::<HashMap<_, _>>();
        let accepted_request_counts = (window_start..=now_sec)
            .map(|second| accepted_by_second.get(&second).copied().unwrap_or(0))
            .collect();
        let latency_samples_ms = guard
            .completed_latencies_ms
            .iter()
            .filter_map(|(second, latency_ms)| (*second >= window_start).then_some(*latency_ms))
            .collect();

        LocalRequestMetricsSnapshot {
            accepted_request_counts,
            latency_samples_ms,
        }
    }
}

impl LocalRequestMetricsWindow {
    pub(crate) fn prune(&mut self, now_sec: u64) {
        let oldest_kept_second = now_sec.saturating_sub(PRETTY_LOCAL_REQUEST_WINDOW_SECS - 1);
        while let Some((second, _)) = self.accepted_by_second.front() {
            if *second < oldest_kept_second {
                self.accepted_by_second.pop_front();
            } else {
                break;
            }
        }
        while let Some((second, _)) = self.completed_latencies_ms.front() {
            if *second < oldest_kept_second {
                self.completed_latencies_ms.pop_front();
            } else {
                break;
            }
        }
    }
}
