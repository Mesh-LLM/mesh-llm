//! Anonymous serving analytics: what real inference looks like in the field.
//!
//! Three events, all through [`mesh_llm_analytics`] and so under its consent,
//! key, and closed-vocabulary rules:
//!
//! - `inference_request`: one generation this node served, from the skippy
//!   frontend's own measurement — prefill, decode, context length, prompt
//!   cache, speculation, lanes, and the split activation wire.
//! - `inference_route`: one routing attempt this node fronted, with the link
//!   class and round-trip time when the target was a peer.
//! - `mesh_links`: an hourly summary of this node's peer links by class.
//!
//! Per-request events are rate limited per process. An event that goes out
//! after others were dropped carries `weight`, the number of requests it
//! stands for, so totals stay right when summed by weight.
//!
//! Model names are reported only when `model_loaded` reported them: an exact
//! catalog name for a model this node loaded from a catalog or repository
//! reference. Anything else, including a local file, a configured alias, or a
//! name a client typed into a request, is `redacted`.

use std::collections::HashSet;
use std::net::IpAddr;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use mesh_llm_analytics::{Event, Label, Properties};
use mesh_llm_membership::selected_path::SelectedPathObservation;
use skippy_serving::generation_stats::{GenerationStats, GenerationStatsObserver};

use crate::network::metrics::{AttemptOutcome, AttemptTarget};

/// Per-request events each of `inference_request` and `inference_route` may
/// send per minute from one process. A busy shared node serves many requests
/// a second; the distribution survives sampling, the bill does not.
const EVENTS_PER_MINUTE: u32 = 30;

/// First `mesh_links` after start, once peers have had time to connect.
const MESH_LINKS_FIRST_DELAY: Duration = Duration::from_secs(5 * 60);
const MESH_LINKS_INTERVAL: Duration = Duration::from_secs(60 * 60);

static REPORTABLE_MODELS: Mutex<Option<HashSet<String>>> = Mutex::new(None);
static REQUEST_LIMITER: RateLimiter = RateLimiter::new(EVENTS_PER_MINUTE);
static ROUTE_LIMITER: RateLimiter = RateLimiter::new(EVENTS_PER_MINUTE);

/// Start reporting `inference_request` for every generation in this process.
///
/// Installing is cheap when analytics is off: `capture` drops the event, and
/// the observer does no more than build a handful of properties first.
pub(crate) fn install() {
    let _ = skippy_serving::generation_stats::set_generation_stats_observer(Arc::new(
        AnalyticsObserver,
    ));
}

/// Allow `name` to be reported. Call only for an exact catalog name loaded
/// from a catalog or repository reference, never a local file or an alias.
pub(crate) fn allow_model_name(name: &str) {
    if let Ok(mut models) = REPORTABLE_MODELS.lock() {
        models
            .get_or_insert_with(HashSet::new)
            .insert(name.to_owned());
    }
}

fn model_label(name: &str) -> Label {
    let allowed = REPORTABLE_MODELS
        .lock()
        .ok()
        .is_some_and(|models| models.as_ref().is_some_and(|set| set.contains(name)));
    if allowed {
        Label::sanitize_or_redact(name)
    } else {
        Label::redacted()
    }
}

struct AnalyticsObserver;

impl GenerationStatsObserver for AnalyticsObserver {
    fn observe(&self, stats: &GenerationStats) {
        let Some(weight) = REQUEST_LIMITER.admit(Instant::now()) else {
            return;
        };
        mesh_llm_analytics::capture(
            Event::InferenceRequest,
            request_properties(stats, model_label(&stats.model_id), weight),
        );
    }
}

/// Throughput in tokens per second, or `None` when it would be noise.
fn tokens_per_second(tokens: u64, ms: f64) -> Option<f64> {
    (tokens > 0 && ms > 0.0).then(|| round2(tokens as f64 * 1000.0 / ms))
}

fn ratio(numerator: u64, denominator: u64) -> Option<f64> {
    (denominator > 0).then(|| round4(numerator as f64 / denominator as f64))
}

fn round2(value: f64) -> f64 {
    (value * 100.0).round() / 100.0
}

fn round4(value: f64) -> f64 {
    (value * 10_000.0).round() / 10_000.0
}

fn request_properties(stats: &GenerationStats, model: Label, weight: u64) -> Properties {
    let prompt = u64::from(stats.prompt_tokens);
    let completion = u64::from(stats.completion_tokens);
    let context = prompt + completion;
    let proposed = stats.draft_proposed_tokens + stats.mtp_proposed_tokens;
    let accepted = stats.draft_accepted_tokens + stats.mtp_accepted_tokens;
    let windows = stats.draft_windows + stats.mtp_verifications;

    let mut properties = Properties::new()
        .with("model", model)
        .with("weight", weight)
        .with("serving", if stats.split { "split" } else { "local" })
        .with("concurrency_limit", stats.concurrency_limit)
        .with("context_size", stats.context_size)
        .with("prompt_tokens", prompt)
        .with("completion_tokens", completion)
        .with("context_tokens", context)
        .with(
            "cached_prompt_tokens",
            u64::from(stats.cached_prompt_tokens),
        )
        .with("prefill_tokens", u64::from(stats.prefill_tokens))
        .with("cache_hit_kind", stats.cache_hit_kind)
        .with("queue_wait_ms", round2(stats.queue_wait_ms))
        .with("restore_ms", round2(stats.restore_ms))
        .with("prefill_ms", round2(stats.prefill_ms))
        .with("decode_ms", round2(stats.decode_ms))
        .with("total_ms", round2(stats.total_ms))
        .with(
            "ttft_ms",
            round2(stats.queue_wait_ms + stats.restore_ms + stats.prefill_ms),
        )
        .with("finish_reason", stats.finish_reason)
        .with(
            "spec_strategy",
            Label::sanitize_or_redact(&stats.speculative_strategy),
        )
        .with("spec_allowed", stats.speculation_allowed)
        .with("spec_proposed_tokens", proposed)
        .with("spec_accepted_tokens", accepted)
        .with("spec_windows", windows)
        .with("spec_draft_proposed_tokens", stats.draft_proposed_tokens)
        .with("spec_draft_accepted_tokens", stats.draft_accepted_tokens)
        .with("spec_mtp_proposed_tokens", stats.mtp_proposed_tokens)
        .with("spec_mtp_accepted_tokens", stats.mtp_accepted_tokens);

    if let Some(fill) = ratio(context, stats.context_size as u64) {
        properties = properties.with("context_fill", fill);
    }
    if let Some(rate) = tokens_per_second(u64::from(stats.prefill_tokens), stats.prefill_ms) {
        properties = properties.with("prefill_tps", rate);
    }
    if let Some(rate) = tokens_per_second(completion, stats.decode_ms) {
        properties = properties.with("decode_tps", rate);
    }
    if let Some(rate) = ratio(accepted, proposed) {
        properties = properties.with("spec_acceptance", rate);
    }
    if windows > 0 {
        // Tokens committed per verify step: the figure speculation exists
        // to raise above one.
        properties = properties.with(
            "spec_accepted_per_window",
            round2(accepted as f64 / windows as f64),
        );
    }
    if let Some(lanes) = stats.downstream_lanes {
        properties = properties.with("downstream_lanes", lanes);
    }
    if let Some(codec) = stats.activation_codec {
        properties = properties
            .with("activation_codec", Label::sanitize_or_redact(codec))
            .with("activation_prefill_bytes", stats.activation_prefill_bytes)
            .with("activation_decode_bytes", stats.activation_decode_bytes);
    }
    properties
}

/// How a peer link is carried.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum LinkClass {
    /// Direct, to a private, link-local, or loopback address.
    Lan,
    /// Direct, to a public address.
    Wan,
    /// Through a relay.
    Relay,
    /// No selected path observed yet.
    Unknown,
}

impl LinkClass {
    pub(crate) const ALL: [Self; 4] = [Self::Lan, Self::Wan, Self::Relay, Self::Unknown];

    pub(crate) const fn as_str(self) -> &'static str {
        match self {
            Self::Lan => "lan",
            Self::Wan => "wan",
            Self::Relay => "relay",
            Self::Unknown => "unknown",
        }
    }

    /// Classify from the selected path. The address decides LAN versus WAN
    /// and is not reported itself.
    pub(crate) fn of(path: Option<&SelectedPathObservation>) -> Self {
        let Some(path) = path else {
            return Self::Unknown;
        };
        if path.path_type != "direct" {
            return Self::Relay;
        }
        match path.observed_direct_remote_addr {
            Some(addr) if is_local_network(addr.ip()) => Self::Lan,
            Some(_) => Self::Wan,
            None => Self::Unknown,
        }
    }
}

fn is_local_network(ip: IpAddr) -> bool {
    match ip {
        IpAddr::V4(v4) => v4.is_private() || v4.is_link_local() || v4.is_loopback(),
        IpAddr::V6(v6) => match v6.to_ipv4_mapped() {
            Some(v4) => is_local_network(IpAddr::V4(v4)),
            None => {
                let first = v6.segments()[0];
                v6.is_loopback()
                    // fc00::/7 unique local, fe80::/10 link local.
                    || (first & 0xfe00) == 0xfc00
                    || (first & 0xffc0) == 0xfe80
            }
        },
    }
}

/// A peer's link class and the round trip that belongs to it.
///
/// A peer that fell back to a relay can still hold its last direct RTT.
/// Reporting that under `relay` would make relays look as fast as the direct
/// path they replaced, so a relay link reports only the relay path's own RTT.
/// With no selected path the only figure is a gossip-derived one that belongs
/// to no link, so none is reported.
fn link_and_rtt(
    path: Option<SelectedPathObservation>,
    direct_rtt_ms: Option<u32>,
) -> (LinkClass, Option<u32>) {
    let link = LinkClass::of(path.as_ref());
    let path_rtt = path.and_then(|path| path.rtt_ms);
    let rtt = match link {
        LinkClass::Relay => path_rtt,
        LinkClass::Lan | LinkClass::Wan => direct_rtt_ms.or(path_rtt),
        LinkClass::Unknown => None,
    };
    (link, rtt)
}

fn outcome_str(outcome: AttemptOutcome) -> &'static str {
    match outcome {
        AttemptOutcome::Success => "success",
        AttemptOutcome::Timeout => "timeout",
        AttemptOutcome::Unavailable => "unavailable",
        AttemptOutcome::ContextOverflow => "context_overflow",
        AttemptOutcome::Rejected => "rejected",
    }
}

/// One routing attempt, as `Node::record_*_attempt` sees it.
pub(crate) struct RouteAttempt<'a> {
    pub(crate) model: Option<&'a str>,
    pub(crate) target: &'a AttemptTarget,
    pub(crate) queue_wait: Duration,
    pub(crate) attempt_time: Duration,
    pub(crate) outcome: AttemptOutcome,
    pub(crate) completion_tokens: Option<u64>,
}

/// Report one routing attempt. A remote target's link is looked up off the
/// request path; the request never waits on analytics.
pub(crate) fn report_route(
    node: &crate::mesh::Node,
    remote_peer: Option<iroh::EndpointId>,
    attempt: RouteAttempt<'_>,
) {
    let Some(weight) = ROUTE_LIMITER.admit(Instant::now()) else {
        return;
    };
    let properties = route_properties(&attempt, weight);
    let Some(peer_id) = remote_peer else {
        mesh_llm_analytics::capture(Event::InferenceRoute, properties);
        return;
    };
    let Ok(runtime) = tokio::runtime::Handle::try_current() else {
        return;
    };
    let node = node.clone();
    runtime.spawn(async move {
        let peer = node
            .peers()
            .await
            .into_iter()
            .find(|peer| peer.id == peer_id);
        let (link, rtt) = link_and_rtt(
            peer.as_ref().and_then(|peer| peer.selected_path),
            peer.as_ref().and_then(|peer| peer.current_direct_rtt_ms()),
        );
        let mut properties = properties.with("link", link.as_str());
        if let Some(rtt) = rtt {
            properties = properties.with("rtt_ms", rtt);
        }
        mesh_llm_analytics::capture(Event::InferenceRoute, properties);
    });
}

fn route_properties(attempt: &RouteAttempt<'_>, weight: u64) -> Properties {
    let target = match attempt.target {
        AttemptTarget::Local(_) => "local",
        AttemptTarget::Remote(_) => "remote",
        AttemptTarget::Endpoint(_) => "endpoint",
    };
    let attempt_ms = attempt.attempt_time.as_secs_f64() * 1000.0;
    let mut properties = Properties::new()
        .with(
            "model",
            attempt.model.map_or_else(Label::redacted, model_label),
        )
        .with("weight", weight)
        .with("target", target)
        .with("outcome", outcome_str(attempt.outcome))
        .with(
            "queue_wait_ms",
            round2(attempt.queue_wait.as_secs_f64() * 1000.0),
        )
        .with("attempt_ms", round2(attempt_ms));
    if let Some(tokens) = attempt.completion_tokens {
        properties = properties.with("completion_tokens", tokens);
        if let Some(rate) = tokens_per_second(tokens, attempt_ms) {
            properties = properties.with("end_to_end_tps", rate);
        }
    }
    properties
}

/// Report `mesh_links` shortly after start and then hourly.
pub(crate) fn spawn_mesh_links_reporter(node: crate::mesh::Node) {
    tokio::spawn(async move {
        tokio::time::sleep(MESH_LINKS_FIRST_DELAY).await;
        loop {
            let links = node
                .peers()
                .await
                .iter()
                .map(|peer| link_and_rtt(peer.selected_path, peer.current_direct_rtt_ms()))
                .collect::<Vec<_>>();
            mesh_llm_analytics::capture(Event::MeshLinks, mesh_links_properties(&links));
            tokio::time::sleep(MESH_LINKS_INTERVAL).await;
        }
    });
}

fn mesh_links_properties(links: &[(LinkClass, Option<u32>)]) -> Properties {
    let mut properties = Properties::new().with("peers", links.len());
    for class in LinkClass::ALL {
        let in_class = links
            .iter()
            .filter(|(link, _)| *link == class)
            .collect::<Vec<_>>();
        properties = properties.with(count_key(class), in_class.len());
        let mut measured = in_class
            .iter()
            .filter_map(|(_, rtt)| *rtt)
            .collect::<Vec<_>>();
        if measured.is_empty() {
            continue;
        }
        measured.sort_unstable();
        let (median_key, max_key) = rtt_keys(class);
        properties = properties
            .with(median_key, measured[measured.len() / 2])
            .with(max_key, measured[measured.len() - 1]);
    }
    properties
}

const fn count_key(class: LinkClass) -> &'static str {
    match class {
        LinkClass::Lan => "lan_peers",
        LinkClass::Wan => "wan_peers",
        LinkClass::Relay => "relay_peers",
        LinkClass::Unknown => "unknown_peers",
    }
}

const fn rtt_keys(class: LinkClass) -> (&'static str, &'static str) {
    match class {
        LinkClass::Lan => ("lan_rtt_ms_median", "lan_rtt_ms_max"),
        LinkClass::Wan => ("wan_rtt_ms_median", "wan_rtt_ms_max"),
        LinkClass::Relay => ("relay_rtt_ms_median", "relay_rtt_ms_max"),
        LinkClass::Unknown => ("unknown_rtt_ms_median", "unknown_rtt_ms_max"),
    }
}

/// A fixed-window limiter that remembers how much it dropped.
struct RateLimiter {
    per_window: u32,
    state: Mutex<LimiterState>,
}

struct LimiterState {
    window_start: Option<Instant>,
    sent: u32,
    skipped: u64,
}

impl RateLimiter {
    const WINDOW: Duration = Duration::from_secs(60);

    const fn new(per_window: u32) -> Self {
        Self {
            per_window,
            state: Mutex::new(LimiterState {
                window_start: None,
                sent: 0,
                skipped: 0,
            }),
        }
    }

    /// `Some(weight)` when this request may be reported, where `weight` is
    /// itself plus every request dropped since the last one that was.
    fn admit(&self, now: Instant) -> Option<u64> {
        let mut state = self.state.lock().ok()?;
        let expired = state
            .window_start
            .is_none_or(|start| now.duration_since(start) >= Self::WINDOW);
        if expired {
            state.window_start = Some(now);
            state.sent = 0;
        }
        if state.sent >= self.per_window {
            state.skipped += 1;
            return None;
        }
        state.sent += 1;
        let weight = state.skipped + 1;
        state.skipped = 0;
        Some(weight)
    }
}

#[cfg(test)]
#[path = "inference_analytics/tests.rs"]
mod tests;
