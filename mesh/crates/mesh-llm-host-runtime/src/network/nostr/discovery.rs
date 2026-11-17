//! Nostr mesh discovery.

#[cfg(test)]
use super::contracts::DEFAULT_RELAYS;
use super::contracts::{DiscoveredMesh, MESH_SERVICE_KIND, MeshListing};
#[cfg(test)]
use super::publish::Publisher;
use anyhow::Result;
use nostr_sdk::prelude::*;
use std::time::Duration;

// ---------------------------------------------------------------------------
// Discovery — find meshes on Nostr
// ---------------------------------------------------------------------------

/// Criteria for filtering discovered meshes.
#[derive(Debug, Clone, Default)]
pub struct MeshFilter {
    /// Match meshes by name (case-insensitive exact match)
    pub name: Option<String>,
    /// Match meshes serving (or wanting) this model name (substring match)
    pub model: Option<String>,
    /// Minimum total VRAM in GB
    pub min_vram_gb: Option<f64>,
    /// Geographic region
    pub region: Option<String>,
}

impl MeshFilter {
    pub fn matches(&self, mesh: &DiscoveredMesh) -> bool {
        if let Some(ref name) = self.name {
            match &mesh.listing.name {
                Some(n) if n.eq_ignore_ascii_case(name) => {}
                _ => return false,
            }
        }
        if let Some(ref model) = self.model {
            let model_lower = model.to_lowercase();
            let has_model = mesh
                .listing
                .serving
                .iter()
                .any(|m| m.to_lowercase().contains(&model_lower))
                || mesh
                    .listing
                    .wanted
                    .iter()
                    .any(|m| m.to_lowercase().contains(&model_lower))
                || mesh
                    .listing
                    .on_disk
                    .iter()
                    .any(|m| m.to_lowercase().contains(&model_lower));
            if !has_model {
                return false;
            }
        }
        if let Some(min_gb) = self.min_vram_gb {
            let vram_gb = mesh.listing.total_vram_bytes as f64 / 1e9;
            if vram_gb < min_gb {
                return false;
            }
        }
        if let Some(ref region) = self.region {
            match &mesh.listing.region {
                Some(r) if r.eq_ignore_ascii_case(region) => {}
                _ => return false,
            }
        }
        true
    }
}

/// How many relays a discovery round actually heard back from.
///
/// `fetch_events` swallows per-relay errors, so "zero meshes" alone
/// cannot distinguish "the mesh plane is empty" from "no relay
/// answered" — callers that act on absence (the publish watchdog) or
/// report to users (the discover CLI) need this alongside the results.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct RelayHealth {
    /// Relays the discovery client tried to query.
    pub relays_queried: usize,
    /// Relays that completed the round trip — an EOSE, possibly empty.
    /// Silence-until-deadline and never-issued requests do not count.
    pub relays_answered: usize,
}

impl RelayHealth {
    /// False only when no relay answered at all — the "cannot tell"
    /// state for absence-based decisions.
    pub fn any_relay_answered(&self) -> bool {
        self.relays_answered > 0
    }
}

/// One relay's raw fetch outcome: whether its connection was live when
/// the attempt started, and the deadline-wrapped per-relay fetch result.
type RelayFetchOutcome = (
    bool,
    Result<
        Result<std::collections::BTreeSet<Event>, nostr_sdk::prelude::Error>,
        tokio::time::error::Elapsed,
    >,
);

/// One relay's fetch outcome, normalized for counting and merging.
#[derive(Debug)]
pub(crate) enum RelayFetchResult {
    /// The relay completed the round trip: events, or an empty EOSE from
    /// a connection that was live when the attempt started.
    Answered(std::collections::BTreeSet<Event>),
    /// The relay never spoke: silence until the deadline, or a fetch
    /// error that means the request was never issued to it.
    Silent,
}

/// Classify one relay's fetch outcome.
///
/// `relay_connected` is the relay's live connection status, consulted for
/// empty successes because nostr-sdk resolves relays it has given up on
/// to a fast `Ok(empty)`: only a connected relay's empty result means
/// "healthy relay, nothing stored" (EOSE).
///
/// An `Err` counts as silent, not spoken. In nostr-sdk 0.45.x per-relay
/// failures never surface as `Err` — the fetch collector logs them and
/// keeps collecting — so an `Err` here means the request never reached
/// the relay (target resolution, buffer limits). Counting it as answered
/// would inflate `relays_answered` in the unsafe direction if a future
/// SDK ever did propagate relay failures as `Err`.
pub(crate) fn classify_relay_fetch<E>(
    events: Result<std::collections::BTreeSet<Event>, E>,
    relay_connected: bool,
) -> RelayFetchResult {
    match events {
        Ok(set) if !set.is_empty() => RelayFetchResult::Answered(set),
        Ok(_) if relay_connected => RelayFetchResult::Answered(Default::default()),
        Ok(_) | Err(_) => RelayFetchResult::Silent,
    }
}

/// Count answered relays and merge their events.
///
/// Only a completed round trip counts as answered: deadline silence and
/// never-issued requests do not. Events merge into one set, deduplicated
/// by event id.
pub(crate) fn merge_relay_fetch_results(
    relays_queried: usize,
    results: Vec<RelayFetchResult>,
) -> (std::collections::BTreeSet<Event>, RelayHealth) {
    let mut events = std::collections::BTreeSet::new();
    let mut relays_answered = 0usize;
    for result in results {
        match result {
            RelayFetchResult::Answered(relay_events) => {
                relays_answered += 1;
                events.extend(relay_events);
            }
            RelayFetchResult::Silent => {}
        }
    }
    (
        events,
        RelayHealth {
            relays_queried,
            relays_answered,
        },
    )
}

/// A reusable read-only Nostr client for discovery.
/// Create once, pass to repeated `discover()` calls to avoid opening
/// new websocket connections and generating throwaway keys every time.
pub struct DiscoveryClient {
    client: Client,
}

impl DiscoveryClient {
    pub async fn new(relays: &[String]) -> Result<Self> {
        let _ = rustls::crypto::ring::default_provider().install_default();
        let client = Client::new();
        let mut added = 0;
        for relay in relays {
            match client.add_relay(relay).await {
                Ok(_) => added += 1,
                Err(e) => tracing::warn!("Nostr relay {relay}: {e}"),
            }
        }
        if added == 0 {
            anyhow::bail!(
                "Could not connect to any Nostr relay (tried {})",
                relays.len()
            );
        }
        client.connect().await;
        wait_for_relay_connections(&client, CONNECT_WAIT_CAP).await;
        Ok(Self { client })
    }
}

/// Discover meshes from Nostr relays.
///
/// If `cached_client` is provided, reuses its connections.  Otherwise
/// creates (and drops) a one-shot client — fine for the initial
/// `--auto` join but wasteful in tight loops. Discards relay health;
/// callers that need to distinguish "no meshes" from "no relay
/// answered" use [`discover_with_health`].
pub async fn discover(
    relays: &[String],
    filter: &MeshFilter,
    cached_client: Option<&DiscoveryClient>,
) -> Result<Vec<DiscoveredMesh>> {
    let (meshes, _health) = discover_with_health(relays, filter, cached_client).await?;
    Ok(meshes)
}

/// Turn per-relay fetch outcomes into counted classifications.
///
/// Empty successes are checked against the relay's live connection
/// status (nostr-sdk resolves relays it has given up on to a fast
/// Ok(empty)), deadline silence stays silent, and a fetch error means
/// the request never issued (nostr-sdk 0.45.x swallows per-relay
/// failures inside the collector), so the relay did not speak.
fn classify_relay_round(
    relay_urls: &[nostr_sdk::prelude::RelayUrl],
    results: Vec<RelayFetchOutcome>,
) -> Vec<RelayFetchResult> {
    let mut outcomes: Vec<RelayFetchResult> = Vec::with_capacity(results.len());
    for (url, (connected_at_start, result)) in relay_urls.iter().cloned().zip(results) {
        match result {
            Ok(Ok(relay_events)) => {
                // The SDK also resolves a relay it has given up
                // connecting to (or one mid-reconnect) to a fast
                // Ok(empty) — a spoke-relay is only one whose connection
                // was live when the attempt started.
                outcomes.push(classify_relay_fetch::<nostr_sdk::prelude::Error>(
                    Ok(relay_events),
                    connected_at_start,
                ));
            }
            Ok(Err(e)) => {
                tracing::warn!("Nostr relay {url} fetch failed: {e}");
                outcomes.push(classify_relay_fetch::<()>(Err(()), false));
            }
            Err(_) => {
                tracing::debug!("Nostr relay {url} stayed silent until the deadline");
                outcomes.push(RelayFetchResult::Silent);
            }
        }
    }
    outcomes
}

/// Discover meshes and report how many relays answered.
///
/// The fetch fans out per relay (one request each, 5 s timeout) so a
/// relay that errors or stays silent is visible instead of folded into
/// an all-or-nothing result; events are merged and deduplicated across
/// relays before the usual per-publisher dedupe. An all-relays failure
/// still returns `Ok` with empty results — plus `relays_answered == 0`
/// — rather than a hard error, matching `discover`'s contract.
pub async fn discover_with_health(
    relays: &[String],
    filter: &MeshFilter,
    cached_client: Option<&DiscoveryClient>,
) -> Result<(Vec<DiscoveredMesh>, RelayHealth)> {
    // Build a temporary client only when no cached one is supplied.
    let _tmp;
    let client: &Client = if let Some(cc) = cached_client {
        &cc.client
    } else {
        _tmp = build_discovery_client(relays).await?;
        &_tmp
    };

    // With a cached client the pool is the one it was built with; health
    // describes that pool, which is what the watchdog's long-lived
    // client re-checks against.
    let relay_urls: Vec<nostr_sdk::prelude::RelayUrl> =
        client.relays().await.keys().cloned().collect();

    let fetch_one = |url: nostr_sdk::prelude::RelayUrl| {
        let nostr_filter = Filter::new()
            .kind(Kind::Custom(MESH_SERVICE_KIND))
            .custom_tag(SingleLetterTag::LOWERCASE_K, "mesh-llm".to_string())
            .limit(100);
        // The timeout is ours, not the SDK's: fetch_events resolves a
        // silent relay to Ok(empty) at its own timeout, which would make
        // an outage indistinguishable from an empty relay. Wrapping the
        // un-timed future means only a relay that actually completed
        // (EOSE or error) resolves before the deadline; silence surfaces
        // as Err(Elapsed).
        // Connection status is sampled BEFORE the fetch, describing the
        // state the attempt started from; sampling after would let a
        // gave-up relay that reconnects mid-round masquerade as having
        // answered with its earlier fast Ok(empty).
        async move {
            let connected_at_start = client
                .relay(&url)
                .await
                .ok()
                .flatten()
                .is_some_and(|relay| relay.status().is_connected());
            let outcome = tokio::time::timeout(
                Duration::from_secs(5),
                client.fetch_events(ReqTarget::single(url, vec![nostr_filter])),
            )
            .await;
            (connected_at_start, outcome)
        }
    };
    let results = futures_util::future::join_all(relay_urls.iter().cloned().map(fetch_one)).await;

    let outcomes = classify_relay_round(&relay_urls, results);
    let (events, health) = merge_relay_fetch_results(relay_urls.len(), outcomes);

    let now = Timestamp::now().as_secs();

    // Dedupe by publisher (keep latest per pubkey, using replaceable event semantics)
    let latest = latest_events_by_pubkey(&events);

    let mut meshes = Vec::new();
    for event in latest.values() {
        let Some(discovered) = parse_discovered_mesh(event, now) else {
            continue;
        };

        if filter.matches(&discovered) {
            meshes.push(discovered);
        }
    }

    // Sort by node count (bigger meshes first), then VRAM
    meshes.sort_by(|a, b| {
        b.listing
            .node_count
            .cmp(&a.listing.node_count)
            .then(b.listing.total_vram_bytes.cmp(&a.listing.total_vram_bytes))
    });

    Ok((meshes, health))
}

/// How long a fresh discovery client waits for its relay connections
/// before the first round: long enough for a websocket handshake against
/// a healthy relay (observed well under a second), short enough that a
/// dead relay set does not stall the command.
const CONNECT_WAIT_CAP: Duration = Duration::from_secs(2);

/// Wait, capped, until every relay's first connection attempt has
/// settled.
///
/// nostr-sdk 0.45.x `connect()` only spawns connection tasks; it does
/// not wait for the handshakes. Sampling `relay.status()` immediately
/// afterwards therefore reports every relay of a freshly built client —
/// including healthy ones that answer moments later — as not connected,
/// which would classify a genuine empty EOSE as silence. Waiting here
/// makes the one-shot path's connection sample meaningful.
///
/// "Settled" means Connected or a status a first attempt ends in
/// (disconnected through banned, sleeping, shutdown): the wait exits as
/// soon as no relay is still initializing or connecting, so a pool with
/// one dead relay costs the handshake time of the healthy ones plus the
/// dead one's failed attempt, not the full cap. A relay that would
/// auto-reconnect later still samples as not connected, which can only
/// undercount `relays_answered` — never inflate it.
async fn wait_for_relay_connections(client: &Client, cap: Duration) {
    use nostr_sdk::relay::RelayStatus;

    let deadline = tokio::time::Instant::now() + cap;
    loop {
        let first_attempts_settled = client.relays().await.values().all(|relay| {
            matches!(
                relay.status(),
                RelayStatus::Connected
                    | RelayStatus::Disconnected
                    | RelayStatus::Terminated
                    | RelayStatus::Banned
                    | RelayStatus::Sleeping
                    | RelayStatus::Shutdown
            )
        });
        if first_attempts_settled || tokio::time::Instant::now() >= deadline {
            return;
        }
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
}

async fn build_discovery_client(relays: &[String]) -> Result<Client> {
    let _ = rustls::crypto::ring::default_provider().install_default();
    let client = Client::new();
    let mut added = 0;
    for relay in relays {
        match client.add_relay(relay).await {
            Ok(_) => added += 1,
            Err(e) => tracing::warn!("Nostr relay {relay}: {e}"),
        }
    }
    if added == 0 {
        anyhow::bail!(
            "Could not connect to any Nostr relay (tried {})",
            relays.len()
        );
    }
    client.connect().await;
    wait_for_relay_connections(&client, CONNECT_WAIT_CAP).await;
    Ok(client)
}

fn latest_events_by_pubkey<'a>(
    events: &'a std::collections::BTreeSet<Event>,
) -> std::collections::HashMap<String, &'a Event> {
    let mut latest: std::collections::HashMap<String, &'a Event> = std::collections::HashMap::new();
    for event in events.iter() {
        let pubkey = event.pubkey.to_hex();
        match latest.get(&pubkey) {
            Some(existing) if event.created_at.as_secs() <= existing.created_at.as_secs() => {}
            _ => {
                latest.insert(pubkey, event);
            }
        }
    }
    latest
}

fn parse_discovered_mesh(event: &Event, now: u64) -> Option<DiscoveredMesh> {
    let expires_at = event
        .tags
        .iter()
        .find(|t| t.as_slice().first().map(|s| s.as_str()) == Some("expiration"))
        .and_then(|t| t.as_slice().get(1))
        .and_then(|s| s.parse::<u64>().ok());
    if expires_at.is_some_and(|exp| exp < now) {
        return None;
    }

    let listing: MeshListing = match serde_json::from_str(&event.content) {
        Ok(listing) => listing,
        Err(err) => {
            tracing::warn!(
                "Skipping Nostr listing from {}: bad JSON: {err}",
                event.pubkey.to_bech32().unwrap_or_default()
            );
            return None;
        }
    };
    if let Err(err) = crate::mesh::Node::decode_invite_token(&listing.invite_token) {
        tracing::warn!(
            "Skipping Nostr listing from {}: {err}",
            event.pubkey.to_bech32().unwrap_or_default()
        );
        return None;
    }

    Some(DiscoveredMesh {
        listing,
        publisher_npub: event.pubkey.to_bech32().unwrap_or_default(),
        published_at: event.created_at.as_secs(),
        expires_at,
    })
}

#[cfg(test)]
mod relay_health_tests {
    use super::*;

    fn answered(events: &[&str]) -> RelayFetchResult {
        // Events are opaque to the accumulator (merged by set identity),
        // so any Event works; reuse one per unique id.
        let mut set = std::collections::BTreeSet::new();
        for id in events {
            let event = EventBuilder::new(Kind::Custom(1), *id)
                .finalize(&nostr_sdk::prelude::Keys::generate())
                .expect("sign");
            set.insert(event);
        }
        RelayFetchResult::Answered(set)
    }

    #[test]
    fn silent_relays_do_not_count_as_answered() {
        let (events, health) = merge_relay_fetch_results(
            3,
            vec![
                answered(&[]),
                RelayFetchResult::Silent,
                RelayFetchResult::Silent,
            ],
        );
        assert_eq!(
            health,
            RelayHealth {
                relays_queried: 3,
                relays_answered: 1
            }
        );
        assert!(events.is_empty());
    }

    #[test]
    fn an_all_silent_round_reads_as_zero_answered() {
        let (_, health) =
            merge_relay_fetch_results(5, (0..5).map(|_| RelayFetchResult::Silent).collect());
        assert!(!health.any_relay_answered());
    }

    #[test]
    fn duplicate_events_across_relays_merge_to_one() {
        // The same event fetched from two relays must not duplicate.
        let event = EventBuilder::new(Kind::Custom(1), "same")
            .finalize(&nostr_sdk::prelude::Keys::generate())
            .expect("sign");
        let mut first = std::collections::BTreeSet::new();
        first.insert(event.clone());
        let mut second = std::collections::BTreeSet::new();
        second.insert(event);
        let (events, health) = merge_relay_fetch_results(
            2,
            vec![
                RelayFetchResult::Answered(first),
                RelayFetchResult::Answered(second),
            ],
        );
        assert_eq!(events.len(), 1);
        assert_eq!(health.relays_answered, 2);
    }

    #[test]
    fn an_empty_result_from_a_disconnected_relay_is_silence() {
        // The SDK resolves relays it gave up connecting to a fast
        // Ok(empty); without the connection check a sustained outage
        // would read as every relay answering.
        let classified = classify_relay_fetch::<()>(Ok(Default::default()), false);
        assert!(matches!(classified, RelayFetchResult::Silent));
    }

    #[test]
    fn an_empty_result_from_a_connected_relay_is_an_answer() {
        // EOSE with nothing stored is a healthy, speaking relay.
        let classified = classify_relay_fetch::<()>(Ok(Default::default()), true);
        assert!(matches!(classified, RelayFetchResult::Answered(_)));
    }

    #[test]
    fn an_errored_fetch_never_reached_the_relay_and_is_silence() {
        // In nostr-sdk 0.45.x per-relay failures are swallowed by the
        // fetch collector; an Err means the request was never issued
        // (target resolution, buffer limits). The relay did not speak.
        let classified = classify_relay_fetch::<()>(Err(()), true);
        assert!(matches!(classified, RelayFetchResult::Silent));
        let classified = classify_relay_fetch::<()>(Err(()), false);
        assert!(matches!(classified, RelayFetchResult::Silent));
    }

    #[test]
    fn a_non_empty_result_is_an_answer_regardless_of_status() {
        let event = EventBuilder::new(Kind::Custom(1), "x")
            .finalize(&nostr_sdk::prelude::Keys::generate())
            .expect("sign");
        let mut set = std::collections::BTreeSet::new();
        set.insert(event);
        let classified = classify_relay_fetch::<()>(Ok(set), false);
        assert!(matches!(classified, RelayFetchResult::Answered(_)));
    }
}

#[cfg(test)]
mod filter_tests {
    use super::*;

    fn make_mesh_for_filter(
        serving: &[&str],
        wanted: &[&str],
        on_disk: &[&str],
        vram: u64,
        region: Option<&str>,
    ) -> DiscoveredMesh {
        make_mesh_for_filter_named(serving, wanted, on_disk, vram, region, None)
    }

    fn make_mesh_for_filter_named(
        serving: &[&str],
        wanted: &[&str],
        on_disk: &[&str],
        vram: u64,
        region: Option<&str>,
        name: Option<&str>,
    ) -> DiscoveredMesh {
        DiscoveredMesh {
            listing: MeshListing {
                invite_token: "tok".into(),
                serving: serving.iter().map(|s| s.to_string()).collect(),
                wanted: wanted.iter().map(|s| s.to_string()).collect(),
                on_disk: on_disk.iter().map(|s| s.to_string()).collect(),
                total_vram_bytes: vram,
                node_count: 1,
                client_count: 0,
                max_clients: 0,
                name: name.map(|s| s.to_string()),
                region: region.map(|s| s.to_string()),
                mesh_id: None,
                publisher_serves: None,
            },
            publisher_npub: "npub-test".into(),
            published_at: 1000,
            expires_at: Some(2000),
        }
    }

    #[test]
    fn filter_default_matches_all() {
        let m = make_mesh_for_filter(&["Qwen3-8B-Q4_K_M"], &[], &[], 8_000_000_000, None);
        assert!(MeshFilter::default().matches(&m));
    }

    #[test]
    fn filter_model_serving() {
        let m = make_mesh_for_filter(&["Qwen3-8B-Q4_K_M"], &[], &[], 8_000_000_000, None);
        let f = MeshFilter {
            model: Some("qwen3-8b".into()),
            ..Default::default()
        };
        assert!(f.matches(&m));
    }

    #[test]
    fn filter_model_wanted() {
        let m = make_mesh_for_filter(&[], &["Qwen3-32B-Q4_K_M"], &[], 8_000_000_000, None);
        let f = MeshFilter {
            model: Some("32b".into()),
            ..Default::default()
        };
        assert!(f.matches(&m));
    }

    #[test]
    fn filter_model_on_disk() {
        let m = make_mesh_for_filter(&[], &[], &["MiniMax-M2.5-Q4_K_M"], 8_000_000_000, None);
        let f = MeshFilter {
            model: Some("minimax".into()),
            ..Default::default()
        };
        assert!(f.matches(&m));
    }

    #[test]
    fn filter_model_no_match() {
        let m = make_mesh_for_filter(&["Qwen3-8B-Q4_K_M"], &[], &[], 8_000_000_000, None);
        let f = MeshFilter {
            model: Some("llama".into()),
            ..Default::default()
        };
        assert!(!f.matches(&m));
    }

    #[test]
    fn filter_min_vram() {
        let m = make_mesh_for_filter(&[], &[], &[], 8_000_000_000, None);
        let pass = MeshFilter {
            min_vram_gb: Some(5.0),
            ..Default::default()
        };
        let fail = MeshFilter {
            min_vram_gb: Some(16.0),
            ..Default::default()
        };
        assert!(pass.matches(&m));
        assert!(!fail.matches(&m));
    }

    #[test]
    fn filter_region() {
        let m = make_mesh_for_filter(&[], &[], &[], 8_000_000_000, Some("us-east"));
        let pass = MeshFilter {
            region: Some("us-east".into()),
            ..Default::default()
        };
        let fail = MeshFilter {
            region: Some("eu-west".into()),
            ..Default::default()
        };
        assert!(pass.matches(&m));
        assert!(!fail.matches(&m));
    }

    #[test]
    fn filter_region_case_insensitive() {
        let m = make_mesh_for_filter(&[], &[], &[], 8_000_000_000, Some("US-East"));
        let f = MeshFilter {
            region: Some("us-east".into()),
            ..Default::default()
        };
        assert!(f.matches(&m));
    }

    #[test]
    fn filter_combined() {
        let m = make_mesh_for_filter(
            &["Qwen3-8B-Q4_K_M"],
            &[],
            &[],
            16_000_000_000,
            Some("us-east"),
        );
        let pass = MeshFilter {
            model: Some("qwen3".into()),
            min_vram_gb: Some(10.0),
            region: Some("us-east".into()),
            ..Default::default()
        };
        let fail_model = MeshFilter {
            model: Some("llama".into()),
            min_vram_gb: Some(10.0),
            region: Some("us-east".into()),
            ..Default::default()
        };
        assert!(pass.matches(&m));
        assert!(!fail_model.matches(&m));
    }

    #[test]
    fn filter_name_exact() {
        let m = make_mesh_for_filter_named(&[], &[], &[], 8_000_000_000, None, Some("poker-night"));
        let f = MeshFilter {
            name: Some("poker-night".into()),
            ..Default::default()
        };
        assert!(f.matches(&m));
    }

    #[test]
    fn filter_name_case_insensitive() {
        let m = make_mesh_for_filter_named(&[], &[], &[], 8_000_000_000, None, Some("Poker-Night"));
        let f = MeshFilter {
            name: Some("poker-night".into()),
            ..Default::default()
        };
        assert!(f.matches(&m));
    }

    #[test]
    fn filter_name_no_match() {
        let m = make_mesh_for_filter_named(&[], &[], &[], 8_000_000_000, None, Some("other-mesh"));
        let f = MeshFilter {
            name: Some("poker-night".into()),
            ..Default::default()
        };
        assert!(!f.matches(&m));
    }

    #[test]
    fn filter_name_mesh_unnamed() {
        // Mesh has no name — filter by name should not match.
        let m = make_mesh_for_filter_named(&[], &[], &[], 8_000_000_000, None, None);
        let f = MeshFilter {
            name: Some("poker-night".into()),
            ..Default::default()
        };
        assert!(!f.matches(&m));
    }

    #[test]
    fn filter_name_none_matches_all() {
        // No name filter — matches meshes with and without names.
        let named =
            make_mesh_for_filter_named(&[], &[], &[], 8_000_000_000, None, Some("poker-night"));
        let unnamed = make_mesh_for_filter_named(&[], &[], &[], 8_000_000_000, None, None);
        let f = MeshFilter::default();
        assert!(f.matches(&named));
        assert!(f.matches(&unnamed));
    }
}

// ---------------------------------------------------------------------------
// Integration test — publish/discover against real Nostr relays
// ---------------------------------------------------------------------------
#[cfg(test)]
mod integration_tests {
    use super::*;
    use base64::Engine;

    fn fake_invite_token() -> String {
        // Build a syntactically valid invite token by encoding a minimal
        // EndpointAddr JSON. We use Node::invite_token indirectly by just
        // crafting the JSON that decode_invite_token expects.
        use std::sync::atomic::{AtomicU64, Ordering};
        static COUNTER: AtomicU64 = AtomicU64::new(1);
        let n = COUNTER.fetch_add(1, Ordering::Relaxed);
        // EndpointAddr serialises as {"id":"<base32>","addrs":[]}
        // We need a valid 32-byte public key. Use a deterministic one.
        let mut seed = [0u8; 32];
        seed[..8].copy_from_slice(&n.to_le_bytes());
        let key = iroh::SecretKey::from_bytes(&seed);
        let addr = iroh::EndpointAddr {
            id: iroh::EndpointId::from(key.public()),
            addrs: Default::default(),
        };
        let json = serde_json::to_vec(&addr).expect("serialize");
        base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(json)
    }

    /// End-to-end: two publishers advertise the same mesh, a reusable
    /// DiscoveryClient finds both listings, and fields round-trip correctly.
    /// Covers publish, discover, multi-publisher, and client reuse in one test.
    #[tokio::test]
    #[ignore = "requires live public Nostr relays"]
    async fn publish_discover_round_trip() {
        let relays: Vec<String> = DEFAULT_RELAYS.iter().map(|s| s.to_string()).collect();
        let mesh_name = format!("mesh-llm-test-{}", rand::random::<u32>());
        let mesh_id = format!("test-id-{}", rand::random::<u32>());

        // Publisher A
        let keys_a = Keys::generate();
        let pub_a = Publisher::new(keys_a.clone(), &relays)
            .await
            .expect("pub_a");
        let token_a = fake_invite_token();
        let token_b = fake_invite_token();
        let listing_a = MeshListing {
            invite_token: token_a.clone(),
            serving: vec!["Qwen3-8B-Q4_K_M".into()],
            wanted: vec![],
            on_disk: vec![],
            total_vram_bytes: 16_000_000_000,
            node_count: 2,
            client_count: 0,
            max_clients: 0,
            name: Some(mesh_name.clone()),
            region: Some("test-region".into()),
            mesh_id: Some(mesh_id.clone()),
            publisher_serves: None,
        };
        pub_a.publish(&listing_a, 120).await.expect("publish A");

        // Publisher B — same mesh, different invite token
        let keys_b = Keys::generate();
        let pub_b = Publisher::new(keys_b.clone(), &relays)
            .await
            .expect("pub_b");
        let mut listing_b = listing_a.clone();
        listing_b.invite_token = token_b.clone();
        pub_b.publish(&listing_b, 120).await.expect("publish B");

        tokio::time::sleep(Duration::from_secs(3)).await;

        // Discover with reusable client (tests DiscoveryClient + discover)
        let dc = DiscoveryClient::new(&relays).await.expect("dc");
        let meshes = discover(&relays, &MeshFilter::default(), Some(&dc))
            .await
            .expect("discover");

        let found: Vec<_> = meshes
            .iter()
            .filter(|m| m.listing.mesh_id.as_deref() == Some(mesh_id.as_str()))
            .collect();
        assert!(
            found.len() >= 2,
            "should find both publishers for mesh_id={mesh_id}, found {}",
            found.len()
        );

        // Verify fields round-tripped
        let m = &found[0];
        assert_eq!(m.listing.name.as_deref(), Some(mesh_name.as_str()));
        assert_eq!(m.listing.serving, vec!["Qwen3-8B-Q4_K_M"]);
        assert_eq!(m.listing.node_count, 2);
        assert_eq!(m.listing.total_vram_bytes, 16_000_000_000);

        // Both invite tokens present
        let tokens: Vec<_> = found
            .iter()
            .map(|m| m.listing.invite_token.as_str())
            .collect();
        assert!(
            tokens.contains(&token_a.as_str()),
            "missing token_a in {tokens:?}"
        );
        assert!(
            tokens.contains(&token_b.as_str()),
            "missing token_b in {tokens:?}"
        );

        // Second discover with same client still works
        let r2 = discover(&relays, &MeshFilter::default(), Some(&dc))
            .await
            .expect("second discover");
        let found2: Vec<_> = r2
            .iter()
            .filter(|m| m.listing.mesh_id.as_deref() == Some(mesh_id.as_str()))
            .collect();
        assert!(found2.len() >= 2, "reused client should still find both");

        // Cleanup
        pub_a.unpublish().await.ok();
        pub_b.unpublish().await.ok();
    }
}
