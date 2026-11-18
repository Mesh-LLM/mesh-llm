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

/// One relay's round outcome, accumulated from the notification stream.
#[derive(Debug, Default)]
pub(crate) struct RelayRound {
    /// The relay sent EOSE for this round's subscription: the query
    /// completed, so an empty event set still counts as answered.
    eose: bool,
    /// The relay terminated the subscription without EOSE (Closed), so
    /// no completion should be expected even if the round waits.
    closed: bool,
    /// Events delivered before the round ended — kept even when the
    /// relay never finished, because a delivered listing is evidence
    /// regardless of completion.
    events: std::collections::BTreeSet<Event>,
}

impl RelayRound {
    fn terminal(&self) -> bool {
        self.eose || self.closed
    }
}

/// One relay's fetch outcome, normalized for counting and merging.
#[derive(Debug)]
pub(crate) enum RelayFetchResult {
    /// The relay completed the round trip (EOSE) or delivered events
    /// before the round ended — either way it spoke.
    Answered(std::collections::BTreeSet<Event>),
    /// The relay neither completed nor delivered anything: never
    /// connected, disconnected before EOSE, closed without EOSE, or
    /// silent until the deadline.
    Silent,
}

/// Classify one relay's round from completion evidence.
///
/// A relay counts as answered when its query completed (EOSE observed on
/// the notification stream) or when it demonstrably spoke (delivered
/// events, even if it never finished). Only a relay that neither
/// completed nor delivered anything is silent: never connected,
/// disconnected before EOSE, or hung to the deadline. Unlike a
/// pre-fetch connection snapshot, EOSE cannot be observed for a relay
/// that dropped mid-fetch, so a truncated round can no longer pose as a
/// completed empty one.
pub(crate) fn classify_relay_round_outcome(round: RelayRound) -> RelayFetchResult {
    if round.eose || !round.events.is_empty() {
        RelayFetchResult::Answered(round.events)
    } else {
        RelayFetchResult::Silent
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

    // Drive one subscription across every relay and route the
    // notification stream per relay. This replaces per-relay
    // `fetch_events` futures, which have two blind spots this round
    // cannot afford: their collected events are dropped if the future
    // is cancelled at the deadline (a delivered listing would vanish),
    // and their `Ok(empty)` cannot distinguish a completed empty query
    // (EOSE) from a stream that ended because the relay disconnected
    // mid-fetch. The notification stream exposes both signals
    // per relay: `EndOfStoredEvents` is completion evidence a
    // connection snapshot can never fake, and events accumulate
    // outside any cancellable future.
    let subscription_id = SubscriptionId::generate();
    // Subscribe to the notification channel BEFORE sending the REQ:
    // anything received before the channel exists is not delivered to
    // it (nostr-sdk documents this ordering requirement).
    let mut notifications = client.notifications();
    let nostr_filter = Filter::new()
        .kind(Kind::Custom(MESH_SERVICE_KIND))
        .custom_tag(SingleLetterTag::LOWERCASE_K, "mesh-llm".to_string())
        .limit(100);
    let targets: Vec<(nostr_sdk::prelude::RelayUrl, Vec<Filter>)> = relay_urls
        .iter()
        .cloned()
        .map(|url| (url, vec![nostr_filter.clone()]))
        .collect();
    let _output = client
        .subscribe(ReqTarget::manual(targets))
        .with_id(subscription_id.clone())
        .await?;

    let mut rounds: std::collections::HashMap<nostr_sdk::prelude::RelayUrl, RelayRound> =
        relay_urls
            .iter()
            .cloned()
            .map(|url| (url, RelayRound::default()))
            .collect();
    let deadline = tokio::time::Instant::now() + Duration::from_secs(5);
    loop {
        let pending: usize = rounds.values().filter(|r| !r.terminal()).count();
        if pending == 0 {
            break;
        }
        let next = tokio::time::timeout_at(deadline, notifications.next()).await;
        let notification = match next {
            Ok(Some(notification)) => notification,
            // Deadline: relays that neither completed nor delivered
            // anything classify as silent; their partial events (if any)
            // are already accumulated and are kept.
            Ok(None) | Err(_) => break,
        };
        match notification {
            ClientNotification::Event {
                relay_url,
                subscription_id: id,
                event,
            } if id == subscription_id => {
                // This variant is client-deduplicated ("first time the
                // event is seen"), so an event two relays both deliver
                // attributes only to whichever arrived first. The merged
                // result is unaffected (it deduplicates anyway), and the
                // counting effect is confined to a relay whose only
                // deliveries were duplicates — it can under-count toward
                // Silent, never inflate toward Answered.
                if let Some(round) = rounds.get_mut(&relay_url) {
                    round.events.insert(*event);
                }
            }
            ClientNotification::Message { relay_url, message } => match *message {
                RelayMessage::EndOfStoredEvents(id) if *id == subscription_id => {
                    if let Some(round) = rounds.get_mut(&relay_url) {
                        round.eose = true;
                    }
                }
                // A relay that closes the subscription (rejected,
                // auth-required, rate-limited) will never send EOSE for
                // it: mark it terminal so the round does not wait out
                // the deadline on its behalf.
                RelayMessage::Closed {
                    subscription_id: id,
                    ..
                } if *id == subscription_id => {
                    if let Some(round) = rounds.get_mut(&relay_url) {
                        round.closed = true;
                    }
                }
                _ => {}
            },
            _ => {}
        }
    }
    // Always drop the subscription, including on the cached client the
    // watchdog reuses — a leaked REQ would keep streaming events into
    // later rounds' notification channels.
    let _ = client.unsubscribe(&subscription_id).await;

    let outcomes = relay_urls
        .iter()
        .filter_map(|url| rounds.remove(url))
        .map(classify_relay_round_outcome)
        .collect();
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
    fn a_round_without_eose_or_events_is_silence() {
        // A relay that never connected, disconnected before EOSE, or
        // hung to the deadline leaves neither completion evidence nor
        // speech — the exact round a truncated `Ok(empty)` from the
        // old fetch path used to fake.
        let classified = classify_relay_round_outcome(RelayRound::default());
        assert!(matches!(classified, RelayFetchResult::Silent));

        // A subscription the relay Closed without EOSE is the same
        // non-answer.
        let closed = RelayRound {
            closed: true,
            ..RelayRound::default()
        };
        let classified = classify_relay_round_outcome(closed);
        assert!(matches!(classified, RelayFetchResult::Silent));
    }

    #[test]
    fn a_completed_empty_round_is_an_answer() {
        // EOSE with nothing stored is a healthy relay that finished the
        // query — completion evidence a connection snapshot cannot
        // fake.
        let completed_empty = RelayRound {
            eose: true,
            ..RelayRound::default()
        };
        let classified = classify_relay_round_outcome(completed_empty);
        assert!(matches!(classified, RelayFetchResult::Answered(set) if set.is_empty()));
    }

    #[test]
    fn a_relay_that_delivered_events_without_finishing_is_an_answer_and_keeps_events() {
        // The round deadline hit before EOSE, but the relay already
        // spoke: its events must survive the bound and it counts as
        // answered — a delivered listing is evidence regardless of
        // completion.
        let event = EventBuilder::new(Kind::Custom(1), "x")
            .finalize(&nostr_sdk::prelude::Keys::generate())
            .expect("sign");
        let mut set = std::collections::BTreeSet::new();
        set.insert(event);
        let unfinished = RelayRound {
            events: set,
            ..RelayRound::default()
        };
        let classified = classify_relay_round_outcome(unfinished);
        match classified {
            RelayFetchResult::Answered(events) => assert_eq!(events.len(), 1),
            other => panic!("expected Answered, got {other:?}"),
        }
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
