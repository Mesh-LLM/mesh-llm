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
    /// Relays that spoke: completed the query (EOSE) or delivered
    /// events before the round ended. Silence-until-deadline and
    /// never-issued requests do not count.
    pub relays_answered: usize,
    /// Relays that completed their query with EOSE. Higher bar than
    /// `relays_answered`: a relay that delivered partial events and
    /// disconnected did not finish, so absence inference must not rest
    /// on it — the listing it never sent may still have been coming.
    pub queries_completed: usize,
}

impl RelayHealth {
    /// False only when no relay answered at all — the "cannot tell"
    /// state for the CLI's outage reporting.
    pub fn any_relay_answered(&self) -> bool {
        self.relays_answered > 0
    }

    /// True when at least one relay finished its query. Only a
    /// completed query that did not contain the listing is evidence of
    /// absence; the absence-based takeover decision requires this.
    pub fn any_query_completed(&self) -> bool {
        self.queries_completed > 0
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
    /// The relay spoke: completed the query (EOSE) or delivered events
    /// before the round ended. `completed_query` distinguishes the two —
    /// only a completed query supports absence inference.
    Answered {
        events: std::collections::BTreeSet<Event>,
        completed_query: bool,
    },
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
        RelayFetchResult::Answered {
            events: round.events,
            completed_query: round.eose,
        }
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
    let mut queries_completed = 0usize;
    for result in results {
        match result {
            RelayFetchResult::Answered {
                events: relay_events,
                completed_query,
            } => {
                relays_answered += 1;
                queries_completed += usize::from(completed_query);
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
            queries_completed,
        },
    )
}

/// The relays a discovery round asks.
///
/// Each round opens its own short-lived connection per relay rather than
/// reusing a long-lived nostr-sdk client: the SDK's notification streams drop
/// lagged messages and end a disconnected subscription exactly as they end a
/// completed one, and absence inference needs to tell those apart. Kept as a
/// type so the publish watchdog can hold its relay set across rounds.
pub struct DiscoveryClient {
    relays: Vec<String>,
}

impl DiscoveryClient {
    pub async fn new(relays: &[String]) -> Result<Self> {
        if relays.is_empty() {
            anyhow::bail!("Could not connect to any Nostr relay (tried 0)");
        }
        Ok(Self {
            relays: relays.to_vec(),
        })
    }
}

/// Discover meshes from Nostr relays.
///
/// Discards relay health; callers that need to distinguish "no meshes" from
/// "no relay answered" use [`discover_with_health`].
pub async fn discover(
    relays: &[String],
    filter: &MeshFilter,
    cached_client: Option<&DiscoveryClient>,
) -> Result<Vec<DiscoveredMesh>> {
    let (meshes, _health) = discover_with_health(relays, filter, cached_client).await?;
    Ok(meshes)
}

/// One deadline for a whole round: connecting, the query and its answer.
const DISCOVERY_ROUND_DEADLINE: Duration = Duration::from_secs(5);

/// Most events kept from one relay. Listings are one replaceable event per
/// publisher, so a relay sending more is misbehaving; its round stops there
/// and does not count as completed.
const MAX_EVENTS_PER_RELAY: usize = 500;

/// Discover meshes and report how many relays answered.
///
/// Every relay is queried concurrently over its own websocket under one
/// deadline. A relay counts as answered when it completed the query (EOSE)
/// or delivered events; everything else — refused, failed handshake,
/// dropped, closed the subscription, hung — is silent. An all-silent round
/// still returns `Ok` with empty results and `relays_answered == 0`.
pub async fn discover_with_health(
    relays: &[String],
    filter: &MeshFilter,
    cached_client: Option<&DiscoveryClient>,
) -> Result<(Vec<DiscoveredMesh>, RelayHealth)> {
    let relays = cached_client.map_or(relays, |client| client.relays.as_slice());
    discover_round(relays, filter, DISCOVERY_ROUND_DEADLINE).await
}

async fn discover_round(
    relays: &[String],
    filter: &MeshFilter,
    deadline: Duration,
) -> Result<(Vec<DiscoveredMesh>, RelayHealth)> {
    if relays.is_empty() {
        anyhow::bail!("Could not connect to any Nostr relay (tried 0)");
    }
    let _ = rustls::crypto::ring::default_provider().install_default();
    let deadline = tokio::time::Instant::now() + deadline;
    let rounds = futures_util::future::join_all(
        relays
            .iter()
            .map(|relay| query_relay_until(relay, deadline)),
    )
    .await;
    let outcomes = rounds
        .into_iter()
        .map(classify_relay_round_outcome)
        .collect();
    let (events, health) = merge_relay_fetch_results(relays.len(), outcomes);

    let now = Timestamp::now().as_secs();
    let latest = latest_events_by_pubkey(&events);
    let mut meshes: Vec<DiscoveredMesh> = latest
        .values()
        .filter_map(|event| parse_discovered_mesh(event, now))
        .filter(|mesh| filter.matches(mesh))
        .collect();

    // Sort by node count (bigger meshes first), then VRAM
    meshes.sort_by(|a, b| {
        b.listing
            .node_count
            .cmp(&a.listing.node_count)
            .then(b.listing.total_vram_bytes.cmp(&a.listing.total_vram_bytes))
    });

    Ok((meshes, health))
}

/// Ask one relay for mesh listings and record what it did before `deadline`.
async fn query_relay_until(relay: &str, deadline: tokio::time::Instant) -> RelayRound {
    let mut round = RelayRound::default();
    match tokio::time::timeout_at(deadline, query_relay(relay, &mut round)).await {
        Ok(Err(error)) => tracing::debug!("Nostr relay {relay}: {error:#}"),
        Ok(Ok(())) => {}
        Err(_) => tracing::debug!("Nostr relay {relay}: no answer before the round deadline"),
    }
    round
}

/// Events stay in `round` as they arrive, so a relay cut off by the deadline
/// keeps what it already delivered.
async fn query_relay(relay: &str, round: &mut RelayRound) -> Result<()> {
    use futures_util::{SinkExt, StreamExt};
    use tokio_tungstenite::tungstenite::Message;

    let (mut socket, _) = tokio_tungstenite::connect_async(relay).await?;
    let subscription_id = SubscriptionId::generate();
    let request = ClientMessage::req(subscription_id.clone(), mesh_listing_filter());
    socket.send(Message::text(request.as_json())).await?;
    while !round.terminal() {
        let Some(message) = socket.next().await else {
            anyhow::bail!("closed the connection before EOSE");
        };
        match message? {
            Message::Text(text) => apply_relay_message(round, &subscription_id, &text),
            Message::Close(_) => anyhow::bail!("closed the connection before EOSE"),
            _ => {}
        }
    }
    let close = ClientMessage::close(subscription_id);
    let _ = socket.send(Message::text(close.as_json())).await;
    let _ = socket.close(None).await;
    Ok(())
}

fn mesh_listing_filter() -> Filter {
    Filter::new()
        .kind(Kind::Custom(MESH_SERVICE_KIND))
        .custom_tag(SingleLetterTag::LOWERCASE_K, "mesh-llm".to_string())
        .limit(100)
}

/// Fold one relay message into the round.
///
/// Every event is verified here: id hash and signature must match the content
/// actually received, and it must be a mesh listing. Collection stops at EOSE,
/// at the subscription's CLOSED, and at [`MAX_EVENTS_PER_RELAY`].
fn apply_relay_message(round: &mut RelayRound, subscription_id: &SubscriptionId, text: &str) {
    let Ok(message) = RelayMessage::from_json(text) else {
        return;
    };
    match message {
        RelayMessage::Event {
            subscription_id: id,
            event,
        } if *id == *subscription_id => {
            let event = event.into_owned();
            if event.kind != Kind::Custom(MESH_SERVICE_KIND) {
                return;
            }
            if let Err(error) = event.verify() {
                tracing::warn!("Nostr relay sent an invalid event: {error}");
                return;
            }
            round.events.insert(event);
            if round.events.len() >= MAX_EVENTS_PER_RELAY {
                round.closed = true;
            }
        }
        RelayMessage::EndOfStoredEvents(id) if *id == *subscription_id => round.eose = true,
        // A relay that closes the subscription (rejected, auth-required,
        // rate-limited) will never send EOSE for it.
        RelayMessage::Closed {
            subscription_id: id,
            ..
        } if *id == *subscription_id => round.closed = true,
        _ => {}
    }
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
        RelayFetchResult::Answered {
            events: set,
            completed_query: true,
        }
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
                relays_answered: 1,
                queries_completed: 1
            }
        );
        assert!(events.is_empty());
    }

    #[test]
    fn an_all_silent_round_reads_as_zero_answered() {
        let (_, health) =
            merge_relay_fetch_results(5, (0..5).map(|_| RelayFetchResult::Silent).collect());
        assert!(!health.any_relay_answered());
        assert!(!health.any_query_completed());
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
                RelayFetchResult::Answered {
                    events: first,
                    completed_query: true,
                },
                RelayFetchResult::Answered {
                    events: second,
                    completed_query: true,
                },
            ],
        );
        assert_eq!(events.len(), 1);
        assert_eq!(health.relays_answered, 2);
        assert_eq!(health.queries_completed, 2);
    }

    fn event_message(id: &SubscriptionId, event: &Event) -> String {
        RelayMessage::event(id.clone(), event.clone()).as_json()
    }

    /// A relay can replay a valid event's id over altered content. The
    /// collector verifies every event itself, so the forged one is dropped.
    #[test]
    fn an_event_reusing_a_valid_id_with_altered_content_is_rejected() {
        let id = SubscriptionId::generate();
        let valid = EventBuilder::new(Kind::Custom(MESH_SERVICE_KIND), "legitimate")
            .finalize(&Keys::generate())
            .expect("sign");
        let other = EventBuilder::new(Kind::Custom(MESH_SERVICE_KIND), "forged")
            .finalize(&Keys::generate())
            .expect("sign");
        let mut value = serde_json::to_value(&other).expect("serialize");
        value["id"] = serde_json::to_value(valid.id).expect("id json");
        let forged: Event = serde_json::from_value(value).expect("deserialize");

        let mut accepted = RelayRound::default();
        apply_relay_message(&mut accepted, &id, &event_message(&id, &valid));
        assert_eq!(accepted.events.len(), 1);

        let mut rejected = RelayRound::default();
        apply_relay_message(&mut rejected, &id, &event_message(&id, &forged));
        assert!(rejected.events.is_empty());
    }

    #[test]
    fn events_for_another_subscription_or_kind_are_ignored() {
        let id = SubscriptionId::generate();
        let note = EventBuilder::new(Kind::TextNote, "x")
            .finalize(&Keys::generate())
            .expect("sign");
        let listing = EventBuilder::new(Kind::Custom(MESH_SERVICE_KIND), "x")
            .finalize(&Keys::generate())
            .expect("sign");
        let mut round = RelayRound::default();
        apply_relay_message(&mut round, &id, &event_message(&id, &note));
        let other = SubscriptionId::generate();
        apply_relay_message(&mut round, &id, &event_message(&other, &listing));
        apply_relay_message(&mut round, &id, &RelayMessage::eose(other).as_json());
        assert!(round.events.is_empty());
        assert!(!round.terminal());
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
        assert!(
            matches!(classified, RelayFetchResult::Answered { events, completed_query: true } if events.is_empty())
        );
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
            RelayFetchResult::Answered {
                events,
                completed_query: false,
            } => assert_eq!(events.len(), 1),
            other => panic!("expected Answered without completion, got {other:?}"),
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

    pub(super) fn fake_invite_token() -> String {
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

/// Full discovery rounds against scripted local relays.
#[cfg(test)]
mod round_tests {
    use super::*;
    use futures_util::{SinkExt, StreamExt};
    use tokio_tungstenite::tungstenite::Message;

    /// What a scripted relay does after it receives the REQ.
    enum Script {
        /// Send these events, then EOSE, then these live events.
        Answer {
            stored: Vec<Event>,
            live: Vec<Event>,
        },
        /// Send these events, then drop the connection before EOSE.
        DropAfter(Vec<Event>),
        /// Close the subscription without EOSE.
        Closed,
        /// Accept the REQ and never answer.
        Hang,
    }

    async fn scripted_relay(script: Script) -> String {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            let (stream, _) = listener.accept().await.unwrap();
            let mut socket = tokio_tungstenite::accept_async(stream).await.unwrap();
            let id = loop {
                let Some(Ok(Message::Text(text))) = socket.next().await else {
                    return;
                };
                if let Ok(ClientMessage::Req {
                    subscription_id, ..
                }) = ClientMessage::from_json(text.as_str())
                {
                    break subscription_id.into_owned();
                }
            };
            let send = |message: RelayMessage| Message::text(message.as_json());
            match script {
                Script::Answer { stored, live } => {
                    for event in stored {
                        let _ = socket
                            .send(send(RelayMessage::event(id.clone(), event)))
                            .await;
                    }
                    let _ = socket.send(send(RelayMessage::eose(id.clone()))).await;
                    for event in live {
                        let _ = socket
                            .send(send(RelayMessage::event(id.clone(), event)))
                            .await;
                    }
                }
                Script::DropAfter(events) => {
                    for event in events {
                        let _ = socket
                            .send(send(RelayMessage::event(id.clone(), event)))
                            .await;
                    }
                    return;
                }
                Script::Closed => {
                    let closed = RelayMessage::closed(id, "error: rate-limited");
                    let _ = socket.send(send(closed)).await;
                }
                Script::Hang => {}
            }
            // Keep the connection open until the client goes away.
            while socket.next().await.is_some() {}
        });
        format!("ws://{addr}")
    }

    fn listing(name: &str) -> Event {
        let listing = MeshListing {
            invite_token: super::integration_tests::fake_invite_token(),
            serving: vec![],
            wanted: vec![],
            on_disk: vec![],
            total_vram_bytes: 0,
            node_count: 1,
            client_count: 0,
            max_clients: 0,
            name: Some(name.to_string()),
            region: None,
            mesh_id: Some(format!("id-{name}")),
            publisher_serves: None,
        };
        EventBuilder::new(
            Kind::Custom(MESH_SERVICE_KIND),
            serde_json::to_string(&listing).unwrap(),
        )
        .tag(Tag::parse(["k", "mesh-llm"]).unwrap())
        .finalize(&Keys::generate())
        .unwrap()
    }

    /// A port nothing listens on: the connection is refused at once.
    async fn refused_relay() -> String {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        drop(listener);
        format!("ws://{addr}")
    }

    const DEADLINE: Duration = Duration::from_secs(2);

    #[tokio::test]
    async fn completed_queries_count_and_refused_relays_are_silent() {
        let relays = vec![
            scripted_relay(Script::Answer {
                stored: vec![listing("a")],
                live: vec![],
            })
            .await,
            scripted_relay(Script::Answer {
                stored: vec![],
                live: vec![],
            })
            .await,
            refused_relay().await,
        ];
        let (meshes, health) = discover_round(&relays, &MeshFilter::default(), DEADLINE)
            .await
            .unwrap();
        assert_eq!(meshes.len(), 1);
        assert_eq!(
            health,
            RelayHealth {
                relays_queried: 3,
                relays_answered: 2,
                queries_completed: 2,
            }
        );
    }

    #[tokio::test]
    async fn a_relay_that_drops_before_eose_keeps_its_events_but_completes_nothing() {
        let relays = vec![scripted_relay(Script::DropAfter(vec![listing("other")])).await];
        let filter = MeshFilter {
            name: Some("target".to_string()),
            ..MeshFilter::default()
        };
        let (meshes, health) = discover_round(&relays, &filter, DEADLINE).await.unwrap();
        // The listing it sent is not the target, and it never finished: an
        // empty result here is an incomplete search, not absence.
        assert!(meshes.is_empty());
        assert!(health.any_relay_answered());
        assert!(!health.any_query_completed());
    }

    #[tokio::test]
    async fn closed_and_hung_relays_are_silent_and_the_round_ends_on_time() {
        let relays = vec![
            scripted_relay(Script::Closed).await,
            scripted_relay(Script::Hang).await,
        ];
        let started = tokio::time::Instant::now();
        let (_, health) = discover_round(&relays, &MeshFilter::default(), DEADLINE)
            .await
            .unwrap();
        assert!(started.elapsed() < DEADLINE + Duration::from_secs(1));
        assert!(!health.any_relay_answered());
    }

    #[tokio::test]
    async fn events_sent_after_eose_are_not_collected() {
        let relays = vec![
            scripted_relay(Script::Answer {
                stored: vec![listing("stored")],
                live: vec![listing("live")],
            })
            .await,
        ];
        let (meshes, health) = discover_round(&relays, &MeshFilter::default(), DEADLINE)
            .await
            .unwrap();
        let names: Vec<_> = meshes
            .iter()
            .filter_map(|m| m.listing.name.as_deref())
            .collect();
        assert_eq!(names, ["stored"]);
        assert_eq!(health.queries_completed, 1);
    }
}
