//! Fairness of scarce big-tier access across a simulated fleet.
//!
//! `fleet_sim_tests.rs` answers "how WIDE is the committee?" — a property of
//! one origin's view. This file answers a different question: when a fleet has
//! a few big machines and many small ones, and many origins are all asking for
//! the big tier at once, **how is that scarce capacity shared?**
//!
//! Replica choice is `hosts_for_model`'s rendezvous hash over
//! `(origin_id, peer_id)` (`mesh/peer_state.rs`). That gives per-origin
//! stickiness (good for KV reuse) and statistical spread across origins, with
//! no coordination. What it does *not* give is any response to congestion:
//! nothing in gossip carries queue depth or inflight count, so a saturated big
//! replica keeps ranking `Healthy` until its own host-activity policy
//! deprioritizes it.
//!
//! These tests measure the resulting distribution rather than asserting a
//! target, except where a property is genuinely load-bearing.
//!
//! Origin count is deliberately small (32). Each simulated origin is a real
//! `mesh::Node`, which binds sockets, and a larger sweep exhausts the file
//! descriptor limit when the whole crate's tests run in parallel. A 300-origin
//! run measured locally gives worst/ideal 1.05-1.28 across 2/4/8 replicas; the
//! committed size is enough to pin the *properties* (every replica served, a
//! deprioritized replica sheds all first-choice traffic) without pinning a
//! noisy ratio.

use super::fleet_sim_tests::{BIG_MODELS, SMALL_MODELS, fleet_peer, fleet_peer_with_health};
use crate::mesh;
use std::collections::BTreeMap;

/// One origin node that can see the whole fleet. Distinct origins get distinct
/// endpoint ids, which is what makes the rendezvous hash spread them.
async fn origin_with_fleet(peers: &[mesh::PeerInfo]) -> mesh::Node {
    let node = mesh::Node::new_for_tests(mesh::NodeRole::Client)
        .await
        .expect("test node");
    for peer in peers {
        node.insert_test_peer(peer.clone()).await;
    }
    node
}

/// Build `big` replicas of one big model plus `small` replicas of one small
/// model, as distinct peers.
fn bimodal_fleet(big: u32, small: u32) -> Vec<mesh::PeerInfo> {
    let mut peers = Vec::new();
    let mut seed = 1u32;
    for _ in 0..big {
        peers.push(fleet_peer(seed, BIG_MODELS[0]));
        seed += 1;
    }
    for _ in 0..small {
        peers.push(fleet_peer(seed, SMALL_MODELS[0]));
        seed += 1;
    }
    peers
}

/// Count, per big replica, how many distinct short-lived origins would send it
/// their first-choice big-tier request. Each node is explicitly closed before
/// the next is created so this simulation does not accumulate bound sockets.
async fn first_choice_histogram(
    peers: &[mesh::PeerInfo],
    model: &str,
    origins: usize,
) -> BTreeMap<String, usize> {
    let mut hist: BTreeMap<String, usize> = BTreeMap::new();
    for _ in 0..origins {
        let node = origin_with_fleet(peers).await;
        let hosts = node.hosts_for_model(model).await;
        if let Some(first) = hosts.first() {
            *hist.entry(first.fmt_short().to_string()).or_default() += 1;
        }
        node.close_endpoint().await;
    }
    hist
}

async fn first_choice_histogram_for_origins(
    origins: &[mesh::Node],
    model: &str,
) -> BTreeMap<String, usize> {
    let mut hist = BTreeMap::new();
    for node in origins {
        let hosts = node.hosts_for_model(model).await;
        if let Some(first) = hosts.first() {
            *hist.entry(first.fmt_short().to_string()).or_default() += 1;
        }
    }
    hist
}

async fn origins_with_fleet(peers: &[mesh::PeerInfo], count: usize) -> Vec<mesh::Node> {
    let mut origins = Vec::with_capacity(count);
    for _ in 0..count {
        origins.push(origin_with_fleet(peers).await);
    }
    origins
}

async fn close_origins(origins: &[mesh::Node]) {
    for node in origins {
        node.close_endpoint().await;
    }
}

fn spread_summary(hist: &BTreeMap<String, usize>, origins: usize) -> (usize, usize, f64) {
    let counts: Vec<usize> = hist.values().copied().collect();
    let min = counts.iter().copied().min().unwrap_or(0);
    let max = counts.iter().copied().max().unwrap_or(0);
    let ideal = origins as f64 / counts.len().max(1) as f64;
    (min, max, max as f64 / ideal)
}

/// Scarce big tier, many origins: how lumpy is the allocation?
///
/// Measurement with one hard semantic floor: rendezvous hashing must spread
/// first choices beyond a single replica. Exact coverage is probabilistic for
/// a bounded sample, especially with eight replicas.
#[tokio::test]
async fn scarce_big_tier_spreads_across_origins() {
    let origins = 32usize;
    for big in [2u32, 4, 8] {
        let peers = bimodal_fleet(big, 40);
        let hist = first_choice_histogram(&peers, BIG_MODELS[0].name, origins).await;
        let (min, max, worst_vs_ideal) = spread_summary(&hist, origins);
        tracing::debug!(
            "big_replicas={big:>2} origins={origins} distinct_first_choices={n} \
             min={min} max={max} worst/ideal={worst_vs_ideal:.2}",
            n = hist.len()
        );
        assert!(
            hist.len() > 1,
            "rendezvous hashing must spread first-choice traffic across replicas: {hist:?}"
        );
    }
}

/// The load-blindness this file exists to document.
///
/// A big replica that is *saturated with inference* looks identical to an idle
/// one from every other node's perspective: `PeerInfo` carries no inflight or
/// queue-depth field, and `advertised_model_throughput` is a historical
/// capability hint (see the comment at `pool.rs`, "not live load pressure").
/// The only signal that moves a replica down the ordering is
/// `InferenceAdmissionState`, which `runtime/activity_policy.rs` derives from
/// **host activity**, not from inference load.
///
/// So this test asserts the current, deliberate behavior: busy-ness alone
/// changes nothing. If a future change adds a live load hint, this test should
/// be updated to show the ordering responding to it.
#[tokio::test]
async fn inference_load_is_invisible_to_replica_choice() {
    use crate::proto::node::InferenceAdmissionState;

    // ONE origin for both observations: the rendezvous hash is keyed on the
    // origin's endpoint id, so a fresh origin would reorder replicas for a
    // reason that has nothing to do with load.
    let node = mesh::Node::new_for_tests(mesh::NodeRole::Client)
        .await
        .expect("test node");

    let flat: Vec<mesh::PeerInfo> = (1..=3)
        .map(|seed| {
            fleet_peer_with_health(
                seed,
                BIG_MODELS[0],
                Some(InferenceAdmissionState::Accepting),
                Some(90_000),
            )
        })
        .collect();
    for peer in &flat {
        node.insert_test_peer(peer.clone()).await;
    }
    let baseline = node.hosts_for_model(BIG_MODELS[0].name).await;

    // Now make the origin's own first choice look maximally unattractive on
    // every signal that actually crosses the wire short of admission state:
    // near-zero advertised throughput.
    let slowest = baseline[0];
    let mut hobbled = flat
        .iter()
        .find(|peer| peer.id == slowest)
        .expect("first-choice peer")
        .clone();
    hobbled.advertised_model_throughput = vec![crate::network::metrics::ModelThroughputHint {
        model_name: BIG_MODELS[0].name.to_string(),
        avg_tokens_per_second_milli: 1_000,
        throughput_samples: 64,
    }];
    node.insert_test_peer(hobbled).await;

    let after = node.hosts_for_model(BIG_MODELS[0].name).await;
    assert_eq!(
        baseline, after,
        "replica ordering is rendezvous-hash only; a 90x throughput drop does not move it"
    );
}

/// The one lever that *does* work today, quantified: when a hot replica
/// deprioritizes itself, how much of the fleet's first-choice traffic moves?
///
/// This is the fallback path a saturated big box has, and it is all-or-nothing
/// — the replica goes to the back of the ordering for every origin at once.
#[tokio::test]
async fn deprioritizing_a_hot_replica_moves_all_of_its_traffic() {
    use crate::proto::node::InferenceAdmissionState;

    let origins = 8usize;
    let mut peers: Vec<mesh::PeerInfo> = (1..=4)
        .map(|seed| {
            fleet_peer_with_health(
                seed,
                BIG_MODELS[0],
                Some(InferenceAdmissionState::Accepting),
                None,
            )
        })
        .collect();

    let origin_nodes = origins_with_fleet(&peers, origins).await;
    let before = first_choice_histogram_for_origins(&origin_nodes, BIG_MODELS[0].name).await;
    let (_, max, _) = spread_summary(&before, origins);
    let hottest = before
        .iter()
        .max_by_key(|(_, count)| **count)
        .map(|(id, _)| id.clone())
        .expect("a hottest replica");
    tracing::debug!("before: {before:?}  hottest={hottest} max={max}");

    for peer in peers.iter_mut() {
        if peer.id.fmt_short().to_string() == hottest {
            peer.inference_admission_state = Some(InferenceAdmissionState::AcceptingDeprioritized);
            for node in &origin_nodes {
                node.insert_test_peer(peer.clone()).await;
            }
        }
    }

    let after = first_choice_histogram_for_origins(&origin_nodes, BIG_MODELS[0].name).await;
    tracing::debug!("after:  {after:?}");
    assert_eq!(
        after.get(&hottest),
        None,
        "a deprioritized replica must take no first-choice traffic while healthy peers exist"
    );
    assert_eq!(
        after.values().sum::<usize>(),
        origins,
        "all origins still reach a big replica — deprioritize sheds load, never capacity"
    );
    close_origins(&origin_nodes).await;
}
