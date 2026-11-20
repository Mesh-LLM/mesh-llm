//! Process-local in-flight counts used to spread concurrent new sessions
//! across equivalent inference targets.

use crate::inference::election;
use crate::network::router::PlacementWeight;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

#[derive(Clone, Debug, Hash, PartialEq, Eq)]
struct ReservationKey {
    model: String,
    target: election::InferenceTarget,
}

type ReservationCounts = Arc<Mutex<HashMap<ReservationKey, usize>>>;

fn decrement(counts: &mut HashMap<ReservationKey, usize>, key: &ReservationKey) {
    if let Some(count) = counts.get_mut(key) {
        *count = count.saturating_sub(1);
        if *count == 0 {
            counts.remove(key);
        }
    }
}

/// Per-process counts of requests currently in flight to each `(model,
/// target)` pair. Entries exist only while a [`RoutingReservation`] guard is
/// alive, so the map stays bounded by the number of in-flight requests.
#[derive(Clone, Default)]
pub(crate) struct RoutingReservations {
    counts: ReservationCounts,
    /// Round-robin position per model for equal-weight two-choice first
    /// draws, so traffic for one model never moves another model's turn.
    round_robin: Arc<Mutex<HashMap<String, u64>>>,
}

impl RoutingReservations {
    /// Choose and reserve a target in one critical section. A pinned
    /// `preferred` (`affinity_applied`: session/cache affinity, or a
    /// [`Self::two_choice`] draw that already weighed load per unit of
    /// capacity) is reserved as-is; otherwise the current picker result is the
    /// stable tie-breaker among targets with the fewest local in-flight
    /// requests.
    ///
    /// `spread_limit` bounds spreading to the leading `candidates[..limit]`,
    /// which callers set to the run of throughput-equivalent targets at the
    /// head of the ranked candidate order. Reservation pressure only trades
    /// off targets the measurements cannot distinguish; it never redirects a
    /// request to a lower-ranked (measurably slower or smaller-context)
    /// target, and a preferred target outside the leading run is reserved
    /// as-is.
    pub(crate) fn reserve(
        &self,
        model: &str,
        candidates: &[election::InferenceTarget],
        spread_limit: usize,
        preferred: &election::InferenceTarget,
        affinity_applied: bool,
    ) -> Option<(election::InferenceTarget, RoutingReservation)> {
        if candidates.is_empty() {
            return None;
        }
        let spread_limit = spread_limit.clamp(1, candidates.len());
        let mut counts = self.counts.lock().unwrap();
        let count = |counts: &HashMap<ReservationKey, usize>,
                     target: &election::InferenceTarget| {
            counts
                .get(&ReservationKey {
                    model: model.to_string(),
                    target: target.clone(),
                })
                .copied()
                .unwrap_or(0)
        };
        let preferred_index = candidates
            .iter()
            .position(|candidate| candidate == preferred)
            .unwrap_or(0);
        let target = if affinity_applied || preferred_index >= spread_limit {
            candidates[preferred_index].clone()
        } else {
            let spread = &candidates[..spread_limit];
            let minimum = spread
                .iter()
                .map(|candidate| count(&counts, candidate))
                .min()
                .unwrap_or(0);
            (0..spread.len())
                .map(|offset| (preferred_index + offset) % spread.len())
                .find(|index| count(&counts, &spread[*index]) == minimum)
                .map(|index| spread[index].clone())
                .unwrap_or_else(|| candidates[preferred_index].clone())
        };
        let key = ReservationKey {
            model: model.to_string(),
            target: target.clone(),
        };
        *counts.entry(key.clone()).or_insert(0) += 1;
        drop(counts);
        Some((
            target,
            RoutingReservation {
                counts: Arc::clone(&self.counts),
                key,
            },
        ))
    }

    /// Power-of-two-choices host pick for requests without affinity.
    ///
    /// Each candidate's [`PlacementWeight`] carries its capacity, RTT and
    /// dense rank class; list position carries no meaning here, so reordering
    /// equivalent candidates cannot change which moves are allowed. The first
    /// draw is proportional to capacity × locality when the draw weights
    /// differ, and round-robin when they are all equal, so equal hosts share
    /// evenly and a faster host carries proportionally more. A second draw
    /// from the remaining candidates may replace the first when it carries
    /// fewer local in-flight requests per unit of capacity, or on a load tie
    /// when it is near and the first is far — but never when its rank class is worse than the first's:
    /// pressure never redirects a request to a measurably slower or
    /// smaller-context target. One load comparison stops a gateway herding
    /// onto a host whose gossiped speed is stale.
    ///
    /// Targets `spillover` marks (penalized or deprioritized hosts) are drawn
    /// only when every candidate is one. The pick is reserved under the same
    /// lock that read the loads, so concurrent requests cannot all see the
    /// same idle host and pile onto it. `None` only when `candidates` is empty.
    pub(crate) fn reserve_two_choice(
        &self,
        model: &str,
        candidates: &[election::InferenceTarget],
        spillover: impl Fn(&election::InferenceTarget) -> bool,
        weight_of: impl Fn(&election::InferenceTarget) -> PlacementWeight,
        seed: u64,
    ) -> Option<(election::InferenceTarget, RoutingReservation)> {
        let preferred: Vec<election::InferenceTarget> = candidates
            .iter()
            .filter(|candidate| !spillover(candidate))
            .cloned()
            .collect();
        let candidates = if preferred.is_empty() {
            candidates
        } else {
            preferred.as_slice()
        };
        if candidates.is_empty() {
            return None;
        }
        let mut counts = self.counts.lock().unwrap();
        let winner = self.two_choice_index(model, candidates, weight_of, seed, &counts);
        let key = ReservationKey {
            model: model.to_string(),
            target: candidates[winner].clone(),
        };
        *counts.entry(key.clone()).or_insert(0) += 1;
        drop(counts);
        Some((
            candidates[winner].clone(),
            RoutingReservation {
                counts: Arc::clone(&self.counts),
                key,
            },
        ))
    }

    /// The two-choice pick alone, released at once; for tests that tally draws.
    #[cfg(test)]
    pub(crate) fn two_choice(
        &self,
        model: &str,
        candidates: &[election::InferenceTarget],
        weight_of: impl Fn(&election::InferenceTarget) -> PlacementWeight,
        seed: u64,
    ) -> Option<election::InferenceTarget> {
        self.reserve_two_choice(model, candidates, |_| false, weight_of, seed)
            .map(|(target, _reservation)| target)
    }

    fn two_choice_index(
        &self,
        model: &str,
        candidates: &[election::InferenceTarget],
        weight_of: impl Fn(&election::InferenceTarget) -> PlacementWeight,
        seed: u64,
        counts: &HashMap<ReservationKey, usize>,
    ) -> usize {
        if candidates.len() == 1 {
            return 0;
        }
        let placements: Vec<PlacementWeight> = candidates.iter().map(&weight_of).collect();
        let mut rng = crate::network::router::SplitMix64::new(seed);
        let mut weights: Vec<f64> = placements
            .iter()
            .map(|placement| {
                let weight = placement.draw_weight();
                if weight.is_finite() && weight > 0.0 {
                    weight
                } else {
                    f64::MIN_POSITIVE
                }
            })
            .collect();
        let all_equal = weights
            .iter()
            .all(|weight| (weight - weights[0]).abs() <= f64::EPSILON * weights[0]);
        let first = if all_equal {
            let mut turns = self.round_robin.lock().unwrap();
            let turn = turns.entry(model.to_string()).or_insert(0);
            let first = *turn as usize % candidates.len();
            *turn = turn.wrapping_add(1);
            first
        } else {
            crate::network::router::weighted_index(&weights, rng.next_f64())
        };
        weights[first] = 0.0;
        let second = crate::network::router::weighted_index(&weights, rng.next_f64());
        if second == first || placements[second].rank > placements[first].rank {
            // Pressure and locality may not promote a lower-ranked target.
            tracing::debug!(
                model,
                first = ?candidates[first],
                second = ?candidates[second],
                first_capacity = placements[first].capacity,
                second_capacity = placements[second].capacity,
                first_rank = placements[first].rank,
                second_rank = placements[second].rank,
                "two-choice host pick: second draw not eligible"
            );
            return first;
        }
        let load = |index: usize| {
            counts
                .get(&ReservationKey {
                    model: model.to_string(),
                    target: candidates[index].clone(),
                })
                .copied()
                .unwrap_or(0)
        };
        // Load is in-flight requests per unit of capacity, not a raw count: a
        // host measured at 3× the speed is only "busier" once it holds more
        // than 3× the requests, so a synchronized burst keeps its capacity
        // share instead of equalising in-flight across unequal hosts. Equal
        // capacities reduce to the plain count comparison.
        let relative_load = |index: usize| {
            let capacity = placements[index].capacity;
            let capacity = if capacity.is_finite() && capacity > 0.0 {
                capacity
            } else {
                f64::MIN_POSITIVE
            };
            load(index) as f64 / capacity
        };
        // Locality breaks a load tie only across a material gap (far = the
        // locality factor is below 0.75, about >175 ms RTT); among nearby
        // hosts the first draw stands, so a crowd of idle clients does not
        // all converge on whichever host is marginally closest.
        let key = |index: usize| (relative_load(index), placements[index].is_far());
        let winner = if key(second) < key(first) {
            second
        } else {
            first
        };
        tracing::debug!(
            model,
            first = ?candidates[first],
            second = ?candidates[second],
            winner = ?candidates[winner],
            first_capacity = placements[first].capacity,
            second_capacity = placements[second].capacity,
            first_rank = placements[first].rank,
            second_rank = placements[second].rank,
            first_in_flight = load(first),
            second_in_flight = load(second),
            "two-choice host pick"
        );
        winner
    }

    /// Total in-flight reservations across all models and targets.
    pub(crate) fn active_total(&self) -> usize {
        self.counts.lock().unwrap().values().sum()
    }

    #[cfg(test)]
    fn active_count(&self, model: &str, target: &election::InferenceTarget) -> usize {
        self.counts
            .lock()
            .unwrap()
            .get(&ReservationKey {
                model: model.to_string(),
                target: target.clone(),
            })
            .copied()
            .unwrap_or(0)
    }
}

/// RAII guard for one in-flight request. Dropping it — on success, failure,
/// cancellation, or panic — releases the reservation.
pub(crate) struct RoutingReservation {
    counts: ReservationCounts,
    key: ReservationKey,
}

impl RoutingReservation {
    /// Move the reservation to the failover target so retries keep the
    /// in-flight count on the target actually serving the request.
    pub(crate) fn transfer_to(&mut self, target: &election::InferenceTarget) {
        if self.key.target == *target {
            return;
        }
        let mut counts = self.counts.lock().unwrap();
        decrement(&mut counts, &self.key);
        self.key.target = target.clone();
        *counts.entry(self.key.clone()).or_insert(0) += 1;
    }
}

impl Drop for RoutingReservation {
    fn drop(&mut self) {
        decrement(&mut self.counts.lock().unwrap(), &self.key);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;
    use std::sync::Barrier;

    fn local(port: u16) -> election::InferenceTarget {
        election::InferenceTarget::Local(port)
    }

    fn concurrent_burst(
        targets: Vec<election::InferenceTarget>,
        requests: usize,
    ) -> BTreeMap<u16, usize> {
        let reservations = RoutingReservations::default();
        let barrier = Arc::new(Barrier::new(requests));
        let selected = Arc::new(Mutex::new(Vec::new()));
        std::thread::scope(|scope| {
            for _ in 0..requests {
                let reservations = reservations.clone();
                let barrier = Arc::clone(&barrier);
                let selected = Arc::clone(&selected);
                let targets = targets.clone();
                scope.spawn(move || {
                    let (target, reservation) = reservations
                        .reserve("model", &targets, targets.len(), &targets[0], false)
                        .expect("reservation");
                    selected.lock().unwrap().push(target);
                    barrier.wait();
                    drop(reservation);
                });
            }
        });
        assert_eq!(reservations.active_total(), 0);
        let mut counts = BTreeMap::new();
        for target in selected.lock().unwrap().drain(..) {
            let election::InferenceTarget::Local(port) = target else {
                panic!("expected local target");
            };
            *counts.entry(port).or_insert(0) += 1;
        }
        counts
    }

    fn placement(capacity: f64, rtt_ms: Option<u32>, rank: usize) -> PlacementWeight {
        PlacementWeight {
            capacity,
            rtt_ms,
            rank,
        }
    }

    fn two_choice_tally(
        reservations: &RoutingReservations,
        targets: &[election::InferenceTarget],
        weight_of: impl Fn(&election::InferenceTarget) -> PlacementWeight,
        draws: u64,
    ) -> BTreeMap<u16, usize> {
        let mut counts = BTreeMap::new();
        for seed in 0..draws {
            let election::InferenceTarget::Local(port) = reservations
                .two_choice("model", targets, &weight_of, seed.wrapping_mul(0x9E37_79B9))
                .expect("pick")
            else {
                panic!("expected local target");
            };
            *counts.entry(port).or_insert(0) += 1;
        }
        counts
    }

    #[test]
    fn two_choice_follows_capacity_when_idle() {
        // 80 vs 20 tok/s, nobody busy: the fast host carries about 80 %.
        let reservations = RoutingReservations::default();
        let targets = vec![local(1), local(2)];
        let weight = |target: &election::InferenceTarget| match target {
            election::InferenceTarget::Local(1) => placement(80.0, None, 0),
            _ => placement(20.0, None, 1),
        };
        let counts = two_choice_tally(&reservations, &targets, weight, 2000);
        let fast = counts[&1] as f64 / 2000.0;
        assert!((0.74..0.86).contains(&fast), "fast share {fast}");
    }

    #[test]
    fn two_choice_moves_off_the_busy_host_regardless_of_list_position() {
        // Three equivalent hosts (one rank class). Whichever list slot the
        // busy host occupies, every pair that draws it moves to the idle
        // partner, and the idle hosts split the traffic.
        for busy_slot in 0..3usize {
            let reservations = RoutingReservations::default();
            let targets = vec![local(1), local(2), local(3)];
            let (_, _held) = reservations
                .reserve("model", &targets, 1, &targets[busy_slot], true)
                .expect("reserve");
            let counts =
                two_choice_tally(&reservations, &targets, |_| placement(50.0, None, 0), 300);
            let busy_port = busy_slot as u16 + 1;
            assert_eq!(
                counts.get(&busy_port),
                None,
                "busy host in slot {busy_slot} must lose every pair: {counts:?}"
            );
            for (port, count) in &counts {
                assert!(
                    (120..=180).contains(count),
                    "idle host {port} should take about half: {counts:?}"
                );
            }
        }
    }

    #[test]
    fn pinned_two_choice_pick_survives_reservation() {
        // Same rank class, 3:1 capacity, fast host holding 2 against the slow
        // host's 1: per unit of capacity the fast host is the lighter one, so
        // the draw keeps it. Reserving that pick within an equivalent prefix
        // of 2 must not hand it to the slow host for having the fewer raw
        // in-flight requests; only an unpinned round-robin default spreads.
        let targets = vec![local(1), local(2)];
        let weight = |target: &election::InferenceTarget| match target {
            election::InferenceTarget::Local(1) => placement(75.0, None, 0),
            _ => placement(25.0, None, 0),
        };
        let reservations = RoutingReservations::default();
        let held = vec![
            reservations.reserve("model", &targets, 2, &targets[0], true),
            reservations.reserve("model", &targets, 2, &targets[0], true),
            reservations.reserve("model", &targets, 2, &targets[1], true),
        ];
        for seed in 0..200u64 {
            let picked = reservations
                .two_choice("model", &targets, weight, seed.wrapping_mul(0x9E37_79B9))
                .expect("pick");
            assert_eq!(picked, targets[0], "seed {seed}");
            let (reserved, guard) = reservations
                .reserve("model", &targets, 2, &picked, true)
                .expect("reserve");
            assert_eq!(reserved, picked, "pinned pick re-spread at seed {seed}");
            drop(guard);
        }
        let (spread, _guard) = reservations
            .reserve("model", &targets, 2, &targets[0], false)
            .expect("reserve");
        assert_eq!(
            spread, targets[1],
            "an unpinned default still spreads by raw count"
        );
        drop(held);
    }

    #[test]
    fn two_choice_compares_load_per_unit_of_capacity() {
        // Host 1 is measured at 3× host 2 (same rank class: say both have
        // samples that round to one class in the caller). Holding 2 requests
        // against 1 leaves it the *less* loaded host per unit of capacity
        // (2/3 < 1/1), so every pair resolves to it; holding 4 against 1
        // (4/3 > 1/1) hands the pairs that draw host 2 second over to host 2.
        let targets = vec![local(1), local(2)];
        let weight = |target: &election::InferenceTarget| match target {
            election::InferenceTarget::Local(1) => placement(75.0, None, 0),
            _ => placement(25.0, None, 0),
        };
        let reservations = RoutingReservations::default();
        let mut held = Vec::new();
        for _ in 0..2 {
            held.push(reservations.reserve("model", &targets, 2, &targets[0], true));
        }
        held.push(reservations.reserve("model", &targets, 2, &targets[1], true));
        let counts = two_choice_tally(&reservations, &targets, weight, 400);
        assert_eq!(
            counts.get(&2),
            None,
            "a fast host under proportionally lighter load keeps every pair: {counts:?}"
        );
        for _ in 0..2 {
            held.push(reservations.reserve("model", &targets, 2, &targets[0], true));
        }
        let counts = two_choice_tally(&reservations, &targets, weight, 400);
        let slow = counts.get(&2).copied().unwrap_or(0);
        assert!(
            (300..=400).contains(&slow),
            "once the fast host is busier per unit of capacity the pairs move: {counts:?}"
        );
        drop(held);
    }

    #[test]
    fn two_choice_never_promotes_a_worse_rank_class_under_pressure() {
        // Host 1 forms rank class 0, host 2 class 1 (measurably slower). With
        // a request in flight on host 1 and none on host 2, host 1 still keeps
        // every first draw that lands on it; host 2 gets only its capacity
        // share, and its own draws may move up to the better class only when
        // that class is less loaded — which it is not here.
        let reservations = RoutingReservations::default();
        let targets = vec![local(1), local(2)];
        let (_, _held) = reservations
            .reserve("model", &targets, 1, &targets[0], true)
            .expect("reserve");
        let weight = |target: &election::InferenceTarget| match target {
            election::InferenceTarget::Local(1) => placement(80.0, None, 0),
            _ => placement(20.0, None, 1),
        };
        let counts = two_choice_tally(&reservations, &targets, weight, 2000);
        let fast = counts[&1] as f64 / 2000.0;
        assert!(
            (0.74..0.86).contains(&fast),
            "fast share {fast}: {counts:?}"
        );
        // Reversed list, same classes: identical policy.
        let reservations = RoutingReservations::default();
        let targets = vec![local(2), local(1)];
        let (_, _held) = reservations
            .reserve("model", &targets, 1, &targets[1], true)
            .expect("reserve");
        let counts = two_choice_tally(&reservations, &targets, weight, 2000);
        let fast = counts[&1] as f64 / 2000.0;
        assert!(
            (0.74..0.86).contains(&fast),
            "fast share {fast}: {counts:?}"
        );
        // Idle better class: the slower host's draws move up to it.
        let reservations = RoutingReservations::default();
        let (_, _held) = reservations
            .reserve("model", &targets, 1, &targets[0], true)
            .expect("reserve");
        let counts = two_choice_tally(&reservations, &targets, weight, 2000);
        assert_eq!(counts.get(&2), None, "{counts:?}");
    }

    #[test]
    fn two_choice_breaks_a_load_tie_toward_the_near_host_only_across_a_real_gap() {
        let reservations = RoutingReservations::default();
        let targets = vec![local(1), local(2)];
        // Equal capacity and rank, 20 ms vs 250 ms (relay-grade): the near
        // host wins every idle pair, whichever list slot it occupies.
        let counts = two_choice_tally(
            &reservations,
            &targets,
            |target| match target {
                election::InferenceTarget::Local(2) => placement(50.0, Some(250), 0),
                _ => placement(50.0, Some(20), 0),
            },
            200,
        );
        assert_eq!(
            counts.get(&2),
            None,
            "far host must lose idle pairs: {counts:?}"
        );
        let counts = two_choice_tally(
            &reservations,
            &[local(2), local(1)],
            |target| match target {
                election::InferenceTarget::Local(2) => placement(50.0, Some(250), 0),
                _ => placement(50.0, Some(20), 0),
            },
            200,
        );
        assert_eq!(
            counts.get(&2),
            None,
            "far host listed first must still lose idle pairs: {counts:?}"
        );
        // 20 ms vs 60 ms: both near, the round-robin first draw stands.
        let counts = two_choice_tally(
            &reservations,
            &targets,
            |target| match target {
                election::InferenceTarget::Local(2) => placement(50.0, Some(60), 0),
                _ => placement(50.0, Some(20), 0),
            },
            400,
        );
        assert!(
            counts[&1] > 120 && counts[&2] > 120,
            "nearby hosts share: {counts:?}"
        );
    }

    #[test]
    fn two_choice_single_candidate_and_empty() {
        let reservations = RoutingReservations::default();
        assert_eq!(
            reservations.two_choice("model", &[local(7)], |_| placement(1.0, None, 0), 1),
            Some(local(7))
        );
        assert_eq!(
            reservations.two_choice("model", &[], |_| placement(1.0, None, 0), 1),
            None
        );
    }

    #[test]
    fn two_choice_round_robin_turns_are_per_model() {
        // Interleaved models must not share one turn counter: with a shared
        // counter, model-a would always land on even turns, so on one host.
        let reservations = RoutingReservations::default();
        let targets = [local(1), local(2)];
        let mut picks = Vec::new();
        for _ in 0..4 {
            for model in ["model-a", "model-b"] {
                let target = reservations
                    .two_choice(model, &targets, |_| placement(1.0, None, 0), 7)
                    .expect("pick");
                if model == "model-a" {
                    picks.push(target);
                }
            }
        }
        assert_eq!(picks, [local(1), local(2), local(1), local(2)]);
    }

    #[test]
    fn reserve_two_choice_holds_its_pick_and_keeps_spillover_last() {
        let reservations = RoutingReservations::default();
        let targets = [local(1), local(2)];
        let spill = |target: &election::InferenceTarget| *target == local(2);
        let mut held = Vec::new();
        for seed in 0..6 {
            let (target, reservation) = reservations
                .reserve_two_choice("model", &targets, spill, |_| placement(1.0, None, 0), seed)
                .expect("pick");
            assert_eq!(
                target,
                local(1),
                "a spillover host waits while another remains"
            );
            held.push(reservation);
        }
        assert_eq!(reservations.active_count("model", &local(1)), 6);
        drop(held);
        assert_eq!(reservations.active_total(), 0);
        let (target, _reservation) = reservations
            .reserve_two_choice("model", &[local(2)], spill, |_| placement(1.0, None, 0), 1)
            .expect("spillover still serves when it is all there is");
        assert_eq!(target, local(2));
    }

    #[test]
    fn concurrent_new_requests_spread_across_two_targets() {
        assert_eq!(
            concurrent_burst(vec![local(1), local(2)], 8),
            BTreeMap::from([(1, 4), (2, 4)])
        );
    }

    #[test]
    fn concurrent_new_requests_spread_across_three_targets() {
        assert_eq!(
            concurrent_burst(vec![local(1), local(2), local(3)], 9),
            BTreeMap::from([(1, 3), (2, 3), (3, 3)])
        );
    }

    #[test]
    fn single_target_routing_is_unchanged() {
        assert_eq!(
            concurrent_burst(vec![local(1)], 8),
            BTreeMap::from([(1, 8)])
        );
    }

    #[test]
    fn reservation_pressure_never_displaces_a_higher_ranked_target() {
        // Candidate order encodes measured throughput rank: target 1 is
        // materially faster and forms a rank tier of its own (spread limit 1).
        // Even with in-flight requests on the fast target and none on the
        // slow one, new sessions must keep going to the fast target.
        let reservations = RoutingReservations::default();
        let targets = vec![local(1), local(2)];
        let mut guards = Vec::new();
        for _ in 0..4 {
            let (selected, guard) = reservations
                .reserve("model", &targets, 1, &targets[0], false)
                .expect("reservation");
            assert_eq!(selected, targets[0]);
            guards.push(guard);
        }
        assert_eq!(reservations.active_count("model", &targets[0]), 4);
        assert_eq!(reservations.active_count("model", &targets[1]), 0);
    }

    #[test]
    fn reservation_pressure_spreads_only_within_the_equivalent_tier() {
        // Targets 1 and 2 are throughput-equivalent (spread limit 2); target 3
        // is a lower tier. Concurrent requests alternate between the first two
        // and never spill to the third.
        let reservations = RoutingReservations::default();
        let targets = vec![local(1), local(2), local(3)];
        let mut guards = Vec::new();
        let mut selections = Vec::new();
        for _ in 0..4 {
            let (selected, guard) = reservations
                .reserve("model", &targets, 2, &targets[0], false)
                .expect("reservation");
            selections.push(selected);
            guards.push(guard);
        }
        assert_eq!(reservations.active_count("model", &targets[0]), 2);
        assert_eq!(reservations.active_count("model", &targets[1]), 2);
        assert_eq!(reservations.active_count("model", &targets[2]), 0);
    }

    #[test]
    fn preferred_target_outside_the_spread_window_is_reserved_as_is() {
        // A sticky/round-robin pick may land past the equivalent tier; the
        // reservation must follow it rather than pull the request forward.
        let reservations = RoutingReservations::default();
        let targets = vec![local(1), local(2)];
        let (selected, _guard) = reservations
            .reserve("model", &targets, 1, &targets[1], false)
            .expect("reservation");
        assert_eq!(selected, targets[1]);
        assert_eq!(reservations.active_count("model", &targets[1]), 1);
    }

    #[test]
    fn established_affinity_ignores_reservation_pressure() {
        let reservations = RoutingReservations::default();
        let targets = vec![local(1), local(2)];
        let (_, _first) = reservations
            .reserve("model", &targets, targets.len(), &targets[0], false)
            .expect("first reservation");
        let (selected, _sticky) = reservations
            .reserve("model", &targets, targets.len(), &targets[0], true)
            .expect("sticky reservation");

        assert_eq!(selected, targets[0]);
        assert_eq!(reservations.active_count("model", &targets[0]), 2);
        assert_eq!(reservations.active_count("model", &targets[1]), 0);
    }

    #[test]
    fn reservation_pressure_is_isolated_per_model() {
        let reservations = RoutingReservations::default();
        let targets = vec![local(1), local(2)];
        let (_, _model_a) = reservations
            .reserve("model-a", &targets, targets.len(), &targets[0], false)
            .expect("model-a reservation");
        let (selected, _model_b) = reservations
            .reserve("model-b", &targets, targets.len(), &targets[0], false)
            .expect("model-b reservation");

        assert_eq!(selected, targets[0]);
        assert_eq!(reservations.active_count("model-a", &targets[0]), 1);
        assert_eq!(reservations.active_count("model-b", &targets[0]), 1);
    }

    #[test]
    fn failover_transfers_and_drop_releases_reservation() {
        let reservations = RoutingReservations::default();
        let targets = vec![local(1), local(2)];
        let (_, mut reservation) = reservations
            .reserve("model", &targets, targets.len(), &targets[0], false)
            .expect("reservation");

        reservation.transfer_to(&targets[1]);
        assert_eq!(reservations.active_count("model", &targets[0]), 0);
        assert_eq!(reservations.active_count("model", &targets[1]), 1);
        drop(reservation);

        assert_eq!(reservations.active_total(), 0);
    }
}
