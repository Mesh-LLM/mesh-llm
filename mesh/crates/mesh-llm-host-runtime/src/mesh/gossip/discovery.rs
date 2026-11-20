use super::{EndpointAddr, EndpointId, Node, PeerAnnouncement};
use crate::mesh::{
    DEAD_PEER_TTL, MembershipState, NodeRole, PEER_CONNECT_AND_GOSSIP_TIMEOUT,
    PendingConnectionReservation,
};
use iroh::endpoint::Connection;
use mesh_llm_membership::announcements::{
    peer_is_idle_transitive_client, version_allowed_for_rebroadcast,
};
use mesh_llm_membership::connection_reservation::{
    PendingConnectionAttemptOwner, PendingConnectionWaiter,
};
use std::collections::VecDeque;

// Bounds the memory that relayed announcements can claim, with headroom for
// large meshes. One worker dials these in turn, so the limit does not raise
// the number of concurrent dials. A dropped peer is queued again when a
// later discovery-enabled gossip exchange mentions it.
const MAX_DISCOVERY_CANDIDATES: usize = 4096;
const MAX_DISCOVERY_PATHS_WITH_FALLBACKS: usize = 16;

#[derive(Default)]
pub(in crate::mesh) struct PendingDiscovery {
    // The front retains its slot while the worker attempts it.
    candidates: VecDeque<DiscoveryCandidate>,
    active: Option<ActiveDiscovery>,
    next_generation: u64,
}

enum ActiveDiscovery {
    Selected(EndpointAddr, u64),
    Dialing(EndpointAddr, u64),
    // The worker parked the front candidate. Its wait may unpark it before
    // the worker finishes, so finishing this round must leave it queued.
    Parked,
}

struct DiscoveryCandidate {
    addr: EndpointAddr,
    generation: u64,
    // Inbound gossip only dials peers absent from the directory.
    inbound_role: Option<NodeRole>,
    // Another dial owns the peer. The worker skips it until that dial ends.
    parked: bool,
}

impl PendingDiscovery {
    pub(super) fn upsert(
        &mut self,
        addr: &EndpointAddr,
        discover: bool,
        inbound_role: Option<NodeRole>,
    ) {
        if let Some(index) = self
            .candidates
            .iter()
            .position(|item| item.addr.id == addr.id)
        {
            let candidate = &mut self.candidates[index];
            if inbound_role.is_none() && candidate.inbound_role.is_some() && !discover {
                // Directory ingestion makes an inbound candidate known.
                self.candidates.remove(index);
                return;
            }
            if candidate.inbound_role.is_some() != inbound_role.is_some() {
                // A source change starts with that source's accepted address set.
                candidate.addr.clone_from(addr);
                candidate.inbound_role = inbound_role;
                candidate.generation = self.next_generation;
                self.next_generation = self.next_generation.wrapping_add(1);
                return;
            }
            candidate.inbound_role = inbound_role;
            let already_dialing = matches!(&self.active, Some(ActiveDiscovery::Dialing(active, generation))
                if *generation == candidate.generation && addr.addrs.is_subset(&active.addrs));
            if already_dialing || addr.addrs.is_empty() {
                return;
            }
            let mut updated = addr.clone();
            // A poorer reply can add a new route without erasing useful hints.
            if updated.addrs.len() < candidate.addr.addrs.len() {
                for path in &candidate.addr.addrs {
                    if updated.addrs.len() >= MAX_DISCOVERY_PATHS_WITH_FALLBACKS {
                        break;
                    }
                    updated.addrs.insert(path.clone());
                }
            }
            candidate.addr = updated;
        } else if discover && self.candidates.len() < MAX_DISCOVERY_CANDIDATES {
            self.candidates.push_back(DiscoveryCandidate {
                addr: addr.clone(),
                generation: self.next_generation,
                inbound_role,
                parked: false,
            });
            self.next_generation = self.next_generation.wrapping_add(1);
        } else if discover {
            tracing::debug!(
                peer = %addr.id.fmt_short(),
                "Discovery queue is full; dropping candidate"
            );
        }
    }

    pub(super) fn withdraw(&mut self, id: EndpointId) {
        self.candidates.retain(|candidate| candidate.addr.id != id);
    }

    fn unpark(&mut self, id: EndpointId) {
        if let Some(candidate) = self
            .candidates
            .iter_mut()
            .find(|candidate| candidate.addr.id == id)
        {
            candidate.parked = false;
        }
    }

    fn finish(&mut self, failed: bool) {
        let (
            Some(
                ActiveDiscovery::Selected(attempted, generation)
                | ActiveDiscovery::Dialing(attempted, generation),
            ),
            Some(current),
        ) = (self.active.as_ref(), self.candidates.front())
        else {
            return;
        };
        if current.addr.id == attempted.id {
            if failed && (&current.addr != attempted || current.generation != *generation) {
                // Give a changed address its turn behind already waiting peers.
                self.candidates.rotate_left(1);
            } else if current.generation == *generation {
                self.candidates.pop_front();
            }
        }
    }

    fn begin_next(&mut self) -> Option<EndpointId> {
        // Parked candidates wait behind the others until their owner finishes.
        if let Some(index) = self.candidates.iter().position(|item| !item.parked) {
            self.candidates.rotate_left(index);
        }
        let candidate = self.candidates.front().filter(|item| !item.parked);
        self.active = candidate.map(|candidate| {
            ActiveDiscovery::Selected(candidate.addr.clone(), candidate.generation)
        });
        candidate.map(|candidate| candidate.addr.id)
    }

    fn address_for_dial(&mut self, id: EndpointId) -> Option<EndpointAddr> {
        let candidate = self
            .candidates
            .front()
            .filter(|candidate| candidate.addr.id == id)?;
        self.active = Some(ActiveDiscovery::Dialing(
            candidate.addr.clone(),
            candidate.generation,
        ));
        Some(candidate.addr.clone())
    }
}

fn recently_dead(state: &MembershipState, id: EndpointId) -> bool {
    state
        .dead_peers
        .get(&id)
        .is_some_and(|time| time.elapsed() < DEAD_PEER_TTL)
}

pub(in crate::mesh) struct ReservedDial {
    pub addr: EndpointAddr,
    pub owner: PendingConnectionAttemptOwner,
    pub cached: Option<Connection>,
}

impl Node {
    pub(super) async fn enqueue_inbound_discovery(
        &self,
        announcements: &[(EndpointAddr, PeerAnnouncement)],
    ) {
        {
            let my_role = self.role.lock().await.clone();
            let state = self.state.lock().await;
            let mut pending = self.peer_discovery.lock().await;
            for (addr, ann) in announcements {
                if Self::discovered_peer_is_filtered(addr.id, ann)
                    || !self.inbound_discovery_eligible(&state, addr.id, &my_role, &ann.role)
                {
                    continue;
                }
                pending.upsert(addr, true, Some(ann.role.clone()));
            }
        }
        self.start_peer_discovery().await;
    }

    fn inbound_discovery_eligible(
        &self,
        state: &MembershipState,
        id: EndpointId,
        my_role: &NodeRole,
        peer_role: &NodeRole,
    ) -> bool {
        !self.should_skip_discovered_peer(my_role, id, peer_role)
            && !self.local_endpoints.contains(id)
            && !state.peers.contains_key(&id)
            && (state.connections.contains_key(&id) || !recently_dead(state, id))
    }

    pub(in crate::mesh) async fn start_peer_discovery(&self) {
        let mut pending = self.peer_discovery.lock().await;
        if pending.active.is_some() || self.endpoint.is_closed() {
            return;
        }
        let Some(id) = pending.begin_next() else {
            return;
        };
        drop(pending);
        self.spawn_peer_discovery(id);
    }

    fn spawn_peer_discovery(&self, id: EndpointId) {
        let node = self.clone();
        tokio::spawn(async move { node.run_peer_discovery(id).await });
    }

    pub(in crate::mesh) async fn reserve_discovery_connection(
        &self,
        id: EndpointId,
    ) -> Option<ReservedDial> {
        let my_role = self.role.lock().await.clone();
        let mut state = self.state.lock().await;
        let mut pending = self.peer_discovery.lock().await;
        let candidate = pending
            .candidates
            .front()
            .filter(|candidate| candidate.addr.id == id)?;
        let cached = state.connections.get(&id).cloned();
        let directory = candidate.inbound_role.is_none();
        let eligible = if let Some(role) = candidate.inbound_role.as_ref() {
            // An earlier candidate may have kept this peer in the queue long
            // enough for a competing handshake to finish. Inbound hints are
            // only for peers outside the directory, so a peer that another
            // attempt is already reaching is left to that attempt.
            self.inbound_discovery_eligible(&state, id, &my_role, role)
                && !state.pending_connection_is_active(id)
        } else {
            !recently_dead(&state, id)
                && state
                    .peers
                    .get(&id)
                    .is_some_and(|peer| !self.should_skip_discovered_peer(&my_role, id, &peer.role))
        };
        if !eligible || self.endpoint.is_closed() {
            pending.candidates.pop_front();
            return None;
        }
        let owner = match state.reserve_pending_connection(id) {
            PendingConnectionReservation::Owner(owner) => owner,
            PendingConnectionReservation::Waiter(waiter) => {
                // Keep the candidate, and any newer address, for a retry if
                // the other dial fails. Later candidates go first meanwhile.
                if let Some(candidate) = pending.candidates.front_mut() {
                    candidate.parked = true;
                }
                pending.active = Some(ActiveDiscovery::Parked);
                let node = self.clone();
                tokio::spawn(async move { node.wait_for_parked_discovery(id, waiter).await });
                return None;
            }
        };
        if directory && cached.is_some() {
            state.remove_pending_connection_if_attempt(id, owner.attempt_id);
            pending.candidates.pop_front();
            return None;
        }
        let addr = pending
            .address_for_dial(id)
            .expect("checked the candidate under the queue lock");
        Some(ReservedDial {
            addr,
            owner,
            cached,
        })
    }

    async fn wait_for_parked_discovery(&self, id: EndpointId, waiter: PendingConnectionWaiter) {
        let result = tokio::select! {
            biased;
            _ = self.endpoint.closed() => return,
            result = tokio::time::timeout(
                PEER_CONNECT_AND_GOSSIP_TIMEOUT * 2,
                self.await_pending_connection(waiter),
            ) => result,
        };
        {
            let mut pending = self.peer_discovery.lock().await;
            if matches!(result, Ok(Ok(()))) {
                pending.withdraw(id);
                // Not every owner refreshes the selected path after admission.
                self.schedule_selected_path_recheck(id);
            } else {
                pending.unpark(id);
            }
        }
        self.start_peer_discovery().await;
    }

    async fn run_peer_discovery(&self, mut id: EndpointId) {
        loop {
            let result = Box::pin(self.connect_to_discovered_peer(id)).await;
            let next = {
                let mut pending = self.peer_discovery.lock().await;
                pending.finish(result.is_err());
                if self.endpoint.is_closed() {
                    pending.candidates.clear();
                }
                pending.begin_next()
            };
            if let Err(error) = result {
                tracing::debug!(peer = %id.fmt_short(), "Could not connect to discovered peer: {error}");
            }
            let Some(next) = next else {
                return;
            };
            id = next;
        }
    }

    pub(crate) fn discovered_peer_is_filtered(peer_id: EndpointId, ann: &PeerAnnouncement) -> bool {
        if !version_allowed_for_rebroadcast(ann.version.as_deref())
            || peer_is_idle_transitive_client(ann)
        {
            tracing::debug!(
                "Skipping discovered peer {} (filtered: version={:?} role={:?})",
                peer_id.fmt_short(),
                ann.version,
                ann.role
            );
            return true;
        }
        false
    }

    pub(crate) fn should_skip_discovered_peer(
        &self,
        my_role: &NodeRole,
        peer_id: EndpointId,
        peer_role: &NodeRole,
    ) -> bool {
        peer_id == self.endpoint.id()
            || (matches!(my_role, NodeRole::Client) && matches!(peer_role, NodeRole::Client))
    }
}

#[cfg(test)]
mod tests {
    use super::super::tests::{test_addr, test_announcement};
    use super::*;
    use crate::mesh::connections::encode_endpoint_addr_token;
    use crate::protocol::{STREAM_GOSSIP, connect_mesh, connection_protocol};
    use mesh_llm_membership::connection_reservation::{
        PendingConnectionAttemptOwner, PendingConnectionOutcome, PendingConnectionReservation,
    };
    use std::time::Duration;

    fn queued_addresses(pending: &PendingDiscovery) -> Vec<EndpointAddr> {
        pending
            .candidates
            .iter()
            .map(|candidate| candidate.addr.clone())
            .collect()
    }

    #[test]
    fn overlapping_replies_keep_one_candidate_and_retry_only_a_changed_address() {
        let [old, other] = [0x60, 0x61].map(test_addr);
        let mut updated = old.clone();
        updated
            .addrs
            .insert(iroh::TransportAddr::Ip(([127, 0, 0, 1], 9337).into()));
        let mut pending = PendingDiscovery::default();
        pending.upsert(&old, true, None);
        assert_eq!(pending.begin_next(), Some(old.id));
        assert_eq!(pending.address_for_dial(old.id).as_ref(), Some(&old));
        pending.upsert(&other, true, None);
        pending.upsert(&updated, true, None);
        pending.upsert(&old, true, None);
        pending.upsert(&other, true, None);
        assert_eq!(queued_addresses(&pending), [updated.clone(), other.clone()]);

        pending.finish(true);
        assert_eq!(queued_addresses(&pending), [other.clone(), updated.clone()]);
        assert_eq!(pending.begin_next(), Some(other.id));
        pending.finish(true);
        assert_eq!(pending.begin_next(), Some(updated.id));
        pending.finish(true);
        assert!(pending.candidates.is_empty());
    }

    #[test]
    fn addresses_refresh_in_acceptance_order_until_the_worker_owns_the_dial() {
        let mut original = test_addr(0x69);
        original
            .addrs
            .insert(iroh::TransportAddr::Ip(([127, 0, 0, 1], 9337).into()));
        let mut alternate = EndpointAddr::new(original.id);
        alternate
            .addrs
            .insert(iroh::TransportAddr::Ip(([127, 0, 0, 1], 9338).into()));
        let mut pending = PendingDiscovery::default();
        pending.upsert(&original, true, None);
        pending.begin_next();
        pending.upsert(&alternate, true, None);
        pending.upsert(&original, true, None);
        assert_eq!(queued_addresses(&pending), [original.clone()]);
        pending.withdraw(original.id);
        pending.upsert(&original, true, None);
        assert_eq!(pending.address_for_dial(original.id), Some(original));
    }

    #[test]
    fn poorer_announcements_preserve_queued_paths() {
        let mut old = test_addr(0x63);
        old.addrs = [9337, 9338]
            .map(|port| iroh::TransportAddr::Ip(([127, 0, 0, 1], port).into()))
            .into();
        let mut fresh = EndpointAddr::new(old.id);
        fresh
            .addrs
            .insert(iroh::TransportAddr::Ip(([127, 0, 0, 1], 9339).into()));
        let mut pending = PendingDiscovery::default();
        pending.upsert(&old, true, None);
        pending.upsert(&fresh, true, None);
        pending.upsert(&EndpointAddr::new(old.id), true, None);
        assert_eq!(pending.candidates[0].addr.addrs, &old.addrs | &fresh.addrs);
    }

    #[test]
    fn source_changes_replace_hints_and_finishing_a_skip_keeps_replacement_work() {
        let mut inbound = test_addr(0x75);
        inbound
            .addrs
            .insert(iroh::TransportAddr::Ip(([127, 0, 0, 1], 9337).into()));
        let admitted = EndpointAddr::new(inbound.id);
        let mut pending = PendingDiscovery::default();
        pending.upsert(&inbound, true, Some(NodeRole::Worker));
        pending.upsert(&admitted, true, None);
        assert_eq!(queued_addresses(&pending), std::slice::from_ref(&admitted));
        assert!(pending.candidates[0].inbound_role.is_none());

        pending.upsert(&inbound, true, Some(NodeRole::Worker));
        assert_eq!(pending.begin_next(), Some(inbound.id));
        pending.candidates.pop_front();
        pending.upsert(&admitted, true, None);
        pending.finish(false);
        assert_eq!(pending.begin_next(), Some(admitted.id));
        assert_eq!(pending.address_for_dial(admitted.id), Some(admitted));
    }

    #[test]
    fn active_address_hints_only_suppress_updates_from_the_same_source() {
        let [first, second] =
            [9337, 9338].map(|port| iroh::TransportAddr::Ip(([127, 0, 0, 1], port).into()));
        let inbound = test_addr(0x77).with_addrs([first.clone(), second.clone()]);
        let accepted = EndpointAddr::new(inbound.id).with_addrs([first]);
        let latest = EndpointAddr::new(inbound.id).with_addrs([second]);
        let mut pending = PendingDiscovery::default();
        pending.upsert(&inbound, true, Some(NodeRole::Worker));
        pending.begin_next();
        assert_eq!(pending.address_for_dial(inbound.id), Some(inbound));
        pending.upsert(&accepted, true, None);
        pending.upsert(&latest, true, None);
        pending.finish(true);
        assert_eq!(pending.begin_next(), Some(latest.id));
        assert_eq!(pending.address_for_dial(latest.id), Some(latest));
    }

    #[tokio::test]
    async fn clients_leave_queue_capacity_for_eligible_peers() {
        let node = Node::new_for_tests(NodeRole::Client).await.unwrap();
        let bridge = iroh::SecretKey::generate().public();
        let mut ann = test_announcement(None);
        ann.role = NodeRole::Client;
        ann.hostname = Some("non-idle client".into());
        for _ in 0..MAX_DISCOVERY_CANDIDATES {
            let addr = EndpointAddr::new(iroh::SecretKey::generate().public());
            ann.addr = addr.clone();
            node.apply_transitive_peer(addr.id, &addr, &ann, bridge, true)
                .await;
        }
        let eligible = EndpointAddr::new(iroh::SecretKey::generate().public());
        ann.addr = eligible.clone();
        ann.role = NodeRole::Worker;
        node.apply_transitive_peer(eligible.id, &eligible, &ann, bridge, true)
            .await;
        assert_eq!(
            queued_addresses(&*node.peer_discovery.lock().await),
            [eligible]
        );
        assert_eq!(
            node.state.lock().await.peers.len(),
            MAX_DISCOVERY_CANDIDATES + 1
        );
        node.close_endpoint().await;
    }

    #[tokio::test]
    async fn inbound_discovery_skips_known_pending_and_filtered_peers() {
        let node = Node::new_for_tests(NodeRole::Worker).await.unwrap();
        let [bridge, known, pending, old, idle, dead] =
            [0x70, 0x71, 0x72, 0x73, 0x74, 0x76].map(test_addr);
        let worker = test_announcement(None);
        node.apply_transitive_peer(known.id, &known, &worker, bridge.id, false)
            .await;
        let PendingConnectionReservation::Owner(owner) =
            node.reserve_pending_connection(pending.id).await
        else {
            panic!("fixture must reserve the pending peer");
        };
        let mut old_ann = worker.clone();
        old_ann.version = Some("0.57.0".into());
        let mut idle_ann = worker.clone();
        idle_ann.role = NodeRole::Client;
        node.state
            .lock()
            .await
            .dead_peers
            .insert(dead.id, std::time::Instant::now());
        let initial_attempt_count = node.state.lock().await.next_pending_connection_attempt;

        node.enqueue_inbound_discovery(&[
            (known, worker.clone()),
            (pending, worker.clone()),
            (old, old_ann),
            (idle, idle_ann),
            (dead, worker),
        ])
        .await;
        tokio::time::timeout(Duration::from_secs(10), async {
            while node.peer_discovery.lock().await.active.is_some() {
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("the worker must skip a handshake that is still pending");
        let queue = node.peer_discovery.lock().await;
        assert!(queue.candidates.is_empty());
        assert!(queue.active.is_none());
        drop(queue);
        assert_eq!(owner.outcome_tx.receiver_count(), 1);
        assert_eq!(
            node.state.lock().await.next_pending_connection_attempt,
            initial_attempt_count,
            "ineligible inbound candidates must never reserve a dial"
        );
        node.finish_pending_connection(
            owner,
            PendingConnectionOutcome::Failed("fixture complete".into()),
        )
        .await;
        node.close_endpoint().await;
    }

    #[tokio::test]
    async fn inbound_gossip_returns_before_queued_cached_admission_finishes() {
        let receiver = Node::new_for_tests(NodeRole::Worker).await.unwrap();
        let sender = Node::new_for_tests(NodeRole::Worker).await.unwrap();
        let peer = Node::new_for_tests(NodeRole::Worker).await.unwrap();

        let addr = peer.endpoint_addr_for_advertisement();
        let mut ann = test_announcement(None);
        ann.addr = addr.clone();
        sender
            .apply_transitive_peer(addr.id, &addr, &ann, sender.id(), false)
            .await;
        receiver
            .apply_transitive_peer(addr.id, &addr, &ann, sender.id(), true)
            .await;
        let accept_cached = async { peer.endpoint.accept().await.unwrap().await.unwrap() };
        let (cached, cached_remote) = tokio::time::timeout(Duration::from_secs(20), async {
            tokio::join!(connect_mesh(&receiver.endpoint, addr), accept_cached)
        })
        .await
        .unwrap();
        let cached = cached.unwrap();
        {
            let mut state = receiver.state.lock().await;
            state.connections.insert(peer.id(), cached.clone());
            state.peers.remove(&peer.id());
            state
                .dead_peers
                .insert(peer.id(), std::time::Instant::now());
        }

        let outgoing = async {
            let conn =
                connect_mesh(&sender.endpoint, receiver.endpoint_addr_for_advertisement()).await?;
            sender
                .initiate_gossip_inner(conn, receiver.id(), false)
                .await
        };
        let incoming = async {
            let conn = receiver.endpoint.accept().await.unwrap().await.unwrap();
            let (send, mut recv) = conn.accept_bi().await.unwrap();
            let mut tag = [0];
            recv.read_exact(&mut tag).await.unwrap();
            assert_eq!(tag, [STREAM_GOSSIP]);
            receiver
                .handle_gossip_stream(sender.id(), connection_protocol(&conn), send, recv)
                .await?;
            Ok::<_, anyhow::Error>(conn)
        };
        let ((), _connection) = tokio::time::timeout(Duration::from_secs(20), async {
            tokio::try_join!(outgoing, incoming).unwrap()
        })
        .await
        .expect("the inbound handler must return before the cached peer replies");

        tokio::time::timeout(Duration::from_secs(20), async {
            let (send, mut recv) = cached_remote.accept_bi().await.unwrap();
            let mut tag = [0];
            recv.read_exact(&mut tag).await.unwrap();
            assert_eq!(tag, [STREAM_GOSSIP]);
            let waiter = {
                let state = receiver.state.lock().await;
                assert!(!state.peers.contains_key(&peer.id()));
                state.pending_connections[&peer.id()].waiter(peer.id())
            };
            peer.handle_gossip_stream(
                receiver.id(),
                connection_protocol(&cached_remote),
                send,
                recv,
            )
            .await
            .unwrap();
            receiver.await_pending_connection(waiter).await.unwrap();
        })
        .await
        .expect("the worker must complete admission on the retained connection");
        {
            let state = receiver.state.lock().await;
            assert!(state.peers[&peer.id()].is_admitted());
            assert_eq!(
                state.connections[&peer.id()].stable_id(),
                cached.stable_id()
            );
        }

        receiver.close_endpoint().await;
        sender.close_endpoint().await;
        peer.close_endpoint().await;
    }

    async fn wait_for_connection(node: &Node, id: EndpointId, expectation: &str) {
        tokio::time::timeout(Duration::from_secs(20), async {
            while !node.state.lock().await.connections.contains_key(&id) {
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect(expectation);
    }

    #[tokio::test]
    async fn parked_candidate_lets_later_peers_dial_then_uses_its_newer_address() {
        let node = Node::new_for_tests(NodeRole::Worker).await.unwrap();
        let remote = Node::new_for_tests(NodeRole::Worker).await.unwrap();
        let next = Node::new_for_tests(NodeRole::Worker).await.unwrap();
        let PendingConnectionReservation::Owner(stale) =
            node.reserve_pending_connection(remote.id()).await
        else {
            panic!("fixture must own the original attempt");
        };
        let accept_remote = remote.clone();
        let accept_next = next.clone();
        let remote_listener = tokio::spawn(async move { accept_remote.accept_loop().await });
        let next_listener = tokio::spawn(async move { accept_next.accept_loop().await });
        remote.start_accepting();
        next.start_accepting();

        // The remote peer is queued first, with an address that cannot be dialed.
        let bridge = test_addr(0x62).id;
        let mut ann = test_announcement(None);
        ann.role = NodeRole::Worker;
        for addr in [
            EndpointAddr::new(remote.id()),
            next.endpoint_addr_for_advertisement(),
        ] {
            ann.addr = addr.clone();
            node.apply_transitive_peer(addr.id, &addr, &ann, bridge, true)
                .await;
        }
        node.start_peer_discovery().await;

        wait_for_discovery_waiter(&stale).await;
        wait_for_connection(
            &node,
            next.id(),
            "a parked candidate must not hold up later peers",
        )
        .await;

        let addr = remote.endpoint_addr_for_advertisement();
        ann.addr = addr.clone();
        node.apply_transitive_peer(addr.id, &addr, &ann, bridge, true)
            .await;
        node.finish_pending_connection(
            stale,
            PendingConnectionOutcome::Failed("old address failed".into()),
        )
        .await;
        wait_for_connection(
            &node,
            remote.id(),
            "the worker must retry the parked peer with its newer address",
        )
        .await;

        node.close_endpoint().await;
        remote.close_endpoint().await;
        next.close_endpoint().await;
        remote_listener.await.unwrap();
        next_listener.await.unwrap();
    }

    #[tokio::test]
    async fn finishing_a_parked_round_keeps_a_candidate_its_wait_already_unparked() {
        let node = Node::new_for_tests(NodeRole::Worker).await.unwrap();
        let [bridge, addr] = [0x7b, 0x7c].map(test_addr);
        let PendingConnectionReservation::Owner(owner) =
            node.reserve_pending_connection(addr.id).await
        else {
            panic!("fixture must own the original attempt");
        };
        let mut ann = test_announcement(None);
        ann.addr = addr.clone();
        ann.role = NodeRole::Worker;
        node.apply_transitive_peer(addr.id, &addr, &ann, bridge.id, true)
            .await;
        assert_eq!(node.peer_discovery.lock().await.begin_next(), Some(addr.id));
        assert!(node.reserve_discovery_connection(addr.id).await.is_none());

        // The other attempt fails and the wait unparks the candidate before
        // the worker finishes its round.
        node.finish_pending_connection(
            owner,
            PendingConnectionOutcome::Failed("other attempt failed".into()),
        )
        .await;
        tokio::time::timeout(Duration::from_secs(10), async {
            while node.peer_discovery.lock().await.candidates[0].parked {
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("the failed attempt must unpark the candidate");
        let mut pending = node.peer_discovery.lock().await;
        pending.finish(false);
        assert_eq!(queued_addresses(&pending), std::slice::from_ref(&addr));
        assert_eq!(pending.begin_next(), Some(addr.id));
        drop(pending);
        node.close_endpoint().await;
    }

    #[tokio::test]
    async fn filtered_transitive_announcements_never_stay_queued() {
        let node = Node::new_for_tests(NodeRole::Worker).await.unwrap();
        let [bridge, queued, old] = [0x78, 0x79, 0x7a].map(test_addr);
        let mut worker = test_announcement(None);
        worker.addr = queued.clone();
        worker.role = NodeRole::Worker;
        node.apply_transitive_peer(queued.id, &queued, &worker, bridge.id, true)
            .await;
        assert_eq!(
            queued_addresses(&*node.peer_discovery.lock().await),
            std::slice::from_ref(&queued)
        );

        let mut idle = worker.clone();
        idle.role = NodeRole::Client;
        node.apply_transitive_peer(queued.id, &queued, &idle, bridge.id, true)
            .await;
        let mut below_floor = worker;
        below_floor.addr = old.clone();
        below_floor.version = Some("0.57.0".into());
        node.apply_transitive_peer(old.id, &old, &below_floor, bridge.id, true)
            .await;
        assert!(node.peer_discovery.lock().await.candidates.is_empty());
        node.close_endpoint().await;
    }

    #[tokio::test]
    async fn explicit_join_tries_invite_address_after_discovery_dial_fails() {
        let node = Node::new_for_tests(NodeRole::Worker).await.unwrap();
        let remote = Node::new_for_tests(NodeRole::Worker).await.unwrap();
        let stale_addr = EndpointAddr::new(remote.id());
        let mut ann = test_announcement(None);
        ann.addr = stale_addr.clone();
        ann.role = NodeRole::Worker;
        node.apply_transitive_peer(remote.id(), &stale_addr, &ann, test_addr(0x66).id, true)
            .await;
        assert_eq!(
            node.peer_discovery.lock().await.begin_next(),
            Some(remote.id())
        );
        let ReservedDial {
            addr, owner: stale, ..
        } = node
            .reserve_discovery_connection(remote.id())
            .await
            .expect("discovery must own the original attempt");
        assert_eq!(addr, stale_addr);

        let accepting = remote.clone();
        let listener = tokio::spawn(async move { accepting.accept_loop().await });
        remote.start_accepting();
        let invite = encode_endpoint_addr_token(&remote.endpoint_addr_for_advertisement());
        let old_invite = encode_endpoint_addr_token(&stale_addr);
        let join = node.join(&invite);
        let other_join = node.join(&old_invite);
        tokio::pin!(join, other_join);
        tokio::time::timeout(Duration::from_secs(20), async {
            while stale.outcome_tx.receiver_count() < 3 {
                assert!(futures_util::poll!(&mut join).is_pending());
                assert!(futures_util::poll!(&mut other_join).is_pending());
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("both joins must wait for the discovery attempt");
        let attempts_before_retry = node.state.lock().await.next_pending_connection_attempt;

        node.finish_pending_connection(
            stale,
            PendingConnectionOutcome::Failed("discovery address failed".into()),
        )
        .await;

        tokio::time::timeout(Duration::from_secs(20), join)
            .await
            .expect("explicit join must complete")
            .expect("explicit join must try the invite address");
        tokio::time::timeout(Duration::from_secs(20), other_join)
            .await
            .expect("the second waiter must complete")
            .expect("the second waiter must reuse the admitted connection");
        {
            let state = node.state.lock().await;
            assert!(state.connections.contains_key(&remote.id()));
            assert_eq!(
                state.next_pending_connection_attempt,
                attempts_before_retry + 1,
                "the second waiter must not start another dial"
            );
        }

        node.close_endpoint().await;
        remote.close_endpoint().await;
        listener.await.unwrap();
    }

    async fn wait_for_discovery_waiter(owner: &PendingConnectionAttemptOwner) {
        tokio::time::timeout(Duration::from_secs(20), async {
            while owner.outcome_tx.receiver_count() < 2 {
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("discovery must wait for the original attempt");
    }

    #[tokio::test]
    async fn closing_endpoint_releases_a_parked_discovery_wait() {
        let node = Node::new_for_tests(NodeRole::Worker).await.unwrap();
        let [bridge, addr] = [0x64, 0x65].map(test_addr);
        let PendingConnectionReservation::Owner(owner) =
            node.reserve_pending_connection(addr.id).await
        else {
            panic!("fixture must own the original attempt");
        };
        let mut ann = test_announcement(None);
        ann.addr = addr.clone();
        ann.role = NodeRole::Worker;
        node.apply_transitive_peer(addr.id, &addr, &ann, bridge.id, true)
            .await;
        node.start_peer_discovery().await;
        wait_for_discovery_waiter(&owner).await;

        node.close_endpoint().await;
        tokio::time::timeout(Duration::from_secs(10), async {
            while owner.outcome_tx.receiver_count() > 1 {
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("endpoint closure must release the parked wait");
        node.finish_pending_connection(
            owner,
            PendingConnectionOutcome::Failed("test ended".into()),
        )
        .await;
    }
}
