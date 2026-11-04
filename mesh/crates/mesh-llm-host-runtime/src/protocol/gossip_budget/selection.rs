//! Fair, bounded outbound gossip without changing the wire protocol.

use super::{GossipFrameBudgetExceeded, MAX_CONTROL_FRAME_BYTES, encode_gossip};
use crate::proto::node::{GossipFrame, PeerAnnouncement, SignedCacheAffinity, SignedNodeRecord};
use iroh::EndpointId;
use mesh_llm_membership::signed_record::{CACHE_AFFINITY_RECORD, NODE_RECORD, RecordHeader};
use prost::Message;
use std::collections::{BTreeMap, VecDeque};
use std::sync::Mutex;

/// Match the receiver's per-batch signature work limit.
pub(crate) const MAX_GOSSIP_RECORDS: usize = 4096;
const MAX_CURSOR_PEERS: usize = 2048;

/// Keeps an independent rotation for each destination. Evicting the least
/// recently used cursor only restarts its rotation; it grants no admission or
/// verification quota. Memory stays bounded under connection churn.
#[derive(Default)]
pub(crate) struct GossipFrameEncoder {
    cursors: Mutex<VecDeque<(EndpointId, EndpointId)>>,
}

impl GossipFrameEncoder {
    /// Preserve small frames exactly. Split oversized frames by origin, keeping
    /// the sender and pairing each relayed description with its signed cache.
    pub(crate) fn encode(&self, frame: GossipFrame, remote: EndpointId) -> anyhow::Result<Vec<u8>> {
        self.encode_with_limit(frame, remote, MAX_CONTROL_FRAME_BYTES)
    }

    fn encode_with_limit(
        &self,
        mut frame: GossipFrame,
        remote: EndpointId,
        limit: usize,
    ) -> anyhow::Result<Vec<u8>> {
        let size = frame.encoded_len();
        if size <= limit && signed_count(&frame) <= MAX_GOSSIP_RECORDS {
            return encode_gossip(&frame);
        }
        // A locally signed record can itself exceed its envelope limit. An
        // unselectable local snapshot must remain a deferral, not a peer fault.
        let mut origins = split_by_origin(&mut frame)
            .map_err(|error| error.context(GossipFrameBudgetExceeded { size }))?;
        let sender = EndpointId::from_bytes(frame.sender_id.as_slice().try_into()?)?;
        if let Some(own) = origins.remove(&sender) {
            own.append_to(&mut frame);
        }
        let mut size = frame.encoded_len();
        if size > limit || signed_count(&frame) > MAX_GOSSIP_RECORDS {
            return Err(GossipFrameBudgetExceeded { size }.into());
        }
        let own_size = size;
        let own_count = signed_count(&frame);
        let mut cursors = self.cursors.lock().unwrap_or_else(|e| e.into_inner());
        let previous = cursors
            .iter()
            .position(|(peer, _)| *peer == remote)
            .and_then(|index| cursors.remove(index))
            .map(|(_, last)| last);
        let mut origins: Vec<_> = origins.into_iter().collect();
        let start = previous.map_or(0, |last| origins.partition_point(|(id, _)| *id <= last));
        origins.rotate_left(start);
        let mut last = previous;
        for (id, records) in origins {
            let added = records.encoded_len();
            let count = records.signed_count();
            // An individually untransportable origin must not block other peers.
            if added > limit - own_size || count > MAX_GOSSIP_RECORDS - own_count {
                tracing::warn!(peer = %id.fmt_short(), "gossip origin exceeds the outbound frame budget");
                last = Some(id);
                continue;
            }
            if added > limit - size || count > MAX_GOSSIP_RECORDS - signed_count(&frame) {
                break;
            }
            size += added;
            records.append_to(&mut frame);
            last = Some(id);
        }
        if let Some(last) = last {
            if cursors.len() == MAX_CURSOR_PEERS {
                cursors.pop_front();
            }
            cursors.push_back((remote, last));
        }
        drop(cursors);
        debug_assert_eq!(size, frame.encoded_len());
        encode_gossip(&frame)
    }
}

#[derive(Default)]
struct OriginRecords {
    peers: Vec<PeerAnnouncement>,
    nodes: Vec<SignedNodeRecord>,
    caches: Vec<SignedCacheAffinity>,
}

impl OriginRecords {
    fn encoded_len(&self) -> usize {
        self.peers
            .iter()
            .map(|p| prost::encoding::message::encoded_len(2, p))
            .sum::<usize>()
            + self
                .nodes
                .iter()
                .map(|p| prost::encoding::message::encoded_len(4, p))
                .sum::<usize>()
            + self
                .caches
                .iter()
                .map(|p| prost::encoding::message::encoded_len(5, p))
                .sum::<usize>()
    }

    fn signed_count(&self) -> usize {
        self.nodes.len() + self.caches.len()
    }

    fn append_to(self, frame: &mut GossipFrame) {
        frame.peers.extend(self.peers);
        frame.signed_records.extend(self.nodes);
        frame.signed_cache_affinity.extend(self.caches);
    }
}

fn signed_count(frame: &GossipFrame) -> usize {
    frame.signed_records.len() + frame.signed_cache_affinity.len()
}

/// Group trusted outbound bytes without decoding or re-signing their bodies.
fn split_by_origin(frame: &mut GossipFrame) -> anyhow::Result<BTreeMap<EndpointId, OriginRecords>> {
    let mut origins = BTreeMap::<EndpointId, OriginRecords>::new();
    for peer in std::mem::take(&mut frame.peers) {
        let id = EndpointId::from_bytes(peer.endpoint_id.as_slice().try_into()?)?;
        origins.entry(id).or_default().peers.push(peer);
    }
    for record in std::mem::take(&mut frame.signed_records) {
        let id = RecordHeader::parse(&NODE_RECORD, &record.signed)?.endpoint_id;
        origins.entry(id).or_default().nodes.push(record);
    }
    for record in std::mem::take(&mut frame.signed_cache_affinity) {
        let id = RecordHeader::parse(&CACHE_AFFINITY_RECORD, &record.signed)?.endpoint_id;
        origins.entry(id).or_default().caches.push(record);
    }
    Ok(origins)
}

#[cfg(test)]
mod tests;
