//! Signed self-records: data a node signs about itself with its endpoint key
//! and relays forward byte-for-byte. Anything gossiped mesh-wide travels as
//! one of these, each kind under its own domain tag.
//!
//! The signature covers the bytes as transmitted, never a re-encoding, so a
//! record body needs no canonical form. A verifier checks the signature over
//! the received bytes and then decodes those same bytes:
//!
//! ```text
//! signed    = endpoint_id[32] || seq[8, u64 BE] || issued_at_unix_ms[8, u64 BE]
//!             || body (protobuf bytes)
//! signature = Ed25519(endpoint key, kind.domain_tag || signed)
//! ```
//!
//! The header is fixed-width so relays can check the signer, ordering and
//! age without touching protobuf, and the body is the only variable-length
//! part, so no length prefixes are needed. A different layout gets a new
//! domain tag rather than a version field.

use ed25519_dalek::{Signature, Signer, SigningKey, VerifyingKey};
use iroh::EndpointId;
use mesh_llm_routing::cache_inventory::{CACHE_AFFINITY_MAX_FUTURE_SKEW_MS, CACHE_AFFINITY_TTL};
use std::collections::HashMap;

const ENDPOINT_ID_LEN: usize = 32;
const HEADER_LEN: usize = ENDPOINT_ID_LEN + 8 + 8;
const SIGNATURE_LEN: usize = 64;

/// What a record kind signs under and how long relays keep it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RecordKind {
    /// Domain separation for this kind's signatures. Ends in NUL so it
    /// cannot be a prefix of, or prefixed by, another tag signed with the
    /// endpoint key.
    pub domain_tag: &'static [u8],
    /// Upper bound on `signed`, checked before any signature or decode work.
    pub max_signed_bytes: usize,
    /// Relays stop accepting and forwarding a record this long after it was
    /// issued, so a departed node's last record cannot circulate forever.
    pub max_age_ms: u64,
    /// How far into the future a relayed record's issue time may be.
    pub max_future_skew_ms: u64,
    /// A node re-signs an unchanged body this often, well inside the max age.
    pub refresh_ms: u64,
}

/// The node's self-description (`NodeRecord`).
pub const NODE_RECORD: RecordKind = RecordKind {
    domain_tag: b"mesh-llm-node-record-v1\0",
    max_signed_bytes: 256 * 1024,
    max_age_ms: 60 * 60 * 1000,
    max_future_skew_ms: 10 * 60 * 1000,
    refresh_ms: 15 * 60 * 1000,
};

/// The node's cache-affinity evidence (`CacheAffinityAdvertisement`). Its
/// lifetime matches the advertisement's own TTL.
pub const CACHE_AFFINITY_RECORD: RecordKind = RecordKind {
    domain_tag: b"mesh-llm-cache-affinity-record-v1\0",
    max_signed_bytes: 64 * 1024,
    max_age_ms: CACHE_AFFINITY_TTL.as_millis() as u64,
    max_future_skew_ms: CACHE_AFFINITY_MAX_FUTURE_SKEW_MS,
    refresh_ms: 30 * 1000,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RecordError {
    Oversize,
    Truncated,
    InvalidEndpointId,
    InvalidSignatureLength,
    BadSignature,
    Decode,
}

impl std::fmt::Display for RecordError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let reason = match self {
            Self::Oversize => "signed record exceeds the size limit",
            Self::Truncated => "signed record is shorter than its header",
            Self::InvalidEndpointId => "signed record names an invalid endpoint id",
            Self::InvalidSignatureLength => "record signature is not 64 bytes",
            Self::BadSignature => "record signature does not verify",
            Self::Decode => "record body is not valid for its kind",
        };
        f.write_str(reason)
    }
}

impl std::error::Error for RecordError {}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RecordHeader {
    pub endpoint_id: EndpointId,
    pub seq: u64,
    pub issued_at_unix_ms: u64,
}

impl RecordHeader {
    /// Reads the fixed-width header. Does not verify anything beyond the
    /// size limit and the endpoint id being a valid Ed25519 point.
    pub fn parse(kind: &RecordKind, signed: &[u8]) -> Result<Self, RecordError> {
        if signed.len() > kind.max_signed_bytes {
            return Err(RecordError::Oversize);
        }
        if signed.len() < HEADER_LEN {
            return Err(RecordError::Truncated);
        }
        let mut id = [0u8; ENDPOINT_ID_LEN];
        id.copy_from_slice(&signed[..ENDPOINT_ID_LEN]);
        let mut seq = [0u8; 8];
        seq.copy_from_slice(&signed[ENDPOINT_ID_LEN..ENDPOINT_ID_LEN + 8]);
        let mut issued_at = [0u8; 8];
        issued_at.copy_from_slice(&signed[ENDPOINT_ID_LEN + 8..HEADER_LEN]);
        let endpoint_id =
            EndpointId::from_bytes(&id).map_err(|_| RecordError::InvalidEndpointId)?;
        Ok(Self {
            endpoint_id,
            seq: u64::from_be_bytes(seq),
            issued_at_unix_ms: u64::from_be_bytes(issued_at),
        })
    }

    fn encode(&self) -> [u8; HEADER_LEN] {
        let mut header = [0u8; HEADER_LEN];
        header[..ENDPOINT_ID_LEN].copy_from_slice(self.endpoint_id.as_bytes());
        header[ENDPOINT_ID_LEN..ENDPOINT_ID_LEN + 8].copy_from_slice(&self.seq.to_be_bytes());
        header[ENDPOINT_ID_LEN + 8..].copy_from_slice(&self.issued_at_unix_ms.to_be_bytes());
        header
    }

    /// Whether a relay should still accept and forward a record of `kind`.
    pub fn is_fresh_at(&self, kind: &RecordKind, now_unix_ms: u64) -> bool {
        let not_expired = now_unix_ms.saturating_sub(self.issued_at_unix_ms) <= kind.max_age_ms;
        not_expired && !self.is_ahead_of(kind, now_unix_ms)
    }

    /// Whether the record was issued further in the future than `kind`
    /// tolerates. No receiver keeps such a record, from any sender.
    pub fn is_ahead_of(&self, kind: &RecordKind, now_unix_ms: u64) -> bool {
        self.issued_at_unix_ms.saturating_sub(now_unix_ms) > kind.max_future_skew_ms
    }
}

/// A record whose signature has been checked against the endpoint id in its
/// header under its kind's domain tag. Only `sign` and `verify` construct one.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VerifiedRecord {
    kind: RecordKind,
    header: RecordHeader,
    signed: Vec<u8>,
    signature: [u8; SIGNATURE_LEN],
}

fn signing_input(kind: &RecordKind, signed: &[u8]) -> Vec<u8> {
    let mut message = Vec::with_capacity(kind.domain_tag.len() + signed.len());
    message.extend_from_slice(kind.domain_tag);
    message.extend_from_slice(signed);
    message
}

impl VerifiedRecord {
    pub fn sign(
        kind: RecordKind,
        key: &SigningKey,
        seq: u64,
        issued_at_unix_ms: u64,
        body: &[u8],
    ) -> Self {
        let endpoint_id = EndpointId::from_bytes(key.verifying_key().as_bytes())
            .expect("an Ed25519 verifying key is a valid endpoint id");
        let header = RecordHeader {
            endpoint_id,
            seq,
            issued_at_unix_ms,
        };
        let mut signed = Vec::with_capacity(HEADER_LEN + body.len());
        signed.extend_from_slice(&header.encode());
        signed.extend_from_slice(body);
        let signature = key.sign(&signing_input(&kind, &signed)).to_bytes();
        Self {
            kind,
            header,
            signed,
            signature,
        }
    }

    pub fn verify(kind: RecordKind, signed: &[u8], signature: &[u8]) -> Result<Self, RecordError> {
        let header = RecordHeader::parse(&kind, signed)?;
        let signature: [u8; SIGNATURE_LEN] = signature
            .try_into()
            .map_err(|_| RecordError::InvalidSignatureLength)?;
        let verifying_key = VerifyingKey::from_bytes(header.endpoint_id.as_bytes())
            .map_err(|_| RecordError::InvalidEndpointId)?;
        verifying_key
            .verify_strict(
                &signing_input(&kind, signed),
                &Signature::from_bytes(&signature),
            )
            .map_err(|_| RecordError::BadSignature)?;
        Ok(Self {
            kind,
            header,
            signed: signed.to_vec(),
            signature,
        })
    }

    pub fn header(&self) -> RecordHeader {
        self.header
    }

    pub fn endpoint_id(&self) -> EndpointId {
        self.header.endpoint_id
    }

    /// The encoded body, exactly as the node signed it.
    pub fn body(&self) -> &[u8] {
        &self.signed[HEADER_LEN..]
    }

    /// The bytes to forward unchanged on the wire.
    pub fn signed_bytes(&self) -> &[u8] {
        &self.signed
    }

    pub fn signature(&self) -> &[u8; SIGNATURE_LEN] {
        &self.signature
    }

    pub fn is_fresh_at(&self, now_unix_ms: u64) -> bool {
        self.header.is_fresh_at(&self.kind, now_unix_ms)
    }
}

/// How an incoming record relates to the one already held for that node.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RecordOrdering {
    /// Nothing held yet.
    New,
    /// Byte-identical to the held record; no need to verify again.
    Duplicate,
    /// A higher sequence number than the held record.
    Newer,
    /// A lower sequence number, or the same one with different bytes.
    Superseded,
}

pub fn order_against_held(
    held: Option<&VerifiedRecord>,
    incoming: &RecordHeader,
    incoming_signed: &[u8],
) -> RecordOrdering {
    let Some(held) = held else {
        return RecordOrdering::New;
    };
    if held.signed == incoming_signed {
        RecordOrdering::Duplicate
    } else if incoming.seq > held.header.seq {
        RecordOrdering::Newer
    } else {
        RecordOrdering::Superseded
    }
}

/// Sequence number for this node's next record of a kind: the wall clock in
/// milliseconds, bumped so it strictly increases within a process.
pub fn next_record_seq(previous: Option<u64>, now_unix_ms: u64) -> u64 {
    previous.map_or(now_unix_ms, |previous| now_unix_ms.max(previous + 1))
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RecordSkip {
    /// Our own record echoed back by a peer.
    OwnRecord,
    /// Older than, or conflicting with, the record already held.
    Superseded,
    /// Issued too long ago, or too far in the future.
    Expired,
    Invalid(RecordError),
}

#[derive(Debug, Clone, PartialEq)]
pub enum RecordAdmission<T> {
    /// The record is held (newly or already) and its body decoded.
    Accepted { record: VerifiedRecord, body: T },
    Skipped {
        endpoint_id: EndpointId,
        reason: RecordSkip,
    },
}

/// The latest verified record of one kind per node, plus this node's own.
#[derive(Debug, Clone)]
pub struct HeldRecords {
    kind: RecordKind,
    peers: HashMap<EndpointId, VerifiedRecord>,
    local: Option<VerifiedRecord>,
}

impl HeldRecords {
    pub fn new(kind: RecordKind) -> Self {
        Self {
            kind,
            peers: HashMap::new(),
            local: None,
        }
    }

    pub fn get(&self, id: &EndpointId) -> Option<&VerifiedRecord> {
        self.peers.get(id)
    }

    /// The held record for `id` if relays should still forward it.
    pub fn fresh(&self, id: &EndpointId, now_unix_ms: u64) -> Option<&VerifiedRecord> {
        self.peers
            .get(id)
            .filter(|record| record.is_fresh_at(now_unix_ms))
    }

    pub fn contains(&self, id: &EndpointId) -> bool {
        self.peers.contains_key(id)
    }

    pub fn remove(&mut self, id: &EndpointId) -> Option<VerifiedRecord> {
        self.peers.remove(id)
    }

    pub fn retain(&mut self, mut keep: impl FnMut(&EndpointId) -> bool) {
        self.peers.retain(|id, _| keep(id));
    }

    pub fn is_empty(&self) -> bool {
        self.peers.is_empty()
    }

    /// Checks an incoming record against the held one, verifying it only if
    /// it would replace that record, and holds it once `decode` accepts its
    /// body. Errors only when the header is too malformed to name a node.
    ///
    /// `sender` is the authenticated peer the frame came from. A node
    /// speaking for itself may roll its sequence back, for example after
    /// restarting with a clock behind the one that signed its last record,
    /// and may send an expired record; relays may only move a node's record
    /// forward. No sender may send a record issued too far in the future:
    /// its clock-based sequence would block the node's corrected records,
    /// and relays would never forward it.
    pub fn admit<T>(
        &mut self,
        local_id: EndpointId,
        sender: EndpointId,
        signed: &[u8],
        signature: &[u8],
        now_unix_ms: u64,
        decode: impl FnOnce(&[u8]) -> Result<T, RecordError>,
    ) -> Result<RecordAdmission<T>, RecordError> {
        let header = RecordHeader::parse(&self.kind, signed)?;
        let endpoint_id = header.endpoint_id;
        let skipped = |reason| RecordAdmission::Skipped {
            endpoint_id,
            reason,
        };
        if endpoint_id == local_id {
            return Ok(skipped(RecordSkip::OwnRecord));
        }
        let held = self.peers.get(&endpoint_id);
        let ordering = order_against_held(held, &header, signed);
        if endpoint_id != sender {
            if !header.is_fresh_at(&self.kind, now_unix_ms) {
                return Ok(skipped(RecordSkip::Expired));
            }
            if ordering == RecordOrdering::Superseded {
                return Ok(skipped(RecordSkip::Superseded));
            }
        } else if header.is_ahead_of(&self.kind, now_unix_ms) {
            return Ok(skipped(RecordSkip::Expired));
        }
        let duplicate = held.filter(|_| ordering == RecordOrdering::Duplicate);
        let record = match duplicate {
            Some(held) => held.clone(),
            None => match VerifiedRecord::verify(self.kind, signed, signature) {
                Ok(record) => record,
                Err(error) => return Ok(skipped(RecordSkip::Invalid(error))),
            },
        };
        let body = match decode(record.body()) {
            Ok(body) => body,
            Err(error) => return Ok(skipped(RecordSkip::Invalid(error))),
        };
        self.peers.insert(endpoint_id, record.clone());
        Ok(RecordAdmission::Accepted { record, body })
    }

    /// This node's signed record for `body`. Re-signs only when the body
    /// changed or the held record is due for refresh, so relays see a new
    /// sequence number only when there is something new.
    pub fn refresh_local(
        &mut self,
        key: &SigningKey,
        body: &[u8],
        now_unix_ms: u64,
    ) -> &VerifiedRecord {
        let previous_seq = self.local.as_ref().map(|held| held.header.seq);
        let reusable = self.local.as_ref().is_some_and(|held| {
            held.body() == body
                && now_unix_ms.saturating_sub(held.header.issued_at_unix_ms) < self.kind.refresh_ms
        });
        if !reusable {
            self.local = None;
        }
        let kind = self.kind;
        self.local.get_or_insert_with(|| {
            let seq = next_record_seq(previous_seq, now_unix_ms);
            VerifiedRecord::sign(kind, key, seq, now_unix_ms, body)
        })
    }
}

#[cfg(test)]
mod tests;
