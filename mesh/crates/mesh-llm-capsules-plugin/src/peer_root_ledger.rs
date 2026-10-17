//! Peer checkpoint-root reconciliation — the receiving half of the peer
//! checkpoint-root exchange. mesh-llm's fork carries an optional
//! `checkpoint: {log_id, mmr_size, root, timestamp_unix_ms, signature}` head
//! on `PeerAnnouncement` (and the plugin-facing `MeshPeer` mirror); this
//! module is what a plugin does with it once received.
//!
//! **Wiring status.** The published `mesh-llm-plugin` this plugin depends on
//! (see `Cargo.toml`) does not carry the `checkpoint` field yet. Until that
//! field ships in a published
//! `mesh-llm-plugin` this plugin depends on, `event.peer.checkpoint` does not
//! exist to read, so `on_mesh_event` cannot be wired yet. This module is
//! therefore self-contained (its own `CheckpointHead`, not
//! `mesh_llm_plugin::proto::PeerCheckpointHead`) and fully tested in
//! isolation; wiring it into `main.rs`'s `on_mesh_event` handler is a
//! follow-up gated on that dependency bump (itself gated on the fork PR's
//! upstream merge + release — see the task's own "commit 1 / commit 2"
//! split). Field names and shapes match the wire type exactly so that wiring
//! is a mechanical `CheckpointHead::from(&event.peer.checkpoint)` once it
//! exists.
//!
//! **Store:** `(peer_id, log_id, mmr_size) -> root`, keyed across every
//! source that reports a checkpoint head for a peer (gossip today; a bundle
//! or the witness later — the store does not care which). A second,
//! different root for the same key is a fork: sealed as a `ForkObserved`
//! record citing both signed heads, never silently overwritten.
//!
//! **Verification.** A checkpoint head's `signature` is an Ed25519 signature
//! by the announcing peer's OWN mesh node key (mesh-llm node ids are Ed25519
//! public keys) over the canonical JCS bytes of
//! `{log_id, mmr_size, root, timestamp_unix_ms}` — see `node.proto`'s
//! `PeerCheckpointHead` doc comment for the same contract on the wire side.
//! `ForkObserved::verifies_offline` re-derives and checks both signatures
//! from the record alone: no network call, no trust in whatever produced the
//! record.

use anyhow::{Context, Result};
use ed25519_dalek::{Signature, Verifier, VerifyingKey};
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};

/// A peer's signed checkpoint head, hex-encoded for JSON persistence. Mirrors
/// `meshllm.node.v1.PeerCheckpointHead` field-for-field.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CheckpointHead {
    pub log_id: String,
    pub mmr_size: u64,
    /// hex-encoded root (32 bytes)
    pub root: String,
    pub timestamp_unix_ms: u64,
    /// hex-encoded Ed25519 signature (64 bytes)
    pub signature: String,
}

impl CheckpointHead {
    fn signing_bytes(&self) -> Result<Vec<u8>> {
        let value = json!({
            "log_id": self.log_id,
            "mmr_size": self.mmr_size,
            "root": self.root,
            "timestamp_unix_ms": self.timestamp_unix_ms,
        });
        crate::producer::jcs::jcs(&value).context("canonicalize checkpoint head signing body")
    }

    /// Verify this head's signature against `peer_id_hex`'s own mesh node
    /// key. `false` on ANY malformed input (bad hex, wrong length, bad
    /// signature) — never a panic, never an assumed-valid default.
    pub fn verify(&self, peer_id_hex: &str) -> bool {
        (|| -> Result<bool> {
            let pk_bytes = hex::decode(peer_id_hex)?;
            let pk_arr: [u8; 32] = pk_bytes
                .try_into()
                .map_err(|_| anyhow::anyhow!("peer id is not 32 bytes"))?;
            let verifying_key = VerifyingKey::from_bytes(&pk_arr)?;
            let sig_bytes = hex::decode(&self.signature)?;
            let sig_arr: [u8; 64] = sig_bytes
                .try_into()
                .map_err(|_| anyhow::anyhow!("signature is not 64 bytes"))?;
            let signature = Signature::from_bytes(&sig_arr);
            let message = self.signing_bytes()?;
            Ok(verifying_key.verify(&message, &signature).is_ok())
        })()
        .unwrap_or(false)
    }
}

/// Where a checkpoint head observation came from — never invented, always
/// the caller's own label for the channel it read the head from.
pub type Source = String;

/// Two different signed checkpoint heads seen for the same peer, log, and
/// size — the evidence a fork happened (or that two sources disagree about
/// this peer's history).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ForkObserved {
    pub peer_id: String,
    pub log_id: String,
    pub mmr_size: u64,
    pub head_a: CheckpointHead,
    pub source_a: Source,
    pub head_b: CheckpointHead,
    pub source_b: Source,
}

impl ForkObserved {
    /// Offline, from the record alone: both heads name the same (peer,
    /// log_id, mmr_size), carry DIFFERENT roots, and both signatures verify
    /// against the claimed peer's own key. A record failing any of these is
    /// not real fork evidence, no matter what the store believed when it was
    /// written.
    pub fn verifies_offline(&self) -> bool {
        self.head_a.log_id == self.log_id
            && self.head_b.log_id == self.log_id
            && self.head_a.mmr_size == self.mmr_size
            && self.head_b.mmr_size == self.mmr_size
            && self.head_a.root != self.head_b.root
            && self.head_a.verify(&self.peer_id)
            && self.head_b.verify(&self.peer_id)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ObservedEntry {
    head: CheckpointHead,
    source: Source,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
struct PersistedState {
    /// key: "{peer_id}:{log_id}:{mmr_size}"
    observed: HashMap<String, ObservedEntry>,
    reconciled_peers: HashSet<String>,
    forks: Vec<ForkObserved>,
}

/// What `observe` learned from one incoming checkpoint head.
#[derive(Debug, Clone)]
pub enum Observation {
    /// First time this (peer, log_id, mmr_size) has been seen.
    New,
    /// Same root already on file for this key — nothing changed.
    Unchanged,
    /// A second, different root for the same key: `ForkObserved` names both.
    Fork(Box<ForkObserved>),
}

/// Persists to `<dir>/reconciliation_state.json` (current counts + observed
/// roots, overwritten atomically) and appends every fork to
/// `<dir>/fork_observations.jsonl` (audit trail, one self-contained,
/// independently offline-verifiable record per line — never rewritten).
pub struct PeerRootLedger {
    dir: PathBuf,
    state: PersistedState,
}

impl PeerRootLedger {
    pub fn open(dir: &Path) -> Result<Self> {
        std::fs::create_dir_all(dir)
            .with_context(|| format!("create peer-root ledger dir {}", dir.display()))?;
        let state_path = dir.join("reconciliation_state.json");
        let state = if state_path.exists() {
            let raw = std::fs::read_to_string(&state_path)
                .with_context(|| format!("read {}", state_path.display()))?;
            serde_json::from_str(&raw).context("parse reconciliation_state.json")?
        } else {
            PersistedState::default()
        };
        Ok(Self {
            dir: dir.to_path_buf(),
            state,
        })
    }

    fn key(peer_id: &str, log_id: &str, mmr_size: u64) -> String {
        format!("{peer_id}:{log_id}:{mmr_size}")
    }

    /// Record a checkpoint head observed for `peer_id` from `source`
    /// ("gossip", "bundle", "witness", ...). Persists before returning.
    pub fn observe(
        &mut self,
        peer_id: &str,
        head: &CheckpointHead,
        source: &str,
    ) -> Result<Observation> {
        self.state.reconciled_peers.insert(peer_id.to_string());
        let key = Self::key(peer_id, &head.log_id, head.mmr_size);
        let outcome = match self.state.observed.get(&key) {
            None => {
                self.state.observed.insert(
                    key,
                    ObservedEntry {
                        head: head.clone(),
                        source: source.to_string(),
                    },
                );
                Observation::New
            }
            Some(existing) if existing.head.root == head.root => Observation::Unchanged,
            Some(existing) => {
                let fork = ForkObserved {
                    peer_id: peer_id.to_string(),
                    log_id: head.log_id.clone(),
                    mmr_size: head.mmr_size,
                    head_a: existing.head.clone(),
                    source_a: existing.source.clone(),
                    head_b: head.clone(),
                    source_b: source.to_string(),
                };
                self.state.forks.push(fork.clone());
                Observation::Fork(Box::new(fork))
            }
        };
        self.persist()?;
        if let Observation::Fork(ref fork) = outcome {
            self.append_fork_log(fork)?;
        }
        Ok(outcome)
    }

    /// Count of distinct peers this node has ever recorded a checkpoint head
    /// for — the history card's `reconciled_with`.
    pub fn reconciled_with(&self) -> usize {
        self.state.reconciled_peers.len()
    }

    /// Count of forks ever sealed — the history card's `forks_observed`.
    pub fn forks_observed(&self) -> usize {
        self.state.forks.len()
    }

    pub fn forks(&self) -> &[ForkObserved] {
        &self.state.forks
    }

    fn persist(&self) -> Result<()> {
        let tmp = self.dir.join("reconciliation_state.json.tmp");
        std::fs::write(&tmp, serde_json::to_string_pretty(&self.state)?)
            .with_context(|| format!("write {}", tmp.display()))?;
        std::fs::rename(&tmp, self.dir.join("reconciliation_state.json"))
            .context("publish reconciliation_state.json")?;
        Ok(())
    }

    fn append_fork_log(&self, fork: &ForkObserved) -> Result<()> {
        use std::io::Write;
        let mut file = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(self.dir.join("fork_observations.jsonl"))
            .context("open fork_observations.jsonl")?;
        writeln!(file, "{}", serde_json::to_string(fork)?)
            .context("append fork_observations.jsonl")?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ed25519_dalek::{Signer, SigningKey};

    fn keypair(seed: u8) -> (SigningKey, String) {
        let signing_key = SigningKey::from_bytes(&[seed; 32]);
        let peer_id_hex = hex::encode(signing_key.verifying_key().to_bytes());
        (signing_key, peer_id_hex)
    }

    fn signed_head(
        signing_key: &SigningKey,
        log_id: &str,
        mmr_size: u64,
        root_byte: u8,
    ) -> CheckpointHead {
        let root = hex::encode([root_byte; 32]);
        let mut head = CheckpointHead {
            log_id: log_id.to_string(),
            mmr_size,
            root,
            timestamp_unix_ms: 1_725_000_000_000,
            signature: String::new(),
        };
        let message = head.signing_bytes().unwrap();
        let signature = signing_key.sign(&message);
        head.signature = hex::encode(signature.to_bytes());
        head
    }

    #[test]
    fn a_correctly_signed_head_verifies() {
        let (signing_key, peer_id) = keypair(0x11);
        let head = signed_head(&signing_key, "log-a", 4, 0xAA);
        assert!(head.verify(&peer_id));
    }

    #[test]
    fn a_head_signed_by_a_different_key_does_not_verify() {
        let (signing_key, _peer_id) = keypair(0x11);
        let (_other_key, other_peer_id) = keypair(0x22);
        let head = signed_head(&signing_key, "log-a", 4, 0xAA);
        assert!(!head.verify(&other_peer_id));
    }

    #[test]
    fn a_tampered_root_does_not_verify() {
        let (signing_key, peer_id) = keypair(0x11);
        let mut head = signed_head(&signing_key, "log-a", 4, 0xAA);
        head.root = hex::encode([0xBB; 32]);
        assert!(!head.verify(&peer_id));
    }

    #[test]
    fn malformed_signature_hex_fails_closed_not_panics() {
        let (signing_key, peer_id) = keypair(0x11);
        let mut head = signed_head(&signing_key, "log-a", 4, 0xAA);
        head.signature = "not-hex".to_string();
        assert!(!head.verify(&peer_id));
    }

    #[test]
    fn first_observation_is_new_and_counts_the_peer() {
        let dir = tempdir();
        let mut ledger = PeerRootLedger::open(dir.path()).unwrap();
        let (signing_key, peer_id) = keypair(0x11);
        let head = signed_head(&signing_key, "log-a", 4, 0xAA);

        let outcome = ledger.observe(&peer_id, &head, "gossip").unwrap();
        assert!(matches!(outcome, Observation::New));
        assert_eq!(ledger.reconciled_with(), 1);
        assert_eq!(ledger.forks_observed(), 0);
    }

    #[test]
    fn same_root_twice_is_unchanged() {
        let dir = tempdir();
        let mut ledger = PeerRootLedger::open(dir.path()).unwrap();
        let (signing_key, peer_id) = keypair(0x11);
        let head = signed_head(&signing_key, "log-a", 4, 0xAA);

        ledger.observe(&peer_id, &head, "gossip").unwrap();
        let outcome = ledger.observe(&peer_id, &head, "gossip").unwrap();
        assert!(matches!(outcome, Observation::Unchanged));
        assert_eq!(ledger.forks_observed(), 0);
    }

    #[test]
    fn three_peers_gossiping_heads_reconciles_with_all_three() {
        let dir = tempdir();
        let mut ledger = PeerRootLedger::open(dir.path()).unwrap();
        for seed in [0x11u8, 0x22, 0x33] {
            let (signing_key, peer_id) = keypair(seed);
            let head = signed_head(&signing_key, "test-log", 4, seed);
            let outcome = ledger.observe(&peer_id, &head, "gossip").unwrap();
            assert!(matches!(outcome, Observation::New));
        }
        assert_eq!(ledger.reconciled_with(), 3);
        assert_eq!(ledger.forks_observed(), 0);
    }

    /// The acceptance mutant: a peer presents root A via one source and root
    /// B via another for the same (log_id, mmr_size) — a fork must be sealed
    /// citing both signed heads, and that record must verify offline on its
    /// own, using only the two heads and the claimed peer id.
    #[test]
    fn a_second_different_root_from_any_source_seals_a_verifiable_fork() {
        let dir = tempdir();
        let mut ledger = PeerRootLedger::open(dir.path()).unwrap();
        let (signing_key, peer_id) = keypair(0x33);
        let head_a = signed_head(&signing_key, "test-log", 4, 0xA1);
        let head_b = signed_head(&signing_key, "test-log", 4, 0xB2);

        ledger.observe(&peer_id, &head_a, "gossip").unwrap();
        let outcome = ledger.observe(&peer_id, &head_b, "bundle").unwrap();

        let Observation::Fork(fork) = outcome else {
            panic!("expected a fork, got {outcome:?}");
        };
        assert_eq!(fork.source_a, "gossip");
        assert_eq!(fork.source_b, "bundle");
        assert!(
            fork.verifies_offline(),
            "a fork citing two validly-signed, differing heads must verify offline"
        );
        assert_eq!(ledger.forks_observed(), 1);
        assert_eq!(ledger.reconciled_with(), 1);
    }

    #[test]
    fn a_fork_record_with_a_forged_signature_does_not_verify_offline() {
        let (signing_key, peer_id) = keypair(0x44);
        let head_a = signed_head(&signing_key, "log-x", 1, 0x01);
        let mut forged_b = signed_head(&signing_key, "log-x", 1, 0x02);
        forged_b.signature = hex::encode([0u8; 64]);

        let fork = ForkObserved {
            peer_id,
            log_id: "log-x".to_string(),
            mmr_size: 1,
            head_a,
            source_a: "gossip".to_string(),
            head_b: forged_b,
            source_b: "gossip".to_string(),
        };

        assert!(
            !fork.verifies_offline(),
            "a fork record must not verify offline if either cited signature is invalid \
             — a fabricated fork must not pass as evidence"
        );
    }

    #[test]
    fn a_fork_record_with_a_matching_root_does_not_verify_offline() {
        // Guards against constructing a "fork" from two identical heads --
        // verifies_offline must require the roots to actually differ.
        let (signing_key, peer_id) = keypair(0x55);
        let head = signed_head(&signing_key, "log-y", 2, 0x09);

        let fork = ForkObserved {
            peer_id,
            log_id: "log-y".to_string(),
            mmr_size: 2,
            head_a: head.clone(),
            source_a: "gossip".to_string(),
            head_b: head,
            source_b: "witness".to_string(),
        };

        assert!(!fork.verifies_offline());
    }

    #[test]
    fn reconciliation_state_persists_across_reopen() {
        let dir = tempdir();
        let (signing_key, peer_id) = keypair(0x66);
        let head = signed_head(&signing_key, "log-z", 5, 0xEE);
        {
            let mut ledger = PeerRootLedger::open(dir.path()).unwrap();
            ledger.observe(&peer_id, &head, "gossip").unwrap();
        }

        let reopened = PeerRootLedger::open(dir.path()).unwrap();
        assert_eq!(reopened.reconciled_with(), 1);
        assert_eq!(reopened.forks_observed(), 0);
    }

    #[test]
    fn fork_observations_jsonl_gets_one_line_per_fork() {
        let dir = tempdir();
        let mut ledger = PeerRootLedger::open(dir.path()).unwrap();
        let (signing_key, peer_id) = keypair(0x77);
        let head_a = signed_head(&signing_key, "log-w", 9, 0x01);
        let head_b = signed_head(&signing_key, "log-w", 9, 0x02);
        ledger.observe(&peer_id, &head_a, "gossip").unwrap();
        ledger.observe(&peer_id, &head_b, "gossip").unwrap();

        let log_path = dir.path().join("fork_observations.jsonl");
        let contents = std::fs::read_to_string(log_path).unwrap();
        let lines: Vec<&str> = contents.lines().collect();
        assert_eq!(lines.len(), 1);
        let parsed: ForkObserved = serde_json::from_str(lines[0]).unwrap();
        assert!(parsed.verifies_offline());
    }

    // Minimal temp-dir helper -- avoids pulling in the `tempfile` crate for
    // a handful of unit tests.
    struct TempDir(PathBuf);
    impl TempDir {
        fn path(&self) -> &Path {
            &self.0
        }
    }
    impl Drop for TempDir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }
    fn tempdir() -> TempDir {
        // A counter, not the clock: tests run in parallel, and two clock reads
        // can return the same value.
        use std::sync::atomic::{AtomicU64, Ordering};
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        let dir = std::env::temp_dir().join(format!(
            "peer-root-ledger-test-{}-{}",
            std::process::id(),
            COUNTER.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir_all(&dir).unwrap();
        TempDir(dir)
    }
}
