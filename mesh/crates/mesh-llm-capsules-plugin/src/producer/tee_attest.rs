//! The `tee_measured` rung: an Intel TDX quote whose REPORT_DATA binds ONE
//! record, carried inline in `x-mesh-poc-v1.evidence_refs.tee_attestation`.
//!
//! The slot's shape, the binding and the checks are fixed by the spec desk's
//! u229 ruling; every name it fixes is in [`names`], so a change to the ruling
//! is a change there.
//!
//! What capsules checks (offline, each failing hard; u229 checks 1-4 and
//! u231 checks 5-8): the quote's SHA-256 is `digest`; the quote parses as a
//! TDX DCAP v4 quote; its REPORT_DATA is `report_data` and both equal the
//! recomputation over the record's own fields; the key bound in that
//! recomputation verifies the record; the quote's signature by its
//! attestation key; the QE report's signature by the PCK key; the QE report's
//! binding of the attestation key; and the PCK chain to the root pinned in
//! [`PINNED_ROOT_CA_SHA256`], at the record's own time. Only all eight make a
//! quote bound. What it leaves to the reader is [`names::LEFT_TO_READER`].
//!
//! A quote is taken through the kernel's configfs-tsm report interface
//! ([`ConfigfsTsm`]). On a host without it the slot is recorded empty, as
//! before, with no error.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use serde_json::{json, Value};
use sha2::{Digest, Sha256, Sha512};

/// Every name and value the u229 ruling fixes.
pub mod names {
    pub const TYPE: &str = "tee_attestation";
    pub const MEASUREMENT_CLASS: &str = "tee_measured";
    pub const QUOTE_FORMAT: &str = "tdx-dcap-v4";
    pub const DIGEST_ALG: &str = "sha-256";
    /// The binding's domain tag, ended by one 0x00 byte in REPORT_DATA's preimage.
    pub const TAG: &str = "x-mesh-poc-v1/tee-binding/v1";
    /// REPORT_DATA's inputs, in order, each 32 bytes (u229, nonce per u230).
    pub const INPUTS: [&str; 4] = [
        "agent_input_digest",
        "agent_output_digest",
        "client_nonce_sha256",
        "producer_public_key",
    ];
    /// The longest client nonce a quote is taken for, in bytes.
    pub const MAX_NONCE_BYTES: usize = 256;
    /// `client_nonce_source` values naming a nonce the CLIENT chose: only
    /// then is a quote fresh against the client's own challenge.
    pub const CLIENT_CHOSEN_SOURCES: [&str; 1] = ["client_supplied"];
    /// `client_nonce_source` values naming a real nonce that the host may
    /// have minted: bound to the exchange, no freshness claim.
    pub const OTHER_REAL_SOURCES: [&str; 1] = ["host_forwarded_nonce"];
    /// The empty slot's context when the record has no real client nonce.
    pub const NO_CLIENT_NONCE: &str =
        "no TEE quote taken: no client nonce; recorded absent, never fabricated";
    /// The members of a filled slot, and no others.
    pub const MEMBERS: [&str; 9] = [
        "type",
        "measurement_class",
        "quote_format",
        "digest_alg",
        "digest",
        "quote",
        "report_data",
        "binding",
        "context",
    ];
    /// What capsules does not check, said in the slot and on the page.
    pub const LEFT_TO_READER: &str = "Not checked by the producer or by capsules: revocation (PCK CRLs), TCB status and collateral recency (TCB info, QE identity), and MRTD and RTMRs against reference values. capsules checks the quote's signatures and its PCK chain to the pinned root when it shows this record.";
    /// The empty slot's context on a host that took no quote at all.
    pub const NO_TEE: &str = "no TEE quote measured (no TDX hardware on this host, or the producer leg has not run here); recorded absent, never fabricated";
}

/// A record's client nonce and where it came from (`client_nonce`,
/// `client_nonce_source`).
#[derive(Clone, Copy, Debug)]
pub struct ClientNonce<'a> {
    pub value: &'a str,
    pub source: &'a str,
}

impl ClientNonce<'_> {
    /// Whether the record shows the client chose this nonce.
    pub fn client_chose(&self) -> bool {
        names::CLIENT_CHOSEN_SOURCES.contains(&self.source)
    }

    /// `SHA-256(UTF8(value))` over the value exactly as recorded, when it
    /// is a real nonce: non-empty, at most [`names::MAX_NONCE_BYTES`], from a
    /// source that names one (never a placeholder).
    fn bytes(&self) -> Option<[u8; 32]> {
        let real = self.client_chose() || names::OTHER_REAL_SOURCES.contains(&self.source);
        (real && !self.value.is_empty() && self.value.len() <= names::MAX_NONCE_BYTES)
            .then(|| Sha256::digest(self.value.as_bytes()).into())
    }
}

/// The 64 REPORT_DATA bytes for one record: `SHA-512(UTF8(TAG) || 0x00 ||
/// input || output || SHA-256(UTF8(nonce)) || producer key)`, each input 32
/// bytes. `Err` names the input that is missing or malformed; then no quote
/// may be taken.
pub fn report_data(
    agent_input_digest: Option<&str>,
    agent_output_digest: Option<&str>,
    client_nonce: Option<ClientNonce<'_>>,
    producer_public_key: &[u8; 32],
) -> Result<[u8; 64], &'static str> {
    let input = agent_input_digest.and_then(hex32).ok_or(names::INPUTS[0])?;
    let output = agent_output_digest
        .and_then(hex32)
        .ok_or(names::INPUTS[1])?;
    let nonce = client_nonce
        .and_then(|n| n.bytes())
        .ok_or(names::INPUTS[2])?;
    let mut hasher = Sha512::new();
    hasher.update(names::TAG.as_bytes());
    hasher.update([0u8]);
    for value in [&input, &output, &nonce, producer_public_key] {
        hasher.update(value);
    }
    Ok(hasher.finalize().into())
}

/// Exactly 64 lowercase hex characters, decoded.
fn hex32(value: &str) -> Option<[u8; 32]> {
    if value.len() != 64
        || !value
            .bytes()
            .all(|b| matches!(b, b'0'..=b'9' | b'a'..=b'f'))
    {
        return None;
    }
    hex::decode(value).ok()?.try_into().ok()
}

// ---------------------------------------------------------------------------
// The quote
// ---------------------------------------------------------------------------

const HEADER_LEN: usize = 48;
const BODY_LEN: usize = 584;
const REPORT_DATA_AT: usize = HEADER_LEN + 520;
const SIGNED_LEN: usize = HEADER_LEN + BODY_LEN;
const TEE_TYPE_TDX: u32 = 0x81;
const ECDSA_P256: u16 = 2;
const SIG_AND_KEY_LEN: usize = 64 + 64;
const QE_REPORT_LEN: usize = 384;
/// Certification data type 6: the QE report, its signature and auth data,
/// then the PCK chain as type 5.
const CERT_QE_REPORT: u16 = 6;
const CERT_PCK_CHAIN: u16 = 5;

/// A TDX DCAP v4 quote, parsed for its structure. Nothing in it is
/// cryptographically checked: see the module doc.
#[derive(Debug)]
pub struct TdxQuote {
    pub report_data: [u8; 64],
    /// Header and TD report body: what the quote signature covers.
    signed: Vec<u8>,
    quote_signature: [u8; 64],
    /// The attestation key, raw `x || y`.
    attestation_key: [u8; 64],
    qe_report: [u8; QE_REPORT_LEN],
    qe_report_signature: [u8; 64],
    qe_auth_data: Vec<u8>,
    pck_chain_pem: Vec<u8>,
}

fn le16(b: &[u8], at: usize) -> Option<u16> {
    Some(u16::from_le_bytes(b.get(at..at + 2)?.try_into().ok()?))
}

fn le32(b: &[u8], at: usize) -> Option<u32> {
    Some(u32::from_le_bytes(b.get(at..at + 4)?.try_into().ok()?))
}

impl TdxQuote {
    /// Parse `bytes` as a TDX DCAP v4 quote: version 4, an ECDSA P-256
    /// attestation key, TEE type TDX, and a signature section whose nested
    /// lengths account for every byte, ending in a PEM PCK chain. Bytes after
    /// the signature section are accepted only when all are zero (the quote
    /// buffer's padding).
    pub fn parse(bytes: &[u8]) -> Result<Self, String> {
        let header = |what: &str| format!("not a TDX DCAP v4 quote: {what}");
        if le16(bytes, 0) != Some(4) {
            return Err(header("version is not 4"));
        }
        if le16(bytes, 2) != Some(ECDSA_P256) {
            return Err(header("attestation key is not ECDSA P-256"));
        }
        if le32(bytes, 4) != Some(TEE_TYPE_TDX) {
            return Err(header("TEE type is not TDX"));
        }
        let section_len = le32(bytes, SIGNED_LEN).ok_or_else(|| header("too short"))? as usize;
        let section_at = SIGNED_LEN + 4;
        let section_end = section_at
            .checked_add(section_len)
            .filter(|&end| end <= bytes.len())
            .ok_or_else(|| header("signature section runs past the end"))?;
        let section = &bytes[section_at..section_end];
        if bytes[section_end..].iter().any(|&b| b != 0) {
            return Err(header("non-zero bytes after the signature section"));
        }
        let (outer_type, outer) = cert_data(section, SIG_AND_KEY_LEN)
            .ok_or_else(|| header("certification data does not fit"))?;
        if outer_type != CERT_QE_REPORT || SIG_AND_KEY_LEN + 6 + outer.len() != section.len() {
            return Err(header(
                "certification data is not a QE report (type 6) filling the section",
            ));
        }
        let auth_at = QE_REPORT_LEN + 64;
        let auth_len =
            le16(outer, auth_at).ok_or_else(|| header("QE report data too short"))? as usize;
        let (inner_type, chain) = cert_data(outer, auth_at + 2 + auth_len)
            .ok_or_else(|| header("PCK chain does not fit"))?;
        if inner_type != CERT_PCK_CHAIN || auth_at + 2 + auth_len + 6 + chain.len() != outer.len() {
            return Err(header(
                "QE report data does not end in a PCK chain (type 5)",
            ));
        }
        if !chain.starts_with(b"-----BEGIN CERTIFICATE-----") {
            return Err(header("PCK chain is not PEM"));
        }
        let fixed = |range: std::ops::Range<usize>| -> &[u8] { &section[range] };
        let report_data = bytes[REPORT_DATA_AT..SIGNED_LEN]
            .try_into()
            .expect("64 bytes inside the checked header and body");
        Ok(Self {
            report_data,
            signed: bytes[..SIGNED_LEN].to_vec(),
            quote_signature: fixed(0..64).try_into().expect("64"),
            attestation_key: fixed(64..128).try_into().expect("64"),
            qe_report: outer[..QE_REPORT_LEN]
                .try_into()
                .expect("384, inside the checked QE report data"),
            qe_report_signature: outer[QE_REPORT_LEN..QE_REPORT_LEN + 64]
                .try_into()
                .expect("64"),
            qe_auth_data: outer[auth_at + 2..auth_at + 2 + auth_len].to_vec(),
            pck_chain_pem: chain.to_vec(),
        })
    }
}

/// The SHA-256 of the DER of the root CA every TDX DCAP v4 quote's PCK chain
/// ends in (Intel SGX Root CA). Checked against the root Intel publishes,
/// and never taken from the quote.
pub const PINNED_ROOT_CA_SHA256: [u8; 32] =
    hex_literal("44a0196b2b99f889b8e149e95b807a350e7424964399e885a7cbb8ccfab674d3");

const fn hex_literal(text: &str) -> [u8; 32] {
    const fn nibble(c: u8) -> u8 {
        match c {
            b'0'..=b'9' => c - b'0',
            b'a'..=b'f' => c - b'a' + 10,
            _ => panic!("lowercase hex"),
        }
    }
    let b = text.as_bytes();
    assert!(b.len() == 64);
    let mut out = [0u8; 32];
    let mut i = 0;
    while i < 32 {
        out[i] = nibble(b[2 * i]) << 4 | nibble(b[2 * i + 1]);
        i += 1;
    }
    out
}

/// Where the QE report keeps its REPORT_DATA (SGX report body layout).
const QE_REPORT_DATA_AT: usize = 320;

fn p256_key(sec1: &[u8]) -> Option<p256::ecdsa::VerifyingKey> {
    p256::ecdsa::VerifyingKey::from_sec1_bytes(sec1).ok()
}

fn raw_signature_verifies(key: &p256::ecdsa::VerifyingKey, message: &[u8], raw: &[u8; 64]) -> bool {
    use p256::ecdsa::signature::Verifier;
    p256::ecdsa::Signature::from_slice(raw).is_ok_and(|sig| key.verify(message, &sig).is_ok())
}

impl TdxQuote {
    /// Checks 5 to 8 (u231), in an order where each failure is its own:
    /// the quote signature, the PCK chain to the pinned root at `at` (seconds
    /// since the epoch, the record's own time), the QE report signature by
    /// the PCK key, and the attestation key's binding in the QE report.
    pub fn check_signatures(&self, at: u64, pinned_root: &[u8; 32]) -> Result<(), CheckFailure> {
        // 5
        let mut sec1 = vec![0x04];
        sec1.extend_from_slice(&self.attestation_key);
        let attestation_key = p256_key(&sec1).ok_or(CheckFailure::QuoteSignature)?;
        if !raw_signature_verifies(&attestation_key, &self.signed, &self.quote_signature) {
            return Err(CheckFailure::QuoteSignature);
        }
        // 8
        let pck =
            pck_chain_key(&self.pck_chain_pem, at, pinned_root).map_err(CheckFailure::PckChain)?;
        // 6
        if !raw_signature_verifies(&pck, &self.qe_report, &self.qe_report_signature) {
            return Err(CheckFailure::QeReportSignature);
        }
        // 7
        let mut expected = Sha256::new();
        expected.update(self.attestation_key);
        expected.update(&self.qe_auth_data);
        let qe_report_data = &self.qe_report[QE_REPORT_DATA_AT..QE_REPORT_DATA_AT + 64];
        if qe_report_data[..32] != expected.finalize()[..]
            || qe_report_data[32..].iter().any(|&b| b != 0)
        {
            return Err(CheckFailure::AttestationKeyBinding);
        }
        Ok(())
    }
}

/// Check 8: the PEM chain, leaf first, ends in the pinned root; every
/// certificate is signed (ECDSA P-256 with SHA-256) by the next, the root by
/// itself; and every certificate is valid at `at`. Returns the leaf's (PCK) key.
fn pck_chain_key(
    pem: &[u8],
    at: u64,
    pinned_root: &[u8; 32],
) -> Result<p256::ecdsa::VerifyingKey, String> {
    use x509_cert::der::Encode;
    const ECDSA_WITH_SHA256: x509_cert::der::oid::ObjectIdentifier =
        x509_cert::der::oid::ObjectIdentifier::new_unwrap("1.2.840.10045.4.3.2");
    let certs = x509_cert::Certificate::load_pem_chain(pem)
        .map_err(|e| format!("PCK chain is not PEM certificates: {e}"))?;
    if certs.len() < 2 {
        return Err(format!(
            "PCK chain has {} certificate(s), not a leaf and a root",
            certs.len()
        ));
    }
    let root = certs.last().expect("at least two");
    let root_der = root.to_der().map_err(|e| e.to_string())?;
    if Sha256::digest(&root_der)[..] != pinned_root[..] {
        return Err("PCK chain does not end in the pinned root CA".into());
    }
    let key_of = |cert: &x509_cert::Certificate| {
        p256_key(
            cert.tbs_certificate
                .subject_public_key_info
                .subject_public_key
                .raw_bytes(),
        )
        .ok_or_else(|| format!("{} has no P-256 key", cert.tbs_certificate.subject))
    };
    for (i, cert) in certs.iter().enumerate() {
        let issuer = certs.get(i + 1).unwrap_or(root);
        let name = cert.tbs_certificate.subject.to_string();
        if cert.signature_algorithm.oid != ECDSA_WITH_SHA256 {
            return Err(format!("{name} is not signed with ECDSA P-256 / SHA-256"));
        }
        let tbs = cert.tbs_certificate.to_der().map_err(|e| e.to_string())?;
        let signature = cert
            .signature
            .as_bytes()
            .and_then(|der| p256::ecdsa::Signature::from_der(der).ok())
            .ok_or_else(|| format!("{name}'s signature is not an ECDSA signature"))?;
        {
            use p256::ecdsa::signature::Verifier;
            if key_of(issuer)?.verify(&tbs, &signature).is_err() {
                return Err(format!("{name}'s signature does not verify"));
            }
        }
        let validity = &cert.tbs_certificate.validity;
        let (from, to) = (
            validity.not_before.to_unix_duration().as_secs(),
            validity.not_after.to_unix_duration().as_secs(),
        );
        if at < from || at > to {
            return Err(format!("{name} is not valid at the record's time"));
        }
    }
    key_of(&certs[0])
}

/// A `(type u16, size u32, data)` certification data block at `at`.
fn cert_data(b: &[u8], at: usize) -> Option<(u16, &[u8])> {
    let kind = le16(b, at)?;
    let size = le32(b, at + 2)? as usize;
    let start = at.checked_add(6)?;
    Some((kind, b.get(start..start.checked_add(size)?)?))
}

// ---------------------------------------------------------------------------
// Taking a quote
// ---------------------------------------------------------------------------

/// Why no quote was taken.
#[derive(Debug, PartialEq, Eq)]
pub enum NoQuote {
    /// This host has no quote interface: not a TEE.
    NoTee,
    /// The interface is there but no quote came back.
    Failed(String),
}

/// Where a quote comes from, given its 64 REPORT_DATA bytes.
pub trait QuoteSource: Send + Sync {
    /// Whether this host has a quote interface at all.
    fn available(&self) -> bool;
    fn quote(&self, report_data: &[u8; 64]) -> Result<Vec<u8>, NoQuote>;
}

/// The kernel's configfs-tsm report interface (`Documentation/ABI/testing/
/// configfs-tsm-report`): make an entry, write REPORT_DATA to `inblob`, read
/// the quote from `outblob`, remove the entry. The plugin's user needs write
/// access to the report directory; nothing else is asked of the host.
pub struct ConfigfsTsm {
    pub root: PathBuf,
    /// Whether this host's report provider is `tdx_guest` and this user can
    /// use it, found once ([`ConfigfsTsm::probe`]).
    usable: std::sync::OnceLock<bool>,
}

pub const TSM_REPORT_DIR: &str = "/sys/kernel/config/tsm/report";

impl Default for ConfigfsTsm {
    fn default() -> Self {
        Self {
            root: PathBuf::from(TSM_REPORT_DIR),
            usable: std::sync::OnceLock::new(),
        }
    }
}

impl ConfigfsTsm {
    fn entry(&self) -> PathBuf {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        self.root.join(format!(
            "capsules-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ))
    }

    /// A kernel built with configfs-tsm has the report directory whether or
    /// not it runs in a TD, so the directory alone says nothing. A TDX host
    /// is one whose report entry names the `tdx_guest` provider; this makes
    /// one entry, reads its provider and removes it. A TDX host this user
    /// cannot make an entry on is said once in the log.
    fn probe(&self) -> bool {
        if !self.root.is_dir() {
            return false;
        }
        let entry = self.entry();
        if let Err(error) = std::fs::create_dir(&entry) {
            if error.kind() == std::io::ErrorKind::PermissionDenied {
                tracing::info!(dir = %self.root.display(), %error, "this user cannot take TEE quotes here; records carry no TEE quote");
            }
            return false;
        }
        let provider = std::fs::read_to_string(entry.join("provider")).unwrap_or_default();
        let _ = std::fs::remove_dir(&entry);
        provider.trim() == "tdx_guest"
    }
}

impl QuoteSource for ConfigfsTsm {
    fn available(&self) -> bool {
        *self.usable.get_or_init(|| self.probe())
    }

    fn quote(&self, report_data: &[u8; 64]) -> Result<Vec<u8>, NoQuote> {
        if !self.available() {
            return Err(NoQuote::NoTee);
        }
        let entry = self.entry();
        std::fs::create_dir(&entry).map_err(|e| failed("make a report entry", &e))?;
        let quote = read_quote(&entry, report_data);
        let _ = std::fs::remove_dir(&entry);
        quote
    }
}

fn failed(step: &str, error: &std::io::Error) -> NoQuote {
    NoQuote::Failed(format!("could not {step}: {error}"))
}

fn read_quote(entry: &Path, report_data: &[u8; 64]) -> Result<Vec<u8>, NoQuote> {
    let provider = std::fs::read_to_string(entry.join("provider"))
        .map_err(|e| failed("read the report provider", &e))?;
    if provider.trim() != "tdx_guest" {
        return Err(NoQuote::Failed(format!(
            "the report provider is {:?}, not tdx_guest",
            provider.trim()
        )));
    }
    std::fs::write(entry.join("inblob"), report_data)
        .map_err(|e| failed("write REPORT_DATA", &e))?;
    let generation =
        |when| std::fs::read_to_string(entry.join("generation")).map_err(|e| failed(when, &e));
    let before = generation("read the generation")?;
    let quote = std::fs::read(entry.join("outblob")).map_err(|e| failed("read the quote", &e))?;
    if generation("read the generation again")? != before {
        return Err(NoQuote::Failed(
            "the report entry changed while the quote was read".into(),
        ));
    }
    Ok(quote)
}

// ---------------------------------------------------------------------------
// The slot
// ---------------------------------------------------------------------------

/// The empty slot, unchanged from before u229, with `context` saying why.
pub fn empty(context: &str) -> Value {
    json!({
        "type": names::TYPE,
        "measurement_class": null,
        "digest": null,
        "context": context,
    })
}

fn filled(quote: &[u8], report_data: &[u8; 64], client_chose_nonce: bool) -> Value {
    let fresh = if client_chose_nonce {
        " The client supplied the nonce, so the quote is fresh against the client's challenge."
    } else {
        ""
    };
    json!({
        "type": names::TYPE,
        "measurement_class": names::MEASUREMENT_CLASS,
        "quote_format": names::QUOTE_FORMAT,
        "digest_alg": names::DIGEST_ALG,
        "digest": hex::encode(Sha256::digest(quote)),
        "quote": hex::encode(quote),
        "report_data": hex::encode(report_data),
        "binding": {"tag": names::TAG, "inputs": names::INPUTS},
        "context": format!(
            "tee_measured, the producer's claim: a TDX quote taken for this record, its REPORT_DATA binding the request and answer digests, the key that signs this record, and bound to this exchange's nonce.{fresh} {}",
            names::LEFT_TO_READER
        ),
    })
}

/// The slot for one record about to be sealed: a quote bound to these
/// values when `source` gives one that parses and carries exactly the
/// requested REPORT_DATA, else the empty slot saying why. On a host with no
/// quote interface the slot is exactly the one recorded before quotes
/// existed. No quote is asked for unless every input is present and well
/// formed.
pub fn slot_for(
    source: &dyn QuoteSource,
    agent_input_digest: Option<&str>,
    agent_output_digest: Option<&str>,
    client_nonce: Option<ClientNonce<'_>>,
    producer_public_key: &[u8; 32],
) -> Value {
    if !source.available() {
        return empty(names::NO_TEE);
    }
    let wanted = match report_data(
        agent_input_digest,
        agent_output_digest,
        client_nonce,
        producer_public_key,
    ) {
        Ok(wanted) => wanted,
        Err(input) if input == names::INPUTS[2] => return empty(names::NO_CLIENT_NONCE),
        Err(input) => {
            return empty(&format!(
                "no TEE quote taken: this record has no usable {input} (64 lowercase hex) to bind; recorded absent, never fabricated"
            ))
        }
    };
    let quote = match source.quote(&wanted) {
        Ok(quote) => quote,
        Err(NoQuote::NoTee) => return empty(names::NO_TEE),
        Err(NoQuote::Failed(why)) => {
            tracing::warn!(%why, "no TEE quote for this record");
            return empty(&format!(
                "no TEE quote taken: {why}; recorded absent, never fabricated"
            ));
        }
    };
    match TdxQuote::parse(&quote) {
        Ok(parsed) if parsed.report_data == wanted => filled(
            &quote,
            &wanted,
            client_nonce.is_some_and(|n| n.client_chose()),
        ),
        Ok(_) => empty(
            "a TEE quote was taken but did not carry this record's REPORT_DATA; recorded absent",
        ),
        Err(why) => empty(&format!("a TEE quote was taken but {why}; recorded absent")),
    }
}

// ---------------------------------------------------------------------------
// The checks
// ---------------------------------------------------------------------------

/// The outcome of [`check`] on one sealed record.
#[derive(Debug, PartialEq, Eq)]
pub enum Checked {
    /// The slot is empty: nothing claimed, nothing to check.
    NotFilled,
    /// All four checks passed: the quote is bound to this record.
    Bound,
    /// A check failed. Never shown as a pass.
    Failed(CheckFailure),
}

/// Which check failed.
#[derive(Debug, PartialEq, Eq)]
pub enum CheckFailure {
    /// The slot claims a quote but its members are not exactly u229's.
    Shape(&'static str),
    /// Check 1: `digest` is not the SHA-256 of the quote bytes.
    Digest,
    /// Check 2: the quote does not parse as `tdx-dcap-v4`.
    Parse(String),
    /// Check 3: the quote's REPORT_DATA, `report_data` and the recomputation
    /// over the record's fields are not all equal.
    ReportData,
    /// Check 4: the bound key is not the key that verifies the record.
    ProducerKey,
    /// Check 5: the quote's signature by its attestation key does not verify.
    QuoteSignature,
    /// Check 6: the QE report's signature by the PCK key does not verify.
    QeReportSignature,
    /// Check 7: the QE report does not bind the attestation key.
    AttestationKeyBinding,
    /// Check 8: the PCK chain does not verify to the pinned root at the
    /// record's time.
    PckChain(String),
}

/// What the Evidence page shows for `record`'s slot: `state` is `absent`
/// (with the slot's own `context`), `bound`, or `failed` (with the failed
/// check). The page holds the words; this is the outcome of [`check`].
pub fn page_fact(record: &Value) -> Value {
    page_fact_pinned(record, &PINNED_ROOT_CA_SHA256)
}

fn page_fact_pinned(record: &Value, pinned_root: &[u8; 32]) -> Value {
    match check_pinned(record, pinned_root) {
        Checked::NotFilled => json!({
            "state": "absent",
            "context": poc(record)
                .and_then(|p| p.pointer("/evidence_refs/tee_attestation/context"))
                .cloned()
                .unwrap_or(Value::Null),
        }),
        Checked::Bound => json!({"state": "bound"}),
        Checked::Failed(failure) => json!({
            "state": "failed",
            "check": match failure {
                CheckFailure::Shape(_) => "shape",
                CheckFailure::Digest => "digest",
                CheckFailure::Parse(_) => "parse",
                CheckFailure::ReportData => "report_data",
                CheckFailure::ProducerKey => "producer_key",
                CheckFailure::QuoteSignature => "quote_signature",
                CheckFailure::QeReportSignature => "qe_report_signature",
                CheckFailure::AttestationKeyBinding => "attestation_key_binding",
                CheckFailure::PckChain(_) => "pck_chain",
            },
        }),
    }
}

fn poc(record: &Value) -> Option<&Value> {
    record.pointer("/model_attestation/compute_attestation/x-mesh-poc-v1")
}

/// Run every check (1 to 8) on `record`'s `tee_attestation` slot, the PCK
/// chain against [`PINNED_ROOT_CA_SHA256`].
pub fn check(record: &Value) -> Checked {
    check_pinned(record, &PINNED_ROOT_CA_SHA256)
}

fn check_pinned(record: &Value, pinned_root: &[u8; 32]) -> Checked {
    let Some(slot) = poc(record).and_then(|p| p.pointer("/evidence_refs/tee_attestation")) else {
        return Checked::NotFilled;
    };
    if slot.get("quote").is_none() && slot.get("measurement_class").is_none_or(Value::is_null) {
        return Checked::NotFilled;
    }
    match run_checks(record, slot, pinned_root) {
        Ok(()) => Checked::Bound,
        Err(failure) => Checked::Failed(failure),
    }
}

fn run_checks(record: &Value, slot: &Value, pinned_root: &[u8; 32]) -> Result<(), CheckFailure> {
    let members = slot
        .as_object()
        .ok_or(CheckFailure::Shape("not an object"))?;
    if members.len() != names::MEMBERS.len()
        || !names::MEMBERS.iter().all(|m| members.contains_key(*m))
    {
        return Err(CheckFailure::Shape(
            "members are not exactly the ruled ones",
        ));
    }
    let text = |key: &str| slot.get(key).and_then(Value::as_str);
    if text("type") != Some(names::TYPE)
        || text("measurement_class") != Some(names::MEASUREMENT_CLASS)
        || text("quote_format") != Some(names::QUOTE_FORMAT)
        || text("digest_alg") != Some(names::DIGEST_ALG)
        || slot["binding"] != json!({"tag": names::TAG, "inputs": names::INPUTS})
    {
        return Err(CheckFailure::Shape("a fixed member has another value"));
    }
    let quote_hex = text("quote").ok_or(CheckFailure::Shape("quote is not a string"))?;
    if quote_hex.bytes().any(|b| b.is_ascii_uppercase()) {
        return Err(CheckFailure::Shape("quote is not lowercase hex"));
    }
    let quote = hex::decode(quote_hex).map_err(|_| CheckFailure::Shape("quote is not hex"))?;

    // 1
    if text("digest") != Some(hex::encode(Sha256::digest(&quote)).as_str()) {
        return Err(CheckFailure::Digest);
    }
    // 2
    let parsed = TdxQuote::parse(&quote).map_err(CheckFailure::Parse)?;
    // 4: the key the record is signed with, which is the key bound below.
    let key_id = record.get("key_id").and_then(Value::as_str);
    let signer = capsule_emit_lib::cose::verify_producer_envelope(record).ok();
    let producer_key = match (key_id.and_then(hex32), signer) {
        (Some(key), Some(signer)) if Some(signer.as_str()) == key_id => key,
        _ => return Err(CheckFailure::ProducerKey),
    };
    // 3
    let digest = |name: &str| {
        record
            .pointer(&format!("/model_attestation/compute_attestation/{name}"))
            .and_then(Value::as_str)
    };
    let field = |name: &str| {
        poc(record)
            .and_then(|p| p.get(name))
            .and_then(Value::as_str)
    };
    let nonce = field("client_nonce")
        .zip(field("client_nonce_source"))
        .map(|(value, source)| ClientNonce { value, source });
    let recomputed = report_data(
        digest("agent_input_digest"),
        digest("agent_output_digest"),
        nonce,
        &producer_key,
    )
    .map_err(|_| CheckFailure::ReportData)?;
    let carried = hex::encode(recomputed);
    let bound = text("report_data") == Some(carried.as_str()) && parsed.report_data == recomputed;
    if !bound && !cfg!(feature = "mutant-tee-skips-binding-check") {
        return Err(CheckFailure::ReportData);
    }
    // 5 to 8, at the record's own time.
    let at = record
        .get("timestamp")
        .and_then(Value::as_str)
        .and_then(|t| chrono::DateTime::parse_from_rfc3339(t).ok())
        .and_then(|t| u64::try_from(t.timestamp()).ok())
        .ok_or_else(|| {
            CheckFailure::PckChain("the record has no time to check the chain at".into())
        })?;
    parsed.check_signatures(at, pinned_root)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::capsule_emit::{CapsuleState, ExchangeRecord};
    use std::sync::atomic::AtomicUsize;
    use std::sync::Arc;

    /// A real TDX DCAP v4 quote (vectors/tdx/, see SOURCES.md).
    fn fixture() -> Vec<u8> {
        std::fs::read(Path::new(env!("CARGO_MANIFEST_DIR")).join("vectors/tdx/tdx_quote.bin"))
            .expect("vectors/tdx/tdx_quote.bin")
    }

    /// The real fixture with `report_data` written over its REPORT_DATA:
    /// its fields match whatever record asks, but its signature no longer
    /// covers them (check 5).
    fn spliced(report_data: &[u8; 64]) -> Vec<u8> {
        let mut quote = fixture();
        quote[REPORT_DATA_AT..SIGNED_LEN].copy_from_slice(report_data);
        quote
    }

    /// A test DCAP chain: a root, a platform CA and a PCK key, and an
    /// attestation key, from fixed scalars. Certificates are valid from a day
    /// ago for a year.
    struct TestChain {
        pck: p256::ecdsa::SigningKey,
        attestation: p256::ecdsa::SigningKey,
        pem: Vec<u8>,
        root_sha256: [u8; 32],
    }

    fn test_chain() -> &'static TestChain {
        use p256::ecdsa::SigningKey;
        use std::str::FromStr;
        use x509_cert::builder::{Builder, CertificateBuilder, Profile};
        use x509_cert::der::{Encode, EncodePem};
        use x509_cert::name::Name;
        use x509_cert::serial_number::SerialNumber;
        use x509_cert::spki::SubjectPublicKeyInfoOwned;
        use x509_cert::time::{Time, Validity};
        static CHAIN: std::sync::OnceLock<TestChain> = std::sync::OnceLock::new();
        CHAIN.get_or_init(|| {
            let key = |n: u8| SigningKey::from_slice(&[n; 32]).unwrap();
            let (root, platform, pck) = (key(1), key(2), key(3));
            let now = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap();
            let day = std::time::Duration::from_secs(86_400);
            let validity = Validity {
                not_before: Time::try_from(std::time::UNIX_EPOCH + (now - day)).unwrap(),
                not_after: Time::try_from(std::time::UNIX_EPOCH + now + 365 * day).unwrap(),
            };
            let name = |cn: &str| Name::from_str(&format!("CN={cn}")).unwrap();
            let spki =
                |k: &SigningKey| SubjectPublicKeyInfoOwned::from_key(*k.verifying_key()).unwrap();
            let build = |profile, serial: u32, subject: &str, of: &SigningKey, by: &SigningKey| {
                CertificateBuilder::new(
                    profile,
                    SerialNumber::from(serial),
                    validity,
                    name(subject),
                    spki(of),
                    by,
                )
                .unwrap()
                .build::<p256::ecdsa::DerSignature>()
                .unwrap()
            };
            let root_cert = build(Profile::Root, 1, "test root", &root, &root);
            let platform_cert = build(
                Profile::SubCA {
                    issuer: name("test root"),
                    path_len_constraint: Some(0),
                },
                2,
                "test platform CA",
                &platform,
                &root,
            );
            let pck_cert = build(
                Profile::Leaf {
                    issuer: name("test platform CA"),
                    enable_key_agreement: false,
                    enable_key_encipherment: false,
                },
                3,
                "test PCK",
                &pck,
                &platform,
            );
            let pem = [&pck_cert, &platform_cert, &root_cert]
                .iter()
                .map(|c| c.to_pem(x509_cert::der::pem::LineEnding::LF).unwrap())
                .collect::<String>()
                .into_bytes();
            TestChain {
                pck,
                attestation: key(4),
                pem,
                root_sha256: Sha256::digest(root_cert.to_der().unwrap()).into(),
            }
        })
    }

    /// A TDX DCAP v4 quote signed through [`test_chain`], carrying
    /// `report_data`. With `bind_key` false the QE report does not bind the
    /// attestation key (check 7), though every signature holds.
    fn signed_quote(report_data: &[u8; 64], bind_key: bool) -> Vec<u8> {
        use p256::ecdsa::signature::Signer;
        let chain = test_chain();
        let mut signed = Vec::new();
        signed.extend_from_slice(&4u16.to_le_bytes());
        signed.extend_from_slice(&ECDSA_P256.to_le_bytes());
        signed.extend_from_slice(&TEE_TYPE_TDX.to_le_bytes());
        signed.resize(HEADER_LEN, 0);
        signed.resize(REPORT_DATA_AT, 0);
        signed.extend_from_slice(report_data);
        let sign = |key: &p256::ecdsa::SigningKey, message: &[u8]| -> Vec<u8> {
            let signature: p256::ecdsa::Signature = key.sign(message);
            signature.to_bytes().to_vec()
        };
        let attestation_key = &chain
            .attestation
            .verifying_key()
            .to_encoded_point(false)
            .as_bytes()[1..]
            .to_vec();
        let auth = [0xaa; 32];
        let mut qe_report = vec![0u8; QE_REPORT_LEN];
        if bind_key {
            let mut binding = Sha256::new();
            binding.update(attestation_key);
            binding.update(auth);
            qe_report[QE_REPORT_DATA_AT..QE_REPORT_DATA_AT + 32]
                .copy_from_slice(&binding.finalize());
        }
        let mut outer = qe_report.clone();
        outer.extend(sign(&chain.pck, &qe_report));
        outer.extend_from_slice(&(auth.len() as u16).to_le_bytes());
        outer.extend_from_slice(&auth);
        outer.extend_from_slice(&CERT_PCK_CHAIN.to_le_bytes());
        outer.extend_from_slice(&(chain.pem.len() as u32).to_le_bytes());
        outer.extend_from_slice(&chain.pem);
        let mut section = sign(&chain.attestation, &signed);
        section.extend_from_slice(attestation_key);
        section.extend_from_slice(&CERT_QE_REPORT.to_le_bytes());
        section.extend_from_slice(&(outer.len() as u32).to_le_bytes());
        section.extend(outer);
        let mut quote = signed;
        quote.extend_from_slice(&(section.len() as u32).to_le_bytes());
        quote.extend(section);
        quote
    }

    /// A TEE host whose quotes carry the REPORT_DATA asked for (or, with
    /// `wrong`, another one). Counts the quotes asked for.
    #[derive(Clone, Default)]
    struct FakeTee {
        asked: Arc<AtomicUsize>,
        wrong: bool,
    }

    impl QuoteSource for FakeTee {
        fn available(&self) -> bool {
            true
        }
        fn quote(&self, report_data: &[u8; 64]) -> Result<Vec<u8>, NoQuote> {
            self.asked.fetch_add(1, Ordering::SeqCst);
            Ok(signed_quote(
                if self.wrong { &[7; 64] } else { report_data },
                true,
            ))
        }
    }

    /// Every check, the chain pinned to [`test_chain`]'s root.
    fn checked(record: &Value) -> Checked {
        check_pinned(record, &test_chain().root_sha256)
    }

    fn unix(time: &str) -> u64 {
        chrono::DateTime::parse_from_rfc3339(time)
            .unwrap()
            .timestamp() as u64
    }

    /// When the fixture quote was taken.
    const TAKEN: &str = "2026-10-12T15:04:16Z";

    /// Run by hand over a record a live TDX run sealed:
    /// `TEE_RECORD=<record.json> cargo test --locked --bin capsules -- --ignored a_live_records_quote_is_bound`.
    /// Every check, against the pinned Intel root, on whatever machine runs it.
    #[test]
    #[ignore = "needs TEE_RECORD: a record a live TDX run sealed"]
    fn a_live_records_quote_is_bound() {
        let path = std::env::var("TEE_RECORD").expect("TEE_RECORD names a record file");
        let record: Value = serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
        assert_eq!(check(&record), Checked::Bound);
        assert_eq!(page_fact(&record), json!({"state": "bound"}));
    }

    #[test]
    fn checks_5_to_8_pass_on_a_real_quote_at_its_time_against_the_pinned_root() {
        let quote = TdxQuote::parse(&fixture()).unwrap();
        assert_eq!(
            quote.check_signatures(unix(TAKEN), &PINNED_ROOT_CA_SHA256),
            Ok(())
        );
    }

    #[test]
    fn check_5_a_quote_whose_signed_bytes_changed_fails() {
        let mut bytes = fixture();
        bytes[REPORT_DATA_AT] ^= 1;
        let quote = TdxQuote::parse(&bytes).unwrap();
        assert_eq!(
            quote.check_signatures(unix(TAKEN), &PINNED_ROOT_CA_SHA256),
            Err(CheckFailure::QuoteSignature)
        );
    }

    #[test]
    fn check_6_a_qe_report_the_pck_key_did_not_sign_fails() {
        let mut bytes = fixture();
        bytes[SIGNED_LEN + 4 + SIG_AND_KEY_LEN + 6] ^= 1; // the QE report's first byte
        let quote = TdxQuote::parse(&bytes).unwrap();
        assert_eq!(
            quote.check_signatures(unix(TAKEN), &PINNED_ROOT_CA_SHA256),
            Err(CheckFailure::QeReportSignature)
        );
    }

    #[test]
    fn check_7_a_qe_report_that_does_not_bind_the_attestation_key_fails() {
        let chain = test_chain();
        let now = unix(&chrono::Utc::now().to_rfc3339());
        let bound = TdxQuote::parse(&signed_quote(&[1; 64], true)).unwrap();
        assert_eq!(bound.check_signatures(now, &chain.root_sha256), Ok(()));
        let unbound = TdxQuote::parse(&signed_quote(&[1; 64], false)).unwrap();
        assert_eq!(
            unbound.check_signatures(now, &chain.root_sha256),
            Err(CheckFailure::AttestationKeyBinding)
        );
    }

    #[test]
    fn check_8_a_chain_to_another_root_or_outside_its_validity_fails() {
        let real = TdxQuote::parse(&fixture()).unwrap();
        let chain_failure = |quote: &TdxQuote, at: u64, root: &[u8; 32]| {
            matches!(
                quote.check_signatures(at, root),
                Err(CheckFailure::PckChain(_))
            )
        };
        assert!(
            chain_failure(&real, unix(TAKEN), &[0; 32]),
            "another pinned root"
        );
        assert!(
            chain_failure(&real, unix("2034-01-01T00:00:00Z"), &PINNED_ROOT_CA_SHA256),
            "after the PCK certificate expires"
        );
        assert!(
            chain_failure(&real, unix("2026-08-01T00:00:00Z"), &PINNED_ROOT_CA_SHA256),
            "before the PCK certificate starts"
        );
        for at in [0, 1] {
            use x509_cert::der::EncodePem;
            let mut certs = x509_cert::Certificate::load_pem_chain(&real.pck_chain_pem).unwrap();
            certs[at].tbs_certificate.serial_number =
                x509_cert::serial_number::SerialNumber::from(42u32);
            let pem: String = certs
                .iter()
                .map(|c| c.to_pem(x509_cert::der::pem::LineEnding::LF).unwrap())
                .collect();
            assert!(
                pck_chain_key(pem.as_bytes(), unix(TAKEN), &PINNED_ROOT_CA_SHA256).is_err(),
                "certificate {at} changed after it was signed"
            );
        }
        let test_signed = TdxQuote::parse(&signed_quote(&[1; 64], true)).unwrap();
        let now = unix(&chrono::Utc::now().to_rfc3339());
        assert!(
            chain_failure(&test_signed, now, &PINNED_ROOT_CA_SHA256),
            "a self-made root is not the pinned one"
        );
    }

    #[test]
    fn a_record_carrying_a_spliced_quote_is_not_bound() {
        // Its fields all match the record (checks 1 to 4); only its
        // signature (check 5) shows it was not made for this record.
        struct Splicer;
        impl QuoteSource for Splicer {
            fn available(&self) -> bool {
                true
            }
            fn quote(&self, report_data: &[u8; 64]) -> Result<Vec<u8>, NoQuote> {
                Ok(spliced(report_data))
            }
        }
        let dir = tempfile::tempdir().unwrap();
        let mut state = CapsuleState::open(dir.path(), "splice-node").unwrap();
        state.set_quote_source(Box::new(Splicer));
        let record = state
            .emit_for_exchange(&ExchangeRecord {
                model: "m",
                client_nonce: Some(NONCE),
                request_bytes: b"{}",
                response_bytes: b"{}",
                latency_ms: 1.0,
                exchange_id: None,
                requesting_party: None,
                host_provenance: None,
            })
            .unwrap()
            .capsule;
        assert_eq!(
            check(&record),
            Checked::Failed(CheckFailure::QuoteSignature)
        );
    }

    struct NoTeeHost;
    impl QuoteSource for NoTeeHost {
        fn available(&self) -> bool {
            false
        }
        fn quote(&self, _: &[u8; 64]) -> Result<Vec<u8>, NoQuote> {
            panic!("a host with no quote interface is never asked for a quote")
        }
    }

    const NONCE: &str = "abababababababababababababababababababababababababababababababab";
    const KEY: [u8; 32] = [0x44; 32];

    fn hex64(byte: &str) -> String {
        byte.repeat(32)
    }

    #[test]
    fn the_fixture_quote_parses_and_its_report_data_is_where_the_layout_says() {
        let quote = fixture();
        let parsed = TdxQuote::parse(&quote).expect("a real quote parses");
        assert_eq!(parsed.report_data[..], quote[568..632]);
        assert_ne!(parsed.report_data, [0; 64]);
    }

    #[test]
    fn a_quote_that_is_not_tdx_dcap_v4_does_not_parse() {
        type Damage = Box<dyn Fn(&mut Vec<u8>)>;
        let cases: Vec<(&str, Damage)> = vec![
            ("version", Box::new(|q| q[0] = 3)),
            ("key type", Box::new(|q| q[2] = 3)),
            ("tee type", Box::new(|q| q[4] = 0)),
            ("truncated", Box::new(|q| q.truncate(700))),
            ("trailing byte", Box::new(|q| *q.last_mut().unwrap() = 1)),
            (
                "outer cert type",
                Box::new(|q| q[SIGNED_LEN + 4 + SIG_AND_KEY_LEN] = 5),
            ),
            ("section length", Box::new(|q| q[SIGNED_LEN] ^= 1)),
        ];
        for (what, damage) in cases {
            let mut quote = fixture();
            damage(&mut quote);
            assert!(TdxQuote::parse(&quote).is_err(), "{what}: still parses");
        }
        assert!(TdxQuote::parse(&[]).is_err());
    }

    const UUID: &str = "3f1e2a2c-6b3e-4d2f-9a1b-0c8d7e6f5a4b";

    fn forwarded(value: &str) -> Option<ClientNonce<'_>> {
        Some(ClientNonce {
            value,
            source: "host_forwarded_nonce",
        })
    }

    #[test]
    fn report_data_is_the_ruled_construction() {
        // Computed outside this code: hashlib.sha512(tag + b"\0" + 11*32 + 22*32
        // + sha256(UUID.encode()) + 44*32).
        let expected = "2353ad70c837564822d322776d87468a97d30b797350af7019450077446a2509af0cad5f981dab5ed8923e1ee76f8a7f421908870ec332b00338ed166d6b179d";
        let got = report_data(
            Some(&hex64("11")),
            Some(&hex64("22")),
            forwarded(UUID),
            &[0x44; 32],
        )
        .unwrap();
        assert_eq!(hex::encode(got), expected);
    }

    #[test]
    fn the_nonce_is_hashed_exactly_as_recorded() {
        let at = |value: &str| {
            report_data(
                Some(&hex64("11")),
                Some(&hex64("22")),
                forwarded(value),
                &KEY,
            )
        };
        let recorded = at(UUID).unwrap();
        assert_ne!(at(&format!(" {UUID}")).unwrap(), recorded, "no trimming");
        assert_ne!(
            at(&UUID.to_uppercase()).unwrap(),
            recorded,
            "no case folding"
        );
        assert_ne!(
            at(&UUID.replace('-', "")).unwrap(),
            recorded,
            "no UUID re-formatting"
        );
    }

    #[test]
    fn report_data_refuses_digests_that_are_not_64_lowercase_hex() {
        let good = hex64("11");
        let short = "1".repeat(63);
        let upper = "A".repeat(64);
        for bad in [
            None,
            Some(short.as_str()),
            Some(upper.as_str()),
            Some("unknown-request:m"),
        ] {
            assert_eq!(
                report_data(bad, Some(&good), forwarded(UUID), &KEY),
                Err("agent_input_digest")
            );
            assert_eq!(
                report_data(Some(&good), bad, forwarded(UUID), &KEY),
                Err("agent_output_digest")
            );
        }
    }

    #[test]
    fn only_a_real_client_nonce_can_be_bound() {
        let good = hex64("11");
        let long = "n".repeat(257);
        let at = |nonce| report_data(Some(&good), Some(&good), nonce, &KEY);
        let not_real = [
            None,
            forwarded(""),
            forwarded(&long),
            Some(ClientNonce {
                value: "host-served-no-nonce",
                source: "host_served_observed",
            }),
            Some(ClientNonce {
                value: "plugin_generated",
                source: "plugin_generated_fallback",
            }),
            Some(ClientNonce {
                value: UUID,
                source: "host_served_observed",
            }),
        ];
        for nonce in not_real {
            assert_eq!(at(nonce), Err("client_nonce_sha256"), "{nonce:?}");
        }
        let max = "n".repeat(256);
        assert!(at(forwarded(&max)).is_ok(), "256 bytes is allowed");
        assert!(at(Some(ClientNonce {
            value: UUID,
            source: "client_supplied"
        }))
        .is_ok());
    }

    #[test]
    fn a_filled_slot_has_exactly_the_ruled_members() {
        let tee = FakeTee::default();
        let slot = slot_for(
            &tee,
            Some(&hex64("11")),
            Some(&hex64("22")),
            forwarded(UUID),
            &KEY,
        );
        let keys: Vec<&str> = slot
            .as_object()
            .unwrap()
            .keys()
            .map(String::as_str)
            .collect();
        let mut ruled = names::MEMBERS.to_vec();
        ruled.sort_unstable();
        let mut got = keys.clone();
        got.sort_unstable();
        assert_eq!(got, ruled);
        let quote = hex::decode(slot["quote"].as_str().unwrap()).unwrap();
        assert_eq!(slot["digest"], hex::encode(Sha256::digest(&quote)));
        assert_eq!(slot["measurement_class"], "tee_measured");
        assert_eq!(slot["binding"]["tag"], "x-mesh-poc-v1/tee-binding/v1");
        assert!(slot["context"]
            .as_str()
            .unwrap()
            .contains(names::LEFT_TO_READER));
    }

    #[test]
    fn no_quote_is_taken_without_a_usable_nonce_and_a_host_with_no_tee_records_todays_slot() {
        let tee = FakeTee::default();
        let placeholders = [
            None,
            Some(ClientNonce {
                value: "host-served-no-nonce",
                source: "host_served_observed",
            }),
            Some(ClientNonce {
                value: "plugin_generated",
                source: "plugin_generated_fallback",
            }),
        ];
        for nonce in placeholders {
            let slot = slot_for(&tee, Some(&hex64("11")), Some(&hex64("22")), nonce, &KEY);
            assert_eq!(slot, empty(names::NO_CLIENT_NONCE), "{nonce:?}");
            assert!(slot["context"]
                .as_str()
                .unwrap()
                .contains("no client nonce"));
        }
        assert_eq!(
            tee.asked.load(Ordering::SeqCst),
            0,
            "never asked for a quote"
        );
        let slot = slot_for(
            &NoTeeHost,
            Some(&hex64("11")),
            Some(&hex64("22")),
            forwarded(UUID),
            &KEY,
        );
        assert_eq!(slot, empty(names::NO_TEE));
        let wrong = FakeTee {
            wrong: true,
            ..FakeTee::default()
        };
        let slot = slot_for(
            &wrong,
            Some(&hex64("11")),
            Some(&hex64("22")),
            forwarded(UUID),
            &KEY,
        );
        assert!(
            slot["measurement_class"].is_null(),
            "a quote for another value is not recorded"
        );
    }

    #[test]
    fn a_report_directory_that_is_not_tdx_is_no_tee_and_leaves_nothing_behind() {
        // A kernel with configfs-tsm has the directory on any host; here an
        // entry can be made but names no `tdx_guest` provider.
        let root = tempfile::tempdir().unwrap();
        let tsm = ConfigfsTsm {
            root: root.path().to_path_buf(),
            usable: std::sync::OnceLock::new(),
        };
        assert!(!tsm.available());
        assert_eq!(tsm.quote(&[0; 64]), Err(NoQuote::NoTee));
        let slot = slot_for(
            &tsm,
            Some(&hex64("11")),
            Some(&hex64("22")),
            forwarded(UUID),
            &KEY,
        );
        assert_eq!(
            slot,
            empty(names::NO_TEE),
            "exactly the slot recorded before quotes"
        );
        assert_eq!(
            std::fs::read_dir(root.path()).unwrap().count(),
            0,
            "the probe entry is removed"
        );
        let missing = ConfigfsTsm {
            root: root.path().join("absent"),
            usable: std::sync::OnceLock::new(),
        };
        assert!(!missing.available());
    }

    #[test]
    fn the_context_claims_freshness_only_for_a_nonce_the_client_chose() {
        let tee = FakeTee::default();
        let context = |source| {
            slot_for(
                &tee,
                Some(&hex64("11")),
                Some(&hex64("22")),
                Some(ClientNonce {
                    value: UUID,
                    source,
                }),
                &KEY,
            )["context"]
                .as_str()
                .unwrap()
                .to_string()
        };
        let forwarded = context("host_forwarded_nonce");
        assert!(forwarded.contains("bound to this exchange's nonce"));
        assert!(
            !forwarded.contains("fresh"),
            "the host may have minted it: {forwarded}"
        );
        let chosen = context("client_supplied");
        assert!(chosen.contains("bound to this exchange's nonce"));
        assert!(chosen.contains("fresh against the client's challenge"));
    }

    /// One record sealed by a node on a TEE host, with `prompt` as its
    /// request, and that node's signing key.
    fn sealed(dir: &Path, prompt: &str) -> (Value, ed25519_dalek::SigningKey) {
        let mut state = CapsuleState::open(dir, "tee-node").unwrap();
        state.set_quote_source(Box::new(FakeTee::default()));
        let request =
            format!(r#"{{"model":"m","messages":[{{"role":"user","content":"{prompt}"}}]}}"#);
        let record = state
            .emit_for_exchange(&ExchangeRecord {
                model: "m",
                client_nonce: Some(NONCE),
                request_bytes: request.as_bytes(),
                response_bytes:
                    br#"{"id":"x","choices":[{"message":{"role":"assistant","content":"y"}}]}"#,
                latency_ms: 1.0,
                exchange_id: Some("e-1"),
                requesting_party: Some("party-1"),
                host_provenance: None,
            })
            .unwrap()
            .capsule;
        (record, state.signing_key().clone())
    }

    fn slot_mut(record: &mut Value) -> &mut Value {
        record
            .pointer_mut("/model_attestation/compute_attestation/x-mesh-poc-v1/evidence_refs/tee_attestation")
            .unwrap()
    }

    /// Give a changed record a fresh id and sign it again with `key`, so the
    /// only thing wrong with it is the change under test.
    fn reseal(record: &mut Value, key: &ed25519_dalek::SigningKey) {
        record["capsule_id"] = json!(capsule_emit_lib::jcs::compute_capsule_id(record).unwrap());
        capsule_emit_lib::capsule::attach_producer_envelope(record, key).unwrap();
    }

    #[test]
    fn a_record_sealed_on_a_tee_host_carries_a_quote_bound_to_it() {
        let dir = tempfile::tempdir().unwrap();
        let (record, _) = sealed(dir.path(), "hi");
        assert_eq!(checked(&record), Checked::Bound);
    }

    #[test]
    fn a_record_without_a_quote_has_nothing_to_check() {
        let dir = tempfile::tempdir().unwrap();
        let mut state = CapsuleState::open(dir.path(), "plain-node").unwrap();
        state.set_quote_source(Box::new(NoTeeHost));
        let record = state
            .emit_for_exchange(&ExchangeRecord {
                model: "m",
                client_nonce: Some(NONCE),
                request_bytes: b"{}",
                response_bytes: b"{}",
                latency_ms: 1.0,
                exchange_id: None,
                requesting_party: None,
                host_provenance: None,
            })
            .unwrap()
            .capsule;
        assert_eq!(checked(&record), Checked::NotFilled);
    }

    #[test]
    fn check_1_a_digest_that_is_not_the_quotes_fails() {
        let dir = tempfile::tempdir().unwrap();
        let (mut record, key) = sealed(dir.path(), "hi");
        slot_mut(&mut record)["digest"] = json!(hex64("00"));
        reseal(&mut record, &key);
        assert_eq!(checked(&record), Checked::Failed(CheckFailure::Digest));
    }

    #[test]
    fn check_2_a_quote_that_does_not_parse_fails() {
        let dir = tempfile::tempdir().unwrap();
        let (mut record, key) = sealed(dir.path(), "hi");
        let slot = slot_mut(&mut record);
        let mut quote = hex::decode(slot["quote"].as_str().unwrap()).unwrap();
        quote[0] = 3;
        slot["quote"] = json!(hex::encode(&quote));
        slot["digest"] = json!(hex::encode(Sha256::digest(&quote)));
        reseal(&mut record, &key);
        assert!(matches!(
            checked(&record),
            Checked::Failed(CheckFailure::Parse(_))
        ));
    }

    #[test]
    fn check_3_a_quote_bound_to_another_record_fails() {
        let dir = tempfile::tempdir().unwrap();
        let (other, _) = sealed(dir.path(), "a different question");
        let (mut record, key) = sealed(dir.path(), "hi");
        *slot_mut(&mut record) = other
            .pointer("/model_attestation/compute_attestation/x-mesh-poc-v1/evidence_refs/tee_attestation")
            .unwrap()
            .clone();
        reseal(&mut record, &key);
        assert_eq!(checked(&record), Checked::Failed(CheckFailure::ReportData));
    }

    #[test]
    fn check_3_report_data_that_disagrees_with_the_quote_fails() {
        let dir = tempfile::tempdir().unwrap();
        let (mut record, key) = sealed(dir.path(), "hi");
        slot_mut(&mut record)["report_data"] = json!("00".repeat(64));
        reseal(&mut record, &key);
        assert_eq!(checked(&record), Checked::Failed(CheckFailure::ReportData));
    }

    #[test]
    fn check_4_a_record_not_signed_by_the_bound_key_fails() {
        let dir = tempfile::tempdir().unwrap();
        let (record, _) = sealed(dir.path(), "hi");
        let outsider = ed25519_dalek::SigningKey::from_bytes(&[5; 32]);

        // The key named is not the key that signed.
        let mut named = record.clone();
        named["key_id"] = json!(hex::encode(outsider.verifying_key().to_bytes()));
        assert_eq!(checked(&named), Checked::Failed(CheckFailure::ProducerKey));

        // Signed again outside the TD with another key: the quote binds the
        // TD's key, so the recomputation over the new key does not match.
        let mut resigned = record.clone();
        reseal(&mut resigned, &outsider);
        assert_eq!(
            checked(&resigned),
            Checked::Failed(CheckFailure::ReportData)
        );
    }

    #[test]
    fn the_page_is_told_the_outcome_of_the_checks_never_more() {
        let dir = tempfile::tempdir().unwrap();
        let (record, key) = sealed(dir.path(), "hi");
        assert_eq!(
            page_fact_pinned(&record, &test_chain().root_sha256),
            json!({"state": "bound"})
        );
        let mut broken = record.clone();
        slot_mut(&mut broken)["report_data"] = json!("00".repeat(64));
        reseal(&mut broken, &key);
        assert_eq!(
            page_fact_pinned(&broken, &test_chain().root_sha256),
            json!({"state": "failed", "check": "report_data"})
        );
        let mut empty_slot = record.clone();
        *slot_mut(&mut empty_slot) = empty(names::NO_CLIENT_NONCE);
        assert_eq!(
            page_fact_pinned(&empty_slot, &test_chain().root_sha256),
            json!({"state": "absent", "context": names::NO_CLIENT_NONCE})
        );
    }

    #[test]
    fn a_slot_with_other_members_fails_on_shape() {
        let dir = tempfile::tempdir().unwrap();
        let (mut record, key) = sealed(dir.path(), "hi");
        slot_mut(&mut record)["mrtd"] = json!("00");
        reseal(&mut record, &key);
        assert!(matches!(
            checked(&record),
            Checked::Failed(CheckFailure::Shape(_))
        ));
    }
}
