//! Runtime / binary attestation rung: measure
//! the serving binary the node actually runs and record a signed reference to
//! it in the exchange capsule's provenance, shaped like executable
//! code-signing.
//!
//! ## HONESTY GRADES — read this before trusting the rung
//!
//! Two grades ship today. Both are carried IN the signed record's
//! `measurement_class` field (not only in these docs), so a verifier reading
//! the capsule sees the honesty grade inline and cannot be told "attested
//! binary" without also being told which grade backs it.
//!
//! ### `self_measured` (the floor — every platform)
//!
//! The very process being attested hashes *its own* on-disk executable
//! ([`std::env::current_exe`]) and signs the digest with the node key it
//! already holds. It proves only:
//!
//!   > "a process holding the node key reported *this* SHA-256 for the file it
//!   >  believes is its own executable, at the moment it measured."
//!
//! It does **NOT** prove the running binary was un-tampered before it hashed
//! itself. A compromised binary can compute the hash of a *pristine* copy on
//! disk (or of anything else) and sign that. Self-measurement has no root of
//! trust below the thing doing the measuring.
//!
//! ### `os_measured` (macOS — an independent measurer beneath the process)
//!
//! On macOS the KERNEL (AMFI) independently computes and validates the
//! running process's code-directory hash (cdhash) at `exec` — a value this
//! process does not compute and cannot redirect: it is bound to the calling
//! PID, not to any path the process supplies. [`measure_self`] reads it via
//! `csops(2) CS_OPS_CDHASH` and signs it JOINTLY with the self-computed
//! `binary_sha256` (see [`BinaryAttestation::verify_signature`]), so a
//! verifier sees BOTH the OS's measurement and the process's own claim, and
//! neither can be swapped independently of the other after signing.
//!
//! `os_measured` is trustworthy only up to, and the record says so IN-BAND
//! ([`KernelAttestation`] + the `context` string in
//! [`BinaryAttestation::to_value`]):
//!   * **SIP enabled** — the whole argument rests on AMFI enforcing signature
//!     validity at `exec`; SIP-off lets a root attacker disable that.
//!   * **a non-root attacker** — root can disable SIP / bypass AMFI.
//!   * **this binary's signing strength** — a local `cargo build` is only
//!     **ad-hoc** signed (cdhash exists, but no Developer-ID / notarization /
//!     library-validation entitlement); the `ad_hoc_signature` caveat carries
//!     that honestly rather than letting `os_measured` read as a stronger
//!     claim than it is.
//!
//! `os_measured` does **NOT** prove hardware-rooted measurement — no TEE, no
//! hardware register is involved — so it must never be confused with
//! `tee_measured` (a TDX quote bound to the record, carried in the separate
//! `tee_attestation` slot: [`crate::producer::tee_attest`]). A root attacker with
//! SIP disabled (able to swap the binary under a valid code signature) is
//! the named residual only `tee_measured` closes.
//!
//! ## Graceful degradation
//!
//! [`measure_self`] NEVER fabricates a measurement. If the kernel query is
//! unavailable (non-macOS, a sandbox that denies `csops`, or the kernel
//! reporting the process's own signature as not currently valid) it degrades
//! to `self_measured` — never a zero/placeholder cdhash, never a fabricated
//! `os_measured`. If the running binary's path cannot be resolved or read at
//! all, the rung records nothing: [`measure_self`] returns `None` and the
//! capsule carries the honest empty `binary_attestation` evidence slot.

use crate::producer::keys::{key_id_of, KeyPair};
use ed25519_dalek::Signer;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Mutex, OnceLock};

/// How the binary digest in a [`BinaryAttestation`] was obtained — the honesty
/// grade of the rung. Serialized verbatim into the signed record's
/// `measurement_class` field so the grade travels WITH the attestation.
///
/// `SelfMeasured` and `OsMeasured` are both produced today — see the module
/// docs for which and when. `TeeMeasured` is the named future rung a hardware
/// TEE measurer would emit; it is declared here so the vocabulary (and the
/// fact that self-measurement is the *weakest* of the three) is legible in
/// this file, not just in docs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[allow(clippy::enum_variant_names)] // the wire values: self_measured, os_measured, tee_measured
pub enum MeasurementClass {
    /// The attested process hashed its own executable. Trustworthy only up to
    /// an OS/TEE that independently measures it (see module docs). This is
    /// the weakest grade — the cross-platform floor and the fallback when an
    /// independent measurer is unavailable.
    SelfMeasured,
    /// The kernel (AMFI, on macOS) independently measured and validated the
    /// running process's code-directory hash at `exec`. Trustworthy only up
    /// to SIP-enabled + non-root + the recorded signing strength (see module
    /// docs and [`KernelAttestation`]). NOT hardware-rooted — do not confuse
    /// with `TeeMeasured`.
    OsMeasured,
    /// A TEE measured the binary into a hardware register before it ran. NOT
    /// emitted yet (out of scope for this rung).
    TeeMeasured,
}

impl MeasurementClass {
    /// The exact string label carried in the record. `self_measured` is the
    /// deliverable honesty label the task requires; the other two are reserved
    /// for the future independent-measurer rungs.
    pub const fn as_str(self) -> &'static str {
        match self {
            MeasurementClass::SelfMeasured => "self_measured",
            MeasurementClass::OsMeasured => "os_measured",
            MeasurementClass::TeeMeasured => "tee_measured",
        }
    }
}

/// Kernel-reported code-signing facts accompanying an
/// [`MeasurementClass::OsMeasured`] attestation. Present only when the kernel
/// (AMFI) could be queried for THIS process's own signing state via
/// `csops(2)`; absent means the rung degraded to `self_measured` (see
/// [`measure_self`]).
#[derive(Clone, Debug)]
pub struct KernelAttestation {
    /// Lowercase-hex code-directory hash (cdhash) the kernel validated for the
    /// RUNNING process at `exec`, from `csops(2) CS_OPS_CDHASH` — a value this
    /// process does not compute and cannot choose; it is bound to the calling
    /// PID, not to any path the process claims.
    pub cdhash: String,
    /// How `cdhash` was obtained. Always `"csops(CS_OPS_CDHASH)"` today;
    /// recorded so a future Security-framework path
    /// (`SecCodeCopySigningInformation`) would be distinguishable if it were
    /// ever added.
    pub cdhash_source: String,
    /// `true` iff the kernel's `CS_OPS_STATUS` flags include `CS_VALID` at the
    /// moment of measurement — the kernel currently considers the running
    /// pages to match the signature. [`measure_self`] never builds an
    /// `os_measured` attestation when this would be `false`; it degrades to
    /// `self_measured` instead.
    pub kernel_valid: bool,
    /// `true` iff `CS_OPS_STATUS` includes `CS_SIGNED` — the process carries a
    /// code signature at all (an ad-hoc signature counts).
    pub kernel_signed: bool,
    /// `true` iff `CS_OPS_STATUS` includes `CS_PLATFORM_BINARY` — an Apple
    /// platform binary. Never true for a locally built binary like this
    /// crate's own test/serving binaries.
    pub platform_binary: bool,
    /// Signing-strength caveat. Conservative by construction: `true` (ad-hoc)
    /// unless the kernel positively reports `platform_binary`. A locally
    /// `cargo build`-produced binary is always ad-hoc — no Developer-ID
    /// certificate, no notarization, no library-validation entitlement — so
    /// this is `true` for the ordinary case this rung ships today. The
    /// default direction is "assume weaker", never "assume stronger".
    pub ad_hoc_signature: bool,
    /// Human-legible signing-authority label derived from `platform_binary`:
    /// `"apple_platform"` or `"unknown_local_or_ad_hoc"`. NOT a full
    /// certificate-chain identity (that would need the Security framework,
    /// out of scope for this rung) — do not read it as a verified issuer
    /// name.
    pub signing_authority: String,
    /// `true` iff `csr_get_active_config()` reports no `CSR_ALLOW_*` bits set
    /// (System Integrity Protection fully enabled) at measurement time.
    /// `false` on ANY customization, not only a full disable — the
    /// conservative direction again. `os_measured`'s entire root-of-trust
    /// argument (AMFI enforcing signature validity at `exec`) depends on SIP
    /// being enabled; this field lets a verifier see that precondition
    /// instead of assuming it.
    pub sip_enabled: bool,
}

/// A signed reference to the running serving binary, shaped like executable
/// code-signing: `sha256(binary)` — plus, when available, the kernel's own
/// cdhash — with an Ed25519 signature over the combined message by the node
/// key, carrying its own honesty grade (`measurement_class`).
///
/// Every field is a real measured/derived value — there is no fabricated
/// fallback. When the binary cannot be measured, no `BinaryAttestation` is
/// produced at all (see [`measure_self`]).
#[derive(Clone, Debug)]
pub struct BinaryAttestation {
    /// The honesty grade of this measurement: [`MeasurementClass::SelfMeasured`]
    /// or [`MeasurementClass::OsMeasured`] — see module docs.
    pub measurement_class: MeasurementClass,
    /// File name (basename only) of the executable that was hashed, from the
    /// process's own `current_exe`. Recorded so a reader knows WHICH file was
    /// measured; the directory is dropped because the sealed record travels
    /// to counterparties and an absolute path exposes the operator's home
    /// directory. The digest, not the name, identifies the binary.
    pub binary_path: String,
    /// Lowercase-hex SHA-256 of the executable's bytes, computed by THIS
    /// process (the `self_measured` claim). Always present, even when
    /// `kernel_signing` is also present, so a reader sees BOTH the OS
    /// measurement and the process's own claim side by side.
    pub binary_sha256: String,
    /// Byte length of the executable that was hashed — a cheap cross-check a
    /// verifier can use alongside the digest.
    pub binary_size_bytes: u64,
    /// Signature algorithm: `ed25519` (the node key's algorithm).
    pub signature_algorithm: String,
    /// Lowercase-hex Ed25519 signature by the node key over
    /// [`signed_message`] of this attestation — the code-signing gesture:
    /// "this node vouches it observed this binary hash [and this kernel
    /// cdhash]". Hex to match every other digest/id in this codebase (no
    /// base64 runtime dep).
    pub signature: String,
    /// `key_id` of the node key that signed — the same 16-hex-char scheme used
    /// across capsule-anchor and this crate's `keys::key_id_of`, so a verifier
    /// can tie the signature to the capsule's signing key.
    pub signer_key_id: String,
    /// Kernel-reported code-signing facts, present iff
    /// `measurement_class == OsMeasured`. `None` for `self_measured`.
    pub kernel_signing: Option<KernelAttestation>,
    /// When the measurement was taken (ISO-8601 UTC). A self-measurement is a
    /// point-in-time observation, not a standing guarantee.
    pub measured_at: String,
}

/// The exact bytes signed for a [`BinaryAttestation`]: the self-measured hex
/// digest alone for `self_measured`, or `"<digest>:<cdhash>"` when a kernel
/// measurement is present — binding BOTH claims into one signature so
/// tampering with either after signing breaks verification. Shared by the
/// signer ([`build_attestation`]) and [`BinaryAttestation::verify_signature`]
/// so they can never drift apart.
fn signed_message(binary_sha256: &str, kernel_signing: &Option<KernelAttestation>) -> String {
    match kernel_signing {
        Some(k) => format!("{binary_sha256}:{}", k.cdhash),
        None => binary_sha256.to_string(),
    }
}

impl BinaryAttestation {
    /// Verify this attestation's node-key signature, against a candidate
    /// verifying key. Reconstructs the SAME message that was signed —
    /// [`signed_message`]: the hex digest alone for `self_measured`, or
    /// `"<digest>:<cdhash>"` for `os_measured` — so tampering with EITHER the
    /// self-reported digest or the kernel-reported cdhash after signing
    /// breaks verification. A verifier reading a sealed capsule uses this to
    /// confirm the node that signed the capsule is the same node that vouched
    /// for the binary hash (and cdhash, when present). (Malformed signature
    /// hex -> `false`, never a panic.)
    pub fn verify_signature(&self, vk: &ed25519_dalek::VerifyingKey) -> bool {
        use ed25519_dalek::Verifier;
        let Ok(sig_bytes) = hex::decode(&self.signature) else {
            return false;
        };
        let Ok(sig_arr): Result<[u8; 64], _> = sig_bytes.try_into() else {
            return false;
        };
        let signature = ed25519_dalek::Signature::from_bytes(&sig_arr);
        let message = signed_message(&self.binary_sha256, &self.kernel_signing);
        vk.verify(message.as_bytes(), &signature).is_ok()
    }

    /// The evidence-ref JSON object for this attestation, slotted into the
    /// capsule's `evidence_refs.binary_attestation`. The honesty label rides
    /// here (`measurement_class`) AND is echoed by the `type` tag, so the
    /// grade is unmissable in the sealed record. When a kernel measurement is
    /// present, its fields (`code_directory_hash`, `ad_hoc_signature`,
    /// `sip_enabled`, ...) ship alongside the self-computed digest, and
    /// `context` states the `os_measured` trust ceiling in-band.
    pub fn to_value(&self) -> Value {
        let mut v = json!({
            "type": "binary_attestation",
            "measurement_class": self.measurement_class.as_str(),
            "digest": self.binary_sha256,
            "digest_alg": "sha256",
            "binary_path": self.binary_path,
            "binary_size_bytes": self.binary_size_bytes,
            "signature": self.signature,
            "signature_algorithm": self.signature_algorithm,
            "signer_key_id": self.signer_key_id,
            "measured_at": self.measured_at,
        });
        match &self.kernel_signing {
            Some(k) => {
                v["code_directory_hash"] = json!(k.cdhash);
                v["cdhash_source"] = json!(k.cdhash_source);
                v["kernel_valid"] = json!(k.kernel_valid);
                v["kernel_signed"] = json!(k.kernel_signed);
                v["platform_binary"] = json!(k.platform_binary);
                v["ad_hoc_signature"] = json!(k.ad_hoc_signature);
                v["signing_authority"] = json!(k.signing_authority);
                v["sip_enabled"] = json!(k.sip_enabled);
                // Belt-and-suspenders honesty for a human reading the raw
                // capsule: the os_measured trust ceiling stated in-band, not
                // only in the code comments.
                v["context"] = json!(os_measured_context(k));
            }
            None => {
                v["context"] = json!(
                    "self-measured: the node hashed its own serving binary; \
trustworthy only up to an OS/TEE that independently measures it. Does NOT prove \
the binary was untampered before hashing."
                );
            }
        }
        v
    }

    /// The value for the capsule's `compute_attestation.runtime` field: the real
    /// binary digest, labeled with its grade and the runtime name, replacing the
    /// old `0*64` placeholder. Shape `"<sha256>:<class>:<runtime-name>"` keeps
    /// the existing `<hash>:<runtime>` convention while making the grade legible.
    pub fn runtime_field(&self, runtime_name: &str) -> String {
        format!(
            "{}:{}:{}",
            self.binary_sha256,
            self.measurement_class.as_str(),
            runtime_name
        )
    }

    /// The runtime/model extension draft's `compute_attestation.runtime`
    /// object shape: `{name, runtime_digest, measurement_class,
    /// platform_integrity}`, replacing the flat [`Self::runtime_field`]
    /// string on the producer paths that emit that draft's block.
    /// `platform_integrity` is present only when a kernel measurement
    /// (`os_measured`) backs it -- absent, never fabricated, for
    /// `self_measured`.
    pub fn runtime_value(&self, runtime_name: &str) -> Value {
        let mut v = json!({
            "name": runtime_name,
            "runtime_digest": self.binary_sha256,
            "measurement_class": self.measurement_class.as_str(),
        });
        if let Some(k) = &self.kernel_signing {
            v["platform_integrity"] = json!({
                "sip_enabled": k.sip_enabled,
                "source": {"sip_enabled": "os_reported"},
            });
        }
        v
    }
}

/// The `os_measured` trust-ceiling text, generated from the actual kernel
/// facts in `k` rather than hard-coded — so the wording tracks what was
/// really observed (an ad-hoc, non-platform binary gets the ad-hoc sentence;
/// a platform binary would not).
fn os_measured_context(k: &KernelAttestation) -> String {
    format!(
        "os_measured: the KERNEL (AMFI) reported this process's code-directory \
hash (cdhash) at exec time via csops(2) CS_OPS_CDHASH — an independent measurer \
beneath the process, so a compromised process cannot repoint this hash at a \
decoy file the way self-measurement can; the signature also binds this cdhash \
to the self-measured digest, so tampering with either after signing breaks \
verification. Trustworthy only up to: SIP {sip}, a NON-ROOT attacker (root can \
disable SIP / bypass AMFI), and this binary's signing strength — \
signing_authority={authority}{adhoc}. Does NOT prove hardware-rooted \
measurement: no TEE and no hardware register are involved, so this is NOT \
tee_measured. A root attacker with SIP disabled is the named residual, closed \
only by tee_measured (out of scope here).",
        sip = if k.sip_enabled {
            "enabled"
        } else {
            "NOT confirmed enabled"
        },
        authority = k.signing_authority,
        adhoc = if k.ad_hoc_signature {
            ", ad-hoc signed (no Developer-ID / notarization / library validation)"
        } else {
            ""
        },
    )
}

/// Sign the measured fields into a [`BinaryAttestation`], shared by every
/// measurement path so the signed message and the record shape can never
/// drift apart between `self_measured` and `os_measured`.
fn build_attestation(
    keys: &KeyPair,
    measurement_class: MeasurementClass,
    path: &std::path::Path,
    binary_sha256: String,
    binary_size_bytes: u64,
    kernel_signing: Option<KernelAttestation>,
    measured_at: String,
) -> BinaryAttestation {
    let message = signed_message(&binary_sha256, &kernel_signing);
    let signature = keys.signing_key.sign(message.as_bytes());
    BinaryAttestation {
        measurement_class,
        binary_path: path
            .file_name()
            .map(|name| name.to_string_lossy().into_owned())
            .unwrap_or_default(),
        binary_sha256,
        binary_size_bytes,
        signature_algorithm: "ed25519".to_string(),
        signature: hex::encode(signature.to_bytes()),
        signer_key_id: key_id_of(&keys.verifying_key()),
        kernel_signing,
        measured_at,
    }
}

/// This process's executable as it was read and hashed at its first
/// measurement: the path it ran from, its SHA-256 and its size.
struct ExeMeasurement {
    path: PathBuf,
    sha256: String,
    size: u64,
}

/// Set once, by the first measurement that succeeds; reused for the life of
/// the process.
static SELF_EXE: OnceLock<ExeMeasurement> = OnceLock::new();
/// Serialises the first measurement, so concurrent first seals read once.
static SELF_EXE_INIT: Mutex<()> = Mutex::new(());
/// How many times this process has read its own executable (see
/// [`exe_reads`]).
static EXE_READS: AtomicUsize = AtomicUsize::new(0);

/// How many times this process has read its own executable to measure it:
/// 1 once anything was sealed, and never more.
pub fn exe_reads() -> usize {
    EXE_READS.load(Ordering::SeqCst)
}

/// This process's executable, read and hashed once: at the first
/// measurement, then reused. Reading the whole executable is the expensive
/// part of a measurement (it scales with the binary's size), and the bytes a
/// running process was loaded from do not change under it, so there is
/// nothing to gain by reading it again for every record.
///
/// What is measured is the file at first use. Replacing the executable on
/// disk while the process runs is not seen until the process restarts, which
/// is also when the new binary would actually run.
///
/// A first read that fails (path unresolvable, file unreadable) is not kept:
/// the next measurement tries again.
fn self_exe() -> Option<&'static ExeMeasurement> {
    if let Some(exe) = SELF_EXE.get() {
        return Some(exe);
    }
    let _first = SELF_EXE_INIT
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    if let Some(exe) = SELF_EXE.get() {
        return Some(exe);
    }
    let path = std::env::current_exe().ok()?;
    EXE_READS.fetch_add(1, Ordering::SeqCst);
    let bytes = std::fs::read(&path).ok()?;
    let _ = SELF_EXE.set(ExeMeasurement {
        sha256: hex::encode(Sha256::digest(&bytes)),
        size: bytes.len() as u64,
        path,
    });
    SELF_EXE.get()
}

/// Measure and sign the RUNNING serving binary — `os_measured` when the
/// kernel can be queried for this process's own signing state, else
/// `self_measured`.
///
/// Resolves this process's own executable ([`std::env::current_exe`]), reads
/// and SHA-256-hashes it (always — the `self_measured` claim ships even
/// alongside `os_measured`), then tries the kernel measurement
/// ([`kernel_attest::measure_own_process`]). The executable is read and
/// hashed once per process ([`self_exe`]); the kernel is asked on every call,
/// because whether it still vouches for this process's pages can change
/// while the process runs. Returns:
///   * `Some(BinaryAttestation)` on success — `os_measured` when the kernel
///     query succeeded, else `self_measured`. Both are real, signed
///     references to the binary this node runs; neither is ever fabricated.
///   * `None` if the path can't be resolved OR the file can't be read. The rung
///     then degrades gracefully: the capsule records the attestation as ABSENT
///     rather than fabricating a hash or a path.
///
/// `measured_at` is supplied by the caller so it matches the capsule timestamp
/// clock (`timestamp::utc_now_iso8601`) and stays testable.
pub fn measure_self(keys: &KeyPair, measured_at: String) -> Option<BinaryAttestation> {
    // current_exe() can fail (sandbox denies it, /proc unavailable, exe
    // unlinked). Degrade: no attestation, never a fabricated one.
    let exe = self_exe()?;
    let kernel_signing = kernel_attest::measure_own_process();
    let measurement_class = if kernel_signing.is_some() {
        MeasurementClass::OsMeasured
    } else {
        MeasurementClass::SelfMeasured
    };
    Some(build_attestation(
        keys,
        measurement_class,
        &exe.path,
        exe.sha256.clone(),
        exe.size,
        kernel_signing,
        measured_at,
    ))
}

/// The self-measured-only core, split out so a test can point it at a known
/// file instead of the test-runner binary — and so a caller can force
/// `self_measured` for an arbitrary path, which is NOT meaningful for
/// `os_measured` (the kernel cdhash is bound to a running PID, not a path;
/// see [`kernel_attest::measure_own_process`]). Same graceful-degradation
/// contract: `None` when the file cannot be read.
pub fn measure_path(
    keys: &KeyPair,
    path: &std::path::Path,
    measured_at: String,
) -> Option<BinaryAttestation> {
    let bytes = std::fs::read(path).ok()?;
    let binary_sha256 = hex::encode(Sha256::digest(&bytes));
    Some(build_attestation(
        keys,
        MeasurementClass::SelfMeasured,
        path,
        binary_sha256,
        bytes.len() as u64,
        None,
        measured_at,
    ))
}

/// Kernel-level code-signing measurement, macOS only. Never fabricates a
/// value: every `Some` is a real kernel-reported byte string; failure at any
/// FFI step returns `None` and the caller falls back to `self_measured`.
#[cfg(target_os = "macos")]
mod kernel_attest {
    use super::KernelAttestation;
    use std::os::raw::{c_int, c_uint, c_void};

    // Constants from XNU's `bsd/sys/codesign.h` (`CS_OPS_*`, `CS_*` status
    // flags) and `bsd/sys/csr.h` (`csr_get_active_config`). Not exposed by the
    // Command Line Tools headers on this host, so declared here directly;
    // `CS_OPS_CDHASH` / `CS_VALID` / `CS_SIGNED` / `CS_PLATFORM_BINARY` and
    // `csr_get_active_config`'s "0 == fully enabled" convention were each
    // cross-checked against `codesign -dvvv` / `csrutil status` ground truth
    // on a macOS host before shipping.
    const CS_OPS_STATUS: c_uint = 0;
    const CS_OPS_CDHASH: c_uint = 5;
    const CS_CDHASH_LEN: usize = 20;
    const CS_VALID: u32 = 0x0000_0001;
    const CS_SIGNED: u32 = 0x2000_0000;
    const CS_PLATFORM_BINARY: u32 = 0x0400_0000;

    extern "C" {
        fn csops(pid: i32, ops: c_uint, useraddr: *mut c_void, usersize: usize) -> c_int;
        fn csr_get_active_config(config: *mut u32) -> c_int;
    }

    fn read_status(pid: i32) -> Option<u32> {
        let mut flags: u32 = 0;
        let rc = unsafe {
            csops(
                pid,
                CS_OPS_STATUS,
                &mut flags as *mut u32 as *mut c_void,
                std::mem::size_of::<u32>(),
            )
        };
        if rc == 0 {
            Some(flags)
        } else {
            None
        }
    }

    fn read_cdhash(pid: i32) -> Option<[u8; CS_CDHASH_LEN]> {
        let mut buf = [0u8; CS_CDHASH_LEN];
        let rc = unsafe {
            csops(
                pid,
                CS_OPS_CDHASH,
                buf.as_mut_ptr() as *mut c_void,
                buf.len(),
            )
        };
        if rc == 0 {
            Some(buf)
        } else {
            None
        }
    }

    /// `true` iff System Integrity Protection is fully enabled: the kernel
    /// reports no `CSR_ALLOW_*` bits set. Any bit set (a partial disable) OR a
    /// failed query is conservatively reported as NOT confirmed enabled.
    fn sip_enabled() -> bool {
        let mut cfg: u32 = 0;
        let rc = unsafe { csr_get_active_config(&mut cfg as *mut u32) };
        rc == 0 && cfg == 0
    }

    /// Measure the CALLING process's own kernel-validated code-signing state.
    /// `None` on ANY failure — the syscall being unavailable, a sandbox
    /// denial, or the kernel reporting `CS_VALID` unset (it does not
    /// currently vouch for this process's pages matching its signature) — so
    /// the caller degrades to `self_measured` rather than claiming
    /// `os_measured` on shaky ground.
    pub fn measure_own_process() -> Option<KernelAttestation> {
        let pid = std::process::id() as i32;
        let flags = read_status(pid)?;
        if flags & CS_VALID == 0 {
            return None;
        }
        let cdhash = read_cdhash(pid)?;
        let platform_binary = flags & CS_PLATFORM_BINARY != 0;
        Some(KernelAttestation {
            cdhash: hex::encode(cdhash),
            cdhash_source: "csops(CS_OPS_CDHASH)".to_string(),
            kernel_valid: true,
            kernel_signed: flags & CS_SIGNED != 0,
            platform_binary,
            ad_hoc_signature: !platform_binary,
            signing_authority: if platform_binary {
                "apple_platform".to_string()
            } else {
                "unknown_local_or_ad_hoc".to_string()
            },
            sip_enabled: sip_enabled(),
        })
    }
}

/// No kernel-level measurer wired up on this platform — always degrades
/// gracefully to `self_measured`. Never fabricates a value.
#[cfg(not(target_os = "macos"))]
mod kernel_attest {
    use super::KernelAttestation;

    pub fn measure_own_process() -> Option<KernelAttestation> {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ed25519_dalek::{Signature, Verifier};

    /// The rung records the real SHA-256 of the measured file AND carries the
    /// `self_measured` honesty label — in the struct, in the runtime field, and
    /// in the evidence-ref JSON — and the signature verifies under the node key.
    #[test]
    fn measures_hash_and_ships_self_measured_label_and_valid_signature() {
        let keys = KeyPair::generate();
        let tmp = std::env::temp_dir().join(format!("b3-attest-{}", std::process::id()));
        std::fs::write(&tmp, b"fake serving binary bytes").expect("write tmp binary");

        let att = measure_path(&keys, &tmp, "2026-08-30T00:00:00Z".to_string())
            .expect("measurable file yields an attestation");

        // Records the REAL hash of the file (independently recomputed here).
        let expected = hex::encode(Sha256::digest(b"fake serving binary bytes"));
        assert_eq!(att.binary_sha256, expected, "records the real binary hash");
        assert_eq!(
            att.binary_size_bytes,
            b"fake serving binary bytes".len() as u64
        );

        // THE honesty label, in every place it must ship.
        assert_eq!(att.measurement_class, MeasurementClass::SelfMeasured);
        assert_eq!(att.measurement_class.as_str(), "self_measured");
        let v = att.to_value();
        assert_eq!(v["measurement_class"], "self_measured");
        assert_eq!(v["digest"], expected);
        assert_eq!(v["type"], "binary_attestation");
        // The trust ceiling is stated in-band.
        assert!(v["context"].as_str().unwrap().contains("self-measured"));
        assert!(v["context"]
            .as_str()
            .unwrap()
            .contains("untampered before hashing"));
        // Runtime field carries the real hash + the self_measured grade.
        let rt = att.runtime_field("mesh-llm-host-runtime");
        assert!(rt.starts_with(&expected));
        assert!(rt.contains("self_measured"));
        assert!(rt.contains("mesh-llm-host-runtime"));
        assert_ne!(rt, "0".repeat(64) + ":mesh-llm-host-runtime");

        // The signature is a real Ed25519 signature over the hex digest by the
        // node key — the code-signing gesture verifies.
        let sig_bytes: [u8; 64] = hex::decode(&att.signature)
            .expect("hex sig")
            .try_into()
            .expect("64-byte sig");
        let signature = Signature::from_bytes(&sig_bytes);
        keys.verifying_key()
            .verify(att.binary_sha256.as_bytes(), &signature)
            .expect("node key must verify its own binary-hash signature");
        assert_eq!(att.signer_key_id, key_id_of(&keys.verifying_key()));

        // The crate's own verify helper agrees, and rejects a DIFFERENT key.
        assert!(att.verify_signature(&keys.verifying_key()));
        let other = KeyPair::generate();
        assert!(
            !att.verify_signature(&other.verifying_key()),
            "a different node key must NOT verify this binary-hash signature"
        );
    }

    /// Graceful degradation: an unresolvable/unreadable path yields NO
    /// attestation — never a fabricated hash or placeholder record.
    #[test]
    fn degrades_to_none_when_binary_unreadable() {
        let keys = KeyPair::generate();
        let missing = std::path::Path::new("/nonexistent/b3/serving-binary-xyz");
        assert!(
            measure_path(&keys, missing, "2026-08-30T00:00:00Z".to_string()).is_none(),
            "an unreadable binary path must record absence, never a fabricated attestation"
        );
    }

    /// `measure_path` is path-parameterized and can ONLY ever produce
    /// `self_measured` — there is no kernel cdhash for an arbitrary file (the
    /// kernel measurer is PID-bound, see [`kernel_attest`]). This holds on
    /// every platform, including macOS, so a record measured this way never
    /// claims more than `self_measured`.
    #[test]
    fn measure_path_never_produces_kernel_signing_even_on_macos() {
        let keys = KeyPair::generate();
        let tmp = std::env::temp_dir().join(format!("rung3-measure-path-{}", std::process::id()));
        std::fs::write(
            &tmp,
            b"arbitrary decoy bytes measure_path can be pointed at",
        )
        .expect("write tmp file");

        let att =
            measure_path(&keys, &tmp, "2026-08-31T00:00:00Z".to_string()).expect("measurable file");
        assert_eq!(att.measurement_class, MeasurementClass::SelfMeasured);
        assert!(att.kernel_signing.is_none());
        assert_eq!(att.to_value()["measurement_class"], "self_measured");

        let _ = std::fs::remove_file(&tmp);
    }

    /// Measuring many times reads the executable once and gives the same
    /// file measurement every time; only `measured_at` and what depends on it
    /// differ.
    #[test]
    fn measure_self_reads_the_executable_once() {
        let keys = KeyPair::generate();
        let first = measure_self(&keys, "2026-10-05T00:00:00Z".to_string()).expect("measurable");
        for n in 1..6 {
            let again = measure_self(&keys, format!("2026-10-05T00:0{n}:00Z")).expect("measurable");
            assert_eq!(again.binary_sha256, first.binary_sha256);
            assert_eq!(again.binary_size_bytes, first.binary_size_bytes);
            assert_eq!(again.binary_path, first.binary_path);
            assert_eq!(
                again.signature, first.signature,
                "the signature covers the hash, not the time"
            );
        }
        assert_eq!(exe_reads(), 1);
    }

    /// REAL kernel attestation, on the REAL kernel of this Apple-Silicon lab
    /// host — no mock. `measure_self()` must upgrade to `os_measured` (every
    /// executing process on arm64 macOS carries at least an ad-hoc cdhash,
    /// since the kernel refuses to exec unsigned code), and the reported
    /// cdhash must EXACTLY match independent ground truth from `codesign
    /// -dvvv` run against the same binary — proving the value came from the
    /// kernel, not from anything this code chose.
    #[cfg(target_os = "macos")]
    #[test]
    fn measure_self_produces_real_os_measured_matching_codesign_ground_truth() {
        let keys = KeyPair::generate();
        let att = measure_self(&keys, "2026-08-31T00:00:00Z".to_string())
            .expect("current_exe is measurable in a cargo test binary");

        assert_eq!(att.measurement_class, MeasurementClass::OsMeasured);
        let kernel = att
            .kernel_signing
            .as_ref()
            .expect("os_measured always carries kernel_signing");
        assert!(kernel.kernel_valid);
        assert_eq!(kernel.cdhash_source, "csops(CS_OPS_CDHASH)");
        assert_eq!(kernel.cdhash.len(), 40, "20-byte cdhash, hex-encoded");
        assert_ne!(kernel.cdhash, "0".repeat(40), "never a placeholder cdhash");

        // GROUND TRUTH cross-check against `codesign -dvvv` on the exact
        // running binary.
        let exe = std::env::current_exe().expect("current_exe");
        let output = std::process::Command::new("codesign")
            .args(["-dvvv", exe.to_str().expect("utf8 path")])
            .output()
            .expect("codesign must be runnable on this macOS host");
        let stderr = String::from_utf8_lossy(&output.stderr);
        let ground_truth = stderr
            .lines()
            .find_map(|l| l.strip_prefix("CDHash="))
            .expect("codesign -dvvv prints a CDHash= line");
        assert_eq!(
            kernel.cdhash, ground_truth,
            "kernel-reported cdhash must exactly match codesign's own CDHash"
        );

        // The joint signature (digest + cdhash) verifies under the node key.
        assert!(att.verify_signature(&keys.verifying_key()));
    }

    /// Signing strength must be labeled HONESTLY: a `cargo test` binary on
    /// this host is ad-hoc/linker-signed, never a platform binary, never
    /// Developer-ID/notarized — and the caveat must say so IN-BAND (not only
    /// in code comments), naming the full os_measured ceiling (SIP, root,
    /// tee_measured boundary) so a reader can never mistake this for a
    /// stronger grade.
    #[cfg(target_os = "macos")]
    #[test]
    fn os_measured_carries_honest_ad_hoc_and_ceiling_caveats() {
        let keys = KeyPair::generate();
        let att = measure_self(&keys, "2026-08-31T00:00:00Z".to_string()).expect("measurable");
        let kernel = att.kernel_signing.as_ref().expect("os_measured");

        assert!(
            !kernel.platform_binary,
            "a cargo test binary is not an Apple platform binary"
        );
        assert!(
            kernel.ad_hoc_signature,
            "cargo-built test binary must be labeled ad-hoc"
        );
        assert_eq!(kernel.signing_authority, "unknown_local_or_ad_hoc");

        let v = att.to_value();
        assert_eq!(v["measurement_class"], "os_measured");
        assert_eq!(v["ad_hoc_signature"], true);
        assert_eq!(v["platform_binary"], false);
        let context = v["context"].as_str().expect("context is a string");
        assert!(
            context.contains("ad-hoc"),
            "ad-hoc caveat must be in-band: {context}"
        );
        assert!(
            context.contains("SIP"),
            "SIP ceiling must be in-band: {context}"
        );
        assert!(
            context.contains("NON-ROOT"),
            "root ceiling must be in-band: {context}"
        );
        assert!(
            context.contains("NOT") && context.contains("tee_measured"),
            "must explicitly deny reading as tee_measured: {context}"
        );
    }

    /// The signature binds BOTH the self-measured digest AND the kernel
    /// cdhash jointly — tampering with EITHER field after signing must break
    /// verification. This is what upgrades the rung-2 attack-9 decoy trick
    /// (self_measured's residual: an internally-perfect record pointed at a
    /// pristine decoy) from residual to CAUGHT under os_measured: even if an
    /// attacker could get a kernel cdhash for a decoy (they cannot: the kernel
    /// reports the cdhash of the running process only), splicing it onto a
    /// different digest, or vice
    /// versa, invalidates the signature.
    #[cfg(target_os = "macos")]
    #[test]
    fn os_measured_signature_breaks_if_either_digest_or_cdhash_is_tampered() {
        let keys = KeyPair::generate();
        let att = measure_self(&keys, "2026-08-31T00:00:00Z".to_string()).expect("measurable");
        assert!(
            att.verify_signature(&keys.verifying_key()),
            "untampered record verifies"
        );

        let mut cdhash_tampered = att.clone();
        let mut k = cdhash_tampered.kernel_signing.clone().expect("os_measured");
        let original_cdhash = k.cdhash.clone();
        k.cdhash = "ff".repeat(20);
        assert_ne!(k.cdhash, original_cdhash);
        cdhash_tampered.kernel_signing = Some(k);
        assert!(
            !cdhash_tampered.verify_signature(&keys.verifying_key()),
            "swapping the cdhash after signing must be CAUGHT"
        );

        let mut digest_tampered = att.clone();
        digest_tampered.binary_sha256 = "ab".repeat(32);
        assert!(
            !digest_tampered.verify_signature(&keys.verifying_key()),
            "swapping the self-measured digest after signing must be CAUGHT"
        );

        // Original is untouched and still verifies (sanity: the tampering above
        // operated on clones, not the real record).
        assert!(att.verify_signature(&keys.verifying_key()));
    }

    /// Off macOS there is no kernel measurer wired up — `measure_self` must
    /// degrade to `self_measured` gracefully, never panic, never fabricate a
    /// cdhash. Runs on Linux CI, giving real portable coverage of the
    /// fallback contract (the macOS-only tests above cannot run there).
    #[cfg(not(target_os = "macos"))]
    #[test]
    fn kernel_measurement_gracefully_absent_off_macos() {
        assert!(
            super::kernel_attest::measure_own_process().is_none(),
            "off macOS there is no kernel measurer — must degrade, never fabricate"
        );
        if let Some(att) = measure_self(&KeyPair::generate(), "2026-08-31T00:00:00Z".to_string()) {
            assert_eq!(att.measurement_class, MeasurementClass::SelfMeasured);
            assert!(att.kernel_signing.is_none());
        }
    }
}
