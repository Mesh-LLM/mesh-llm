use ed25519_dalek::{Signer, SigningKey};
use getrandom::fill as fill_random;
use mesh_llm_system::embedded_release_footer::{
    EmbeddedReleaseFooterStatus, EmbeddedReleasePayloadSummary, EmbeddedReleasePayloadVerifier,
    read_embedded_release_footer, stamp_embedded_release_payload, strip_embedded_release_footer,
    verify_embedded_release_footer,
};
use serde::Deserialize;
use std::collections::BTreeSet;
use std::error::Error;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};
use std::time::{SystemTime, UNIX_EPOCH};

type DynError = Box<dyn Error>;
type DynResult<T> = Result<T, DynError>;

const RELEASE_BUILD_ATTESTATION_VERSION: u32 = 1;
const RELEASE_SIGNING_PRIVATE_KEY_KIND: &str = "mesh-llm-release-signing-private-key-v1";
const RELEASE_SIGNING_PUBLIC_KEY_KIND: &str = "mesh-llm-release-signing-public-key-v1";
const RELEASE_BUILD_ATTESTATION_DOMAIN_TAG: &[u8] = b"mesh-llm-release-attestation-v1:";
const ED25519_SIGNATURE_ALGORITHM: &str = "ed25519";

#[derive(Debug, Clone, serde::Serialize, Deserialize)]
struct ReleaseBuildAttestationClaims {
    version: u32,
    node_version: String,
    build_id: String,
    commit: String,
    target_triple: String,
    supported_protocol_generation_min: Option<u32>,
    supported_protocol_generation_max: Option<u32>,
    artifact_digest: String,
}

#[derive(Debug, Clone, serde::Serialize, Deserialize)]
struct EmbeddedReleaseAttestation {
    version: u32,
    signer_key_id: String,
    signature_algorithm: String,
    claims: ReleaseBuildAttestationClaims,
    signed_payload_hex: String,
    signature_hex: String,
}

#[derive(Debug, Clone, serde::Serialize, Deserialize)]
struct ReleaseSigningPrivateKeyFile {
    kind: String,
    version: u32,
    algorithm: String,
    signer_key_id: String,
    seed_hex: String,
}

#[derive(Debug, Clone, serde::Serialize, Deserialize)]
struct ReleaseSigningPublicKeyFile {
    kind: String,
    version: u32,
    algorithm: String,
    signer_key_id: String,
    public_key_hex: String,
}

#[derive(Debug, Clone, serde::Serialize)]
struct ReleaseAttestationInspectSummary {
    status: String,
    version: Option<u32>,
    signer_key_id: Option<String>,
    artifact_digest: Option<String>,
    error: Option<String>,
}

impl ReleaseBuildAttestationClaims {
    fn validate(&self) -> DynResult<()> {
        if self.version != RELEASE_BUILD_ATTESTATION_VERSION
            || self.node_version.trim().is_empty()
            || self.build_id.trim().is_empty()
            || self.commit.trim().is_empty()
            || self.target_triple.trim().is_empty()
            || self.artifact_digest.trim().is_empty()
        {
            return Err("invalid release build attestation shape".into());
        }
        if let (Some(min), Some(max)) = (
            self.supported_protocol_generation_min,
            self.supported_protocol_generation_max,
        ) && min > max
        {
            return Err("invalid release build attestation protocol bounds".into());
        }
        if !self.artifact_digest.starts_with("sha256:") {
            return Err("release build attestation artifact digest must start with sha256:".into());
        }
        Ok(())
    }

    fn canonical_bytes(
        &self,
        signer_key_id: &str,
        signature_algorithm: &str,
    ) -> DynResult<Vec<u8>> {
        self.validate()?;
        let mut buf = Vec::with_capacity(256);
        buf.extend_from_slice(RELEASE_BUILD_ATTESTATION_DOMAIN_TAG);
        buf.extend_from_slice(&self.version.to_le_bytes());
        write_canonical_string(&mut buf, self.node_version.trim());
        write_canonical_string(&mut buf, self.build_id.trim());
        write_canonical_string(&mut buf, self.commit.trim());
        write_canonical_string(&mut buf, self.target_triple.trim());
        write_optional_u32(&mut buf, self.supported_protocol_generation_min);
        write_optional_u32(&mut buf, self.supported_protocol_generation_max);
        write_optional_string(&mut buf, Some(self.artifact_digest.trim()));
        write_canonical_string(&mut buf, signer_key_id.trim());
        write_canonical_string(&mut buf, signature_algorithm.trim());
        Ok(buf)
    }
}

impl EmbeddedReleaseAttestation {
    fn validate(&self) -> DynResult<()> {
        if self.version != RELEASE_BUILD_ATTESTATION_VERSION
            || self.signer_key_id.trim().is_empty()
            || self.signature_algorithm.trim().is_empty()
            || self.signed_payload_hex.trim().is_empty()
            || self.signature_hex.trim().is_empty()
        {
            return Err("invalid embedded release attestation shape".into());
        }
        if self.signature_algorithm.trim() != ED25519_SIGNATURE_ALGORITHM {
            return Err("invalid embedded release attestation signature algorithm".into());
        }
        parse_release_signer_public_key(self.signer_key_id.trim())?;
        if self.signature_bytes()?.len() != 64 {
            return Err("invalid embedded release attestation signature shape".into());
        }
        let _ = self.signed_payload_bytes()?;
        self.claims.validate()?;
        Ok(())
    }

    fn signed_payload_bytes(&self) -> DynResult<Vec<u8>> {
        Ok(hex::decode(self.signed_payload_hex.trim())?)
    }

    fn signature_bytes(&self) -> DynResult<Vec<u8>> {
        Ok(hex::decode(self.signature_hex.trim())?)
    }

    fn claims(&self) -> DynResult<ReleaseBuildAttestationClaims> {
        let claims = self.claims.clone();
        claims.validate()?;
        Ok(claims)
    }

    fn canonical_hash_hex(&self) -> DynResult<String> {
        use sha2::{Digest, Sha256};

        self.validate()?;
        Ok(hex::encode(Sha256::digest(serde_json::to_vec(self)?)))
    }

    fn verify_with_public_key(
        &self,
        supplied_public_key: &ed25519_dalek::VerifyingKey,
    ) -> DynResult<ReleaseBuildAttestationClaims> {
        self.validate()?;
        let embedded_signer_public_key =
            parse_release_signer_public_key(self.signer_key_id.trim())?;
        if embedded_signer_public_key != *supplied_public_key {
            return Err("supplied public key does not match embedded signer_key_id".into());
        }
        let signature_bytes = self.signature_bytes()?;
        let signature = ed25519_dalek::Signature::from_bytes(
            &signature_bytes
                .as_slice()
                .try_into()
                .map_err(|_| "invalid embedded release attestation signature length")?,
        );
        let signed_payload_bytes = self.signed_payload_bytes()?;
        let claims = self.claims()?;
        let canonical_bytes =
            claims.canonical_bytes(&self.signer_key_id, &self.signature_algorithm)?;
        if signed_payload_bytes != canonical_bytes {
            return Err("embedded release attestation signed payload does not match claims".into());
        }
        supplied_public_key.verify_strict(&signed_payload_bytes, &signature)?;
        claims.validate()?;
        Ok(claims)
    }
}

fn write_canonical_string(buf: &mut Vec<u8>, value: &str) {
    buf.extend_from_slice(&(value.len() as u64).to_le_bytes());
    buf.extend_from_slice(value.as_bytes());
}

fn write_optional_string(buf: &mut Vec<u8>, value: Option<&str>) {
    match value {
        Some(value) => {
            buf.push(1);
            write_canonical_string(buf, value);
        }
        None => buf.push(0),
    }
}

fn write_optional_u32(buf: &mut Vec<u8>, value: Option<u32>) {
    match value {
        Some(value) => {
            buf.push(1);
            buf.extend_from_slice(&value.to_le_bytes());
        }
        None => buf.push(0),
    }
}

impl ReleaseSigningPrivateKeyFile {
    fn from_signing_key(signing_key: &SigningKey) -> Self {
        Self {
            kind: RELEASE_SIGNING_PRIVATE_KEY_KIND.to_string(),
            version: RELEASE_BUILD_ATTESTATION_VERSION,
            algorithm: ED25519_SIGNATURE_ALGORITHM.to_string(),
            signer_key_id: release_signer_key_id(&signing_key.verifying_key()),
            seed_hex: hex::encode(signing_key.as_bytes()),
        }
    }

    fn validate(&self) -> DynResult<()> {
        if self.kind != RELEASE_SIGNING_PRIVATE_KEY_KIND
            || self.version != RELEASE_BUILD_ATTESTATION_VERSION
            || self.algorithm != ED25519_SIGNATURE_ALGORITHM
            || self.signer_key_id.trim().is_empty()
            || self.seed_hex.trim().is_empty()
        {
            return Err("invalid release signing private key file".into());
        }
        let signing_key = signing_key_from_seed_hex(&self.seed_hex)?;
        ensure_eq(
            &release_signer_key_id(&signing_key.verifying_key()),
            self.signer_key_id.trim(),
            "release signing private key signer_key_id",
        )?;
        Ok(())
    }

    fn signing_key(&self) -> DynResult<SigningKey> {
        self.validate()?;
        signing_key_from_seed_hex(&self.seed_hex)
    }
}

impl ReleaseSigningPublicKeyFile {
    fn from_verifying_key(verifying_key: &ed25519_dalek::VerifyingKey) -> Self {
        Self {
            kind: RELEASE_SIGNING_PUBLIC_KEY_KIND.to_string(),
            version: RELEASE_BUILD_ATTESTATION_VERSION,
            algorithm: ED25519_SIGNATURE_ALGORITHM.to_string(),
            signer_key_id: release_signer_key_id(verifying_key),
            public_key_hex: hex::encode(verifying_key.as_bytes()),
        }
    }

    fn validate(&self) -> DynResult<()> {
        if self.kind != RELEASE_SIGNING_PUBLIC_KEY_KIND
            || self.version != RELEASE_BUILD_ATTESTATION_VERSION
            || self.algorithm != ED25519_SIGNATURE_ALGORITHM
            || self.signer_key_id.trim().is_empty()
            || self.public_key_hex.trim().is_empty()
        {
            return Err("invalid release signing public key file".into());
        }
        let public_key = parse_release_signer_public_key(self.signer_key_id.trim())?;
        ensure_eq(
            &hex::encode(public_key.as_bytes()),
            self.public_key_hex.trim(),
            "release signing public key hex",
        )?;
        Ok(())
    }

    fn verifying_key(&self) -> DynResult<ed25519_dalek::VerifyingKey> {
        self.validate()?;
        parse_release_signer_public_key(self.signer_key_id.trim())
    }
}

struct XtaskReleasePayloadVerifier {
    supplied_public_key: ed25519_dalek::VerifyingKey,
}

impl EmbeddedReleasePayloadVerifier for XtaskReleasePayloadVerifier {
    type Error = String;

    fn verify_payload(
        &self,
        payload_bytes: &[u8],
    ) -> Result<EmbeddedReleasePayloadSummary, Self::Error> {
        let attestation: EmbeddedReleaseAttestation =
            serde_json::from_slice(payload_bytes).map_err(|error| error.to_string())?;
        let claims = attestation
            .verify_with_public_key(&self.supplied_public_key)
            .map_err(|error| error.to_string())?;
        Ok(EmbeddedReleasePayloadSummary {
            artifact_digest: claims.artifact_digest,
        })
    }
}

#[derive(Default)]
struct GenerateKeypairArgs {
    private_key_out: Option<PathBuf>,
    public_key_out: Option<PathBuf>,
}

#[derive(Default)]
struct StampArgs {
    binary: Option<PathBuf>,
    signing_key_file: Option<PathBuf>,
    node_version: Option<String>,
    build_id: Option<String>,
    commit: Option<String>,
    target_triple: Option<String>,
    protocol_min: Option<u32>,
    protocol_max: Option<u32>,
}

#[derive(Default)]
struct InspectArgs {
    binary: Option<PathBuf>,
    public_key_file: Option<PathBuf>,
    json: bool,
}

fn release_signer_key_id(verifying_key: &ed25519_dalek::VerifyingKey) -> String {
    format!("ed25519:{}", hex::encode(verifying_key.as_bytes()))
}

fn write_json_file<T: serde::Serialize>(path: &Path, value: &T) -> DynResult<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(path, serde_json::to_vec_pretty(value)?)?;
    Ok(())
}

fn print_json<T: serde::Serialize>(value: &T) -> DynResult<()> {
    println!("{}", serde_json::to_string_pretty(value)?);
    Ok(())
}

fn main() {
    if let Err(error) = run() {
        eprintln!("error: {error}");
        std::process::exit(1);
    }
}

fn run() -> DynResult<()> {
    let args = std::env::args().skip(1).collect::<Vec<_>>();
    match args.as_slice() {
        [command, scope] if command == "repo-consistency" && scope == "release-targets" => {
            check_release_targets()
        }
        [command, scope] if command == "repo-consistency" && scope == "ci-crate-lists" => {
            let repo_root = repo_root()?;
            check_ci_script_workspace_members(&repo_root)?;
            check_attestation_default_version(&repo_root)?;
            println!("repo consistency checks passed: ci-crate-lists");
            Ok(())
        }
        [command, scope, rest @ ..]
            if command == "release-attestation" && scope == "generate-keypair" =>
        {
            generate_release_attestation_keypair(rest)
        }
        [command, scope, rest @ ..] if command == "release-attestation" && scope == "stamp" => {
            stamp_release_attestation(rest)
        }
        [command, scope, rest @ ..]
            if command == "release-attestation" && scope == "inspect" =>
        {
            inspect_release_attestation(rest)
        }
        _ => Err(
            "usage:\n  cargo run -p xtask -- repo-consistency release-targets\n  cargo run -p xtask -- repo-consistency ci-crate-lists\n  cargo run -p xtask -- release-attestation generate-keypair --private-key-out <path> --public-key-out <path>\n  cargo run -p xtask -- release-attestation stamp --binary <path> --signing-key-file <path> [--node-version <semver>] [--build-id <id>] [--commit <sha>] [--target-triple <triple>] [--protocol-min <n>] [--protocol-max <n>]\n  cargo run -p xtask -- release-attestation inspect --binary <path> [--public-key-file <path>] [--json]"
                .to_string()
                .into(),
        ),
    }
}

fn generate_release_attestation_keypair(args: &[String]) -> DynResult<()> {
    let parsed = parse_generate_keypair_args(args)?;
    let private_key_out = parsed
        .private_key_out
        .ok_or("--private-key-out is required")?;
    let public_key_out = parsed
        .public_key_out
        .ok_or("--public-key-out is required")?;
    let mut seed = [0u8; 32];
    fill_random(&mut seed).map_err(|error| error.to_string())?;
    let signing_key = SigningKey::from_bytes(&seed);
    let private_key_file = ReleaseSigningPrivateKeyFile::from_signing_key(&signing_key);
    let public_key_file =
        ReleaseSigningPublicKeyFile::from_verifying_key(&signing_key.verifying_key());
    write_json_file(&private_key_out, &private_key_file)?;
    write_json_file(&public_key_out, &public_key_file)?;
    print_json(&serde_json::json!({
        "private_key_file": private_key_out,
        "public_key_file": public_key_out,
        "signer_key_id": private_key_file.signer_key_id,
        "public_key_hex": public_key_file.public_key_hex,
    }))
}

fn stamp_release_attestation(args: &[String]) -> DynResult<()> {
    let parsed = parse_stamp_args(args)?;
    let binary = parsed.binary.ok_or("--binary is required")?;
    let signing_key_file = parsed
        .signing_key_file
        .ok_or("--signing-key-file is required")?;
    let signing_key = load_release_signing_key_file(&signing_key_file)?.signing_key()?;
    let verifying_key = signing_key.verifying_key();
    let binary_bytes = fs::read(&binary)?;
    let base_binary_bytes = strip_embedded_release_footer(&binary_bytes)?.to_vec();
    let artifact_digest = format!("sha256:{}", sha256_bytes(&base_binary_bytes));
    let node_version = match parsed.node_version {
        Some(version) => version,
        None => default_node_version()?,
    };

    let claims = ReleaseBuildAttestationClaims {
        version: RELEASE_BUILD_ATTESTATION_VERSION,
        node_version,
        build_id: parsed
            .build_id
            .unwrap_or_else(|| default_build_id(&binary, &artifact_digest)),
        commit: parsed.commit.unwrap_or_else(default_commit),
        target_triple: parsed.target_triple.unwrap_or_else(default_target_triple),
        supported_protocol_generation_min: parsed.protocol_min,
        supported_protocol_generation_max: parsed.protocol_max,
        artifact_digest,
    };
    let signer_key_id = release_signer_key_id(&verifying_key);
    let signed_payload_bytes =
        claims.canonical_bytes(&signer_key_id, ED25519_SIGNATURE_ALGORITHM)?;
    let signature = signing_key.sign(&signed_payload_bytes);
    let attestation = EmbeddedReleaseAttestation {
        version: RELEASE_BUILD_ATTESTATION_VERSION,
        signer_key_id,
        signature_algorithm: ED25519_SIGNATURE_ALGORITHM.to_string(),
        claims: claims.clone(),
        signed_payload_hex: hex::encode(&signed_payload_bytes),
        signature_hex: hex::encode(signature.to_bytes()),
    };
    attestation.validate()?;
    let payload_bytes = serde_json::to_vec(&attestation)?;
    let stamped_bytes = stamp_embedded_release_payload(&binary_bytes, &payload_bytes)?;
    fs::write(&binary, stamped_bytes)?;

    print_json(&serde_json::json!({
        "binary": binary,
        "version": claims.version,
        "node_version": claims.node_version,
        "build_id": claims.build_id,
        "commit": claims.commit,
        "target_triple": claims.target_triple,
        "supported_protocol_generation_min": claims.supported_protocol_generation_min,
        "supported_protocol_generation_max": claims.supported_protocol_generation_max,
        "artifact_digest": claims.artifact_digest,
        "signer_key_id": attestation.signer_key_id,
        "attestation_hash": attestation.canonical_hash_hex()?,
    }))
}

fn inspect_release_attestation(args: &[String]) -> DynResult<()> {
    let parsed = parse_inspect_args(args)?;
    let summary = inspect_release_attestation_summary(&parsed)?;
    let _emit_json = parsed.json;
    print_json(&summary)
}

fn inspect_release_attestation_summary(
    parsed: &InspectArgs,
) -> DynResult<ReleaseAttestationInspectSummary> {
    let binary = parsed.binary.as_ref().ok_or("--binary is required")?;
    let binary_bytes = fs::read(binary)?;

    let footer = match read_embedded_release_footer(&binary_bytes) {
        Ok(footer) => footer,
        Err(error) => {
            return Ok(ReleaseAttestationInspectSummary {
                status: EmbeddedReleaseFooterStatus::Invalid.as_str().to_string(),
                version: None,
                signer_key_id: None,
                artifact_digest: None,
                error: Some(error.to_string()),
            });
        }
    };

    let Some(footer) = footer else {
        return Ok(ReleaseAttestationInspectSummary {
            status: EmbeddedReleaseFooterStatus::Missing.as_str().to_string(),
            version: None,
            signer_key_id: None,
            artifact_digest: None,
            error: None,
        });
    };

    let attestation =
        match serde_json::from_slice::<EmbeddedReleaseAttestation>(footer.payload_bytes) {
            Ok(attestation) => attestation,
            Err(error) => {
                return Ok(ReleaseAttestationInspectSummary {
                    status: EmbeddedReleaseFooterStatus::Invalid.as_str().to_string(),
                    version: None,
                    signer_key_id: None,
                    artifact_digest: None,
                    error: Some(format!(
                        "embedded release attestation payload is invalid JSON: {error}"
                    )),
                });
            }
        };

    let claims = attestation.claims().ok();
    let version = claims
        .as_ref()
        .map(|claims| claims.version)
        .or(Some(attestation.version));
    let signer_key_id = Some(attestation.signer_key_id.clone());
    let artifact_digest = claims.as_ref().map(|claims| claims.artifact_digest.clone());

    let public_key_file = match parsed.public_key_file.as_ref() {
        Some(path) => path,
        None => {
            return Ok(ReleaseAttestationInspectSummary {
                status: EmbeddedReleaseFooterStatus::Invalid.as_str().to_string(),
                version,
                signer_key_id,
                artifact_digest,
                error: Some(
                    "--public-key-file is required when an embedded release attestation is present"
                        .to_string(),
                ),
            });
        }
    };
    let supplied_public_key =
        load_release_signing_public_key_file(public_key_file)?.verifying_key()?;
    let verification = verify_embedded_release_footer(
        &binary_bytes,
        &XtaskReleasePayloadVerifier {
            supplied_public_key,
        },
    );

    Ok(ReleaseAttestationInspectSummary {
        status: verification.status.as_str().to_string(),
        version,
        signer_key_id,
        artifact_digest,
        error: (verification.status == EmbeddedReleaseFooterStatus::Invalid)
            .then_some(verification.error)
            .flatten(),
    })
}

fn parse_generate_keypair_args(args: &[String]) -> DynResult<GenerateKeypairArgs> {
    let mut parsed = GenerateKeypairArgs::default();
    let mut iter = args.iter();
    while let Some(flag) = iter.next() {
        let value = iter
            .next()
            .ok_or_else(|| format!("missing value for {flag}"))?;
        match flag.as_str() {
            "--private-key-out" => parsed.private_key_out = Some(PathBuf::from(value)),
            "--public-key-out" => parsed.public_key_out = Some(PathBuf::from(value)),
            _ => return Err(format!("unknown flag for generate-keypair: {flag}").into()),
        }
    }
    Ok(parsed)
}

fn parse_stamp_args(args: &[String]) -> DynResult<StampArgs> {
    let mut parsed = StampArgs::default();
    let mut iter = args.iter();
    while let Some(flag) = iter.next() {
        let value = iter
            .next()
            .ok_or_else(|| format!("missing value for {flag}"))?;
        match flag.as_str() {
            "--binary" => parsed.binary = Some(PathBuf::from(value)),
            "--signing-key-file" => parsed.signing_key_file = Some(PathBuf::from(value)),
            "--node-version" => parsed.node_version = Some(value.clone()),
            "--build-id" => parsed.build_id = Some(value.clone()),
            "--commit" => parsed.commit = Some(value.clone()),
            "--target-triple" => parsed.target_triple = Some(value.clone()),
            "--protocol-min" => parsed.protocol_min = Some(value.parse()?),
            "--protocol-max" => parsed.protocol_max = Some(value.parse()?),
            _ => return Err(format!("unknown flag for stamp: {flag}").into()),
        }
    }
    Ok(parsed)
}

fn parse_inspect_args(args: &[String]) -> DynResult<InspectArgs> {
    let mut parsed = InspectArgs::default();
    let mut iter = args.iter();
    while let Some(flag) = iter.next() {
        match flag.as_str() {
            "--json" => parsed.json = true,
            "--binary" => {
                let value = iter
                    .next()
                    .ok_or_else(|| format!("missing value for {flag}"))?;
                parsed.binary = Some(PathBuf::from(value));
            }
            "--public-key-file" => {
                let value = iter
                    .next()
                    .ok_or_else(|| format!("missing value for {flag}"))?;
                parsed.public_key_file = Some(PathBuf::from(value));
            }
            _ => return Err(format!("unknown flag for inspect: {flag}").into()),
        }
    }
    Ok(parsed)
}

fn load_release_signing_key_file(path: &Path) -> DynResult<ReleaseSigningPrivateKeyFile> {
    let key_file: ReleaseSigningPrivateKeyFile = serde_json::from_slice(&fs::read(path)?)?;
    key_file.validate()?;
    Ok(key_file)
}

fn load_release_signing_public_key_file(path: &Path) -> DynResult<ReleaseSigningPublicKeyFile> {
    let key_file: ReleaseSigningPublicKeyFile = serde_json::from_slice(&fs::read(path)?)?;
    key_file.validate()?;
    Ok(key_file)
}

fn signing_key_from_seed_hex(seed_hex: &str) -> DynResult<SigningKey> {
    let seed = hex::decode(seed_hex)?;
    let seed: [u8; 32] = seed
        .try_into()
        .map_err(|_| "release signing seed must decode to exactly 32 bytes")?;
    Ok(SigningKey::from_bytes(&seed))
}

fn sha256_bytes(bytes: &[u8]) -> String {
    use sha2::{Digest, Sha256};

    hex::encode(Sha256::digest(bytes))
}

fn parse_release_signer_public_key(signer_key_id: &str) -> DynResult<ed25519_dalek::VerifyingKey> {
    let encoded = signer_key_id
        .strip_prefix("ed25519:")
        .ok_or("release signer key id must start with ed25519:")?;
    let bytes = hex::decode(encoded)?;
    let bytes: [u8; 32] = bytes
        .try_into()
        .map_err(|_| "release signer key id must contain a 32-byte public key")?;
    Ok(ed25519_dalek::VerifyingKey::from_bytes(&bytes)?)
}

fn default_build_id(binary: &Path, artifact_digest: &str) -> String {
    let stem = binary
        .file_stem()
        .and_then(|value| value.to_str())
        .unwrap_or("mesh-llm");
    format!("{stem}-{}", &artifact_digest[..12])
}

fn default_commit() -> String {
    std::env::var("GIT_COMMIT").unwrap_or_else(|_| "task8-local".to_string())
}

fn default_target_triple() -> String {
    std::env::var("TARGET")
        .unwrap_or_else(|_| format!("{}-{}", std::env::consts::ARCH, std::env::consts::OS))
}

fn check_release_targets() -> DynResult<()> {
    let repo_root = repo_root()?;
    let fixture_rows = fixture_rows(&repo_root)?;
    let fixture_version = fixture_release_tag(&fixture_rows)?;

    if host_supports_shell_parity_checks() {
        check_installer_outcomes(&repo_root, &fixture_rows)?;
        check_package_release_assets(&repo_root, &fixture_rows, &fixture_version)?;
    } else {
        println!(
            "note: skipping bash-dependent release parity checks on native Windows; run `just check-release` on macOS/Linux for install.sh and package-release.sh parity"
        );
    }
    check_windows_name_invariance(&fixture_rows, &fixture_version)?;
    check_ci_script_workspace_members(&repo_root)?;
    check_attestation_default_version(&repo_root)?;
    check_docs_and_workflow_invariants(&repo_root)?;

    println!("repo consistency checks passed: release-targets");
    Ok(())
}

fn check_attestation_default_version(repo_root: &Path) -> DynResult<()> {
    let runtime_version = resolve_runtime_version(repo_root)?;
    let default_node_version = host_runtime_package_version(repo_root)?;
    ensure_eq(
        runtime_version.as_str(),
        default_node_version.as_str(),
        "xtask release-attestation default node version",
    )
}

fn default_node_version() -> DynResult<String> {
    host_runtime_package_version(&repo_root()?)
}

fn resolve_runtime_version(repo_root: &Path) -> DynResult<String> {
    let runtime_lib = repo_root
        .join("crates")
        .join("mesh-llm-host-runtime")
        .join("src")
        .join("lib.rs");
    let contents = fs::read_to_string(runtime_lib)?;
    extract_runtime_version(repo_root, &contents)
}

fn extract_runtime_version(repo_root: &Path, contents: &str) -> DynResult<String> {
    const LITERAL_PREFIX: &str = "pub const VERSION: &str = \"";
    const CARGO_PKG_VERSION: &str = "pub const VERSION: &str = env!(\"CARGO_PKG_VERSION\");";
    for line in contents.lines().map(str::trim) {
        if line == CARGO_PKG_VERSION {
            return host_runtime_package_version(repo_root);
        }
        if let Some(rest) = line.strip_prefix(LITERAL_PREFIX) {
            return Ok(rest
                .strip_suffix("\";")
                .ok_or("malformed mesh-llm-host-runtime VERSION constant")?
                .to_string());
        }
    }
    Err("missing mesh-llm-host-runtime VERSION constant".into())
}

fn host_runtime_package_version(repo_root: &Path) -> DynResult<String> {
    let runtime_manifest = repo_root
        .join("crates")
        .join("mesh-llm-host-runtime")
        .join("Cargo.toml");
    let runtime_contents = fs::read_to_string(runtime_manifest)?;
    if let Some(version) = extract_manifest_string(&runtime_contents, "version")? {
        return Ok(version);
    }
    if has_manifest_bool(&runtime_contents, "version.workspace", true) {
        let workspace_manifest = repo_root.join("Cargo.toml");
        let workspace_contents = fs::read_to_string(workspace_manifest)?;
        return extract_manifest_string(&workspace_contents, "version")?
            .ok_or_else(|| "missing workspace package version".into());
    }
    Err("missing mesh-llm-host-runtime package version".into())
}

fn extract_manifest_string(contents: &str, key: &str) -> DynResult<Option<String>> {
    let prefix = format!("{key} = \"");
    for line in contents.lines().map(str::trim) {
        if let Some(rest) = line.strip_prefix(&prefix) {
            return Ok(Some(
                rest.strip_suffix('"')
                    .ok_or_else(|| format!("malformed manifest {key} value"))?
                    .to_string(),
            ));
        }
    }
    Ok(None)
}

fn has_manifest_bool(contents: &str, key: &str, expected: bool) -> bool {
    let expected_value = if expected { "true" } else { "false" };
    let expected_line = format!("{key} = {expected_value}");
    contents
        .lines()
        .map(str::trim)
        .any(|line| line == expected_line)
}

fn host_supports_shell_parity_checks() -> bool {
    !cfg!(windows)
}

fn repo_root() -> DynResult<PathBuf> {
    // CARGO_MANIFEST_DIR is <repo>/tools/xtask; go up two levels to reach the repo root.
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .map(Path::to_path_buf)
        .ok_or_else(|| "could not determine repo root from xtask manifest directory".into())
}

#[derive(Clone, Debug, Deserialize)]
struct FixtureRow {
    os: String,
    arch: String,
    flavor: String,
    support: String,
    stable_asset: Option<String>,
    versioned_asset: Option<String>,
}

#[derive(Debug, Deserialize)]
struct CargoMetadata {
    packages: Vec<CargoPackage>,
    workspace_members: Vec<String>,
}

#[derive(Debug, Deserialize)]
struct CargoPackage {
    id: String,
    name: String,
}

fn fixture_rows(repo_root: &Path) -> DynResult<Vec<FixtureRow>> {
    let fixture_path = fixture_path(repo_root);
    let contents = fs::read_to_string(&fixture_path)?;
    Ok(serde_json::from_str(&contents)?)
}

fn fixture_path(repo_root: &Path) -> PathBuf {
    repo_root
        .join("crates")
        .join("mesh-llm-system")
        .join("tests")
        .join("fixtures")
        .join("release-target-matrix.json")
}

fn fixture_release_tag(rows: &[FixtureRow]) -> DynResult<String> {
    for row in rows {
        let (Some(stable), Some(versioned)) = (&row.stable_asset, &row.versioned_asset) else {
            continue;
        };

        let stable_tail = stable
            .strip_prefix("mesh-llm-")
            .ok_or("stable asset missing mesh-llm- prefix")?;
        let versioned_tail = versioned
            .strip_prefix("mesh-llm-")
            .ok_or("versioned asset missing mesh-llm- prefix")?;
        let suffix = format!("-{stable_tail}");
        if let Some(version) = versioned_tail.strip_suffix(&suffix) {
            return Ok(version.to_string());
        }
    }

    Err("could not derive fixture release tag".into())
}

fn fixture_row<'a>(
    rows: &'a [FixtureRow],
    os: &str,
    arch: &str,
    flavor: &str,
) -> DynResult<&'a FixtureRow> {
    rows.iter()
        .find(|row| row.os == os && row.arch == arch && row.flavor == flavor)
        .ok_or_else(|| format!("missing fixture row for {os}/{arch}/{flavor}").into())
}

fn check_installer_outcomes(repo_root: &Path, rows: &[FixtureRow]) -> DynResult<()> {
    let linux_arm64_asset = fixture_row(rows, "linux", "aarch64", "cpu")?
        .stable_asset
        .clone()
        .ok_or("linux/aarch64/cpu stable asset missing")?;
    let macos_arm64_asset = fixture_row(rows, "macos", "aarch64", "metal")?
        .stable_asset
        .clone()
        .ok_or("macos/aarch64/metal stable asset missing")?;

    let cases = [
        InstallerCase {
            raw_os: "Linux",
            raw_arch: "arm64",
            flavor: "cpu",
            expected_platform: "Linux/aarch64",
            expected_supported_flavors: "cpu",
            expected_asset: linux_arm64_asset.as_str(),
            label: "Linux/arm64",
        },
        InstallerCase {
            raw_os: "Linux",
            raw_arch: "aarch64",
            flavor: "cpu",
            expected_platform: "Linux/aarch64",
            expected_supported_flavors: "cpu",
            expected_asset: linux_arm64_asset.as_str(),
            label: "Linux/aarch64",
        },
        InstallerCase {
            raw_os: "Darwin",
            raw_arch: "arm64",
            flavor: "metal",
            expected_platform: "Darwin/arm64",
            expected_supported_flavors: "metal",
            expected_asset: macos_arm64_asset.as_str(),
            label: "Darwin/arm64",
        },
    ];

    for case in cases {
        let envs = [
            ("MESH_LLM_TEST_UNAME_S", case.raw_os),
            ("MESH_LLM_TEST_UNAME_M", case.raw_arch),
        ];
        let actual_platform =
            sourced_script_stdout(repo_root, "install.sh", "platform_id", &envs, &[])?;
        ensure_eq(
            case.expected_platform,
            &actual_platform,
            &format!("{} normalized platform", case.label),
        )?;

        let actual_supported_flavors =
            sourced_script_stdout(repo_root, "install.sh", "supported_flavors", &envs, &[])?;
        ensure_eq(
            case.expected_supported_flavors,
            &actual_supported_flavors,
            &format!("{} supported flavors", case.label),
        )?;

        let actual_asset = sourced_script_stdout(
            repo_root,
            "install.sh",
            "asset_name \"$2\"",
            &envs,
            &[case.flavor],
        )?;
        ensure_eq(
            case.expected_asset,
            &actual_asset,
            &format!("{} asset parity", case.label),
        )?;
    }

    let arm_fixture = fixture_row(rows, "linux", "arm", "cpu")?;
    let arm_envs = [
        ("MESH_LLM_TEST_UNAME_S", "Linux"),
        ("MESH_LLM_TEST_UNAME_M", "armv7l"),
    ];
    let actual_support = sourced_script_stdout(
        repo_root,
        "install.sh",
        "platform_support_status",
        &arm_envs,
        &[],
    )?;
    ensure_eq(
        &arm_fixture.support,
        &actual_support,
        "Linux/armv7l installer support classification",
    )?;
    let actual_message = sourced_script_stdout(
        repo_root,
        "install.sh",
        "platform_error_message",
        &arm_envs,
        &[],
    )?;
    ensure_eq(
        "error: recognized but unsupported platform: Linux/arm (32-bit ARM release bundles are not published)",
        &actual_message,
        "Linux/armv7l installer error",
    )?;

    Ok(())
}

struct InstallerCase<'a> {
    raw_os: &'a str,
    raw_arch: &'a str,
    flavor: &'a str,
    expected_platform: &'a str,
    expected_supported_flavors: &'a str,
    expected_asset: &'a str,
    label: &'a str,
}

fn check_package_release_assets(
    repo_root: &Path,
    rows: &[FixtureRow],
    fixture_version: &str,
) -> DynResult<()> {
    for row in rows {
        if row.os != "linux" && row.os != "macos" {
            continue;
        }
        if row.support == "recognized-unsupported" {
            continue;
        }

        for raw_case in raw_targets(row)? {
            let mut envs = vec![
                ("MESH_RELEASE_OS", raw_case.raw_os),
                ("MESH_RELEASE_ARCH", raw_case.raw_arch),
            ];
            if row.flavor != implicit_release_flavor(row) {
                envs.push(("MESH_RELEASE_FLAVOR", row.flavor.as_str()));
            }

            let actual_support = sourced_script_stdout(
                repo_root,
                "scripts/package-release.sh",
                "release_target_support",
                &envs,
                &[],
            )?;
            ensure_eq(
                shell_support(row),
                &actual_support,
                &format!(
                    "{}/{}/{} package support ({})",
                    row.os, row.arch, row.flavor, raw_case.label
                ),
            )?;

            if row.support != "supported" {
                let tmp_output_dir = unique_temp_dir("check-release-unsupported");
                let output = run_command(
                    Command::new("bash")
                        .current_dir(repo_root)
                        .envs(envs.iter().copied())
                        .arg("scripts/package-release.sh")
                        .arg(fixture_version)
                        .arg(&tmp_output_dir),
                );
                let _ = std::fs::remove_dir_all(&tmp_output_dir);
                let output = output?;
                ensure_status(
                    1,
                    output.status.code(),
                    &format!(
                        "{}/{}/{} unsupported packaging exit code ({})",
                        row.os, row.arch, row.flavor, raw_case.label
                    ),
                )?;
                ensure_eq(
                    &unsupported_release_target_message(&raw_case, row),
                    &trimmed_stderr_or_stdout(&output),
                    &format!(
                        "{}/{}/{} unsupported packaging message ({})",
                        row.os, row.arch, row.flavor, raw_case.label
                    ),
                )?;
                continue;
            }

            let actual_stable = sourced_script_stdout(
                repo_root,
                "scripts/package-release.sh",
                "resolve_release_target; printf '%s\\n' \"$STABLE_ASSET\"",
                &envs,
                &[],
            )?;
            ensure_eq_option(
                row.stable_asset.as_deref(),
                Some(actual_stable.as_str()),
                &format!(
                    "{}/{}/{} package stable asset ({})",
                    row.os, row.arch, row.flavor, raw_case.label
                ),
            )?;

            let actual_versioned = sourced_script_stdout(
                repo_root,
                "scripts/package-release.sh",
                "versioned_asset_name \"$2\"",
                &envs,
                &[fixture_version],
            )?;
            ensure_eq_option(
                row.versioned_asset.as_deref(),
                Some(actual_versioned.as_str()),
                &format!(
                    "{}/{}/{} package versioned asset ({})",
                    row.os, row.arch, row.flavor, raw_case.label
                ),
            )?;
        }
    }

    let arm_row = fixture_row(rows, "linux", "arm", "cpu")?;
    ensure_eq(
        "recognized-unsupported",
        &arm_row.support,
        "linux/arm fixture support",
    )?;
    ensure_eq_option(
        None,
        arm_row.stable_asset.as_deref(),
        "linux/arm fixture stable asset",
    )?;
    ensure_eq_option(
        None,
        arm_row.versioned_asset.as_deref(),
        "linux/arm fixture versioned asset",
    )?;

    let tmp_output_dir = unique_temp_dir("check-release");
    let output = run_command(
        Command::new("bash")
            .current_dir(repo_root)
            .env("MESH_RELEASE_OS", "Linux")
            .env("MESH_RELEASE_ARCH", "armv7l")
            .arg("scripts/package-release.sh")
            .arg(fixture_version)
            .arg(&tmp_output_dir),
    );
    // Clean up before propagating any error so the temp dir is always removed.
    let _ = std::fs::remove_dir_all(&tmp_output_dir);
    let output = output?;
    ensure_status(1, output.status.code(), "Linux/armv7l packaging exit code")?;
    let actual_message = trimmed_stderr_or_stdout(&output);
    ensure_eq(
        "Recognized but unsupported release target: Linux/armv7l (normalized: linux/arm)",
        &actual_message,
        "Linux/armv7l packaging error",
    )?;

    Ok(())
}

struct RawTargetCase {
    raw_os: &'static str,
    raw_arch: &'static str,
    label: &'static str,
}

fn raw_targets(row: &FixtureRow) -> DynResult<Vec<RawTargetCase>> {
    match (row.os.as_str(), row.arch.as_str()) {
        ("macos", "aarch64") => Ok(vec![RawTargetCase {
            raw_os: "Darwin",
            raw_arch: "arm64",
            label: "Darwin/arm64",
        }]),
        ("linux", "x86_64") => Ok(vec![RawTargetCase {
            raw_os: "Linux",
            raw_arch: "x86_64",
            label: "Linux/x86_64",
        }]),
        ("linux", "aarch64") => Ok(vec![
            RawTargetCase {
                raw_os: "Linux",
                raw_arch: "arm64",
                label: "Linux/arm64",
            },
            RawTargetCase {
                raw_os: "Linux",
                raw_arch: "aarch64",
                label: "Linux/aarch64",
            },
        ]),
        _ => Err(format!("unsupported raw target mapping for {}/{}", row.os, row.arch).into()),
    }
}

fn implicit_release_flavor(row: &FixtureRow) -> &'static str {
    match (row.os.as_str(), row.arch.as_str()) {
        ("macos", "aarch64") => "metal",
        ("linux", "x86_64") | ("linux", "aarch64") | ("linux", "arm") => "cpu",
        _ => "",
    }
}

fn shell_support(row: &FixtureRow) -> &str {
    match row.support.as_str() {
        "unknown" => "unsupported",
        other => other,
    }
}

fn unsupported_release_target_message(raw_case: &RawTargetCase, row: &FixtureRow) -> String {
    format!(
        "Unsupported release target/flavor for packaging: {}/{} with flavor {} (normalized: {}/{})",
        raw_case.raw_os, raw_case.raw_arch, row.flavor, row.os, row.arch
    )
}

fn check_windows_name_invariance(rows: &[FixtureRow], fixture_version: &str) -> DynResult<()> {
    for row in rows {
        if row.os != "windows" {
            continue;
        }

        ensure_eq(
            "x86_64",
            &row.arch,
            &format!("windows/{}/{}/canonical arch", row.arch, row.flavor),
        )?;
        ensure_eq(
            "supported",
            &row.support,
            &format!("windows/{}/{}/support", row.arch, row.flavor),
        )?;
        let stable_expected = windows_asset_name(&row.flavor, "");
        let versioned_expected = windows_asset_name(&row.flavor, &format!("-{fixture_version}"));
        ensure_eq_option(
            Some(stable_expected.as_str()),
            row.stable_asset.as_deref(),
            &format!("windows/{}/{}/stable asset", row.arch, row.flavor),
        )?;
        ensure_eq_option(
            Some(versioned_expected.as_str()),
            row.versioned_asset.as_deref(),
            &format!("windows/{}/{}/versioned asset", row.arch, row.flavor),
        )?;
    }

    Ok(())
}

fn windows_asset_name(flavor: &str, version_prefix: &str) -> String {
    let suffix = match flavor {
        "cpu" | "metal" => "",
        other => other,
    };

    if suffix.is_empty() {
        format!("mesh-llm{version_prefix}-x86_64-pc-windows-msvc.zip")
    } else {
        format!("mesh-llm{version_prefix}-x86_64-pc-windows-msvc-{suffix}.zip")
    }
}

fn check_docs_and_workflow_invariants(repo_root: &Path) -> DynResult<()> {
    let readme = fs::read_to_string(repo_root.join("README.md"))?;
    let contributing = fs::read_to_string(repo_root.join("CONTRIBUTING.md"))?;
    let release = fs::read_to_string(repo_root.join("RELEASE.md"))?;
    let justfile = fs::read_to_string(repo_root.join("Justfile"))?;
    let release_workflow = fs::read_to_string(repo_root.join(".github/workflows/release.yml"))?;
    let ci_workflow = fs::read_to_string(repo_root.join(".github/workflows/ci.yml"))?;
    let pr_builds_workflow = fs::read_to_string(repo_root.join(".github/workflows/pr_builds.yml"))?;
    let pr_quality_workflow =
        fs::read_to_string(repo_root.join(".github/workflows/pr_quality.yml"))?;
    let pr_cleanup_workflow =
        fs::read_to_string(repo_root.join(".github/workflows/pr_cleanup.yml"))?;
    let windows_warm_caches_workflow =
        fs::read_to_string(repo_root.join(".github/workflows/windows-warm-caches.yml"))?;

    ensure_contains(
        &readme,
        "mesh-llm-aarch64-unknown-linux-gnu.tar.gz",
        "README Linux ARM64 asset note",
    )?;
    ensure_contains(
        &release,
        "mesh-llm-aarch64-unknown-linux-gnu.tar.gz",
        "RELEASE Linux ARM64 asset note",
    )?;
    ensure_contains_normalized(
        &readme,
        "Windows CPU, Windows CUDA, Windows ROCm, and Windows Vulkan bundles",
        "README Windows publish note",
    )?;
    ensure_contains(
        &release,
        "Windows release artifacts use the `x86_64-pc-windows-msvc` target triple",
        "RELEASE Windows publish note",
    )?;
    ensure_contains(
        &release_workflow,
        "runs-on: blacksmith-4vcpu-ubuntu-2404-arm",
        "release workflow ARM64 runner",
    )?;
    ensure_contains(
        &release_workflow,
        "name: release-linux-arm64",
        "release workflow ARM64 artifact",
    )?;
    ensure_contains(
        &release_workflow,
        "build_windows_cpu:",
        "release workflow Windows CPU build",
    )?;
    ensure_contains(
        &release_workflow,
        "build_windows_gpu:",
        "release workflow Windows GPU build",
    )?;
    ensure_contains(
        &release_workflow,
        "- build_windows_cpu",
        "release workflow Windows CPU publish need",
    )?;
    ensure_contains(
        &release_workflow,
        "- build_windows_gpu",
        "release workflow Windows GPU publish need",
    )?;
    ensure_contains(
        &justfile,
        "check-release:",
        "Justfile release consistency wrapper",
    )?;
    ensure_contains(
        &justfile,
        "cargo run -p xtask -- repo-consistency release-targets",
        "Justfile xtask command",
    )?;
    ensure_contains(
        &contributing,
        "just check-release",
        "CONTRIBUTING release consistency command",
    )?;
    ensure_contains(
        &contributing,
        "On native Windows, `just check-release` runs the host-safe Rust/doc invariant subset and skips the Bash-only `install.sh` / `package-release.sh` parity checks",
        "CONTRIBUTING Windows check-release note",
    )?;
    ensure_contains(
        &release,
        "On native Windows, `just check-release` still runs the Rust/docs/workflow invariant checks, but it skips the Bash-only `install.sh` and `scripts/package-release.sh` parity checks",
        "RELEASE Windows check-release note",
    )?;
    ensure_contains(
        &pr_builds_workflow,
        "cargo run -p xtask -- repo-consistency release-targets",
        "PR Builds xtask release-target check",
    )?;
    ensure_contains(
        &pr_quality_workflow,
        "name: PR Quality Checks",
        "PR quality workflow display name",
    )?;
    ensure_contains(
        &pr_quality_workflow,
        "cargo run -p xtask -- repo-consistency ci-crate-lists",
        "PR quality CI crate-list drift check",
    )?;
    ensure_contains(
        &pr_cleanup_workflow,
        "pull_request_target:",
        "PR cache cleanup trigger",
    )?;
    ensure_contains(
        &ci_workflow,
        "push:\n    branches: [main]",
        "main CI push trigger",
    )?;
    check_windows_abi_cache_key_alignment(
        &ci_workflow,
        &pr_builds_workflow,
        &windows_warm_caches_workflow,
    )?;
    check_ci_crate_test_coverage(&pr_builds_workflow)?;

    Ok(())
}

fn check_windows_abi_cache_key_alignment(
    ci_workflow: &str,
    pr_builds_workflow: &str,
    windows_warm_caches_workflow: &str,
) -> DynResult<()> {
    const WINDOWS_ABI_CACHE_HASH_INPUTS: &str = concat!(
        "hashFiles('scripts/build-windows.ps1', 'scripts/install-windows-sdk.ps1', ",
        "'.github/actions/setup-windows-rocm-sdk/action.yml', ",
        "'third_party/llama.cpp/upstream.txt', 'third_party/llama.cpp/patches/**', ",
        "'Justfile', '.github/cache-version.txt')",
    );
    let windows_cpu_abi_cache_key =
        format!("windows-2025-skippy-abi-cpu--cpu-${{{{ {WINDOWS_ABI_CACHE_HASH_INPUTS} }}}}");

    ensure_contains(
        ci_workflow,
        &windows_cpu_abi_cache_key,
        "main CI Windows CPU ABI cache key",
    )?;
    ensure_contains(
        windows_warm_caches_workflow,
        &windows_cpu_abi_cache_key,
        "Windows warm-cache CPU ABI cache key",
    )?;
    ensure_contains(
        pr_builds_workflow,
        "windows-2025-skippy-abi-${{ matrix.backend }}-${{ matrix.build_args }}-",
        "PR Builds Windows ABI cache key template",
    )?;
    ensure_contains(
        pr_builds_workflow,
        "|| 'cpu' }}-${{ hashFiles(",
        "PR Builds Windows CPU ABI cache discriminator",
    )?;
    ensure_contains(
        pr_builds_workflow,
        WINDOWS_ABI_CACHE_HASH_INPUTS,
        "PR Builds Windows ABI cache hash inputs",
    )?;

    Ok(())
}

fn check_ci_crate_test_coverage(ci_workflow: &str) -> DynResult<()> {
    const REQUIRED_TEST_CRATES: &[(&str, &str)] = &[
        ("mesh-llm-client", "mesh client crate tests"),
        ("mesh-llm-api-client", "mesh LLM client API crate tests"),
        ("mesh-llm-api-server", "mesh LLM API crate tests"),
        ("mesh-llm-config", "mesh LLM config crate tests"),
        (
            "mesh-llm-console-server",
            "mesh LLM console server crate tests",
        ),
        ("mesh-llm-ffi", "mesh LLM FFI crate tests"),
        ("mesh-llm-nodejs", "mesh LLM Node.js crate tests"),
        ("skippy-protocol", "skippy protocol crate tests"),
        ("skippy-server", "skippy server crate tests"),
        ("openai-frontend", "OpenAI frontend crate tests"),
        ("skippy-runtime", "skippy runtime crate tests"),
        ("skippy-topology", "skippy topology crate tests"),
        ("skippy-model-package", "skippy model-package crate tests"),
        ("skippy-prompt", "skippy prompt crate tests"),
        ("metrics-server", "metrics server crate tests"),
    ];
    const LIB_ONLY_CRATE_PATTERN: &str = "skippy-protocol|skippy-server|openai-frontend)";

    ensure_contains(
        ci_workflow,
        "cargo test -p \"$c\"",
        "CI dynamic crate test command",
    )?;
    ensure_contains(
        ci_workflow,
        "for c in mesh-llm-client mesh-llm-api-client mesh-llm-api-server mesh-llm-config mesh-llm-console-server mesh-llm-ffi mesh-llm-nodejs; do",
        "CI SDK/API crate test loop",
    )?;
    ensure_contains(
        ci_workflow,
        "for c in skippy-protocol skippy-server openai-frontend skippy-runtime skippy-topology skippy-model-package skippy-prompt metrics-server; do",
        "CI Skippy crate test loop",
    )?;
    ensure_contains(
        ci_workflow,
        LIB_ONLY_CRATE_PATTERN,
        "CI lib-only crate test flag selector",
    )?;
    ensure_contains(ci_workflow, "--lib", "CI lib-only crate test flag")?;

    for (crate_name, context) in REQUIRED_TEST_CRATES {
        ensure_contains(ci_workflow, crate_name, &format!("CI {context}"))?;
    }

    Ok(())
}

fn check_ci_script_workspace_members(repo_root: &Path) -> DynResult<()> {
    let expected = workspace_package_names(repo_root)?;
    let scripts = [
        "scripts/affected-crates.sh",
        "scripts/plan-clippy-batches.sh",
    ];

    for script in scripts {
        let actual = script_workspace_members(repo_root, script)?;
        ensure_set_eq(&expected, &actual, &format!("{script} WORKSPACE_MEMBERS"))?;
    }

    Ok(())
}

fn workspace_package_names(repo_root: &Path) -> DynResult<BTreeSet<String>> {
    let mut cargo = Command::new("cargo");
    cargo
        .current_dir(repo_root)
        .arg("metadata")
        .arg("--format-version=1")
        .arg("--no-deps");
    let output = run_command(&mut cargo)?;
    if !output.status.success() {
        return Err(format!(
            "cargo metadata failed while checking CI crate lists: {}",
            trimmed_stderr_or_stdout(&output)
        )
        .into());
    }

    let metadata: CargoMetadata = serde_json::from_slice(&output.stdout)?;
    let workspace_members = metadata
        .workspace_members
        .into_iter()
        .collect::<BTreeSet<_>>();
    let mut names = BTreeSet::new();
    for package in metadata.packages {
        if workspace_members.contains(&package.id) {
            names.insert(package.name);
        }
    }

    if names.is_empty() {
        return Err("cargo metadata returned no workspace package names".into());
    }

    Ok(names)
}

fn script_workspace_members(repo_root: &Path, relative_path: &str) -> DynResult<BTreeSet<String>> {
    let contents = fs::read_to_string(repo_root.join(relative_path))?;
    let mut in_array = false;
    let mut members = BTreeSet::new();

    for line in contents.lines() {
        let trimmed = line.trim();
        if !in_array {
            if trimmed == "WORKSPACE_MEMBERS=(" {
                in_array = true;
            }
            continue;
        }

        if trimmed == ")" {
            return Ok(members);
        }

        let Some(member) = trimmed
            .strip_prefix('"')
            .and_then(|value| value.strip_suffix('"'))
        else {
            return Err(format!(
                "{relative_path} WORKSPACE_MEMBERS: expected quoted crate name, got `{trimmed}`"
            )
            .into());
        };
        if !members.insert(member.to_string()) {
            return Err(format!(
                "{relative_path} WORKSPACE_MEMBERS: duplicate crate name `{member}`"
            )
            .into());
        }
    }

    Err(format!("{relative_path}: missing WORKSPACE_MEMBERS array").into())
}

fn sourced_script_stdout(
    repo_root: &Path,
    script_relative_path: &str,
    expression: &str,
    envs: &[(&str, &str)],
    extra_args: &[&str],
) -> DynResult<String> {
    let script_path = repo_root.join(script_relative_path);
    let command = format!("source \"$1\"; {expression}");
    let mut bash = Command::new("bash");
    bash.current_dir(repo_root)
        .arg("-lc")
        .arg(command)
        .arg("bash")
        .arg(script_path);
    for extra_arg in extra_args {
        bash.arg(extra_arg);
    }
    for (key, value) in envs {
        bash.env(key, value);
    }

    let output = run_command(&mut bash)?;
    if !output.status.success() {
        return Err(format!(
            "script command failed: {}",
            trimmed_stderr_or_stdout(&output)
        )
        .into());
    }
    Ok(trimmed_stdout(&output))
}

fn run_command(command: &mut Command) -> DynResult<Output> {
    Ok(command.output()?)
}

fn trimmed_stdout(output: &Output) -> String {
    String::from_utf8_lossy(&output.stdout).trim().to_string()
}

fn trimmed_stderr_or_stdout(output: &Output) -> String {
    let stderr = String::from_utf8_lossy(&output.stderr).trim().to_string();
    if !stderr.is_empty() {
        stderr
    } else {
        trimmed_stdout(output)
    }
}

fn unique_temp_dir(prefix: &str) -> PathBuf {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_nanos())
        .unwrap_or_default();
    std::env::temp_dir().join(format!(".tmp-{prefix}-{}-{nanos}", std::process::id()))
}

fn ensure_eq(expected: &str, actual: &str, context: &str) -> DynResult<()> {
    if expected == actual {
        Ok(())
    } else {
        Err(format!("{context}: expected `{expected}`, got `{actual}`").into())
    }
}

fn ensure_eq_option(expected: Option<&str>, actual: Option<&str>, context: &str) -> DynResult<()> {
    if expected == actual {
        Ok(())
    } else {
        Err(format!("{context}: expected {:?}, got {:?}", expected, actual).into())
    }
}

fn ensure_set_eq(
    expected: &BTreeSet<String>,
    actual: &BTreeSet<String>,
    context: &str,
) -> DynResult<()> {
    if expected == actual {
        return Ok(());
    }

    let missing = expected
        .difference(actual)
        .cloned()
        .collect::<Vec<_>>()
        .join(", ");
    let extra = actual
        .difference(expected)
        .cloned()
        .collect::<Vec<_>>()
        .join(", ");
    Err(format!(
        "{context}: workspace crate list drift detected; missing [{}], extra [{}]",
        missing, extra
    )
    .into())
}

fn ensure_status(expected: i32, actual: Option<i32>, context: &str) -> DynResult<()> {
    match actual {
        Some(status) if status == expected => Ok(()),
        Some(status) => {
            Err(format!("{context}: expected exit code {expected}, got {status}").into())
        }
        None => Err(format!("{context}: process terminated by signal").into()),
    }
}

fn ensure_contains(haystack: &str, needle: &str, context: &str) -> DynResult<()> {
    if haystack.contains(needle) {
        Ok(())
    } else {
        Err(format!("{context}: missing `{needle}`").into())
    }
}

fn ensure_contains_normalized(haystack: &str, needle: &str, context: &str) -> DynResult<()> {
    let normalized_haystack = normalize_whitespace(haystack);
    let normalized_needle = normalize_whitespace(needle);
    if normalized_haystack.contains(&normalized_needle) {
        Ok(())
    } else {
        Err(format!("{context}: missing `{needle}`").into())
    }
}

fn normalize_whitespace(value: &str) -> String {
    value.split_whitespace().collect::<Vec<_>>().join(" ")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_signing_key(seed: u8) -> SigningKey {
        SigningKey::from_bytes(&[seed; 32])
    }

    fn write_test_keypair(dir: &Path, seed: u8) -> DynResult<(PathBuf, PathBuf)> {
        let signing_key = test_signing_key(seed);
        let private_key_path = dir.join("release-key");
        let public_key_path = dir.join("release-key.pub");
        write_json_file(
            &private_key_path,
            &ReleaseSigningPrivateKeyFile::from_signing_key(&signing_key),
        )?;
        write_json_file(
            &public_key_path,
            &ReleaseSigningPublicKeyFile::from_verifying_key(&signing_key.verifying_key()),
        )?;
        Ok((private_key_path, public_key_path))
    }

    fn make_temp_dir(label: &str) -> DynResult<PathBuf> {
        let dir = unique_temp_dir(label);
        fs::create_dir_all(&dir)?;
        Ok(dir)
    }

    #[test]
    fn release_signing_key_files_round_trip() -> DynResult<()> {
        let signing_key = test_signing_key(7);
        let private_key_file = ReleaseSigningPrivateKeyFile::from_signing_key(&signing_key);
        let public_key_file =
            ReleaseSigningPublicKeyFile::from_verifying_key(&signing_key.verifying_key());

        let loaded_signing_key = private_key_file.signing_key()?;
        let loaded_public_key = public_key_file.verifying_key()?;
        assert_eq!(loaded_signing_key.as_bytes(), signing_key.as_bytes());
        assert_eq!(
            loaded_public_key.as_bytes(),
            signing_key.verifying_key().as_bytes()
        );
        assert_eq!(
            private_key_file.signer_key_id,
            public_key_file.signer_key_id
        );
        Ok(())
    }

    #[test]
    fn release_attestation_inspect_reports_missing_without_public_key() -> DynResult<()> {
        let dir = make_temp_dir("xtask-attestation-missing")?;
        let binary_path = dir.join("mesh-llm");
        fs::write(&binary_path, b"plain release binary")?;

        let summary = inspect_release_attestation_summary(&InspectArgs {
            binary: Some(binary_path),
            public_key_file: None,
            json: true,
        })?;
        assert_eq!(summary.status, "missing");
        assert_eq!(summary.version, None);
        assert_eq!(summary.signer_key_id, None);
        assert_eq!(summary.artifact_digest, None);
        assert_eq!(summary.error, None);
        Ok(())
    }

    #[test]
    fn release_attestation_stamp_and_inspect_round_trip() -> DynResult<()> {
        let dir = make_temp_dir("xtask-attestation-valid")?;
        let binary_path = dir.join("mesh-llm");
        fs::write(&binary_path, b"release-binary-v1")?;
        let (private_key_path, public_key_path) = write_test_keypair(&dir, 11)?;

        stamp_release_attestation(&[
            "--binary".to_string(),
            binary_path.display().to_string(),
            "--signing-key-file".to_string(),
            private_key_path.display().to_string(),
            "--node-version".to_string(),
            "9.9.9".to_string(),
            "--build-id".to_string(),
            "build-123".to_string(),
            "--commit".to_string(),
            "abcdef".to_string(),
            "--target-triple".to_string(),
            "x86_64-unknown-linux-gnu".to_string(),
        ])?;

        let summary = inspect_release_attestation_summary(&InspectArgs {
            binary: Some(binary_path.clone()),
            public_key_file: Some(public_key_path),
            json: true,
        })?;
        assert_eq!(summary.status, "valid");
        assert_eq!(summary.version, Some(1));
        assert!(
            summary
                .signer_key_id
                .as_deref()
                .is_some_and(|value| value.starts_with("ed25519:"))
        );
        assert!(
            summary
                .artifact_digest
                .as_deref()
                .is_some_and(|value| value.starts_with("sha256:"))
        );

        let binary_bytes = fs::read(binary_path)?;
        let footer = read_embedded_release_footer(&binary_bytes)?
            .expect("stamped binary should contain embedded footer");
        let embedded: EmbeddedReleaseAttestation = serde_json::from_slice(footer.payload_bytes)?;
        let claims = embedded.claims()?;
        let canonical_payload =
            claims.canonical_bytes(&embedded.signer_key_id, &embedded.signature_algorithm)?;
        assert_eq!(
            hex::decode(&embedded.signed_payload_hex)?,
            canonical_payload
        );
        assert_eq!(claims.node_version, "9.9.9");
        assert_eq!(claims.build_id, "build-123");
        assert_eq!(claims.commit, "abcdef");
        Ok(())
    }

    #[test]
    fn release_attestation_inspect_reports_invalid_after_tamper() -> DynResult<()> {
        let dir = make_temp_dir("xtask-attestation-invalid")?;
        let binary_path = dir.join("mesh-llm");
        fs::write(&binary_path, b"release-binary-v1")?;
        let (private_key_path, public_key_path) = write_test_keypair(&dir, 13)?;

        stamp_release_attestation(&[
            "--binary".to_string(),
            binary_path.display().to_string(),
            "--signing-key-file".to_string(),
            private_key_path.display().to_string(),
        ])?;

        let mut tampered = fs::read(&binary_path)?;
        tampered[0] ^= 0x01;
        fs::write(&binary_path, tampered)?;

        let summary = inspect_release_attestation_summary(&InspectArgs {
            binary: Some(binary_path),
            public_key_file: Some(public_key_path),
            json: true,
        })?;
        assert_eq!(summary.status, "invalid");
        assert!(
            summary
                .error
                .as_deref()
                .is_some_and(|error| error.contains("artifact digest mismatch"))
        );
        assert!(summary.artifact_digest.is_some());
        Ok(())
    }
}
