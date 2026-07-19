use crate::command::{DynResult, ensure_eq, print_json, write_json_file};
use crate::repo_consistency::default_node_version;
use ed25519_dalek::{Signer, SigningKey};
use getrandom::fill as fill_random;
use mesh_llm_system::embedded_release_footer::{
    EmbeddedReleaseFooterStatus, EmbeddedReleasePayloadSummary, EmbeddedReleasePayloadVerifier,
    read_embedded_release_footer, stamp_embedded_release_payload, strip_embedded_release_footer,
    verify_embedded_release_footer,
};
use serde::Deserialize;
use std::fs;
use std::path::{Path, PathBuf};

const RELEASE_BUILD_ATTESTATION_VERSION: u32 = 1;
const RELEASE_SIGNING_PRIVATE_KEY_KIND: &str = "mesh-llm-release-signing-private-key-v1";
const RELEASE_SIGNING_PUBLIC_KEY_KIND: &str = "mesh-llm-release-signing-public-key-v1";
const RELEASE_BUILD_ATTESTATION_DOMAIN_TAG: &[u8] = b"mesh-llm-release-attestation-v1:";
const ED25519_SIGNATURE_ALGORITHM: &str = "ed25519";

#[derive(Debug, Clone, serde::Serialize, Deserialize)]
pub(crate) struct ReleaseBuildAttestationClaims {
    pub(crate) version: u32,
    pub(crate) node_version: String,
    pub(crate) build_id: String,
    pub(crate) commit: String,
    pub(crate) target_triple: String,
    pub(crate) supported_protocol_generation_min: Option<u32>,
    pub(crate) supported_protocol_generation_max: Option<u32>,
    pub(crate) artifact_digest: String,
}

#[derive(Debug, Clone, serde::Serialize, Deserialize)]
pub(crate) struct EmbeddedReleaseAttestation {
    pub(crate) version: u32,
    pub(crate) signer_key_id: String,
    pub(crate) signature_algorithm: String,
    pub(crate) claims: ReleaseBuildAttestationClaims,
    pub(crate) signed_payload_hex: String,
    pub(crate) signature_hex: String,
}

#[derive(Debug, Clone, serde::Serialize, Deserialize)]
pub(crate) struct ReleaseSigningPrivateKeyFile {
    pub(crate) kind: String,
    pub(crate) version: u32,
    pub(crate) algorithm: String,
    pub(crate) signer_key_id: String,
    pub(crate) seed_hex: String,
}

#[derive(Debug, Clone, serde::Serialize, Deserialize)]
pub(crate) struct ReleaseSigningPublicKeyFile {
    pub(crate) kind: String,
    pub(crate) version: u32,
    pub(crate) algorithm: String,
    pub(crate) signer_key_id: String,
    pub(crate) public_key_hex: String,
}

#[derive(Debug, Clone, serde::Serialize)]
pub(crate) struct ReleaseAttestationInspectSummary {
    pub(crate) status: String,
    pub(crate) version: Option<u32>,
    pub(crate) signer_key_id: Option<String>,
    pub(crate) artifact_digest: Option<String>,
    pub(crate) error: Option<String>,
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
        match (
            self.supported_protocol_generation_min,
            self.supported_protocol_generation_max,
        ) {
            (Some(min), Some(max)) if min > max => {
                return Err("invalid release build attestation protocol bounds".into());
            }
            _ => {}
        }
        if !self.artifact_digest.starts_with("sha256:") {
            return Err("release build attestation artifact digest must start with sha256:".into());
        }
        Ok(())
    }

    pub(crate) fn canonical_bytes(
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

    pub(crate) fn claims(&self) -> DynResult<ReleaseBuildAttestationClaims> {
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
    pub(crate) fn from_signing_key(signing_key: &SigningKey) -> Self {
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

    pub(crate) fn signing_key(&self) -> DynResult<SigningKey> {
        self.validate()?;
        signing_key_from_seed_hex(&self.seed_hex)
    }
}

impl ReleaseSigningPublicKeyFile {
    pub(crate) fn from_verifying_key(verifying_key: &ed25519_dalek::VerifyingKey) -> Self {
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

    pub(crate) fn verifying_key(&self) -> DynResult<ed25519_dalek::VerifyingKey> {
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
    pub(crate) binary: Option<PathBuf>,
    signing_key_file: Option<PathBuf>,
    node_version: Option<String>,
    build_id: Option<String>,
    commit: Option<String>,
    target_triple: Option<String>,
    protocol_min: Option<u32>,
    protocol_max: Option<u32>,
}

#[derive(Default)]
pub(crate) struct InspectArgs {
    pub(crate) binary: Option<PathBuf>,
    pub(crate) public_key_file: Option<PathBuf>,
    pub(crate) json: bool,
}

fn release_signer_key_id(verifying_key: &ed25519_dalek::VerifyingKey) -> String {
    format!("ed25519:{}", hex::encode(verifying_key.as_bytes()))
}

pub(crate) fn generate_release_attestation_keypair(args: &[String]) -> DynResult<()> {
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

pub(crate) fn stamp_release_attestation(args: &[String]) -> DynResult<()> {
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

pub(crate) fn inspect_release_attestation(args: &[String]) -> DynResult<()> {
    let parsed = parse_inspect_args(args)?;
    let summary = inspect_release_attestation_summary(&parsed)?;
    if parsed.json {
        print_json(&summary)
    } else {
        print_release_attestation_summary(&summary);
        Ok(())
    }
}

fn print_release_attestation_summary(summary: &ReleaseAttestationInspectSummary) {
    println!("release attestation: {}", summary.status);
    if let Some(version) = summary.version {
        println!("version: {version}");
    }
    if let Some(signer_key_id) = &summary.signer_key_id {
        println!("signer key: {signer_key_id}");
    }
    if let Some(artifact_digest) = &summary.artifact_digest {
        println!("artifact digest: {artifact_digest}");
    }
    if let Some(error) = &summary.error {
        println!("error: {error}");
    }
}

pub(crate) fn inspect_release_attestation_summary(
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
    let digest = artifact_digest
        .strip_prefix("sha256:")
        .unwrap_or(artifact_digest);
    format!("{stem}-{}", digest.get(..12).unwrap_or(digest))
}

fn default_commit() -> String {
    std::env::var("GIT_COMMIT").unwrap_or_else(|_| "task8-local".to_string())
}

fn default_target_triple() -> String {
    std::env::var("TARGET")
        .unwrap_or_else(|_| format!("{}-{}", std::env::consts::ARCH, std::env::consts::OS))
}
