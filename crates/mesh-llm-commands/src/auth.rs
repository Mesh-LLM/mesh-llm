use std::io::IsTerminal;
use std::path::{Path, PathBuf};
use std::result::Result as StdResult;

use anyhow::{Context, Result, bail};
use iroh::{EndpointId, SecretKey};
use mesh_llm_cli::{AuthCommand, TrustCommand};
use mesh_llm_identity::{
    KEYCHAIN_SERVICE, OwnerKeychainLoadError, OwnerKeypair, SignedNodeOwnership, TrustPolicy,
    TrustStore, default_keystore_path, default_node_key_path, default_node_ownership_path,
    default_trust_store_path, keystore_exists, keystore_metadata, load_keystore,
    load_node_key_bytes_from_path, load_node_ownership, load_owner_keypair_from_keychain,
    load_trust_store, save_keystore, save_keystore_with_keychain, save_node_key_bytes_to_path,
    save_node_ownership, save_trust_store, sign_node_ownership, verify_node_ownership,
};
use zeroize::Zeroizing;

fn now_unix_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}

fn resolve_owner_key_path(owner_key: Option<PathBuf>) -> Result<PathBuf> {
    match owner_key {
        Some(path) => Ok(path),
        None => Ok(default_keystore_path()?),
    }
}

fn resolve_node_key_path(node_key: Option<PathBuf>) -> Result<PathBuf> {
    match node_key {
        Some(path) => Ok(path),
        None => Ok(default_node_key_path()?),
    }
}

fn load_node_key_from_path(path: &Path) -> Result<SecretKey> {
    Ok(SecretKey::from_bytes(&load_node_key_bytes_from_path(path)?))
}

fn save_node_key_to_path(path: &Path, key: &SecretKey) -> Result<()> {
    save_node_key_bytes_to_path(path, &key.to_bytes())?;
    Ok(())
}

pub fn run_auth_command(command: &AuthCommand) -> Result<()> {
    match command {
        AuthCommand::Init {
            owner_key,
            force,
            no_passphrase,
            keychain,
        } => run_init(owner_key.clone(), *force, *no_passphrase, *keychain),
        AuthCommand::Status {
            owner_key,
            node_key,
            node_ownership,
            trust_store,
        } => run_status(
            owner_key.clone(),
            node_key.clone(),
            node_ownership.clone(),
            trust_store.clone(),
        ),
        AuthCommand::SignNode {
            owner_key,
            node_key,
            out,
            hostname_hint,
            node_label,
            expires_in_hours,
        } => run_sign_node(
            owner_key.clone(),
            node_key.clone(),
            out.clone(),
            node_label.clone(),
            hostname_hint.clone(),
            *expires_in_hours,
        ),
        AuthCommand::RenewNode {
            owner_key,
            node_key,
            out,
            hostname_hint,
            node_label,
            expires_in_hours,
        } => run_renew_node(
            owner_key.clone(),
            node_key.clone(),
            out.clone(),
            node_label.clone(),
            hostname_hint.clone(),
            *expires_in_hours,
        ),
        AuthCommand::VerifyNode {
            file,
            node_id,
            trust_store,
            trust_policy,
        } => run_verify_node(
            file.clone(),
            node_id.clone(),
            trust_store.clone(),
            trust_policy.map(cli_trust_policy_to_identity),
        ),
        AuthCommand::RotateNode {
            owner_key,
            node_key,
            out,
            hostname_hint,
            node_label,
            expires_in_hours,
            revoke_current,
            reason,
            trust_store,
        } => run_rotate_node(
            owner_key.clone(),
            node_key.clone(),
            out.clone(),
            node_label.clone(),
            hostname_hint.clone(),
            *expires_in_hours,
            *revoke_current,
            reason.clone(),
            trust_store.clone(),
        ),
        AuthCommand::RevokeOwner {
            owner_id,
            reason,
            trust_store,
        } => run_revoke_owner(owner_id.clone(), reason.clone(), trust_store.clone()),
        AuthCommand::RevokeNode {
            cert_id,
            node_id,
            reason,
            trust_store,
        } => run_revoke_node(
            cert_id.clone(),
            node_id.clone(),
            reason.clone(),
            trust_store.clone(),
        ),
        AuthCommand::RotateOwner {
            owner_key,
            no_passphrase,
            force,
        } => run_rotate_owner(owner_key.clone(), *no_passphrase, *force),
        AuthCommand::Trust { command } => run_trust_command(command),
    }
}

fn cli_trust_policy_to_identity(value: mesh_llm_cli::TrustPolicy) -> TrustPolicy {
    match value {
        mesh_llm_cli::TrustPolicy::Off => TrustPolicy::Off,
        mesh_llm_cli::TrustPolicy::PreferOwned => TrustPolicy::PreferOwned,
        mesh_llm_cli::TrustPolicy::RequireOwned => TrustPolicy::RequireOwned,
        mesh_llm_cli::TrustPolicy::Allowlist => TrustPolicy::Allowlist,
    }
}

fn resolve_node_ownership_path(path: Option<PathBuf>) -> Result<PathBuf> {
    match path {
        Some(path) => Ok(path),
        None => Ok(default_node_ownership_path()?),
    }
}

fn resolve_trust_store_path(path: Option<PathBuf>) -> Result<PathBuf> {
    match path {
        Some(path) => Ok(path),
        None => Ok(default_trust_store_path()?),
    }
}

fn prompt_new_passphrase(no_passphrase: bool) -> Result<Option<Zeroizing<String>>> {
    if no_passphrase {
        return Ok(None);
    }

    let passphrase = Zeroizing::new(rpassword::prompt_password_stderr(
        "Enter passphrase (empty for none): ",
    )?);
    if passphrase.is_empty() {
        return Ok(None);
    }

    let confirm = Zeroizing::new(rpassword::prompt_password_stderr("Confirm passphrase: ")?);
    if passphrase.as_str() != confirm.as_str() {
        bail!("Passphrases do not match.");
    }

    Ok(Some(passphrase))
}

fn resolve_keystore_passphrase(path: &Path) -> Result<Option<Zeroizing<String>>> {
    let info = keystore_metadata(path)?;
    if !info.encrypted {
        return Ok(None);
    }

    if let Ok(passphrase) = std::env::var("MESH_LLM_OWNER_PASSPHRASE") {
        return Ok(Some(Zeroizing::new(passphrase)));
    }

    if std::io::stdin().is_terminal() && std::io::stderr().is_terminal() {
        let prompt = format!("Enter owner keystore passphrase for {}: ", path.display());
        let passphrase = rpassword::prompt_password_stderr(&prompt)?;
        return Ok(Some(Zeroizing::new(passphrase)));
    }

    Err(mesh_llm_identity::CryptoError::MissingPassphrase.into())
}

fn load_owner_keypair_from_path(path: &Path) -> Result<OwnerKeypair> {
    let info = keystore_metadata(path)?;
    if info.encrypted && std::env::var("MESH_LLM_OWNER_PASSPHRASE").is_err() {
        match load_owner_keypair_from_keychain(path) {
            Ok(keypair) => return Ok(keypair),
            Err(OwnerKeychainLoadError::NoEntry)
            | Err(OwnerKeychainLoadError::Crypto(
                mesh_llm_identity::CryptoError::DecryptionFailed,
            ))
            | Err(OwnerKeychainLoadError::Crypto(
                mesh_llm_identity::CryptoError::KeychainUnavailable { .. },
            ))
            | Err(OwnerKeychainLoadError::Crypto(
                mesh_llm_identity::CryptoError::KeychainAccessDenied { .. },
            )) => {}
            Err(OwnerKeychainLoadError::Crypto(err)) => {
                return Err(err)
                    .with_context(|| format!("Failed to load owner keystore {}", path.display()));
            }
        }
    }

    let passphrase = resolve_keystore_passphrase(path)?;
    load_keystore(path, passphrase.as_deref().map(|value| value.as_str()))
        .with_context(|| format!("Failed to load owner keystore {}", path.display()))
}

fn sign_node_certificate(
    owner: &OwnerKeypair,
    node_secret_key: &SecretKey,
    expires_in_hours: u64,
    node_label: Option<String>,
    hostname_hint: Option<String>,
) -> Result<SignedNodeOwnership> {
    if expires_in_hours == 0 {
        bail!("--expires-in-hours must be greater than 0");
    }

    let endpoint_id = EndpointId::from(node_secret_key.public());
    let expires_at_unix_ms =
        now_unix_ms().saturating_add(expires_in_hours.saturating_mul(60 * 60 * 1000));
    Ok(sign_node_ownership(
        owner,
        endpoint_id.as_bytes(),
        expires_at_unix_ms,
        node_label,
        hostname_hint,
    )?)
}

fn parse_node_id_hex(node_id: &str) -> Result<[u8; 32]> {
    let bytes = hex::decode(node_id).with_context(|| format!("Invalid node ID hex: {node_id}"))?;
    let array: [u8; 32] = bytes
        .try_into()
        .map_err(|_| anyhow::anyhow!("Node ID must be 32 bytes of hex"))?;
    Ok(array)
}

fn load_effective_trust_store(path: &Path) -> Result<TrustStore> {
    load_trust_store(path).with_context(|| format!("Failed to load trust store {}", path.display()))
}

/// Run `mesh-llm auth init`.
pub(crate) fn run_init(
    owner_key: Option<PathBuf>,
    force: bool,
    no_passphrase: bool,
    keychain: bool,
) -> Result<()> {
    let custom_owner_key = owner_key.is_some();
    let path = resolve_owner_key_path(owner_key)?;
    let existing_keystore = keystore_exists(&path);

    if existing_keystore && !force {
        bail!(
            "Owner keystore already exists at {}\nUse --force to overwrite.",
            path.display()
        );
    }

    let use_keychain = if keychain {
        if !mesh_llm_identity::keychain_available() {
            bail!(
                "No OS keychain backend is available on this host.\n\
                 Retry without --keychain to set a passphrase, or with --no-passphrase \
                 to store keys unencrypted."
            );
        }
        true
    } else {
        let available =
            (!existing_keystore && !no_passphrase) && mesh_llm_identity::keychain_available();
        should_default_to_keychain(existing_keystore, no_passphrase, available)
    };

    let keypair = OwnerKeypair::generate();
    let owner_id = keypair.owner_id();
    let sign_pk = hex::encode(keypair.verifying_key().as_bytes());
    let enc_pk = hex::encode(keypair.encryption_public_key().as_bytes());
    let source = if no_passphrase {
        save_keystore(&path, &keypair, None, force)?;
        PassphraseSource::None
    } else if use_keychain {
        let account = save_keystore_with_keychain(&path, &keypair, force)?;
        PassphraseSource::Keychain { account }
    } else {
        match prompt_new_passphrase(false)? {
            Some(passphrase) => {
                save_keystore(&path, &keypair, Some(passphrase.as_str()), force)?;
                PassphraseSource::Prompt
            }
            None => {
                save_keystore(&path, &keypair, None, force)?;
                PassphraseSource::None
            }
        }
    };
    let encrypted = !matches!(source, PassphraseSource::None);

    eprintln!();
    eprintln!("Owner keystore created.");
    eprintln!("Owner ID:        {owner_id}");
    eprintln!("Signing key:     {sign_pk}");
    eprintln!("Encryption key:  {enc_pk}");
    eprintln!("Path:            {}", path.display());
    eprintln!("Encrypted:       {}", if encrypted { "yes" } else { "no" });
    match &source {
        PassphraseSource::Keychain { account } => {
            eprintln!(
                "Unlock:          OS keychain (service={KEYCHAIN_SERVICE}, account={account})"
            );
        }
        PassphraseSource::Prompt => {
            eprintln!("Unlock:          passphrase prompt");
        }
        PassphraseSource::None => {}
    }
    eprintln!();
    eprintln!("Next steps:");
    match &source {
        PassphraseSource::Keychain { account } => {
            eprintln!(
                "- This keystore is unlock-bound to this machine's keychain. To share the same \
                 owner identity on another node, retrieve the passphrase from your OS keychain \
                 (service={KEYCHAIN_SERVICE}, account={account}) and enter it there, or re-run \
                 `auth init` with a manual passphrase so the same passphrase can be used everywhere."
            );
        }
        PassphraseSource::Prompt | PassphraseSource::None => {
            eprintln!(
                "- Copy this keystore to other trusted nodes that should share the same owner identity."
            );
        }
    }
    eprintln!("- Start mesh-llm and it will automatically attest nodes from this keystore.");
    if custom_owner_key {
        eprintln!(
            "- Pass --owner-key {} when starting mesh-llm.",
            path.display()
        );
    }

    Ok(())
}

/// Run `mesh-llm auth status`.
pub(crate) fn run_status(
    owner_key: Option<PathBuf>,
    node_key: Option<PathBuf>,
    node_ownership: Option<PathBuf>,
    trust_store: Option<PathBuf>,
) -> Result<()> {
    let owner_key_path = resolve_owner_key_path(owner_key)?;
    let node_key_path = resolve_node_key_path(node_key)?;
    let node_ownership_path = resolve_node_ownership_path(node_ownership)?;
    let trust_store_path = resolve_trust_store_path(trust_store)?;

    if !keystore_exists(&owner_key_path) {
        eprintln!("No owner keystore found at {}", owner_key_path.display());
        eprintln!("Run `mesh-llm auth init` to create one.");
    } else {
        let info = keystore_metadata(&owner_key_path)?;
        eprintln!("Owner keystore:  {}", owner_key_path.display());
        eprintln!("Status:          present");
        eprintln!(
            "Encrypted:       {}",
            if info.encrypted { "yes" } else { "no" }
        );
        eprintln!("Owner ID:        {}", info.owner_id);
        if let Some(ref spk) = info.signing_public_key {
            eprintln!("Signing key:     {spk}");
        }
        if let Some(ref epk) = info.encryption_public_key {
            eprintln!("Encryption key:  {epk}");
        }
        eprintln!("Created:         {}", info.created_at);
        if info.encrypted {
            match load_owner_keypair_from_keychain(&owner_key_path) {
                Ok(_) => {
                    eprintln!("Keystore:        valid (unlocked from OS keychain)");
                }
                Err(OwnerKeychainLoadError::Crypto(e)) => {
                    eprintln!(
                        "{}",
                        encrypted_keystore_keychain_status(OwnerKeychainLoadError::Crypto(e))
                    );
                }
                Err(e) => eprintln!("{}", encrypted_keystore_keychain_status(e)),
            }
        } else {
            match load_keystore(&owner_key_path, None) {
                Ok(_) => {
                    eprintln!("Keystore:        valid (keys loaded successfully)");
                }
                Err(e) => {
                    eprintln!("Keystore:        ERROR loading keys: {e}");
                }
            }
        }
    }

    eprintln!();

    let node_secret_key = if node_key_path.exists() {
        let node_secret_key = load_node_key_from_path(&node_key_path)?;
        let node_id = EndpointId::from(node_secret_key.public());
        eprintln!("Node key:        {}", node_key_path.display());
        eprintln!("Node ID:         {}", hex::encode(node_id.as_bytes()));
        Some(node_secret_key)
    } else {
        eprintln!("Node key:        missing ({})", node_key_path.display());
        None
    };

    let trust_store = load_effective_trust_store(&trust_store_path)?;
    eprintln!("Trust store:     {}", trust_store_path.display());
    eprintln!("Trust policy:    {:?}", trust_store.policy);
    eprintln!("Trusted owners:  {}", trust_store.trusted_owners.len());
    eprintln!("Revoked owners:  {}", trust_store.revoked_owners.len());
    eprintln!("Revoked certs:   {}", trust_store.revoked_node_certs.len());
    eprintln!("Revoked node IDs:{}", trust_store.revoked_node_ids.len());

    eprintln!();

    if node_ownership_path.exists() {
        let ownership = load_node_ownership(&node_ownership_path)?;
        let actual_node_id = node_secret_key
            .as_ref()
            .map(|key| EndpointId::from(key.public()).as_bytes().to_owned())
            .map(Ok)
            .unwrap_or_else(|| {
                parse_node_id_hex(&ownership.claim.node_endpoint_id).with_context(|| {
                    format!(
                        "failed to parse claim.node_endpoint_id in {}: {}",
                        node_ownership_path.display(),
                        ownership.claim.node_endpoint_id
                    )
                })
            })?;
        let summary = verify_node_ownership(
            Some(&ownership),
            &actual_node_id,
            &trust_store,
            trust_store.policy,
            now_unix_ms(),
        );
        eprintln!("Node cert:       {}", node_ownership_path.display());
        eprintln!("Cert ID:         {}", ownership.claim.cert_id);
        eprintln!("Claim node ID:   {}", ownership.claim.node_endpoint_id);
        eprintln!(
            "Owner ID:        {}",
            summary
                .owner_id
                .as_deref()
                .unwrap_or(ownership.claim.owner_id.as_str())
        );
        eprintln!("Status:          {:?}", summary.status);
        eprintln!(
            "Verified:        {}",
            if summary.verified { "yes" } else { "no" }
        );
        eprintln!("Expires at:      {}", ownership.claim.expires_at_unix_ms);
        if let Some(node_label) = summary.node_label.as_deref() {
            eprintln!("Node label:      {node_label}");
        }
        if let Some(hostname_hint) = summary.hostname_hint.as_deref() {
            eprintln!("Hostname hint:   {hostname_hint}");
        }
    } else {
        eprintln!(
            "Node cert:       missing ({})",
            node_ownership_path.display()
        );
    }

    Ok(())
}

pub(crate) fn run_sign_node(
    owner_key: Option<PathBuf>,
    node_key: Option<PathBuf>,
    out: Option<PathBuf>,
    node_label: Option<String>,
    hostname_hint: Option<String>,
    expires_in_hours: u64,
) -> Result<()> {
    let owner_key_path = resolve_owner_key_path(owner_key)?;
    let node_key_path = resolve_node_key_path(node_key)?;
    let output_path = resolve_node_ownership_path(out)?;
    let owner = load_owner_keypair_from_path(&owner_key_path)?;
    let node_secret_key = load_node_key_from_path(&node_key_path)?;
    let ownership = sign_node_certificate(
        &owner,
        &node_secret_key,
        expires_in_hours,
        node_label,
        hostname_hint,
    )?;
    save_node_ownership(&output_path, &ownership)?;

    eprintln!(
        "Signed node certificate written to {}",
        output_path.display()
    );
    eprintln!("Owner ID:        {}", ownership.claim.owner_id);
    eprintln!("Node ID:         {}", ownership.claim.node_endpoint_id);
    eprintln!("Cert ID:         {}", ownership.claim.cert_id);
    eprintln!("Expires at:      {}", ownership.claim.expires_at_unix_ms);

    Ok(())
}

pub(crate) fn run_renew_node(
    owner_key: Option<PathBuf>,
    node_key: Option<PathBuf>,
    out: Option<PathBuf>,
    node_label: Option<String>,
    hostname_hint: Option<String>,
    expires_in_hours: u64,
) -> Result<()> {
    run_sign_node(
        owner_key,
        node_key,
        out,
        node_label,
        hostname_hint,
        expires_in_hours,
    )
}

pub(crate) fn run_verify_node(
    file: Option<PathBuf>,
    node_id: Option<String>,
    trust_store: Option<PathBuf>,
    trust_policy: Option<TrustPolicy>,
) -> Result<()> {
    let certificate_path = resolve_node_ownership_path(file)?;
    let trust_store_path = resolve_trust_store_path(trust_store)?;
    let ownership = load_node_ownership(&certificate_path)?;
    let trust_store = load_effective_trust_store(&trust_store_path)?;
    let policy = trust_policy.unwrap_or(trust_store.policy);
    let actual_node_id = match node_id {
        Some(node_id) => parse_node_id_hex(&node_id)?,
        None => parse_node_id_hex(&ownership.claim.node_endpoint_id)?,
    };

    let summary = verify_node_ownership(
        Some(&ownership),
        &actual_node_id,
        &trust_store,
        policy,
        now_unix_ms(),
    );

    eprintln!("Certificate:     {}", certificate_path.display());
    eprintln!("Owner ID:        {}", ownership.claim.owner_id);
    eprintln!("Node ID:         {}", ownership.claim.node_endpoint_id);
    eprintln!("Cert ID:         {}", ownership.claim.cert_id);
    eprintln!("Trust policy:    {:?}", policy);
    eprintln!("Status:          {:?}", summary.status);
    eprintln!(
        "Verified:        {}",
        if summary.verified { "yes" } else { "no" }
    );
    eprintln!("Expires at:      {}", ownership.claim.expires_at_unix_ms);

    Ok(())
}

type RunRotateNodeFn = fn(
    Option<PathBuf>,
    Option<PathBuf>,
    Option<PathBuf>,
    Option<String>,
    Option<String>,
    u64,
    bool,
    Option<String>,
    Option<PathBuf>,
) -> StdResult<(), anyhow::Error>;

pub(crate) const RUN_ROTATE_NODE: RunRotateNodeFn =
    |owner_key,
     node_key,
     out,
     node_label,
     hostname_hint,
     expires_in_hours,
     revoke_current,
     reason,
     trust_store| {
        let node_key_path = resolve_node_key_path(node_key)?;
        let certificate_path = resolve_node_ownership_path(out)?;
        let trust_store_path = resolve_trust_store_path(trust_store)?;

        let previous_node_key = node_key_path
            .exists()
            .then(|| load_node_key_from_path(&node_key_path))
            .transpose()?;
        let previous_node_id = previous_node_key
            .as_ref()
            .map(|key| hex::encode(EndpointId::from(key.public()).as_bytes()));
        let previous_ownership = certificate_path
            .exists()
            .then(|| load_node_ownership(&certificate_path))
            .transpose()?;

        let mut trust_store = load_effective_trust_store(&trust_store_path)?;
        if revoke_current {
            if let Some(ref previous_ownership) = previous_ownership {
                trust_store
                    .revoke_node_cert(previous_ownership.claim.cert_id.clone(), reason.clone());
            }
            if let Some(ref previous_node_id) = previous_node_id {
                trust_store.revoke_node_id(previous_node_id.clone(), reason.clone());
            }
            save_trust_store(&trust_store_path, &trust_store)?;
        }

        let new_key = SecretKey::generate();
        save_node_key_to_path(&node_key_path, &new_key)?;

        eprintln!("Node key rotated at {}", node_key_path.display());
        if let Some(previous_node_id) = previous_node_id {
            eprintln!("Previous node ID:{previous_node_id}");
        }
        let new_node_id = hex::encode(EndpointId::from(new_key.public()).as_bytes());
        eprintln!("New node ID:     {new_node_id}");

        let owner_key_path = resolve_owner_key_path(owner_key)?;
        if !owner_key_path.exists() {
            eprintln!("No owner keystore found at {}", owner_key_path.display());
            eprintln!(
                "Run `mesh-llm auth init` or `mesh-llm auth sign-node` later to attest this node."
            );
            return Ok(());
        }

        let owner = load_owner_keypair_from_path(&owner_key_path)?;
        let ownership = sign_node_certificate(
            &owner,
            &new_key,
            expires_in_hours,
            node_label,
            hostname_hint,
        )?;
        save_node_ownership(&certificate_path, &ownership)?;
        eprintln!("New node certificate: {}", certificate_path.display());
        eprintln!("New cert ID:      {}", ownership.claim.cert_id);

        Ok(())
    };

pub(crate) use RUN_ROTATE_NODE as run_rotate_node;

pub(crate) fn run_revoke_owner(
    owner_id: String,
    reason: Option<String>,
    trust_store: Option<PathBuf>,
) -> Result<()> {
    let trust_store_path = resolve_trust_store_path(trust_store)?;
    let mut trust_store = load_effective_trust_store(&trust_store_path)?;
    trust_store.remove_trusted_owner(&owner_id);
    trust_store.revoke_owner(owner_id.clone(), reason);
    save_trust_store(&trust_store_path, &trust_store)?;

    eprintln!("Revoked owner {owner_id} in {}", trust_store_path.display());
    Ok(())
}

pub(crate) fn run_revoke_node(
    cert_id: Option<String>,
    node_id: Option<String>,
    reason: Option<String>,
    trust_store: Option<PathBuf>,
) -> Result<()> {
    if cert_id.is_none() && node_id.is_none() {
        bail!("Pass --cert-id, --node-id, or both.");
    }

    let trust_store_path = resolve_trust_store_path(trust_store)?;
    let mut trust_store = load_effective_trust_store(&trust_store_path)?;

    if let Some(cert_id) = cert_id {
        trust_store.revoke_node_cert(cert_id.clone(), reason.clone());
        eprintln!("Revoked cert ID {cert_id}");
    }
    if let Some(node_id) = node_id {
        let normalized = hex::encode(parse_node_id_hex(&node_id)?);
        trust_store.revoke_node_id(normalized.clone(), reason);
        eprintln!("Revoked node ID {normalized}");
    }

    save_trust_store(&trust_store_path, &trust_store)?;
    eprintln!("Updated trust store {}", trust_store_path.display());
    Ok(())
}

pub(crate) fn run_rotate_owner(
    owner_key: Option<PathBuf>,
    no_passphrase: bool,
    force: bool,
) -> Result<()> {
    let owner_key_path = resolve_owner_key_path(owner_key)?;
    let passphrase = prompt_new_passphrase(no_passphrase)?;
    let new_keypair = OwnerKeypair::generate();

    let backup_path = if owner_key_path.exists() {
        let timestamp = chrono::Utc::now().format("%Y%m%d%H%M%S");
        let backup_path =
            owner_key_path.with_file_name(format!("owner-keystore.{timestamp}.bak.json"));
        if backup_path.exists() && !force {
            bail!(
                "Backup file already exists at {} (use --force to overwrite).",
                backup_path.display()
            );
        }
        if backup_path.exists() {
            std::fs::remove_file(&backup_path)?;
        }
        std::fs::rename(&owner_key_path, &backup_path)?;
        Some(backup_path)
    } else {
        None
    };

    save_keystore(
        &owner_key_path,
        &new_keypair,
        passphrase.as_deref().map(|value| value.as_str()),
        true,
    )?;

    eprintln!("Rotated owner keystore at {}", owner_key_path.display());
    eprintln!("New owner ID:    {}", new_keypair.owner_id());
    if let Some(backup_path) = backup_path {
        eprintln!("Backup:          {}", backup_path.display());
    }

    Ok(())
}

pub(crate) fn run_trust_command(command: &TrustCommand) -> Result<()> {
    match command {
        TrustCommand::Add {
            owner_id,
            label,
            trust_store,
        } => {
            let trust_store_path = resolve_trust_store_path(trust_store.clone())?;
            let mut store = load_effective_trust_store(&trust_store_path)?;
            store.add_trusted_owner(owner_id.clone(), label.clone());
            save_trust_store(&trust_store_path, &store)?;
            eprintln!("Trusted owner {owner_id} in {}", trust_store_path.display());
        }
        TrustCommand::Remove {
            owner_id,
            trust_store,
        } => {
            let trust_store_path = resolve_trust_store_path(trust_store.clone())?;
            let mut store = load_effective_trust_store(&trust_store_path)?;
            if store.remove_trusted_owner(owner_id) {
                save_trust_store(&trust_store_path, &store)?;
                eprintln!(
                    "Removed trusted owner {owner_id} from {}",
                    trust_store_path.display()
                );
            } else {
                eprintln!("Trusted owner {owner_id} was not present.");
            }
        }
        TrustCommand::List { trust_store } => {
            let trust_store_path = resolve_trust_store_path(trust_store.clone())?;
            let store = load_effective_trust_store(&trust_store_path)?;
            eprintln!("Trust store:     {}", trust_store_path.display());
            eprintln!("Policy:          {:?}", store.policy);
            eprintln!();
            eprintln!("Trusted owners:");
            if store.trusted_owners.is_empty() {
                eprintln!("- none");
            } else {
                for owner in &store.trusted_owners {
                    match owner.label.as_deref() {
                        Some(label) => eprintln!("- {} ({label})", owner.owner_id),
                        None => eprintln!("- {}", owner.owner_id),
                    }
                }
            }
            eprintln!();
            eprintln!("Revoked owners:");
            if store.revoked_owners.is_empty() {
                eprintln!("- none");
            } else {
                for owner in &store.revoked_owners {
                    match owner.reason.as_deref() {
                        Some(reason) => eprintln!("- {} ({reason})", owner.owner_id),
                        None => eprintln!("- {}", owner.owner_id),
                    }
                }
            }
            eprintln!();
            eprintln!("Revoked node certs:");
            if store.revoked_node_certs.is_empty() {
                eprintln!("- none");
            } else {
                for cert in &store.revoked_node_certs {
                    match cert.reason.as_deref() {
                        Some(reason) => eprintln!("- {} ({reason})", cert.cert_id),
                        None => eprintln!("- {}", cert.cert_id),
                    }
                }
            }
            eprintln!();
            eprintln!("Revoked node IDs:");
            if store.revoked_node_ids.is_empty() {
                eprintln!("- none");
            } else {
                for node in &store.revoked_node_ids {
                    match node.reason.as_deref() {
                        Some(reason) => eprintln!("- {} ({reason})", node.node_endpoint_id),
                        None => eprintln!("- {}", node.node_endpoint_id),
                    }
                }
            }
        }
    }

    Ok(())
}

#[derive(Debug)]
enum PassphraseSource {
    None,
    Prompt,
    Keychain { account: String },
}

fn encrypted_keystore_keychain_status(error: OwnerKeychainLoadError) -> String {
    match error {
        OwnerKeychainLoadError::NoEntry => {
            "Keystore:        encrypted (no keychain entry for this path; provide the \
             passphrase when the owner keystore is consumed)"
                .into()
        }
        OwnerKeychainLoadError::Crypto(mesh_llm_identity::CryptoError::DecryptionFailed) => {
            "Keystore:        encrypted (keychain entry could not unlock this keystore; \
             provide the passphrase when the owner keystore is consumed or remove the stale \
             keychain entry for this path)"
                .into()
        }
        OwnerKeychainLoadError::Crypto(mesh_llm_identity::CryptoError::KeychainUnavailable {
            reason,
        }) => format!("Keystore:        encrypted (keychain unavailable: {reason})"),
        OwnerKeychainLoadError::Crypto(mesh_llm_identity::CryptoError::KeychainAccessDenied {
            reason,
        }) => format!(
            "Keystore:        encrypted (keychain is locked or access was denied: {reason}; \
             unlock the keychain and retry, or provide the passphrase when the owner keystore \
             is consumed)"
        ),
        OwnerKeychainLoadError::Crypto(e) => format!("Keystore:        ERROR loading keys: {e}"),
    }
}

fn should_default_to_keychain(
    existing_keystore: bool,
    no_passphrase: bool,
    keychain_available: bool,
) -> bool {
    !existing_keystore && !no_passphrase && keychain_available
}

#[cfg(test)]
mod tests;
