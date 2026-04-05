use std::path::PathBuf;

use anyhow::{bail, Result};
use zeroize::Zeroizing;

use crate::crypto::{
    default_keystore_path, keystore_exists, keystore_metadata, load_keystore, save_keystore,
    OwnerKeypair,
};

/// Run `mesh-llm auth init`.
pub(crate) fn run_init(owner_key: Option<PathBuf>, force: bool, no_passphrase: bool) -> Result<()> {
    let path = match owner_key {
        Some(p) => p,
        None => default_keystore_path()?,
    };

    if keystore_exists(&path) && !force {
        bail!(
            "Owner keystore already exists at {}\nUse --force to overwrite.",
            path.display()
        );
    }

    let passphrase = if no_passphrase {
        None
    } else {
        let pass = Zeroizing::new(rpassword::prompt_password_stderr(
            "Enter passphrase (empty for none): ",
        )?);
        if pass.is_empty() {
            None
        } else {
            let confirm =
                Zeroizing::new(rpassword::prompt_password_stderr("Confirm passphrase: ")?);
            if pass.as_str() != confirm.as_str() {
                bail!("Passphrases do not match.");
            }
            Some(pass)
        }
    };
    let encrypted = passphrase.is_some();

    let keypair = OwnerKeypair::generate();
    let owner_id = keypair.owner_id();
    let sign_pk = hex::encode(keypair.verifying_key().as_bytes());
    let enc_pk = hex::encode(keypair.encryption_public_key().as_bytes());

    save_keystore(
        &path,
        &keypair,
        passphrase.as_ref().map(|pass| pass.as_str()),
        force,
    )?;

    eprintln!();
    eprintln!("Owner keystore created.");
    eprintln!("Owner ID:        {owner_id}");
    eprintln!("Signing key:     {sign_pk}");
    eprintln!("Encryption key:  {enc_pk}");
    eprintln!("Path:            {}", path.display());
    eprintln!("Encrypted:       {}", if encrypted { "yes" } else { "no" });
    eprintln!();
    eprintln!("Next steps:");
    eprintln!(
        "- Copy this keystore to other trusted nodes that should share the same owner identity."
    );
    eprintln!("- Start mesh-llm with --owner-key {}", path.display());

    Ok(())
}

/// Run `mesh-llm auth status`.
pub(crate) fn run_status(owner_key: Option<PathBuf>) -> Result<()> {
    let path = match owner_key {
        Some(p) => p,
        None => default_keystore_path()?,
    };

    if !keystore_exists(&path) {
        eprintln!("No owner keystore found at {}", path.display());
        eprintln!("Run `mesh-llm auth init` to create one.");
        return Ok(());
    }

    let info = keystore_metadata(&path)?;

    eprintln!("Owner keystore:  {}", path.display());
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

    // If not encrypted, verify we can load the full keypair.
    if !info.encrypted {
        match load_keystore(&path, None) {
            Ok(_) => eprintln!("Keystore:        valid (keys loaded successfully)"),
            Err(e) => eprintln!("Keystore:        ERROR loading keys: {e}"),
        }
    }

    Ok(())
}
