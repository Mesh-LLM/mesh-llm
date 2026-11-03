//! Local key management. No private key or password crosses a Mesh operation.
use alloy::signers::local::PrivateKeySigner;
use anyhow::{Context, Result, ensure};
use serde::{Deserialize, Serialize};
use std::{
    fs::{self, OpenOptions},
    io::Write,
    path::{Path, PathBuf},
};
use wallet_contract::{CHAIN_ID, OUSD};
use zeroize::Zeroizing;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Profile {
    pub wallet_id: String,
    pub address: String,
    pub chain_id: u64,
    pub token: String,
    pub key_file: String,
}

pub fn secure_dir(path: &Path) -> Result<()> {
    if !path.exists() {
        #[cfg(unix)]
        {
            use std::os::unix::fs::DirBuilderExt;
            fs::DirBuilder::new()
                .recursive(true)
                .mode(0o700)
                .create(path)?;
        }
        #[cfg(not(unix))]
        fs::create_dir_all(path)?;
    }
    ensure!(
        fs::symlink_metadata(path)?.file_type().is_dir(),
        "state directory must be a real directory"
    );
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        ensure!(
            fs::metadata(path)?.permissions().mode() & 0o077 == 0,
            "state directory must have owner-only permissions (chmod 700)"
        );
    }
    Ok(())
}

pub fn private_file(path: &Path) -> Result<()> {
    let metadata = fs::symlink_metadata(path)?;
    ensure!(
        metadata.is_file() && !metadata.file_type().is_symlink(),
        "credential/keystore must be a regular non-symlink file"
    );
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        ensure!(
            metadata.permissions().mode() & 0o077 == 0,
            "credential/keystore file must have owner-only permissions (chmod 600)"
        );
    }
    Ok(())
}

fn password(path: &Path) -> Result<Zeroizing<Vec<u8>>> {
    private_file(path)?;
    let mut bytes = Zeroizing::new(fs::read(path).context("reading password file")?);
    while bytes.last().is_some_and(|b| *b == b'\n' || *b == b'\r') {
        bytes.pop();
    }
    ensure!(bytes.len() >= 16, "use a password of at least 16 bytes");
    ensure!(bytes.len() <= 4096, "password file is too large");
    Ok(bytes)
}

pub fn load_profile(dir: &Path) -> Result<Profile> {
    let profile: Profile = serde_json::from_slice(
        &fs::read(dir.join("wallet.json"))
            .context("wallet not initialized; run the local init command first")?,
    )?;
    ensure!(
        profile.chain_id == CHAIN_ID && profile.token.eq_ignore_ascii_case(OUSD),
        "wallet profile is not Tempo mainnet OUSD"
    );
    ensure!(
        matches!(
            Path::new(&profile.key_file).components().next(),
            Some(std::path::Component::Normal(_))
        ) && Path::new(&profile.key_file).components().count() == 1
            && !profile.key_file.starts_with('.'),
        "invalid keystore filename"
    );
    wallet_contract::validate_address(&profile.address)?;
    Ok(profile)
}

pub fn unlock(
    dir: &Path,
    credential: Option<&Path>,
    profile: &Profile,
) -> Result<Option<PrivateKeySigner>> {
    let Some(credential) = credential else {
        return Ok(None);
    };
    let key_path = dir.join(&profile.key_file);
    private_file(&key_path)?;
    let password = password(credential)?;
    let signer = PrivateKeySigner::decrypt_keystore(key_path, password.as_slice())
        .context("unable to unlock wallet keystore")?;
    ensure!(
        format!("{}", signer.address()).eq_ignore_ascii_case(&profile.address),
        "keystore address does not match wallet profile"
    );
    Ok(Some(signer))
}

/// Called only by an explicit local CLI command, never during plugin startup.
pub fn initialize(dir: &Path, credential: &Path) -> Result<Profile> {
    secure_dir(dir)?;
    ensure!(
        !dir.join("wallet.json").exists(),
        "wallet already initialized; refusing to replace it"
    );
    let password = password(credential)?;
    let staging = dir.join(format!("init-{}", uuid::Uuid::new_v4()));
    secure_dir(&staging)?;
    let key_name = format!("keystore-{}.json", uuid::Uuid::new_v4());
    let (signer, _) = PrivateKeySigner::new_keystore(
        &staging,
        &mut rand::rngs::OsRng,
        password.as_slice(),
        Some(&key_name),
    )?;
    let staged_key = staging.join(&key_name);
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        fs::set_permissions(&staged_key, fs::Permissions::from_mode(0o600))?;
    }
    let restored = PrivateKeySigner::decrypt_keystore(&staged_key, password.as_slice())?;
    ensure!(
        restored.address() == signer.address(),
        "keystore round-trip verification failed"
    );
    OpenOptions::new()
        .read(true)
        .open(&staged_key)?
        .sync_all()?;
    fs::rename(&staged_key, dir.join(&key_name))?;
    let profile = Profile {
        wallet_id: uuid::Uuid::new_v4().to_string(),
        address: signer.address().to_string(),
        chain_id: CHAIN_ID,
        token: OUSD.into(),
        key_file: key_name,
    };
    let mut options = OpenOptions::new();
    options.create_new(true).write(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.mode(0o600);
    }
    let mut file = options.open(dir.join("wallet.json"))?;
    file.write_all(&serde_json::to_vec_pretty(&profile)?)?;
    file.sync_all()?;
    fs::remove_dir(&staging)?;
    #[cfg(unix)]
    OpenOptions::new().read(true).open(dir)?.sync_all()?;
    Ok(profile)
}

pub fn canonical_state_dir(path: &Path) -> Result<PathBuf> {
    secure_dir(path)?;
    Ok(fs::canonicalize(path)?)
}

#[cfg(all(test, unix))]
mod tests {
    use super::*;
    use alloy::primitives::B256;
    use rand::SeedableRng;
    use std::os::unix::fs::PermissionsExt;

    #[test]
    fn encrypted_fixture_unlocks_only_with_private_correct_credential() {
        let dir = tempfile::tempdir().unwrap();
        let password_file = dir.path().join("password");
        fs::write(&password_file, b"fixture-only-credential-password\n").unwrap();
        fs::set_permissions(&password_file, fs::Permissions::from_mode(0o600)).unwrap();
        // A published deterministic fixture key, never a production wallet.
        let signer = PrivateKeySigner::from_bytes(&B256::repeat_byte(1)).unwrap();
        let mut rng = rand::rngs::StdRng::seed_from_u64(7);
        let key_file = String::from("fixture-key.json");
        let (_, _uuid) = PrivateKeySigner::encrypt_keystore(
            dir.path(),
            &mut rng,
            signer.to_bytes(),
            b"fixture-only-credential-password",
            Some(&key_file),
        )
        .unwrap();
        fs::set_permissions(
            dir.path().join(&key_file),
            fs::Permissions::from_mode(0o600),
        )
        .unwrap();
        let profile = Profile {
            wallet_id: "fixture".into(),
            address: signer.address().to_string(),
            chain_id: CHAIN_ID,
            token: OUSD.into(),
            key_file,
        };
        assert!(unlock(dir.path(), None, &profile).unwrap().is_none());
        assert_eq!(
            unlock(dir.path(), Some(&password_file), &profile)
                .unwrap()
                .unwrap()
                .address(),
            signer.address()
        );
        fs::write(&password_file, b"another-fake-password-for-tests").unwrap();
        assert!(unlock(dir.path(), Some(&password_file), &profile).is_err());
        fs::set_permissions(&password_file, fs::Permissions::from_mode(0o644)).unwrap();
        assert!(unlock(dir.path(), Some(&password_file), &profile).is_err());
    }
}
