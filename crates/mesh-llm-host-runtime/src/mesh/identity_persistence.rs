use super::*;

/// Generate a mesh ID for a new mesh.
/// Named meshes: `sha256("mesh-llm:" + name + ":" + nostr_pubkey)` — deterministic, unique per creator.
/// Unnamed meshes: random UUID, persisted to `~/.mesh-llm/mesh-id`.
pub fn generate_mesh_id(name: Option<&str>, nostr_pubkey: Option<&str>) -> Result<String> {
    if let Some(name) = name {
        use sha2::{Digest, Sha256};
        let mut hasher = Sha256::new();
        hasher.update(b"mesh-llm:");
        hasher.update(name.as_bytes());
        hasher.update(b":");
        hasher.update(nostr_pubkey.unwrap_or_default().as_bytes());
        Ok(hex::encode(hasher.finalize()))
    } else {
        generate_random_mesh_id_at(&mesh_id_path())
    }
}

fn random_mesh_id() -> String {
    format!(
        "{:016x}{:016x}",
        rand::random::<u64>(),
        rand::random::<u64>()
    )
}

pub(crate) fn generate_random_mesh_id_at(path: &std::path::Path) -> Result<String> {
    match std::fs::read_to_string(path) {
        Ok(id) => {
            let id = id.trim().to_string();
            if !id.is_empty() {
                return Ok(id);
            }
        }
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
        Err(error) => return Err(error).with_context(|| format!("read {}", path.display())),
    }
    let id = random_mesh_id();
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).with_context(|| format!("create {}", parent.display()))?;
    }
    std::fs::write(path, &id).with_context(|| format!("write {}", path.display()))?;
    Ok(id)
}

pub(crate) fn mesh_id_path() -> std::path::PathBuf {
    identity_state_dir().join("mesh-id")
}

pub(crate) fn mesh_genesis_policy_path() -> std::path::PathBuf {
    identity_state_dir().join("mesh-genesis-policy.json")
}

/// Save the mesh ID of the last mesh we successfully joined.
pub fn save_last_mesh_id(mesh_id: &str) -> Result<()> {
    let path = identity_state_dir().join("last-mesh");
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).with_context(|| format!("create {}", parent.display()))?;
    }
    std::fs::write(&path, mesh_id).with_context(|| format!("write {}", path.display()))?;
    Ok(())
}

/// Load the mesh ID of the last mesh we successfully joined.
pub fn load_last_mesh_id() -> Option<String> {
    let path = identity_state_dir().join("last-mesh");
    std::fs::read_to_string(&path)
        .ok()
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
}

// ---------------------------------------------------------------------------
// Public-to-private identity transition
// ---------------------------------------------------------------------------

/// Return the state directory owned by the active node key.
///
/// The historical default key keeps using `~/.mesh-llm`, while an explicit
/// key path gets a private namespace under `~/.mesh-llm/identities`. This
/// prevents one process from rotating another process's Nostr and mesh state.
pub(crate) fn identity_state_dir() -> std::path::PathBuf {
    let home = identity_home_dir();
    let active_key_path =
        default_node_key_path().unwrap_or_else(|_| home.join(".mesh-llm").join("key"));
    identity_state_dir_for(&home, &active_key_path)
}

pub(crate) fn identity_home_dir() -> std::path::PathBuf {
    #[cfg(test)]
    if let Some(home) = std::env::var_os("MESH_LLM_TEST_HOME") {
        return home.into();
    }
    dirs::home_dir().unwrap_or_else(|| std::path::PathBuf::from("."))
}

fn identity_state_dir_for(
    home: &std::path::Path,
    active_key_path: &std::path::Path,
) -> std::path::PathBuf {
    let dir = home.join(".mesh-llm");
    if active_key_path == dir.join("key") {
        return dir;
    }
    dir.join("identities")
        .join(key_path_digest(active_key_path))
}

fn key_path_digest(path: &std::path::Path) -> String {
    use sha2::{Digest, Sha256};
    let mut hasher = Sha256::new();
    hasher.update(key_path_bytes(path));
    hex::encode(hasher.finalize())
}

#[cfg(unix)]
fn key_path_bytes(path: &std::path::Path) -> Vec<u8> {
    use std::os::unix::ffi::OsStrExt;
    path.as_os_str().as_bytes().to_vec()
}

#[cfg(windows)]
fn key_path_bytes(path: &std::path::Path) -> Vec<u8> {
    use std::os::windows::ffi::OsStrExt;
    path.as_os_str()
        .encode_wide()
        .flat_map(u16::to_le_bytes)
        .collect()
}

#[cfg(not(any(unix, windows)))]
fn key_path_bytes(path: &std::path::Path) -> Vec<u8> {
    path.to_string_lossy().as_bytes().to_vec()
}

pub(crate) fn was_public_path() -> std::path::PathBuf {
    let home = identity_home_dir();
    let active_key_path =
        default_node_key_path().unwrap_or_else(|_| home.join(".mesh-llm").join("key"));
    identity_state_dir_for(&home, &active_key_path).join("was-public")
}

fn was_public_path_for(
    home: &std::path::Path,
    active_key_path: &std::path::Path,
) -> std::path::PathBuf {
    identity_state_dir_for(home, active_key_path).join("was-public")
}

pub(crate) fn clear_public_identity_file(path: &std::path::Path) -> Result<()> {
    if !path.exists() {
        return Ok(());
    }
    std::fs::remove_file(path).with_context(|| format!("remove {}", path.display()))?;
    tracing::info!("Cleared {}", path.display());
    Ok(())
}

/// Record that this node was started in public mode (--auto / --publish / --mesh-name).
/// Called at startup so we can detect a public→private transition next time.
pub fn mark_was_public() -> Result<()> {
    let home = identity_home_dir();
    let key_path = default_node_key_path()?;
    mark_was_public_at(&home, &key_path)
}

fn mark_was_public_at(home: &std::path::Path, active_key_path: &std::path::Path) -> Result<()> {
    let path = was_public_path_for(home, active_key_path);
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).with_context(|| format!("create {}", parent.display()))?;
    }
    std::fs::write(&path, "1").with_context(|| format!("write {}", path.display()))?;
    Ok(())
}

/// Returns true if the previous run was public (marker file exists).
pub fn was_previously_public() -> bool {
    was_public_path().exists()
}

#[cfg(test)]
fn was_previously_public_at(home: &std::path::Path, active_key_path: &std::path::Path) -> bool {
    was_public_path_for(home, active_key_path).exists()
}

/// Clear identity files (key, nostr.nsec, mesh-id, last-mesh, and the marker for
/// the active key) so the next start gets a completely fresh identity. Called
/// when transitioning from public → private to avoid reusing a publicly-known
/// identity in a private mesh.
pub fn clear_public_identity() -> Result<()> {
    let home = identity_home_dir();
    let key_path = default_node_key_path()?;
    clear_public_identity_at(&home, &key_path)
}

fn clear_public_identity_at(
    home: &std::path::Path,
    active_key_path: &std::path::Path,
) -> Result<()> {
    // The active key may live outside ~/.mesh-llm when a second process uses
    // MESH_LLM_NODE_KEY_PATH. Rotate that key too; otherwise a public-to-private
    // transition claims to rotate identity and immediately reloads the old key.
    clear_public_identity_file(active_key_path)?;
    let state_dir = identity_state_dir_for(home, active_key_path);
    if active_key_path == home.join(".mesh-llm").join("key") {
        clear_public_identity_file(&state_dir.join("key"))?;
    }
    for name in &["nostr.nsec", "mesh-id", "last-mesh"] {
        clear_public_identity_file(&state_dir.join(name))?;
    }
    let marker = state_dir.join("was-public");
    match std::fs::remove_file(&marker) {
        Ok(()) => Ok(()),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(error) => Err(error).with_context(|| format!("remove {}", marker.display())),
    }
}

/// Load secret key from ~/.mesh-llm/key, or create a new one and save it.
pub(crate) async fn load_or_create_key() -> Result<SecretKey> {
    let key_path = default_node_key_path()?;
    if key_path.exists() {
        let key = load_node_key_from_path(&key_path)?;
        tracing::info!("Loaded key from {}", key_path.display());
        return Ok(key);
    }

    let key = SecretKey::generate();
    save_node_key_to_path(&key_path, &key)?;
    tracing::info!("Generated new key, saved to {}", key_path.display());
    Ok(key)
}

pub fn default_node_key_path() -> Result<std::path::PathBuf> {
    #[cfg(test)]
    if std::env::var_os("MESH_LLM_TEST_HOME").is_some()
        && !std::env::var_os("MESH_LLM_NODE_KEY_PATH").is_some_and(|path| !path.is_empty())
    {
        return Ok(identity_home_dir().join(".mesh-llm").join("key"));
    }
    let path = mesh_llm_identity::default_node_key_path()?;
    if path.is_absolute() {
        return Ok(path);
    }
    Ok(std::env::current_dir()
        .context("resolve relative MESH_LLM_NODE_KEY_PATH")?
        .join(path))
}

pub fn load_node_key_from_path(path: &std::path::Path) -> Result<SecretKey> {
    Ok(SecretKey::from_bytes(
        &mesh_llm_identity::load_node_key_bytes_from_path(path)?,
    ))
}

pub fn save_node_key_to_path(path: &std::path::Path, key: &SecretKey) -> Result<()> {
    mesh_llm_identity::save_node_key_bytes_to_path(path, &key.to_bytes())?;
    Ok(())
}

#[cfg(test)]
mod clear_identity_tests {
    use super::*;
    use serial_test::serial;
    use std::ffi::OsString;

    struct NodeKeyPathEnvGuard(Option<OsString>);

    impl NodeKeyPathEnvGuard {
        fn set(value: &str) -> Self {
            let previous = std::env::var_os("MESH_LLM_NODE_KEY_PATH");
            // SAFETY: this guard is used only by #[serial] tests and restores the prior value.
            unsafe { std::env::set_var("MESH_LLM_NODE_KEY_PATH", value) };
            Self(previous)
        }
    }

    impl Drop for NodeKeyPathEnvGuard {
        fn drop(&mut self) {
            match self.0.take() {
                Some(value) => {
                    // SAFETY: this guard is used only by #[serial] tests and restores the prior value.
                    unsafe { std::env::set_var("MESH_LLM_NODE_KEY_PATH", value) }
                }
                None => {
                    // SAFETY: this guard is used only by #[serial] tests and restores the prior value.
                    unsafe { std::env::remove_var("MESH_LLM_NODE_KEY_PATH") }
                }
            }
        }
    }

    #[test]
    fn custom_rotation_preserves_default_and_clears_owned_state() {
        let root = tempfile::tempdir().expect("temp identity root");
        let home = root.path().join("home");
        let state_dir = home.join(".mesh-llm");
        let custom_key = root.path().join("second-node.key");
        let custom_state_dir = identity_state_dir_for(&home, &custom_key);
        std::fs::create_dir_all(&state_dir).expect("create state dir");
        std::fs::create_dir_all(&custom_state_dir).expect("create custom state dir");
        std::fs::write(&custom_key, b"public custom key").expect("write custom key");
        for name in ["key", "nostr.nsec", "mesh-id", "last-mesh"] {
            std::fs::write(state_dir.join(name), b"public state").expect("write public state");
        }
        for name in ["nostr.nsec", "mesh-id", "last-mesh"] {
            std::fs::write(custom_state_dir.join(name), b"custom public state")
                .expect("write custom public state");
        }
        let marker = was_public_path_for(&home, &custom_key);
        std::fs::write(&marker, b"public state").expect("write public marker");

        clear_public_identity_at(&home, &custom_key).expect("clear public identity");

        assert!(!custom_key.exists());
        for name in ["key", "nostr.nsec", "mesh-id", "last-mesh"] {
            assert!(state_dir.join(name).exists(), "default {name} was removed");
        }
        for name in ["nostr.nsec", "mesh-id", "last-mesh"] {
            assert!(
                !custom_state_dir.join(name).exists(),
                "custom {name} was not removed"
            );
        }
        assert!(!marker.exists());
    }

    #[test]
    fn public_transition_preserves_the_other_custom_key_marker() {
        let root = tempfile::tempdir().expect("temp identity root");
        let home = root.path().join("home");
        let state_dir = home.join(".mesh-llm");
        let first_key = root.path().join("first-node.key");
        let second_key = root.path().join("second-node.key");
        std::fs::create_dir_all(&state_dir).expect("create state dir");
        std::fs::write(&first_key, b"first public key").expect("write first key");
        std::fs::write(&second_key, b"second private key").expect("write second key");

        mark_was_public_at(&home, &first_key).expect("mark first key public");

        assert!(was_previously_public_at(&home, &first_key));
        assert!(!was_previously_public_at(&home, &second_key));
        assert!(second_key.exists());

        // A later public run for the second process gets its own marker, and
        // clearing the first process must preserve it.
        mark_was_public_at(&home, &second_key).expect("mark second key public");
        assert!(was_previously_public_at(&home, &second_key));

        clear_public_identity_at(&home, &first_key).expect("clear first identity");
        assert!(!first_key.exists());
        assert!(second_key.exists());
        assert!(was_previously_public_at(&home, &second_key));
    }

    #[test]
    fn default_key_keeps_legacy_marker_mapping() {
        let root = tempfile::tempdir().expect("temp identity root");
        let home = root.path().join("home");
        let state_dir = home.join(".mesh-llm");
        let default_key = state_dir.join("key");
        std::fs::create_dir_all(&state_dir).expect("create state dir");
        std::fs::write(&default_key, b"default public key").expect("write default key");
        std::fs::write(state_dir.join("was-public"), b"public state").expect("write legacy marker");

        assert_eq!(
            was_public_path_for(&home, &default_key),
            state_dir.join("was-public")
        );
        assert!(was_previously_public_at(&home, &default_key));

        clear_public_identity_at(&home, &default_key).expect("clear default identity");
        assert!(!default_key.exists());
        assert!(!state_dir.join("was-public").exists());
    }

    #[test]
    #[serial]
    fn relative_custom_key_paths_resolve_before_use() {
        let _env = NodeKeyPathEnvGuard::set("relative-node.key");
        let expected = std::env::current_dir()
            .expect("current dir")
            .join("relative-node.key");

        assert_eq!(
            default_node_key_path().expect("resolve node key path"),
            expected
        );
        assert!(expected.is_absolute());
    }

    #[cfg(unix)]
    #[test]
    fn key_path_digest_preserves_non_utf8_path_bytes() {
        use std::os::unix::ffi::OsStringExt;

        let first =
            std::path::PathBuf::from(OsString::from_vec(vec![b'n', b'o', b'd', b'e', 0x80]));
        let second =
            std::path::PathBuf::from(OsString::from_vec(vec![b'n', b'o', b'd', b'e', 0x81]));

        assert_ne!(key_path_digest(&first), key_path_digest(&second));
    }
}
