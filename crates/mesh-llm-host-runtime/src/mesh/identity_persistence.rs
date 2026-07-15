use super::*;

/// Generate a mesh ID for a new mesh.
/// Named meshes: `sha256("mesh-llm:" + name + ":" + nostr_pubkey)` — deterministic, unique per creator.
/// Unnamed meshes: random UUID, persisted to `~/.mesh-llm/mesh-id`.
pub fn generate_mesh_id(name: Option<&str>, nostr_pubkey: Option<&str>) -> String {
    if let Some(name) = name {
        use std::hash::{Hash, Hasher};
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        "mesh-llm:".hash(&mut hasher);
        name.hash(&mut hasher);
        if let Some(pk) = nostr_pubkey {
            pk.hash(&mut hasher);
        }
        format!("{:016x}", hasher.finish())
    } else {
        // Try to load persisted mesh-id
        let path = mesh_id_path();
        if let Ok(id) = std::fs::read_to_string(&path) {
            let id = id.trim().to_string();
            if !id.is_empty() {
                return id;
            }
        }
        // Generate new random ID and persist
        let id = format!(
            "{:016x}{:016x}",
            rand::random::<u64>(),
            rand::random::<u64>()
        );
        if let Some(parent) = path.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        let _ = std::fs::write(&path, &id);
        id
    }
}

pub(crate) fn mesh_id_path() -> std::path::PathBuf {
    dirs::home_dir()
        .unwrap_or_else(|| std::path::PathBuf::from("."))
        .join(".mesh-llm")
        .join("mesh-id")
}

pub(crate) fn mesh_genesis_policy_path() -> std::path::PathBuf {
    dirs::home_dir()
        .unwrap_or_else(|| std::path::PathBuf::from("."))
        .join(".mesh-llm")
        .join("mesh-genesis-policy.json")
}

/// Save the mesh ID of the last mesh we successfully joined.
pub fn save_last_mesh_id(mesh_id: &str) {
    let path = dirs::home_dir()
        .unwrap_or_else(|| std::path::PathBuf::from("."))
        .join(".mesh-llm")
        .join("last-mesh");
    if let Some(parent) = path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    let _ = std::fs::write(&path, mesh_id);
}

/// Load the mesh ID of the last mesh we successfully joined.
pub fn load_last_mesh_id() -> Option<String> {
    let path = dirs::home_dir()
        .unwrap_or_else(|| std::path::PathBuf::from("."))
        .join(".mesh-llm")
        .join("last-mesh");
    std::fs::read_to_string(&path)
        .ok()
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
}

// ---------------------------------------------------------------------------
// Public-to-private identity transition
// ---------------------------------------------------------------------------

pub(crate) fn was_public_path() -> std::path::PathBuf {
    dirs::home_dir()
        .unwrap_or_else(|| std::path::PathBuf::from("."))
        .join(".mesh-llm")
        .join("was-public")
}

pub(crate) fn clear_public_identity_file(path: &std::path::Path) -> bool {
    if !path.exists() {
        return true;
    }
    match std::fs::remove_file(path) {
        Ok(()) => {
            tracing::info!("Cleared {}", path.display());
            true
        }
        Err(_) => {
            tracing::warn!("Failed to clear {}", path.display());
            false
        }
    }
}

/// Record that this node was started in public mode (--auto / --publish / --mesh-name).
/// Called at startup so we can detect a public→private transition next time.
pub fn mark_was_public() {
    let path = was_public_path();
    if let Some(parent) = path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    let _ = std::fs::write(&path, "1");
}

/// Returns true if the previous run was public (marker file exists).
pub fn was_previously_public() -> bool {
    was_public_path().exists()
}

/// Clear identity files (key, nostr.nsec, mesh-id, last-mesh, was-public) so the
/// next start gets a completely fresh identity. Called when transitioning from
/// public → private to avoid reusing a publicly-known identity in a private mesh.
pub fn clear_public_identity() {
    let home = dirs::home_dir().unwrap_or_else(|| std::path::PathBuf::from("."));
    let dir = home.join(".mesh-llm");
    let mut ok = true;
    for name in &["key", "nostr.nsec", "mesh-id", "last-mesh"] {
        ok &= clear_public_identity_file(&dir.join(name));
    }
    // Only remove the marker after identity files are gone, so a failed
    // cleanup is retried on the next private start.
    let marker = dir.join("was-public");
    if ok {
        let _ = std::fs::remove_file(&marker);
    } else {
        tracing::warn!("Keeping was-public marker — will retry cleanup next start");
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
    Ok(mesh_llm_identity::default_node_key_path()?)
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
