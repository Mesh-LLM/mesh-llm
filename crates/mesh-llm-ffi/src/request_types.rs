#[derive(uniffi::Record)]
pub struct ModelNative {
    pub id: String,
    pub name: String,
}

#[derive(uniffi::Record)]
pub struct ClientStatus {
    pub connected: bool,
    pub peer_count: u64,
}

#[derive(uniffi::Record)]
pub struct ConsoleOptionsNative {
    pub asset_dir: String,
    pub port: Option<u16>,
    pub listen_all: bool,
}

#[derive(uniffi::Record)]
pub struct PublicMeshQuery {
    pub model: Option<String>,
    pub min_vram_gb: Option<f64>,
    pub region: Option<String>,
    pub target_name: Option<String>,
    pub relays: Vec<String>,
}

#[derive(uniffi::Record)]
pub struct PublicMesh {
    pub invite_token: String,
    pub serving: Vec<String>,
    pub wanted: Vec<String>,
    pub on_disk: Vec<String>,
    pub total_vram_bytes: u64,
    pub node_count: u64,
    pub client_count: u64,
    pub max_clients: u64,
    pub name: Option<String>,
    pub region: Option<String>,
    pub mesh_id: Option<String>,
    pub publisher_npub: String,
    pub published_at: u64,
    pub expires_at: Option<u64>,
}

#[derive(uniffi::Record)]
pub struct ChatRequestNative {
    pub model: String,
    pub messages: Vec<ChatMessageNative>,
}

#[derive(uniffi::Record)]
pub struct ChatMessageNative {
    pub role: String,
    pub content: String,
}

#[derive(uniffi::Record)]
pub struct ResponsesRequestNative {
    pub model: String,
    pub input: String,
}
