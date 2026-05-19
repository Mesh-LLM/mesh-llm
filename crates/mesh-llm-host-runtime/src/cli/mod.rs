use clap::{Parser, Subcommand, ValueEnum};
use std::ffi::OsString;
use std::net::IpAddr;
use std::path::PathBuf;

use crate::cli::benchmark::BenchmarkCommand;
use crate::cli::runtime::RuntimeCommand;
use crate::crypto::TrustPolicy;
use crate::network::discovery::MeshDiscoveryMode;

/// Parse a `URL=TOKEN` pair for `--relay-auth`. Splits on the first `=` only,
/// so tokens may contain `=` (base64 padding, JWTs).
///
/// Error messages must never include the token portion of the input —
/// `--relay-auth` carries bearer credentials, and a parse failure could
/// otherwise leak them into terminal output, logs, and bug reports. The URL
/// is safe to echo back (it's the public identity of the relay).
fn parse_relay_auth_pair(s: &str) -> Result<(String, String), String> {
    let Some((url, token)) = s.split_once('=') else {
        return Err("expected URL=TOKEN, no '=' separator found (token redacted)".to_string());
    };
    if url.is_empty() {
        return Err("expected URL=TOKEN, got empty URL (token redacted)".to_string());
    }
    if token.is_empty() {
        return Err(format!(
            "expected URL=TOKEN, got empty token for URL {url:?}"
        ));
    }
    Ok((url.to_string(), token.to_string()))
}

#[cfg(test)]
mod relay_auth_parser_tests {
    use super::parse_relay_auth_pair;

    #[test]
    fn parses_simple_pair() {
        let (url, token) = parse_relay_auth_pair("https://r.example/=abc123").unwrap();
        assert_eq!(url, "https://r.example/");
        assert_eq!(token, "abc123");
    }

    #[test]
    fn preserves_equals_in_token() {
        // Base64-padded tokens and NIP-98-style payloads often contain `=`.
        let (_, token) = parse_relay_auth_pair("https://r/=eyJhbGciOiJFZERTQSJ9.payload==")
            .expect("token with '=' must parse");
        assert_eq!(token, "eyJhbGciOiJFZERTQSJ9.payload==");
    }

    #[test]
    fn rejects_missing_separator() {
        assert!(parse_relay_auth_pair("no-separator").is_err());
    }

    #[test]
    fn rejects_empty_url() {
        assert!(parse_relay_auth_pair("=token").is_err());
    }

    #[test]
    fn rejects_empty_token() {
        assert!(parse_relay_auth_pair("https://r/=").is_err());
    }

    #[test]
    fn parser_errors_never_leak_token_portion() {
        // --relay-auth carries bearer credentials; if parsing fails, the
        // token portion of the input must never appear in the error
        // message (which lands in terminal output, logs, and bug reports).
        // The URL is safe to echo back — it's the public identity of the
        // relay — but everything after the first `=` is secret.
        let secret_token = "super-secret-bearer-token-xyz-12345";

        // Case 1: no `=` separator. Whole input is treated as a malformed
        // URL-or-token blob; we cannot tell which it is, so redact both.
        let err = parse_relay_auth_pair(secret_token).expect_err("should fail");
        assert!(
            !err.contains(secret_token),
            "missing-separator error must not echo the input: {err}"
        );

        // Case 2: empty URL (`=token`). URL is empty, the token portion is
        // the secret — must not appear.
        let err = parse_relay_auth_pair(&format!("={secret_token}")).expect_err("should fail");
        assert!(
            !err.contains(secret_token),
            "empty-URL error must not echo the token: {err}"
        );

        // Case 3: empty token (`URL=`). Token is empty, no secret to leak;
        // the URL is fine to include and helps the user diagnose.
        let err = parse_relay_auth_pair("https://r.example/=").expect_err("should fail");
        assert!(
            err.contains("https://r.example/"),
            "empty-token error should name the URL: {err}"
        );
    }
}

#[derive(Subcommand, Debug)]
pub(crate) enum TrustCommand {
    /// Add an owner to the local trust store allowlist.
    Add {
        /// Owner ID to trust.
        owner_id: String,
        /// Optional human label for this owner.
        #[arg(long)]
        label: Option<String>,
        /// Path to the trust store file.
        #[arg(long)]
        trust_store: Option<PathBuf>,
    },
    /// Remove an owner from the local trust store allowlist.
    Remove {
        /// Owner ID to remove.
        owner_id: String,
        /// Path to the trust store file.
        #[arg(long)]
        trust_store: Option<PathBuf>,
    },
    /// Show the current trust store contents.
    List {
        /// Path to the trust store file.
        #[arg(long)]
        trust_store: Option<PathBuf>,
    },
}

#[derive(Subcommand, Debug)]
pub(crate) enum AuthCommand {
    /// Generate a new owner keypair and save to keystore.
    Init {
        /// Path to the owner keystore.
        #[arg(long)]
        owner_key: Option<PathBuf>,
        /// Overwrite an existing keystore.
        #[arg(long)]
        force: bool,
        /// Skip passphrase prompt (store keys unencrypted).
        #[arg(long, conflicts_with = "keychain")]
        no_passphrase: bool,
        /// Store a random unlock passphrase in the OS keychain (macOS Keychain,
        /// Windows Credential Manager, Linux Secret Service). New keystores
        /// already default to this when a backend is available; use this flag
        /// to force it when overwriting an existing keystore.
        #[arg(long)]
        keychain: bool,
    },
    /// Show current owner identity status.
    Status {
        /// Path to the owner keystore.
        #[arg(long)]
        owner_key: Option<PathBuf>,
        /// Path to the node identity file (default: ~/.mesh-llm/key).
        #[arg(long)]
        node_key: Option<PathBuf>,
        /// Path to the node ownership certificate.
        #[arg(long)]
        node_ownership: Option<PathBuf>,
        /// Path to the trust store file.
        #[arg(long)]
        trust_store: Option<PathBuf>,
    },
    /// Sign the current node identity with the existing owner keystore.
    SignNode {
        /// Path to the owner keystore.
        #[arg(long)]
        owner_key: Option<PathBuf>,
        /// Path to the node identity file (default: ~/.mesh-llm/key).
        #[arg(long)]
        node_key: Option<PathBuf>,
        /// Output path for the signed node certificate.
        #[arg(long)]
        out: Option<PathBuf>,
        /// Optional hostname hint attached to the certificate.
        #[arg(long)]
        hostname_hint: Option<String>,
        /// Optional human label attached to this node certificate.
        #[arg(long)]
        node_label: Option<String>,
        /// Certificate lifetime in hours.
        #[arg(long, default_value = "168")]
        expires_in_hours: u64,
    },
    /// Renew the local node ownership certificate in place.
    RenewNode {
        /// Path to the owner keystore.
        #[arg(long)]
        owner_key: Option<PathBuf>,
        /// Path to the node identity file (default: ~/.mesh-llm/key).
        #[arg(long)]
        node_key: Option<PathBuf>,
        /// Output path for the signed node certificate.
        #[arg(long)]
        out: Option<PathBuf>,
        /// Optional hostname hint attached to the certificate.
        #[arg(long)]
        hostname_hint: Option<String>,
        /// Optional human label attached to this node certificate.
        #[arg(long)]
        node_label: Option<String>,
        /// Certificate lifetime in hours.
        #[arg(long, default_value = "168")]
        expires_in_hours: u64,
    },
    /// Verify a node ownership certificate.
    VerifyNode {
        /// Path to the signed node certificate.
        #[arg(long)]
        file: Option<PathBuf>,
        /// Override the node ID to verify against.
        #[arg(long)]
        node_id: Option<String>,
        /// Path to the trust store file.
        #[arg(long)]
        trust_store: Option<PathBuf>,
        /// Override trust policy used for verification.
        #[arg(long = "verify-trust-policy", value_enum)]
        trust_policy: Option<TrustPolicy>,
    },
    /// Rotate the local node identity key.
    RotateNode {
        /// Path to the owner keystore.
        #[arg(long)]
        owner_key: Option<PathBuf>,
        /// Path to the node identity file (default: ~/.mesh-llm/key).
        #[arg(long)]
        node_key: Option<PathBuf>,
        /// Output path for the signed node certificate.
        #[arg(long)]
        out: Option<PathBuf>,
        /// Optional hostname hint attached to the certificate.
        #[arg(long)]
        hostname_hint: Option<String>,
        /// Optional human label attached to this node certificate.
        #[arg(long)]
        node_label: Option<String>,
        /// Certificate lifetime in hours.
        #[arg(long, default_value = "168")]
        expires_in_hours: u64,
        /// Revoke the current certificate and node ID in the local trust store first.
        #[arg(long)]
        revoke_current: bool,
        /// Optional revocation reason stored in the trust store.
        #[arg(long)]
        reason: Option<String>,
        /// Path to the trust store file.
        #[arg(long)]
        trust_store: Option<PathBuf>,
    },
    /// Revoke an owner in the local trust store.
    RevokeOwner {
        /// Owner ID to revoke.
        owner_id: String,
        /// Optional reason stored in the trust store.
        #[arg(long)]
        reason: Option<String>,
        /// Path to the trust store file.
        #[arg(long)]
        trust_store: Option<PathBuf>,
    },
    /// Revoke a node certificate or node ID in the local trust store.
    RevokeNode {
        /// Certificate ID to revoke.
        #[arg(long)]
        cert_id: Option<String>,
        /// Node endpoint ID to revoke.
        #[arg(long)]
        node_id: Option<String>,
        /// Optional reason stored in the trust store.
        #[arg(long)]
        reason: Option<String>,
        /// Path to the trust store file.
        #[arg(long)]
        trust_store: Option<PathBuf>,
    },
    /// Rotate the existing owner keystore identity.
    RotateOwner {
        /// Path to the owner keystore.
        #[arg(long)]
        owner_key: Option<PathBuf>,
        /// Skip passphrase prompt (store keys unencrypted).
        #[arg(long)]
        no_passphrase: bool,
        /// Overwrite an existing backup file if present.
        #[arg(long)]
        force: bool,
    },
    /// Manage the local trust store.
    Trust {
        #[command(subcommand)]
        command: TrustCommand,
    },
}

#[derive(Subcommand, Debug)]
pub(crate) enum GpuCommand {
    /// Force a fresh local GPU benchmark and rewrite the cached fingerprint.
    Benchmark {
        /// Print machine-readable JSON output.
        #[arg(long)]
        json: bool,
    },
}

pub(crate) mod benchmark;
pub(crate) mod commands;
pub mod models;
pub mod output;
pub(crate) mod pager;
pub(crate) mod runtime;
pub(crate) mod shell;
pub(crate) mod terminal_progress;

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, ValueEnum)]
pub enum LogFormat {
    #[default]
    Pretty,
    Json,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, ValueEnum)]
pub(crate) enum MeshGuardrailCliMode {
    #[default]
    Disabled,
    Metrics,
    Enforce,
}

impl MeshGuardrailCliMode {
    pub(crate) const fn as_str(self) -> &'static str {
        match self {
            Self::Disabled => "disabled",
            Self::Metrics => "metrics",
            Self::Enforce => "enforce",
        }
    }

    pub(crate) const fn to_guardrail_mode(self) -> openai_frontend::GuardrailMode {
        match self {
            Self::Disabled => openai_frontend::GuardrailMode::Disabled,
            Self::Metrics => openai_frontend::GuardrailMode::MetricsOnly,
            Self::Enforce => openai_frontend::GuardrailMode::Enforce,
        }
    }
}

#[derive(Parser, Debug)]
#[command(
    name = "mesh-llm",
    version = crate::VERSION,
    about = "Pool GPUs over the internet for LLM inference",
    after_help = "Preferred runtime entrypoints:\n  mesh-llm serve\n  mesh-llm serve --model Qwen3-8B-Q4_K_M\n  mesh-llm client --auto\n  mesh-llm gpus\n\n`mesh-llm serve` loads startup models from ~/.mesh-llm/config.toml.\nRun with --help-advanced for all options.\n\nExternal backends (vLLM, TGI, Ollama):\n  Install the plugin:\n    mesh-llm plugins install openai-endpoint\n  Add to ~/.mesh-llm/config.toml:\n    [[plugin]]\n    name = \"openai-endpoint\"\n    url = \"http://gpu-box:8000/v1\"\n  Then: mesh-llm serve     (or: mesh-llm client  for client-only mode)\n\nFlash-MoE SSD backend:\n  Install the plugin:\n    mesh-llm plugins install flash-moe\n  Add [[plugin]] name = \"flash-moe\" with url or plugin-owned args.\n  Then: mesh-llm serve     (or: mesh-llm client  for client-only mode)"
)]
pub(crate) struct Cli {
    #[command(subcommand)]
    pub(crate) command: Option<Command>,

    /// Terminal output format for app-owned runtime events.
    #[arg(long, value_enum, default_value_t = LogFormat::Pretty)]
    pub(crate) log_format: LogFormat,

    /// Enable mesh runtime debug output; set MESH_LLM_DEBUG_NATIVE_VERBOSE=1 for verbose llama.cpp native logs.
    #[arg(long)]
    pub(crate) debug: bool,

    /// OTLP/gRPC endpoint for embedded Skippy debug telemetry, for example http://127.0.0.1:14317.
    #[arg(long, hide = true)]
    pub(crate) skippy_metrics_otlp_grpc: Option<String>,

    /// Server-side mesh guardrail mode for hosted Skippy backends.
    #[arg(long = "mesh-guardrails", value_enum, default_value_t = MeshGuardrailCliMode::Disabled)]
    pub(crate) mesh_guardrails: MeshGuardrailCliMode,

    /// Show all options (including advanced/niche ones).
    #[arg(long, hide = true)]
    pub(crate) help_advanced: bool,

    /// Join a mesh via invite token (can repeat).
    #[arg(long, short)]
    pub(crate) join: Vec<String>,

    /// Discover a mesh and join it.
    #[arg(long, default_missing_value = "", num_args = 0..=1)]
    pub(crate) discover: Option<String>,

    /// Auto-join the best mesh found via discovery.
    #[arg(long)]
    pub(crate) auto: bool,

    /// Discovery provider for --auto, --discover, --publish, and the discover command.
    #[arg(long, value_enum, default_value_t = MeshDiscoveryMode::Nostr, global = true)]
    pub(crate) mesh_discovery_mode: MeshDiscoveryMode,

    /// Model to serve (path, remote catalog name, or Hugging Face ref).
    #[arg(long)]
    pub(crate) model: Vec<PathBuf>,

    /// Raw local GGUF file to serve directly (repeatable).
    #[arg(long)]
    pub(crate) gguf: Vec<PathBuf>,

    /// Explicit mmproj sidecar for the primary served model.
    #[arg(long, hide = true)]
    pub(crate) mmproj: Option<PathBuf>,

    /// API port (default: 9337).
    #[arg(long, default_value = "9337")]
    pub(crate) port: u16,

    /// Run as a client — no GPU, no model needed.
    #[arg(long)]
    pub(crate) client: bool,

    /// Web console port (default: 3131).
    #[arg(long, default_value = "3131")]
    pub(crate) console: u16,

    /// Disable the embedded web UI but keep the management API on the --console port.
    #[arg(long)]
    pub(crate) headless: bool,

    /// Write passive swarm debug capture JSONL to this local directory (opt-in, no telemetry egress).
    #[arg(long)]
    pub(crate) swarm_capture: Option<PathBuf>,

    /// Publish this mesh for discovery by other nodes.
    /// Without this flag, your mesh is private and only joinable via invite token.
    #[arg(long)]
    pub(crate) publish: bool,

    /// Human-readable name for this mesh (shown in discovery when combined with --publish).
    /// Naming a mesh does NOT make it publicly discoverable — use --publish for that.
    #[arg(long)]
    pub(crate) mesh_name: Option<String>,

    /// Region tag, e.g. "US", "EU", "AU" (shown in discovery).
    #[arg(long)]
    pub(crate) region: Option<String>,

    /// Minimum mesh-llm node version required when creating a new mesh.
    #[arg(long)]
    pub(crate) min_node_version: Option<String>,

    /// Maximum mesh-llm node version allowed when creating a new mesh.
    #[arg(long)]
    pub(crate) max_node_version: Option<String>,

    /// Minimum protocol generation required when creating a new mesh.
    #[arg(long)]
    pub(crate) min_protocol_version: Option<u32>,

    /// Maximum protocol generation allowed when creating a new mesh.
    #[arg(long)]
    pub(crate) max_protocol_version: Option<u32>,

    /// Require release attestation when creating a new mesh.
    #[arg(long)]
    pub(crate) require_release_attestation: bool,

    /// Allowed release signer key for mesh creation-time attestation policy (repeatable).
    #[arg(long = "release-signer-key")]
    pub(crate) release_signer_key: Vec<String>,

    /// Display name for this node.
    #[arg(long)]
    pub(crate) name: Option<String>,

    /// Internal plugin service mode.
    #[arg(long, hide = true)]
    pub(crate) plugin: Option<String>,

    /// Update mesh-llm before continuing for release-bundle installs if a newer bundled release is available.
    #[arg(long, global = true)]
    pub(crate) auto_update: bool,

    // ── Advanced options (hidden from default --help) ─────────────
    /// Draft model for speculative decoding.
    #[arg(long, hide = true)]
    pub(crate) draft: Option<PathBuf>,

    /// Max draft tokens (default: 8).
    #[arg(long, default_value = "8", hide = true)]
    pub(crate) draft_max: u16,

    /// Disable automatic draft model detection.
    #[arg(long, hide = true)]
    pub(crate) no_draft: bool,

    /// Force tensor split even if the model fits on one node.
    #[arg(long, hide = true)]
    pub(crate) split: bool,

    /// Override context size (tokens). Default: auto-scaled to available VRAM.
    #[arg(long, hide = true)]
    pub(crate) ctx_size: Option<u32>,

    /// Cap VRAM used for planning, local-fit decisions, and mesh advertisement (GB).
    #[arg(long)]
    pub(crate) max_vram: Option<f64>,

    /// Disable broadcasting GPU name, hostname, VRAM, and reserved bytes to peers. By default all nodes announce this hardware info.
    #[arg(long = "no-enumerate-host", hide = true)]
    pub(crate) no_enumerate_host: bool,

    /// Path to bundled mesh support binaries.
    #[arg(long, hide = true)]
    pub(crate) bin_dir: Option<PathBuf>,

    /// Override which bundled llama.cpp flavor to use.
    #[arg(long, value_enum)]
    pub(crate) llama_flavor: Option<crate::system::backend::BinaryFlavor>,

    /// Device override for local backend selection.
    #[arg(long, hide = true)]
    pub(crate) device: Option<String>,

    /// Deprecated tensor split override retained for CLI compatibility.
    #[arg(long, hide = true)]
    pub(crate) tensor_split: Option<String>,

    /// Override iroh relay URLs.
    #[arg(long, hide = true)]
    pub(crate) relay: Vec<String>,

    /// Per-relay bearer token for gated iroh relays, formatted as
    /// `URL=TOKEN`. Repeatable. The token is sent as
    /// `Authorization: Bearer <TOKEN>` on the WebSocket upgrade to the
    /// matching `--relay` URL. Relays not listed here register without
    /// authentication (the correct behavior for public relays).
    ///
    /// Splits on the first `=` only, so tokens may contain `=` (base64
    /// padding, JWTs, etc.).
    #[arg(long = "relay-auth", value_parser = parse_relay_auth_pair, hide = true)]
    pub(crate) relay_auth: Vec<(String, String)>,

    /// Bind QUIC to a fixed UDP port (for NAT port forwarding).
    #[arg(long, hide = true)]
    pub(crate) bind_port: Option<u16>,

    /// Bind mesh QUIC to a specific local IP address.
    #[arg(long, hide = true)]
    pub(crate) bind_ip: Option<IpAddr>,

    /// Bind to 0.0.0.0 (for containers/Fly.io).
    #[arg(long, hide = true)]
    pub(crate) listen_all: bool,

    /// Stop advertising when N clients connected.
    #[arg(long, hide = true)]
    pub(crate) max_clients: Option<usize>,

    /// Custom Nostr relay URLs.
    #[arg(long, hide = true)]
    pub(crate) nostr_relay: Vec<String>,

    /// Ignored (backward compat).
    #[arg(long, hide = true)]
    pub(crate) no_console: bool,

    /// Optional path to the mesh-llm config file.
    #[arg(long)]
    pub(crate) config: Option<PathBuf>,

    /// Path to the owner keystore used to attest this node.
    #[arg(long)]
    pub(crate) owner_key: Option<PathBuf>,

    /// Bind address for the owner-control listener. Defaults to 127.0.0.1:0 when owner identity is configured.
    #[arg(long, hide = true)]
    pub(crate) control_bind: Option<std::net::SocketAddr>,

    /// Advertised owner-control address encoded into the local-only bootstrap token.
    #[arg(long, hide = true)]
    pub(crate) control_advertise_addr: Option<std::net::SocketAddr>,

    /// Fail startup if owner attestation cannot be loaded or signed.
    #[arg(long)]
    pub(crate) owner_required: bool,

    /// Optional human label attached to this node certificate.
    #[arg(long)]
    pub(crate) node_label: Option<String>,

    /// Override peer ownership trust policy.
    #[arg(long, value_enum)]
    pub(crate) trust_policy: Option<TrustPolicy>,

    /// Add trusted owner IDs on top of the local trust store.
    #[arg(long)]
    pub(crate) trust_owner: Vec<String>,

    /// Internal: set when this node joined via Nostr discovery (not --join).
    #[arg(skip)]
    pub(crate) nostr_discovery: bool,
}

pub(crate) fn validate_discovery_mode_args(cli: &Cli) -> anyhow::Result<()> {
    if cli.mesh_discovery_mode != MeshDiscoveryMode::Mdns {
        return Ok(());
    }

    if !cli.nostr_relay.is_empty() {
        anyhow::bail!("--nostr-relay is only valid with --mesh-discovery-mode nostr");
    }
    if let Some(Command::Discover { relay, .. }) = cli.command.as_ref()
        && !relay.is_empty()
    {
        anyhow::bail!("discover --relay is only valid with --mesh-discovery-mode nostr");
    }

    Ok(())
}

#[derive(Subcommand, Debug)]
pub(crate) enum Command {
    /// Manage model storage, migration, and update checks.
    Models {
        #[command(subcommand)]
        command: models::ModelsCommand,
    },
    /// Download a model from the remote catalog or Hugging Face
    Download {
        /// Model name (e.g. "Qwen2.5-32B-Instruct-Q4_K_M" or just "32b")
        name: Option<String>,
        /// Also download the recommended draft model for speculative decoding
        #[arg(long)]
        draft: bool,
    },
    /// Update mesh-llm to a bundled release and exit.
    Update {
        /// Install this specific release tag or version (e.g. v0.60.0 or 0.60.0-rc.1).
        #[arg(long)]
        version: Option<String>,
        /// Install this release bundle flavor instead of the default installed flavor.
        #[arg(long, value_enum, conflicts_with = "detect_flavor")]
        flavor: Option<crate::system::backend::BinaryFlavor>,
        /// Re-detect the best host backend flavor before selecting the release bundle.
        #[arg(long, conflicts_with = "flavor")]
        detect_flavor: bool,
    },
    /// Inspect local GPUs, stable IDs, and cached bandwidth.
    #[command(alias = "gpu")]
    Gpus {
        /// Print machine-readable JSON output.
        #[arg(long)]
        json: bool,
        #[command(subcommand)]
        command: Option<GpuCommand>,
    },
    /// Inspect and manage local runtime-served models.
    #[command(hide = true)]
    Runtime {
        #[command(subcommand)]
        command: Option<RuntimeCommand>,
    },
    /// Diagnose local mesh, runtime, and split-readiness problems.
    Doctor {
        #[command(subcommand)]
        command: DoctorCommand,
    },
    /// Load a local model into a running mesh-llm instance.
    Load {
        /// Model name/path/url to load
        name: String,
        /// Console/API port of the running mesh-llm instance (default: 3131)
        #[arg(long, default_value = "3131")]
        port: u16,
    },
    /// Unload a local model from a running mesh-llm instance.
    #[command(alias = "drop")]
    Unload {
        /// Model name to unload
        name: String,
        /// Console/API port of the running mesh-llm instance (default: 3131)
        #[arg(long, default_value = "3131")]
        port: u16,
    },
    /// Show local model status on a running mesh-llm instance.
    Status {
        /// Console/API port of the running mesh-llm instance (default: 3131)
        #[arg(long, default_value = "3131")]
        port: u16,
    },
    /// Discover meshes and optionally auto-join one.
    Discover {
        /// Filter by mesh name (case-insensitive exact match)
        #[arg(long)]
        name: Option<String>,
        /// Filter by model name (substring match)
        #[arg(long)]
        model: Option<String>,
        /// Filter by minimum VRAM (GB)
        #[arg(long)]
        min_vram: Option<f64>,
        /// Filter by region
        #[arg(long)]
        region: Option<String>,
        /// Print the invite token of the best match (for piping to --join)
        #[arg(long)]
        auto: bool,
        /// Nostr relay URLs (default: see DEFAULT_RELAYS)
        #[arg(long)]
        relay: Vec<String>,
    },
    /// Rotate all identity keys (node + Nostr).
    #[command(hide = true)]
    RotateKey,
    /// Launch Goose with mesh-llm as the inference provider.
    ///
    /// If no mesh is running on --port, this auto-joins the mesh as a client.
    #[command(name = "goose")]
    Goose {
        /// Model id to use from /v1/models (default: auto = mesh picks best)
        #[arg(long)]
        model: Option<String>,
        /// API port for mesh-llm (default: 9337)
        #[arg(long, default_value = "9337")]
        port: u16,
    },
    /// Launch Claude Code with mesh-llm as the inference provider.
    ///
    /// If no mesh is running on --port, this auto-joins the mesh as a client.
    #[command(name = "claude")]
    Claude {
        /// Model id to use from /v1/models (default: auto = mesh picks best)
        #[arg(long)]
        model: Option<String>,
        /// API port for mesh-llm (default: 9337)
        #[arg(long, default_value = "9337")]
        port: u16,
    },
    /// Launch pi with mesh-llm as the inference provider.
    ///
    /// If no mesh is running on a loopback/localhost target, this auto-joins the mesh as a client.
    /// Writes a mesh provider into ~/.pi/agent/models.json and launches pi unless --write is set.
    #[command(name = "pi")]
    Pi {
        /// Model id to use from /v1/models (default: auto = mesh picks best)
        #[arg(long)]
        model: Option<String>,
        /// mesh-llm host or URL for Pi (default: 127.0.0.1:9337)
        #[arg(long, default_value = "127.0.0.1:9337")]
        host: String,
        /// Write the mesh provider config to Pi's models.json instead of launching.
        #[arg(long)]
        write: bool,
    },
    /// Launch OpenCode with mesh-llm as the inference provider.
    ///
    /// If no mesh is running on a loopback/localhost target, this auto-joins the mesh as a client.
    #[command(name = "opencode")]
    Opencode {
        /// Model id to use from /v1/models (default: auto = mesh picks best)
        #[arg(long)]
        model: Option<String>,
        /// mesh-llm host or URL for OpenCode (default: 127.0.0.1:9337)
        #[arg(long, default_value = "127.0.0.1:9337")]
        host: String,
        /// Write the mesh provider config to opencode's config file instead of launching.
        #[arg(long)]
        write: bool,
    },
    /// Stop running mesh-llm processes.
    Stop,
    /// Plugin management.
    #[command(name = "plugins", alias = "plugin")]
    Plugin {
        #[command(subcommand)]
        command: PluginCommand,
    },
    /// Install agent skills exposed by installed plugins.
    Skills {
        #[command(subcommand)]
        command: SkillCommand,
    },
    /// Benchmark and compare model/runtime strategies.
    #[command(hide = true)]
    Benchmark {
        #[command(subcommand)]
        command: BenchmarkCommand,
    },
    /// Prepare a model for distributed inference by splitting it into
    /// per-layer files on HF compute.
    ///
    /// Submits an HF Job that builds skippy-model-package from source,
    /// splits the model, publishes the layer package, and updates the
    /// meshllm/catalog.
    #[command(name = "model-prepare", hide = true, alias = "model-package")]
    ModelPrepare {
        /// Source HuggingFace model ref (e.g. unsloth/Qwen3-235B-A22B-GGUF:UD-Q4_K_XL).
        source_repo: Option<String>,

        /// Quantization variant (deprecated; prefer source refs like repo:Q4_K_M).
        #[arg(long)]
        quant: Option<String>,

        /// Target repo for the layer package (auto-derived if omitted).
        #[arg(long)]
        target: Option<String>,

        /// Override model ID in the manifest.
        #[arg(long)]
        model_id: Option<String>,

        /// HF Job hardware flavor. Use auto for the default CPU splitter baseline.
        #[arg(long, default_value = "auto")]
        flavor: String,

        /// Requested job timeout; raised automatically by model-size minimums.
        #[arg(long, default_value = "1h")]
        timeout: String,

        /// Branch or tag of mesh-llm to build in the job [default: main].
        #[arg(long, default_value = "main")]
        mesh_llm_ref: String,

        /// Explicitly keep this as a dry run. This is the default unless --confirm is set.
        #[arg(long)]
        dry_run: bool,

        /// Actually submit the HF Job. Without this, the command only prints plan, spec, and max cost.
        #[arg(long)]
        confirm: bool,

        /// Stream job logs after submission until completion.
        #[arg(long)]
        follow: bool,

        /// Emit JSON output.
        #[arg(long)]
        json: bool,

        /// Check status of a previously submitted job.
        #[arg(long)]
        status: Option<String>,

        /// Fetch logs for a previously submitted job.
        #[arg(long)]
        logs: Option<String>,

        /// Cancel a running job.
        #[arg(long)]
        cancel: Option<String>,

        /// List recent model-package jobs.
        #[arg(long)]
        list: bool,

        /// Upload the latest job script to the meshllm bucket (requires org access).
        #[arg(long)]
        update_script: bool,
    },
    /// Manage owner identity and keystore.
    Auth {
        #[command(subcommand)]
        command: AuthCommand,
    },
    /// Run a CLI command contributed by a configured plugin.
    #[command(external_subcommand)]
    ExternalPlugin(Vec<OsString>),
}

#[derive(Subcommand, Debug)]
pub(crate) enum PluginCommand {
    /// Install a native plugin from the catalog or a GitHub repository.
    Install {
        /// Plugin catalog name, GitHub owner/repo, or GitHub URL.
        reference: String,
    },
    /// Update an installed native plugin.
    Update {
        /// Plugin name.
        name: String,
    },
    /// Enable an installed native plugin.
    Enable {
        /// Plugin name.
        name: String,
    },
    /// Disable an installed native plugin.
    Disable {
        /// Plugin name.
        name: String,
    },
    /// Delete an installed native plugin.
    Delete {
        /// Plugin name.
        name: String,
    },
    /// Show installed plugin details.
    Info {
        /// Plugin name.
        name: String,
    },
    /// Search the plugin catalog.
    Search {
        /// Optional search query.
        query: Option<String>,
    },
    /// List installed, auto-registered, and configured plugins.
    List,
}

#[derive(Subcommand, Debug)]
pub(crate) enum SkillCommand {
    /// Install skills exposed by installed plugins into supported agent skill folders.
    Install {
        /// Agent to install for. Repeat to install to several agents.
        #[arg(long, value_enum, conflicts_with = "all")]
        agent: Vec<SkillAgentArg>,
        /// Install to all supported agent locations, even if the agent is not detected.
        #[arg(long)]
        all: bool,
        /// Show what would be installed without writing files.
        #[arg(long)]
        dry_run: bool,
        /// Replace an existing non-mesh-managed skill with the same directory name.
        #[arg(long)]
        force: bool,
    },
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum)]
pub(crate) enum SkillAgentArg {
    Global,
    Goose,
    Pi,
    Codex,
    Opencode,
    Claude,
}

impl From<SkillAgentArg> for mesh_llm_plugin_manager::SkillAgent {
    fn from(value: SkillAgentArg) -> Self {
        match value {
            SkillAgentArg::Global => Self::Global,
            SkillAgentArg::Goose => Self::Goose,
            SkillAgentArg::Pi => Self::Pi,
            SkillAgentArg::Codex => Self::Codex,
            SkillAgentArg::Opencode => Self::Opencode,
            SkillAgentArg::Claude => Self::Claude,
        }
    }
}

#[derive(Subcommand, Debug)]
pub(crate) enum DoctorCommand {
    /// Diagnose split-readiness for a model on a running local mesh node.
    Split {
        /// Model ref/name to diagnose.
        #[arg(long, visible_alias = "model")]
        model_ref: String,
        /// Console/API port of the running mesh-llm instance.
        #[arg(long, default_value = "3131")]
        port: u16,
        /// Print machine-readable JSON.
        #[arg(long)]
        json: bool,
        /// Write split-readiness.json to this directory for sharing with maintainers.
        #[arg(long)]
        output_dir: Option<PathBuf>,
    },
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum RuntimeSurface {
    Serve,
    Client,
}

#[derive(Clone, Debug)]
pub(crate) struct NormalizedRuntimeArgs {
    pub(crate) original: Vec<OsString>,
    pub(crate) normalized: Vec<OsString>,
    pub(crate) explicit_surface: Option<RuntimeSurface>,
}

pub(crate) fn normalize_runtime_surface_args<I, S>(args: I) -> NormalizedRuntimeArgs
where
    I: IntoIterator<Item = S>,
    S: Into<OsString>,
{
    let original: Vec<OsString> = args.into_iter().map(Into::into).collect();
    let mut normalized = original.clone();
    let mut explicit_surface = None;

    // Skip leading global flags to find the pseudo-subcommand position.
    // Recognized value-taking flags: --log-format, --mesh-discovery-mode, --max-vram,
    // --llama-flavor, --device, --tensor-split, --bind-port, --bind-ip, --max-clients,
    // --port, --console, --swarm-capture, --draft-max, --ctx-size.
    // Boolean flags: --help-advanced, --auto, --client, --headless, --publish,
    // --plugin, --auto-update, --no-draft, --split, --no-enumerate-host, --listen-all,
    // --no-console, --owner-required.
    let value_taking_flags = [
        "--log-format",
        "--mesh-discovery-mode",
        "--max-vram",
        "--llama-flavor",
        "--device",
        "--tensor-split",
        "--bind-port",
        "--bind-ip",
        "--max-clients",
        "--port",
        "--console",
        "--swarm-capture",
        "--draft-max",
        "--ctx-size",
        "--model",
        "--gguf",
        "--mmproj",
        "--join",
        "--discover",
        "--mesh-name",
        "--region",
        "--name",
        "--plugin",
        "--draft",
        "--bin-dir",
        "--relay",
        "--relay-auth",
        "--nostr-relay",
        "--config",
        "--owner-key",
        "--control-bind",
        "--control-advertise-addr",
        "--node-label",
        "--trust-policy",
        "--trust-owner",
    ];

    let mut pos = 1;
    while pos < original.len() {
        let arg_str = original.get(pos).and_then(|arg| arg.to_str()).unwrap_or("");

        // Check for --flag=value form
        if let Some(eq_idx) = arg_str.find('=') {
            let flag_part = &arg_str[..eq_idx];
            if value_taking_flags.contains(&flag_part) {
                pos += 1;
                continue;
            }
        }

        // Check for --flag value form
        if value_taking_flags.contains(&arg_str) {
            // Advance by 2 if next token exists and doesn't start with '-'
            if let Some(next) = original.get(pos + 1).and_then(|arg| arg.to_str())
                && !next.starts_with('-')
            {
                pos += 2;
                continue;
            }
            // If next doesn't exist or starts with '-', advance by 1 (let Clap handle the error)
            pos += 1;
            continue;
        }

        // If it starts with '-' but isn't a recognized flag, it's likely a parse error or unknown flag
        if arg_str.starts_with('-') {
            pos += 1;
            continue;
        }

        // Found the first positional argument (serve/client/other subcommand)
        break;
    }

    // Now apply the serve/client normalization logic at the discovered position
    match original.get(pos).and_then(|arg| arg.to_str()) {
        Some("serve") => match original.get(pos + 1).and_then(|arg| arg.to_str()) {
            Some(arg) if arg.starts_with('-') => {
                normalized.remove(pos);
                explicit_surface = Some(RuntimeSurface::Serve);
            }
            None => {
                normalized.remove(pos);
                explicit_surface = Some(RuntimeSurface::Serve);
            }
            _ => {}
        },
        Some("client") => {
            normalized.remove(pos);
            normalized.insert(pos, OsString::from("--client"));
            explicit_surface = Some(RuntimeSurface::Client);
        }
        _ => {}
    }

    NormalizedRuntimeArgs {
        original,
        normalized,
        explicit_surface,
    }
}

pub(crate) fn legacy_runtime_surface_warning(
    cli: &Cli,
    original_args: &[OsString],
    explicit_surface: Option<RuntimeSurface>,
) -> Option<String> {
    if explicit_surface.is_some() || cli.command.is_some() {
        return None;
    }

    if cli.client {
        return Some(format!(
            "⚠️ top-level `--client` now maps to `mesh-llm client`.\n  Please use: {}",
            suggested_client_command(original_args)
        ));
    }

    if !cli.model.is_empty() || !cli.gguf.is_empty() || cli.mmproj.is_some() {
        return Some(format!(
            "⚠️ top-level serving flags now map to `mesh-llm serve`.\n  Please use: {}",
            suggested_serve_command(original_args)
        ));
    }

    None
}

fn suggested_serve_command(original_args: &[OsString]) -> String {
    let mut args = Vec::with_capacity(original_args.len() + 1);
    if let Some(program) = original_args.first() {
        args.push(program.clone());
    } else {
        args.push(OsString::from("mesh-llm"));
    }
    args.push(OsString::from("serve"));
    args.extend(original_args.iter().skip(1).cloned());
    shell_join(&args)
}

fn suggested_client_command(original_args: &[OsString]) -> String {
    let mut args = Vec::with_capacity(original_args.len());
    if let Some(program) = original_args.first() {
        args.push(program.clone());
    } else {
        args.push(OsString::from("mesh-llm"));
    }
    args.push(OsString::from("client"));
    let mut skipped_client = false;
    for arg in original_args.iter().skip(1) {
        if !skipped_client && arg.to_string_lossy() == "--client" {
            skipped_client = true;
            continue;
        }
        args.push(arg.clone());
    }
    shell_join(&args)
}

fn shell_join(args: &[OsString]) -> String {
    args.iter().map(shell_display).collect::<Vec<_>>().join(" ")
}

fn shell_display(arg: &OsString) -> String {
    let text = arg.to_string_lossy();
    if text.is_empty() {
        "\"\"".into()
    } else if text
        .chars()
        .any(|ch| ch.is_whitespace() || matches!(ch, '"' | '\'' | '\\'))
    {
        format!("{text:?}")
    } else {
        text.into_owned()
    }
}

#[cfg(test)]
pub(crate) fn assert_mesh_requirements_docs_examples_parse() {
    let unrestricted_args =
        normalize_runtime_surface_args(["mesh-llm", "serve", "--model", "Qwen3-8B-Q4_K_M"]);
    let unrestricted = Cli::parse_from(unrestricted_args.normalized.clone());
    assert!(unrestricted.command.is_none());
    assert_eq!(unrestricted.model, vec![PathBuf::from("Qwen3-8B-Q4_K_M")]);
    assert!(!unrestricted.publish);

    let signed_public_args = normalize_runtime_surface_args([
        "mesh-llm",
        "serve",
        "--model",
        "Qwen3-8B-Q4_K_M",
        "--publish",
        "--require-release-attestation",
        "--release-signer-key",
        "ed25519:0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
        "--owner-key",
        "~/.mesh-llm/owner-keystore.json",
        "--owner-required",
        "--trust-policy",
        "require-owned",
        "--node-label",
        "lab-a",
    ]);
    let signed_public = Cli::parse_from(signed_public_args.normalized.clone());
    assert!(signed_public.command.is_none());
    assert_eq!(signed_public.model, vec![PathBuf::from("Qwen3-8B-Q4_K_M")]);
    assert!(signed_public.publish);
    assert!(signed_public.require_release_attestation);
    assert_eq!(
        signed_public.release_signer_key,
        vec![
            "ed25519:0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef".to_string()
        ]
    );
    assert_eq!(
        signed_public.owner_key,
        Some(PathBuf::from("~/.mesh-llm/owner-keystore.json"))
    );
    assert!(signed_public.owner_required);
    assert_eq!(signed_public.trust_policy, Some(TrustPolicy::RequireOwned));
    assert_eq!(signed_public.node_label, Some("lab-a".to_string()));

    let signed_bootstrap_args =
        normalize_runtime_surface_args(["mesh-llm", "serve", "--join", "signed-bootstrap-token"]);
    let signed_bootstrap = Cli::parse_from(signed_bootstrap_args.normalized.clone());
    assert!(signed_bootstrap.command.is_none());
    assert_eq!(
        signed_bootstrap.join,
        vec!["signed-bootstrap-token".to_string()]
    );

    let runtime_bootstrap = Cli::parse_from(["mesh-llm", "runtime", "bootstrap", "--port", "3131"]);
    match runtime_bootstrap.command.expect("runtime command expected") {
        Command::Runtime {
            command: Some(RuntimeCommand::Bootstrap { port, json }),
        } => {
            assert_eq!(port, 3131);
            assert!(!json);
        }
        other => panic!("unexpected command: {other:?}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cli::models::{ModelSearchSort, ModelsCommand};
    use clap::{CommandFactory, Parser, error::ErrorKind};

    #[test]
    fn normalize_runtime_surface_args_rewrites_serve_invocation() {
        let normalized = normalize_runtime_surface_args([
            "mesh-llm",
            "serve",
            "--auto",
            "--model",
            "Qwen3-8B-Q4_K_M",
        ]);

        assert_eq!(normalized.explicit_surface, Some(RuntimeSurface::Serve));
        assert_eq!(
            normalized.normalized,
            vec!["mesh-llm", "--auto", "--model", "Qwen3-8B-Q4_K_M"]
                .into_iter()
                .map(OsString::from)
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn normalize_runtime_surface_args_bare_serve_loads_default_config() {
        let normalized = normalize_runtime_surface_args(["mesh-llm", "serve"]);

        assert_eq!(normalized.explicit_surface, Some(RuntimeSurface::Serve));
        assert_eq!(
            normalized.normalized,
            vec!["mesh-llm"]
                .into_iter()
                .map(OsString::from)
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn normalize_runtime_surface_args_rewrites_client_invocation() {
        let normalized =
            normalize_runtime_surface_args(["mesh-llm", "client", "--auto", "--port", "9337"]);

        assert_eq!(normalized.explicit_surface, Some(RuntimeSurface::Client));
        assert_eq!(
            normalized.normalized,
            vec!["mesh-llm", "--client", "--auto", "--port", "9337"]
                .into_iter()
                .map(OsString::from)
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn normalize_runtime_surface_args_treats_relay_auth_as_value_taking_before_serve() {
        // Regression: --relay-auth carries a `URL=TOKEN` value, so the
        // pseudo-subcommand scanner must skip the value and still discover
        // `serve` (or `client`) as the runtime surface. If --relay-auth is not
        // in the value-taking list the scanner stops at the token and Clap
        // sees a malformed command.
        let normalized = normalize_runtime_surface_args([
            "mesh-llm",
            "--relay-auth",
            "https://gated.example/=token",
            "serve",
            "--relay",
            "https://gated.example/",
            "--auto",
        ]);

        assert_eq!(normalized.explicit_surface, Some(RuntimeSurface::Serve));
        assert_eq!(
            normalized.normalized,
            vec![
                "mesh-llm",
                "--relay-auth",
                "https://gated.example/=token",
                "--relay",
                "https://gated.example/",
                "--auto",
            ]
            .into_iter()
            .map(OsString::from)
            .collect::<Vec<_>>()
        );

        // And the resulting argv must actually parse cleanly through Clap so
        // the relay-auth value reaches `Cli::relay_auth`.
        let cli = Cli::try_parse_from(&normalized.normalized).expect("clap parse");
        assert_eq!(
            cli.relay_auth,
            vec![("https://gated.example/".to_string(), "token".to_string())],
        );
    }

    #[test]
    fn normalize_runtime_surface_args_relay_auth_before_client_invocation() {
        // Same regression but for the `client` surface, including a token
        // containing `=` (NIP-98-style base64 padding).
        let normalized = normalize_runtime_surface_args([
            "mesh-llm",
            "--relay-auth",
            "https://gated.example/=eyJhbGciOiJFZERTQSJ9.payload==",
            "client",
            "--auto",
        ]);

        assert_eq!(normalized.explicit_surface, Some(RuntimeSurface::Client));
        let cli = Cli::try_parse_from(&normalized.normalized).expect("clap parse");
        assert!(cli.client, "client surface flag should be set");
        assert_eq!(
            cli.relay_auth,
            vec![(
                "https://gated.example/".to_string(),
                "eyJhbGciOiJFZERTQSJ9.payload==".to_string()
            )],
        );
    }

    #[test]
    fn normalize_runtime_surface_args_keeps_non_runtime_subcommands() {
        let normalized = normalize_runtime_surface_args(["mesh-llm", "download", "foo"]);

        assert_eq!(normalized.explicit_surface, None);
        assert_eq!(
            normalized.normalized,
            vec!["mesh-llm", "download", "foo"]
                .into_iter()
                .map(OsString::from)
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn legacy_runtime_surface_warning_for_top_level_serve_flags() {
        let normalized =
            normalize_runtime_surface_args(["mesh-llm", "--auto", "--model", "Qwen3-8B-Q4_K_M"]);
        let cli = Cli::parse_from(normalized.normalized.clone());

        let warning =
            legacy_runtime_surface_warning(&cli, &normalized.original, normalized.explicit_surface)
                .expect("warning should be present");

        assert!(warning.contains("mesh-llm serve --auto --model Qwen3-8B-Q4_K_M"));
    }

    #[test]
    fn legacy_runtime_surface_warning_for_top_level_client_flag() {
        let normalized = normalize_runtime_surface_args(["mesh-llm", "--auto", "--client"]);
        let cli = Cli::parse_from(normalized.normalized.clone());

        let warning =
            legacy_runtime_surface_warning(&cli, &normalized.original, normalized.explicit_surface)
                .expect("warning should be present");

        assert!(warning.contains("mesh-llm client --auto"));
    }

    #[test]
    fn explicit_runtime_surface_suppresses_legacy_warning() {
        let normalized = normalize_runtime_surface_args(["mesh-llm", "client", "--auto"]);
        let cli = Cli::parse_from(normalized.normalized.clone());

        assert!(
            legacy_runtime_surface_warning(&cli, &normalized.original, normalized.explicit_surface)
                .is_none()
        );
    }

    #[test]
    fn auth_status_accepts_owner_key_locally() {
        let cli = Cli::parse_from(["mesh-llm", "auth", "status", "--owner-key", "owner.json"]);

        match cli.command.expect("auth command expected") {
            Command::Auth {
                command: AuthCommand::Status { owner_key, .. },
            } => {
                assert_eq!(owner_key, Some(PathBuf::from("owner.json")));
            }
            other => panic!("unexpected command: {other:?}"),
        }
    }

    #[test]
    fn mesh_requirements_docs_examples_parse() {
        super::assert_mesh_requirements_docs_examples_parse();
    }

    #[test]
    fn auth_status_rejects_runtime_only_owner_required_flag() {
        let err = Cli::try_parse_from(["mesh-llm", "auth", "status", "--owner-required"])
            .expect_err("runtime-only flag should be rejected for auth status");

        let rendered = err.to_string();
        assert!(rendered.contains("--owner-required"));
    }

    #[test]
    fn gpus_command_parses_without_subcommand() {
        let cli = Cli::parse_from(["mesh-llm", "gpus"]);

        match cli.command.expect("gpu command expected") {
            Command::Gpus { json, command } => {
                assert!(!json);
                assert!(command.is_none());
            }
            other => panic!("unexpected command: {other:?}"),
        }
    }

    #[test]
    fn gpu_alias_parses_without_subcommand() {
        let cli = Cli::parse_from(["mesh-llm", "gpu"]);

        match cli.command.expect("gpu command expected") {
            Command::Gpus { json, command } => {
                assert!(!json);
                assert!(command.is_none());
            }
            other => panic!("unexpected command: {other:?}"),
        }
    }

    #[test]
    fn gpus_command_accepts_json_flag() {
        let cli = Cli::parse_from(["mesh-llm", "gpus", "--json"]);

        match cli.command.expect("gpu command expected") {
            Command::Gpus { json, command } => {
                assert!(json);
                assert!(command.is_none());
            }
            other => panic!("unexpected command: {other:?}"),
        }
    }

    #[test]
    fn gpu_benchmark_subcommand_parses() {
        let cli = Cli::parse_from(["mesh-llm", "gpu", "benchmark"]);

        match cli.command.expect("gpu command expected") {
            Command::Gpus {
                json: false,
                command: Some(GpuCommand::Benchmark { json: false }),
            } => {}
            other => panic!("unexpected command: {other:?}"),
        }
    }

    #[test]
    fn gpu_benchmark_subcommand_accepts_json_flag() {
        let cli = Cli::parse_from(["mesh-llm", "gpu", "benchmark", "--json"]);

        match cli.command.expect("gpu command expected") {
            Command::Gpus {
                json: false,
                command: Some(GpuCommand::Benchmark { json: true }),
            } => {}
            other => panic!("unexpected command: {other:?}"),
        }
    }

    #[test]
    fn cli_accepts_headless_flag_for_serve_surface() {
        let args = vec!["mesh-llm", "serve", "--headless", "--auto"];
        let normalized = normalize_runtime_surface_args(args);
        let cli = Cli::try_parse_from(&normalized.normalized).unwrap();
        assert!(cli.headless);
    }

    #[test]
    fn cli_accepts_headless_flag_for_client_surface() {
        let args = vec!["mesh-llm", "client", "--headless", "--auto"];
        let normalized = normalize_runtime_surface_args(args);
        let cli = Cli::try_parse_from(&normalized.normalized).unwrap();
        assert!(cli.headless);
    }

    #[test]
    fn cli_accepts_swarm_capture_flag_for_client_surface() {
        let args = vec![
            "mesh-llm",
            "client",
            "--swarm-capture",
            "/tmp/mesh-capture",
            "--auto",
        ];
        let normalized = normalize_runtime_surface_args(args);
        let cli = Cli::try_parse_from(&normalized.normalized).unwrap();

        assert!(cli.client);
        assert_eq!(cli.swarm_capture, Some(PathBuf::from("/tmp/mesh-capture")));
    }

    #[test]
    fn cli_accepts_global_swarm_capture_before_client() {
        let normalized = normalize_runtime_surface_args([
            "mesh-llm",
            "--swarm-capture",
            "/tmp/mesh-capture",
            "client",
            "--auto",
        ]);
        let cli = Cli::parse_from(normalized.normalized);

        assert!(cli.client);
        assert_eq!(cli.swarm_capture, Some(PathBuf::from("/tmp/mesh-capture")));
        assert_eq!(normalized.explicit_surface, Some(RuntimeSurface::Client));
    }

    #[test]
    fn legacy_no_console_remains_ignored_in_headless_tests() {
        let args = vec!["mesh-llm", "serve", "--no-console"];
        let normalized = normalize_runtime_surface_args(args);
        let cli = Cli::try_parse_from(&normalized.normalized).unwrap();
        assert!(
            !cli.headless,
            "--no-console must not activate headless mode"
        );
    }

    #[test]
    fn help_text_mentions_headless_keeps_management_api() {
        let help = Cli::command().render_help().to_string();
        assert!(
            help.contains("headless") || help.contains("management API"),
            "help text should mention headless or management API"
        );
    }

    #[test]
    fn opencode_command_accepts_host_flag() {
        let cli = Cli::parse_from([
            "mesh-llm",
            "opencode",
            "--host",
            "https://mesh.example.com:9443",
        ]);

        match cli.command.expect("opencode command expected") {
            Command::Opencode { model, host, write } => {
                assert_eq!(model, None);
                assert_eq!(host, "https://mesh.example.com:9443");
                assert!(!write);
            }
            other => panic!("unexpected command: {other:?}"),
        }
    }

    #[test]
    fn opencode_command_rejects_port_flag() {
        let err = Cli::try_parse_from(["mesh-llm", "opencode", "--port", "9337"])
            .expect_err("opencode should reject --port");

        let rendered = err.to_string();
        assert!(rendered.contains("--port"));
    }

    #[test]
    fn unknown_top_level_command_is_captured_for_plugin_dispatch() {
        let normalized = normalize_runtime_surface_args([
            "mesh-llm",
            "goose-next",
            "--model",
            "auto",
            "--",
            "prompt.txt",
        ]);
        let cli = Cli::parse_from(normalized.normalized);

        match cli.command.expect("external plugin command expected") {
            Command::ExternalPlugin(args) => {
                assert_eq!(
                    args,
                    vec![
                        OsString::from("goose-next"),
                        OsString::from("--model"),
                        OsString::from("auto"),
                        OsString::from("--"),
                        OsString::from("prompt.txt"),
                    ]
                );
            }
            other => panic!("unexpected command: {other:?}"),
        }
    }

    #[test]
    fn cli_defaults_log_format_to_pretty() {
        let normalized = normalize_runtime_surface_args(["mesh-llm", "serve", "--auto"]);
        let cli = Cli::parse_from(normalized.normalized);

        assert_eq!(cli.log_format, LogFormat::Pretty);
    }

    #[test]
    fn skills_install_accepts_global_agent_target() {
        let cli = Cli::parse_from(["mesh-llm", "skills", "install", "--agent", "global"]);

        match cli.command.expect("skills command expected") {
            Command::Skills {
                command:
                    SkillCommand::Install {
                        agent, all: false, ..
                    },
            } => {
                assert_eq!(agent, vec![SkillAgentArg::Global]);
            }
            other => panic!("unexpected command: {other:?}"),
        }
    }

    #[test]
    fn cli_accepts_json_log_format() {
        let normalized =
            normalize_runtime_surface_args(["mesh-llm", "serve", "--log-format", "json", "--auto"]);
        let cli = Cli::parse_from(normalized.normalized);

        assert_eq!(cli.log_format, LogFormat::Json);
    }

    #[test]
    fn cli_accepts_global_log_format_before_serve() {
        let normalized =
            normalize_runtime_surface_args(["mesh-llm", "--log-format", "json", "serve", "--auto"]);
        let cli = Cli::parse_from(normalized.normalized);

        assert_eq!(cli.log_format, LogFormat::Json);
        assert_eq!(normalized.explicit_surface, Some(RuntimeSurface::Serve));
    }

    #[test]
    fn cli_accepts_global_log_format_before_serve_with_model() {
        let normalized = normalize_runtime_surface_args([
            "mesh-llm",
            "--log-format",
            "json",
            "serve",
            "--model",
            "Qwen3-8B-Q4_K_M",
        ]);
        let cli = Cli::parse_from(normalized.normalized);

        assert_eq!(cli.log_format, LogFormat::Json);
        assert_eq!(cli.model, vec![std::path::PathBuf::from("Qwen3-8B-Q4_K_M")]);
        assert_eq!(normalized.explicit_surface, Some(RuntimeSurface::Serve));
    }

    #[test]
    fn cli_accepts_global_log_format_equals_before_serve() {
        let normalized =
            normalize_runtime_surface_args(["mesh-llm", "--log-format=json", "serve", "--auto"]);
        let cli = Cli::parse_from(normalized.normalized);

        assert_eq!(cli.log_format, LogFormat::Json);
        assert_eq!(normalized.explicit_surface, Some(RuntimeSurface::Serve));
    }

    #[test]
    fn cli_accepts_global_log_format_before_client() {
        let normalized = normalize_runtime_surface_args([
            "mesh-llm",
            "--log-format",
            "json",
            "client",
            "--auto",
        ]);
        let cli = Cli::parse_from(normalized.normalized);

        assert_eq!(cli.log_format, LogFormat::Json);
        assert_eq!(normalized.explicit_surface, Some(RuntimeSurface::Client));
    }

    #[test]
    fn cli_accepts_global_bind_ip_before_serve() {
        let normalized = normalize_runtime_surface_args([
            "mesh-llm",
            "--bind-ip",
            "10.1.2.3",
            "serve",
            "--bind-port",
            "47916",
        ]);
        let cli = Cli::parse_from(normalized.normalized);

        assert_eq!(cli.bind_ip, Some("10.1.2.3".parse().unwrap()));
        assert_eq!(cli.bind_port, Some(47916));
        assert_eq!(normalized.explicit_surface, Some(RuntimeSurface::Serve));
    }

    #[test]
    fn cli_accepts_global_mesh_discovery_mode_before_serve() {
        let normalized = normalize_runtime_surface_args([
            "mesh-llm",
            "--mesh-discovery-mode",
            "mdns",
            "serve",
            "--auto",
        ]);
        let cli = Cli::parse_from(normalized.normalized);

        assert_eq!(
            cli.mesh_discovery_mode,
            crate::network::discovery::MeshDiscoveryMode::Mdns
        );
        assert_eq!(normalized.explicit_surface, Some(RuntimeSurface::Serve));
    }

    #[test]
    fn cli_defaults_mesh_discovery_mode_to_nostr() {
        let normalized = normalize_runtime_surface_args(["mesh-llm", "serve", "--auto"]);
        let cli = Cli::parse_from(normalized.normalized);

        assert_eq!(
            cli.mesh_discovery_mode,
            crate::network::discovery::MeshDiscoveryMode::Nostr
        );
    }

    #[test]
    fn cli_accepts_mdns_discovery_mode_for_runtime_surfaces() {
        let normalized =
            normalize_runtime_surface_args(["mesh-llm", "client", "--mesh-discovery-mode", "mdns"]);
        let cli = Cli::parse_from(normalized.normalized);

        assert!(cli.client);
        assert_eq!(
            cli.mesh_discovery_mode,
            crate::network::discovery::MeshDiscoveryMode::Mdns
        );
    }

    #[test]
    fn cli_rejects_nostr_relays_in_mdns_mode() {
        let normalized = normalize_runtime_surface_args([
            "mesh-llm",
            "serve",
            "--mesh-discovery-mode",
            "mdns",
            "--nostr-relay",
            "wss://relay.example",
        ]);
        let cli = Cli::parse_from(normalized.normalized);

        let err = validate_discovery_mode_args(&cli)
            .expect_err("mdns mode must reject Nostr relay overrides");
        assert!(err.to_string().contains("--nostr-relay"));
    }

    #[test]
    fn cli_rejects_invalid_log_format_values() {
        let err = Cli::try_parse_from(["mesh-llm", "--log-format", "invalid"])
            .expect_err("invalid log format should be rejected");

        assert_eq!(err.kind(), ErrorKind::InvalidValue);
        let rendered = err.to_string();
        assert!(rendered.contains("--log-format <LOG_FORMAT>"));
        assert!(rendered.contains("pretty"));
        assert!(rendered.contains("json"));
    }

    #[test]
    fn cli_help_documents_log_format_flag() {
        let mut command = Cli::command();
        let help = command.render_long_help().to_string();

        assert!(help.contains("--log-format <LOG_FORMAT>"));
        assert!(help.contains("Terminal output format for app-owned runtime events"));
        assert!(help.contains("[default: pretty]"));
        assert!(help.contains("[possible values: pretty, json]"));
    }

    #[test]
    fn cli_log_format_selection_is_independent_across_runs() {
        let pretty = Cli::parse_from(["mesh-llm", "--log-format", "pretty"]);
        assert_eq!(pretty.log_format, LogFormat::Pretty);

        let json = Cli::parse_from(["mesh-llm", "--log-format", "json"]);
        assert_eq!(json.log_format, LogFormat::Json);

        let pretty_again = Cli::parse_from(["mesh-llm", "--log-format", "pretty"]);
        assert_eq!(pretty_again.log_format, LogFormat::Pretty);

        let json_again = Cli::parse_from(["mesh-llm", "--log-format", "json"]);
        assert_eq!(json_again.log_format, LogFormat::Json);
    }

    #[test]
    fn models_search_accepts_canonical_parameter_sort_names() {
        let cli = Cli::parse_from([
            "mesh-llm",
            "models",
            "search",
            "qwen",
            "--sort",
            "parameters-desc",
        ]);

        match cli.command.expect("models command expected") {
            Command::Models {
                command:
                    ModelsCommand::Search {
                        sort: ModelSearchSort::ParametersDesc,
                        ..
                    },
            } => {}
            other => panic!("unexpected command: {other:?}"),
        }
    }

    #[test]
    fn models_search_keeps_legacy_parameter_sort_aliases_parsing() {
        let cli = Cli::parse_from([
            "mesh-llm",
            "models",
            "search",
            "qwen",
            "--sort",
            "most-parameters",
        ]);

        match cli.command.expect("models command expected") {
            Command::Models {
                command:
                    ModelsCommand::Search {
                        sort: ModelSearchSort::ParametersDesc,
                        ..
                    },
            } => {}
            other => panic!("unexpected command: {other:?}"),
        }
    }

    #[test]
    fn models_certify_parses_package_gate_options() {
        let cli = Cli::parse_from([
            "mesh-llm",
            "models",
            "certify",
            "hf://meshllm/demo-layers@abc123",
            "--package-only",
            "--report-out",
            "/tmp/cert.json",
            "--json",
            "--prompt",
            "Say ok.",
            "--max-tokens",
            "2",
        ]);

        match cli.command.expect("models command expected") {
            Command::Models {
                command:
                    ModelsCommand::Certify {
                        model,
                        package_only: true,
                        json: true,
                        report_out: Some(report_out),
                        prompt,
                        max_tokens: 2,
                        ..
                    },
            } => {
                assert_eq!(model, "hf://meshllm/demo-layers@abc123");
                assert_eq!(report_out, std::path::PathBuf::from("/tmp/cert.json"));
                assert_eq!(prompt, "Say ok.");
            }
            other => panic!("unexpected command: {other:?}"),
        }
    }
}
