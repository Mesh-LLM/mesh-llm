//! Groups of the top-level `mesh-llm` options, flattened into [`Cli`](super::Cli).
//!
//! clap's derive code defines every option of one struct in one function, and
//! in a debug build that function's frame grows with each option. `Cli`'s frame
//! used to stay on the stack while the subcommands were defined, which nearly
//! filled a 2 MiB thread (clap-rs/clap#5134). Each group gets its own
//! function, and none of them is on the stack while the subcommands are
//! defined. The groups keep the options' order, so `--help` doesn't change.

use clap::Args;
use std::net::IpAddr;
use std::path::PathBuf;

use super::commands::{
    BinaryFlavor, MeshDiscoveryMode, ServingStrategyCli, SpeculativeNgramFallbackCli,
    SpeculativeNgramProposerCli, TrustPolicy, parse_relay_auth_pair,
};
use mesh_llm_events::LogFormat;
use mesh_llm_events::audit::{AuditLevel, AuditLogFormat};

/// Terminal, audit and debug logging.
#[derive(Args, Debug)]
pub struct LoggingOptions {
    /// Terminal output format for app-owned runtime events.
    #[arg(long, value_enum, default_value_t = LogFormat::Pretty)]
    pub log_format: LogFormat,

    /// Audit log file path (enables audit logging when set).
    #[arg(long, hide = true)]
    pub audit_log_path: Option<PathBuf>,

    /// Audit log format.
    #[arg(long, value_enum, default_value_t = AuditLogFormat::JsonLines, hide = true)]
    pub audit_log_format: AuditLogFormat,

    /// Minimum audit log level.
    #[arg(long, value_enum, default_value_t = AuditLevel::Info, hide = true)]
    pub audit_log_level: AuditLevel,

    /// Enable mesh runtime debug output; set MESH_LLM_DEBUG_NATIVE_VERBOSE=1 for verbose llama.cpp native logs.
    #[arg(long)]
    pub debug: bool,

    /// OTLP/gRPC endpoint for embedded Skippy debug telemetry, for example http://127.0.0.1:14317.
    #[arg(long, hide = true)]
    pub skippy_metrics_otlp_grpc: Option<String>,
}

/// How this node finds and joins a mesh.
#[derive(Args, Debug)]
pub struct JoiningOptions {
    /// Join a mesh via invite token (can repeat).
    ///
    /// The value is a credential: the legacy-surface warning must never
    /// echo it. `parser::suggestion_redaction` redacts every spelling, and
    /// its staleness guards turn red on a rename or a new alias.
    #[arg(long, short)]
    pub join: Vec<String>,

    /// Read an invite token from a file (can repeat).
    ///
    /// The file is re-read on every rejoin attempt, so a rotated token is
    /// picked up without restarting a service.
    #[arg(long, value_name = "PATH")]
    pub join_file: Vec<PathBuf>,

    /// Discover a mesh and join it.
    // require_equals: without it, `--discover status` swallows the `status`
    // subcommand as the mesh name (optional values grab the next token).
    #[arg(long, default_missing_value = "", num_args = 0..=1, require_equals = true)]
    pub discover: Option<String>,

    /// Auto-join the best mesh found via discovery.
    #[arg(long)]
    pub auto: bool,

    /// Discovery provider for --auto, --discover, --publish, and the discover command.
    #[arg(long, value_enum, default_value_t = MeshDiscoveryMode::Nostr, global = true)]
    pub mesh_discovery_mode: MeshDiscoveryMode,
}

/// The models this node serves and its local API.
#[derive(Args, Debug)]
pub struct ServingOptions {
    /// Model to serve (path, remote catalog name, or Hugging Face ref).
    #[arg(long)]
    pub model: Vec<PathBuf>,

    /// Raw local GGUF file to serve directly (repeatable).
    #[arg(long)]
    pub gguf: Vec<PathBuf>,

    /// Explicit mmproj sidecar for the primary served model.
    #[arg(long, hide = true)]
    pub mmproj: Option<PathBuf>,

    /// Quantize SafeTensors weights while loading.
    #[arg(
        long = "quant",
        visible_alias = "checkpoint-quantization",
        long_help = "Quantize SafeTensors weights while loading. The default is preserve.\n\nValid recipes: preserve, F32, F16, BF16, Q1_0, Q2_0, Q4_0, Q4_1, Q5_0, Q5_1, IQ2_XXS, IQ2_XS, IQ2_S, IQ2_M, IQ1_S, IQ1_M, TQ1_0, TQ2_0, Q2_K, Q2_K_S, IQ3_XS, IQ3_XXS, IQ3_S, IQ3_M, Q3_K_S, Q3_K_M, Q3_K_L, IQ4_NL, IQ4_XS, Q4_K_S, Q4_K_M, Q5_K_S, Q5_K_M, Q6_K, Q8_0, MXFP4_MOE. Some low-bit recipes require --checkpoint-imatrix."
    )]
    pub checkpoint_quantization: Option<String>,

    /// Importance matrix for load-time checkpoint quantization.
    #[arg(long, hide = true)]
    pub checkpoint_imatrix: Option<PathBuf>,

    /// API port (default: 9337).
    #[arg(long, default_value = "9337")]
    pub port: u16,

    /// Serve one model directly through the local OpenAI API without starting a mesh node.
    #[arg(long)]
    pub local_model_only: bool,

    /// Load a native serving plugin into the dedicated local-model path.
    #[arg(
        long,
        hide = true,
        requires = "local_model_only",
        requires_all = [
            "native_serving_plugin_config",
            "native_serving_plugin_state",
            "native_serving_plugin_deadline_ms"
        ]
    )]
    pub native_serving_plugin: Option<PathBuf>,

    /// Absolute path to the native serving plugin configuration.
    #[arg(long, hide = true, requires = "native_serving_plugin")]
    pub native_serving_plugin_config: Option<PathBuf>,

    /// Absolute path to the native serving plugin state directory.
    #[arg(long, hide = true, requires = "native_serving_plugin")]
    pub native_serving_plugin_state: Option<PathBuf>,

    /// Mesh-enforced hard proposal deadline in milliseconds.
    #[arg(
        long,
        hide = true,
        requires = "native_serving_plugin",
        value_parser = clap::value_parser!(u64).range(1..)
    )]
    pub native_serving_plugin_deadline_ms: Option<u64>,
}

/// Client mode, the web console and local debug capture.
#[derive(Args, Debug)]
pub struct NodeModeOptions {
    /// Run as a client — no GPU, no model needed.
    #[arg(long)]
    pub client: bool,

    /// Web console port (default: 3131).
    #[arg(long, default_value = "3131")]
    pub console: u16,

    /// Disable the embedded web UI but keep the management API on the --console port.
    #[arg(long)]
    pub headless: bool,

    /// Write passive swarm debug capture JSONL to this local directory (opt-in, no telemetry egress).
    #[arg(long)]
    pub swarm_capture: Option<PathBuf>,
}

/// Publication and creation-time policy of a new mesh.
#[derive(Args, Debug)]
pub struct MeshPolicyOptions {
    /// Publish this mesh for discovery by other nodes.
    /// Without this flag, your mesh is private and only joinable via invite token.
    #[arg(long)]
    pub publish: bool,

    /// Human-readable name for this mesh (shown in discovery when combined with --publish).
    /// Naming a mesh does NOT make it publicly discoverable — use --publish for that.
    #[arg(long)]
    pub mesh_name: Option<String>,

    /// Region tag, e.g. "US", "EU", "AU" (shown in discovery).
    #[arg(long)]
    pub region: Option<String>,

    /// Minimum mesh-llm node version required when creating a new mesh.
    #[arg(long)]
    pub min_node_version: Option<String>,

    /// Maximum mesh-llm node version allowed when creating a new mesh.
    #[arg(long)]
    pub max_node_version: Option<String>,

    /// Minimum protocol generation required when creating a new mesh.
    #[arg(long)]
    pub min_protocol_version: Option<u32>,

    /// Maximum protocol generation allowed when creating a new mesh.
    #[arg(long)]
    pub max_protocol_version: Option<u32>,

    /// Require release attestation when creating a new mesh.
    #[arg(long)]
    pub require_release_attestation: bool,

    /// Require origin-signed relayed gossip in a new mesh (upgraded peers only).
    #[arg(long)]
    pub require_signed_gossip: bool,

    /// Allowed release signer key for mesh creation-time attestation policy (repeatable).
    #[arg(long = "release-signer-key")]
    pub release_signer_key: Vec<String>,
}

/// Speculative decoding and draft models (hidden from default --help).
#[derive(Args, Debug)]
pub struct SpeculativeOptions {
    /// Override speculative decoding (`mtp`, `ngram-cache`, `ngram-suffix`, `dflash`, or a package strategy id).
    #[arg(long, hide = true)]
    pub speculative_strategy: Option<String>,

    /// Minimum matching N-gram length for the request-local MTP cache extension.
    #[arg(long, hide = true)]
    pub speculative_ngram_min: Option<u32>,

    /// Maximum matching N-gram length for the request-local MTP cache extension.
    #[arg(long, hide = true)]
    pub speculative_ngram_max: Option<u32>,

    /// Cap N-gram tokens proposed in one verify window.
    #[arg(long, hide = true)]
    pub speculative_ngram_max_proposal_tokens: Option<u32>,

    /// Standalone N-gram proposer kind (`cache` or `suffix`).
    #[arg(long, hide = true, value_enum)]
    pub speculative_ngram_proposer: Option<SpeculativeNgramProposerCli>,

    /// Maximum N-gram extension length for a composite MTP strategy.
    #[arg(long, hide = true)]
    pub speculative_extension_max_tokens: Option<u32>,

    /// Native MTP rejection cooldown in generated tokens.
    #[arg(long, hide = true)]
    pub speculative_native_mtp_reject_cooldown_tokens: Option<u32>,

    /// Suppress native MTP drafts while its rejection cooldown is active.
    #[arg(long, hide = true)]
    pub speculative_native_mtp_suppress_cooldown_drafts: bool,

    /// Keep native MTP drafts during its rejection cooldown.
    #[arg(
        long,
        hide = true,
        conflicts_with = "speculative_native_mtp_suppress_cooldown_drafts"
    )]
    pub speculative_native_mtp_allow_cooldown_drafts: bool,

    /// Maximum native MTP drafts suppressed by a cooldown.
    #[arg(long, hide = true)]
    pub speculative_native_mtp_suppress_cooldown_draft_limit: Option<u32>,

    /// Minimum tokens to include in a pipelined verify window.
    #[arg(long, hide = true)]
    pub speculative_verify_window_min_tokens: Option<u32>,

    /// Maximum tokens to include in a pipelined verify window.
    #[arg(long, hide = true)]
    pub speculative_verify_window_max_tokens: Option<u32>,

    /// Number of in-flight pipelined verify windows.
    #[arg(long, hide = true)]
    pub speculative_verify_window_pipeline_depth: Option<u32>,

    /// Admit pipelined verify windows by speculative-token budget instead of a
    /// fixed depth. 0 keeps fixed-depth admission.
    #[arg(long, hide = true)]
    pub speculative_verify_window_runahead_tokens: Option<u32>,

    /// Propose from the draft model when the N-gram proposer misses
    /// (`draft` or `none`). Requires a draft model and pipeline depth > 1.
    #[arg(long, hide = true, value_enum)]
    pub speculative_ngram_fallback: Option<SpeculativeNgramFallbackCli>,

    /// Draft model for speculative decoding.
    #[arg(long, hide = true)]
    pub draft: Option<PathBuf>,

    /// Max draft tokens (default: 8).
    #[arg(long, default_value = "8", hide = true)]
    pub draft_max: u16,

    /// Disable automatic draft model detection.
    #[arg(long, hide = true)]
    pub no_draft: bool,
}

/// Serving strategy, split placement and local hardware selection.
#[derive(Args, Debug)]
pub struct PlacementOptions {
    /// What to optimise serving for: `balanced` (default), `interactive`, or
    /// `throughput`. Composes the settings each one needs; any explicit flag or
    /// config value you set still wins.
    #[arg(long, value_enum)]
    pub strategy: Option<ServingStrategyCli>,

    /// Force tensor split even if the model fits on one node.
    #[arg(long, hide = true)]
    pub split: bool,

    /// Experimentally allow split serving for an architecture not certified by the llama canary.
    #[arg(long, requires = "split", hide = true)]
    pub allow_uncertified_split: bool,

    /// Pin split-serving node order and layer ranges from a JSON topology lock.
    #[arg(long, value_name = "PATH", requires = "split", hide = true)]
    pub split_topology_lock: Option<PathBuf>,

    /// Place split layers by node speed and rebalance them while serving, so a
    /// slower node does not hold back faster ones. Split-only: requires --split.
    #[arg(long, requires = "split")]
    pub auto_balance: bool,

    /// Override context size (tokens). Default: auto-scaled to available VRAM.
    #[arg(long, hide = true)]
    pub ctx_size: Option<u32>,

    /// Parallel lanes (concurrent sequences) for served models. Overrides `[gpu].parallel`
    /// from the config file. Default: planned, currently 4.
    #[arg(long, hide = true)]
    pub parallel: Option<std::num::NonZeroUsize>,

    /// Cap VRAM used for planning, local-fit decisions, and mesh advertisement (GB).
    #[arg(long)]
    pub max_vram: Option<f64>,

    /// Disable broadcasting GPU name, hostname, VRAM, and reserved bytes to peers. By default all nodes announce this hardware info.
    #[arg(long = "no-enumerate-host", hide = true)]
    pub no_enumerate_host: bool,

    /// Path to bundled mesh support binaries.
    #[arg(long, hide = true)]
    pub bin_dir: Option<PathBuf>,

    /// Override which bundled llama.cpp flavor to use.
    #[arg(long, value_enum)]
    pub llama_flavor: Option<BinaryFlavor>,

    /// Device override for local backend selection.
    #[arg(long, hide = true)]
    pub device: Option<String>,

    /// Deprecated tensor split override retained for CLI compatibility.
    #[arg(long, hide = true)]
    pub tensor_split: Option<String>,
}

/// Relays, socket binding and connection limits.
#[derive(Args, Debug)]
pub struct NetworkOptions {
    /// Override iroh relay URLs.
    #[arg(long, hide = true)]
    pub relay: Vec<String>,

    /// Per-relay bearer token for gated iroh relays, formatted as
    /// `URL=TOKEN`. Repeatable. The token is sent as
    /// `Authorization: Bearer <TOKEN>` on the WebSocket upgrade to the
    /// matching `--relay` URL. Relays not listed here register without
    /// authentication (the correct behavior for public relays).
    ///
    /// Splits on the first `=` only, so tokens may contain `=` (base64
    /// padding, JWTs, etc.).
    ///
    /// The value is a credential: the legacy-surface warning must never
    /// echo the token portion. `parser::suggestion_redaction` keeps the
    /// URL and redacts the token in every spelling, and its staleness
    /// guards turn red on a rename or a new alias.
    #[arg(long = "relay-auth", value_parser = parse_relay_auth_pair, hide = true)]
    pub relay_auth: Vec<(String, String)>,

    /// Disable iroh relays even when public mesh discovery would normally use them.
    #[arg(long = "disable-iroh-relays", hide = true)]
    pub disable_iroh_relays: bool,

    /// Bind QUIC to a fixed UDP port (for NAT port forwarding).
    #[arg(long, hide = true)]
    pub bind_port: Option<u16>,

    /// Bind mesh QUIC to a specific local IP address.
    #[arg(long, hide = true)]
    pub bind_ip: Option<IpAddr>,

    /// Bind to 0.0.0.0 (for containers/Fly.io).
    #[arg(long, hide = true)]
    pub listen_all: bool,

    /// Stop advertising when N clients connected.
    #[arg(long, hide = true)]
    pub max_clients: Option<usize>,

    /// Custom Nostr relay URLs.
    #[arg(long, hide = true)]
    pub nostr_relay: Vec<String>,

    /// Ignored (backward compat).
    #[arg(long, hide = true)]
    pub no_console: bool,
}

/// The node-local disk prompt cache.
#[derive(Args, Debug)]
pub struct KvCacheOptions {
    /// Node-local disk prompt cache: off, auto, or an explicit IEC size such as 32GiB.
    #[arg(long, value_name = "off|auto|SIZE")]
    pub kv_cache_disk: Option<String>,

    /// Absolute node-local disk prompt-cache directory.
    #[arg(long, value_name = "ABSOLUTE_PATH")]
    pub kv_cache_disk_dir: Option<PathBuf>,

    /// Minimum free storage to preserve, with an IEC suffix such as 16GiB.
    #[arg(long, value_name = "SIZE")]
    pub kv_cache_min_free: Option<String>,
}

/// Owner attestation, the owner-control listener and peer trust.
#[derive(Args, Debug)]
pub struct OwnerOptions {
    /// Path to the owner keystore used to attest this node.
    #[arg(long)]
    pub owner_key: Option<PathBuf>,

    /// Bind address for the owner-control listener. Defaults to 127.0.0.1:0 when owner identity is configured.
    #[arg(long, hide = true)]
    pub control_bind: Option<std::net::SocketAddr>,

    /// Advertised owner-control address encoded into the local-only bootstrap token.
    #[arg(long, hide = true)]
    pub control_advertise_addr: Option<std::net::SocketAddr>,

    /// Fail startup if owner attestation cannot be loaded or signed.
    #[arg(long)]
    pub owner_required: bool,

    /// Optional human label attached to this node certificate.
    #[arg(long)]
    pub node_label: Option<String>,

    /// Override peer ownership trust policy.
    #[arg(long, value_enum)]
    pub trust_policy: Option<TrustPolicy>,

    /// Add trusted owner IDs on top of the local trust store.
    #[arg(long)]
    pub trust_owner: Vec<String>,
}
