#![forbid(unsafe_code)]

pub mod agent_config;
pub mod benchmark;
pub mod inventory;
pub mod memberships;
pub mod models;
pub mod pager;
pub mod parser;
pub mod runtime;
pub mod shell;

pub use mesh_llm_events::LogFormat;

pub use inventory::{
    CLI_INVENTORY_SCHEMA_VERSION, CliInventory, InventoryNode, InventoryNodeKind,
    build_cli_inventory, check_cli_inventory, cli_inventory_json, serialize_cli_inventory,
    write_cli_inventory,
};

pub use parser::{
    AnalyticsCommand, AuthCommand, BinaryFlavor, Cli, Command, ConfigCommand, DiscoveryScope,
    DoctorCommand, GpuCommand, JoiningOptions, KvCacheCommand, KvCacheOptions, LoggingOptions,
    MeshDiscoveryMode, MeshGuardrailCliMode, MeshPolicyOptions, NetworkOptions, NodeModeOptions,
    NormalizedRuntimeArgs, OwnerOptions, PlacementOptions, PluginCommand, RuntimeSurface,
    ServingOptions, ServingStrategyCli, SkillAgentArg, SkillCommand, SpeculativeOptions,
    TrustCommand, TrustPolicy, legacy_runtime_surface_warning, normalize_runtime_surface_args,
    raw_args_invoke_analytics, validate_discovery_mode_args,
};

pub mod wallet;
