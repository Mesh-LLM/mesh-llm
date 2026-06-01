#![forbid(unsafe_code)]

pub mod benchmark;
pub mod models;
pub mod pager;
pub mod parser;
pub mod runtime;
pub mod shell;
pub mod terminal_progress;

pub use mesh_llm_tui::LogFormat;

pub use parser::{
    AuthCommand, BinaryFlavor, Cli, Command, DiscoveryScope, DoctorCommand, GpuCommand,
    MeshDiscoveryMode, MeshGuardrailCliMode, NormalizedRuntimeArgs, PluginCommand, RuntimeSurface,
    SkillAgentArg, SkillCommand, TrustCommand, TrustPolicy, legacy_runtime_surface_warning,
    normalize_runtime_surface_args, validate_discovery_mode_args,
};
