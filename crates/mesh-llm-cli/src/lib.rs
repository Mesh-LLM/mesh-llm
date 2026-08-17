#![forbid(unsafe_code)]

pub mod benchmark;
pub mod models;
pub mod pager;
pub mod parser;
pub mod runtime;
pub mod shell;

pub use mesh_llm_events::LogFormat;

pub use parser::{
    AuthCommand, BinaryFlavor, Cli, Command, ConfigCommand, DiscoveryScope, DoctorCommand,
    GpuCommand, MeshDiscoveryMode, MeshGuardrailCliMode, NormalizedRuntimeArgs, PluginCommand,
    RuntimeSurface, SkillAgentArg, SkillCommand, TrustCommand, TrustPolicy,
    legacy_runtime_surface_warning, normalize_runtime_surface_args, validate_discovery_mode_args,
};
