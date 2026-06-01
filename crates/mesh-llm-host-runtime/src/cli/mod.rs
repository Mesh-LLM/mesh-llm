pub(crate) mod commands;
pub mod output;
pub(crate) mod pager;
pub(crate) mod shell;
pub(crate) mod terminal_progress;

pub mod models {
    pub use mesh_llm_cli::models::*;
}

pub mod runtime {
    pub use mesh_llm_cli::runtime::*;
}

pub use mesh_llm_cli::{
    BinaryFlavor, Cli, Command, DoctorCommand, LogFormat, MeshGuardrailCliMode,
    NormalizedRuntimeArgs, PluginCommand, RuntimeSurface, SkillAgentArg, SkillCommand, TrustPolicy,
    legacy_runtime_surface_warning, normalize_runtime_surface_args, validate_discovery_mode_args,
};

pub(crate) fn trust_policy_to_crypto(value: TrustPolicy) -> crate::crypto::TrustPolicy {
    match value {
        TrustPolicy::Off => crate::crypto::TrustPolicy::Off,
        TrustPolicy::PreferOwned => crate::crypto::TrustPolicy::PreferOwned,
        TrustPolicy::RequireOwned => crate::crypto::TrustPolicy::RequireOwned,
        TrustPolicy::Allowlist => crate::crypto::TrustPolicy::Allowlist,
    }
}

pub(crate) fn binary_flavor_to_backend(
    flavor: Option<BinaryFlavor>,
) -> Option<crate::system::backend::BinaryFlavor> {
    flavor.map(|flavor| match flavor {
        BinaryFlavor::Cpu => crate::system::backend::BinaryFlavor::Cpu,
        BinaryFlavor::Cuda => crate::system::backend::BinaryFlavor::Cuda,
        BinaryFlavor::Rocm => crate::system::backend::BinaryFlavor::Rocm,
        BinaryFlavor::Vulkan => crate::system::backend::BinaryFlavor::Vulkan,
        BinaryFlavor::Metal => crate::system::backend::BinaryFlavor::Metal,
    })
}

pub(crate) fn skill_agent_arg_to_manager(
    agent: SkillAgentArg,
) -> mesh_llm_plugin_manager::SkillAgent {
    match agent {
        SkillAgentArg::Global => mesh_llm_plugin_manager::SkillAgent::Global,
        SkillAgentArg::Goose => mesh_llm_plugin_manager::SkillAgent::Goose,
        SkillAgentArg::Pi => mesh_llm_plugin_manager::SkillAgent::Pi,
        SkillAgentArg::Codex => mesh_llm_plugin_manager::SkillAgent::Codex,
        SkillAgentArg::Opencode => mesh_llm_plugin_manager::SkillAgent::Opencode,
        SkillAgentArg::Claude => mesh_llm_plugin_manager::SkillAgent::Claude,
    }
}

pub(crate) fn mesh_guardrail_mode_to_openai(
    mode: MeshGuardrailCliMode,
) -> openai_frontend::GuardrailMode {
    match mode {
        MeshGuardrailCliMode::Disabled => openai_frontend::GuardrailMode::Disabled,
        MeshGuardrailCliMode::Metrics => openai_frontend::GuardrailMode::MetricsOnly,
        MeshGuardrailCliMode::Enforce => openai_frontend::GuardrailMode::Enforce,
    }
}
