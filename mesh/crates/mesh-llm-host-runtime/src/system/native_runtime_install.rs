pub use mesh_llm_system::native_runtime_install::*;

/// The host's runtime profile, detected once per process. GPU detection takes
/// most of a second on some hosts (#2528), and the startup runtime load and
/// the unused-GPU check both need it; the hardware doesn't change while the
/// node runs.
pub fn host_runtime_profile() -> skippy_native_runtime::HostRuntimeProfile {
    static PROFILE: std::sync::OnceLock<skippy_native_runtime::HostRuntimeProfile> =
        std::sync::OnceLock::new();
    PROFILE
        .get_or_init(mesh_llm_system::native_runtime_install::host_runtime_profile)
        .clone()
}
