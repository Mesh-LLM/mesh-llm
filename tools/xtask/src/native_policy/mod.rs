//! `native`: the Rust owners of the native runtime selection and host
//! dependency policy scripts. `select-runtime` replaces
//! `scripts/select-native-runtime.py`, `verify-host-dependencies`
//! replaces `scripts/verify-host-dependencies.py`, `linux-runtime-deps`
//! replaces `scripts/linux-native-runtime-deps.py`, and
//! `windows-runtime-deps` replaces `scripts/windows-native-runtime-deps.py`,
//! keeping their argv, streams and exit statuses. Inspection tools run only through the
//! [`toolchain::Toolchain`] adapter.

mod argv;
mod forbidden;
mod glibc;
mod host_dependencies;
mod imports;
mod linux_deps;
mod linux_deps_collect;
mod linux_deps_elf;
mod linux_deps_order;
mod linux_deps_policy;
pub(crate) mod manifest_identity;
mod manifest_json;
mod package_version;
mod package_version_command;
mod release_matrix;
mod runtime_manifest_writer;
mod runtime_package;
mod runtime_package_linux;
mod runtime_package_macos;
mod runtime_package_manifest;
mod select_runtime;
mod toolchain;
mod windows_deps;
mod windows_deps_pe;
mod windows_deps_policy;

use crate::command::DynResult;
use std::path::PathBuf;

/// The same regular-file identity used by the import-policy producer.
pub(crate) fn host_binary_sha256(path: &std::path::Path) -> Result<String, String> {
    host_dependencies::binary_sha256(path)
}

/// Reapply the owning neutral-host import policy when admitting a producer report.
pub(crate) fn rejected_host_imports(imports: &[String]) -> Vec<String> {
    forbidden::forbidden(imports)
}

/// A `native` subcommand.
#[derive(Clone, Copy)]
pub(crate) enum NativeCommand {
    SelectRuntime,
    VerifyHostDependencies,
    LinuxRuntimeDeps,
    WindowsRuntimeDeps,
    ReleaseMatrix,
    VerifyRuntimePackage,
    PackageSourceVersion,
    RuntimeManifestWrite,
}

impl NativeCommand {
    pub(crate) fn parse(name: &str) -> Option<Self> {
        match name {
            "select-runtime" => Some(Self::SelectRuntime),
            "verify-host-dependencies" => Some(Self::VerifyHostDependencies),
            "linux-runtime-deps" => Some(Self::LinuxRuntimeDeps),
            "windows-runtime-deps" => Some(Self::WindowsRuntimeDeps),
            "release-matrix" => Some(Self::ReleaseMatrix),
            "verify-runtime-package" => Some(Self::VerifyRuntimePackage),
            "package-source-version" => Some(Self::PackageSourceVersion),
            "runtime-manifest-write" => Some(Self::RuntimeManifestWrite),
            _ => None,
        }
    }
}

/// `root` resolves the checkout lazily: only `--max-glibc declared` reads
/// `scripts/linux-glibc-floor.txt` from it.
pub(crate) fn run(
    command: NativeCommand,
    args: &[String],
    root: impl FnOnce() -> DynResult<PathBuf>,
) -> DynResult<()> {
    let report = match command {
        NativeCommand::RuntimeManifestWrite => return runtime_manifest_writer::run(args),
        NativeCommand::PackageSourceVersion => package_version_command::run(args),
        NativeCommand::SelectRuntime => select_runtime::run(args),
        NativeCommand::LinuxRuntimeDeps => linux_deps::run(args, &toolchain::HostToolchain),
        NativeCommand::WindowsRuntimeDeps => windows_deps::run(args, &toolchain::HostToolchain),
        NativeCommand::ReleaseMatrix => release_matrix::run(args),
        NativeCommand::VerifyRuntimePackage => {
            runtime_package::run(args, &toolchain::HostToolchain)
        }
        NativeCommand::VerifyHostDependencies => {
            let mut root = Some(root);
            let mut floor_file = || {
                let resolve = root.take().ok_or("checkout already resolved")?;
                let root = resolve().unwrap_or_else(|_| build_checkout());
                Ok(root.join("scripts").join("linux-glibc-floor.txt"))
            };
            host_dependencies::run(args, &toolchain::HostToolchain, &mut floor_file)
        }
    };
    report.emit()
}

/// The checkout xtask was built from. The legacy script read the floor
/// beside itself, so outside any checkout this is the equivalent location.
fn build_checkout() -> PathBuf {
    let manifest = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    manifest
        .ancestors()
        .nth(2)
        .unwrap_or(manifest)
        .to_path_buf()
}
