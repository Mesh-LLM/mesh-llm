//! Checked-in exact cache expressions. Rendering runs during maintenance, never lookup.
use crate::command::DynResult;
use std::collections::BTreeSet;
use std::path::Path;

#[path = "cache_inputs_closure.rs"]
mod closure;
#[path = "cache_inputs_resources.rs"]
mod resources;
#[cfg(test)]
#[path = "cache_inputs_tests.rs"]
mod tests;

#[derive(Clone, Copy, Debug)]
pub(crate) enum Inventory {
    HostedXtask,
    HostedController,
    LinuxRuntime,
    WindowsAbi,
}

pub(crate) const ALL: [Inventory; 4] = [
    Inventory::HostedXtask,
    Inventory::HostedController,
    Inventory::LinuxRuntime,
    Inventory::WindowsAbi,
];

const COMMON: &[&str] = &[
    "Cargo.toml",
    "Cargo.lock",
    ".cargo/**",
    ".github/cache-version.txt",
    "scripts/cargo-linker",
    "scripts/cargo-linker-linux-*",
    "scripts/lib/lld.sh",
];
const XTASK: &[&str] = &[
    "tools/xtask",
    "tools/authority-automation",
    "tools/ci-controller",
    "tools/process-supervision",
    "tools/runner-cleanup",
    "tools/trajectory-reader",
    "mesh/crates/mesh-llm-release-footer",
];
const PACKAGE_TOOL: &[&str] = &[
    "skippy/crates/skippy-package-builder",
    "skippy/crates/skippy-ffi",
    "skippy/crates/skippy-model",
    "skippy/crates/skippy-package-format",
    "skippy/crates/skippy-runtime",
    "skippy/crates/skippy-cache",
    "skippy/crates/skippy-model-ref",
    "skippy/crates/skippy-model-artifact",
    "skippy/crates/skippy-model-hf",
    "skippy/crates/skippy-model-resolver",
    "skippy/crates/skippy-hf-hub",
    "skippy/crates/skippy-protocol",
    "skippy/crates/skippy-tokenizer",
];

impl Inventory {
    pub(crate) fn path(self) -> &'static str {
        match self {
            Self::HostedXtask => ".github/actions/prepare-automation/action.yml",
            Self::HostedController => ".github/actions/prepare-ci-controller/action.yml",
            Self::LinuxRuntime => ".github/workflows/ci-linux-runtime-slice.yml",
            Self::WindowsAbi => ".github/actions/restore-windows-abi-cache/action.yml",
        }
    }

    fn prefix(self) -> &'static str {
        match self {
            Self::HostedXtask => {
                "xtask-hosted-bare-v2-${{ runner.os }}-${{ runner.arch }}-${{ steps.rust_toolchain.outputs.cachekey }}-${{ steps.hosted_image.outputs.identity }}-"
            }
            Self::HostedController => {
                "ci-controller-hosted-v2-${{ runner.os }}-${{ runner.arch }}-${{ steps.rust_toolchain.outputs.cachekey }}-${{ steps.hosted_image.outputs.identity }}-"
            }
            Self::LinuxRuntime => {
                "skippy-runtime-linux-v2-${{ matrix.runtime.target }}-${{ matrix.runtime.toolchain_epoch }}-"
            }
            Self::WindowsAbi => "",
        }
    }

    pub(crate) fn inputs(self) -> BTreeSet<String> {
        let mut inputs: BTreeSet<String> = COMMON.iter().map(|s| (*s).into()).collect();
        inputs.insert(self.path().into());
        let packages: Vec<&str> = match self {
            Self::HostedXtask => XTASK.to_vec(),
            Self::HostedController => vec!["tools/ci-controller"],
            Self::LinuxRuntime => XTASK.iter().chain(PACKAGE_TOOL).copied().collect(),
            Self::WindowsAbi => XTASK.to_vec(),
        };
        for package in packages {
            inputs.insert(format!("{package}/Cargo.toml"));
            inputs.insert(format!("{package}/src/**"));
        }
        match self {
            Self::HostedController => {}
            Self::WindowsAbi => {
                inputs.extend(
                    [
                        "skippy/llama_cpp/**",
                        "skippy/scripts/**",
                        "scripts/**",
                        "tools/xtask/src/native_policy/**",
                        "tools/xtask/src/prepared_input/**",
                        "tools/xtask/build.rs",
                        "ci/python-research-source.json",
                        "scripts/ci-compat-smoke.sh",
                        "scripts/linux-glibc-floor.txt",
                        "skippy/crates/skippy-gpu-bench/native/**",
                        ".github/workflows/ci-windows-runtime-slice.yml",
                        ".github/workflows/windows-warm-caches.yml",
                        ".github/workflows/release.yml",
                        ".github/actions/save-and-verify-actions-cache/**",
                        ".github/actions/resolve-native-toolchain-epoch/**",
                        ".github/actions/prepare-native-runtime-input/**",
                        ".github/actions/setup-windows-rocm-sdk/**",
                    ]
                    .map(str::to_owned),
                );
            }
            Self::HostedXtask | Self::LinuxRuntime => {
                inputs.extend(
                    [
                        "tools/xtask/build.rs",
                        "ci/python-research-source.json",
                        "scripts/ci-compat-smoke.sh",
                        "scripts/linux-glibc-floor.txt",
                    ]
                    .map(str::to_owned),
                );
            }
        }
        if let Self::LinuxRuntime = self {
            // Keep native recipes whole: their shell/CMake source closure is not a Rust graph.
            inputs.extend(
                [
                    "skippy/llama_cpp/**",
                    "skippy/scripts/**",
                    "scripts/**",
                    "skippy/crates/skippy-ffi/build.rs",
                    "skippy/crates/skippy-ffi/build/**",
                    "skippy/crates/skippy-ffi/bindgen/**",
                    "skippy/crates/skippy-protocol/build.rs",
                    "skippy/crates/skippy-protocol/proto/**",
                    "skippy/crates/skippy-native-runtime/RUNTIME_VERSION",
                    "skippy/crates/skippy-gpu-bench/native/**",
                    ".github/actions/prepare-native-runtime-input/**",
                    ".github/actions/resolve-native-toolchain-epoch/**",
                ]
                .map(str::to_owned),
            );
        }
        inputs
    }

    pub(crate) fn expression(self) -> String {
        let inputs = self
            .inputs()
            .into_iter()
            .map(|p| format!("'{p}'"))
            .collect::<Vec<_>>()
            .join(", ");
        format!("{}${{{{ hashFiles({inputs}) }}}}", self.prefix())
    }
}

/// `render` prints YAML projections; Windows is a CACHE_INPUT_HASH value.
pub(crate) fn run(root: &Path, args: &[String]) -> DynResult<()> {
    match args
        .iter()
        .map(String::as_str)
        .collect::<Vec<_>>()
        .as_slice()
    {
        ["render"] => {
            for inventory in ALL {
                let field = match inventory {
                    Inventory::WindowsAbi => "CACHE_INPUT_HASH",
                    Inventory::HostedXtask
                    | Inventory::HostedController
                    | Inventory::LinuxRuntime => "key",
                };
                println!("{}\n{field}: {}", inventory.path(), inventory.expression());
            }
            Ok(())
        }
        ["check"] => check(root),
        _ => Err("usage: ci-ops cache-inputs <render|check>".into()),
    }
}

pub(crate) fn check(root: &Path) -> DynResult<()> {
    for inventory in ALL {
        let source = std::fs::read_to_string(root.join(inventory.path()))?;
        let field = match inventory {
            Inventory::WindowsAbi => "CACHE_INPUT_HASH: ",
            Inventory::HostedXtask | Inventory::HostedController | Inventory::LinuxRuntime => {
                "key: "
            }
        };
        let keys: Vec<_> = source
            .lines()
            .filter_map(|line| line.trim().strip_prefix(field))
            .filter(|key| {
                key.starts_with(inventory.prefix().split("-${{").next().unwrap_or_default())
            })
            .collect();
        if keys != [inventory.expression()] {
            return Err(format!(
                "{}: cache input projection is stale; run ci-ops cache-inputs render",
                inventory.path()
            )
            .into());
        }
        closure::check(root, inventory)?;
        resources::check(root, inventory)?;
    }
    Ok(())
}
