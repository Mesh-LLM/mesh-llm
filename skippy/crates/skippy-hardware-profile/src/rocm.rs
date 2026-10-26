use skippy_native_runtime::HostGpuProfile;
use std::collections::BTreeSet;
#[cfg(any(target_os = "linux", test))]
use std::path::Path;

#[cfg(target_os = "linux")]
const KFD_TOPOLOGY_NODES: &str = "/sys/class/kfd/kfd/topology/nodes";

/// Prints the `gcnArchName` of each GPU the HIP runtime of AMD's Windows driver
/// sees, from `amdhip64_6.dll`, else `amdhip64_7.dll`. HIP 6.4 and 7.2 share
/// the `hipDeviceProp_tR0600` layout: 1,472 bytes on x86_64, with
/// `gcnArchName[256]` at offset 1,160. Single quotes only, so the script passes
/// through `-Command` unchanged; `[char]34` supplies the quotes C# needs.
#[cfg(target_os = "windows")]
const HIP_ARCH_PROBE: &str = concat!(
    "$ErrorActionPreference = 'Stop'; ",
    "foreach ($version in 6, 7) { try { ",
    "$hip = Add-Type -PassThru -Namespace MeshLlm -Name ('Hip' + $version) -MemberDefinition (",
    "'[DllImport({0}amdhip64_{1}.dll{0})] public static extern int hipGetDeviceCount(out int count); ",
    "[DllImport({0}amdhip64_{1}.dll{0})] public static extern int hipGetDevicePropertiesR0600(byte[] prop, int device);'",
    " -f [char]34, $version); ",
    "$count = 0; ",
    "if ($hip::hipGetDeviceCount([ref]$count) -ne 0) { continue }; ",
    "for ($device = 0; $device -lt $count; $device++) { ",
    "$prop = New-Object byte[] 1472; ",
    "if ($hip::hipGetDevicePropertiesR0600($prop, $device) -eq 0) { ",
    "[Text.Encoding]::ASCII.GetString($prop, 1160, 256).Split([char]0)[0] } }; ",
    "exit 0 } catch { } }; ",
    "exit 1",
);

/// Returns ROCm architecture evidence for native-runtime selection.
///
/// This deliberately does not construct GPU inventory. Once the selected
/// native runtime is loaded, Skippy's backend ABI remains authoritative for
/// device identity, ordering, memory, and runtime selectability.
#[cfg(target_os = "linux")]
pub(super) fn gpu_arches(_gpus: &[HostGpuProfile]) -> BTreeSet<String> {
    gpu_arches_from_topology(Path::new(KFD_TOPOLOGY_NODES))
}

/// Windows has no KFD topology, so this asks the HIP runtime that ships with
/// AMD's driver, once per process and only when an AMD GPU is listed. It runs
/// in a child process: loading the driver's `amdhip64_6.dll` here would make a
/// ROCm runtime loaded later bind to that copy instead of the one it bundles,
/// since Windows reuses an already loaded DLL of the same name.
#[cfg(target_os = "windows")]
pub(super) fn gpu_arches(gpus: &[HostGpuProfile]) -> BTreeSet<String> {
    static PROBED: std::sync::OnceLock<BTreeSet<String>> = std::sync::OnceLock::new();
    if !super::has_rocm_label(gpus) {
        return BTreeSet::new();
    }
    PROBED
        .get_or_init(|| {
            crate::platform::command_output(
                "powershell",
                &["-NoProfile", "-NonInteractive", "-Command", HIP_ARCH_PROBE],
            )
            .map(|output| gfx_arches_from_hip_probe(&output))
            .unwrap_or_default()
        })
        .clone()
}

#[cfg(not(any(target_os = "linux", target_os = "windows")))]
pub(super) fn gpu_arches(_gpus: &[HostGpuProfile]) -> BTreeSet<String> {
    BTreeSet::new()
}

/// The HIP probe's architectures, one `gcnArchName` per line. Target features
/// (`gfx90a:sramecc+:xnack-`) are dropped, since runtimes list bare targets as
/// KFD reports them, and so is anything else the driver prints.
#[cfg(any(target_os = "windows", test))]
fn gfx_arches_from_hip_probe(output: &str) -> BTreeSet<String> {
    output
        .lines()
        .filter_map(|line| line.trim().split(':').next())
        .filter(|arch| {
            arch.strip_prefix("gfx").is_some_and(|target| {
                !target.is_empty() && target.bytes().all(|byte| byte.is_ascii_hexdigit())
            })
        })
        .map(str::to_string)
        .collect()
}

#[cfg(any(target_os = "linux", test))]
fn gpu_arches_from_topology(root: &Path) -> BTreeSet<String> {
    let Ok(nodes) = std::fs::read_dir(root) else {
        return BTreeSet::new();
    };

    nodes
        .flatten()
        .filter_map(|entry| gfx_arch_from_kfd_node(&entry.path()))
        .collect()
}

#[cfg(any(target_os = "linux", test))]
fn gfx_arch_from_kfd_node(node: &Path) -> Option<String> {
    let gpu_id = std::fs::read_to_string(node.join("gpu_id")).ok()?;
    if gpu_id.trim().parse::<u64>().ok()? == 0 {
        return None;
    }

    let properties = std::fs::read_to_string(node.join("properties")).ok()?;
    let version = property_value(&properties, "gfx_target_version")?
        .parse()
        .ok()?;
    gfx_arch_from_target_version(version)
}

#[cfg(any(target_os = "linux", test))]
fn property_value<'a>(properties: &'a str, name: &str) -> Option<&'a str> {
    properties.lines().find_map(|line| {
        let mut fields = line.split_whitespace();
        (fields.next()? == name).then(|| fields.next()).flatten()
    })
}

#[cfg(any(target_os = "linux", test))]
fn gfx_arch_from_target_version(version: u64) -> Option<String> {
    if version == 0 {
        return None;
    }

    let major = version / 10_000;
    let minor = (version / 100) % 100;
    let stepping = version % 100;
    if major == 0 || minor > 0xf || stepping > 0xf {
        return None;
    }

    Some(format!("gfx{major}{minor:x}{stepping:x}"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU64, Ordering};

    static NEXT_FIXTURE: AtomicU64 = AtomicU64::new(0);

    struct TopologyFixture {
        root: std::path::PathBuf,
    }

    impl TopologyFixture {
        fn new() -> Self {
            let suffix = NEXT_FIXTURE.fetch_add(1, Ordering::Relaxed);
            let root = std::env::temp_dir().join(format!(
                "mesh-llm-kfd-topology-{}-{suffix}",
                std::process::id()
            ));
            std::fs::create_dir_all(&root).expect("create KFD topology fixture");
            Self { root }
        }

        fn add_node(&self, index: usize, gpu_id: u64, gfx_target_version: u64) {
            let node = self.root.join(index.to_string());
            std::fs::create_dir_all(&node).expect("create KFD node fixture");
            std::fs::write(node.join("gpu_id"), format!("{gpu_id}\n"))
                .expect("write KFD gpu_id fixture");
            std::fs::write(
                node.join("properties"),
                format!(
                    "vendor_id 4098\ndevice_id 29857\ngfx_target_version {gfx_target_version}\n"
                ),
            )
            .expect("write KFD properties fixture");
        }
    }

    impl Drop for TopologyFixture {
        fn drop(&mut self) {
            std::fs::remove_dir_all(&self.root).ok();
        }
    }

    #[test]
    fn discovers_gpu_arches_and_ignores_cpu_nodes() {
        let fixture = TopologyFixture::new();
        fixture.add_node(0, 0, 0);
        fixture.add_node(1, 51_844, 90_402);
        fixture.add_node(2, 74_492, 110_000);

        assert_eq!(
            gpu_arches_from_topology(&fixture.root),
            BTreeSet::from(["gfx942".to_string(), "gfx1100".to_string()])
        );
    }

    #[test]
    fn formats_hexadecimal_kfd_stepping() {
        assert_eq!(
            gfx_arch_from_target_version(90_010).as_deref(),
            Some("gfx90a")
        );
    }

    #[test]
    fn reads_bare_targets_from_the_hip_probe() {
        let output = "HIP Library Path: C:\\WINDOWS\\SYSTEM32\\amdhip64_7.dll\r\n\
                      gfx1103\r\ngfx90a:sramecc+:xnack-\r\n\r\ngfx\r\ngfxzzz\r\n";

        assert_eq!(
            gfx_arches_from_hip_probe(output),
            BTreeSet::from(["gfx1103".to_string(), "gfx90a".to_string()])
        );
    }

    #[test]
    fn rejects_missing_or_malformed_kfd_versions() {
        assert_eq!(gfx_arch_from_target_version(0), None);
        assert_eq!(gfx_arch_from_target_version(91_600), None);
        assert_eq!(gfx_arch_from_target_version(90_016), None);
    }
}
