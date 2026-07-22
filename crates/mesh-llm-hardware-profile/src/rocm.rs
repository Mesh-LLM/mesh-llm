use std::collections::BTreeSet;
#[cfg(any(target_os = "linux", test))]
use std::path::Path;

#[cfg(target_os = "linux")]
const KFD_TOPOLOGY_NODES: &str = "/sys/class/kfd/kfd/topology/nodes";

/// Returns ROCm architecture evidence for native-runtime selection.
///
/// This deliberately does not construct GPU inventory. Once the selected
/// native runtime is loaded, Skippy's backend ABI remains authoritative for
/// device identity, ordering, memory, and runtime selectability.
#[cfg(target_os = "linux")]
pub(super) fn gpu_arches() -> BTreeSet<String> {
    gpu_arches_from_topology(Path::new(KFD_TOPOLOGY_NODES))
}

#[cfg(not(target_os = "linux"))]
pub(super) fn gpu_arches() -> BTreeSet<String> {
    BTreeSet::new()
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
    fn rejects_missing_or_malformed_kfd_versions() {
        assert_eq!(gfx_arch_from_target_version(0), None);
        assert_eq!(gfx_arch_from_target_version(91_600), None);
        assert_eq!(gfx_arch_from_target_version(90_016), None);
    }
}
