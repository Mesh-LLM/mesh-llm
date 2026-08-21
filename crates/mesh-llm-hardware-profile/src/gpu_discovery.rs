//! GPU inventory and NVIDIA probe parsing.

use crate::platform::{command_output, gpu_labels, looks_like_display_controller};
use mesh_llm_native_runtime::HostGpuProfile;
use mesh_llm_native_runtime::host::HostGpuProbe;
use std::collections::BTreeMap;
pub(crate) fn detect_gpus() -> Vec<HostGpuProfile> {
    merge_nvidia_and_fallback_gpus(detect_nvidia_gpu_profiles(), fallback_gpu_profiles())
}

pub(crate) fn merge_nvidia_and_fallback_gpus(
    mut nvidia_gpus: Vec<HostGpuProfile>,
    mut fallback_gpus: Vec<HostGpuProfile>,
) -> Vec<HostGpuProfile> {
    if nvidia_gpus.is_empty() {
        return fallback_gpus;
    }

    fallback_gpus.retain(|gpu| !looks_like_nvidia_gpu_label(&gpu.display_name));
    nvidia_gpus.extend(fallback_gpus);
    nvidia_gpus
}

pub(crate) fn fallback_gpu_profiles() -> Vec<HostGpuProfile> {
    gpu_labels()
        .into_iter()
        .map(fallback_gpu_profile_from_label)
        .collect()
}

pub(crate) fn fallback_gpu_profile_from_label(label: String) -> HostGpuProfile {
    HostGpuProfile {
        display_name: label,
        backend_device: None,
        stable_id: None,
        vram_bytes: None,
        unified_memory: cfg!(target_os = "macos"),
        probe: None,
        cuda_sm: None,
        rocm_gfx: None,
    }
}

pub(crate) fn looks_like_nvidia_gpu_label(label: &str) -> bool {
    let label = label.to_ascii_lowercase();
    label.contains("nvidia") || label.contains("cuda")
}

pub(crate) fn detect_nvidia_gpu_profiles() -> Vec<HostGpuProfile> {
    let Some(nvidia_smi) = command_output("nvidia-smi", &["-L"]) else {
        return Vec::new();
    };
    let compute_caps = command_output(
        "nvidia-smi",
        &[
            "--query-gpu=index,compute_cap",
            "--format=csv,noheader,nounits",
        ],
    )
    .map(|output| nvidia_compute_caps_by_index(&output))
    .unwrap_or_default();
    let lspci = command_output("lspci", &[]).unwrap_or_default();
    let proc_entries = linux_nvidia_proc_information_entries();
    let borrowed_entries: Vec<(&str, &str)> = proc_entries
        .iter()
        .map(|entry| (entry.path.as_str(), entry.info.as_str()))
        .collect();
    nvidia_gpu_profiles_from_probe_outputs(&nvidia_smi, &compute_caps, &lspci, &borrowed_entries)
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct NvidiaSmiGpu {
    index: usize,
    name: String,
    vendor_uuid: Option<String>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct NvidiaProcInformationEntry {
    path: String,
    info: String,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct NvidiaProcProbe {
    pci_bdf: Option<String>,
    vendor_uuid: Option<String>,
    probe: HostGpuProbe,
}

pub(crate) fn nvidia_gpu_profiles_from_probe_outputs(
    nvidia_smi_output: &str,
    compute_caps: &BTreeMap<usize, String>,
    lspci_output: &str,
    proc_entries: &[(&str, &str)],
) -> Vec<HostGpuProfile> {
    let mut proc_probes = proc_entries
        .iter()
        .map(|(path, info)| nvidia_proc_probe(path, info))
        .collect::<Vec<_>>();

    parse_nvidia_smi_list(nvidia_smi_output)
        .into_iter()
        .map(|gpu| {
            let pci_bdf = nvidia_lspci_bdf_for_name(lspci_output, &gpu.name);
            let probe = take_matching_nvidia_probe(
                &mut proc_probes,
                gpu.vendor_uuid.as_deref(),
                pci_bdf.as_deref(),
            );
            HostGpuProfile {
                display_name: gpu.name,
                backend_device: Some(format!("CUDA{}", gpu.index)),
                stable_id: gpu
                    .vendor_uuid
                    .as_ref()
                    .map(|uuid| format!("uuid:{uuid}"))
                    .or_else(|| pci_bdf.as_ref().map(|bdf| format!("pci:{bdf}"))),
                vram_bytes: None,
                unified_memory: false,
                probe,
                cuda_sm: compute_caps.get(&gpu.index).cloned(),
                rocm_gfx: None,
            }
        })
        .collect()
}

pub(crate) fn nvidia_compute_caps_by_index(output: &str) -> BTreeMap<usize, String> {
    output
        .lines()
        .filter_map(|line| {
            let (index, compute_cap) = line.split_once(',')?;
            let index = index.trim().parse::<usize>().ok()?;
            let cuda_sm = cuda_sm_from_compute_cap(compute_cap.trim())?;
            Some((index, cuda_sm))
        })
        .collect()
}

pub(crate) fn cuda_sm_from_compute_cap(value: &str) -> Option<String> {
    let (major, minor) = value.split_once('.')?;
    let major = major.trim();
    let minor = minor.trim();
    if major.is_empty()
        || minor.is_empty()
        || !major.chars().all(|ch| ch.is_ascii_digit())
        || !minor.chars().all(|ch| ch.is_ascii_digit())
    {
        return None;
    }
    Some(format!("{major}{minor}"))
}

pub(crate) fn parse_nvidia_smi_list(output: &str) -> Vec<NvidiaSmiGpu> {
    output
        .lines()
        .filter_map(|line| {
            let line = line.trim();
            let body = line.strip_prefix("GPU ")?;
            let (index, rest) = body.split_once(':')?;
            let index = index.trim().parse::<usize>().ok()?;
            let rest = rest.trim();
            let (name, vendor_uuid) = match rest.rsplit_once(" (UUID: ") {
                Some((name, uuid)) => (name.trim(), uuid.strip_suffix(')').map(str::trim)),
                None => (rest, None),
            };
            (!name.is_empty()).then(|| NvidiaSmiGpu {
                index,
                name: name.to_string(),
                vendor_uuid: vendor_uuid.map(ToOwned::to_owned),
            })
        })
        .collect()
}

pub(crate) fn nvidia_lspci_bdf_for_name(output: &str, name: &str) -> Option<String> {
    let name = name.to_ascii_lowercase();
    output.lines().find_map(|line| {
        let line = line.trim();
        if !looks_like_display_controller(line) {
            return None;
        }
        let lower = line.to_ascii_lowercase();
        if !name
            .split_whitespace()
            .filter(|token| *token != "nvidia" && *token != "geforce")
            .all(|token| lower.contains(token))
        {
            return None;
        }
        line.split_whitespace().next().map(normalize_pci_bdf)
    })
}

pub(crate) fn normalize_pci_bdf(bdf: &str) -> String {
    if bdf.matches(':').count() == 1 {
        format!("0000:{bdf}").to_ascii_lowercase()
    } else {
        bdf.to_ascii_lowercase()
    }
}

pub(crate) fn nvidia_proc_probe(path: &str, info: &str) -> NvidiaProcProbe {
    let fields = nvidia_proc_fields(info);
    NvidiaProcProbe {
        pci_bdf: fields
            .get("Bus Location")
            .map(String::as_str)
            .map(normalize_pci_bdf),
        vendor_uuid: fields.get("GPU UUID").cloned(),
        probe: HostGpuProbe {
            source: "linux_nvidia_proc".to_string(),
            path: Some(path.to_string()),
            fields,
            raw_lines: info.lines().map(str::to_string).collect(),
        },
    }
}

pub(crate) fn nvidia_proc_fields(info: &str) -> BTreeMap<String, String> {
    info.lines()
        .filter_map(|line| {
            let (key, value) = line.split_once(':')?;
            let key = key.trim();
            if key.is_empty() {
                return None;
            }
            Some((key.to_string(), value.trim().to_string()))
        })
        .collect()
}

pub(crate) fn take_matching_nvidia_probe(
    probes: &mut Vec<NvidiaProcProbe>,
    vendor_uuid: Option<&str>,
    pci_bdf: Option<&str>,
) -> Option<HostGpuProbe> {
    let index = probes.iter().position(|probe| {
        vendor_uuid.is_some_and(|uuid| probe.vendor_uuid.as_deref() == Some(uuid))
            || pci_bdf.is_some_and(|bdf| probe.pci_bdf.as_deref() == Some(bdf))
    })?;
    Some(probes.remove(index).probe)
}

#[cfg(target_os = "linux")]
pub(crate) fn linux_nvidia_proc_information_entries() -> Vec<NvidiaProcInformationEntry> {
    let Ok(entries) = std::fs::read_dir("/proc/driver/nvidia/gpus") else {
        return Vec::new();
    };
    entries
        .flatten()
        .filter_map(|entry| {
            let path = entry.path().join("information");
            let info = std::fs::read_to_string(&path).ok()?;
            Some(NvidiaProcInformationEntry {
                path: path.display().to_string(),
                info,
            })
        })
        .collect()
}

#[cfg(not(target_os = "linux"))]
pub(crate) fn linux_nvidia_proc_information_entries() -> Vec<NvidiaProcInformationEntry> {
    Vec::new()
}

#[cfg(test)]
mod tests {
    use super::normalize_pci_bdf;

    #[test]
    fn normalizes_short_and_long_pci_bdf_case() {
        assert_eq!(normalize_pci_bdf("AB:0C.0"), "0000:ab:0c.0");
        assert_eq!(normalize_pci_bdf("0000:AB:0C.0"), "0000:ab:0c.0");
    }
}
