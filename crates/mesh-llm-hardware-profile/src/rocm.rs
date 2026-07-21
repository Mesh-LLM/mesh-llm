use mesh_llm_native_runtime::HostGpuProfile;

#[derive(Default)]
struct RocmAgent {
    name: Option<String>,
    marketing_name: Option<String>,
    device_type: Option<String>,
}

pub(super) fn gpu_profiles_from_rocminfo(output: &str) -> Vec<HostGpuProfile> {
    let mut profiles = Vec::new();
    let mut agent = RocmAgent::default();

    for line in output.lines().map(str::trim) {
        if is_agent_header(line) {
            push_gpu_profile(&mut profiles, &mut agent);
            continue;
        }
        let Some((key, value)) = line.split_once(':') else {
            continue;
        };
        let value = value.trim();
        if value.is_empty() {
            continue;
        }
        match key.trim().to_ascii_lowercase().as_str() {
            "name" => agent.name = Some(value.to_string()),
            "marketing name" => agent.marketing_name = Some(value.to_string()),
            "device type" => agent.device_type = Some(value.to_string()),
            _ => {}
        }
    }
    push_gpu_profile(&mut profiles, &mut agent);

    profiles
}

fn is_agent_header(line: &str) -> bool {
    line.strip_prefix("Agent ")
        .is_some_and(|index| index.trim().parse::<usize>().is_ok())
}

fn push_gpu_profile(profiles: &mut Vec<HostGpuProfile>, agent: &mut RocmAgent) {
    let Some(gfx_arch) = agent.name.as_deref().and_then(parse_gfx_arch) else {
        *agent = RocmAgent::default();
        return;
    };
    let is_gpu = agent
        .device_type
        .as_deref()
        .is_none_or(|kind| kind.eq_ignore_ascii_case("GPU"));
    if is_gpu {
        profiles.push(HostGpuProfile {
            display_name: agent
                .marketing_name
                .take()
                .filter(|name| !name.is_empty())
                .unwrap_or_else(|| gfx_arch.clone()),
            backend_device: None,
            stable_id: None,
            vram_bytes: None,
            unified_memory: false,
            probe: None,
            cuda_sm: None,
            rocm_gfx: Some(gfx_arch),
        });
    }
    *agent = RocmAgent::default();
}

fn parse_gfx_arch(value: &str) -> Option<String> {
    let token = value.split_whitespace().next()?;
    let arch = token.split(':').next()?.trim().to_ascii_lowercase();
    let suffix = arch.strip_prefix("gfx")?;
    (!suffix.is_empty() && suffix.chars().all(|ch| ch.is_ascii_alphanumeric())).then_some(arch)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_mi300x_agent_and_ignores_cpu_agent() {
        let output = r#"
*******
Agent 1
*******
  Name:                    AMD EPYC 9654 96-Core Processor
  Marketing Name:          AMD EPYC 9654 96-Core Processor
  Device Type:             CPU
*******
Agent 2
*******
  Name:                    gfx942
  Marketing Name:          AMD Instinct MI300X
  Device Type:             GPU
"#;

        let profiles = gpu_profiles_from_rocminfo(output);

        assert_eq!(profiles.len(), 1);
        assert_eq!(profiles[0].display_name, "AMD Instinct MI300X");
        assert_eq!(profiles[0].rocm_gfx.as_deref(), Some("gfx942"));
        assert_eq!(profiles[0].backend_device, None);
    }

    #[test]
    fn parses_multiple_gpu_agents_and_normalizes_feature_suffixes() {
        let output = r#"
Agent 1
  Name: gfx942:sramecc+:xnack-
  Marketing Name: AMD Instinct MI300X
  Device Type: GPU
Agent 2
  Name: GFX1100
  Marketing Name: AMD Radeon RX 7900 XTX
  Device Type: GPU
"#;

        let profiles = gpu_profiles_from_rocminfo(output);

        assert_eq!(profiles.len(), 2);
        assert_eq!(profiles[0].rocm_gfx.as_deref(), Some("gfx942"));
        assert_eq!(profiles[1].rocm_gfx.as_deref(), Some("gfx1100"));
    }

    #[test]
    fn uses_gfx_arch_when_marketing_name_is_unavailable() {
        let profiles = gpu_profiles_from_rocminfo("Agent 1\n  Name: gfx1201\n");

        assert_eq!(profiles.len(), 1);
        assert_eq!(profiles[0].display_name, "gfx1201");
        assert_eq!(profiles[0].rocm_gfx.as_deref(), Some("gfx1201"));
    }
}
