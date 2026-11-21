use super::*;

impl Collector for DefaultCollector {
    fn collect(&self, metrics: &[Metric]) -> HardwareSurvey {
        let mut survey = HardwareSurvey::default();

        #[cfg(feature = "skippy-devices")]
        if apply_skippy_backend_devices_to_survey(&mut survey, metrics) {
            return survey;
        }

        #[cfg(all(
            target_os = "macos",
            any(not(feature = "skippy-devices"), feature = "dynamic-native-runtime")
        ))]
        {
            if metrics.contains(&Metric::IsSoc) {
                survey.is_soc = true;
            }
            let metal_budget = if metrics.contains(&Metric::VramBytes) {
                macos_metal_gpu_budget(query_metal_recommended_working_set_bytes())
            } else {
                None
            };
            if let Some((vram_bytes, reserved_bytes)) = metal_budget {
                survey.vram_bytes = vram_bytes;
                survey.gpu_vram = vec![vram_bytes];
                survey.gpu_reserved = vec![reserved_bytes];
            }
            if metrics.contains(&Metric::GpuName) {
                let name = sanitize_macos_gpu_name(query_metal_device_name());
                survey.gpu_name_source =
                    name.is_some().then_some(GpuNameSource::MetalDefaultDevice);
                survey.gpu_name = name;
            }
            if metrics.contains(&Metric::GpuCount) {
                survey.gpu_count = 1;
            }
        }

        #[cfg(all(
            target_os = "linux",
            any(not(feature = "skippy-devices"), feature = "dynamic-native-runtime")
        ))]
        {
            let system_ram = read_system_ram_bytes();
            if system_ram > 0 {
                survey.system_ram_bytes = Some(system_ram);
            }

            if metrics.contains(&Metric::VramBytes) {
                // Try NVIDIA (mesh.rs:284-316)
                let nvidia_vram: Option<(u64, Vec<u64>)> = (|| {
                    let out = skippy_hardware_profile::probe::output(
                        std::process::Command::new("nvidia-smi").args([
                            "--query-gpu=memory.total,memory.reserved",
                            "--format=csv,noheader,nounits",
                        ]),
                    )
                    .ok();
                    match out {
                        Some(out) if out.status.success() => {
                            let s = String::from_utf8(out.stdout).ok()?;
                            let parsed = parse_nvidia_gpu_memory_and_reserved(&s);
                            if !parsed.is_empty() {
                                survey.gpu_reserved =
                                    parsed.iter().map(|(_, reserved)| *reserved).collect();
                                let per_gpu: Vec<u64> =
                                    parsed.iter().map(|(total, _)| *total).collect();
                                let total: u64 = per_gpu.iter().sum();
                                if total > 0 {
                                    return Some((total, per_gpu));
                                }
                            }
                        }
                        Some(_) | None => {}
                    }
                    let out = skippy_hardware_profile::probe::output(
                        std::process::Command::new("nvidia-smi")
                            .args(["--query-gpu=memory.total", "--format=csv,noheader,nounits"]),
                    )
                    .ok()?;
                    if !out.status.success() {
                        return None;
                    }
                    let s = String::from_utf8(out.stdout).ok()?;
                    let per_gpu: Vec<u64> = s
                        .lines()
                        .filter_map(|line| {
                            let mib = line.trim().parse::<u64>().ok()?;
                            Some(mib * 1024 * 1024)
                        })
                        .collect();
                    let total: u64 = per_gpu.iter().sum();
                    if total > 0 {
                        survey.gpu_reserved = vec![None; per_gpu.len()];
                        Some((total, per_gpu))
                    } else {
                        None
                    }
                })();

                if let Some((vram, per_gpu)) = nvidia_vram {
                    survey.gpu_vram = per_gpu;
                    let ram_offload = system_ram.saturating_sub(vram);
                    survey.vram_bytes = vram + (ram_offload as f64 * 0.90) as u64;
                } else {
                    // Try AMD ROCm (mesh.rs:295-316)
                    let rocm_vram: Option<(Vec<u64>, bool)> = (|| {
                        let out = skippy_hardware_profile::probe::output(
                            std::process::Command::new("rocm-smi").args([
                                "--showmeminfo",
                                "vram",
                                "--csv",
                            ]),
                        )
                        .ok()?;
                        if !out.status.success() {
                            return None;
                        }
                        let s = String::from_utf8(out.stdout).ok()?;
                        let parsed = parse_rocm_gpu_memory_and_used(&s);
                        // ROCm exposes total and live used VRAM here, not a
                        // true reserved/unavailable metric, so leave
                        // reserved_bytes unavailable for this backend.
                        survey.gpu_reserved = vec![None; parsed.len()];
                        let vrams: Vec<u64> = parsed.iter().map(|(total, _)| *total).collect();
                        if vrams.is_empty() {
                            None
                        } else {
                            let gtt_totals = skippy_hardware_profile::probe::output(
                                std::process::Command::new("rocm-smi").args([
                                    "--showmeminfo",
                                    "gtt",
                                    "--csv",
                                ]),
                            )
                            .ok()
                            .filter(|out| out.status.success())
                            .and_then(|out| String::from_utf8(out.stdout).ok())
                            .map(|stdout| {
                                parse_rocm_gpu_memory_and_used(&stdout)
                                    .into_iter()
                                    .map(|(total, _)| total)
                                    .collect::<Vec<_>>()
                            })
                            .unwrap_or_default();
                            if let Some(usable_bytes) =
                                rocm_unified_memory_usable_bytes(&vrams, &gtt_totals, system_ram)
                            {
                                Some((vec![usable_bytes], true))
                            } else {
                                Some((vrams, false))
                            }
                        }
                    })();

                    if let Some((per_gpu, unified_memory)) = rocm_vram {
                        let vram: u64 = per_gpu.iter().sum();
                        survey.gpu_vram = per_gpu;
                        if unified_memory {
                            survey.is_soc = true;
                            survey.vram_bytes = vram;
                        } else {
                            let ram_offload = system_ram.saturating_sub(vram);
                            survey.vram_bytes = vram + (ram_offload as f64 * 0.90) as u64;
                        }
                    } else {
                        let intel_gpus: Option<Vec<XpuSmiGpuInfo>> = (|| {
                            for args in [["discovery", "--json"], ["discovery", "-j"]] {
                                let out = skippy_hardware_profile::probe::output(
                                    std::process::Command::new("xpu-smi").args(args),
                                )
                                .ok()?;
                                if !out.status.success() {
                                    continue;
                                }
                                let stdout = String::from_utf8(out.stdout).ok()?;
                                let gpus = parse_xpu_smi_discovery_json(&stdout);
                                if !gpus.is_empty() {
                                    return Some(gpus);
                                }
                            }
                            None
                        })();

                        if let Some(intel_gpus) = intel_gpus {
                            // xpu-smi discovery reports capacity plus used
                            // bytes, but not a true reserved/unavailable
                            // metric, so leave reserved_bytes unavailable.
                            survey.gpu_reserved = vec![None; intel_gpus.len()];
                            let per_gpu: Vec<u64> = intel_gpus
                                .iter()
                                .map(|gpu| gpu.total_bytes.unwrap_or(0))
                                .collect();
                            let total: u64 = per_gpu.iter().sum();
                            survey.gpu_vram = per_gpu;
                            if total > 0 {
                                let ram_offload = system_ram.saturating_sub(total);
                                survey.vram_bytes = total + (ram_offload as f64 * 0.90) as u64;
                            } else if system_ram > 0 {
                                survey.vram_bytes = (system_ram as f64 * 0.90) as u64;
                            }
                        } else if system_ram > 0 {
                            // CPU-only (mesh.rs:320-322)
                            survey.vram_bytes = (system_ram as f64 * 0.90) as u64;
                        }
                    }
                }
            }

            if metrics.contains(&Metric::GpuName) || metrics.contains(&Metric::GpuCount) {
                let nvidia_names: Option<Vec<String>> = (|| {
                    let out = skippy_hardware_profile::probe::output(
                        std::process::Command::new("nvidia-smi")
                            .args(["--query-gpu=name", "--format=csv,noheader"]),
                    )
                    .ok()?;
                    if !out.status.success() {
                        return None;
                    }
                    let s = String::from_utf8(out.stdout).ok()?;
                    let names = parse_nvidia_gpu_names(&s);
                    if names.is_empty() { None } else { Some(names) }
                })();

                if let Some(ref names) = nvidia_names {
                    if metrics.contains(&Metric::GpuName) {
                        let name = summarize_gpu_name(names);
                        survey.gpu_name_source = name.is_some().then_some(GpuNameSource::NvidiaSmi);
                        survey.gpu_name = name;
                    }
                    if metrics.contains(&Metric::GpuCount) {
                        survey.gpu_count = u8::try_from(names.len()).unwrap_or(u8::MAX);
                    }
                } else {
                    let out = skippy_hardware_profile::probe::output(
                        std::process::Command::new("rocm-smi").args(["--showproductname"]),
                    )
                    .ok();
                    match out {
                        Some(out) if out.status.success() => {
                            if let Ok(s) = String::from_utf8(out.stdout) {
                                let names = parse_rocm_gpu_names(&s);
                                if metrics.contains(&Metric::GpuName) {
                                    let name = summarize_gpu_name(&names);
                                    survey.gpu_name_source =
                                        name.is_some().then_some(GpuNameSource::RocmSmi);
                                    survey.gpu_name = name;
                                }
                                if metrics.contains(&Metric::GpuCount) {
                                    survey.gpu_count = u8::try_from(names.len()).unwrap_or(u8::MAX);
                                }
                            }
                        }
                        None => {
                            for args in [["discovery", "--json"], ["discovery", "-j"]] {
                                let out = skippy_hardware_profile::probe::output(
                                    std::process::Command::new("xpu-smi").args(args),
                                )
                                .ok();
                                if let Some(out) = out {
                                    if !out.status.success() {
                                        continue;
                                    }
                                    let Ok(stdout) = String::from_utf8(out.stdout) else {
                                        continue;
                                    };
                                    let gpus = parse_xpu_smi_discovery_json(&stdout);
                                    if !gpus.is_empty() {
                                        let names: Vec<String> =
                                            gpus.iter().map(|gpu| gpu.name.clone()).collect();
                                        if metrics.contains(&Metric::GpuName) {
                                            let name = summarize_gpu_name(&names);
                                            survey.gpu_name_source =
                                                name.is_some().then_some(GpuNameSource::XpuSmi);
                                            survey.gpu_name = name;
                                        }
                                        if metrics.contains(&Metric::GpuCount) {
                                            survey.gpu_count =
                                                u8::try_from(names.len()).unwrap_or(u8::MAX);
                                        }
                                        break;
                                    }
                                }
                            }
                        }
                        Some(_) => {}
                    }
                }
            }
        }

        #[cfg(all(
            target_os = "windows",
            any(not(feature = "skippy-devices"), feature = "dynamic-native-runtime")
        ))]
        {
            let system_ram = read_windows_total_ram_bytes().unwrap_or(0);
            if system_ram > 0 {
                survey.system_ram_bytes = Some(system_ram);
            }
            let want_gpu_info =
                metrics.contains(&Metric::GpuName) || metrics.contains(&Metric::GpuCount);
            let want_vram = metrics.contains(&Metric::VramBytes);

            let nvidia_names = if want_gpu_info {
                skippy_hardware_profile::probe::output(
                    std::process::Command::new("nvidia-smi")
                        .args(["--query-gpu=name", "--format=csv,noheader"]),
                )
                .ok()
                .and_then(|out| {
                    if !out.status.success() {
                        return None;
                    }
                    let s = String::from_utf8(out.stdout).ok()?;
                    let names = parse_nvidia_gpu_names(&s);
                    if names.is_empty() { None } else { Some(names) }
                })
            } else {
                None
            };

            let nvidia_vram = if want_vram {
                skippy_hardware_profile::probe::output(
                    std::process::Command::new("nvidia-smi")
                        .args(["--query-gpu=memory.total", "--format=csv,noheader,nounits"]),
                )
                .ok()
                .and_then(|out| {
                    if !out.status.success() {
                        return None;
                    }
                    let s = String::from_utf8(out.stdout).ok()?;
                    let per_gpu = parse_nvidia_gpu_memory(&s);
                    if per_gpu.is_empty() {
                        None
                    } else {
                        Some(per_gpu)
                    }
                })
            } else {
                None
            };

            let windows_gpus = if want_gpu_info || want_vram {
                read_windows_video_controllers()
            } else {
                Vec::new()
            };

            if want_vram {
                if let Some(per_gpu) = nvidia_vram {
                    let total: u64 = per_gpu.iter().sum();
                    if total > 0 {
                        survey.gpu_vram = per_gpu;
                        let ram_offload = system_ram.saturating_sub(total);
                        survey.vram_bytes = total + (ram_offload as f64 * 0.90) as u64;
                    }
                } else {
                    let per_gpu: Vec<u64> = windows_per_gpu_vram(&windows_gpus);
                    let total: u64 = per_gpu.iter().sum();
                    survey.gpu_vram = per_gpu;
                    if total > 0 {
                        let ram_offload = system_ram.saturating_sub(total);
                        survey.vram_bytes = total + (ram_offload as f64 * 0.90) as u64;
                    } else if system_ram > 0 {
                        survey.vram_bytes = (system_ram as f64 * 0.90) as u64;
                    }
                }
            }

            if want_gpu_info {
                if let Some(ref names) = nvidia_names {
                    if metrics.contains(&Metric::GpuName) {
                        let name = summarize_gpu_name(names);
                        survey.gpu_name_source = name.is_some().then_some(GpuNameSource::NvidiaSmi);
                        survey.gpu_name = name;
                    }
                    if metrics.contains(&Metric::GpuCount) {
                        survey.gpu_count = u8::try_from(names.len()).unwrap_or(u8::MAX);
                    }
                } else {
                    let names: Vec<String> =
                        windows_gpus.iter().map(|(name, _)| name.clone()).collect();
                    if metrics.contains(&Metric::GpuName) {
                        let name = summarize_gpu_name(&names);
                        survey.gpu_name_source = name
                            .is_some()
                            .then_some(GpuNameSource::WindowsVideoController);
                        survey.gpu_name = name;
                    }
                    if metrics.contains(&Metric::GpuCount) {
                        survey.gpu_count = u8::try_from(names.len()).unwrap_or(u8::MAX);
                    }
                }
            }
        }

        survey
    }
}
