use crate::BenchmarkOutput;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BenchmarkBackend {
    Metal,
    Cuda,
    Hip,
    Intel,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BenchmarkRunner {
    pub backend: BenchmarkBackend,
}

pub fn runner_for(
    os: &str,
    gpu_count: u8,
    gpu_name: Option<&str>,
    is_soc: bool,
) -> Option<BenchmarkRunner> {
    if gpu_count == 0 {
        tracing::debug!("no GPUs detected; skipping benchmark");
        return None;
    }

    let gpu_upper = gpu_name.unwrap_or("").to_uppercase();

    if os == "macos" && is_soc {
        return Some(BenchmarkRunner {
            backend: BenchmarkBackend::Metal,
        });
    }

    if os == "linux" || os == "windows" {
        if gpu_upper.contains("NVIDIA")
            || gpu_upper.contains("ORIN")
            || gpu_upper.contains("NVGPU")
            || gpu_upper.contains("TEGRA")
        {
            return Some(BenchmarkRunner {
                backend: BenchmarkBackend::Cuda,
            });
        }

        if gpu_upper.contains("AMD") || gpu_upper.contains("RADEON") {
            return Some(BenchmarkRunner {
                backend: BenchmarkBackend::Hip,
            });
        }

        if gpu_upper.contains("INTEL") || gpu_upper.contains("ARC") {
            tracing::info!(
                "Intel GPU benchmark is not supported in standard mesh-llm builds; skipping"
            );
            return None;
        }

        if os == "linux" && is_soc {
            tracing::warn!("Jetson benchmark is unvalidated for ARM CUDA; attempting");
            return Some(BenchmarkRunner {
                backend: BenchmarkBackend::Cuda,
            });
        }
    }

    tracing::warn!("could not identify benchmark runner for GPU platform: {gpu_name:?}");
    None
}

pub fn parse_benchmark_output(stdout: &[u8]) -> Option<Vec<BenchmarkOutput>> {
    match serde_json::from_slice::<Vec<BenchmarkOutput>>(stdout) {
        Ok(outputs) if !outputs.is_empty() => Some(outputs),
        Ok(_) => {
            tracing::debug!("benchmark returned empty device list");
            None
        }
        Err(err) => {
            let error_message = serde_json::from_slice::<serde_json::Value>(stdout)
                .ok()
                .and_then(|val| {
                    val.get("error")
                        .and_then(|v| v.as_str())
                        .map(ToOwned::to_owned)
                });
            if let Some(msg) = error_message {
                tracing::warn!("benchmark reported error: {msg}");
                return None;
            }
            tracing::warn!("failed to parse benchmark output: {err}");
            None
        }
    }
}
