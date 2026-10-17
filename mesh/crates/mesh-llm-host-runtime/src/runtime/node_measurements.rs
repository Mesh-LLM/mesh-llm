//! Hardware measurements a serving node gossips: GPU memory bandwidth and
//! compute from the native benchmark, and random-read throughput of the
//! local model store for split planning's expert tiering.

use crate::mesh;
use crate::models;
use crate::system::benchmark;
use crate::system::hardware;
use std::path::Path;

pub(super) async fn store_benchmark_metrics(
    mem_arc: std::sync::Arc<tokio::sync::Mutex<Option<Vec<f64>>>>,
    fp32_arc: std::sync::Arc<tokio::sync::Mutex<Option<Vec<f64>>>>,
    fp16_arc: std::sync::Arc<tokio::sync::Mutex<Option<Vec<f64>>>>,
    result: Option<&benchmark::BenchmarkResult>,
) {
    *mem_arc.lock().await = result.map(|r| r.mem_bandwidth_gbps.clone());
    *fp32_arc.lock().await = result.and_then(|r| r.compute_tflops_fp32.clone());
    *fp16_arc.lock().await = result.and_then(|r| r.compute_tflops_fp16.clone());
}

/// Lift the soft open-file limit to the hard limit. A serving node holds a
/// socket per lane, per stage bridge relay and per client, and macOS starts
/// processes at a soft limit as low as 256, which a busy split exhausts —
/// after which accepts fail and stage bridges stop taking connections.
fn raise_open_file_limit() {
    #[cfg(unix)]
    {
        let mut limit = libc::rlimit {
            rlim_cur: 0,
            rlim_max: 0,
        };
        // SAFETY: getrlimit/setrlimit read and write a caller-owned struct.
        if unsafe { libc::getrlimit(libc::RLIMIT_NOFILE, &mut limit) } != 0 {
            return;
        }
        // macOS rejects RLIM_INFINITY for NOFILE; OPEN_MAX-style caps apply.
        let target = if limit.rlim_max == libc::RLIM_INFINITY {
            65_536
        } else {
            limit.rlim_max.min(1_048_576)
        };
        if limit.rlim_cur >= target {
            return;
        }
        let previous = limit.rlim_cur;
        limit.rlim_cur = target;
        // SAFETY: as above.
        if unsafe { libc::setrlimit(libc::RLIMIT_NOFILE, &limit) } == 0 {
            tracing::info!(previous, raised_to = target, "raised open-file limit");
        } else {
            limit.rlim_cur = 10_240.max(previous);
            // SAFETY: as above; fall back to a limit macOS always accepts.
            let _ = unsafe { libc::setrlimit(libc::RLIMIT_NOFILE, &limit) };
        }
    }
}

pub(super) fn spawn_node_benchmark_task(node: &mesh::Node, bin_dir: &Path) {
    let mem_arc = node.gpu_mem_bandwidth_gbps.clone();
    let compute_fp32_arc = node.gpu_compute_tflops_fp32.clone();
    let compute_fp16_arc = node.gpu_compute_tflops_fp16.clone();
    let bin_dir_clone = bin_dir.to_path_buf();
    let node_bench = node.clone();
    tokio::spawn(async move {
        let result = tokio::time::timeout(
            std::time::Duration::from_secs(30),
            tokio::task::spawn_blocking(move || {
                let hw = hardware::survey();
                if hw.gpu_count == 0 {
                    tracing::debug!("no GPUs detected — skipping memory bandwidth benchmark");
                    return None;
                }
                benchmark::run_or_load(&hw, &bin_dir_clone, benchmark::BENCHMARK_TIMEOUT)
            }),
        )
        .await
        .map_err(|_| {
            tracing::warn!("benchmark timed out after 30s — bandwidth will not be gossiped")
        })
        .ok()
        .and_then(|r| r.ok())
        .flatten();

        if let Some(ref run) = result {
            let total: f64 = run.mem_bandwidth_gbps.iter().sum();
            tracing::info!(
                "Memory bandwidth fingerprint: {} GPUs, {:.1} GB/s total",
                run.mem_bandwidth_gbps.len(),
                total
            );
            for (i, gbps) in run.mem_bandwidth_gbps.iter().enumerate() {
                tracing::debug!("  GPU {}: {:.1} GB/s", i, gbps);
            }
            if let Some(fp32s) = &run.compute_tflops_fp32 {
                let total_fp32: f64 = fp32s.iter().sum();
                tracing::info!(
                    "Compute FP32 TFLOPS: {} GPUs, {:.1} TFLOPS total",
                    fp32s.len(),
                    total_fp32
                );
                for (i, tf) in fp32s.iter().enumerate() {
                    tracing::debug!("  GPU {}: {:.1} TF32", i, tf);
                }
            }
            if let Some(fp16s) = &run.compute_tflops_fp16 {
                let total_fp16: f64 = fp16s.iter().sum();
                tracing::info!(
                    "Compute FP16 TFLOPS: {} GPUs, {:.1} TFLOPS total",
                    fp16s.len(),
                    total_fp16
                );
                for (i, tf) in fp16s.iter().enumerate() {
                    tracing::debug!("  GPU {}: {:.1} TF16", i, tf);
                }
            }
        }
        store_benchmark_metrics(
            mem_arc.clone(),
            compute_fp32_arc.clone(),
            compute_fp16_arc.clone(),
            result.as_ref(),
        )
        .await;
        node_bench.regossip().await;
    });
}

/// Start the hardware measurements a serving node gossips: model-store read
/// throughput and, when a native runtime is loaded, memory bandwidth.
pub(super) fn spawn_node_measurement_tasks(node: &mesh::Node, bin_dir: &Path, is_client: bool) {
    if is_client {
        tracing::debug!("client node — skipping memory bandwidth benchmark");
        return;
    }
    spawn_model_store_probe_task(node);
    if skippy_runtime::native_runtime_loaded() {
        spawn_node_benchmark_task(node, bin_dir);
        raise_open_file_limit();
    } else {
        tracing::debug!("native runtime not loaded — skipping memory bandwidth benchmark");
    }
}

/// Measure how fast the local model store serves random reads and gossip it,
/// so split planning can price routed experts that page in from this disk.
pub(super) fn spawn_model_store_probe_task(node: &mesh::Node) {
    let rate_arc = node.model_store_read_bytes_per_second.clone();
    let node_probe = node.clone();
    tokio::spawn(async move {
        let store = models::catalog::models_dir();
        let measured = tokio::task::spawn_blocking(move || {
            crate::system::storage_probe::measure_model_store_read_rate(&store)
        })
        .await
        .ok()
        .flatten();
        let Some(rate) = measured else {
            tracing::debug!("model store has no file large enough to probe read throughput");
            return;
        };
        tracing::info!(
            bytes_per_second = rate.bytes_per_second,
            bytes_read = rate.bytes_read,
            elapsed_ms = rate.elapsed.as_millis() as u64,
            "model store random-read throughput: {:.2} GB/s",
            rate.bytes_per_second as f64 / 1e9
        );
        *rate_arc.lock().await = Some(rate.bytes_per_second);
        node_probe.regossip().await;
    });
}
