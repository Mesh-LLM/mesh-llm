use crate::{BenchmarkOutput, capture::capture_stdout, parse_benchmark_output};
use anyhow::{Context, Result};
use std::ffi::c_int;

unsafe extern "C" {
    fn mesh_llm_gpu_bench_cuda_main() -> c_int;
}

pub fn run() -> Result<Vec<BenchmarkOutput>> {
    let stdout = capture_stdout(mesh_llm_gpu_bench_cuda_main)?;
    parse_benchmark_output(&stdout).context("CUDA benchmark returned invalid output")
}
