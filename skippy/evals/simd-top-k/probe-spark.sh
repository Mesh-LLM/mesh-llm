#!/usr/bin/env bash
set -euo pipefail

if [[ $# != 2 ]]; then
    echo 'usage: probe-spark.sh MODEL_FILE ARTIFACT_DIRECTORY' >&2
    exit 2
fi

model_file=$1
artifact_directory=$2
mkdir -p "$artifact_directory"
test "$(uname -m)" = aarch64
{
    git rev-parse HEAD
    uname -srm
    lscpu
    cat /etc/os-release
    free -b
    cmake --version
    ninja --version
    c++ --version
    if command -v nvcc >/dev/null; then
        nvcc --version
    else
        echo 'nvcc not on PATH; inspecting installed CUDA toolchains'
        for compiler_path in /usr/local/cuda*/bin/nvcc; do
            if [[ -x "$compiler_path" ]]; then printf '%s\n' "$compiler_path"; "$compiler_path" --version; fi
        done
    fi
    nvidia-smi --query-gpu=name,driver_version,compute_cap --format=csv,noheader
    printf 'model_size_bytes='
    stat -c %s "$model_file"
} > "$artifact_directory/toolchain.txt" 2>&1
test "$(stat -c %s "$model_file")" = 5680522464
printf '%s  %s\n' 03b74727a860a56338e042c4420bb3f04b2fec5734175f4cb9fa853daf52b7e8 "$model_file" | sha256sum -c - > "$artifact_directory/model-hash.txt"
nvidia-smi -q -d TEMPERATURE,PERFORMANCE,CLOCK,POWER > "$artifact_directory/gpu-telemetry.txt"
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader > "$artifact_directory/gpu-processes.txt"
{
    if command -v perf >/dev/null; then
        if perf --version; then echo 'perf executable'; else echo 'perf wrapper unavailable for this kernel'; fi
    else
        echo 'perf unavailable'
    fi
    if [[ -r /proc/sys/kernel/perf_event_paranoid ]]; then cat /proc/sys/kernel/perf_event_paranoid; fi
    for thermal_path in /sys/class/thermal/thermal_zone*/type /sys/class/thermal/thermal_zone*/temp /sys/devices/system/cpu/cpu*/thermal_throttle/*_throttle_count; do
        if [[ -r "$thermal_path" ]]; then printf '%s=' "$thermal_path"; cat "$thermal_path"; fi
    done
} > "$artifact_directory/cpu-telemetry-capabilities.txt"
printf 'Prerequisite probe only: no model generation or qualification timings.\n' > "$artifact_directory/scope.txt"
