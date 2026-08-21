#!/usr/bin/env bash
# detect-cuda-toolkit-version.sh — detect the installed CUDA *toolkit* version
# (the compiler that will actually run), for release-build-cuda's SM-list
# selection. This is a different axis than detect-cuda-arch.sh, which detects
# which SM the installed *GPU* needs — this detects which SM values the
# installed *nvcc* is able to target at all (pre-12.8 toolkits reject
# Blackwell's sm_100/103/120/121 outright).
#
# CI always sets MESH_CUDA_VERSION explicitly from its build matrix and never
# reaches this script. It exists so a bare `just release-build-cuda` run on a
# workstation (no MESH_CUDA_VERSION set) selects the SM list for the toolkit
# that is actually installed, instead of silently falling back to the
# pre-Blackwell list regardless of what's on the machine.
#
# Prints "<major>.<minor>" on stdout. Falls back to "12" (the historical,
# pre-Blackwell-safe default) when no toolkit can be detected, so callers
# never have to special-case failure.

set -uo pipefail

# ── Strategy 1: nvcc (the compiler that will actually run) ────────────────
# build-llama.sh honors CUDACXX/CMAKE_CUDA_COMPILER to pick a specific nvcc
# on hosts with multiple toolkits installed side by side (a carrack host has
# 12.x and 13.x toolkits at once); check the same override here so the
# detected version always matches the compiler that will actually run,
# falling back to whatever `nvcc` resolves to on PATH.
for NVCC in "${CUDACXX:-}" "${NVCC:-}" nvcc; do
    [[ -n "$NVCC" ]] || continue
    if command -v "$NVCC" &>/dev/null; then
        VER="$("$NVCC" --version 2>/dev/null | grep -oP 'release \K[0-9]+\.[0-9]+')"
        if [[ -n "$VER" ]]; then
            echo "$VER"
            exit 0
        fi
    fi
done

# ── Strategy 2: CUDA toolkit install metadata ──────────────────────────────
for CANDIDATE in /usr/local/cuda /opt/cuda; do
    if [[ -f "$CANDIDATE/version.json" ]]; then
        VER="$(grep -oP '"version"\s*:\s*"\K[0-9]+\.[0-9]+' "$CANDIDATE/version.json" | head -1)"
        if [[ -n "$VER" ]]; then
            echo "$VER"
            exit 0
        fi
    fi
    if [[ -f "$CANDIDATE/version.txt" ]]; then
        VER="$(grep -oP 'CUDA Version \K[0-9]+\.[0-9]+' "$CANDIDATE/version.txt" | head -1)"
        if [[ -n "$VER" ]]; then
            echo "$VER"
            exit 0
        fi
    fi
done

# ── Strategy 3: nvidia-smi's reported CUDA version ─────────────────────────
# This is the max version the *driver* supports, not proof a matching
# toolkit is installed, so it's a last resort ahead of the static fallback.
if command -v nvidia-smi &>/dev/null; then
    VER="$(nvidia-smi 2>/dev/null | grep -oP 'CUDA (UMD )?Version:\s*\K[0-9]+\.[0-9]+' | head -1)"
    if [[ -n "$VER" ]]; then
        echo "$VER"
        exit 0
    fi
fi

# ── All strategies exhausted: keep the historical, pre-Blackwell-safe default ──
echo "12"
