#!/usr/bin/env python3
"""Parallelism-budget contract tests for scripts/build-llama.sh.

The CUDA native-runtime lanes build inside a memory-limited container on a
many-core host. Sizing `cmake --build --parallel` from the CPU count alone
overcommits memory there and the host compiler dies with no diagnostic about
our source (gcc internal compiler error, or `cicc terminated (signal: 11)`).
These tests pin the budget logic so that regression cannot return silently.
"""

from __future__ import annotations

import os
import pathlib
import subprocess
import unittest


REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "build-llama.sh"

GIB = 1024 * 1024 * 1024


def print_jobs(
    *,
    backend: str,
    memory_bytes: int | str | None,
    parallel_override: str | None = None,
) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ)
    env["LLAMA_STAGE_BACKEND"] = backend
    env["LLAMA_STAGE_LINK_MODE"] = "dynamic"
    env.pop("CMAKE_BUILD_PARALLEL_LEVEL", None)
    if memory_bytes is not None:
        env["MESH_LLM_LLAMA_BUILD_MEMORY_BYTES"] = str(memory_bytes)
    if parallel_override is not None:
        env["CMAKE_BUILD_PARALLEL_LEVEL"] = parallel_override

    return subprocess.run(
        [str(SCRIPT), "--print-jobs"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )


def cpu_count() -> int:
    return os.cpu_count() or 1


class DetectJobsTests(unittest.TestCase):
    def test_cuda_parallelism_capped_by_memory_not_cpu_count(self) -> None:
        """6 GiB permits 2 CUDA compiles at the assumed 3 GiB each.

        The budget must bind below the CPU count even on a small runner, so
        this asserts a cap lower than any machine CI runs on.
        """
        result = print_jobs(backend="cuda", memory_bytes=6 * GIB)

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), "2")
        self.assertLess(2, cpu_count())

    def test_cap_is_reported_so_a_slow_build_is_explainable(self) -> None:
        result = print_jobs(backend="cuda", memory_bytes=4 * GIB)

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("capping llama.cpp build parallelism", result.stderr)

    def test_cpu_backend_gets_a_larger_budget_than_cuda(self) -> None:
        """CPU translation units are far cheaper, so they may run wider."""
        cuda = print_jobs(backend="cuda", memory_bytes=8 * GIB)
        cpu = print_jobs(backend="cpu", memory_bytes=8 * GIB)

        self.assertEqual(cuda.returncode, 0, cuda.stderr)
        self.assertEqual(cpu.returncode, 0, cpu.stderr)
        self.assertGreater(int(cpu.stdout.strip()), int(cuda.stdout.strip()))

    def test_cpu_count_wins_when_memory_is_plentiful(self) -> None:
        """The cap must not slow down a machine that has the headroom."""
        result = print_jobs(backend="cuda", memory_bytes=4096 * GIB)

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(int(result.stdout.strip()), cpu_count())
        self.assertNotIn("capping", result.stderr)

    def test_never_returns_zero_jobs(self) -> None:
        """A tiny budget must still produce a runnable build."""
        result = print_jobs(backend="cuda", memory_bytes=256 * 1024 * 1024)

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), "1")

    def test_detects_capacity_without_an_explicit_budget(self) -> None:
        """The real probe must yield a usable job count on this machine."""
        result = print_jobs(backend="cuda", memory_bytes=None)

        self.assertEqual(result.returncode, 0, result.stderr)
        jobs = int(result.stdout.strip())
        self.assertGreaterEqual(jobs, 1)
        self.assertLessEqual(jobs, cpu_count())

    def test_explicit_override_is_honored_verbatim(self) -> None:
        """CI must be able to pin parallelism regardless of detection."""
        result = print_jobs(
            backend="cuda",
            memory_bytes=16 * GIB,
            parallel_override="3",
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), "3")

    def test_rejects_a_non_numeric_memory_budget(self) -> None:
        result = print_jobs(backend="cuda", memory_bytes="lots")

        self.assertEqual(result.returncode, 2)
        self.assertIn("positive integer byte count", result.stderr)


if __name__ == "__main__":
    unittest.main()
