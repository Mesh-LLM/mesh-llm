from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "manage-build-cache.py"
SPEC = importlib.util.spec_from_file_location("manage_build_cache", SCRIPT)
assert SPEC and SPEC.loader
CACHE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CACHE)


class ManageBuildCacheTests(unittest.TestCase):
    def test_size_parser_accepts_binary_units(self) -> None:
        self.assertEqual(CACHE.parse_size("80GiB"), 80 * 1024**3)
        self.assertEqual(CACHE.parse_size("max_size=80GiB"), 80 * 1024**3)
        self.assertEqual(CACHE.parse_size("1.5 MiB"), int(1.5 * 1024**2))
        self.assertEqual(CACHE.parse_age("max_age=14"), 14)

    def test_status_emits_machine_readable_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            workspace = Path(temporary)
            artifact = workspace / "target" / "debug" / "deps" / "item"
            artifact.parent.mkdir(parents=True)
            artifact.write_bytes(b"x" * 128)
            result = subprocess.run(
                [sys.executable, str(SCRIPT), "status", "--workspace", str(workspace),
                 "--max-size", "64B", "--json"],
                check=False, capture_output=True, text=True,
            )
        self.assertEqual(result.returncode, 0, result.stderr)
        report = json.loads(result.stdout)
        self.assertEqual(report["schema"], "mesh-llm.local-build-cache")
        self.assertEqual(report["target_bytes"], 128)
        self.assertEqual(report["target_over_limit_bytes"], 64)

    def test_package_metrics_count_direct_dependency_files(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary) / "target"
            artifact = target / "debug" / "deps" / "mesh_llm-abc.rcgu.o"
            artifact.parent.mkdir(parents=True)
            artifact.write_bytes(b"x" * 256)
            metrics = CACHE.package_metrics(target, ["mesh-llm"])
        self.assertEqual(metrics[0]["package"], "mesh-llm")
        self.assertEqual(metrics[0]["bytes"], 256)

    def test_incremental_pruning_is_oldest_first_and_target_scoped(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary) / "target"
            old = target / "debug" / "incremental" / "old"
            fresh = target / "debug" / "incremental" / "fresh"
            old.mkdir(parents=True)
            fresh.mkdir()
            (old / "artifact").write_bytes(b"x" * 100)
            (fresh / "artifact").write_bytes(b"y" * 100)
            old_time = time.time() - 30 * 86400
            os.utime(old / "artifact", (old_time, old_time))
            os.utime(old, (old_time, old_time))
            remaining, actions = CACHE.prune_incremental(
                target, time.time() - 14 * 86400, 200, 150, True,
            )
            self.assertFalse(old.exists())
            self.assertTrue(fresh.exists())
            self.assertEqual(actions[0]["path"], str(old))
            self.assertEqual(remaining, 100)

    def test_execute_refuses_when_a_compiler_is_active(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            workspace = Path(temporary)
            (workspace / "target").mkdir()
            with mock.patch.object(CACHE, "active_compilers", return_value=["1 cargo"]):
                with mock.patch.object(
                    sys, "argv", [str(SCRIPT), "prune", "--workspace", str(workspace), "--execute"],
                ):
                    with self.assertRaises(CACHE.CacheError):
                        CACHE.main()


if __name__ == "__main__":
    unittest.main()
