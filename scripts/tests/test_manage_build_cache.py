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

    def test_package_metrics_count_cross_target_debug_files(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary) / "target"
            artifact = (
                target / "x86_64-unknown-linux-gnu" / "debug" / "deps"
                / "mesh_llm-abc.rcgu.o"
            )
            artifact.parent.mkdir(parents=True)
            artifact.write_bytes(b"x" * 384)
            metrics = CACHE.package_metrics(target, ["mesh-llm"])
        self.assertEqual(metrics[0]["package"], "mesh-llm")
        self.assertEqual(metrics[0]["bytes"], 384)

    def test_package_metrics_count_hyphenated_build_root(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary) / "target"
            artifact = target / "debug" / "build" / "mesh-llm-abc" / "output"
            artifact.parent.mkdir(parents=True)
            artifact.write_bytes(b"x" * 512)
            metrics = CACHE.package_metrics(target, ["mesh-llm"])
        self.assertEqual(metrics[0]["package"], "mesh-llm")
        self.assertEqual(metrics[0]["bytes"], 512)

    def test_separate_cargo_build_directory_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            workspace = Path(temporary)
            (workspace / "Cargo.toml").write_text("[workspace]\n", encoding="utf-8")
            metadata = {
                "target_directory": str(workspace / "target"),
                "build_directory": str(workspace / "build-artifacts"),
            }
            with mock.patch.object(CACHE, "cargo_metadata", return_value=metadata):
                with self.assertRaisesRegex(CACHE.CacheError, "build.build-dir"):
                    CACHE.reject_separate_build_directory(workspace, workspace / "target")

    def test_cargo_build_directory_environment_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            with mock.patch.dict(os.environ, {"CARGO_BUILD_BUILD_DIR": "elsewhere"}):
                with self.assertRaisesRegex(CACHE.CacheError, "CARGO_BUILD_BUILD_DIR"):
                    workspace = Path(temporary)
                    CACHE.reject_separate_build_directory(workspace, workspace / "target")

    def test_cargo_target_directory_environment_mismatch_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            workspace = Path(temporary)
            configured_target = workspace / "configured-target"
            (workspace / "Cargo.toml").write_text("[workspace]\n", encoding="utf-8")
            metadata = {
                "target_directory": str(configured_target),
                "build_directory": str(configured_target),
            }
            with mock.patch.dict(
                os.environ, {"CARGO_TARGET_DIR": str(configured_target)}, clear=False,
            ):
                with mock.patch.object(CACHE, "cargo_metadata", return_value=metadata):
                    with self.assertRaisesRegex(CACHE.CacheError, "effective target"):
                        CACHE.reject_separate_build_directory(
                            workspace, workspace / "target",
                        )

    def test_explicit_target_directory_must_match_cargo_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            workspace = Path(temporary)
            cargo_target = workspace / "target"
            explicit_target = workspace / "other-target"
            (workspace / "Cargo.toml").write_text("[workspace]\n", encoding="utf-8")
            metadata = {
                "target_directory": str(cargo_target),
                "build_directory": str(cargo_target),
            }
            with mock.patch.object(CACHE, "cargo_metadata", return_value=metadata):
                with mock.patch.object(
                    sys,
                    "argv",
                    [
                        str(SCRIPT), "status", "--workspace", str(workspace),
                        "--target-dir", str(explicit_target),
                    ],
                ):
                    with self.assertRaisesRegex(CACHE.CacheError, "effective target"):
                        CACHE.main()

    def test_explicit_target_directory_matching_cargo_metadata_is_valid(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            workspace = Path(temporary)
            explicit_target = workspace / "configured-target"
            (workspace / "Cargo.toml").write_text("[workspace]\n", encoding="utf-8")
            metadata = {
                "target_directory": str(explicit_target),
                "build_directory": str(explicit_target),
            }
            with mock.patch.object(CACHE, "cargo_metadata", return_value=metadata):
                CACHE.reject_separate_build_directory(workspace, explicit_target)

    def test_cargo_operations_use_just_recipes(self) -> None:
        workspace = Path("/workspace")
        metadata_result = subprocess.CompletedProcess(
            ["just", "cache-cargo-metadata"], 0, stdout='{"packages": []}', stderr="",
        )
        clean_result = subprocess.CompletedProcess(
            ["just", "cache-cargo-clean"], 0,
        )
        with mock.patch.object(
            CACHE.subprocess, "run", side_effect=[metadata_result, clean_result],
        ) as run:
            self.assertEqual(CACHE.cargo_metadata(workspace), {"packages": []})
            with mock.patch.object(
                CACHE, "package_metrics",
                return_value=[{"package": "mesh-llm", "bytes": 1, "newest_mtime": 0}],
            ):
                with mock.patch.object(CACHE, "cargo_packages", return_value=["mesh-llm"]):
                    with mock.patch.object(CACHE, "tree_metrics", return_value=(0, 0)):
                        CACHE.prune_packages(workspace, workspace / "target", 1, 0, 0, True)
        self.assertEqual(run.call_args_list[0].args[0], ["just", "cache-cargo-metadata"])
        self.assertEqual(
            run.call_args_list[1].args[0],
            ["just", "cache-cargo-clean"],
        )
        clean_environment = run.call_args_list[1].kwargs["env"]
        self.assertEqual(
            clean_environment["MESH_LLM_CACHE_TARGET_DIR"], "/workspace/target",
        )
        self.assertEqual(clean_environment["MESH_LLM_CACHE_PACKAGE"], "mesh-llm")

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

    def test_incremental_pruning_finds_cross_target_debug_profile(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary) / "target"
            old = (
                target / "x86_64-unknown-linux-gnu" / "debug" / "incremental" / "old"
            )
            old.mkdir(parents=True)
            (old / "artifact").write_bytes(b"x" * 100)
            old_time = time.time() - 30 * 86400
            os.utime(old / "artifact", (old_time, old_time))
            os.utime(old, (old_time, old_time))
            remaining, actions = CACHE.prune_incremental(
                target, time.time() - 14 * 86400, 100, 100, True,
            )
            self.assertFalse(old.exists())
            self.assertEqual(actions[0]["path"], str(old))
            self.assertEqual(remaining, 0)

    def test_remove_tree_unlinks_symlink_without_deleting_target(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary) / "target"
            retained = target / "debug" / "incremental" / "retained"
            retained.mkdir(parents=True)
            (retained / "artifact").write_bytes(b"keep")
            candidate = target / "debug" / "incremental" / "candidate"
            candidate.symlink_to(retained, target_is_directory=True)
            CACHE.remove_tree(candidate, target)
            self.assertFalse(candidate.exists())
            self.assertTrue((retained / "artifact").exists())

    def test_build_command_holds_shared_cache_lock(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            workspace = Path(temporary)
            with mock.patch.object(CACHE, "cache_lock") as cache_lock:
                cache_lock.return_value.__enter__.return_value = None
                cache_lock.return_value.__exit__.return_value = None
                with mock.patch.object(
                    sys,
                    "argv",
                    [str(SCRIPT), "build", "--workspace", str(workspace), "--", "true"],
                ):
                    self.assertEqual(CACHE.main(), 0)
            cache_lock.assert_called_once_with(
                workspace.resolve() / "target", exclusive=False, nonblocking=False,
            )

    def test_shared_build_lock_excludes_pruning(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary) / "target"
            with CACHE.cache_lock(target, exclusive=False, nonblocking=False):
                with self.assertRaises(CACHE.CacheError):
                    with CACHE.cache_lock(target, exclusive=True, nonblocking=True):
                        self.fail("exclusive prune lock unexpectedly acquired")

    def test_status_holds_nonblocking_shared_cache_lock(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            workspace = Path(temporary)
            with mock.patch.object(CACHE, "cache_lock") as cache_lock:
                cache_lock.return_value.__enter__.return_value = None
                cache_lock.return_value.__exit__.return_value = None
                with mock.patch.object(CACHE, "render_status"):
                    with mock.patch.object(
                        sys, "argv", [str(SCRIPT), "status", "--workspace", str(workspace)],
                    ):
                        self.assertEqual(CACHE.main(), 0)
            cache_lock.assert_called_once_with(
                workspace.resolve() / "target", exclusive=False, nonblocking=True,
            )

    def test_dry_run_holds_nonblocking_shared_cache_lock(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            workspace = Path(temporary)
            with mock.patch.object(CACHE, "cache_lock") as cache_lock:
                cache_lock.return_value.__enter__.return_value = None
                cache_lock.return_value.__exit__.return_value = None
                with mock.patch.object(CACHE, "run_prune", return_value=0):
                    with mock.patch.object(
                        sys, "argv", [str(SCRIPT), "prune", "--workspace", str(workspace)],
                    ):
                        self.assertEqual(CACHE.main(), 0)
            cache_lock.assert_called_once_with(
                workspace.resolve() / "target", exclusive=False, nonblocking=True,
            )

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

    def test_execute_acquires_exclusive_lock_before_compiler_check(self) -> None:
        events = []

        class RecordingLock:
            def __enter__(self):
                events.append("lock")

            def __exit__(self, *_args):
                events.append("unlock")

        with tempfile.TemporaryDirectory() as temporary:
            workspace = Path(temporary)
            with mock.patch.object(CACHE, "cache_lock", return_value=RecordingLock()):
                with mock.patch.object(
                    CACHE,
                    "active_compilers",
                    side_effect=lambda: events.append("compiler-check") or ["1 cargo"],
                ):
                    with mock.patch.object(
                        sys,
                        "argv",
                        [str(SCRIPT), "prune", "--workspace", str(workspace), "--execute"],
                    ):
                        with self.assertRaises(CACHE.CacheError):
                            CACHE.main()
        self.assertEqual(events, ["lock", "compiler-check", "unlock"])


if __name__ == "__main__":
    unittest.main()
