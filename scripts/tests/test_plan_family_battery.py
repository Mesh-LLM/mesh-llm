from __future__ import annotations

import copy
import json
from pathlib import Path
import struct
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
PLANNER = ROOT / "scripts" / "plan-family-battery.py"
MANIFEST = ROOT / "ci" / "llama-canary" / "family-certified.json"


class FamilyBatteryPlannerTests(unittest.TestCase):
    @staticmethod
    def _write_gguf(
        path: Path, block_count: int, embedding_length: int = 1024
    ) -> None:
        def gguf_string(value: str) -> bytes:
            encoded = value.encode("utf-8")
            return struct.pack("<Q", len(encoded)) + encoded

        payload = bytearray(b"GGUF")
        payload.extend(struct.pack("<IQQ", 3, 0, 3))
        payload.extend(gguf_string("general.architecture"))
        payload.extend(struct.pack("<I", 8))
        payload.extend(gguf_string("fixture"))
        payload.extend(gguf_string("fixture.block_count"))
        payload.extend(struct.pack("<II", 4, block_count))
        payload.extend(gguf_string("fixture.embedding_length"))
        payload.extend(struct.pack("<II", 4, embedding_length))
        path.write_bytes(payload)

    def _run(
        self, manifest: Path = MANIFEST, *args: str
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [str(PLANNER), "--manifest", str(manifest), *args],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )

    def test_checked_in_policy_resolves_all_certified_models(self) -> None:
        result = self._run()
        self.assertEqual(0, result.returncode, result.stderr)
        plan = json.loads(result.stdout)
        self.assertEqual(32, plan["selected_family_count"])
        self.assertEqual(
            ["single-step", "chain", "dtype-matrix", "state-handoff"],
            plan["required_certification_lanes"],
        )
        glm47 = next(
            model
            for model in plan["selected_models"]
            if model["family"] == "glm47-flash"
        )
        self.assertEqual(47, glm47["execution"]["layer_end"])
        self.assertEqual(0, glm47["execution"]["mtp_layers"])
        by_family = {model["family"]: model for model in plan["selected_models"]}
        expected_ranges = {
            "deepseek2": 27,
            "qwen3-moe": 48,
            "kimi-linear": 27,
            "mamba2": 64,
            "laguna": 40,
        }
        for family, layer_end in expected_ranges.items():
            with self.subTest(family=family):
                self.assertEqual(layer_end, by_family[family]["execution"]["layer_end"])
        self.assertEqual(4096, by_family["qwen3-vl"]["execution"]["activation_width"])
        self.assertEqual(600, by_family["qwen3-vl"]["resources"]["startup_timeout_secs"])

    def test_certified_model_requires_an_explicit_activation_width(self) -> None:
        manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
        del manifest["models"][0]["execution"]["activation_width"]
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "manifest.json"
            path.write_text(json.dumps(manifest), encoding="utf-8")
            result = self._run(path)
        self.assertEqual(2, result.returncode)
        self.assertIn("activation_width must be an integer", result.stderr)

    def test_certified_profile_cannot_drop_a_core_lane(self) -> None:
        manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
        manifest["policy"]["profiles"]["full"]["required_lanes"].remove(
            "state-handoff"
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "manifest.json"
            path.write_text(json.dumps(manifest), encoding="utf-8")
            result = self._run(path)
        self.assertEqual(2, result.returncode)
        self.assertIn("must require exactly the four core lanes", result.stderr)

    def test_certified_profile_cannot_add_or_reorder_core_lanes(self) -> None:
        manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
        lanes = manifest["policy"]["profiles"]["package-oracle"]["required_lanes"]
        lanes.reverse()
        lanes.append("graph-parse")
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "manifest.json"
            path.write_text(json.dumps(manifest), encoding="utf-8")
            result = self._run(path)
        self.assertEqual(2, result.returncode)
        self.assertIn("must require exactly the four core lanes", result.stderr)

    def test_duplicate_family_is_rejected(self) -> None:
        manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
        manifest["models"].append(copy.deepcopy(manifest["models"][0]))
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "manifest.json"
            path.write_text(json.dumps(manifest), encoding="utf-8")
            result = self._run(path)
        self.assertEqual(2, result.returncode)
        self.assertIn("duplicate family", result.stderr)

    def test_cache_gate_requires_every_exact_revision_file(self) -> None:
        source = json.loads(MANIFEST.read_text(encoding="utf-8"))
        source["models"] = [copy.deepcopy(source["models"][0])]
        model = source["models"][0]
        model["execution"]["trunk_layers"] = 3
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            manifest = root / "manifest.json"
            manifest.write_text(json.dumps(source), encoding="utf-8")
            artifact = model["artifact"]
            cached = (
                root
                / "cache"
                / "hub"
                / ("models--" + artifact["repo"].replace("/", "--"))
                / "snapshots"
                / artifact["revision"]
                / artifact["files"][0]
            )
            cached.parent.mkdir(parents=True)
            self._write_gguf(cached, 3)
            present = self._run(
                manifest,
                "--check-cache",
                "--cache-root",
                str(root / "cache"),
            )
            self.assertEqual(0, present.returncode, present.stderr)
            cached.unlink()
            missing = self._run(
                manifest,
                "--check-cache",
                "--cache-root",
                str(root / "cache"),
            )
        self.assertEqual(2, missing.returncode)
        self.assertIn("immutable family cache is incomplete", missing.stderr)
        self.assertIn(model["family"], missing.stderr)

    def test_cache_gate_rejects_runtime_range_drift_before_build(self) -> None:
        source = json.loads(MANIFEST.read_text(encoding="utf-8"))
        source["models"] = [copy.deepcopy(source["models"][0])]
        model = source["models"][0]
        model["execution"]["trunk_layers"] = 3
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            manifest = root / "manifest.json"
            manifest.write_text(json.dumps(source), encoding="utf-8")
            artifact = model["artifact"]
            cached = (
                root
                / "cache"
                / "hub"
                / ("models--" + artifact["repo"].replace("/", "--"))
                / "snapshots"
                / artifact["revision"]
                / artifact["files"][0]
            )
            cached.parent.mkdir(parents=True)
            self._write_gguf(cached, 4)
            result = self._run(
                manifest,
                "--check-cache",
                "--cache-root",
                str(root / "cache"),
            )
        self.assertEqual(2, result.returncode)
        self.assertIn("plans 3 runtime layers", result.stderr)
        self.assertIn("declares 4", result.stderr)

    def test_cache_gate_rejects_activation_width_drift_before_build(self) -> None:
        source = json.loads(MANIFEST.read_text(encoding="utf-8"))
        source["models"] = [copy.deepcopy(source["models"][0])]
        model = source["models"][0]
        model["execution"]["trunk_layers"] = 3
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            manifest = root / "manifest.json"
            manifest.write_text(json.dumps(source), encoding="utf-8")
            artifact = model["artifact"]
            cached = (
                root
                / "cache"
                / "hub"
                / ("models--" + artifact["repo"].replace("/", "--"))
                / "snapshots"
                / artifact["revision"]
                / artifact["files"][0]
            )
            cached.parent.mkdir(parents=True)
            self._write_gguf(cached, 3, 2048)
            result = self._run(
                manifest,
                "--check-cache",
                "--cache-root",
                str(root / "cache"),
            )
        self.assertEqual(2, result.returncode)
        self.assertIn("plans activation width 1024", result.stderr)
        self.assertIn("declares 2048", result.stderr)

    def test_shards_are_deterministic_and_preserve_every_family_once(self) -> None:
        first = self._run(MANIFEST, "--shard-count", "4")
        second = self._run(MANIFEST, "--shard-count", "4")
        self.assertEqual(0, first.returncode, first.stderr)
        self.assertEqual(first.stdout, second.stdout)
        plan = json.loads(first.stdout)
        families = [
            family for shard in plan["shards"] for family in shard["families"]
        ]
        self.assertEqual(32, len(families))
        self.assertEqual(32, len(set(families)))
        self.assertEqual(4, len(plan["github_matrix"]["include"]))


if __name__ == "__main__":
    unittest.main()
