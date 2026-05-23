from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "skippy-llama-parity.py"


def load_module():
    spec = importlib.util.spec_from_file_location("skippy_llama_parity", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


class SkippyLlamaParityTests(unittest.TestCase):
    def setUp(self) -> None:
        self.parity = load_module()

    def test_resolves_first_gguf_shard_for_split_candidates(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            previous_cache = os.environ.get("HF_HUB_CACHE")
            cache_root = Path(tmp) / "hub"
            snapshot = (
                cache_root
                / "models--DevQuasar--CohereLabs.command-a-plus-05-2026-bf16-GGUF"
                / "snapshots"
                / "abc123"
            )
            snapshot.mkdir(parents=True)
            first_shard = (
                snapshot
                / "CohereLabs.command-a-plus-05-2026-bf16-Q4_K_M-00001-of-00009.gguf"
            )
            last_shard = (
                snapshot
                / "CohereLabs.command-a-plus-05-2026-bf16-Q4_K_M-00009-of-00009.gguf"
            )
            mmproj = snapshot / "mmproj-CohereLabs.command-a-plus-05-2026-bf16.gguf"
            first_shard.write_bytes(b"larger-first-shard")
            last_shard.write_bytes(b"x")
            mmproj.write_bytes(b"")
            os.environ["HF_HUB_CACHE"] = str(cache_root)
            try:
                resolved = self.parity.resolve_candidate_file(
                    {
                        "repo": "DevQuasar/CohereLabs.command-a-plus-05-2026-bf16-GGUF",
                        "include": "*Q4_K_M*.gguf",
                    }
                )
            finally:
                if previous_cache is None:
                    os.environ.pop("HF_HUB_CACHE", None)
                else:
                    os.environ["HF_HUB_CACHE"] = previous_cache

        self.assertEqual(resolved, first_shard)


if __name__ == "__main__":
    unittest.main()
