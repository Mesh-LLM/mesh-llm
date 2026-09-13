from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path
import subprocess
import tempfile
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "llama-oracle-source.py"
SPEC = importlib.util.spec_from_file_location("llama_oracle_source", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
source = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(source)


class LlamaOracleSourceTests(unittest.TestCase):
    def test_digest_includes_generated_series_in_prepare_order(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            patches = Path(temp_dir)
            (patches / "0001-base.patch").write_bytes(b"base\n")
            generated = patches / "generated"
            generated.mkdir()
            (generated / "series").write_text("0001-family-test.patch\n", encoding="utf-8")
            (generated / "0001-family-test.patch").write_bytes(b"generated\n")
            expected = hashlib.sha256()
            for name, content in (
                ("0001-base.patch", b"base\n"),
                ("generated/0001-family-test.patch", b"generated\n"),
            ):
                expected.update(
                    f"{name}\n{hashlib.sha256(content).hexdigest()}\n".encode("utf-8")
                )
            self.assertEqual(expected.hexdigest(), source.patch_digest(patches))
            (generated / "series").write_text("0002-family-test.patch\n", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "invalid generated patch sequence"):
                source.patch_digest(patches)

    def test_prepared_checkout_rejects_patch_queue_drift(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            checkout = root / ".deps/llama.cpp"
            checkout.mkdir(parents=True)
            patches = root / "third_party/llama.cpp/patches"
            patches.mkdir(parents=True)
            patch = patches / "0001-test.patch"
            patch.write_bytes(b"original\n")
            upstream = root / "third_party/llama.cpp/upstream.txt"
            upstream.write_text("upstream-sha\n", encoding="utf-8")
            subprocess.run(["git", "init", "-q", str(checkout)], check=True)
            head = subprocess.run(
                ["git", "-C", str(checkout), "rev-parse", "HEAD"],
                capture_output=True, text=True, check=False,
            )
            if head.returncode != 0:
                subprocess.run(
                    ["git", "-C", str(checkout), "-c", "user.name=Oracle Test",
                     "-c", "user.email=oracle@example.invalid", "commit", "--allow-empty",
                     "-qm", "initial"],
                    check=True,
                )
            head_sha = subprocess.run(
                ["git", "-C", str(checkout), "rev-parse", "HEAD"],
                check=True, capture_output=True, text=True,
            ).stdout.strip()
            (checkout / ".mesh-llm-upstream-sha").write_text("upstream-sha\n", encoding="utf-8")
            (checkout / ".mesh-llm-patch-digest").write_text(
                source.patch_digest(patches) + "\n", encoding="utf-8"
            )
            (checkout / ".mesh-llm-patched-sha").write_text(head_sha + "\n", encoding="utf-8")
            (checkout / ".mesh-llm-prepare-schema").write_text("4\n", encoding="utf-8")
            self.assertEqual(head_sha, source.prepared_patched_sha(root))
            patch.write_bytes(b"changed\n")
            with self.assertRaisesRegex(RuntimeError, "does not match the current patch queue"):
                source.prepared_patched_sha(root)


if __name__ == "__main__":
    unittest.main()
