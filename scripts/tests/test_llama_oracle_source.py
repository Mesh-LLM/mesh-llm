from __future__ import annotations

import hashlib
import importlib.util
import re
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
    def test_oracle_and_prepare_require_the_same_schema(self) -> None:
        """A native preparation format change must also update oracle provenance checks."""
        prepare = (SCRIPT.parent / "prepare-llama.sh").read_text(encoding="utf-8")
        schema = re.search(r"^PREPARE_SCHEMA=(\d+)$", prepare, re.MULTILINE)
        self.assertIsNotNone(schema)
        self.assertIn(f'prepared_schema != "{schema.group(1)}"', SCRIPT.read_text(encoding="utf-8"))

    def test_model_support_series_rejects_incomplete_or_unsafe_inputs(self) -> None:
        """Fail closed on missing manifests, orphan patches, and invalid lane sequences."""
        for manifest in (None, "", "../escape.patch\n", "0002-test.patch\n",
                         "0001-test.patch\n0001-test.patch\n", "0001-missing.patch\n"):
            with self.subTest(manifest=manifest), tempfile.TemporaryDirectory() as temp_dir:
                patches = Path(temp_dir)
                support = patches / "model_support"
                support.mkdir()
                (support / "0001-test.patch").write_bytes(b"support\n")
                if manifest is not None:
                    (support / "series").write_text(manifest, encoding="utf-8")
                with self.assertRaises(RuntimeError):
                    source.ordered_patches(patches)

    def test_digest_includes_generated_series_in_prepare_order(self) -> None:
        """Include every patch lane, with model support preceding generated families."""
        with tempfile.TemporaryDirectory() as temp_dir:
            patches = Path(temp_dir)
            (patches / "0001-base.patch").write_bytes(b"base\n")
            support = patches / "model_support"
            support.mkdir()
            (support / "series").write_bytes(b"0001-test-support.patch\r\n")
            (support / "0001-test-support.patch").write_bytes(b"support\n")
            generated = patches / "generated"
            generated.mkdir()
            (generated / "series").write_text("0001-family-test.patch\n", encoding="utf-8")
            (generated / "0001-family-test.patch").write_bytes(b"generated\n")
            expected = hashlib.sha256()
            for name, content in (
                ("0001-base.patch", b"base\n"),
                ("model_support/0001-test-support.patch", b"support\n"),
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
        """Reject an oracle checkout after its source patch queue changes."""
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
            schema = checkout / ".mesh-llm-prepare-schema"
            schema.write_text("4\n", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "does not match the pinned upstream"):
                source.prepared_patched_sha(root)
            schema.write_text("5\n", encoding="utf-8")
            self.assertEqual(head_sha, source.prepared_patched_sha(root))
            patch.write_bytes(b"changed\n")
            with self.assertRaisesRegex(RuntimeError, "does not match the current patch queue"):
                source.prepared_patched_sha(root)


if __name__ == "__main__":
    unittest.main()
