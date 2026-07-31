import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "hf-skippy-convert-job.py"
SPEC = importlib.util.spec_from_file_location("hf_skippy_convert_job", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class ConvertedArtifactValidationTests(unittest.TestCase):
    def write_artifact(self, root: Path, expected_splits: int) -> None:
        (root / "README.md").write_text("beta", encoding="utf-8")
        (root / "skippy-convert-manifest.json").write_text(
            json.dumps(
                {
                    "expected_splits": expected_splits,
                    "output_basename": "Inkling-BF16",
                }
            ),
            encoding="utf-8",
        )

    def test_accepts_every_declared_split(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            self.write_artifact(root, 2)
            (root / "Inkling-BF16-00001-of-00002.gguf").write_bytes(b"one")
            (root / "Inkling-BF16-00002-of-00002.gguf").write_bytes(b"two")

            MODULE.validate_converted_artifact(root)

    def test_rejects_interrupted_multi_split_conversion(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            self.write_artifact(root, 2)
            (root / "Inkling-BF16-00001-of-00002.gguf").write_bytes(b"one")

            with self.assertRaisesRegex(FileNotFoundError, "00002-of-00002"):
                MODULE.validate_converted_artifact(root)

    def test_accepts_declared_unsplit_output(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            self.write_artifact(root, 1)
            (root / "Inkling-BF16.gguf").write_bytes(b"one")

            MODULE.validate_converted_artifact(root)


if __name__ == "__main__":
    unittest.main()
