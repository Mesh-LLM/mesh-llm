from __future__ import annotations

import pathlib
import subprocess
import tempfile
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "check-env-mutation-contract.py"
AUDITED_FILE = "crates/model-hf/src/store/local.rs"


class EnvironmentMutationContractTests(unittest.TestCase):
    def test_repository_census_is_serialized_or_explicitly_deferred(self) -> None:
        result = subprocess.run(
            ["python3", str(SCRIPT)],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("17 audited files", result.stdout)
        self.assertIn("129 mutation sites", result.stdout)

    def test_unserialized_test_mutation_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            source = root / AUDITED_FILE
            source.parent.mkdir(parents=True)
            source.write_text(
                """#[cfg(test)]
mod tests {
    #[test]
    fn mutates_process_environment_without_serialization() {
        // SAFETY: this comment cannot replace the required test lock.
        unsafe { std::env::set_var(\"MESH_LLM_TEST_ENV\", \"1\") };
    }
}
""",
                encoding="utf-8",
            )

            result = subprocess.run(
                [
                    "python3",
                    str(SCRIPT),
                    "--root",
                    str(root),
                    "--file",
                    AUDITED_FILE,
                ],
                cwd=ROOT,
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("not covered by #[serial]", result.stderr)


if __name__ == "__main__":
    unittest.main()
