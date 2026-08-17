from __future__ import annotations

from pathlib import Path
import subprocess
import sys
import unittest


ROOT = Path(__file__).resolve().parents[2]
SELECTOR = ROOT / "scripts" / "select-release-notes-base.py"


class SelectReleaseNotesBaseTests(unittest.TestCase):
    def select(self, target: str, tags: list[str]) -> str:
        result = subprocess.run(
            [sys.executable, str(SELECTOR), target],
            cwd=ROOT,
            input="\n".join(tags),
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        return result.stdout.strip()

    def test_release_candidate_uses_previous_stable_release(self) -> None:
        self.assertEqual(
            self.select(
                "v0.76.0-rc1",
                ["v0.75.0", "v0.75.1", "v0.76.0-rc1"],
            ),
            "v0.75.1",
        )

    def test_stable_release_ignores_release_candidates_for_same_version(
        self,
    ) -> None:
        self.assertEqual(
            self.select(
                "v0.76.0",
                ["v0.75.1", "v0.76.0-rc1", "v0.76.0-rc2"],
            ),
            "v0.75.1",
        )

    def test_next_patch_candidate_uses_current_stable_release(self) -> None:
        self.assertEqual(
            self.select(
                "0.76.1-rc1",
                ["v0.75.1", "v0.76.0", "v0.76.1-rc1"],
            ),
            "v0.76.0",
        )

    def test_selection_is_semantic_and_ignores_nonstable_tags(self) -> None:
        self.assertEqual(
            self.select(
                "v0.11.0",
                ["v0.9.9", "v0.10.0", "v0.10.1-rc1", "not-a-release"],
            ),
            "v0.10.0",
        )

    def test_first_release_has_no_comparison_base(self) -> None:
        self.assertEqual(self.select("v0.1.0", ["v0.1.0-rc1"]), "")

    def test_invalid_target_is_rejected(self) -> None:
        result = subprocess.run(
            [sys.executable, str(SELECTOR), "v0.76"],
            cwd=ROOT,
            input="v0.75.1\n",
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(result.returncode, 2)
        self.assertIn("invalid release tag", result.stderr)


if __name__ == "__main__":
    unittest.main()
