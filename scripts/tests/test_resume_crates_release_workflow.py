#!/usr/bin/env python3

from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = ROOT / ".github" / "workflows" / "resume-crates-release.yml"


class ResumeCratesReleaseWorkflowTests(unittest.TestCase):
    def test_recovery_is_default_branch_only_and_exact_source_bound(self) -> None:
        workflow = WORKFLOW.read_text(encoding="utf-8")

        self.assertIn("github.ref == 'refs/heads/main'", workflow)
        self.assertIn("^[0-9a-f]{40}$", workflow)
        self.assertIn('"refs/tags/${RELEASE_TAG}^{}"', workflow)
        self.assertIn('checked_out_sha="$(git -C release-source rev-parse HEAD)"', workflow)
        self.assertIn('remote_sha" != "$EXPECTED_SOURCE_SHA', workflow)
        self.assertIn('checked_out_sha" != "$EXPECTED_SOURCE_SHA', workflow)

    def test_secret_is_scoped_to_publish_step(self) -> None:
        workflow = WORKFLOW.read_text(encoding="utf-8")

        self.assertEqual(workflow.count("secrets.CARGO_REGISTRY_TOKEN"), 1)
        self.assertIn("persist-credentials: false", workflow)
        self.assertIn("../controller/scripts/publish-crates.sh --resume", workflow)
        self.assertIn("working-directory: release-source", workflow)


if __name__ == "__main__":
    unittest.main()
