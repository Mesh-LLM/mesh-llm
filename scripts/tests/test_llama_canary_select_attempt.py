from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "canary_select", ROOT / "scripts/llama-canary-select-attempt.py"
)
SELECT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SELECT)


def job(**outputs):
    return {"result": "success", "outputs": outputs}


def green(head="a" * 40):
    return job(green="true", head=head, package="package", identity="b" * 64, branch="branch")


def repairable(head="a" * 40):
    return job(repairable="true", head=head, package="package", identity="b" * 64,
               feedback="feedback", failure_class="candidate",
               failure_stage="family-certification")


class AttemptSelectionTests(unittest.TestCase):
    def test_changed_attempt_requires_matching_green_verification(self):
        result = SELECT.select_attempt(True, green(), green())
        self.assertEqual(result["state"], "green")
        self.assertEqual(result["head"], "a" * 40)
        mismatch = SELECT.select_attempt(True, green(), green("c" * 40))
        self.assertEqual(mismatch["state"], "failed")

    def test_candidate_or_verifier_family_failure_can_resume(self):
        for candidate, verify in ((repairable(), job()), (green(), repairable())):
            with self.subTest(candidate=candidate):
                result = SELECT.select_attempt(True, candidate, verify)
                self.assertEqual(result["state"], "repairable")
                self.assertEqual(result["resume_feedback"], "feedback")

    def test_infrastructure_failure_never_resumes(self):
        result = SELECT.select_attempt(True, job(failure_class="infrastructure"), job())
        self.assertEqual(result["state"], "failed")
        self.assertEqual(result["repairable"], "false")

    def test_unchanged_certification_needs_no_verifier(self):
        self.assertEqual(SELECT.select_attempt(False, green(), job())["state"], "green")

    def test_unchanged_failure_never_starts_a_repair_attempt(self):
        result = SELECT.select_attempt(False, repairable(), job())
        self.assertEqual(result["state"], "failed")
        self.assertEqual(result["repairable"], "false")
        self.assertEqual(result["failure_class"], "candidate")

    def test_final_uses_latest_attempt_and_publishes_only_changed_upgrade(self):
        selected = job(state="green", green="true", head="a" * 40, package="package",
                       identity="b" * 64, branch="branch")
        result = SELECT.select_final(True, True, "", "success", {
            "attempt_1": job(state="repairable"), "attempt_2": selected,
            "attempt_3": {"result": "skipped", "outputs": {}},
        })
        self.assertEqual(result["publish"], "true")
        self.assertEqual(result["package"], "package")

    def test_exhausted_third_attempt_denies_publication(self):
        attempts = {f"attempt_{number}": job(state="repairable", failure_class="candidate",
                                             failure_stage="family-certification")
                    for number in range(1, 4)}
        with self.assertRaisesRegex(ValueError, "publication denied"):
            SELECT.select_final(True, True, "", "success", attempts)

    def test_selected_source_must_match_and_never_publishes(self):
        attempts = {"attempt_1": job(state="green", head="a" * 40)}
        result = SELECT.select_final(True, False, "a" * 40, "success", attempts)
        self.assertEqual(result["publish"], "false")
        with self.assertRaisesRegex(ValueError, "selected MeshLLM"):
            SELECT.select_final(True, False, "c" * 40, "success", attempts)

    def test_noop_and_preflight_failure(self):
        self.assertEqual(SELECT.select_final(False, False, "", "skipped", {})["state"], "noop")
        with self.assertRaisesRegex(ValueError, "preflight"):
            SELECT.select_final(True, True, "", "failure", {})


if __name__ == "__main__":
    unittest.main()
