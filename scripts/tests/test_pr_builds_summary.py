from __future__ import annotations

from collections import Counter
import importlib.util
from pathlib import Path
import re
import unittest


ROOT = Path(__file__).resolve().parents[2]
PR_BUILDS = ROOT / ".github" / "workflows" / "pr_builds.yml"
PLANNER_PATH = ROOT / "scripts" / "plan-pr-build-jobs.py"
JOB_HEADER = re.compile(r"(?m)^  ([A-Za-z0-9_-]+):\n")

PLANNER_SPEC = importlib.util.spec_from_file_location(
    "plan_pr_build_jobs",
    PLANNER_PATH,
)
if PLANNER_SPEC is None or PLANNER_SPEC.loader is None:
    raise RuntimeError(f"unable to import {PLANNER_PATH}")
PLANNER = importlib.util.module_from_spec(PLANNER_SPEC)
PLANNER_SPEC.loader.exec_module(PLANNER)


def job_sections(workflow: str) -> dict[str, str]:
    jobs_start = workflow.index("jobs:\n") + len("jobs:\n")
    jobs_body = workflow[jobs_start:]
    matches = list(JOB_HEADER.finditer(jobs_body))
    sections: dict[str, str] = {}
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(jobs_body)
        sections[match.group(1)] = jobs_body[match.start() : end]
    return sections


def summary_needs(summary: str) -> list[str]:
    needs_match = re.search(
        r"(?ms)^    needs:\n(?P<items>(?:      - [A-Za-z0-9_-]+\n)+)",
        summary,
    )
    if needs_match is None:
        raise AssertionError("PR Builds summary must use an explicit needs list")
    return re.findall(r"(?m)^      - ([A-Za-z0-9_-]+)$", needs_match.group("items"))


class PrBuildsSummaryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.workflow = PR_BUILDS.read_text(encoding="utf-8")
        cls.jobs = job_sections(cls.workflow)
        cls.summary = cls.jobs["summary"]

    def test_summary_has_a_stable_branch_protection_name(self) -> None:
        self.assertIn("    name: PR Builds Summary\n", self.summary)
        self.assertNotIn("${{", self.summary.split("    name:", 1)[1].splitlines()[0])
        self.assertIn("    if: ${{ !cancelled() }}\n", self.summary)
        self.assertNotIn("always()", self.summary)

    def test_summary_directly_needs_every_other_top_level_job(self) -> None:
        expected = [job for job in self.jobs if job != "summary"]
        self.assertEqual(summary_needs(self.summary), expected)

    def test_every_conditional_job_is_mapped_exactly_once(self) -> None:
        mapped_jobs = [job_id for job_id, _route in PLANNER.JOB_ROUTES]
        counts = Counter(mapped_jobs)
        self.assertTrue(all(count == 1 for count in counts.values()))
        self.assertEqual(
            mapped_jobs,
            [job for job in self.jobs if job not in {"changes", "summary"}],
        )

    def test_every_conditional_job_consumes_the_shared_plan(self) -> None:
        for job_name, section in self.jobs.items():
            if job_name in {"changes", "summary"}:
                continue
            with self.subTest(job=job_name):
                expected = (
                    "    if: ${{ contains("
                    "fromJson(needs.changes.outputs.required_jobs_json), "
                    f"'{job_name}') }}}}\n"
                )
                job_level_conditions = re.findall(
                    r"(?m)^    if: .+$",
                    section,
                )
                self.assertEqual(job_level_conditions, [expected.rstrip()])

    def test_changes_exports_the_checked_in_plan(self) -> None:
        changes = self.jobs["changes"]
        self.assertIn(
            "required_jobs_json: ${{ steps.plan.outputs.required_jobs_json }}",
            changes,
        )
        self.assertIn("python3 scripts/plan-pr-build-jobs.py", changes)
        self.assertIn("PR_BUILD_PLAN_INPUT:", changes)

    def test_summary_evaluates_the_complete_needs_result_object(self) -> None:
        self.assertIn("NEEDS_RESULTS: ${{ toJson(needs) }}", self.summary)
        self.assertIn(
            "REQUIRED_JOBS: ${{ needs.changes.outputs.required_jobs_json }}",
            self.summary,
        )
        self.assertIn("set -euo pipefail", self.summary)
        self.assertIn('$entry.value.result == "success"', self.summary)
        self.assertIn('$entry.value.result == "skipped"', self.summary)
        self.assertIn("($required | index($entry.key)) == null", self.summary)
        self.assertIn("has($job)", self.summary)
        self.assertIn("select(accepted | not)", self.summary)
        self.assertIn("exit 1", self.summary)


if __name__ == "__main__":
    unittest.main()
