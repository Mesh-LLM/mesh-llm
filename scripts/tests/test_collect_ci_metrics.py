import importlib.util
import json
import pathlib
import subprocess
import sys
import tempfile
import unittest


SCRIPT = pathlib.Path(__file__).resolve().parents[1] / "collect-ci-metrics.py"


def load_collector():
    spec = importlib.util.spec_from_file_location("collect_ci_metrics", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def job(
    name,
    created_at,
    started_at,
    completed_at,
    *,
    conclusion="success",
    labels=None,
    steps=None,
):
    return {
        "id": hash((name, started_at)) & 0xFFFF,
        "name": name,
        "status": "completed",
        "conclusion": conclusion,
        "created_at": created_at,
        "started_at": started_at,
        "completed_at": completed_at,
        "html_url": f"https://example.test/jobs/{name}",
        "labels": labels or ["ubuntu-24.04"],
        "steps": steps or [],
    }


def run(
    run_id,
    created_at,
    started_at,
    updated_at,
    jobs,
    *,
    conclusion="success",
    attempt=1,
):
    return {
        "databaseId": run_id,
        "attempt": attempt,
        "workflowName": "PR Builds",
        "displayTitle": f"run {run_id}",
        "event": "pull_request",
        "status": "completed",
        "conclusion": conclusion,
        "createdAt": created_at,
        "startedAt": started_at,
        "updatedAt": updated_at,
        "url": f"https://example.test/runs/{run_id}",
        "headSha": f"sha-{run_id}",
        "headBranch": "feature",
        "jobs": jobs,
    }


class CollectCiMetricsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.collector = load_collector()

    def sample_runs(self):
        return [
            run(
                101,
                "2026-07-01T00:00:00Z",
                "2026-07-01T00:00:05Z",
                "2026-07-01T00:10:00Z",
                [
                    job(
                        "build",
                        "2026-07-01T00:00:10Z",
                        "2026-07-01T00:00:20Z",
                        "2026-07-01T00:08:20Z",
                    ),
                    job(
                        "smoke",
                        "2026-07-01T00:08:20Z",
                        "2026-07-01T00:08:25Z",
                        "2026-07-01T00:09:50Z",
                    ),
                    job(
                        "unused",
                        "2026-07-01T00:00:00Z",
                        "2026-07-01T00:10:00Z",
                        "2026-07-01T00:10:00Z",
                        conclusion="skipped",
                    ),
                ],
            ),
            run(
                102,
                "2026-07-02T00:00:00Z",
                "2026-07-02T00:00:15Z",
                "2026-07-02T00:20:00Z",
                [
                    job(
                        "build",
                        "2026-07-02T00:00:20Z",
                        "2026-07-02T00:00:50Z",
                        "2026-07-02T00:16:50Z",
                    ),
                    job(
                        "summary",
                        "2026-07-02T00:16:50Z",
                        "2026-07-02T00:16:52Z",
                        "2026-07-02T00:19:52Z",
                    ),
                ],
            ),
        ]

    def test_percentiles_use_linear_interpolation(self):
        summary = self.collector.summarize([10.0, 20.0, 30.0, 40.0])

        self.assertEqual(summary["p50"], 25.0)
        self.assertEqual(summary["p90"], 37.0)
        self.assertEqual(summary["p95"], 38.5)
        self.assertEqual(summary["mean"], 25.0)

    def test_analysis_reports_wall_queue_slow_and_terminal_jobs(self):
        runs = [self.collector.normalize_run(item) for item in self.sample_runs()]

        report = self.collector.analyze(
            runs,
            requested_status="success",
            top=5,
            source={"description": "fixture"},
            labels={"provider": "github"},
        )

        self.assertEqual(report["selection"]["included_run_count"], 2)
        self.assertEqual(report["workflow"]["wall_seconds"]["p50"], 900.0)
        self.assertEqual(report["workflow"]["queue_seconds"]["p50"], 10.0)
        by_name = {item["name"]: item for item in report["jobs"]["by_name"]}
        self.assertEqual(by_name["build"]["duration_seconds"]["p50"], 720.0)
        self.assertEqual(by_name["build"]["queue_seconds"]["p50"], 20.0)
        self.assertNotIn("unused", by_name)
        self.assertEqual(
            report["jobs"]["critical_finish_candidates"],
            [
                {"name": "smoke", "terminal_count": 1, "share": 0.5},
                {"name": "summary", "terminal_count": 1, "share": 0.5},
            ],
        )
        self.assertEqual(
            report["jobs"]["slowest_observations"][0]["name"],
            "build",
        )

    def test_gh_run_view_json_has_unavailable_job_queue_not_fake_queue(self):
        raw = self.sample_runs()[0]
        raw["jobs"][0].pop("created_at")
        raw["jobs"][0]["startedAt"] = raw["jobs"][0].pop("started_at")
        raw["jobs"][0]["completedAt"] = raw["jobs"][0].pop("completed_at")
        normalized = self.collector.normalize_run(raw)

        report = self.collector.analyze(
            [normalized],
            requested_status="success",
            top=5,
            source={"description": "gh run view"},
            labels={},
        )
        build = next(
            item for item in report["jobs"]["by_name"] if item["name"] == "build"
        )

        self.assertEqual(build["queue_seconds"]["count"], 0)
        self.assertEqual(build["start_delay_seconds"]["p50"], 20.0)

    def test_reruns_exclude_workflow_and_start_delay_timing(self):
        first_attempt = self.sample_runs()[0]
        rerun = run(
            103,
            "2026-07-01T00:00:00Z",
            "2026-07-02T00:00:05Z",
            "2026-07-02T00:10:00Z",
            [
                job(
                    "build",
                    "2026-07-02T00:00:10Z",
                    "2026-07-02T00:00:20Z",
                    "2026-07-02T00:08:20Z",
                )
            ],
            attempt=2,
        )
        runs = [
            self.collector.normalize_run(item) for item in (first_attempt, rerun)
        ]

        report = self.collector.analyze(
            runs,
            requested_status="success",
            top=5,
            source={"description": "fixture"},
            labels={},
        )

        self.assertEqual(
            report["selection"]["workflow_timing_excluded_reruns"],
            1,
        )
        self.assertEqual(report["workflow"]["wall_seconds"]["count"], 1)
        self.assertEqual(report["workflow"]["queue_seconds"]["count"], 1)
        rerun_report = next(item for item in report["runs"] if item["id"] == 103)
        self.assertEqual(rerun_report["attempt"], 2)
        self.assertTrue(rerun_report["workflow_timing_excluded"])
        self.assertIsNone(rerun_report["wall_seconds"])
        build = next(
            item for item in report["jobs"]["by_name"] if item["name"] == "build"
        )
        self.assertEqual(build["duration_seconds"]["count"], 2)
        self.assertEqual(build["start_delay_seconds"]["count"], 1)
        self.assertIn("Excluded workflow wall", self.collector.render_markdown(report, 5))

    def test_non_matching_and_in_progress_runs_are_reported_as_skipped(self):
        successful, failed = self.sample_runs()
        failed["conclusion"] = "failure"
        pending = self.sample_runs()[0]
        pending["databaseId"] = 103
        pending["status"] = "in_progress"
        runs = [
            self.collector.normalize_run(item)
            for item in (successful, failed, pending)
        ]

        report = self.collector.analyze(
            runs,
            requested_status="success",
            top=5,
            source={"description": "fixture"},
            labels={},
        )

        self.assertEqual(report["selection"]["included_run_count"], 1)
        self.assertEqual(
            report["selection"]["skipped_runs"],
            {"conclusion_failure": 1, "not_completed": 1},
        )

    def test_cli_reads_saved_json_and_writes_both_report_formats(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            input_path = root / "runs.json"
            json_path = root / "metrics.json"
            markdown_path = root / "metrics.md"
            input_path.write_text(
                json.dumps({"runs": self.sample_runs()}),
                encoding="utf-8",
            )

            subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--input",
                    str(input_path),
                    "--json-out",
                    str(json_path),
                    "--markdown-out",
                    str(markdown_path),
                    "--label",
                    "provider=fixture",
                ],
                check=True,
                capture_output=True,
                text=True,
            )

            report = json.loads(json_path.read_text(encoding="utf-8"))
            markdown = markdown_path.read_text(encoding="utf-8")
            self.assertEqual(report["schema_version"], 3)
            self.assertEqual(report["benchmark_labels"], {"provider": "fixture"})
            self.assertIn("# CI timing summary", markdown)
            self.assertIn("Runner dimensions", markdown)
            self.assertIn("Step timing", markdown)
            self.assertIn("Slow job families", markdown)
            self.assertIn("20m 0s", markdown)

    def test_analysis_preserves_runner_dimensions_and_step_timing(self):
        raw = run(
            104,
            "2026-07-03T00:00:00Z",
            "2026-07-03T00:00:01Z",
            "2026-07-03T00:03:00Z",
            [
                job(
                    "Linux CPU runtime",
                    "2026-07-03T00:00:02Z",
                    "2026-07-03T00:00:03Z",
                    "2026-07-03T00:02:00Z",
                    labels=["depot-ubuntu-24.04-8"],
                    steps=[
                        {
                            "name": "Restore native cache",
                            "number": 1,
                            "conclusion": "success",
                            "started_at": "2026-07-03T00:00:04Z",
                            "completed_at": "2026-07-03T00:00:14Z",
                        },
                        {
                            "name": "Build runtime",
                            "number": 2,
                            "conclusion": "success",
                            "started_at": "2026-07-03T00:00:14Z",
                            "completed_at": "2026-07-03T00:01:44Z",
                        },
                    ],
                ),
                job(
                    "Linux CPU runtime",
                    "2026-07-03T00:00:05Z",
                    "2026-07-03T00:00:06Z",
                    "2026-07-03T00:02:30Z",
                    labels=["depot-ubuntu-24.04-arm-8"],
                    steps=[
                        {
                            "name": "Restore native cache",
                            "number": 1,
                            "conclusion": "success",
                            "started_at": "2026-07-03T00:00:07Z",
                            "completed_at": "2026-07-03T00:00:27Z",
                        },
                        {
                            "name": "Build runtime",
                            "number": 2,
                            "conclusion": "success",
                            "started_at": "2026-07-03T00:00:27Z",
                            "completed_at": "2026-07-03T00:02:27Z",
                        },
                    ],
                ),
            ],
        )
        report = self.collector.analyze(
            [self.collector.normalize_run(raw)],
            requested_status="success",
            top=5,
            source={"description": "step fixture"},
            labels={},
        )

        depot_runners = {
            item["architecture"]: item
            for item in report["jobs"]["by_runner"]
            if item["provider"] == "depot"
        }
        self.assertEqual(set(depot_runners), {"amd64", "arm64"})
        self.assertEqual(depot_runners["amd64"]["runner_size"], "8")
        self.assertEqual(depot_runners["arm64"]["runner_size"], "8")
        steps = {
            (
                item["architecture"],
                item["runner_size"],
                item["step_name"],
            ): item
            for item in report["steps"]["by_name"]
        }
        self.assertEqual(len(steps), 4)
        self.assertEqual(
            steps[("amd64", "8", "Restore native cache")]["duration_seconds"]["p50"],
            10.0,
        )
        self.assertEqual(
            steps[("amd64", "8", "Build runtime")]["duration_seconds"]["p50"],
            90.0,
        )
        self.assertEqual(
            steps[("arm64", "8", "Restore native cache")]["duration_seconds"]["p50"],
            20.0,
        )
        self.assertEqual(
            steps[("arm64", "8", "Build runtime")]["duration_seconds"]["p50"],
            120.0,
        )
        slowest = report["jobs"]["slowest_observations"][0]
        self.assertEqual(slowest["runner_dimensions"]["runner_size"], "8")

    def test_timing_decomposition_keeps_dependency_wait_explicit(self):
        raw = run(
            105,
            "2026-07-04T00:00:00Z",
            "2026-07-04T00:00:01Z",
            "2026-07-04T00:03:00Z",
            [
                job(
                    "runtime",
                    "2026-07-04T00:00:10Z",
                    "2026-07-04T00:00:20Z",
                    "2026-07-04T00:02:20Z",
                    labels=["depot-ubuntu-24.04-8"],
                )
            ],
        )
        raw["jobs"][0]["dependency_ready_at"] = "2026-07-04T00:00:18Z"
        raw["jobs"][0]["runner_role"] = "linux-native-16"
        raw["jobs"][0]["operating_system"] = "Linux"
        normalized = self.collector.normalize_run(raw)
        report = self.collector.analyze(
            [normalized],
            requested_status="success",
            top=5,
            source={"description": "decomposition"},
            labels={"provider": "depot"},
        )
        runner = report["jobs"]["by_runner"][0]
        self.assertEqual(runner["operating_system"], "linux")
        self.assertEqual(runner["runner_role"], "linux-native-16")
        self.assertEqual(report["jobs"]["queue_seconds"]["p50"], 10.0)
        self.assertEqual(report["jobs"]["runner_queue_seconds"]["p50"], 2.0)
        self.assertEqual(report["jobs"]["dependency_wait_seconds"]["p50"], 8.0)
        self.assertEqual(report["jobs"]["execution_seconds"]["p50"], 120.0)
        self.assertEqual(report["jobs"]["wall_seconds"]["p50"], 130.0)
        self.assertEqual(report["heuristics"]["state"], "insufficient_sample")

    def test_capacity_contamination_and_queue_heuristic_are_deterministic(self):
        raw_runs = self.sample_runs()
        for raw_run in raw_runs:
            for raw_job in raw_run["jobs"]:
                if raw_job["conclusion"] == "skipped":
                    continue
                raw_job["created_at"] = "2026-06-30T00:00:00Z"
                raw_job["labels"] = ["depot-ubuntu-24.04-8"]
        report = self.collector.analyze(
            [self.collector.normalize_run(raw) for raw in raw_runs],
            requested_status="success",
            top=5,
            source={"description": "contaminated"},
            labels={"provider": "depot"},
        )
        self.assertTrue(report["heuristics"]["capacity_contaminated"])
        self.assertEqual(report["heuristics"]["state"], "rollback")
        self.assertGreaterEqual(report["capacity"]["runner_minutes"], 28.0)

    def test_compare_reports_requires_provider_cohort_separation(self):
        baseline_raw = self.sample_runs()
        candidate_raw = self.sample_runs()
        for raw_runs in (baseline_raw, candidate_raw):
            raw_runs[0]["jobs"][1]["name"] = "test suite"
            raw_runs[1]["jobs"].append(
                job(
                    "Quality / Runner and cache contracts / Runner and cache contract",
                    "2026-07-02T00:00:20Z",
                    "2026-07-02T00:00:25Z",
                    "2026-07-02T00:00:35Z",
                )
            )
        baseline = self.collector.analyze(
            [self.collector.normalize_run(raw) for raw in baseline_raw],
            requested_status="success",
            top=5,
            source={"description": "github historical"},
            labels={"provider": "github"},
        )
        for raw_run in candidate_raw:
            for raw_job in raw_run["jobs"]:
                if self.collector.is_comparison_executor(raw_job["name"]):
                    raw_job["labels"] = ["depot-ubuntu-24.04-8"]
        candidate = self.collector.analyze(
            [self.collector.normalize_run(raw) for raw in candidate_raw],
            requested_status="success",
            top=5,
            source={"description": "depot candidate"},
            labels={"provider": "depot"},
        )
        comparison = self.collector.compare_reports(baseline, candidate)
        self.assertTrue(comparison["provider_cohort_separation"]["disjoint"])
        self.assertEqual(comparison["recommendation"], "eligible")
        self.assertEqual(
            candidate["jobs"]["comparison_cohort"]["providers"],
            ["depot"],
        )

    def test_compare_reports_holds_when_baseline_cohort_is_too_small(self):
        baseline_raw = [self.sample_runs()[0]]
        candidate_raw = self.sample_runs()
        candidate_raw[0]["jobs"][1]["name"] = "test suite"
        for raw_run in candidate_raw:
            for raw_job in raw_run["jobs"]:
                if self.collector.is_comparison_executor(raw_job["name"]):
                    raw_job["labels"] = ["depot-ubuntu-24.04-8"]
        baseline = self.collector.analyze(
            [self.collector.normalize_run(raw) for raw in baseline_raw],
            requested_status="success",
            top=5,
            source={"description": "small github baseline"},
            labels={},
        )
        candidate = self.collector.analyze(
            [self.collector.normalize_run(raw) for raw in candidate_raw],
            requested_status="success",
            top=5,
            source={"description": "depot candidate"},
            labels={},
        )
        comparison = self.collector.compare_reports(baseline, candidate)
        self.assertFalse(comparison["sample_counts"]["sufficient"])
        self.assertEqual(comparison["recommendation"], "hold")

    def test_runner_dimensions_cover_depot_platforms_and_hosted_intel_macos(self):
        depot_macos = self.collector.runner_dimensions(["depot-macos-15"])
        depot_windows = self.collector.runner_dimensions(["depot-windows-2022-8"])
        hosted_intel = self.collector.runner_dimensions(["macos-15-intel"])
        self.assertEqual(
            depot_macos,
            {
                "provider": "depot",
                "architecture": "arm64",
                "runner_size": "default",
                "operating_system": "macos",
                "runner_role": None,
            },
        )
        self.assertEqual(depot_windows["operating_system"], "windows")
        self.assertEqual(depot_windows["architecture"], "amd64")
        self.assertEqual(hosted_intel["architecture"], "amd64")

    def test_cli_rejects_run_list_json_without_detailed_jobs(self):
        with tempfile.TemporaryDirectory() as directory:
            input_path = pathlib.Path(directory) / "runs.json"
            raw = self.sample_runs()[0]
            raw.pop("jobs")
            input_path.write_text(json.dumps([raw]), encoding="utf-8")

            result = subprocess.run(
                [sys.executable, str(SCRIPT), "--input", str(input_path)],
                capture_output=True,
                text=True,
            )

            self.assertEqual(result.returncode, 2)
            self.assertIn("has no jobs array", result.stderr)


if __name__ == "__main__":
    unittest.main()
