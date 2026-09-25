from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import tempfile
import textwrap
import unittest

import yaml


ROOT = Path(__file__).resolve().parents[2]
WORKFLOWS = ROOT / ".github" / "workflows"


class CiPrCanaryWorkflowTests(unittest.TestCase):
    def workflow(self, name: str) -> str:
        return (WORKFLOWS / name).read_text(encoding="utf-8")

    def _source_matrix_script(self) -> str:
        workflow = self.workflow("ci-pr-canary-lane.yml")
        step = workflow.split(
            "      - name: Resolve the catalog-owned CPU canary row\n", 1
        )[1].split("\n\n  ui_artifact:", 1)[0]
        return textwrap.dedent(step.split("        run: |\n", 1)[1])

    def _run_source_matrix(self, catalog: dict) -> subprocess.CompletedProcess[str]:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "ci").mkdir()
            (root / "ci" / "slices.yml").write_text(
                json.dumps(catalog), encoding="utf-8"
            )
            output = root / "github-output"
            result = subprocess.run(
                ["bash", "-c", self._source_matrix_script()],
                cwd=root,
                env={**os.environ, "GITHUB_OUTPUT": str(output)},
                capture_output=True,
                text=True,
                check=False,
            )
            if output.exists():
                result.stdout = output.read_text(encoding="utf-8")
            return result

    @staticmethod
    def _catalog_row(**overrides: str) -> dict:
        row = {
            "id": "linux-cpu",
            "platform": "linux",
            "architecture": "amd64",
            "backend": "cpu",
            "target": "x86_64-unknown-linux-gnu",
            "build_dir": ".deps/llama.cpp/build-stage-abi-dynamic-cpu",
            "container_image": "ghcr.io/mesh-llm/mesh-llm-cuda-runner@sha256:" + "a" * 64,
            "toolchain_epoch": "mesh-llm-cuda-runner-sha256-" + "a" * 64,
            "verify_backend": "public",
        }
        row.update(overrides)
        return row

    def _reachable_workflows(self) -> dict[str, dict]:
        pending = ["pr_ci_canary.yml"]
        seen: dict[str, dict] = {}
        while pending:
            name = pending.pop()
            if name in seen:
                continue
            document = yaml.safe_load(self.workflow(name)) or {}
            seen[name] = document
            for job in (document.get("jobs") or {}).values():
                uses = job.get("uses") if isinstance(job, dict) else None
                if isinstance(uses, str) and uses.startswith("./.github/workflows/"):
                    pending.append(Path(uses).name)
                elif uses == (
                    "Mesh-LLM/mesh-llm/.github/workflows/ci-pr-canary-lane.yml@main"
                ):
                    pending.append("ci-pr-canary-lane.yml")
        return seen

    def test_entrypoint_is_label_gated_and_does_not_cancel_on_other_labels(self) -> None:
        workflow = self.workflow("pr_ci_canary.yml")

        self.assertIn("  pull_request:\n", workflow)
        self.assertIn(
            "types: [opened, synchronize, reopened, ready_for_review, labeled, unlabeled]",
            workflow,
        )
        self.assertNotIn("paths:", workflow)
        self.assertIn("contains(github.event.pull_request.labels.*.name, 'ci:canary')", workflow)
        self.assertIn("github.event.label.name == 'ci:canary'", workflow)
        self.assertIn("github.event.label.name != 'ci:canary'", workflow)
        self.assertIn("format('unrelated-label-{0}', github.run_id)", workflow)
        self.assertIn("cancel-in-progress: true", workflow)
        self.assertIn("merge_sha: ${{ github.sha }}", workflow)
        self.assertIn(
            "uses: Mesh-LLM/mesh-llm/.github/workflows/ci-pr-canary-lane.yml@main",
            workflow,
        )
        self.assertNotIn("uses: ./.github/workflows/ci-pr-canary-lane.yml", workflow)

        for forbidden in (
            "checks: write",
            "secrets:",
            "secrets: inherit",
            "environment:",
            "id-token:",
            "self-hosted",
            "use_depot: true",
        ):
            self.assertNotIn(forbidden, workflow)

    def test_lane_calls_existing_linux_slices_without_linux_orchestration(self) -> None:
        workflow = self.workflow("ci-pr-canary-lane.yml")

        self.assertIn("  workflow_call:\n", workflow)
        for slice_name in (
            "ci-ui-artifact-slice.yml",
            "ci-linux-host-slice.yml",
            "ci-linux-runtime-slice.yml",
            "ci-linux-product-slice.yml",
        ):
            self.assertEqual(
                1,
                workflow.count(f"uses: ./.github/workflows/{slice_name}"),
                slice_name,
            )
        self.assertNotIn("ci-linux-lane.yml", workflow)
        self.assertIn("name: Canary / CI", workflow)
        self.assertIn("permissions: {}", workflow)
        self.assertIn("native runtime-event gate", workflow)
        self.assertIn("Linux lane orchestration", workflow)
        self.assertIn('refs/pull/${PR_NUMBER}/head', workflow)
        for forbidden in (
            "checks: write",
            "secrets:",
            "secrets: inherit",
            "environment:",
            "id-token:",
            "self-hosted",
            "use_depot: true",
        ):
            self.assertNotIn(forbidden, workflow)

    def test_canary_matrix_is_catalog_owned_and_bounded(self) -> None:
        workflow = self.workflow("ci-pr-canary-lane.yml")
        slices = json.loads((ROOT / "ci" / "slices.yml").read_text(encoding="utf-8"))
        runtime_rows = [row for row in slices["runtime_rows"] if row["id"] == "linux-cpu"]
        self.assertEqual(1, len(runtime_rows))
        row = runtime_rows[0]
        self.assertEqual(
            {
                "platform": "linux",
                "architecture": "amd64",
                "backend": "cpu",
                "target": "x86_64-unknown-linux-gnu",
                "verify_backend": "public",
            },
            {key: row[key] for key in ("platform", "architecture", "backend", "target", "verify_backend")},
        )

        self.assertIn("id: source_matrix", workflow)
        self.assertIn("expected exactly one linux-cpu runtime row", workflow)
        for field in (
            ".platform == \"linux\"",
            ".architecture == \"amd64\"",
            ".backend == \"cpu\"",
            ".target == \"x86_64-unknown-linux-gnu\"",
            ".verify_backend == \"public\"",
        ):
            self.assertIn(field, workflow)
        self.assertIn("runtime_matrix=$runtime_matrix", workflow)
        self.assertIn("runtime_matrix: ${{ needs.plan.outputs.runtime_matrix }}", workflow)
        self.assertIn("hosts_matrix: ${{ needs.plan.outputs.host_matrix }}", workflow)
        self.assertIn("runtime_matrix: ${{ needs.plan.outputs.product_matrix }}", workflow)

        # Build inputs stay catalog-derived; only the selector constraints above
        # may mention the CPU row's identity. This prevents silent image/epoch/
        # build-directory drift from the versioned runtime catalog.
        for field in ("build_dir", "container_image", "toolchain_epoch"):
            self.assertIn(f"(.{field} |", workflow)
            self.assertNotIn(str(row[field]), workflow)

    def test_canary_keeps_runner_policy_on_protected_default_branch(self) -> None:
        lane = self.workflow("ci-pr-canary-lane.yml")
        for slice_name in (
            "ci-ui-artifact-slice.yml",
            "ci-linux-host-slice.yml",
            "ci-linux-runtime-slice.yml",
            "ci-linux-product-slice.yml",
        ):
            slice_workflow = self.workflow(slice_name)
            self.assertIn("policy_source_sha:", slice_workflow)
            self.assertIn(
                "ref: ${{ inputs.policy_source_sha || github.event.repository.default_branch }}",
                slice_workflow,
            )
        self.assertNotIn("policy_source_sha: ${{ inputs.merge_sha }}", lane)

        changes = (ROOT / ".github" / "actions" / "compute-changes" / "derive-outputs.sh").read_text(
            encoding="utf-8"
        )
        self.assertIn("pr_ci_canary", changes)

    def test_source_matrix_step_executes_catalog_row_and_propagates_inputs(self) -> None:
        catalog = {"runtime_rows": [self._catalog_row()]}
        result = self._run_source_matrix(catalog)
        self.assertEqual(0, result.returncode, result.stderr)
        output = dict(
            line.split("=", 1)
            for line in result.stdout.splitlines()
            if "=" in line
        )
        self.assertEqual(
            [
                {
                    "id": "linux-cpu",
                    "platform": "linux",
                    "architecture": "amd64",
                }
            ],
            json.loads(output["host_matrix"]),
        )
        self.assertEqual(
            [
                {
                    "id": "linux-cpu",
                    "platform": "linux",
                    "architecture": "amd64",
                    "backend": "cpu",
                }
            ],
            json.loads(output["product_matrix"]),
        )
        runtime = json.loads(output["runtime_matrix"])
        self.assertEqual(catalog["runtime_rows"], runtime)

        changed = self._catalog_row(
            build_dir=".deps/llama.cpp/build-stage-abi-dynamic-cpu-new",
            container_image="ghcr.io/mesh-llm/mesh-llm-cuda-runner@sha256:" + "b" * 64,
            toolchain_epoch="mesh-llm-cuda-runner-sha256-" + "b" * 64,
        )
        changed_result = self._run_source_matrix({"runtime_rows": [changed]})
        self.assertEqual(0, changed_result.returncode, changed_result.stderr)
        changed_output = dict(
            line.split("=", 1)
            for line in changed_result.stdout.splitlines()
            if "=" in line
        )
        self.assertEqual([changed], json.loads(changed_output["runtime_matrix"]))

    def test_source_matrix_step_rejects_missing_duplicate_or_wrong_cpu_rows(self) -> None:
        cases = {
            "missing": {"runtime_rows": []},
            "duplicate": {"runtime_rows": [self._catalog_row(), self._catalog_row()]},
            "wrong-backend": {"runtime_rows": [self._catalog_row(backend="cuda")]},
            "wrong-architecture": {"runtime_rows": [self._catalog_row(architecture="arm64")]},
        }
        for name, catalog in cases.items():
            with self.subTest(case=name):
                result = self._run_source_matrix(catalog)
                self.assertNotEqual(0, result.returncode)

    def test_reachable_canary_workflows_have_read_only_permissions_and_no_secrets(self) -> None:
        workflows = self._reachable_workflows()
        self.assertEqual(
            {
                "pr_ci_canary.yml",
                "ci-pr-canary-lane.yml",
                "ci-ui-artifact-slice.yml",
                "ci-linux-host-slice.yml",
                "ci-linux-runtime-slice.yml",
                "ci-linux-product-slice.yml",
            },
            set(workflows),
        )

        def walk(value):
            if isinstance(value, dict):
                yield value
                for key, child in value.items():
                    if key == "permissions" and isinstance(child, dict):
                        for permission, mode in child.items():
                            self.assertNotIn(
                                str(mode).lower(),
                                {"write", "write-all"},
                                f"{permission} write permission",
                            )
                    if key in {"secrets", "environment", "id-token"}:
                        self.fail(f"forbidden {key} in canary workflow")
                    yield from walk(child)
            elif isinstance(value, list):
                for child in value:
                    yield from walk(child)

        for name, document in workflows.items():
            raw = self.workflow(name)
            self.assertNotIn("secrets: inherit", raw)
            self.assertNotIn("self-hosted", raw)
            self.assertNotIn("use_depot: true", raw)
            list(walk(document))


if __name__ == "__main__":
    unittest.main()
