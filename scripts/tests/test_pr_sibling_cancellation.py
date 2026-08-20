from __future__ import annotations

import json
from pathlib import Path
import subprocess
import unittest


ROOT = Path(__file__).resolve().parents[2]
ACTION = ROOT / ".github" / "actions" / "cancel-pr-sibling-runs" / "index.js"
WORKFLOW = ROOT / ".github" / "workflows" / "pr-cancel-sibling-runs.yml"


class PrSiblingCancellationTests(unittest.TestCase):
    def run_node(self, expression: str) -> object:
        script = f"""
const action = require(process.argv[1]);
const result = (() => {{ {expression} }})();
process.stdout.write(JSON.stringify(result));
"""
        completed = subprocess.run(
            ["node", "-e", script, str(ACTION)],
            check=True,
            cwd=ROOT,
            capture_output=True,
            text=True,
        )
        return json.loads(completed.stdout)

    def run_node_async(self, expression: str) -> object:
        script = f"""
const action = require(process.argv[1]);
(async () => {{
  const result = await (async () => {{ {expression} }})();
  process.stdout.write(JSON.stringify(result));
}})().catch((error) => {{
  console.error(error);
  process.exitCode = 1;
}});
"""
        completed = subprocess.run(
            ["node", "-e", script, str(ACTION)],
            check=True,
            cwd=ROOT,
            capture_output=True,
            text=True,
        )
        return json.loads(completed.stdout)

    def test_monitor_is_protected_and_pr_only(self) -> None:
        workflow = WORKFLOW.read_text(encoding="utf-8")
        self.assertIn("workflow_run:", workflow)
        self.assertIn("workflows: [PR · Quality]", workflow)
        self.assertIn("types: [in_progress]", workflow)
        self.assertNotIn("pull_request_target:", workflow)
        self.assertNotIn("\n  pull_request:\n", workflow)
        self.assertIn("github.event.workflow_run.event == 'pull_request'", workflow)
        self.assertIn("actions: write", workflow)
        self.assertIn("permissions: {}", workflow)
        self.assertIn(
            "ref: ${{ github.event.repository.default_branch }}",
            workflow,
        )
        self.assertIn("persist-credentials: false", workflow)
        self.assertNotIn("github.event.workflow_run.head_sha", workflow)
        self.assertNotIn("secrets:", workflow)

        for lane in ("quality", "website", "linux", "macos", "windows"):
            pr_workflow = (
                ROOT / ".github" / "workflows" / f"pr_{lane}.yml"
            ).read_text(encoding="utf-8")
            self.assertNotIn("actions: write", pr_workflow)

    def test_trigger_requires_exact_quality_pr_identity(self) -> None:
        result = self.run_node(
            """
const sha = "a".repeat(40);
const payload = {
  repository: {full_name: "Mesh-LLM/mesh-llm"},
  workflow_run: {
    id: 101,
    name: "PR · Quality",
    event: "pull_request",
    head_sha: sha,
    created_at: "2026-08-20T12:00:00Z",
    pull_requests: [{number: 42}],
  },
};
return action.parseTrigger(payload, "Mesh-LLM/mesh-llm");
"""
        )
        self.assertEqual(42, result["pullNumber"])
        self.assertEqual(101, result["triggerRunId"])
        self.assertEqual("a" * 40, result["headSha"])

    def test_target_selection_rejects_other_pr_sha_and_event_epoch(self) -> None:
        result = self.run_node(
            """
const sha = "b".repeat(40);
const trigger = {
  createdAt: Date.parse("2026-08-20T12:00:00Z"),
  headSha: sha,
  pullNumber: 42,
  triggerRunId: 201,
};
const base = {
  event: "pull_request",
  head_sha: sha,
  created_at: "2026-08-20T12:00:20Z",
  pull_requests: [{number: 42}],
  status: "in_progress",
};
const runs = [
  {...base, id: 201, name: "PR · Quality"},
  {...base, id: 202, name: "PR · Linux"},
  {...base, id: 203, name: "PR · Windows", head_sha: "c".repeat(40)},
  {...base, id: 204, name: "PR · macOS", pull_requests: [{number: 43}]},
  {...base, id: 205, name: "PR · Website", created_at: "2026-08-20T11:40:00Z"},
  {...base, id: 206, name: "Release"},
  {...base, id: 207, name: "PR · Quality"},
];
return action.selectTargetRuns(runs, trigger).map((run) => [run.id, run.name]);
"""
        )
        self.assertEqual([[202, "PR · Linux"]], result)

    def test_first_failure_is_preserved_and_only_active_siblings_cancel(self) -> None:
        result = self.run_node(
            """
const runs = [
  {id: 301, name: "PR · Quality", status: "in_progress"},
  {id: 302, name: "PR · Website", status: "completed"},
  {id: 303, name: "PR · Linux", status: "in_progress"},
  {id: 304, name: "PR · macOS", status: "queued"},
  {id: 305, name: "PR · Windows", status: "completed"},
];
const failure = action.findEarliestFailure([
  {run: runs[2], jobs: [{id: 2, name: "later", conclusion: "failure", completed_at: "2026-08-20T12:02:00Z"}]},
  {run: runs[0], jobs: [{id: 1, name: "first", conclusion: "failure", completed_at: "2026-08-20T12:01:00Z"}]},
]);
return {
  failure,
  cancellations: action.cancellableSiblingRuns(runs, failure.runId).map((run) => run.id),
};
"""
        )
        self.assertEqual(301, result["failure"]["runId"])
        self.assertEqual([303, 304], result["cancellations"])

    def test_all_five_runs_must_be_terminal_before_monitor_exits_cleanly(self) -> None:
        result = self.run_node(
            """
const complete = action.TARGET_WORKFLOWS.map((name, index) => ({id: index + 1, name, status: "completed"}));
return {
  complete: action.allTargetsTerminal(complete),
  missing: action.allTargetsTerminal(complete.slice(0, 4)),
  active: action.allTargetsTerminal(complete.map((run, index) => index === 2 ? {...run, status: "in_progress"} : run)),
};
"""
        )
        self.assertEqual(
            {"complete": True, "missing": False, "active": False},
            result,
        )

    def test_cancel_api_accepts_empty_202_response(self) -> None:
        result = self.run_node_async(
            """
global.fetch = async () => ({
  ok: true,
  status: 202,
  text: async () => "",
});
return await action.githubApi("token", "owner", "repo").cancelRun(123);
"""
        )
        self.assertTrue(result)

    def test_monitor_cancels_siblings_created_after_early_failure(self) -> None:
        result = self.run_node_async(
            """
const sha = "d".repeat(40);
const trigger = {
  createdAt: Date.parse("2026-08-20T12:00:00Z"),
  headSha: sha,
  pullNumber: 42,
  triggerRunId: 401,
};
const base = {
  event: "pull_request",
  head_sha: sha,
  created_at: "2026-08-20T12:00:05Z",
  pull_requests: [{number: 42}],
  status: "in_progress",
};
const quality = {...base, id: 401, name: "PR · Quality"};
const allRuns = action.TARGET_WORKFLOWS.map((name, index) => ({
  ...base,
  id: 401 + index,
  name,
}));
let polls = 0;
const cancelled = [];
const api = {
  listRuns: async () => (++polls === 1 ? [quality] : allRuns),
  listJobs: async (runId) => runId === 401 ? [{
    id: 501,
    name: "Plan",
    conclusion: "failure",
    completed_at: "2026-08-20T12:00:06Z",
  }] : [],
  cancelRun: async (runId) => { cancelled.push(runId); return true; },
};
const outcome = await action.monitor({
  api,
  trigger,
  pollSeconds: 0,
  maxMinutes: 1,
  log: () => {},
});
return {cancelled, failedRun: outcome.failure.runId, polls};
"""
        )
        self.assertEqual(401, result["failedRun"])
        self.assertEqual(2, result["polls"])
        self.assertEqual([402, 403, 404, 405], result["cancelled"])


if __name__ == "__main__":
    unittest.main()
