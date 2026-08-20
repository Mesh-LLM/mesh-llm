"use strict";

const fs = require("fs");

const TARGET_WORKFLOWS = Object.freeze([
  "PR · Quality",
  "PR · Website",
  "PR · Linux",
  "PR · macOS",
  "PR · Windows",
]);
const TARGET_WORKFLOW_SET = new Set(TARGET_WORKFLOWS);
const FAILURE_CONCLUSIONS = new Set([
  "action_required",
  "failure",
  "stale",
  "startup_failure",
  "timed_out",
]);
const CANCELLABLE_STATUSES = new Set([
  "in_progress",
  "pending",
  "queued",
  "requested",
  "waiting",
]);
const REQUEST_TIMEOUT_MS = 30_000;
const LATE_SIBLING_WINDOW_MS = 120_000;
const PAGE_SIZE = 100;

function fail(message) {
  throw new Error(`PR sibling cancellation: ${message}`);
}

function positiveInteger(raw, name) {
  if (!/^[1-9][0-9]*$/.test(raw || "")) {
    fail(`${name} must be a positive integer`);
  }
  return Number(raw);
}

function parseRepository(repository) {
  const match = /^([^/]+)\/([^/]+)$/.exec(repository || "");
  if (!match) {
    fail("GITHUB_REPOSITORY is malformed");
  }
  return { owner: match[1], repo: match[2] };
}

function parseTrigger(payload, repository) {
  const workflowRun = payload?.workflow_run;
  if (!workflowRun || workflowRun.event !== "pull_request") {
    return null;
  }
  if (workflowRun.name !== "PR · Quality") {
    fail("triggering workflow is not PR · Quality");
  }
  if (!Number.isSafeInteger(workflowRun.id) || workflowRun.id <= 0) {
    fail("triggering workflow run ID is malformed");
  }
  if (!/^[0-9a-f]{40}$/.test(workflowRun.head_sha || "")) {
    fail("triggering workflow head SHA is malformed");
  }
  const createdAt = Date.parse(workflowRun.created_at || "");
  if (!Number.isFinite(createdAt)) {
    fail("triggering workflow creation time is malformed");
  }
  const pullRequests = Array.isArray(workflowRun.pull_requests)
    ? workflowRun.pull_requests
    : [];
  if (pullRequests.length !== 1) {
    fail("triggering workflow must identify exactly one pull request");
  }
  const pullNumber = pullRequests[0]?.number;
  if (!Number.isSafeInteger(pullNumber) || pullNumber <= 0) {
    fail("pull request number is malformed");
  }
  const payloadRepository = payload?.repository?.full_name;
  if (payloadRepository !== repository) {
    fail("event repository does not match GITHUB_REPOSITORY");
  }
  return {
    createdAt,
    headSha: workflowRun.head_sha,
    pullNumber,
    triggerRunId: workflowRun.id,
  };
}

function runBelongsToTrigger(run, trigger) {
  if (!TARGET_WORKFLOW_SET.has(run?.name)) return false;
  if (run.event !== "pull_request" || run.head_sha !== trigger.headSha) return false;
  const createdAt = Date.parse(run.created_at || "");
  if (!Number.isFinite(createdAt)) return false;
  // All five focused workflows are created by one PR event. Exclude an older
  // reopened/ready run of the same unchanged SHA from this cancellation set.
  if (Math.abs(createdAt - trigger.createdAt) > 120_000) return false;
  const pullRequests = Array.isArray(run.pull_requests) ? run.pull_requests : [];
  return pullRequests.some((pull) => pull?.number === trigger.pullNumber);
}

function selectTargetRuns(runs, trigger) {
  const selected = new Map();
  for (const run of runs || []) {
    if (!runBelongsToTrigger(run, trigger)) continue;
    const existing = selected.get(run.name);
    if (!existing || Number(run.id) > Number(existing.id)) {
      selected.set(run.name, run);
    }
  }
  const quality = selected.get("PR · Quality");
  if (quality && quality.id !== trigger.triggerRunId) {
    selected.delete("PR · Quality");
  }
  return TARGET_WORKFLOWS.flatMap((name) => {
    const run = selected.get(name);
    return run ? [run] : [];
  });
}

function findEarliestFailure(runJobs) {
  const failures = [];
  for (const { run, jobs } of runJobs) {
    for (const job of jobs || []) {
      if (!FAILURE_CONCLUSIONS.has(job?.conclusion)) continue;
      failures.push({
        completedAt: Date.parse(job.completed_at || "") || Number.MAX_SAFE_INTEGER,
        jobId: job.id,
        jobName: job.name,
        runId: run.id,
        workflowName: run.name,
      });
    }
  }
  failures.sort((left, right) => {
    if (left.completedAt !== right.completedAt) {
      return left.completedAt - right.completedAt;
    }
    if (left.runId !== right.runId) return left.runId - right.runId;
    return left.jobId - right.jobId;
  });
  return failures[0] || null;
}

function allTargetsTerminal(runs) {
  return runs.length === TARGET_WORKFLOWS.length
    && runs.every((run) => run.status === "completed");
}

function cancellableSiblingRuns(runs, preservedRunId) {
  return runs.filter((run) => (
    run.id !== preservedRunId && CANCELLABLE_STATUSES.has(run.status)
  ));
}

function sleep(milliseconds) {
  return new Promise((resolve) => setTimeout(resolve, milliseconds));
}

function githubApi(token, owner, repo) {
  const base = `https://api.github.com/repos/${encodeURIComponent(owner)}/${encodeURIComponent(repo)}`;

  async function request(path, options = {}) {
    let timeoutMs = REQUEST_TIMEOUT_MS;
    if (options.remainingMs) {
      const remainingMs = Math.floor(options.remainingMs());
      if (!Number.isFinite(remainingMs) || remainingMs <= 0) {
        fail("monitor deadline elapsed");
      }
      timeoutMs = Math.min(timeoutMs, remainingMs);
    }
    const response = await fetch(`${base}${path}`, {
      method: options.method || "GET",
      signal: AbortSignal.timeout(timeoutMs),
      headers: {
        Accept: "application/vnd.github+json",
        Authorization: `Bearer ${token}`,
        "User-Agent": "mesh-llm-pr-sibling-canceller",
        "X-GitHub-Api-Version": "2022-11-28",
      },
    });
    if (!response.ok) {
      const body = await response.text();
      const detail = body.slice(0, 300).replace(/[\r\n]+/g, " ");
      const error = new Error(`GitHub API ${response.status}: ${detail}`);
      error.status = response.status;
      throw error;
    }
    const body = await response.text();
    return body ? JSON.parse(body) : null;
  }

  async function listAll(pathForPage, field, options) {
    const values = [];
    for (let page = 1; ; page += 1) {
      const result = await request(pathForPage(page), options);
      const batch = result?.[field];
      if (!Array.isArray(batch)) {
        fail(`GitHub API response is missing ${field}`);
      }
      values.push(...batch);
      if (batch.length < PAGE_SIZE) return values;
    }
  }

  return {
    async listRuns(headSha, options = {}) {
      return listAll((page) => {
        const query = new URLSearchParams({
          event: "pull_request",
          head_sha: headSha,
          page: String(page),
          per_page: String(PAGE_SIZE),
        });
        return `/actions/runs?${query}`;
      }, "workflow_runs", options);
    },
    async listJobs(runId, options = {}) {
      return listAll(
        (page) => `/actions/runs/${runId}/jobs?filter=latest&per_page=${PAGE_SIZE}&page=${page}`,
        "jobs",
        options,
      );
    },
    async cancelRun(runId, options = {}) {
      try {
        await request(`/actions/runs/${runId}/cancel`, {
          ...options,
          method: "POST",
        });
        return true;
      } catch (error) {
        // A sibling can finish between the status read and cancel request.
        if (error.status === 409 || error.status === 422) return false;
        throw error;
      }
    },
  };
}

async function monitor({
  api,
  trigger,
  pollSeconds,
  maxMinutes,
  log = console.log,
  now = Date.now,
  sleepFn = sleep,
}) {
  const monitorDeadline = now() + maxMinutes * 60_000;
  const requestOptions = { remainingMs: () => monitorDeadline - now() };
  let consecutiveErrors = 0;
  let failure = null;
  let failureDeadline = null;
  while (now() < monitorDeadline) {
    try {
      const runs = selectTargetRuns(
        await api.listRuns(trigger.headSha, requestOptions),
        trigger,
      );
      if (!failure) {
        const runJobs = [];
        for (const run of runs) {
          if (run.status !== "queued") {
            runJobs.push({
              run,
              jobs: await api.listJobs(run.id, requestOptions),
            });
          }
        }
        failure = findEarliestFailure(runJobs);
      }
      if (failure) {
        if (failureDeadline === null) {
          failureDeadline = now() + LATE_SIBLING_WINDOW_MS;
          log(`::notice::Preserving failed ${failure.workflowName} run ${failure.runId}; cancelling sibling PR lanes for the same revision.`);
        }
        for (const sibling of cancellableSiblingRuns(runs, failure.runId)) {
          const cancelled = await api.cancelRun(sibling.id, requestOptions);
          log(`::notice::${cancelled ? "Cancelled" : "Sibling already terminal"}: ${sibling.name} run ${sibling.id}.`);
        }
        // Workflow records normally appear together, but a planning failure
        // can be nearly immediate. Poll for up to two additional minutes so a
        // late-created sibling from the same event epoch is still cancelled.
        if (runs.length === TARGET_WORKFLOWS.length || now() >= failureDeadline) {
          return { failure, runs };
        }
      }
      if (!failure && allTargetsTerminal(runs)) {
        log("::notice::All five PR validation lanes completed without a definitive job failure.");
        return { failure: null, runs };
      }
      consecutiveErrors = 0;
    } catch (error) {
      consecutiveErrors += 1;
      console.warn(`::warning::PR sibling monitor poll failed (${consecutiveErrors}/3): ${error.message}`);
      if (consecutiveErrors >= 3) throw error;
    }
    const remainingMs = monitorDeadline - now();
    if (remainingMs > 0) {
      await sleepFn(Math.min(pollSeconds * 1000, remainingMs));
    }
  }
  fail(`monitor timed out after ${maxMinutes} minutes`);
}

async function main() {
  const token = process.env.INPUT_TOKEN || "";
  if (!token) fail("token input is required");
  const pollSeconds = positiveInteger(process.env.INPUT_POLL_SECONDS, "poll_seconds");
  const maxMinutes = positiveInteger(process.env.INPUT_MAX_MINUTES, "max_minutes");
  const repository = process.env.GITHUB_REPOSITORY || "";
  const { owner, repo } = parseRepository(repository);
  const eventPath = process.env.GITHUB_EVENT_PATH || "";
  if (!eventPath) fail("GITHUB_EVENT_PATH is missing");
  const payload = JSON.parse(fs.readFileSync(eventPath, "utf8"));
  const trigger = parseTrigger(payload, repository);
  if (!trigger) {
    console.log("::notice::Trigger is not a pull_request workflow run; no monitoring is required.");
    return;
  }
  console.log(`::notice::Monitoring five PR validation lanes for PR #${trigger.pullNumber} at ${trigger.headSha}.`);
  await monitor({
    api: githubApi(token, owner, repo),
    trigger,
    pollSeconds,
    maxMinutes,
  });
}

module.exports = {
  TARGET_WORKFLOWS,
  allTargetsTerminal,
  cancellableSiblingRuns,
  findEarliestFailure,
  githubApi,
  monitor,
  parseTrigger,
  selectTargetRuns,
};

if (require.main === module) {
  main().catch((error) => {
    console.error(`::error::${error.message}`);
    process.exitCode = 1;
  });
}
