from __future__ import annotations

from pathlib import Path
import re
import unittest

import yaml


ROOT = Path(__file__).resolve().parents[2]
WORKFLOWS_DIR = ROOT / ".github" / "workflows"

# GitHub Actions permission levels that count as "granted". Anything absent
# from a permissions: block (or explicitly "none") does not propagate.
_GRANTED_LEVELS = {"read", "write"}

_LOCAL_USES_RE = re.compile(r"^\./\.github/workflows/([^@\s]+\.ya?ml)$")


def _load_workflow(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _granted_scopes(permissions) -> set[str] | None:
    """Return the set of scopes granted by a permissions: value, or None if
    no explicit block is present (meaning the check does not apply — GitHub
    falls back to the repo/org default token, which this test cannot see)."""
    if permissions is None:
        return None
    if isinstance(permissions, str):
        # permissions: read-all / write-all / {}
        return None if permissions in ("read-all", "write-all") else set()
    return {scope for scope, level in permissions.items() if level in _GRANTED_LEVELS}


def _requested_scopes(doc: dict) -> set[str] | None:
    """Every scope a reusable workflow can request: the workflow-level
    permissions block unioned with every explicit job-level block. GitHub
    evaluates permissions at both levels, so a callee that declares
    `packages: read` on one job alone still needs its caller to grant it --
    reading only the workflow-level block returns None there and silently
    skips the caller edge, which is exactly the hop this test exists to
    cover. Returns None when nothing explicit is declared anywhere, or when
    any level uses read-all/write-all (unenumerable -- do not assert)."""
    blocks = [doc.get("permissions")]
    jobs = doc.get("jobs") or {}
    if isinstance(jobs, dict):
        for job in jobs.values():
            if isinstance(job, dict):
                blocks.append(job.get("permissions"))

    requested: set[str] | None = None
    for block in blocks:
        if block is None:
            continue
        scopes = _granted_scopes(block)
        if scopes is None:
            return None
        requested = scopes if requested is None else requested | scopes
    return requested


class CiWorkflowPermissionContractTests(unittest.TestCase):
    """A called reusable workflow can only use permissions its caller job
    actually grants it — GitHub rejects the run at creation time otherwise
    (a zero-job startup_failure, invisible to actionlint and to PR CI since
    the caller's own PR run never requests the scope the callee needs until
    that specific callee executes). This walks every local
    `uses: ./.github/workflows/X.yml` edge in the repo and asserts the
    caller's effective permissions (job-level, else workflow-level) are a
    superset of what X.yml itself requests.
    """

    @classmethod
    def setUpClass(cls) -> None:
        cls.workflows: dict[str, dict] = {
            path.name: _load_workflow(path) for path in sorted(WORKFLOWS_DIR.glob("*.yml"))
        }
        cls.requested: dict[str, set[str] | None] = {
            name: _requested_scopes(doc)
            for name, doc in cls.workflows.items()
            if isinstance(doc.get(True, doc.get("on")), dict)
            and "workflow_call" in doc.get(True, doc.get("on"))
        }

    def test_every_local_reusable_call_site_grants_the_callees_permissions(self) -> None:
        violations = []
        for caller_name, doc in self.workflows.items():
            jobs = doc.get("jobs") or {}
            for job_name, job in jobs.items():
                uses = job.get("uses") if isinstance(job, dict) else None
                if not isinstance(uses, str):
                    continue
                match = _LOCAL_USES_RE.match(uses)
                if not match:
                    continue
                callee_name = match.group(1)
                requested = self.requested.get(callee_name)
                if not requested:
                    continue  # callee declares no permissions, or isn't a workflow_call target we tracked

                job_permissions = job.get("permissions")
                if job_permissions is not None:
                    granted = _granted_scopes(job_permissions)
                else:
                    granted = _granted_scopes(doc.get("permissions"))

                if granted is None:
                    continue  # no explicit block at the effective level — can't assert, GitHub uses the default token

                missing = requested - granted
                if missing:
                    violations.append(
                        f"{caller_name}:{job_name} -> {callee_name} requests "
                        f"{sorted(requested)} but only grants {sorted(granted)} "
                        f"(missing {sorted(missing)})"
                    )

        self.assertEqual(
            [],
            violations,
            "Reusable workflow call sites must grant every permission scope their "
            "callee requests, at every hop — GitHub does not let a called workflow "
            "reach past what its immediate caller job declares:\n" + "\n".join(violations),
        )


if __name__ == "__main__":
    unittest.main()
