from __future__ import annotations

from pathlib import Path
import re
import unittest

import yaml


ROOT = Path(__file__).resolve().parents[2]
WORKFLOWS_DIR = ROOT / ".github" / "workflows"

# GitHub Actions permission levels, ranked. A caller satisfies a callee only if
# it grants at least the level the callee asks for -- `contents: read` does NOT
# satisfy `contents: write`, and GitHub rejects that downgrade at run creation
# exactly like a missing scope. Comparing scope *names* alone accepts it
# silently, so the level has to survive into the comparison.
_LEVEL_RANK = {"none": 0, "read": 1, "write": 2}
_GRANTED_LEVELS = ("read", "write")

_LOCAL_USES_RE = re.compile(r"^\./\.github/workflows/([^@\s]+\.ya?ml)$")


def _load_workflow(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _rank(level: str | None) -> int:
    return _LEVEL_RANK.get(level or "none", 0)


class _AllScopes:
    """`permissions: read-all` / `write-all` -- one blanket level across every
    scope. Modelled rather than skipped, because as a *grant* it is perfectly
    enumerable: `write-all` satisfies any request, and `read-all` satisfies a
    `read` request but not a `write` one. (As a *request* it is still opaque --
    see _requested_scopes -- since it names no scopes to check against.)"""

    def __init__(self, level: str) -> None:
        self.level = level

    def get(self, scope: str, default: str = "none") -> str:
        return self.level

    def __repr__(self) -> str:
        return f"{{every scope: {self.level}}}"


def _format(scopes) -> str:
    if isinstance(scopes, _AllScopes):
        return repr(scopes)
    return "{" + ", ".join(f"{scope}: {level}" for scope, level in sorted(scopes.items())) + "}"


def _scope_levels(permissions) -> dict[str, str] | _AllScopes | None:
    """Map each scope in a permissions: value to the level it is set to, or
    None if no explicit block is present (meaning the check does not apply --
    GitHub falls back to the repo/org default token, which this test cannot
    see). `read-all`/`write-all` become an _AllScopes blanket level. Scopes set
    to `none` are dropped: they neither grant nor request."""
    if permissions is None:
        return None
    if isinstance(permissions, str):
        if permissions == "read-all":
            return _AllScopes("read")
        if permissions == "write-all":
            return _AllScopes("write")
        return {}
    return {scope: level for scope, level in permissions.items() if level in _GRANTED_LEVELS}


def _requested_scopes(doc: dict) -> dict[str, str] | None:
    """Every scope a reusable workflow can request, at the highest level it
    asks for anywhere: the workflow-level permissions block merged with every
    explicit job-level block. GitHub evaluates permissions at both levels, so a
    callee that declares `packages: read` on one job alone still needs its
    caller to grant it -- reading only the workflow-level block returns None
    there and silently skips the caller edge, which is exactly the hop this
    test exists to cover. Where a scope appears in more than one block the
    highest level wins, because the caller has to satisfy the strictest
    request. Returns None when nothing explicit is declared anywhere, or when
    any level uses read-all/write-all (unenumerable -- do not assert)."""
    blocks = [doc.get("permissions")]
    jobs = doc.get("jobs") or {}
    if isinstance(jobs, dict):
        for job in jobs.values():
            if isinstance(job, dict):
                blocks.append(job.get("permissions"))

    requested: dict[str, str] | None = None
    for block in blocks:
        if block is None:
            continue
        levels = _scope_levels(block)
        if levels is None or isinstance(levels, _AllScopes):
            # Unenumerable as a request: it names no scopes to hold the caller
            # to, so asserting anything here would be invention.
            return None
        if requested is None:
            requested = {}
        for scope, level in levels.items():
            if _rank(level) > _rank(requested.get(scope)):
                requested[scope] = level
    return requested


def _unsatisfied(requested: dict[str, str], granted) -> list[str]:
    """Scopes the caller fails to cover, either because it omits them or
    because it grants a weaker level than the callee requests."""
    return [
        f"{scope}: needs {level}, has {granted.get(scope, 'none')}"
        for scope, level in sorted(requested.items())
        if _rank(level) > _rank(granted.get(scope))
    ]


class CiWorkflowPermissionContractTests(unittest.TestCase):
    """A called reusable workflow can only use permissions its caller job
    actually grants it — GitHub rejects the run at creation time otherwise
    (a zero-job startup_failure, invisible to actionlint and to PR CI since
    the caller's own PR run never requests the scope the callee needs until
    that specific callee executes). This walks every local
    `uses: ./.github/workflows/X.yml` edge in the repo and asserts the
    caller's effective permissions (job-level, else workflow-level) cover
    every scope X.yml requests, at no less than the level it requests.
    """

    @classmethod
    def setUpClass(cls) -> None:
        cls.workflows: dict[str, dict] = {
            path.name: _load_workflow(path) for path in sorted(
                [*WORKFLOWS_DIR.glob("*.yml"), *WORKFLOWS_DIR.glob("*.yaml")]
            )
        }
        cls.requested: dict[str, dict[str, str] | None] = {
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
                    granted = _scope_levels(job_permissions)
                else:
                    granted = _scope_levels(doc.get("permissions"))

                if granted is None:
                    continue  # no explicit block at the effective level — can't assert, GitHub uses the default token

                unsatisfied = _unsatisfied(requested, granted)
                if unsatisfied:
                    violations.append(
                        f"{caller_name}:{job_name} -> {callee_name} requests "
                        f"{_format(requested)} but grants {_format(granted)} "
                        f"({'; '.join(unsatisfied)})"
                    )

        self.assertEqual(
            [],
            violations,
            "Reusable workflow call sites must grant every permission scope their "
            "callee requests, at every hop and at no less than the requested level — "
            "GitHub does not let a called workflow reach past what its immediate "
            "caller job declares:\n" + "\n".join(violations),
        )

    def test_a_read_grant_does_not_satisfy_a_write_request(self) -> None:
        """Regression: the contract compares levels, not just scope names. A
        caller granting `contents: read` to a callee that needs
        `contents: write` is the same run-creation failure as omitting the
        scope, so it must be reported, not passed over."""
        requested = _requested_scopes({"permissions": {"contents": "write"}})
        self.assertEqual({"contents": "write"}, requested)

        self.assertEqual(
            ["contents: needs write, has read"],
            _unsatisfied(requested, {"contents": "read"}),
            "a read grant must not satisfy a write request",
        )
        self.assertEqual(
            [],
            _unsatisfied(requested, {"contents": "write"}),
            "an equal grant must satisfy the request",
        )
        self.assertEqual(
            [],
            _unsatisfied(_requested_scopes({"permissions": {"contents": "read"}}), {"contents": "write"}),
            "a stronger grant must satisfy a weaker request",
        )
        self.assertEqual(
            ["contents: needs read, has none"],
            _unsatisfied(_requested_scopes({"permissions": {"contents": "read"}}), {"packages": "read"}),
            "an omitted scope must still be reported",
        )

    def test_all_scope_caller_grants_are_modelled_not_skipped(self) -> None:
        """A caller declaring `read-all`/`write-all` used to return None and
        skip the edge entirely. `write-all` genuinely satisfies anything;
        `read-all` genuinely does not satisfy a `write` request, and skipping
        it hid exactly the run-creation failure this test exists to catch."""
        needs_write = _requested_scopes({"permissions": {"contents": "write"}})
        needs_read = _requested_scopes({"permissions": {"contents": "read"}})

        write_all = _scope_levels("write-all")
        read_all = _scope_levels("read-all")
        self.assertIsInstance(write_all, _AllScopes)
        self.assertIsInstance(read_all, _AllScopes)

        self.assertEqual([], _unsatisfied(needs_write, write_all), "write-all satisfies write")
        self.assertEqual([], _unsatisfied(needs_read, write_all), "write-all satisfies read")
        self.assertEqual([], _unsatisfied(needs_read, read_all), "read-all satisfies read")
        self.assertEqual(
            ["contents: needs write, has read"],
            _unsatisfied(needs_write, read_all),
            "read-all must NOT satisfy a write request",
        )

    def test_an_all_scope_callee_request_stays_unenumerable(self) -> None:
        """The same forms are opaque in the other direction: a callee asking
        `write-all` names no scopes, so there is nothing to hold the caller to
        and the contract declines to assert rather than invent a scope list."""
        self.assertIsNone(_requested_scopes({"permissions": "write-all"}))
        self.assertIsNone(_requested_scopes({"permissions": "read-all"}))
        self.assertIsNone(
            _requested_scopes({
                "permissions": {"contents": "read"},
                "jobs": {"a": {"permissions": "write-all"}},
            }),
            "one all-scope job block makes the whole request unenumerable",
        )

    def test_the_strictest_level_wins_when_a_scope_is_declared_twice(self) -> None:
        """A callee asking `contents: read` at the workflow level and
        `contents: write` on one job needs write from its caller. Merging by
        scope name alone would keep whichever block was seen last."""
        doc = {
            "permissions": {"contents": "read"},
            "jobs": {
                "a": {"permissions": {"contents": "write"}},
                "b": {"permissions": {"packages": "read", "contents": "none"}},
            },
        }
        self.assertEqual({"contents": "write", "packages": "read"}, _requested_scopes(doc))


if __name__ == "__main__":
    unittest.main()
