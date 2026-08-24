from __future__ import annotations

import json
import os
from pathlib import Path
import re
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "build-ui.sh"
UI_DIR = ROOT / "crates" / "mesh-llm-ui"
WORKFLOWS = ROOT / ".github" / "workflows"


def _parse_partial_version(text: str) -> tuple[int, int, int]:
    parts = (text.split(".") + ["0", "0"])[:3]
    return tuple(int(part) for part in parts)


def _version_satisfies_range(version: str, range_spec: str) -> bool:
    """Check a concrete x.y.z version against a small, space-separated-AND
    subset of node-semver ranges: `>=`, `<=`, `>`, `<`, and bare-equals
    comparators only.

    `engines.pnpm` in this repo has never needed `^`, `~`, or `||`; if it
    grows one, extend this rather than reaching for a dependency.
    """
    version_tuple = _parse_partial_version(version)
    comparator = re.compile(r"^(>=|<=|>|<|=)?(\d+(?:\.\d+){0,2})$")
    for token in range_spec.split():
        match = comparator.match(token)
        if match is None:
            raise ValueError(f"unsupported engines.pnpm range token: {token!r}")
        operator, bound = match.group(1) or "=", match.group(2)
        bound_tuple = _parse_partial_version(bound)
        satisfied = {
            ">=": version_tuple >= bound_tuple,
            "<=": version_tuple <= bound_tuple,
            ">": version_tuple > bound_tuple,
            "<": version_tuple < bound_tuple,
            "=": version_tuple == bound_tuple,
        }[operator]
        if not satisfied:
            return False
    return True


class SemverRangeTests(unittest.TestCase):
    """Direct coverage for the range check the engines.pnpm test relies on."""

    def test_satisfies_a_lower_bound(self) -> None:
        self.assertTrue(_version_satisfies_range("10.30.3", ">=10"))

    def test_rejects_a_pin_below_an_exclusive_upper_bound_stated_as_a_lower_bound(
        self,
    ) -> None:
        self.assertFalse(_version_satisfies_range("10.30.3", "<10"))

    def test_rejects_a_pin_outside_a_narrow_conjunctive_range(self) -> None:
        self.assertFalse(_version_satisfies_range("10.30.3", ">=10 <10.30.0"))

    def test_satisfies_a_wide_conjunctive_range(self) -> None:
        self.assertTrue(_version_satisfies_range("10.30.3", ">=10 <11"))


class BuildUiScriptTests(unittest.TestCase):
    def test_ui_pnpm_workspace_names_root_package(self) -> None:
        workspace = ROOT / "crates" / "mesh-llm-ui" / "pnpm-workspace.yaml"

        contents = workspace.read_text(encoding="utf-8")

        self.assertIn("packages:", contents)
        self.assertRegex(contents, r"(?m)^\s*-\s*['\"]?\.[\"']?\s*$")

    def ci_pnpm_majors(self) -> set[str]:
        """Every pnpm major `pnpm/action-setup` is asked to install in CI."""
        majors: set[str] = set()
        workflows = [*WORKFLOWS.glob("*.yml"), *WORKFLOWS.glob("*.yaml")]
        for workflow in sorted(workflows):
            for block in re.finditer(
                r"uses:\s*pnpm/action-setup@[^\n]*\n"
                r"(?P<rest>(?:[ \t]+(?![ \t]*-)[^\n]*\n)*)",
                workflow.read_text(encoding="utf-8"),
            ):
                version = re.search(
                    r"(?m)^[ \t]*version:\s*[\"']?(\d+)", block.group("rest")
                )
                if version:
                    majors.add(version.group(1))
        return majors

    def test_ui_package_declares_the_pnpm_major_that_ci_installs(self) -> None:
        """The lockfile only installs under pnpm 10+, so the repo must say so.

        `overrides` and `allowBuilds` live in `pnpm-workspace.yaml`, which is a
        pnpm 10 layout. pnpm 9 does not read them, so it computes a different
        overrides config than the lockfile records and stops with
        `ERR_PNPM_LOCKFILE_CONFIG_MISMATCH` — a message that says nothing about
        pnpm versions. Declaring the requirement turns that into
        `ERR_PNPM_UNSUPPORTED_ENGINE`, which names the actual problem.
        """
        manifest = json.loads((UI_DIR / "package.json").read_text(encoding="utf-8"))

        required = manifest.get("engines", {}).get("pnpm")
        self.assertIsNotNone(required, "package.json must declare engines.pnpm")
        declared_major = re.search(r"(\d+)", required)
        self.assertIsNotNone(declared_major)
        assert declared_major is not None

        pinned = manifest.get("packageManager", "")
        self.assertRegex(
            pinned,
            r"^pnpm@\d+\.\d+\.\d+$",
            "packageManager must pin an exact pnpm version for corepack",
        )

        ci_majors = self.ci_pnpm_majors()
        self.assertTrue(ci_majors, "no pnpm/action-setup version found to compare against")
        self.assertEqual(
            ci_majors,
            {declared_major.group(1)},
            "engines.pnpm and CI's pnpm/action-setup version have drifted apart",
        )
        pinned_version = pinned.removeprefix("pnpm@")
        self.assertTrue(
            _version_satisfies_range(pinned_version, required),
            f"packageManager {pinned} does not satisfy the declared engines.pnpm range {required!r}",
        )

    def test_ui_npmrc_enforces_the_declared_engine(self) -> None:
        """`engines` is advisory to pnpm unless engine-strict is turned on.

        Without this the field is documentation only and a pnpm 9 operator
        still gets the unrelated lockfile-mismatch error.
        """
        npmrc = UI_DIR / ".npmrc"
        self.assertTrue(npmrc.exists(), "crates/mesh-llm-ui/.npmrc is missing")
        self.assertRegex(
            npmrc.read_text(encoding="utf-8"),
            r"(?m)^engine-strict\s*=\s*true\s*$",
        )

    def test_mixed_case_release_profile_uses_release_ui_env(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ui_dir = Path(tmp) / "ui"
            self._write_up_to_date_ui_fixture(ui_dir, profile="release", debug_ui="false")

            env = os.environ.copy()
            env["MESH_LLM_BUILD_PROFILE"] = "ReLeAsE"
            for key in ("VITE_BASE_PATH", "VITE_ROUTER_BASE_PATH", "VITE_STORAGE_NAMESPACE"):
                env.pop(key, None)

            result = subprocess.run(
                ["bash", str(SCRIPT), str(ui_dir)],
                cwd=ROOT,
                env=env,
                text=True,
                capture_output=True,
                check=False,
            )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("profile: release", result.stdout)
        self.assertIn("debug UI: false", result.stdout)

    def _write_up_to_date_ui_fixture(self, ui_dir: Path, *, profile: str, debug_ui: str) -> None:
        ui_dir.mkdir()
        for relative in (
            "package.json",
            "pnpm-lock.yaml",
            "vite.config.ts",
            "tsconfig.json",
            "tsconfig.app.json",
            "tsconfig.node.json",
            "biome.json",
            "index.html",
        ):
            (ui_dir / relative).write_text("{}\n", encoding="utf-8")
        (ui_dir / "src").mkdir()
        (ui_dir / "public").mkdir()

        dist_dir = ui_dir / "dist"
        dist_dir.mkdir()
        (dist_dir / "asset.js").write_text("// built\n", encoding="utf-8")
        (dist_dir / ".mesh-llm-ui-build-env").write_text(
            f"MESH_LLM_BUILD_PROFILE={profile}\n"
            f"VITE_MESH_LLM_DEBUG_UI={debug_ui}\n"
            "VITE_BASE_PATH=\n"
            "VITE_ROUTER_BASE_PATH=\n"
            "VITE_STORAGE_NAMESPACE=\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    unittest.main()
