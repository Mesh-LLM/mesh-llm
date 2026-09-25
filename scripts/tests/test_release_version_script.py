from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RELEASE_VERSION_SCRIPT = ROOT / "scripts" / "release-version.sh"


def product_sidecar(logical: str) -> Path:
    """The checked-in sidecar for a logical path, in whichever layout exists."""
    for candidate in (ROOT / logical, ROOT / "mesh" / logical, ROOT / "skippy" / logical):
        if candidate.exists():
            return candidate
    raise AssertionError(f"no versioned sidecar for {logical} in either source layout")


def known_versions_file() -> Path:
    script = RELEASE_VERSION_SCRIPT.read_text(encoding="utf-8")
    match = re.search(
        r'^known_versions_logical="(?P<path>[^"]+)"$',
        script,
        re.MULTILINE,
    )
    assert match is not None, "release-version.sh no longer assigns known_versions_logical"
    return product_sidecar(match.group("path"))


# Run update_known_mesh_versions in isolation. The script cannot be sourced --
# it validates its own arguments and performs a full release bump at load time
# -- so extract just the function definition and evaluate that.
_INVOKE_HELPER = """
set -euo pipefail
eval "$(sed -n '/^update_known_mesh_versions()/,/^}/p' "$1")"
update_known_mesh_versions "$2" "$3"
"""

class ReleaseVersionScriptTests(unittest.TestCase):
    def test_discovered_manifests_bump_path_dependencies_in_both_layouts(self) -> None:
        script = RELEASE_VERSION_SCRIPT.read_text()
        discovery = script[script.index('manifests=()'):script.index('versioned_files=()')]
        helper = re.search(r'update_versioned_path_dependency_versions\(\) \{.*?^}', script, re.S | re.M).group()
        for owners in [('crates',), ('mesh/crates', 'skippy/crates')]:
            with self.subTest(owners=owners), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                subprocess.run(['git', 'init', '-q', tmp], check=True)
                manifests = []
                for owner in (*owners, 'tools'):
                    path = root / owner / 'fixture' / 'Cargo.toml'
                    path.parent.mkdir(parents=True)
                    path.write_text('[dependencies]\nlocal = { path = "../local", version = "0.76.1" }\nexternal = "0.76.1"\n')
                    manifests.append(path)
                subprocess.run(['git', 'add', '.'], cwd=root, check=True)
                body = helper + '\nREPO_ROOT="$1"\n' + discovery + '\nfor manifest in "${manifests[@]}"; do update_versioned_path_dependency_versions "$REPO_ROOT/$manifest" 0.77.0; done\n'
                subprocess.run(['bash', '-euc', body, 'bash', tmp], cwd=root, check=True, capture_output=True)
                for path in manifests:
                    updated = path.read_text()
                    self.assertIn('path = "../local", version = "0.77.0"', updated)
                    self.assertIn('external = "0.76.1"', updated)

    def test_known_versions_file_defines_the_function_the_script_edits(self) -> None:
        """The script rewrites known_mesh_llm_versions() by regex.

        If the function moves to another module the substitution silently
        matches nothing and the release fails in the metadata job, after the
        tag has been chosen. Keep the path and the definition together.
        """
        target = known_versions_file()
        self.assertTrue(target.is_file(), f"missing {target}")
        self.assertIn(
            "fn known_mesh_llm_versions()",
            target.read_text(encoding="utf-8"),
        )

    def test_update_known_mesh_versions_prepends_the_new_version(self) -> None:
        source = known_versions_file().read_text(encoding="utf-8")
        with tempfile.TemporaryDirectory() as tmp:
            sample = Path(tmp) / "setting_schema.rs"
            sample.write_text(source, encoding="utf-8")
            subprocess.run(
                [
                    "bash",
                    "-c",
                    _INVOKE_HELPER,
                    "bash",
                    str(RELEASE_VERSION_SCRIPT),
                    str(sample),
                    "99.99.99-rc1",
                ],
                check=True,
                capture_output=True,
            )
            updated = sample.read_text(encoding="utf-8")
        self.assertIn('"99.99.99-rc1",', updated)

    def test_update_known_mesh_versions_fails_loudly_on_a_wrong_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            sample = Path(tmp) / "elsewhere.rs"
            sample.write_text("// no version list here\n", encoding="utf-8")
            result = subprocess.run(
                [
                    "bash",
                    "-c",
                    _INVOKE_HELPER,
                    "bash",
                    str(RELEASE_VERSION_SCRIPT),
                    str(sample),
                    "99.99.99-rc1",
                ],
                capture_output=True,
                text=True,
            )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("known_mesh_llm_versions()", result.stderr)


    def _resolver(self) -> str:
        script = RELEASE_VERSION_SCRIPT.read_text(encoding="utf-8")
        match = re.search(r"^resolve_product_path\(\) \{.*?^\}$", script, re.S | re.M)
        assert match is not None, "release-version.sh no longer defines resolve_product_path"
        return match.group()

    def _run_resolver(self, root: Path, logical: str) -> subprocess.CompletedProcess[str]:
        body = self._resolver() + '\nREPO_ROOT="$1"\nresolve_product_path "$2"\n'
        return subprocess.run(
            ["bash", "-euc", body, "bash", str(root), logical],
            capture_output=True,
            text=True,
            check=False,
        )

    def test_relocated_release_sidecars_resolve_from_the_same_source_layout(self) -> None:
        logical = "sdk/kotlin/build.gradle.kts"
        for prefix, expected in (("", logical), ("mesh/", "mesh/" + logical), ("skippy/", "skippy/" + logical)):
            with self.subTest(prefix=prefix), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                target = root / prefix / logical
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text("")
                result = self._run_resolver(root, logical)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(result.stdout.strip(), expected)

    def test_ambiguous_or_missing_release_sidecar_fails_before_writing(self) -> None:
        logical = "sdk/kotlin/build.gradle.kts"
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for prefix in ("", "mesh/"):
                target = root / prefix / logical
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text("")
            result = self._run_resolver(root, logical)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("ambiguous source layout", result.stderr)
        with tempfile.TemporaryDirectory() as tmp:
            result = self._run_resolver(Path(tmp), logical)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("missing required file", result.stderr)

    @staticmethod
    def _relocated(logical: str) -> str:
        if logical.startswith("crates/skippy-"):
            return "skippy/" + logical
        for prefix in ("crates/", "sdk/", "docs/", "website/"):
            if logical.startswith(prefix):
                return "mesh/" + logical
        return logical

    @staticmethod
    def _sidecar_content(relative: str) -> str:
        if relative.endswith("Cargo.lock"):
            return 'version = "0.76.1"\n'
        if relative.endswith("Cargo.toml"):
            return '[package]\nname = "fixture"\nversion = "0.76.1"\n'
        if relative.endswith("build.gradle.kts"):
            return 'plugins { }\nversion = "0.76.1"\n'
        if relative.endswith("package.json") or relative.endswith("package-lock.json"):
            return '{\n  "name": "fixture",\n  "version": "0.76.1",\n  "packages": {\n    "": {\n      "version": "0.76.1"\n    }\n  }\n}\n'
        if relative.endswith("setting_schema.rs"):
            return "fn known_mesh_llm_versions() -> &'static [&'static str] {\n    &[\n    ]\n}\n"
        if relative.endswith(".rs") or relative.endswith(".json"):
            return "release 0.76.1\n"
        return "release v0.76.1\n"

    @unittest.skipUnless(shutil.which("node") and shutil.which("git"), "needs node and git")
    def test_relocated_only_release_fixture_bumps_every_versioned_sidecar(self) -> None:
        """A checkout that only has mesh/ and skippy/ product trees still releases."""
        script = RELEASE_VERSION_SCRIPT.read_text(encoding="utf-8")
        literal_block = re.search(r"^literal_version_files=\(\n(?P<body>.*?)^\)$", script, re.S | re.M)
        assert literal_block is not None, "release-version.sh no longer lists literal_version_files"
        logical_paths = re.findall(r'^\s*"([^"]+)"$', literal_block.group("body"), re.MULTILINE)
        self.assertGreater(len(logical_paths), 15)
        logical_paths += [
            "sdk/kotlin/build.gradle.kts",
            "crates/mesh-llm-config/src/model/built_in_schema/setting_schema.rs",
        ]
        fixtures = {self._relocated(path) for path in logical_paths}
        fixtures.update({"mesh/crates/mesh-llm-config/Cargo.toml", "skippy/crates/skippy-runtime/Cargo.toml"})
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "scripts").mkdir()
            shutil.copy2(RELEASE_VERSION_SCRIPT, root / "scripts" / "release-version.sh")
            (root / "Cargo.toml").write_text(
                '[workspace]\nmembers = []\nresolver = "2"\n\n[workspace.package]\nversion = "0.76.1"\n'
            )
            for relative in sorted(fixtures):
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(self._sidecar_content(relative), encoding="utf-8")
            subprocess.run(["git", "init", "-q", str(root)], check=True)
            subprocess.run(["git", "add", "."], cwd=root, check=True)
            binaries = root / "bin"
            binaries.mkdir()
            cargo = binaries / "cargo"
            cargo.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
            cargo.chmod(0o755)
            env = {**os.environ, "PATH": f"{binaries}{os.pathsep}{os.environ['PATH']}"}
            result = subprocess.run(
                ["bash", str(root / "scripts" / "release-version.sh"), "0.77.0"],
                cwd=root,
                env=env,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            for relative in sorted(fixtures):
                with self.subTest(fixture=relative):
                    text = (root / relative).read_text(encoding="utf-8")
                    if relative.endswith("package.json") or relative.endswith("package-lock.json"):
                        self.assertEqual(json.loads(text)["version"], "0.77.0")
                        self.assertEqual(json.loads(text)["packages"][""]["version"], "0.77.0")
                    self.assertIn("0.77.0", text)
                    self.assertNotIn("0.76.1", text)
            self.assertIn("mesh/sdk/kotlin/build.gradle.kts", result.stdout)
            self.assertFalse((root / "crates").exists())
            self.assertFalse((root / "website").exists())


if __name__ == "__main__":
    unittest.main()
