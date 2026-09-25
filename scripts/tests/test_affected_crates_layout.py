"""Relocated product crates must reach Cargo ownership matching.

`scripts/affected-crates.sh` used to skip every path that did not begin with
`crates/` or `tools/`, so a change under the supported relocated roots
`mesh/crates/` or `skippy/crates/` was filtered out before reverse-dependency
matching and silently selected no packages at all. These tests drive the real
script against a synthetic Cargo workspace so the relocated roots, their
dependents, the relocated Mesh UI path, and the legacy `crates/` handling are
all exercised together.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import shlex
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / 'scripts/affected-crates.sh'


def _metadata(root: Path) -> str:
    packages = [
        ('relocated-mesh', 'mesh/crates/relocated-mesh', []),
        ('relocated-skippy', 'skippy/crates/relocated-skippy', ['relocated-mesh']),
        ('legacy-mesh', 'crates/legacy-mesh', []),
    ]
    return json.dumps({
        'workspace_root': str(root),
        'packages': [
            {
                'name': name,
                'manifest_path': str(root / relative / 'Cargo.toml'),
                'dependencies': [{'name': dependency} for dependency in dependencies],
            }
            for name, relative, dependencies in packages
        ],
    })


class AffectedCratesLayoutTests(unittest.TestCase):
    def _run(self, changed: list[str]) -> dict:
        with tempfile.TemporaryDirectory(prefix='affected crates ') as tmp:
            root = Path(tmp)
            metadata_path = root / 'metadata.json'
            metadata_path.write_text(_metadata(root), encoding='utf-8')
            bin_dir = root / 'bin'
            bin_dir.mkdir()
            cargo = bin_dir / 'cargo'
            cargo.write_text(f'#!/bin/sh\ncat {shlex.quote(str(metadata_path))}\n', encoding='utf-8')
            cargo.chmod(0o755)
            env = {**os.environ, 'PATH': str(bin_dir) + os.pathsep + os.environ['PATH']}
            result = subprocess.run(
                ['bash', str(SCRIPT), *changed],
                cwd=ROOT,
                env=env,
                capture_output=True,
                text=True,
                timeout=60,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            return json.loads(result.stdout)

    def test_relocated_crate_paths_select_their_own_packages(self) -> None:
        for path, crate in (
            ('mesh/crates/relocated-mesh/src/lib.rs', 'relocated-mesh'),
            ('skippy/crates/relocated-skippy/src/lib.rs', 'relocated-skippy'),
        ):
            with self.subTest(path=path):
                payload = self._run([path])
                self.assertFalse(payload['all_rust'])
                self.assertFalse(payload['ui_changed'])
                self.assertIn(crate, payload['test_crates'])
                self.assertIn(crate, payload['affected'])

    def test_relocated_crate_path_selects_its_dependents(self) -> None:
        payload = self._run(['mesh/crates/relocated-mesh/src/lib.rs'])
        self.assertEqual(payload['test_crates'], ['relocated-mesh'])
        self.assertIn('relocated-skippy', payload['affected'])

    def test_legacy_crate_and_relocated_ui_handling_is_preserved(self) -> None:
        legacy = self._run(['crates/legacy-mesh/src/lib.rs'])
        self.assertIn('legacy-mesh', legacy['affected'])
        self.assertFalse(legacy['ui_changed'])

        relocated_ui = self._run(['mesh/crates/mesh-llm-ui/src/App.tsx'])
        self.assertTrue(relocated_ui['ui_changed'])
        self.assertEqual(relocated_ui['test_crates'], [])


if __name__ == '__main__':
    unittest.main()
