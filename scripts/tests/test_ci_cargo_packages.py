"""Old protected plans must retain complete coverage during extraction."""
from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import tempfile

import yaml
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location('ci_cargo_packages', ROOT / 'scripts/ci-cargo-packages.py')
COMPAT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(COMPAT)
FIXTURES = ROOT / 'scripts/tests/fixtures/ci-source-layout'
LEGACY = json.loads((FIXTURES / 'legacy-packages.json').read_text())
EXTRACTED = json.loads((FIXTURES / 'extracted-packages.json').read_text())


class CargoPackageCompatibilityTests(unittest.TestCase):
    def test_each_extracted_workspace_member_retains_exactly_one_legacy_batch(self):
        # Frozen inventories: protected main d71ceeaf1 and extraction 5a874ba87.
        # These are independent actual workspace censuses, not generated from
        # the mapping being tested. Resolve separate batches to catch overlap.
        resolved = [name for old in LEGACY for name in COMPAT.resolve([old], LEGACY, set(EXTRACTED), "legacy")]
        self.assertCountEqual(resolved, EXTRACTED)
        self.assertEqual(len(resolved), len(set(resolved)))

    def test_unchanged_old_and_new_plans_preserve_their_batches(self):
        for source in (LEGACY, EXTRACTED):
            for name in source:
                self.assertEqual(COMPAT.resolve([name], source, set(source), "legacy" if source is LEGACY else "current"), [name])

    def test_reused_package_name_preserves_both_different_owners(self):
        self.assertEqual(COMPAT.resolve(['model-package'], LEGACY, set(EXTRACTED), "legacy"), ['skippy-model-package'])
        self.assertEqual(COMPAT.resolve(['skippy-model-package'], LEGACY, set(EXTRACTED), "legacy"), ['skippy-package-builder'])
        self.assertEqual(COMPAT.resolve(['skippy-model-package'], EXTRACTED, set(EXTRACTED), "current"), ['skippy-model-package'])

    def test_partial_plan_with_reused_name_has_explicit_generation(self):
        batch = ['skippy-model-package']
        self.assertEqual(COMPAT.resolve(batch, batch, set(EXTRACTED), 'legacy'), ['skippy-package-builder'])
        self.assertEqual(COMPAT.resolve(batch, batch, set(EXTRACTED), 'current'), batch)

    def test_missing_successor_is_an_error_not_a_skipped_test(self):
        available = set(EXTRACTED) - {'mesh-llm-membership'}
        with self.assertRaisesRegex(ValueError, 'missing source owners'):
            COMPAT.resolve(['mesh-llm-host-runtime'], LEGACY, available, "legacy")

    def test_unmapped_planned_package_defers_to_the_workspace_filter(self):
        # A protected plan may name a crate that this revision does not have (a
        # member added on the default branch after the branch was cut). Only a
        # mapped successor proves an incomplete extraction, so an unmapped name
        # is passed through for the executor's workspace filter to drop and
        # annotate instead of failing the whole batch.
        self.assertEqual(
            COMPAT.resolve(['unknown'], ['unknown'], set(EXTRACTED), 'legacy'),
            ['unknown'],
        )
        self.assertEqual(
            COMPAT.resolve(['mesh-llm-analytics'], ['mesh-llm-analytics'], set(EXTRACTED), 'legacy'),
            ['mesh-llm-analytics'],
        )
        self.assertEqual(
            COMPAT.resolve(['unknown'], ['unknown'], set(LEGACY), 'legacy'),
            ['unknown'],
        )

    def test_batch_cannot_add_a_package_outside_its_plan(self):
        with self.assertRaisesRegex(ValueError, 'outside the protected plan'):
            COMPAT.resolve(['skippy-cli'], ['mesh-llm'], set(EXTRACTED), 'current')

    def test_invalid_duplicate_or_shell_shaped_names_are_rejected(self):
        for names in ([], ['mesh-llm', 'mesh-llm'], ['--workspace'], ['$(touch /tmp/unwanted)'], ['x\ny']):
            with self.subTest(names=names), self.assertRaises(ValueError):
                COMPAT.resolve(names, names, set(EXTRACTED), "current")


    def test_safetensors_step_compiles_selected_owner_once_and_requires_exact_test(self):
        job = yaml.safe_load((ROOT / '.github/workflows/ci-rust-tests-slice.yml').read_text())['jobs']['safetensors_runtime_smoke']
        script = next(step['run'] for step in job['steps'] if step.get('id') == 'safetensors_smoke_test')
        for adapter in (False, True):
            for present in (False, True):
                with self.subTest(adapter=adapter, present=present), tempfile.TemporaryDirectory(prefix='smoke step ') as temporary:
                    root = Path(temporary)
                    binaries = root / 'bin'
                    binaries.mkdir()
                    crate = 'mesh-llm-skippy-adapter' if adapter else 'mesh-llm-host-runtime'
                    prefix = 'config::hardware_translation_tests::' if adapter else 'inference::skippy::resolver::tests::'
                    test = prefix + 'safetensors_checkpoint_reaches_mesh_host_runtime'
                    metadata = {'workspace_members': [crate], 'packages': [{'id': crate, 'name': crate}]}
                    (root / 'metadata.json').write_text(json.dumps(metadata))
                    executable = root / 'smoke-test'
                    executable.write_text('#!/bin/sh\nprintf "%s\\n" "$TEST_LISTING"\n')
                    executable.chmod(0o755)
                    artifact = {'reason': 'compiler-artifact', 'target': {'name': crate.replace('-', '_')}, 'profile': {'test': True}, 'executable': str(executable)}
                    (root / 'artifact.json').write_text(json.dumps(artifact))
                    cargo = binaries / 'cargo'
                    cargo.write_text('#!/bin/sh\ncase "$1" in metadata) cat "$FIXTURE/metadata.json";; test) printf "%s\\n" "$*" >> "$FIXTURE/calls"; cat "$FIXTURE/artifact.json";; *) exit 9;; esac\n')
                    cargo.chmod(0o755)
                    env = {**os.environ, 'PATH': str(binaries) + os.pathsep + os.environ['PATH'], 'FIXTURE': str(root), 'GITHUB_OUTPUT': str(root / 'output'), 'TEST_LISTING': test + ': test' if present else 'other: test'}
                    result = subprocess.run(['bash', '-euo', 'pipefail', '-c', script], cwd=root, env=env, capture_output=True, text=True)
                    self.assertEqual(result.returncode, 0 if present else 1, result.stderr)
                    calls = (root / 'calls').read_text().splitlines()
                    self.assertEqual(len(calls), 1)
                    self.assertIn('-p ' + crate, calls[0])
                    if present:
                        self.assertIn('test_name=' + test, (root / 'output').read_text())

if __name__ == '__main__':
    unittest.main()
