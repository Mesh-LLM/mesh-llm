import importlib.util
import os
from unittest.mock import patch
from pathlib import Path
import tempfile
import unittest

import yaml

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location('cleanup', ROOT / 'scripts/cleanup-self-hosted.py')
C = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(C)


class CleanupTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.base = Path(tmp.name).resolve()
        self.workspace = self.base / 'workspace'
        self.workspace.mkdir()
        self.temp = self.base / 'temp'
        self.temp.mkdir()
        self.env = dict(GITHUB_WORKSPACE=str(self.workspace), RUNNER_TEMP=str(self.temp),
                        CANARY_SOURCE_ROOT=str(self.workspace), GITHUB_RUN_ID='123',
                        GITHUB_RUN_ATTEMPT='2', CANARY_PASS_ID='repair-1', CANARY_SHARD_INDEX='3',
                        CLEANUP_ARTIFACT_PATH='ci-artifacts/linux',
                        CLEANUP_BINARY_PATH='target/release/mesh-llm')

    def seed(self, path):
        path.mkdir(parents=True, exist_ok=True)
        (path / 'payload').write_text('keep or remove')

    def test_profiles_remove_outputs_and_preserve_unrelated_files(self):
        for profile in ('build', 'family', 'replay', 'cuda-release', 'smoke', 'runner-contract'):
            with self.subTest(profile=profile):
                paths = C.targets(self.env, profile, True, True)
                for _, path in paths:
                    self.seed(path)
                sentinels = [self.workspace / 'source', self.base / 'model-cache',
                             self.workspace / '.deps/canary-input-999-1-repair-1-3']
                for path in sentinels:
                    self.seed(path)
                C.cleanup(paths)
                C.cleanup(paths)  # repeated cleanup and missing outputs are harmless
                self.assertTrue(all(not p.exists() for _, p in paths))
                self.assertTrue(all((p / 'payload').exists() for p in sentinels))

    def test_failed_uploads_retain_only_recovery_outputs(self):
        for profile in ('build', 'family', 'replay', 'cuda-release'):
            all_paths = C.targets(self.env, profile, True, True)
            safe_paths = C.targets(self.env, profile, False, False)
            retained = set(all_paths) - set(safe_paths)
            self.assertTrue(retained)
            for _, path in all_paths:
                self.seed(path)
            C.cleanup(safe_paths)
            self.assertTrue(all((p / 'payload').exists() for _, p in retained))
            C.cleanup(all_paths)

    def test_leaf_symlink_does_not_delete_external_data(self):
        outside = self.base / 'outside'
        self.seed(outside)
        link = self.workspace / 'target'
        link.symlink_to(outside, target_is_directory=True)
        C.cleanup([(self.workspace, link)])
        self.assertTrue((outside / 'payload').exists())
        self.assertFalse(link.is_symlink())

    def test_parent_symlink_rejects_entire_cleanup_before_deletion(self):
        outside = self.base / 'outside'
        self.seed(outside / 'debug')
        (self.workspace / 'target').symlink_to(outside, target_is_directory=True)
        safe = self.workspace / 'safe'
        self.seed(safe)
        with self.assertRaises(ValueError):
            C.cleanup([(self.workspace, safe), (self.workspace, self.workspace / 'target/debug')])
        self.assertTrue((safe / 'payload').exists())
        self.assertTrue((outside / 'debug/payload').exists())

    def test_invalid_identities_and_sources_fail_closed(self):
        for key, value in [('CANARY_PASS_ID', '../../'), ('GITHUB_RUN_ID', ''),
                           ('CANARY_SHARD_INDEX', '../3'), ('CANARY_SOURCE_ROOT', str(self.base))]:
            with self.subTest(key=key), self.assertRaises(ValueError):
                C.targets({**self.env, key: value}, 'family', True, True)

    def test_selected_source_paths_stay_in_selected_checkout(self):
        env = {**self.env, 'CANARY_SOURCE_ROOT': str(self.workspace / 'canary-source')}
        paths = [p for _, p in C.targets(env, 'family', True, True)]
        self.assertIn(self.workspace / 'canary-source/target/debug', paths)
        self.assertNotIn(self.workspace / 'target/debug', paths)

    def test_smoke_rejects_sources_external_paths_and_traversal(self):
        for value in ('', 'scripts', '/tmp/output', 'target/../../source'):
            with self.subTest(value=value), self.assertRaises((ValueError, IndexError)):
                C.targets({**self.env, 'CLEANUP_ARTIFACT_PATH': value}, 'smoke', False, False)

    def test_every_self_hosted_job_has_final_cleanup_for_every_outcome(self):
        found = set()
        for file in (ROOT / '.github/workflows').glob('*.yml'):
            doc = yaml.safe_load(file.read_text())
            for name, job in doc.get('jobs', {}).items():
                runner = str(job.get('runs-on', ''))
                matrix = job.get('strategy', {}).get('matrix', {})
                rows = matrix.get('include', []) if isinstance(matrix, dict) else []
                custom_matrix = any(row.get('runner', '').startswith('mesh-llm-') for row in rows)
                if 'self-hosted' not in runner and not custom_matrix:
                    continue
                found.add((file.name, name))
                step = job['steps'][-1]
                self.assertIn('cleanup-self-hosted.py', step.get('run', ''))
                for status in ('success()', 'failure()', 'cancelled()'):
                    self.assertIn(status, step.get('if', ''))
                self.assertEqual(step['timeout-minutes'], 5)
        self.assertEqual(len(found), 6)

    def test_replay_builds_use_the_job_owned_worktree_root(self):
        spec = importlib.util.spec_from_file_location('replay_params', ROOT / 'scripts/agentic-replay-params.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        directory = str(self.temp / 'agentic-replay-worktrees')
        with patch.dict(os.environ, AGENTIC_REPLAY_WORKTREE_ROOT=directory):
            command = module.replay_command(ROOT / 'ci/agentic-replay-nightly/matrix.json',
                                            'granite-3.1-2b', ['main=HEAD'], Path('dataset'), Path('output'))
        self.assertEqual(command[command.index('--worktree-root') + 1], directory)
        self.assertIn((self.temp, Path(directory)), C.targets(self.env, 'replay', False, False))

    def test_release_cache_save_precedes_cleanup(self):
        doc = yaml.safe_load((ROOT / '.github/workflows/release.yml').read_text())
        steps = doc['jobs']['build_native_runtime_linux_x86_64_cuda']['steps']
        self.assertIn('actions/cache/save@', steps[-2]['uses'])
        self.assertTrue(any('actions/cache/restore@' in s.get('uses', '') for s in steps))


if __name__ == '__main__':
    unittest.main()
