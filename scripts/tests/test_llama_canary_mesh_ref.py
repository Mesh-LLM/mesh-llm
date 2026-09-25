from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location('mesh_ref', ROOT / 'scripts/llama-canary-resolve-mesh-ref.py')
R = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(R)


class MeshRefTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.repo = Path(self.temp.name) / 'origin'
        self.repo.mkdir()
        self.git(self.repo, 'init', '-b', 'main')
        self.git(self.repo, 'config', 'user.name', 'Test')
        self.git(self.repo, 'config', 'user.email', 'test@example.invalid')
        pin = self.repo / 'third_party/llama.cpp/upstream.txt'
        pin.parent.mkdir(parents=True)
        pin.write_text('a' * 40 + '\n')
        self.git(self.repo, 'add', '.')
        self.git(self.repo, 'commit', '-m', 'fixture')
        self.initial = self.git(self.repo, 'rev-parse', 'HEAD')
        self.git(self.repo, 'branch', 'candidate')
        self.checkout = Path(self.temp.name) / 'checkout'
        self.git(self.repo, 'clone', '--depth=1', self.repo.as_uri(), str(self.checkout))

    def git(self, root, *args):
        return subprocess.check_output(['git', '-C', str(root), *args], text=True, stderr=subprocess.DEVNULL).strip()

    def test_branch_is_frozen_with_existing_pin_and_no_repair(self):
        result = R.resolve(self.checkout, 'candidate')
        self.assertEqual(result, {'mesh_source': self.initial, 'upstream': 'a'*40,
                                 'changed': 'false', 'mode': 'pinned-build', 'certify': 'true'})
        pin = self.repo / 'third_party/llama.cpp/upstream.txt'
        pin.write_text('b'*40 + '\n')
        self.git(self.repo, 'commit', '-am', 'new pin')
        self.git(self.repo, 'branch', '-f', 'candidate', 'HEAD')
        self.assertEqual(result['mesh_source'], self.initial)
        latest = R.resolve(self.checkout, 'refs/heads/candidate')
        self.assertNotEqual(latest['mesh_source'], result['mesh_source'])
        self.assertEqual(latest['upstream'], 'b'*40)
        self.assertEqual(R.resolve(self.checkout, self.initial)['mesh_source'], self.initial)

    def test_invalid_or_external_refs_and_upstream_override_fail(self):
        for ref in ('', ' candidate', 'refs/pull/1977/head', 'refs/tags/v1', 'main~1', '--upload-pack=evil'):
            with self.subTest(ref=ref), self.assertRaises((ValueError, subprocess.CalledProcessError)):
                R.resolve(self.checkout, ref)
        with self.assertRaisesRegex(ValueError, 'cannot be combined'):
            R.resolve(self.checkout, 'candidate', 'latest')
        with self.assertRaises(subprocess.CalledProcessError):
            R.resolve(self.checkout, 'c'*40)

    def test_commit_not_reachable_from_branch_is_rejected(self):
        self.git(self.checkout, 'config', 'user.name', 'Test')
        self.git(self.checkout, 'config', 'user.email', 'test@example.invalid')
        self.git(self.checkout, 'commit', '--allow-empty', '-m', 'local only')
        head = self.git(self.checkout, 'rev-parse', 'HEAD')
        with self.assertRaisesRegex(ValueError, 'not reachable'):
            R.resolve(self.checkout, head)

    def test_invalid_selected_pin_fails(self):
        (self.repo / 'third_party/llama.cpp/upstream.txt').write_text('latest\n')
        self.git(self.repo, 'commit', '-am', 'bad pin')
        with self.assertRaisesRegex(ValueError, 'invalid llama.cpp pin'):
            R.resolve(self.checkout, 'main')

    def test_relocated_pin_and_ambiguous_or_missing_pin(self):
        legacy = self.repo / 'third_party/llama.cpp/upstream.txt'
        relocated = self.repo / 'skippy/third_party/llama.cpp/upstream.txt'
        relocated.parent.mkdir(parents=True)
        legacy.rename(relocated)
        self.git(self.repo, 'add', '-A')
        self.git(self.repo, 'commit', '-m', 'relocate pin')
        self.assertEqual(R.resolve(self.checkout, 'main')['upstream'], 'a'*40)
        legacy.write_text('b'*40 + '\n')
        self.git(self.repo, 'add', '-A')
        self.git(self.repo, 'commit', '-m', 'ambiguous pin')
        with self.assertRaisesRegex(ValueError, 'exactly one'):
            R.resolve(self.checkout, 'main')
        legacy.unlink()
        relocated.unlink()
        self.git(self.repo, 'add', '-A')
        self.git(self.repo, 'commit', '-m', 'missing pin')
        with self.assertRaisesRegex(ValueError, 'exactly one'):
            R.resolve(self.checkout, 'main')

    def test_only_manual_dispatch_can_select_source(self):
        with patch.dict(os.environ, GITHUB_EVENT_NAME='schedule', MESH_REF='candidate'):
            with self.assertRaisesRegex(ValueError, 'manual dispatch'):
                R.main()


if __name__ == '__main__':
    unittest.main()
