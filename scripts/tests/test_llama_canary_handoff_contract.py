"""Exercise source-owned consumers across controller/source version boundaries."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from types import SimpleNamespace
import time
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location('handoff', ROOT / 'scripts/llama-canary-family-evidence.py')
E = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(E)


class HandoffContractTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.source = self.base / 'canary-source'
        self.source.mkdir()

    def command(self, root, *args):
        return subprocess.check_output(['git', '-C', str(root), *args], text=True,
                                       stderr=subprocess.DEVNULL).strip()

    def init(self, root):
        root.mkdir(parents=True, exist_ok=True)
        self.command(root, 'init')
        self.command(root, 'config', 'user.name', 'Fixture')
        self.command(root, 'config', 'user.email', 'fixture@example.invalid')
        self.command(root, 'config', 'commit.gpgsign', 'false')

    def commit(self, root):
        self.command(root, 'add', '-A')
        self.command(root, 'commit', '-m', 'fixture')
        return self.command(root, 'rev-parse', 'HEAD')

    def copy_script(self, name):
        (self.source / 'scripts').mkdir(exist_ok=True)
        shutil.copy2(ROOT / 'scripts' / name, self.source / 'scripts' / name)

    def test_older_planner_and_nested_checkout_validate_with_real_battery(self):
        # Model the previous planner's descending shard order. Both versions
        # insist on exact canonical plans; the controller may only sort its
        # separate scheduling projection.
        for name in ('plan-family-battery.py', 'skippy-family-battery.sh'):
            self.copy_script(name)
        shutil.copytree(ROOT / 'scripts/lib', self.source / 'scripts/lib')
        manifest = self.source / 'ci/llama-canary/family-certified.json'
        manifest.parent.mkdir(parents=True)
        shutil.copy2(ROOT / 'ci/llama-canary/family-certified.json', manifest)
        planner = self.source / 'scripts/plan-family-battery.py'
        text = planner.read_text()
        needle = 'for shard in sorted(shards, key=lambda item: (item["estimated_work_bytes"], item["families"]))'
        self.assertIn(needle, text)
        planner.write_text(text.replace(needle, 'for shard in shards'))
        destination = self.base / 'plan.json'
        plan = E.source_plan(self.source, destination)
        self.assertEqual(plan['manifest'], 'ci/llama-canary/family-certified.json')
        before = destination.read_bytes()
        matrix = E.scheduling_matrix(plan)
        weights = [row['estimated_work_bytes'] for row in matrix['include']]
        self.assertEqual(weights, sorted(weights))
        self.assertNotEqual(matrix, plan['github_matrix'])
        self.assertEqual(destination.read_bytes(), before)
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop('HF_CACHE', None)
            E.preflight_battery(self.source)

    def test_prepared_source_roundtrip_in_empty_worker_and_stale_pin_rejected(self):
        self.copy_script('llama-oracle-source.py')
        pin = self.source / 'third_party/llama.cpp/upstream.txt'
        pin.parent.mkdir(parents=True)
        (pin.parent / 'patches').mkdir()
        pin.write_text('a' * 40 + '\n')
        native = self.source / '.deps/llama.cpp'
        self.init(native)
        (native / 'source.cpp').write_text('first')
        self.commit(native)
        (native / 'source.cpp').write_text('second')
        head = self.commit(native)
        values = ('a' * 40, hashlib.sha256(b'').hexdigest(), head, '5')
        for name, value in zip(E.LLAMA_MARKERS, values):
            (native / name).write_text(value + '\n')
        package = self.base / 'package'
        package.mkdir()
        E.pack_llama_source(self.source, package)
        shutil.rmtree(native)
        E.restore_llama_source(self.source, package)
        self.assertEqual((native / 'source.cpp').read_text(), 'second')
        self.assertEqual(self.command(native, 'rev-parse', 'HEAD'), head)
        pin.write_text('b' * 40 + '\n')
        with self.assertRaises(subprocess.CalledProcessError):
            E.restore_llama_source(self.source, package)

    def test_pack_restore_and_source_owned_workload_verification(self):
        # A fresh worker has no .deps or target outputs. Only binary platform
        # inspection is mocked: fixture scripts stand in for compiled Mach-O.
        for name in ('llama-oracle-source.py', 'check-skippy-workload-candidate.py',
                     'plan-family-battery.py', 'skippy-family-battery.sh'):
            self.copy_script(name)
        shutil.copytree(ROOT / 'scripts/lib', self.source / 'scripts/lib')
        manifest = self.source / 'ci/llama-canary/family-certified.json'
        manifest.parent.mkdir(parents=True)
        shutil.copy2(ROOT / 'ci/llama-canary/family-certified.json', manifest)
        pin = self.source / 'third_party/llama.cpp/upstream.txt'
        pin.parent.mkdir(parents=True)
        (pin.parent / 'patches').mkdir()
        pin.write_text('a'*40 + '\n')
        (self.source / '.gitignore').write_text('.deps/\ntarget/\n')
        self.init(self.source)
        candidate = self.commit(self.source)
        native = self.source / '.deps/llama.cpp'
        self.init(native)
        (native / 'source.cpp').write_text('fixture')
        head = self.commit(native)
        for name, value in zip(E.LLAMA_MARKERS, ('a'*40, hashlib.sha256(b'').hexdigest(), head, '5')):
            (native / name).write_text(value + '\n')
        closure = self.source / '.deps/workloads'
        names = {'candidate': 'cargo/debug/skippy-server', 'test_binary': 'cargo/debug/deps/test',
                 'model_package': 'cargo/debug/skippy-model-package',
                 'correctness': 'cargo/debug/skippy-correctness',
                 'topology_plan': 'cargo/debug/skippy-topology-plan',
                 'native_stamp': 'native/.mesh-llm-build-stamp',
                 **{name: 'native/bin/'+name for name in ('llama-server', 'llama-completion', 'llama-tts')}}
        for relative in names.values():
            file = closure / relative
            file.parent.mkdir(parents=True, exist_ok=True)
            file.write_bytes(b'#!/bin/sh\nexit 0\n')
            file.chmod(0o755)
        os.utime(closure / names['native_stamp'], (time.time()-120, time.time()-120))
        E.write(closure / 'producer.json', {'schema_version': 1,
                'source': {'head': candidate, 'worktree_sha256': hashlib.sha256(b'').hexdigest()},
                'files': {name: {'path': relative, 'sha256': E.sha(closure / relative)}
                          for name, relative in names.items()}})
        for name in (*E.BINS, 'skippy-mm-test'):
            file = self.source / 'target/debug' / name
            file.parent.mkdir(parents=True, exist_ok=True)
            file.write_bytes(b'#!/bin/sh\nexit 0\n')
        test_build = self.base / 'build.jsonl'
        E.write(test_build, {})
        test_build.write_text(json.dumps({'reason': 'compiler-artifact',
            'executable': str(self.source / 'target/debug/skippy-mm-test'),
            'target': {'name': 'skippy_server'}, 'profile': {'test': True}})+'\n')
        summary = self.base / 'summary.md'
        summary.write_text('fixture')
        package = self.base / 'package'
        env = dict(GITHUB_RUN_ID='123', GITHUB_RUN_ATTEMPT='1',
                   CANARY_CONTROLLER_SHA=candidate, CANARY_MESH_SOURCE=candidate)
        actual_plan = E.source_plan
        def without_cache(root, destination, **kwargs):
            return actual_plan(root, destination)
        with patch.dict(os.environ, env), patch.object(E, 'check_binary'), \
                patch.object(E, 'source_plan', side_effect=without_cache):
            os.environ.pop('CANARY_VERIFIED_WORKLOAD_PRODUCER', None)
            E.pack(SimpleNamespace(root=self.source, output=package, candidate=candidate,
                base=candidate, branch='fixture', pass_id='repair-1', test_build=test_build,
                workload_oracles=closure, summary=summary, bundle=self.base/'absent.bundle'))
            worker = self.base / 'worker'
            subprocess.run(['git', 'clone', str(self.source), str(worker)], check=True,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            self.assertFalse((worker / '.deps').exists())
            E.restore(SimpleNamespace(root=worker, package=package, identity=E.sha(package/'identity.json')))
            restored = worker / E.WORKLOAD_CLOSURE_ROOT
            subprocess.run(['python3', str(worker/'scripts/check-skippy-workload-candidate.py'),
                '--candidate-binary', str(restored/names['candidate']),
                '--native-build-dir', str(restored/'native'),
                '--producer-manifest', str(restored/'producer.json')], check=True)
            E.preflight_battery(worker)

    def test_snapshot_rebinding_requires_sealed_manifest_exact_tree_and_files(self):
        self.init(self.source)
        (self.source / 'code').write_text('base')
        self.commit(self.source)
        (self.source / 'code').write_text('candidate')
        candidate = self.commit(self.source)
        closure = self.base / 'closure'
        closure.mkdir()
        (closure / 'binary').write_bytes(b'compiled')
        payload = {'source': {'head': 'a'*40, 'worktree_sha256': 'b'*64},
                   'files': {'candidate': {'path': 'binary', 'sha256': E.sha(closure / 'binary')}}}
        E.write(closure / 'producer.json', payload)
        seal = E.sha(closure / 'producer.json')
        with patch.dict(os.environ, CANARY_VERIFIED_WORKLOAD_PRODUCER=seal):
            result = json.loads(E.packaged_workload_manifest(self.source, closure, candidate))
            self.assertEqual(result['source'], {'head': candidate, 'worktree_sha256': hashlib.sha256(b'').hexdigest()})
            self.assertEqual(E.read(closure / 'producer.json'), payload)
            (closure / 'binary').write_bytes(b'replaced')
            with self.assertRaisesRegex(ValueError, 'artifact changed'):
                E.packaged_workload_manifest(self.source, closure, candidate)
            (closure / 'binary').write_bytes(b'compiled')
            (self.source / 'untracked').write_text('source')
            with self.assertRaisesRegex(ValueError, 'untracked source'):
                E.packaged_workload_manifest(self.source, closure, candidate)
            (self.source / 'untracked').unlink()
            E.write(closure / 'producer.json', {})
            with self.assertRaisesRegex(ValueError, 'changed after snapshot'):
                E.packaged_workload_manifest(self.source, closure, candidate)


if __name__ == '__main__':
    unittest.main()
