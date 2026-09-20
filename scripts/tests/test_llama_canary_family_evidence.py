from __future__ import annotations

import copy
import importlib.util
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import tarfile
import tempfile
from types import SimpleNamespace
import unittest
import yaml
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location('canary_evidence', ROOT / 'scripts/llama-canary-family-evidence.py')
E = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(E)


class FamilyEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.package = self.root / 'package'
        self.package.mkdir()
        self.evidence = self.root / 'evidence'
        self.evidence.mkdir()
        self.env = patch.dict(os.environ, GITHUB_RUN_ID='123', GITHUB_RUN_ATTEMPT='2')
        self.env.start()
        self.addCleanup(self.env.stop)
        self.plan = {
            'required_certification_lanes': sorted(E.CORE),
            'selected_models': [{'family': f, 'mmproj_artifact': None} for f in ('dense', 'hybrid')],
            'shards': [{'shard_index': i, 'families': [f]} for i, f in enumerate(('dense', 'hybrid'))],
            'github_matrix': {'include': [{'shard_index': i, 'families': f} for i, f in enumerate(('dense', 'hybrid'))]},
        }
        E.write(self.package / 'plan.json', self.plan)
        with tarfile.open(self.package / 'binaries.tar', 'w') as archive:
            for name in (*E.BINS, 'skippy-mm-test'):
                info = tarfile.TarInfo(name)
                content = b'#!/bin/sh\nexit 0\n'
                info.size = len(content)
                archive.addfile(info, io.BytesIO(content))
        self.identity = {'schema': 1, 'candidate': 'a'*40, 'base': 'a'*40,
                         'branch': 'llama-canary/repair-123-2-aaaaaaaaaa', 'pass_id': 'repair-1',
                         'platform': 'macos-arm64-metal', 'run_id': '123', 'run_attempt': '2',
                         'plan_sha256': E.sha(self.package / 'plan.json'),
                         'binaries_sha256': E.sha(self.package / 'binaries.tar'),
                         'bundle_sha256': None, 'manifest_sha256': 'b'*64}
        self.save_identity()
        for family in ('dense', 'hybrid'):
            directory = self.evidence / family
            directory.mkdir()
            row = {'family': family, 'exit_code': 0, 'split_layer': 10,
                   'outcomes': [{'name': lane, 'status': 'pass', 'exit_code': 0} for lane in E.CORE]}
            (directory / 'results.jsonl').write_text(json.dumps(row) + '\n')
            self.make_receipt(family)

    def save_identity(self):
        E.write(self.package / 'identity.json', self.identity)
        self.digest = E.sha(self.package / 'identity.json')

    def make_receipt(self, family, outcome='success'):
        E.receipt(SimpleNamespace(package=self.package, identity=self.digest, evidence=self.evidence / family,
                                  family=family, outcome=outcome))

    def aggregate(self):
        E.aggregate(SimpleNamespace(package=self.package, identity=self.digest, evidence=self.evidence))

    def test_complete_distributed_pass(self):
        self.aggregate()

    def test_reports_all_failed_workers_without_emitting_green(self):
        for family in ('dense', 'hybrid'):
            self.make_receipt(family, 'failure')
        summary = self.root / 'summary.md'
        output = self.root / 'outputs'
        with patch.dict(os.environ, GITHUB_STEP_SUMMARY=str(summary), GITHUB_OUTPUT=str(output)):
            with self.assertRaises(ValueError) as caught:
                self.aggregate()
        for family in ('dense', 'hybrid'):
            self.assertIn(f'{family}: failed or mismatched', str(caught.exception))
            self.assertIn(f'{family}: failed or mismatched', summary.read_text())
        self.assertIn('0/2 family receipts passed', summary.read_text())
        self.assertFalse(output.exists())

    def test_missing_worker_cannot_pass(self):
        (self.evidence / 'hybrid/receipt.json').unlink()
        with self.assertRaisesRegex(ValueError, 'missing family'):
            self.aggregate()

    def test_duplicate_worker_cannot_pass(self):
        duplicate = self.evidence / 'duplicate'
        shutil.copytree(self.evidence / 'dense', duplicate)
        receipts = sorted(self.evidence.glob('*/receipt.json'))
        # Filesystem traversal order differs across CI and developer machines.
        # Both workers must have complete evidence so only duplication fails.
        for order in (receipts, list(reversed(receipts))):
            with self.subTest(first=order[0].parent.name):
                with patch.object(Path, 'glob', return_value=iter(order)):
                    with self.assertRaisesRegex(ValueError, 'duplicate'):
                        self.aggregate()

    def test_failed_timed_out_or_cancelled_family_cannot_pass(self):
        for outcome in ('failure', 'cancelled', 'skipped'):
            with self.subTest(outcome=outcome):
                self.make_receipt('dense', outcome)
                with self.assertRaisesRegex(ValueError, 'failed or mismatched'):
                    self.aggregate()

    def test_other_candidate_or_pass_cannot_be_mixed(self):
        path = self.evidence / 'dense/receipt.json'
        original = E.read(path)
        for key, value in (('candidate', 'c'*40), ('pass_id', 'verify-1'), ('identity_sha256', 'd'*64)):
            with self.subTest(key=key):
                item = dict(original, **{key: value})
                E.write(path, item)
                with self.assertRaisesRegex(ValueError, 'mismatched'):
                    self.aggregate()

    def test_changed_results_are_detected(self):
        with (self.evidence / 'dense/results.jsonl').open('a') as stream:
            stream.write('{}\n')
        with self.assertRaisesRegex(ValueError, 'digest'):
            self.aggregate()

    def test_success_exit_without_core_lanes_is_rejected(self):
        path = self.evidence / 'dense/results.jsonl'
        row = json.loads(path.read_text())
        row['outcomes'].pop()
        path.write_text(json.dumps(row)+'\n')
        self.make_receipt('dense')
        with self.assertRaisesRegex(ValueError, 'required lane'):
            self.aggregate()

    def test_skipped_lane_is_not_certification(self):
        path = self.evidence / 'dense/results.jsonl'
        row = json.loads(path.read_text())
        row['outcomes'][0]['status'] = 'skip'
        path.write_text(json.dumps(row)+'\n')
        self.make_receipt('dense')
        with self.assertRaisesRegex(ValueError, 'required lane'):
            self.aggregate()

    def test_duplicate_certification_is_rejected(self):
        path = self.evidence / 'dense/results.jsonl'
        path.write_text(path.read_text()*2)
        self.make_receipt('dense')
        with self.assertRaisesRegex(ValueError, 'one consolidated'):
            self.aggregate()

    def test_missing_multimodal_smoke_is_rejected(self):
        model = {'mmproj_artifact': {'files': ['projector.gguf']}}
        with self.assertRaisesRegex(ValueError, 'multimodal'):
            E.validate_results(self.evidence / 'dense/results.jsonl', 'dense', model)

    def test_foreign_run_and_attempt_are_rejected(self):
        for key in ('run_id', 'run_attempt'):
            with self.subTest(key=key):
                original = self.identity[key]
                self.identity[key] = '999'
                self.save_identity()
                with self.assertRaisesRegex(ValueError, 'foreign workflow'):
                    E.verify_package(self.package, self.digest)
                self.identity[key] = original

    def test_tampered_package_is_rejected(self):
        (self.package / 'binaries.tar').write_bytes(b'wrong')
        with self.assertRaisesRegex(ValueError, 'digest mismatch'):
            E.verify_package(self.package, self.digest)

    def test_plan_requires_exactly_one_job_per_family(self):
        plan = copy.deepcopy(self.plan)
        plan['github_matrix']['include'][0]['families'] = 'dense,hybrid'
        with self.assertRaisesRegex(ValueError, 'one matrix job'):
            E.validate_plan(plan)

    def test_current_roster_generates_one_job_per_family(self):
        planner = E.load_planner(ROOT)
        plan = planner.build_plan(ROOT / 'ci/llama-canary/family-certified.json', shard_count=256)
        self.assertEqual(len(E.validate_plan(plan)), len(plan['github_matrix']['include']))

    def test_restore_pinned_source_moves_exact_executable_bytes(self):
        checkout = self.root / 'checkout'
        checkout.mkdir()
        subprocess.run(['git', 'init', '-q', str(checkout)], check=True)
        manifest = checkout / 'ci/llama-canary/family-certified.json'
        manifest.parent.mkdir(parents=True)
        manifest.write_text('{}\n')
        subprocess.run(['git', '-C', str(checkout), 'add', '.'], check=True)
        subprocess.run(['git', '-C', str(checkout), '-c', 'user.name=Fixture', '-c', 'user.email=fixture@example.invalid',
                        '-c', 'commit.gpgsign=false', 'commit', '-qm', 'fixture'], check=True)
        self.identity['base'] = self.identity['candidate'] = E.git(checkout, 'rev-parse', 'HEAD')
        self.identity['manifest_sha256'] = E.sha(manifest)
        self.save_identity()
        E.restore(SimpleNamespace(package=self.package, identity=self.digest, root=checkout))
        for name in (*E.BINS, 'skippy-mm-test'):
            binary = checkout / 'target/debug' / name
            self.assertTrue(os.access(binary, os.X_OK))
            self.assertEqual(binary.read_bytes(), b'#!/bin/sh\nexit 0\n')

    def changed_candidate(self, protected=False):
        checkout = self.root / 'changed-checkout'
        checkout.mkdir()
        def command(*args):
            return subprocess.run(['git', '-C', str(checkout), '-c', 'user.name=Fixture',
                                   '-c', 'user.email=fixture@example.invalid', '-c', 'commit.gpgsign=false',
                                   *args], check=True, capture_output=True, text=True)
        command('init', '-q')
        manifest = checkout / 'ci/llama-canary/family-certified.json'
        manifest.parent.mkdir(parents=True)
        manifest.write_text('{}\n')
        command('add', '.')
        command('commit', '-qm', 'base')
        base = command('rev-parse', 'HEAD').stdout.strip()
        target = checkout / ('scripts/control.py' if protected else 'candidate.txt')
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text('candidate bytes\n')
        command('add', '.')
        command('commit', '-qm', 'candidate')
        candidate = command('rev-parse', 'HEAD').stdout.strip()
        branch = self.identity['branch']
        command('branch', branch)
        command('bundle', 'create', str(self.package / 'candidate.bundle'), branch, '^'+base)
        command('checkout', '--detach', base)
        self.identity.update(base=base, candidate=candidate,
                             bundle_sha256=E.sha(self.package / 'candidate.bundle'),
                             manifest_sha256=E.sha(manifest))
        self.save_identity()
        return checkout

    def test_restore_changed_candidate_is_exact_and_detached(self):
        checkout = self.changed_candidate()
        E.restore(SimpleNamespace(package=self.package, identity=self.digest, root=checkout))
        self.assertEqual(E.git(checkout, 'rev-parse', 'HEAD'), self.identity['candidate'])
        self.assertEqual((checkout / 'candidate.txt').read_text(), 'candidate bytes\n')
        self.assertEqual(E.git(checkout, 'status', '--porcelain', '--untracked-files=no'), '')

    def test_candidate_cannot_replace_trusted_worker_code(self):
        checkout = self.changed_candidate(protected=True)
        with self.assertRaisesRegex(ValueError, 'trusted orchestration'):
            E.restore(SimpleNamespace(package=self.package, identity=self.digest, root=checkout))
        self.assertEqual(E.git(checkout, 'rev-parse', 'HEAD'), self.identity['base'])

    def test_publisher_rejects_repair_only_package(self):
        with self.assertRaisesRegex(ValueError, 'independent verifier'):
            E.publication(SimpleNamespace(package=self.package, identity=self.digest))


class WorkflowTerminalGateTests(unittest.TestCase):
    def run_gate(self, *, changed=True, certify=True, repair=None, verify=None):
        workflow = yaml.safe_load((ROOT / '.github/workflows/llama-upstream-canary.yml').read_text())
        step = workflow['jobs']['result']['steps'][0]
        body = step['run'].split("python3 - <<'PYCODE'\n", 1)[1].rsplit('PYCODE', 1)[0]
        needs = {'resolve': {'result': 'success', 'outputs': {
            'changed': str(changed).lower(), 'certify': str(certify).lower()}}}
        for attempt in range(1, 4):
            for mode, values in (('repair', repair or {}), ('verify', verify or {})):
                outputs = values.get(attempt, {})
                needs[f'{mode}-{attempt}'] = {'result': 'success' if outputs else 'skipped', 'outputs': outputs}
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'output'
            result = subprocess.run(['python3', '-c', body], env={**os.environ,
                                    'NEEDS_JSON': json.dumps(needs), 'GITHUB_OUTPUT': str(output)},
                                    text=True, capture_output=True)
            return result.returncode, output.read_text() if output.exists() else ''

    def green(self, head='a'*40):
        return {'green': 'true', 'head': head, 'package': 'candidate-package', 'identity': 'b'*64, 'branch': 'branch'}

    def test_success_requires_both_passes_on_same_commit(self):
        code, out = self.run_gate(repair={1: self.green()}, verify={1: self.green()})
        self.assertEqual(code, 0)
        self.assertIn('publish=true', out)
        code, out = self.run_gate(repair={1: self.green()}, verify={1: self.green('c'*40)})
        self.assertNotEqual(code, 0)
        self.assertNotIn('publish=true', out)

    def test_success_after_repair_uses_later_candidate(self):
        code, out = self.run_gate(repair={1: {'head': 'a'*40}, 2: self.green('c'*40)},
                                  verify={2: self.green('c'*40)})
        self.assertEqual(code, 0)
        self.assertIn('head='+'c'*40, out)

    def test_missing_verification_and_exhaustion_deny_publication(self):
        code, out = self.run_gate(repair={i: self.green() for i in range(1, 4)})
        self.assertNotEqual(code, 0)
        self.assertEqual(out, '')

    def test_unchanged_pin_requires_families_but_never_publishes(self):
        code, out = self.run_gate(changed=False, repair={1: self.green()})
        self.assertEqual(code, 0)
        self.assertEqual(out, '')
        code, _ = self.run_gate(changed=False)
        self.assertNotEqual(code, 0)

    def test_non_forced_unchanged_manual_is_read_only_noop(self):
        code, out = self.run_gate(changed=False, certify=False)
        self.assertEqual(code, 0)
        self.assertEqual(out, '')


if __name__ == '__main__':
    unittest.main()
