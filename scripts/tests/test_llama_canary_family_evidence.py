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
            'selected_models': [{
                'family': f,
                'artifact': {'files': ['model.gguf'],
                             'file_integrity': {'model.gguf': {'size_bytes': 1}}},
                'resources': {'estimated_model_bytes': 1},
                'mmproj_artifact': None,
                'class': 'causal_generation',
                'certification_lanes': sorted(E.CORE),
            } for f in ('dense', 'hybrid')],
            'shards': [{'shard_index': i, 'families': [f]} for i, f in enumerate(('dense', 'hybrid'))],
            'github_matrix': {'include': [
                {'shard_index': i, 'families': f, 'estimated_work_bytes': i + 1}
                for i, f in enumerate(('dense', 'hybrid'))
            ]},
        }
        E.write(self.package / 'plan.json', self.plan)
        with tarfile.open(self.package / 'binaries.tar', 'w') as archive:
            for name in (*E.BINS, 'skippy-mm-test'):
                info = tarfile.TarInfo(name)
                content = b'#!/bin/sh\nexit 0\n'
                info.size = len(content)
                archive.addfile(info, io.BytesIO(content))
        self.build_closure()
        for name in (E.LLAMA_BUNDLE, E.LLAMA_PROVENANCE):
            (self.package / name).write_text('fixture')
        restore_source = patch.object(E, 'restore_llama_source')
        restore_source.start()
        self.addCleanup(restore_source.stop)
        self.identity = {'schema': 3, 'candidate': 'a'*40, 'base': 'a'*40,
                         'branch': 'llama-canary/repair-123-2-aaaaaaaaaa', 'pass_id': 'repair-1',
                         'platform': 'macos-arm64-metal', 'run_id': '123', 'run_attempt': '2',
                         'plan_sha256': E.sha(self.package / 'plan.json'),
                         'binaries_sha256': E.sha(self.package / 'binaries.tar'),
                         'workload_oracles_sha256': E.sha(self.package / E.WORKLOAD_ORACLES_TAR),
                         'llama_bundle_sha256': E.sha(self.package / E.LLAMA_BUNDLE),
                         'llama_provenance_sha256': E.sha(self.package / E.LLAMA_PROVENANCE),
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

    def test_mtp_family_cannot_certify_without_its_all_head_lane(self):
        model = copy.deepcopy(self.plan['selected_models'][0])
        model['certification_lanes'].append('native-mtp-heads')
        path = self.evidence / 'dense/results.jsonl'
        with self.assertRaisesRegex(ValueError, 'native-mtp-heads incomplete'):
            E.validate_results(path, 'dense', model)
        row = json.loads(path.read_text())
        row['outcomes'].append({'name': 'native-mtp-heads', 'status': 'pass', 'exit_code': 0})
        path.write_text(json.dumps(row) + '\n')
        E.validate_results(path, 'dense', model)
        row['outcomes'][-1]['status'] = 'fail'
        path.write_text(json.dumps(row) + '\n')
        with self.assertRaisesRegex(ValueError, 'native-mtp-heads incomplete'):
            E.validate_results(path, 'dense', model)

    def build_closure(self):
        """Create a synthetic workload oracle closure and its handoff tar."""
        directory = self.root / 'closure-src'
        directory.mkdir(exist_ok=True)
        files = {
            'candidate': 'cargo/debug/skippy',
            'test_binary': 'cargo/debug/deps/smoke-test',
            'model_package': 'cargo/debug/skippy-package-builder',
            'correctness': 'cargo/debug/skippy-correctness',
            'topology_plan': 'cargo/debug/skippy-topology-plan',
            'native_stamp': 'native/.mesh-llm-build-stamp',
            'oracle_server': 'native/bin/llama-server',
            'oracle_completion': 'native/bin/llama-completion',
            'oracle_tts': 'native/bin/llama-tts',
        }
        manifest = {'schema_version': 1, 'source': {'head': 'x'*40, 'worktree_sha256': 'y'*64},
                    'files': {name: {'path': relative, 'sha256': 'z'*64} for name, relative in files.items()}}
        (directory / 'producer.json').write_text(json.dumps(manifest, sort_keys=True) + '\n')
        for relative in files.values():
            target = directory / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(b'closure:' + relative.encode())
        with tarfile.open(self.package / E.WORKLOAD_ORACLES_TAR, 'w') as archive:
            for relative in sorted({'producer.json', *files.values()}):
                archive.add(directory / relative, arcname=relative)

    def make_receipt(self, family, outcome='success'):
        self.make_receipt_at(self.evidence / family, family, outcome)

    def make_receipt_at(self, directory, family, outcome='success'):
        E.receipt(SimpleNamespace(package=self.package, identity=self.digest, evidence=directory,
                                  family=family, outcome=outcome))

    def aggregate(self, **kwargs):
        E.aggregate(SimpleNamespace(package=self.package, identity=self.digest,
                                    evidence=self.evidence, **kwargs))

    def reconcile(self, previous_feedback, evidence, **kwargs):
        E.reconcile(SimpleNamespace(package=self.package, identity=self.digest,
                                    previous_feedback=previous_feedback, evidence=evidence, **kwargs))

    def test_complete_distributed_pass(self):
        self.aggregate()

    def test_pretty_printed_worker_evidence_is_accepted(self):
        path = self.evidence / 'dense/results.jsonl'
        certification = json.loads(path.read_text())
        preflight = {'family': 'dense', 'exit_code': 0,
                     'outcomes': [{'name': 'model-preflight', 'status': 'pass', 'exit_code': 0}]}
        path.write_text(json.dumps(preflight, indent=2) + '\n' + json.dumps(certification) + '\n')
        E.validate_results(path, 'dense', self.plan['selected_models'][0])

    def test_global_battery_preflight_is_validated_separately(self):
        path = self.evidence / 'dense/results.jsonl'
        certification = path.read_text()
        battery = {'family': 'battery', 'exit_code': 0, 'model_id': 'environment',
                   'outcomes': [{'name': 'environment-preflight', 'status': 'pass',
                                 'outcome': 'pass', 'exit_code': 0}]}
        path.write_text(json.dumps(battery) + '\n' + certification)
        E.validate_results(path, 'dense', self.plan['selected_models'][0])
        battery['outcomes'][0]['status'] = 'fail'
        path.write_text(json.dumps(battery) + '\n' + certification)
        with self.assertRaisesRegex(ValueError, 'global battery preflight incomplete'):
            E.validate_results(path, 'dense', self.plan['selected_models'][0])
        battery['family'] = 'foreign'
        battery['outcomes'][0]['status'] = 'pass'
        path.write_text(json.dumps(battery) + '\n' + certification)
        with self.assertRaisesRegex(ValueError, 'foreign'):
            E.validate_results(path, 'dense', self.plan['selected_models'][0])

    def rerun_receipt(self, family='dense', outcome='success', attempt='3'):
        previous = self.evidence / f'{family}-previous'
        shutil.copytree(self.evidence / family, previous)
        with patch.dict(os.environ, GITHUB_RUN_ATTEMPT=attempt):
            self.make_receipt(family, outcome)

    def test_partial_rerun_reuses_build_and_successful_sibling(self):
        self.make_receipt('dense', 'failure')
        self.rerun_receipt()
        with patch.dict(os.environ, GITHUB_RUN_ATTEMPT='3'):
            self.aggregate()

    def test_aggregate_only_rerun_reuses_complete_prior_pass(self):
        with patch.dict(os.environ, GITHUB_RUN_ATTEMPT='4'):
            self.aggregate()

    def test_newer_failure_never_falls_back_to_old_success(self):
        for outcome in ('failure', 'cancelled', 'skipped'):
            with self.subTest(outcome=outcome):
                if (self.evidence / 'dense-previous').exists():
                    shutil.rmtree(self.evidence / 'dense-previous')
                self.make_receipt('dense')
                self.rerun_receipt(outcome=outcome)
                with patch.dict(os.environ, GITHUB_RUN_ATTEMPT='3'):
                    with self.assertRaisesRegex(ValueError, 'failed or mismatched'):
                        self.aggregate()

    def test_newer_corrupt_results_never_fall_back(self):
        self.rerun_receipt()
        (self.evidence / 'dense/results.jsonl').write_text('{}\n')
        with patch.dict(os.environ, GITHUB_RUN_ATTEMPT='3'):
            with self.assertRaisesRegex(ValueError, 'digest mismatch'):
                self.aggregate()

    def test_worker_provenance_must_be_valid(self):
        path = self.evidence / 'dense/receipt.json'
        original = E.read(path)
        for key, value in (('run_id', '999'), ('run_attempt', '1'),
                           ('run_attempt', '3'), ('run_attempt', '0'),
                           ('run_attempt', 'x'), ('run_attempt', 2)):
            with self.subTest(key=key, value=value):
                E.write(path, dict(original, **{key: value}))
                with self.assertRaises(ValueError):
                    self.aggregate()

    def test_rebuilt_producer_cannot_reuse_previous_receipts(self):
        self.identity['run_attempt'] = '3'
        self.save_identity()
        with patch.dict(os.environ, GITHUB_RUN_ATTEMPT='3'):
            with self.assertRaisesRegex(ValueError, 'mismatched'):
                self.aggregate()

    def test_invalid_producer_attempt_is_rejected(self):
        for value in ('0', '-1', 'x', '02', 2, None):
            with self.subTest(value=value):
                self.identity['run_attempt'] = value
                self.save_identity()
                with self.assertRaisesRegex(ValueError, 'invalid workflow'):
                    E.verify_package(self.package, self.digest)

    def test_rerun_still_rejects_tampered_package(self):
        (self.package / 'binaries.tar').write_bytes(b'wrong')
        with patch.dict(os.environ, GITHUB_RUN_ATTEMPT='3'):
            with self.assertRaisesRegex(ValueError, 'digest mismatch'):
                E.verify_package(self.package, self.digest)

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
        self.assertIn('green=false', output.read_text())
        self.assertIn('failure_class=candidate', output.read_text())

    def test_candidate_failure_emits_digest_bound_feedback(self):
        self.make_receipt('dense', 'failure')
        feedback = self.root / 'feedback'
        output = self.root / 'outputs'
        with patch.dict(os.environ, GITHUB_OUTPUT=str(output)):
            with self.assertRaisesRegex(ValueError, 'failed or mismatched'):
                self.aggregate(feedback=feedback, family_result='failure')
        payload = E.verify_feedback(feedback, self.digest, self.identity)
        self.assertEqual(payload['state'], 'candidate_repairable')
        self.assertEqual(payload['candidate_failures'], ['dense'])
        self.assertEqual(payload['infrastructure_failures'], [])
        self.assertEqual(payload['failed_families'], ['dense'])
        self.assertTrue((feedback / 'dense/results.jsonl').is_file())
        self.assertIn('repairable=true', output.read_text())
        with (feedback / 'dense/results.jsonl').open('a') as stream:
            stream.write('{}\n')
        with self.assertRaisesRegex(ValueError, 'digest mismatch'):
            E.verify_feedback(feedback, self.digest, self.identity)

    def test_mixed_failure_retains_candidate_and_targets_only_missing_family(self):
        self.make_receipt('dense', 'failure')
        (self.evidence / 'hybrid/receipt.json').unlink()
        feedback = self.root / 'feedback'
        output = self.root / 'outputs'
        with patch.dict(os.environ, GITHUB_OUTPUT=str(output)):
            with self.assertRaises(ValueError):
                self.aggregate(feedback=feedback, family_result='failure')
        self.assertIn('repairable=false', output.read_text())
        self.assertIn('failure_class=infrastructure', output.read_text())
        self.assertIn('state=infrastructure_retryable', output.read_text())
        matrix_line = next(line for line in output.read_text().splitlines()
                           if line.startswith('retry_matrix='))
        matrix = json.loads(matrix_line.removeprefix('retry_matrix='))
        self.assertEqual([row['families'] for row in matrix['include']], ['hybrid'])
        payload = E.verify_feedback(feedback, self.digest, self.identity,
                                    expected_state='infrastructure_retryable')
        self.assertEqual(payload['candidate_failures'], ['dense'])
        self.assertEqual(payload['infrastructure_failures'], ['hybrid'])

    def test_cancelled_or_skipped_graph_blocks_candidate_repair(self):
        for family_result in ('cancelled', 'skipped'):
            with self.subTest(family_result=family_result):
                self.make_receipt('dense', 'failure')
                feedback = self.root / f'feedback-{family_result}'
                output = self.root / f'outputs-{family_result}'
                with patch.dict(os.environ, GITHUB_OUTPUT=str(output)):
                    with self.assertRaises(ValueError):
                        self.aggregate(feedback=feedback, family_result=family_result)
                self.assertIn('repairable=false', output.read_text())
                self.assertIn('failure_class=contract', output.read_text())
                self.assertFalse(feedback.exists())

    def test_environment_preflight_failure_is_retried_without_source_repair(self):
        path = self.evidence / 'dense/results.jsonl'
        battery = {'family': 'battery', 'exit_code': 1,
                   'outcomes': [{'name': 'environment-preflight', 'status': 'fail',
                                 'exit_code': 1}]}
        path.write_text(json.dumps(battery) + '\n')
        self.make_receipt('dense', 'failure')
        output = self.root / 'outputs'
        feedback = self.root / 'feedback'
        with patch.dict(os.environ, GITHUB_OUTPUT=str(output)):
            with self.assertRaises(ValueError):
                self.aggregate(feedback=feedback, family_result='failure')
        self.assertIn('repairable=false', output.read_text())
        self.assertIn('failure_class=infrastructure', output.read_text())
        self.assertIn('state=infrastructure_retryable', output.read_text())
        payload = E.verify_feedback(feedback, self.digest, self.identity,
                                    expected_state='infrastructure_retryable')
        self.assertEqual(payload['candidate_failures'], [])
        self.assertEqual(payload['infrastructure_failures'], ['dense'])

    def test_memory_admission_without_results_is_retried(self):
        directory = self.evidence / 'dense'
        (directory / 'results.jsonl').unlink()
        E.write(directory / 'memory-admission.json', {'status': 'failed'})
        self.make_receipt('dense', 'failure')
        output = self.root / 'outputs'
        feedback = self.root / 'feedback'
        with patch.dict(os.environ, GITHUB_OUTPUT=str(output)):
            with self.assertRaises(ValueError):
                self.aggregate(feedback=feedback, family_result='failure')
        self.assertIn('state=infrastructure_retryable', output.read_text())
        payload = E.verify_feedback(feedback, self.digest, self.identity,
                                    expected_state='infrastructure_retryable')
        self.assertEqual(payload['infrastructure_failures'], ['dense'])

    def test_failed_certification_child_is_candidate_failure(self):
        directory = self.evidence / 'dense'
        E.write(directory / 'memory-admission.json', {'status': 'failed', 'exit_code': 1})
        self.make_receipt('dense', 'failure')
        self.assertEqual(E.classify_family_failure(directory / 'receipt.json', 'dense',
                                                   self.identity, self.digest), 'candidate')

    def test_failed_child_without_results_is_retried(self):
        directory = self.evidence / 'dense'
        (directory / 'results.jsonl').unlink()
        E.write(directory / 'memory-admission.json', {'status': 'failed', 'exit_code': 1})
        self.make_receipt('dense', 'failure')
        self.assertEqual(E.classify_family_failure(directory / 'receipt.json', 'dense',
                                                   self.identity, self.digest), 'infrastructure')

    def test_memory_monitor_error_with_results_is_infrastructure_failure(self):
        directory = self.evidence / 'dense'
        E.write(directory / 'memory-admission.json',
                {'status': 'failed', 'error': 'host available memory crossed reserve'})
        self.make_receipt('dense', 'failure')
        self.assertEqual(E.classify_family_failure(directory / 'receipt.json', 'dense',
                                                   self.identity, self.digest), 'infrastructure')

    def test_memory_admission_with_corrupt_results_remains_contract_failure(self):
        directory = self.evidence / 'dense'
        E.write(directory / 'memory-admission.json', {'status': 'failed'})
        (directory / 'results.jsonl').write_text('changed after receipt\n')
        self.assertEqual(E.classify_family_failure(directory / 'receipt.json', 'dense',
                                                   self.identity, self.digest), 'contract')

    def test_failed_memory_admission_cannot_have_success_receipt(self):
        directory = self.evidence / 'dense'
        (directory / 'results.jsonl').unlink()
        E.write(directory / 'memory-admission.json', {'status': 'failed'})
        self.make_receipt('dense', 'success')
        self.assertEqual(E.classify_family_failure(directory / 'receipt.json', 'dense',
                                                   self.identity, self.digest), 'contract')

    def test_infrastructure_recheck_preserves_prior_candidate_failure(self):
        self.make_receipt('dense', 'failure')
        (self.evidence / 'hybrid/receipt.json').unlink()
        previous = self.root / 'previous-feedback'
        with self.assertRaises(ValueError):
            self.aggregate(feedback=previous, family_result='failure')
        retry = self.root / 'retry'
        shutil.copytree(self.evidence / 'hybrid', retry / 'hybrid')
        self.make_receipt_at(retry / 'hybrid', 'hybrid')
        reconciled = self.root / 'reconciled-feedback'
        output = self.root / 'reconcile-output'
        with patch.dict(os.environ, GITHUB_OUTPUT=str(output)):
            with self.assertRaisesRegex(ValueError, 'candidate failures remain'):
                self.reconcile(previous, retry, feedback=reconciled, family_result='success')
        self.assertIn('state=candidate_repairable', output.read_text())
        payload = E.verify_feedback(reconciled, self.digest, self.identity,
                                    expected_state='candidate_repairable')
        self.assertEqual(payload['candidate_failures'], ['dense'])
        self.assertEqual(payload['infrastructure_failures'], [])

    def test_infrastructure_only_recheck_can_complete_the_full_pass(self):
        (self.evidence / 'hybrid/receipt.json').unlink()
        previous = self.root / 'previous-feedback'
        with self.assertRaises(ValueError):
            self.aggregate(feedback=previous, family_result='failure')
        retry = self.root / 'retry'
        shutil.copytree(self.evidence / 'hybrid', retry / 'hybrid')
        self.make_receipt_at(retry / 'hybrid', 'hybrid')
        output = self.root / 'reconcile-output'
        with patch.dict(os.environ, GITHUB_OUTPUT=str(output)):
            self.reconcile(previous, retry, feedback=self.root / 'unused', family_result='success')
        self.assertIn('green=true', output.read_text())
        self.assertIn('state=green', output.read_text())

    def test_failed_retry_job_graph_cannot_certify_green_receipts(self):
        (self.evidence / 'hybrid/receipt.json').unlink()
        previous = self.root / 'previous-feedback'
        with self.assertRaises(ValueError):
            self.aggregate(feedback=previous, family_result='failure')
        retry = self.root / 'retry'
        shutil.copytree(self.evidence / 'hybrid', retry / 'hybrid')
        self.make_receipt_at(retry / 'hybrid', 'hybrid')
        output = self.root / 'reconcile-output'
        with patch.dict(os.environ, GITHUB_OUTPUT=str(output)):
            with self.assertRaisesRegex(ValueError, 'did not produce complete'):
                self.reconcile(previous, retry, feedback=self.root / 'unused', family_result='failure')
        self.assertIn('state=infrastructure_exhausted', output.read_text())
        self.assertIn('green=false', output.read_text())
        self.assertFalse((self.root / 'unused').exists())

    def test_recheck_can_reclassify_an_infrastructure_family_as_candidate(self):
        (self.evidence / 'hybrid/receipt.json').unlink()
        previous = self.root / 'previous-feedback'
        with self.assertRaises(ValueError):
            self.aggregate(feedback=previous, family_result='failure')
        retry = self.root / 'retry'
        shutil.copytree(self.evidence / 'hybrid', retry / 'hybrid')
        self.make_receipt_at(retry / 'hybrid', 'hybrid', 'failure')
        reconciled = self.root / 'reconciled-feedback'
        with self.assertRaisesRegex(ValueError, 'candidate failures remain'):
            self.reconcile(previous, retry, feedback=reconciled, family_result='failure')
        payload = E.verify_feedback(reconciled, self.digest, self.identity,
                                    expected_state='candidate_repairable')
        self.assertEqual(payload['candidate_failures'], ['hybrid'])

    def test_repeated_infrastructure_failure_stops_without_goose_feedback(self):
        (self.evidence / 'hybrid/receipt.json').unlink()
        previous = self.root / 'previous-feedback'
        with self.assertRaises(ValueError):
            self.aggregate(feedback=previous, family_result='failure')
        retry = self.root / 'retry'
        shutil.copytree(self.evidence / 'hybrid', retry / 'hybrid')
        E.write(retry / 'hybrid/memory-admission.json',
                {'status': 'failed', 'error': 'host available memory crossed reserve'})
        self.make_receipt_at(retry / 'hybrid', 'hybrid', 'failure')
        output = self.root / 'reconcile-output'
        with patch.dict(os.environ, GITHUB_OUTPUT=str(output)):
            with self.assertRaisesRegex(ValueError, 'did not produce complete'):
                self.reconcile(previous, retry, feedback=self.root / 'unused', family_result='failure')
        self.assertIn('state=infrastructure_exhausted', output.read_text())
        self.assertIn('repairable=false', output.read_text())
        self.assertFalse((self.root / 'unused').exists())

    def test_foreign_receipt_never_becomes_repair_input(self):
        self.make_receipt('dense', 'failure')
        receipt = E.read(self.evidence / 'dense/receipt.json')
        receipt['identity_sha256'] = 'c' * 64
        E.write(self.evidence / 'dense/receipt.json', receipt)
        output = self.root / 'outputs'
        with patch.dict(os.environ, GITHUB_OUTPUT=str(output)):
            with self.assertRaises(ValueError):
                self.aggregate(feedback=self.root / 'feedback', family_result='failure')
        self.assertIn('repairable=false', output.read_text())
        self.assertIn('failure_class=contract', output.read_text())
        self.assertFalse((self.root / 'feedback').exists())

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
        model = {'mmproj_artifact': {'files': ['projector.gguf']}, 'class': 'causal_generation',
                 'certification_lanes': sorted(E.CORE)}
        with self.assertRaisesRegex(ValueError, 'multimodal'):
            E.validate_results(self.evidence / 'dense/results.jsonl', 'dense', model)

    def test_workload_family_requires_its_class_lanes(self):
        """Non-chat evidence must contain exactly its own smoke and oracle lanes."""
        model = {'family': 'dense', 'class': 'embedding', 'mmproj_artifact': None,
                 'certification_lanes': ['embedding-smoke', 'embedding-oracle']}
        row = {'family': 'dense', 'exit_code': 0, 'workload_class': 'embedding',
               'outcomes': [{'name': lane, 'status': 'pass', 'exit_code': 0}
                            for lane in ('embedding-smoke', 'embedding-oracle')]}
        path = self.evidence / 'dense/results.jsonl'
        path.write_text(json.dumps(row) + '\n')
        E.validate_results(path, 'dense', model)
        row['outcomes'] = row['outcomes'][:1]
        path.write_text(json.dumps(row) + '\n')
        with self.assertRaisesRegex(ValueError, 'required lane embedding-oracle'):
            E.validate_results(path, 'dense', model)

    def test_workload_class_mismatch_is_rejected(self):
        """A family result cannot substitute another class for the immutable plan."""
        model = {'family': 'dense', 'class': 'embedding', 'mmproj_artifact': None,
                 'certification_lanes': ['embedding-smoke', 'embedding-oracle']}
        row = {'family': 'dense', 'exit_code': 0, 'workload_class': 'rerank',
               'outcomes': [{'name': lane, 'status': 'pass', 'exit_code': 0}
                            for lane in ('embedding-smoke', 'embedding-oracle')]}
        path = self.evidence / 'dense/results.jsonl'
        path.write_text(json.dumps(row) + '\n')
        with self.assertRaisesRegex(ValueError, 'class mismatch'):
            E.validate_results(path, 'dense', model)

    def test_tampered_workload_closure_is_rejected(self):
        """Reject a changed producer closure before allowing any worker execution."""
        (self.package / E.WORKLOAD_ORACLES_TAR).write_bytes(b'wrong')
        with self.assertRaisesRegex(ValueError, 'digest mismatch'):
            E.verify_package(self.package, self.digest)

    def test_restore_materializes_relocatable_workload_closure(self):
        """Restored workers consume verified producer bytes without rebuilding them."""
        checkout = self.root / 'closure-checkout'
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
        closure = checkout / E.WORKLOAD_CLOSURE_ROOT
        stamp = closure / 'native/.mesh-llm-build-stamp'
        candidate = closure / 'cargo/debug/skippy'
        self.assertEqual(candidate.read_bytes(), b'closure:cargo/debug/skippy')
        self.assertLess(stamp.stat().st_mtime_ns, candidate.stat().st_mtime_ns,
                        'restored stamp must predate restored executables')

    def test_foreign_run_and_attempt_are_rejected(self):
        for key in ('run_id', 'run_attempt'):
            with self.subTest(key=key):
                original = self.identity[key]
                self.identity[key] = '999'
                self.save_identity()
                with self.assertRaisesRegex(ValueError, 'foreign workflow'):
                    E.verify_package(self.package, self.digest)
                self.identity[key] = original

    def test_prepared_source_inputs_are_digest_bound(self):
        for name in (E.LLAMA_BUNDLE, E.LLAMA_PROVENANCE):
            with self.subTest(name=name):
                path = self.package / name
                original = path.read_bytes()
                path.write_bytes(b'replaced')
                with self.assertRaisesRegex(ValueError, 'digest mismatch'):
                    E.verify_package(self.package, self.digest)
                path.write_bytes(original)

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

    def test_rerun_binds_verifier_to_producer_branch_from_identity(self):
        checkout = self.changed_candidate()
        env = dict(CANARY_SOURCE_ROOT=str(checkout), CANARY_MESH_SOURCE='',
                   CANARY_HARNESS_MODE='verify-build', CANARY_PASS_ID='verify-1',
                   CANARY_PREVIOUS_PACKAGE=str(self.package), CANARY_PREVIOUS_IDENTITY=self.digest,
                   CANARY_CANDIDATE_SHA=self.identity['candidate'], GITHUB_RUN_ATTEMPT='3')
        with patch.dict(os.environ, env), patch.object(E, 'preflight_battery'), \
                patch.object(E, 'git', return_value=self.identity['base']), \
                patch.object(E.subprocess, 'run') as run:
            E.build(SimpleNamespace())
        wrapper_env = run.call_args.kwargs['env']
        self.assertEqual(wrapper_env['CANARY_CANDIDATE_BRANCH'], self.identity['branch'])
        self.assertNotIn('-3-', wrapper_env['CANARY_CANDIDATE_BRANCH'])

    def test_candidate_cannot_replace_trusted_worker_code(self):
        checkout = self.changed_candidate(protected=True)
        with self.assertRaisesRegex(ValueError, 'trusted orchestration'):
            E.restore(SimpleNamespace(package=self.package, identity=self.digest, root=checkout))
        self.assertEqual(E.git(checkout, 'rev-parse', 'HEAD'), self.identity['base'])

    def test_selected_source_and_controller_are_bound_across_reruns(self):
        self.identity.update(controller='c'*40, mesh_source='a'*40, run_attempt='1')
        self.save_identity()
        with patch.dict(os.environ, CANARY_CONTROLLER_SHA='c'*40, CANARY_MESH_SOURCE='a'*40):
            E.verify_package(self.package, self.digest)
            for field, value in (('controller', 'd'*40), ('candidate', 'b'*40),
                                 ('base', 'b'*40), ('mesh_source', ''), ('pass_id', 'verify-1')):
                with self.subTest(field=field):
                    original = self.identity[field]
                    self.identity[field] = value
                    self.save_identity()
                    with self.assertRaises(ValueError):
                        E.verify_package(self.package, self.digest)
                    self.identity[field] = original
            self.save_identity()
        with self.assertRaisesRegex(ValueError, 'selected source identity'):
            E.verify_package(self.package, self.digest)

    def test_selected_source_build_uses_controller_wrapper_and_denies_repair(self):
        env = dict(CANARY_SOURCE_ROOT=str(self.root), CANARY_MESH_SOURCE='a'*40,
                   CANARY_HARNESS_MODE='pinned-build', CANARY_PASS_ID='repair-1',
                   CANARY_PREVIOUS_PACKAGE='')
        with patch.dict(os.environ, env), patch.object(E, 'git', return_value='a'*40), \
                patch.object(E, 'preflight_battery'), patch.object(E.subprocess, 'run') as run:
            E.build(SimpleNamespace())
            self.assertEqual(run.call_args.args[0], [str(ROOT / 'scripts/llama-canary-agent-repair.sh')])
            os.environ['CANARY_HARNESS_MODE'] = 'repair-build'
            with self.assertRaisesRegex(ValueError, 'unchanged pinned-build'):
                E.build(SimpleNamespace())
            self.assertEqual(run.call_count, 1)

    def test_publisher_rejects_repair_only_package(self):
        with self.assertRaisesRegex(ValueError, 'independent verifier'):
            E.publication(SimpleNamespace(package=self.package, identity=self.digest))


class WorkflowRerunContractTests(unittest.TestCase):
    def test_early_workload_gate_has_sdk_and_verifies_produced_candidate(self):
        workflow = yaml.safe_load((ROOT / '.github/workflows/llama-canary-family-pass.yml').read_text())
        build_steps = workflow['jobs']['build']['steps']
        sdk_index = next(i for i, step in enumerate(build_steps)
                         if step.get('uses') == './.github/actions/setup-canary-python')
        gate_index = next(i for i, step in enumerate(build_steps)
                          if step.get('name') == 'Build exact candidate for distributed certification')
        self.assertLess(sdk_index, gate_index)
        battery = (ROOT / 'skippy/scripts/skippy-family-battery.sh').read_text()
        self.assertIn('--candidate-executable "${SKIPPY_WORKLOAD_CANDIDATE_BIN_DIR:-$ROOT/target/debug}/skippy"', battery)
        self.assertNotIn('--candidate-executable "${SKIPPY_WORKLOAD_CANDIDATE_BIN_DIR:-$ROOT/target/debug}/skippy-serving"', battery)

    def test_family_battery_writes_compact_json_lines(self):
        battery = (ROOT / 'skippy/scripts/skippy-family-battery.sh').read_text()
        append = '>> "$RESULTS_JSONL"'
        writers = []
        for command in battery.split(append)[:-1]:
            start = max(command.rfind('\n  jq '), command.rfind('\n    jq '))
            self.assertNotEqual(start, -1)
            writers.append(command[start:].lstrip().splitlines()[0].strip())
        self.assertGreater(len(writers), 1)
        for writer in writers:
            with self.subTest(writer=writer):
                self.assertIn('-c', writer.split())

    def test_artifact_selection_is_bound_to_producer_across_attempts(self):
        workflow = yaml.safe_load((ROOT / '.github/workflows/llama-canary-family-pass.yml').read_text())
        jobs = workflow['jobs']
        package_output = jobs['build']['outputs']['package']
        package_upload = next(step for step in jobs['build']['steps']
                              if step.get('name') == 'Upload exact candidate and executable handoff')['with']['name']
        self.assertEqual(package_output, package_upload)
        self.assertIn('${{ github.run_attempt }}', package_output)
        self.assertIn('${{ steps.build.outputs.identity_sha256 }}', package_output)
        upload = next(step for step in jobs['family']['steps'] if step.get('name') == 'Upload family evidence')
        download = next(step for step in jobs['aggregate']['steps'] if 'pattern' in step.get('with', {}))
        name = upload['with']['name']
        pattern = download['with']['pattern']
        import fnmatch
        def expand(text, attempt, identity):
            for key, value in {'github.run_id': '123', 'github.run_attempt': str(attempt),
                               'needs.build.outputs.identity': identity,
                               'inputs.pass_id': 'repair-1', 'matrix.shard_index': '0'}.items():
                text = text.replace('${{ ' + key + ' }}', value)
            return text
        selected = expand(pattern, 3, 'a'*64)
        self.assertTrue(fnmatch.fnmatchcase(expand(name, 2, 'a'*64), selected))
        self.assertTrue(fnmatch.fnmatchcase(expand(name, 3, 'a'*64), selected))
        self.assertFalse(fnmatch.fnmatchcase(expand(name, 2, 'b'*64), selected))
        self.assertNotIn('merge-multiple', download['with'])
        gate = next(step for step in jobs['aggregate']['steps'] if step.get('id') == 'aggregate')
        self.assertIn('--family-result "$FAMILY_RESULT"', gate['run'])
        self.assertIn('--feedback "$RUNNER_TEMP/canary-family-feedback"', gate['run'])

    def test_failed_certification_is_retryable_after_evidence_upload(self):
        workflow = yaml.safe_load((ROOT / '.github/workflows/llama-canary-family-pass.yml').read_text())
        steps = workflow['jobs']['family']['steps']
        upload = next(i for i, step in enumerate(steps) if step.get('name') == 'Upload family evidence')
        gate = next(i for i, step in enumerate(steps) if step.get('name') == 'Require successful family certification')
        self.assertGreater(gate, upload)
        self.assertNotIn('continue-on-error', steps[gate])
        for outcome in ('success', 'failure', 'cancelled', 'skipped', ''):
            result = subprocess.run(['bash', '-c', steps[gate]['run']], env={**os.environ, 'OUTCOME': outcome})
            self.assertEqual(result.returncode == 0, outcome == 'success')

    def test_family_pass_binds_cross_pass_feedback(self):
        workflow = yaml.safe_load((ROOT / '.github/workflows/llama-canary-family-pass.yml').read_text())
        inputs = workflow[True]['workflow_call']['inputs']
        self.assertIn('previous_feedback', inputs)
        text = (ROOT / '.github/workflows/llama-canary-family-pass.yml').read_text()
        self.assertIn('CANARY_PREVIOUS_FEEDBACK', text)
        self.assertIn('llama-family-feedback-', text)


if __name__ == '__main__':
    unittest.main()
