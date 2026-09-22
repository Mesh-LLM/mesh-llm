"""Memory-tier boundaries, verified-plan routing and real process cleanup."""
import copy
import importlib.util
import json
import os
import signal
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location('memory_evidence', ROOT / 'scripts/llama-canary-family-evidence.py')
E = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(E)
M = E.MEMORY


def model(size=1, kind='causal_generation', projector=0):
    def artifact(value):
        return {'files': ['model.gguf'], 'file_integrity': {'model.gguf': {'size_bytes': value}}}
    return {'family': 'fixture', 'class': kind, 'artifact': artifact(size),
            'resources': {'estimated_model_bytes': size},
            'mmproj_artifact': artifact(projector) if projector else None}


class MemoryTests(unittest.TestCase):
    def test_exact_boundaries_and_invalid_estimates(self):
        small = 128 * M.GIB * 85 // 100
        large = 256 * M.GIB * 85 // 100
        self.assertEqual(M.tier_for(small), 'accelerator-memory-128plus')
        self.assertEqual(M.tier_for(small + 1), 'accelerator-memory-256plus')
        self.assertEqual(M.tier_for(large), 'accelerator-memory-256plus')
        for invalid in (large + 1, 0, -1, True, 1.2, '12'):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                M.tier_for(invalid)

    def test_projector_and_concurrent_oracle_change_routing(self):
        self.assertEqual(M.placement(model(70*M.GIB))['memory_tier'], 'accelerator-memory-128plus')
        for value in (model(70*M.GIB, projector=20*M.GIB), model(70*M.GIB, kind='embedding')):
            self.assertEqual(M.placement(value)['memory_tier'], 'accelerator-memory-256plus')
        with self.assertRaisesRegex(ValueError, 'fixture.*exceeds'):
            M.placement(model(200*M.GIB))

    def test_missing_or_understated_artifact_bytes_fail_safe(self):
        value = model(100*M.GIB)
        value['resources']['estimated_model_bytes'] = 1
        self.assertEqual(M.placement(value)['memory_tier'], 'accelerator-memory-256plus')
        value['artifact']['file_integrity'] = {}
        with self.assertRaises(ValueError):
            M.placement(value)

    def test_real_roster_routes_large_models_without_mutating_plan(self):
        plan = E.load_planner(ROOT).build_plan(ROOT / 'ci/llama-canary/family-certified.json', shard_count=256)
        original = copy.deepcopy(plan)
        rows = {r['families']: r for r in E.scheduling_matrix(plan)['include']}
        self.assertEqual(plan, original)
        for family in ('minimax-m3', 'inkling'):
            self.assertEqual(rows[family]['memory_tier'], 'accelerator-memory-256plus')
        self.assertEqual(rows['lfm2-vl']['memory_tier'], 'accelerator-memory-128plus')

    def test_admission_checks_actual_host_and_other_workloads(self):
        estimate = M.placement(model(100*M.GIB))
        with self.assertRaisesRegex(ValueError, 'too small'):
            M.admission(estimate, 128*M.GIB, 128*M.GIB)
        with self.assertRaisesRegex(ValueError, 'available'):
            M.admission(estimate, 256*M.GIB, 160*M.GIB)
        self.assertEqual(M.admission(estimate, 256*M.GIB, 256*M.GIB), (256*M.GIB*15+99)//100)

    def test_vm_stat_does_not_count_purgeable_twice(self):
        text = ('Mach Virtual Memory Statistics: (page size of 16384 bytes)\n'
                'Pages free: 10.\nPages inactive: 20.\nPages speculative: 3.\nPages purgeable: 5.\n')
        self.assertEqual(M.parse_vm_stat(text), 33*16384)
        with self.assertRaises(ValueError):
            M.parse_vm_stat(text.replace('Pages inactive', 'Missing'))

    def run_guard(self, command, observations, tier='accelerator-memory-128plus'):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        directory = Path(temp.name)
        with patch.object(M, 'host_memory', side_effect=observations):
            result = M.guarded_run(model(), tier, command, directory)
        return result, json.loads((directory / 'memory-admission.json').read_text())

    def test_wrong_tier_never_starts_child(self):
        with self.assertRaisesRegex(ValueError, 'tier differs'):
            self.run_guard(['/should/not/run'], [], 'accelerator-memory-256plus')

    def test_insufficient_memory_never_starts_child(self):
        with self.assertRaisesRegex(ValueError, 'available'):
            self.run_guard(['/should/not/run'], [(128*M.GIB, 20*M.GIB)])

    @unittest.skipIf(os.name != 'posix', 'process-group guard is macOS/POSIX')
    def test_pressure_stops_real_child_and_records_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            evidence = Path(directory)
            pidfile = evidence / 'child.pid'
            command = [sys.executable, '-c',
                       'import os,time,pathlib; pathlib.Path(' + repr(str(pidfile)) + ').write_text(str(os.getpid())); time.sleep(60)']
            def observe():
                if not pidfile.exists():
                    return 128*M.GIB, 128*M.GIB
                return 128*M.GIB, M.GIB
            with patch.object(M, 'host_memory', side_effect=observe):
                with self.assertRaisesRegex(ValueError, '15% reserve'):
                    M.guarded_run(model(), 'accelerator-memory-128plus', command, evidence)
            pid = int(pidfile.read_text())
            with self.assertRaises(ProcessLookupError):
                os.kill(pid, 0)
            report = json.loads((evidence / 'memory-admission.json').read_text())
            self.assertEqual(report['status'], 'failed')
            self.assertIn('15% reserve', report['error'])

    @unittest.skipIf(os.name != 'posix', 'process-group guard is macOS/POSIX')
    def test_child_failure_is_not_certification_success(self):
        with patch.object(M, 'host_memory', return_value=(128*M.GIB,128*M.GIB)):
            with tempfile.TemporaryDirectory() as directory:
                result = M.guarded_run(model(), 'accelerator-memory-128plus',
                                       [sys.executable, '-c', 'raise SystemExit(7)'], Path(directory))
                self.assertEqual(result, 7)
                self.assertEqual(json.loads((Path(directory)/'memory-admission.json').read_text())['status'], 'failed')

    @unittest.skipIf(os.name != 'posix', 'process-group guard is macOS/POSIX')
    def test_successful_parent_does_not_leave_background_writer(self):
        with tempfile.TemporaryDirectory() as directory:
            evidence = Path(directory)
            marker = evidence / 'writer'
            child = (f"import pathlib,time\np=pathlib.Path({str(marker)!r})\n"
                     "while True:\n with p.open('a') as f: f.write('x'); f.flush()\n time.sleep(0.02)\n")
            parent = (f"import subprocess,sys,time,pathlib\nsubprocess.Popen([sys.executable, '-c', {child!r}])\n"
                      f"p=pathlib.Path({str(marker)!r})\nwhile not p.exists(): time.sleep(0.01)\n")
            with patch.object(M, 'host_memory', return_value=(128*M.GIB, 128*M.GIB)):
                result = M.guarded_run(model(), 'accelerator-memory-128plus',
                                       [sys.executable, '-c', parent], evidence)
            self.assertEqual(result, 0)
            length = marker.stat().st_size
            self.assertGreater(length, 0)
            time.sleep(0.1)
            self.assertEqual(marker.stat().st_size, length)

    def test_signal_during_spawn_defers_cleanup_until_child_is_owned(self):
        handlers = {}
        child = Mock()
        spawning = False
        def install(number, handler):
            previous = handlers.get(number, signal.SIG_DFL)
            handlers[number] = handler
            return previous
        def spawn(*args, **kwargs):
            nonlocal spawning
            spawning = True
            handlers[signal.SIGTERM](signal.SIGTERM, None)
            spawning = False
            return child
        def cleanup(process):
            self.assertFalse(spawning)
            self.assertIs(process, child)
            handlers[signal.SIGINT](signal.SIGINT, None)
        with tempfile.TemporaryDirectory() as directory, \
             patch.object(M, 'host_memory', return_value=(128*M.GIB, 128*M.GIB)), \
             patch.object(M.signal, 'signal', side_effect=install), \
             patch.object(M.subprocess, 'Popen', side_effect=spawn), \
             patch.object(M, 'stop_group', side_effect=cleanup) as stop:
            with self.assertRaisesRegex(InterruptedError, 'signal'):
                M.guarded_run(model(), 'accelerator-memory-128plus', ['fixture'], Path(directory))
            stop.assert_called_once_with(child)
            self.assertEqual(json.loads((Path(directory)/'memory-admission.json').read_text())['status'], 'failed')
        self.assertEqual(handlers, {signal.SIGTERM: signal.SIG_DFL, signal.SIGINT: signal.SIG_DFL})

    def test_workflow_uses_guard_and_locked_sdk_for_historical_source(self):
        import yaml
        family = yaml.safe_load((ROOT/'.github/workflows/llama-canary-family-pass.yml').read_text())['jobs']['family']
        self.assertIn('${{ matrix.memory_tier }}', family['runs-on'])
        steps = family['steps']
        self.assertTrue(any(s.get('uses') == './.github/actions/setup-canary-python' for s in steps))
        certify = next(s for s in steps if s.get('id') == 'certify')
        self.assertIn('llama-canary-family-evidence.py certify', certify['run'])
        self.assertIn('--root "$CANARY_SOURCE_ROOT"', certify['run'])
        self.assertIn('--identity "$IDENTITY"', certify['run'])
