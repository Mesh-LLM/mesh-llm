"""Execute the source-layout action and its UI producer/consumer shell boundary."""
from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

import yaml

ROOT = Path(__file__).resolve().parents[2]
ACTION = ROOT / '.github/actions/resolve-source-layout/action.yml'


def workflow(name):
    return yaml.safe_load((ROOT / '.github/workflows' / name).read_text())


class SourceLayoutTests(unittest.TestCase):
    def resolve(self, root, prefix=''):
        for relative in ('crates/mesh-llm-ui', 'website', 'sdk'):
            (root / prefix / relative).mkdir(parents=True, exist_ok=True)
        output = root / 'outputs'
        output.write_text('')
        script = yaml.safe_load(ACTION.read_text())['runs']['steps'][0]['run']
        result = subprocess.run(
            ['bash', '-euo', 'pipefail', '-c', script], cwd=root,
            env={**os.environ, 'GITHUB_OUTPUT': str(output)},
            capture_output=True, text=True, check=False,
        )
        values = dict(line.split('=', 1) for line in output.read_text().splitlines())
        return result, values

    def test_legacy_and_relocated_source_resolve_without_creating_aliases(self):
        for prefix in ('', 'mesh'):
            with self.subTest(prefix=prefix), tempfile.TemporaryDirectory(prefix='layout space ') as tmp:
                root = Path(tmp)
                result, values = self.resolve(root, prefix)
                self.assertEqual(result.returncode, 0, result.stderr)
                for key, relative in [('ui_dir', 'crates/mesh-llm-ui'), ('website_dir', 'website'), ('sdk_dir', 'sdk')]:
                    self.assertEqual(values[key], str(Path(prefix) / relative))
                if prefix:
                    self.assertFalse((root / 'crates').exists())

    def test_ambiguous_component_fails(self):
        for relative in ('crates/mesh-llm-ui', 'website', 'sdk'):
            with self.subTest(component=relative), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                (root / 'mesh' / relative).mkdir(parents=True)
                result, _ = self.resolve(root)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn('ambiguous source layout', result.stderr)

    def test_missing_component_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _, _ = self.resolve(root)
            shutil.rmtree(root / 'website')
            script = yaml.safe_load(ACTION.read_text())['runs']['steps'][0]['run']
            result = subprocess.run(['bash', '-euo', 'pipefail', '-c', script], cwd=root,
                env={**os.environ, 'GITHUB_OUTPUT': str(root / 'outputs')}, capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('missing source directory: website or mesh/website', result.stderr)

    def test_real_ui_producer_and_host_verification_steps_use_same_tree(self):
        producer = workflow('ci-ui-artifact-slice.yml')['jobs']['ui_artifact']['steps']
        verify_build = next(s for s in producer if s.get('name') == 'Verify console distribution')
        upload = next(s for s in producer if s.get('name') == 'Upload immutable console distribution')
        self.assertEqual(upload['with']['path'], '${{ steps.layout.outputs.ui_dir }}/dist')
        for prefix in ('', 'mesh'):
            with self.subTest(prefix=prefix), tempfile.TemporaryDirectory(prefix='layout space ') as tmp:
                root = Path(tmp)
                result, paths = self.resolve(root, prefix)
                self.assertEqual(result.returncode, 0, result.stderr)
                ui = root / paths['ui_dir']
                (ui / 'dist').mkdir()
                (ui / 'dist/index.html').write_text('<html>built console</html>')
                subprocess.run(['bash', '-euc', verify_build['run']], cwd=ui, check=True)
                for name, job in [('ci-linux-host-slice.yml', 'linux_host'), ('ci-macos-host-slice.yml', 'macos_host'), ('sdk-smoke.yml', 'sdk_smoke')]:
                    steps = workflow(name)['jobs'][job]['steps']
                    download = next(s for s in steps if s.get('name') == 'Download immutable UI distribution')
                    self.assertEqual(download['with']['path'], upload['with']['path'])
                    verify = next(s for s in steps if s.get('name') == 'Verify UI distribution input')
                    env = {**os.environ, 'UI_DIR': paths['ui_dir']}
                    subprocess.run(['bash', '-euc', verify['run']], cwd=root, env=env, check=True)
                    (ui / 'dist/index.html').write_text('')
                    rejected = subprocess.run(['bash', '-euc', verify['run']], cwd=root, env=env)
                    self.assertNotEqual(rejected.returncode, 0)
                    (ui / 'dist/index.html').write_text('<html>built console</html>')

    def test_ui_steps_resolve_after_checkout_and_do_not_use_job_defaults(self):
        for name, jobs in [('ci-web-slice.yml', ['ui_quality', 'ui_e2e']), ('ci-ui-artifact-slice.yml', ['ui_artifact'])]:
            for job_name in jobs:
                job = workflow(name)['jobs'][job_name]
                self.assertNotIn('working-directory', job.get('defaults', {}).get('run', {}))
                steps = job['steps']
                resolve = next(i for i, step in enumerate(steps) if step.get('id') == 'layout')
                checkout = next(i for i, step in enumerate(steps) if step.get('uses', '').startswith('actions/checkout@'))
                self.assertGreater(resolve, checkout)
                for step in steps:
                    if step.get('run', '').startswith('pnpm '):
                        self.assertEqual(step['working-directory'], '${{ steps.layout.outputs.ui_dir }}')


if __name__ == '__main__':
    unittest.main()
