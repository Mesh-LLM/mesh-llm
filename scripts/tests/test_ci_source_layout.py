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

    def test_node_addon_staging_uses_resolved_sdk_on_every_platform(self):
        jobs = workflow('node-sdk-addon-artifact.yml')['jobs']
        checked = 0
        for job in jobs.values():
            steps = job.get('steps', [])
            build = next((s for s in steps if s.get('name') == 'Build, smoke, and stage immutable addon'), None)
            if build is None:
                continue
            checked += 1
            self.assertEqual(build['env']['SDK_DIR'], '${{ steps.layout.outputs.sdk_dir }}')
            self.assertLess(next(i for i, s in enumerate(steps) if s.get('id') == 'layout'), steps.index(build))
            script = build['run']
            target = 'linux-x64' if 'unsupported Linux' in script else 'darwin-arm64' if 'unsupported macOS' in script else 'win32-x64'
            # Exercise real version checks, npm prefix selection, and addon lookup.
            # Native compilation is replaced by a fixture-producing npm executable.
            prefix = script.split('          smoke_root=', 1)[0] if '          smoke_root=' in script else script.split('smoke_root=', 1)[0]
            for layout in ('', 'mesh'):
                with self.subTest(target=target, layout=layout), tempfile.TemporaryDirectory(prefix='node layout ') as tmp:
                    root = Path(tmp)
                    result, paths = self.resolve(root, layout)
                    self.assertEqual(result.returncode, 0, result.stderr)
                    node = root / paths['sdk_dir'] / 'node'
                    node.mkdir()
                    (node / 'package.json').write_text('{"version":"1.2.3"}')
                    bins = root / 'bin'
                    bins.mkdir()
                    npm = bins / 'npm'
                    npm.write_text('#!/usr/bin/env python3\nimport os, pathlib, sys\np = pathlib.Path(sys.argv[sys.argv.index("--prefix") + 1])\nassert (p / "package.json").is_file()\nif "build:native" in sys.argv:\n p = p / "native" / os.environ["NODE_SDK_TARGET"] / "mesh_llm_nodejs.node"\n p.parent.mkdir(parents=True)\n p.write_text("fixture")\n')
                    npm.chmod(0o755)
                    env = {**os.environ, 'PATH': str(bins) + os.pathsep + os.environ['PATH'], 'SDK_DIR': paths['sdk_dir'], 'NODE_SDK_TARGET': target, 'RELEASE_TAG': 'v1.2.3', 'PREPARE_RELEASE_VERSION': 'false'}
                    result = subprocess.run(['bash', '-euc', prefix], cwd=root, env=env, capture_output=True, text=True)
                    self.assertEqual(result.returncode, 0, result.stderr)
                    env['RELEASE_TAG'] = 'v1.2.4'
                    rejected = subprocess.run(['bash', '-euc', prefix], cwd=root, env=env, capture_output=True, text=True)
                    self.assertNotEqual(rejected.returncode, 0)
                    self.assertIn('Node SDK version mismatch', rejected.stderr)
        self.assertEqual(checked, 3)

    def test_backend_and_sdk_classification_accepts_both_layouts(self):
        source = (ROOT / '.github/actions/compute-changes/derive-outputs.sh').read_text()
        script = source[source.index('BACKEND_CHANGED="false"'):source.index('# Inference artifacts are needed')]
        script += '\nprintf "%s %s %s %s" "$BACKEND_CHANGED" "$WINDOWS_CPU_BUILD_REQUIRED" "$WINDOWS_GPU_BUILD_REQUIRED" "$SDK_SMOKE_REQUIRED"\n'
        cases = {
            'third_party/llama.cpp/upstream.txt': 'true true true false',
            'skippy/third_party/llama.cpp/patches/test.patch': 'true true true false',
            'sdk/node/index.js': 'false false false true',
            'mesh/sdk/node/index.js': 'false false false true',
            'skippy/scripts/build-llama.sh': 'true false false true',
            'RESEARCH.md': 'false false false false',
        }
        for changed, expected in cases.items():
            with self.subTest(changed=changed):
                env = {**os.environ, 'CHANGED_FILES': changed, 'ALL_RUST': 'false', 'FORCE_ALL': 'false', 'EVENT_NAME': 'push', 'AFFECTED_CRATES': '[]', 'BACKEND_RECIPE_CHANGED': 'false'}
                result = subprocess.run(['bash', '-euc', script], env=env, capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(result.stdout.strip(), expected)

    @unittest.skipUnless(shutil.which('just'), 'needs just')
    def test_website_recipes_consume_the_resolved_layout(self):
        """`just website-build` must not fall back to a root `website/` tree."""
        justfile = (ROOT / 'Justfile').read_text(encoding='utf-8')
        self.assertIn('MESH_LLM_WEBSITE_DIR', justfile)
        for name, job_name, step_name in (
            ('ci-web-slice.yml', 'website', 'Build public website'),
            ('ci-quality-slice.yml', 'cli_docs_sync', 'Verify generated CLI inventory is deterministic and current'),
        ):
            steps = workflow(name)['jobs'][job_name]['steps']
            layout = next(i for i, step in enumerate(steps) if step.get('id') == 'layout')
            index = next(i for i, step in enumerate(steps) if step.get('name') == step_name)
            self.assertLess(layout, index)
            self.assertEqual(
                steps[index]['env']['MESH_LLM_WEBSITE_DIR'],
                '${{ steps.layout.outputs.website_dir }}',
            )
        evaluated = subprocess.run(
            ['just', '--evaluate', 'website_dir'],
            cwd=ROOT,
            env={**os.environ, 'MESH_LLM_WEBSITE_DIR': 'mesh/website'},
            capture_output=True,
            text=True,
        )
        self.assertEqual(evaluated.returncode, 0, evaluated.stderr)
        self.assertEqual(evaluated.stdout.strip(), 'mesh/website')
        fallback = subprocess.run(
            ['just', '--evaluate', 'website_dir'],
            cwd=ROOT,
            env={key: value for key, value in os.environ.items() if key != 'MESH_LLM_WEBSITE_DIR'},
            capture_output=True,
            text=True,
        )
        self.assertEqual(fallback.returncode, 0, fallback.stderr)
        self.assertEqual(fallback.stdout.strip(), 'website')


    def test_relocated_runtime_owner_gates_sdk_smoke_and_inference_artifacts(self):
        """A relocated runtime owner must still select the artifact consumers."""
        source = (ROOT / '.github/actions/compute-changes/derive-outputs.sh').read_text()
        script = source[source.index('SDK_SMOKE_REQUIRED="false"'):source.index('LINUX_TEST_GROUPS_JSON')]
        script += '\nprintf "%s %s" "$SDK_SMOKE_REQUIRED" "$INFERENCE_ARTIFACT_REQUIRED"\n'
        cases = {
            '["mesh-llm-native-runtime"]': 'true true',
            '["skippy-native-runtime"]': 'true true',
            '["mesh-llm-config"]': 'true true',
            '[]': 'false false',
        }
        for affected, expected in cases.items():
            with self.subTest(affected=affected):
                env = {
                    **os.environ,
                    'CHANGED_FILES': 'mesh/crates/skippy-native-runtime/src/lib.rs',
                    'ALL_RUST': 'false',
                    'FORCE_ALL': 'false',
                    'EVENT_NAME': 'push',
                    'UI_CHANGED': 'false',
                    'BACKEND_CHANGED': 'false',
                    'AFFECTED_CRATES': affected,
                }
                result = subprocess.run(['bash', '-euc', script], env=env, capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(result.stdout.strip(), expected)


    def test_nightly_pin_resolution_rejects_missing_and_ambiguous_layouts(self):
        steps = workflow('llama-upstream-canary.yml')['jobs']['resolve']['steps']
        body = next(s['run'] for s in steps if s.get('id') == 'resolve')
        script = body[body.index('pins=()'):body.index('upstream="$UPSTREAM"')]
        script += '\nprintf "%s" "$old"\n'
        for paths in ([], ['third_party/llama.cpp/upstream.txt'], ['skippy/third_party/llama.cpp/upstream.txt'], ['third_party/llama.cpp/upstream.txt', 'skippy/third_party/llama.cpp/upstream.txt']):
            with self.subTest(paths=paths), tempfile.TemporaryDirectory() as tmp:
                for relative in paths:
                    path = Path(tmp) / relative
                    path.parent.mkdir(parents=True)
                    path.write_text('a'*40 + '\n')
                result = subprocess.run(['bash', '-euc', script], cwd=tmp, capture_output=True, text=True)
                self.assertEqual(result.returncode == 0, len(paths) == 1, result.stderr)
                if len(paths) == 1:
                    self.assertEqual(result.stdout, 'a'*40)

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
