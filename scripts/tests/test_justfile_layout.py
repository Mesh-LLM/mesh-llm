from __future__ import annotations

import json
import os
from pathlib import Path
import re
import subprocess
import tarfile
import tempfile
from typing import Final
import unittest


ROOT: Final = Path(__file__).resolve().parents[2]
IMPORTS: Final = (
    "just/build.just",
    "just/release-build.just",
    "just/skippy.just",
    "just/mesh.just",
    "just/release-bundle.just",
    "just/website-ui.just",
    "just/ci.just",
    "just/mesh-client.just",
    "just/utilities.just",
)
RECIPES_BY_FILE: Final = {
    "just/build.just": {
        "bootstrap-build-tools", "build", "build-dev",
        "qa-logging-console-e2e", "with-lld",
        "build-openai-exchange-exemplar", "package-openai-exchange-exemplar",
        "test-openai-exchange-conformance",
    },
    "just/release-build.just": {
        "llama-build", "llama-prepare", "llama-prepare-latest", "release",
        "release-build", "release-build-aarch64", "release-build-aarch64-cuda",
        "release-build-cuda", "release-build-cuda-windows", "release-build-rocm",
        "release-build-rocm-windows", "release-build-vulkan",
        "release-build-vulkan-windows",
        "release-host-build", "release-runtime-build",
    },
    "just/skippy.just": {
        "bench-corpus", "competitive-benchmark-build", "family-certify",
        "metrics-server", "metrics-server-build",
        "skippy-native-full-replay", "skippy-native-tests", "skippy-openai-smoke",
        "skippy-rewriter-build",
        "skippy-workload-oracles-build",
        "skippy-quantize-build",
        "skippy-quantize-release-build", "skippy-quantize-standalone-build",
        "skippy-quantize-standalone-release-build", "skippy-wan-lab-build-bins",
        "skippy", "skippy-cli-build", "skippy-cli-release-build", "skippy-release",
        "spec-bench",
    },
    "just/mesh.just": {"bundle", "download-model", "mesh", "mesh-join", "mesh-worker"},
    "just/release-bundle.just": {
        "check-env-mutation-contract", "check-release", "release-attestation",
        "release-bundle",
    },
    "just/website-ui.just": {
        "cli-inventory-check", "crate-docs", "ui-dev", "ui-dev-public", "ui-test", "website-build",
        "website-clean", "website-dev",
    },
    "just/ci.just": {
        "automation-bootstrap", "automation-run", "ci-crate-lists", "ci-sccache-seed-build", "ci-shellcheck", "ci-validate",
        "no-console-print", "publish-crates", "test-all",
    },
    "just/mesh-client.just": {"auto", "mesh-client"},
    "just/utilities.just": {
        "cache-cargo-clean", "cache-cargo-metadata", "cache-prune",
        "cache-prune-dry-run", "cache-status", "check-commits", "clean",
        "diff", "docker-build-client", "docker-run-client", "hooks-install",
        "llama-summary", "llama-update-pin", "stop", "test", "ui-clean",
    },
}
RECIPE_HEADER: Final = re.compile(r"^([A-Za-z_][\w-]*)(?:\s+[^:]*)?:(?!=)")


class JustfileLayoutTests(unittest.TestCase):
    def test_exemplar_package_uses_current_cargo_artifact_instead_of_default_target(self) -> None:
        source = (ROOT / "just/build.just").read_text(encoding="utf-8")
        recipe = source.split("package-openai-exchange-exemplar:\n", 1)[1].split(
            "\n# Run installed-process", 1
        )[0]
        for target in ("configured-target", "env-target/debug", "env-build-target/aarch64-apple-darwin/debug"):
            with self.subTest(target=target), tempfile.TemporaryDirectory() as temporary:
                directory = Path(temporary)
                artifact = directory / target / "examples/openai-exchange-observer"
                artifact.parent.mkdir(parents=True)
                artifact.write_text('#!/bin/sh\nprintf \'{"fresh":true}\\n\'\n', encoding="utf-8")
                artifact.chmod(0o755)
                stale = directory / "target/debug/examples/openai-exchange-observer"
                stale.parent.mkdir(parents=True)
                stale.write_text("stale executable", encoding="utf-8")
                shim = directory / "bin/just"
                shim.parent.mkdir()
                shim.write_text(
                    '#!/bin/sh\n[ "$*" = "build-openai-exchange-exemplar json" ] || exit 2\n'
                    'printf \'%s\\n\' "$CARGO_ARTIFACT_JSON"\n', encoding="utf-8"
                )
                shim.chmod(0o755)
                justfile = directory / "fixture.just"
                justfile.write_text("package-openai-exchange-exemplar:\n" + recipe, encoding="utf-8")
                environment = dict(os.environ, PATH=str(shim.parent) + os.pathsep + os.environ["PATH"])
                environment["CARGO_ARTIFACT_JSON"] = json.dumps({
                    "reason": "compiler-artifact", "target": {"name": "openai-exchange-observer"},
                    "executable": str(artifact),
                }) + '\n' + json.dumps({"reason": "build-finished", "success": True})
                driver = subprocess.check_output(["which", "just"], text=True).strip()
                subprocess.run([driver, "--justfile", str(justfile), "package-openai-exchange-exemplar"],
                               cwd=directory, env=environment, check=True, capture_output=True)
                with tarfile.open(directory / "dist/openai-exchange-observer.tar.gz") as archive:
                    executable = archive.extractfile("openai-exchange-observer/openai-exchange-observer")
                    manifest = archive.extractfile("openai-exchange-observer/plugin-manifest.json")
                    self.assertEqual(executable.read(), artifact.read_bytes())
                    self.assertEqual(json.load(manifest), {"fresh": True})
                environment["CARGO_ARTIFACT_JSON"] = '{"reason":"build-finished","success":true}'
                (directory / "dist/openai-exchange-observer.tar.gz").unlink()
                missing = subprocess.run([driver, "--justfile", str(justfile), "package-openai-exchange-exemplar"],
                                         cwd=directory, env=environment, capture_output=True)
                self.assertNotEqual(missing.returncode, 0)
                self.assertFalse((directory / "dist/openai-exchange-observer.tar.gz").exists())

    def test_windows_skips_only_unix_exemplar_conformance(self) -> None:
        source = (ROOT / "just/ci.just").read_text(encoding="utf-8")
        exemplar_stage = source.split('echo "=== 6/11 Plugin author exemplar ==="', 1)[1]
        guard = exemplar_stage.split('    case "$(uname -s)" in\n', 1)[1].split("    esac\n", 1)[0]
        script = 'case "$(uname -s)" in\n' + guard + "esac\njust portable-author-check\n"
        for platform in ("MINGW64_NT-10.0", "MSYS_NT-10.0", "CYGWIN_NT-10.0", "Linux", "Darwin"):
            with self.subTest(platform=platform), tempfile.TemporaryDirectory() as temporary:
                directory = Path(temporary)
                for name, body in {"uname": '#!/bin/sh\nprintf \'%s\\n\' "$TEST_PLATFORM"\n',
                                   "just": '#!/bin/sh\nprintf \'called:%s\\n\' "$*"\n'}.items():
                    shim = directory / name
                    shim.write_text(body, encoding="utf-8")
                    shim.chmod(0o755)
                environment = dict(os.environ, PATH=str(directory) + os.pathsep + os.environ["PATH"],
                                   TEST_PLATFORM=platform)
                output = subprocess.check_output(["bash", "-e", "-c", script], env=environment, text=True)
                self.assertIn("called:portable-author-check", output)
                self.assertEqual("called:test-openai-exchange-conformance" in output,
                                 platform in ("Linux", "Darwin"))

    def test_root_keeps_prelude_default_and_ordered_flat_imports(self) -> None:
        source = (ROOT / "Justfile").read_text(encoding="utf-8")
        imports = re.findall(r"(?m)^import '([^']+)'$", source)

        self.assertEqual(tuple(imports), IMPORTS)
        self.assertIn("# Distributed LLM Inference — build & run tasks", source)
        self.assertIn("default: build", source)
        self.assertNotRegex(source, r"(?m)^\s*(?:mod\b|import\?)")

    def test_each_recipe_stays_in_its_owning_import(self) -> None:
        for relative_path, expected in RECIPES_BY_FILE.items():
            with self.subTest(relative_path=relative_path):
                source = (ROOT / relative_path).read_text(encoding="utf-8")
                actual = {
                    match.group(1)
                    for line in source.splitlines()
                    if (match := RECIPE_HEADER.match(line)) is not None
                }
                self.assertEqual(actual, expected)

    def test_only_with_lld_is_private_and_imports_create_no_modules(self) -> None:
        private_sources = {
            relative_path: (ROOT / relative_path).read_text(encoding="utf-8").count("[private]\n")
            for relative_path in IMPORTS
        }
        dump = json.loads(
            subprocess.check_output(
                ["just", "--dump", "--dump-format", "json"], cwd=ROOT, text=True
            )
        )

        self.assertEqual(private_sources["just/build.just"], 2)
        self.assertEqual(private_sources["just/ci.just"], 1)
        self.assertTrue(all(count == 0 for path, count in private_sources.items() if path not in {"just/build.just", "just/ci.just"}))
        self.assertNotIn("automation-run", subprocess.check_output(["just", "--summary"], cwd=ROOT, text=True).split())
        self.assertNotIn("with-lld", subprocess.check_output(["just", "--summary"], cwd=ROOT, text=True).split())
        self.assertEqual(dump["modules"], {})
        self.assertEqual(dump["first"], "default")

    def test_short_product_recipes_keep_mesh_independent(self) -> None:
        skippy = (ROOT / "just/skippy.just").read_text(encoding="utf-8")
        mesh = (ROOT / "just/mesh.just").read_text(encoding="utf-8")
        orchestrator = (ROOT / "mesh/scripts/build-development-product.sh").read_text(
            encoding="utf-8"
        )
        self.assertIn('skippy backend="" cuda_arch="" rocm_arch="":', skippy)
        self.assertIn('skippy/scripts/build-development-product.sh --backend "{{ backend }}"', skippy)
        self.assertIn('mesh profile="debug":', mesh)
        self.assertIn('scripts/build-host.sh --profile "{{ profile }}"', mesh)
        self.assertIn('-HostOnly', mesh)
        self.assertNotIn("skippy/scripts/build-development-product.sh", mesh)
        self.assertLess(
            orchestrator.index('just skippy "$BACKEND" "$CUDA_ARCH" "$ROCM_ARCH"'),
            orchestrator.index('just mesh "$PROFILE"'),
        )


if __name__ == "__main__":
    unittest.main()
