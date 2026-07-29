from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import tarfile
import tempfile
import tomllib
import unittest


ROOT = Path(__file__).resolve().parents[2]
ACTIONS = ROOT / ".github" / "actions"
COMPOSE_SCRIPT = ROOT / "scripts" / "ci-compose-product-input.sh"
RELEASE_FOOTER_MANIFEST = ROOT / "crates" / "mesh-llm-release-footer" / "Cargo.toml"
XTASK_MANIFEST = ROOT / "tools" / "xtask" / "Cargo.toml"


class CiArtifactActionTests(unittest.TestCase):
    def read_action(self, name: str) -> str:
        return (ACTIONS / name / "action.yml").read_text(encoding="utf-8")

    def test_external_actions_have_sha_and_release_provenance(self) -> None:
        action_files = sorted(ACTIONS.glob("*/action.yml"))
        workflow_files = sorted(
            (ROOT / ".github" / "workflows").glob("*.yml"),
        )
        exact_pin = re.compile(
            r"^[^@\s]+@[0-9a-f]{40}\s+#\s+\S",
        )

        for path in (*action_files, *workflow_files):
            for line_number, line in enumerate(
                path.read_text(encoding="utf-8").splitlines(),
                start=1,
            ):
                if "uses:" not in line:
                    continue
                value = line.split("uses:", maxsplit=1)[1].strip()
                if value.startswith("./"):
                    continue
                with self.subTest(
                    path=path.relative_to(ROOT),
                    line=line_number,
                ):
                    self.assertRegex(value, exact_pin)

    def write_fake_product_inputs(
        self,
        workspace: Path,
        *,
        host_version: str = "1.2.3",
    ) -> tuple[Path, Path]:
        host_input = workspace / "host-input"
        runtime_input = workspace / "runtime-input"
        host_input.mkdir()
        runtime_input.mkdir()

        host = host_input / "mesh-llm"
        host.write_text(
            "#!/usr/bin/env bash\n"
            f"printf 'mesh-llm {host_version}\\n'\n",
            encoding="utf-8",
        )
        host.chmod(0o755)
        host_digest = hashlib.sha256(host.read_bytes()).hexdigest()
        (host_input / "mesh-llm.sha256").write_text(
            f"{host_digest}  mesh-llm\n",
            encoding="utf-8",
        )
        (host_input / "host-imports.json").write_text(
            "{}\n",
            encoding="utf-8",
        )

        runtime_id = "meshllm-native-runtime-test-x86_64-cpu"
        runtime = runtime_input / runtime_id
        (runtime / "lib").mkdir(parents=True)
        (runtime / "tools").mkdir()
        library = runtime / "lib" / "libmesh_fake.a"
        library.write_bytes(b"fake static library")
        tool = runtime / "tools" / "mesh-runtime-bench"
        tool.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
        tool.chmod(0o755)
        library_digest = hashlib.sha256(library.read_bytes()).hexdigest()
        tool_digest = hashlib.sha256(tool.read_bytes()).hexdigest()
        manifest = {
            "runtime": {
                "id": runtime_id,
                "mesh_version": "1.2.3",
                "skippy_abi": {"major": 1, "minor": 0, "patch": 0},
                "platform": {"os": "test", "arch": "x86_64"},
                "backend": {"kind": "cpu"},
                "libraries": ["lib/libmesh_fake.a"],
                "tools": {"tools/mesh-runtime-bench": tool_digest},
            },
            "build": {
                "backend": "cpu",
                "primary_library": "lib/libmesh_fake.a",
                "library_sha256": library_digest,
            },
        }
        (runtime / "manifest.json").write_text(
            json.dumps(manifest) + "\n",
            encoding="utf-8",
        )
        return host_input, runtime_input

    def run_product_composer(
        self,
        workspace: Path,
        *,
        host_version: str = "1.2.3",
    ) -> subprocess.CompletedProcess[str]:
        host_input, runtime_input = self.write_fake_product_inputs(
            workspace,
            host_version=host_version,
        )
        return subprocess.run(
            [str(COMPOSE_SCRIPT)],
            cwd=ROOT,
            env={
                **os.environ,
                "GITHUB_WORKSPACE": str(workspace),
                "GITHUB_OUTPUT": str(workspace / "github-output"),
                "INPUT_HOST_INPUT_DIR": str(host_input),
                "INPUT_RUNTIME_INPUT_DIR": str(runtime_input),
                "INPUT_OUTPUT_DIR": str(workspace / "product-input"),
                "INPUT_BACKEND": "cpu",
                "INPUT_VERSION": "1.2.3",
                "INPUT_BINARY_NAME": "mesh-llm",
                "INPUT_READINESS_SMOKE": "false",
            },
            check=False,
            capture_output=True,
            text=True,
        )

    def run_runner_selector(
        self,
        *,
        event_name: str,
        ref: str,
        main_enabled: str,
        manual_enabled: str,
    ) -> dict[str, str]:
        action = self.read_action("select-ci-runners")
        run_block = action.split("      run: |\n", maxsplit=1)[1]
        script = "\n".join(
            line[8:] if line.startswith("        ") else line
            for line in run_block.splitlines()
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "github-output"
            result = subprocess.run(
                ["bash", "-c", script],
                cwd=ROOT,
                env={
                    **os.environ,
                    "GITHUB_OUTPUT": str(output),
                    "INPUT_EVENT_NAME": event_name,
                    "INPUT_REF": ref,
                    "INPUT_DEPOT_MAIN_ENABLED": main_enabled,
                    "INPUT_MANUAL_USE_DEPOT": manual_enabled,
                },
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            return dict(
                line.split("=", maxsplit=1)
                for line in output.read_text(encoding="utf-8").splitlines()
            )

    def test_host_action_uses_canonical_dynamic_host_builder(self) -> None:
        action = self.read_action("prepare-host-input")

        self.assertIn('scripts/build-host.sh --profile "$INPUT_PROFILE"', action)
        self.assertIn("scripts/verify-host-dependencies.py", action)
        self.assertNotIn("package-native-runtime.sh", action)

    def test_windows_host_action_owns_the_neutral_host_integrity_contract(
        self,
    ) -> None:
        action = self.read_action("prepare-windows-host-input")

        self.assertIn(
            "& .\\scripts\\build-windows.ps1 -BuildProfile $profile -HostOnly",
            action,
        )
        self.assertIn("scripts\\verify-host-dependencies.py", action)
        self.assertIn("mesh-llm.exe.sha256", action)
        self.assertIn("cargo build -q -p xtask --bin xtask", action)
        self.assertIn("release-attestation stamp", action)
        self.assertIn("release-attestation inspect", action)
        self.assertIn('"$attestationVerifierPath.sha256"', action)
        self.assertIn(
            '"$verifierHash  release-attestation-verifier.exe"',
            action,
        )
        self.assertNotIn("package-native-runtime.sh", action)
        self.assertNotIn("compose-product", action)

    def test_windows_attestation_verifier_stays_native_abi_free(self) -> None:
        xtask = tomllib.loads(XTASK_MANIFEST.read_text(encoding="utf-8"))
        xtask_dependencies = xtask["dependencies"]
        self.assertEqual(
            xtask_dependencies["mesh-llm-release-footer"],
            {"workspace": True},
        )
        self.assertNotIn("mesh-llm-system", xtask_dependencies)
        self.assertNotIn("skippy-ffi", xtask_dependencies)

        footer = tomllib.loads(RELEASE_FOOTER_MANIFEST.read_text(encoding="utf-8"))
        self.assertEqual(set(footer["dependencies"]), {"hex", "sha2"})

    def test_windows_debug_host_uses_the_package_version_for_composition(
        self,
    ) -> None:
        action = self.read_action("prepare-windows-host-input")

        debug = action[
            action.index('if ($profile -eq "debug")')
            : action.index('if ($env:INPUT_SKIP_UI -eq "true")')
        ]
        self.assertIn("cargo pkgid -p mesh-llm", debug)
        self.assertIn("$env:MESH_LLM_BUILD_VERSION", debug)
        self.assertNotIn("git ", debug)

    def test_windows_routes_cover_every_shared_product_primitive(self) -> None:
        action = self.read_action("compute-changes")
        routing = action[
            action.index("WINDOWS_CPU_INPUTS=")
            : action.index("# SDK smokes are consumer tests")
        ]
        cpu_routing = routing[: routing.index("WINDOWS_GPU_INPUTS=")]
        gpu_routing = routing[routing.index("WINDOWS_GPU_INPUTS=") :]

        self.assertIn("^crates/mesh-llm-release-footer/", cpu_routing)
        self.assertNotIn("^crates/mesh-llm-release-footer/", gpu_routing)

        for primitive in (
            "prepare-windows-host-input",
            "prepare-native-runtime-input",
            "compose-product-input",
            "package-native-runtime",
            "verify-native-runtime-package",
            "compose-product-bundle",
            "ci-compose-product-input",
            "ci-client-readiness-smoke",
        ):
            with self.subTest(primitive=primitive):
                self.assertIn(primitive, routing)

    def test_runtime_action_never_builds_the_host(self) -> None:
        action = self.read_action("prepare-native-runtime-input")

        self.assertIn('scripts/package-native-runtime.sh "${args[@]}"', action)
        self.assertIn("scripts/verify-native-runtime-package.sh", action)
        self.assertNotIn("build-host.sh", action)
        self.assertNotIn("build-release.sh", action)

    def test_product_action_only_composes_verified_inputs(self) -> None:
        action = self.read_action("compose-product-input")

        self.assertIn("scripts/ci-compose-product-input.sh", action)
        self.assertNotIn("cargo build", action)
        self.assertNotIn("package-native-runtime.sh", action)
        script = COMPOSE_SCRIPT.read_text(encoding="utf-8")
        self.assertIn("scripts/compose-product-bundle.py", script)
        self.assertIn("scripts/verify-native-runtime-package.sh", script)
        self.assertIn("scripts/ci-client-readiness-smoke.sh", script)
        self.assertIn('archive_path="$product_dir.tar.gz"', script)
        self.assertIn('tar -C "$product_dir" -czf "$archive_path" .', script)

    def test_product_composer_normalizes_windows_shell_boundaries(self) -> None:
        script = COMPOSE_SCRIPT.read_text(encoding="utf-8")

        self.assertIn("local path=\"${1%$'\\r'}\"", script)
        self.assertIn('cygpath -u "$path"', script)
        self.assertIn('cygpath -m "$path"', script)
        self.assertIn(
            'canonical_paths+=("$(to_shell_path "$path")")',
            script,
        )
        self.assertIn(
            'GITHUB_OUTPUT="$(to_shell_path "$GITHUB_OUTPUT")"',
            script,
        )
        self.assertIn('require_file "immutable host" "$host"', script)
        self.assertNotIn('test -f "$host"', script)

    def test_product_archive_preserves_verified_executable_modes(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            result = self.run_product_composer(workspace)

            self.assertEqual(result.returncode, 0, result.stderr)
            archive = workspace / "product-input.tar.gz"
            self.assertTrue(archive.is_file())
            with tarfile.open(archive, "r:gz") as bundle:
                host = next(
                    member
                    for member in bundle.getmembers()
                    if member.name.endswith("/mesh-llm")
                )
                tool = next(
                    member
                    for member in bundle.getmembers()
                    if member.name.endswith(
                        "/tools/mesh-runtime-bench"
                    )
                )
                self.assertNotEqual(host.mode & 0o111, 0)
                self.assertNotEqual(tool.mode & 0o111, 0)
            output = (workspace / "github-output").read_text(encoding="utf-8")
            self.assertIn(f"archive_path={archive.resolve()}", output)

    def test_product_composer_rejects_host_version_drift(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            result = self.run_product_composer(
                Path(temp_dir),
                host_version="9.9.9",
            )

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("composed host version mismatch", result.stderr)

    def test_release_attestation_is_verified_without_compiling_in_composer(
        self,
    ) -> None:
        host_action = self.read_action("prepare-host-input")
        product_action = self.read_action("compose-product-input")
        product_script = COMPOSE_SCRIPT.read_text(encoding="utf-8")

        self.assertIn("cargo build -q -p xtask --bin xtask", host_action)
        self.assertIn("release-attestation-verifier.sha256", host_action)
        self.assertNotIn("cargo ", product_action)
        self.assertIn(
            '"$attestation_verifier" release-attestation inspect',
            product_script,
        )
        self.assertIn(
            '"$expected_verifier_checksum" \\\n'
            '        "$actual_verifier_checksum"',
            product_script,
        )

    def test_smoke_restore_rechecks_the_archived_product(self) -> None:
        action = self.read_action("restore-smoke-inputs")

        self.assertIn("expected exactly one composed product archive", action)
        self.assertIn("tar -xzf", action)
        self.assertIn("scripts/verify-native-runtime-package.sh", action)
        self.assertIn("--check", action)

    def test_smoke_restore_model_is_optional(self) -> None:
        action = self.read_action("restore-smoke-inputs")
        model_inputs_present = (
            "inputs.model_url != '' && inputs.model_file != ''"
        )

        self.assertEqual(action.count(model_inputs_present), 4)
        self.assertIn(
            f"if: ${{{{ {model_inputs_present} }}}}\n"
            "      id: cache-model",
            action,
        )
        self.assertIn(
            f"if: ${{{{ {model_inputs_present} }}}}\n"
            "      id: model-file",
            action,
        )

    def test_product_action_rejects_destructive_output_paths(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            host_input = workspace / "inputs" / "host"
            runtime_input = workspace / "inputs" / "runtime"
            host_input.mkdir(parents=True)
            runtime_input.mkdir(parents=True)
            sentinel = workspace / "sentinel"
            sentinel.write_text("keep", encoding="utf-8")
            outside = workspace.parent / f"{workspace.name}-outside"
            dangerous_outputs = (
                ".",
                "./",
                "product/..",
                str(workspace),
                str(outside),
                str(host_input),
                str(host_input / "product"),
                str(workspace / "inputs"),
            )

            for output in dangerous_outputs:
                with self.subTest(output=output):
                    result = subprocess.run(
                        [str(COMPOSE_SCRIPT)],
                        cwd=workspace,
                        env={
                            **os.environ,
                            "GITHUB_WORKSPACE": str(workspace),
                            "GITHUB_OUTPUT": str(workspace / "github-output"),
                            "INPUT_HOST_INPUT_DIR": str(host_input),
                            "INPUT_RUNTIME_INPUT_DIR": str(runtime_input),
                            "INPUT_OUTPUT_DIR": output,
                            "INPUT_BACKEND": "cpu",
                            "INPUT_VERSION": "",
                            "INPUT_BINARY_NAME": "mesh-llm",
                            "INPUT_READINESS_SMOKE": "false",
                        },
                        check=False,
                        capture_output=True,
                        text=True,
                    )
                    self.assertNotEqual(result.returncode, 0)
                    self.assertEqual(sentinel.read_text(encoding="utf-8"), "keep")

    def test_sccache_prefers_depot_webdav_with_disk_fallback(self) -> None:
        action = self.read_action("configure-sccache-gha")

        self.assertIn("allow_depot_remote_cache", action)
        self.assertIn('default: "false"', action)
        self.assertIn("SCCACHE_WEBDAV_ENDPOINT", action)
        self.assertIn("DEPOT_CACHE_TOKEN", action)
        self.assertIn("process.env.SCCACHE_DIR", action)
        self.assertIn("process.env.RUNNER_TEMP", action)
        self.assertIn("await io.mkdirP(diskCacheDirectory)", action)
        self.assertIn(
            "core.exportVariable('SCCACHE_DIR', diskCacheDirectory)",
            action,
        )
        self.assertIn("SCCACHE_DIR: diskCacheDirectory", action)
        self.assertIn("'disk,webdav'", action)
        self.assertIn("'disk'", action)
        self.assertIn(
            "Depot cache is present but disabled for this trust context",
            action,
        )
        self.assertIn(
            "'Unable to start baked sccache with its trust-isolated disk cache.'",
            action,
        )
        self.assertIn("env: diskOnlyEnvironment()", action)
        self.assertNotIn(
            "core.exportVariable('ACTIONS_RUNTIME_TOKEN', '')",
            action,
        )

    def test_runner_selection_never_routes_pull_requests_to_depot(self) -> None:
        action = self.read_action("select-ci-runners")

        self.assertIn("depot_main_enabled", action)
        self.assertNotIn("depot_pr_enabled", action)
        self.assertNotIn("head_repository", action)
        self.assertNotIn("\n  repository:", action)
        self.assertIn("\n  ref:", action)

        runtime = action.split("runs:", maxsplit=1)[1]
        pull_request_case = runtime.split(
            "pull_request|pull_request_target)",
            maxsplit=1,
        )[1].split(";;", maxsplit=1)[0]
        self.assertIn("depot_enabled=false", pull_request_case)
        self.assertNotIn("depot_enabled=true", pull_request_case)
        self.assertNotIn("INPUT_DEPOT_PR_ENABLED", runtime)
        self.assertNotIn("INPUT_HEAD_REPOSITORY", runtime)
        self.assertNotIn("INPUT_REPOSITORY", runtime)

        dispatch_case = runtime.split(
            "workflow_dispatch)",
            maxsplit=1,
        )[1].split(";;", maxsplit=1)[0]
        self.assertIn("INPUT_DEPOT_MAIN_ENABLED", dispatch_case)
        self.assertIn("INPUT_MANUAL_USE_DEPOT", dispatch_case)
        self.assertIn('INPUT_REF" == "refs/heads/main"', dispatch_case)
        self.assertIn("depot_enabled=true", dispatch_case)

        push_case = runtime.split(
            "push)",
            maxsplit=1,
        )[1].split(";;", maxsplit=1)[0]
        self.assertIn("INPUT_DEPOT_MAIN_ENABLED", push_case)
        self.assertIn('INPUT_REF" == "refs/heads/main"', push_case)
        self.assertIn("depot_enabled=true", push_case)

        default_case = runtime.split(
            "*)",
            maxsplit=1,
        )[1].split(";;", maxsplit=1)[0]
        self.assertIn("depot_enabled=false", default_case)
        self.assertNotIn("depot_enabled=true", default_case)
        self.assertIn("depot-ubuntu-24.04-16", action)

        cases = (
            ("pull_request", "refs/pull/12/merge", "true", "true", "false", "ubuntu-24.04"),
            ("pull_request_target", "refs/heads/main", "true", "true", "false", "ubuntu-24.04"),
            ("workflow_dispatch", "refs/heads/main", "false", "true", "true", "depot-ubuntu-24.04"),
            ("workflow_dispatch", "refs/heads/feature", "true", "true", "false", "ubuntu-24.04"),
            ("push", "refs/heads/main", "true", "false", "true", "depot-ubuntu-24.04"),
            ("push", "refs/heads/feature", "true", "false", "false", "ubuntu-24.04"),
            ("push", "refs/tags/v1.2.3", "true", "false", "false", "ubuntu-24.04"),
            ("push", "refs/heads/main", "false", "false", "false", "ubuntu-24.04"),
            ("schedule", "refs/heads/main", "true", "true", "false", "ubuntu-24.04"),
        )
        for event_name, ref, main, manual, enabled, runner in cases:
            with self.subTest(event_name=event_name, ref=ref):
                outputs = self.run_runner_selector(
                    event_name=event_name,
                    ref=ref,
                    main_enabled=main,
                    manual_enabled=manual,
                )
                self.assertEqual(outputs["depot_enabled"], enabled)
                self.assertEqual(outputs["allow_depot_remote_cache"], enabled)
                self.assertEqual(outputs["runner"], runner)

    def test_pr_caches_rely_on_github_ref_scoping_while_depot_is_blocked(
        self,
    ) -> None:
        for workflow_name in ("pr_builds.yml", "pr_quality.yml"):
            workflow = (
                ROOT / ".github" / "workflows" / workflow_name
            ).read_text(encoding="utf-8")
            with self.subTest(workflow=workflow_name):
                self.assertIn("CACHE_NAMESPACE: mesh-llm", workflow)
                self.assertNotIn("CACHE_NAMESPACE: mesh-llm-pr", workflow)
                self.assertNotIn("'mesh-llm-pr'", workflow)
                self.assertIn(
                    "save-if: ${{ github.ref == 'refs/heads/main' }}",
                    workflow,
                )

        quality = (
            ROOT / ".github" / "workflows" / "pr_quality.yml"
        ).read_text(encoding="utf-8")
        builds = (
            ROOT / ".github" / "workflows" / "pr_builds.yml"
        ).read_text(encoding="utf-8")
        self.assertIn(
            "allow_depot_remote_cache: "
            "${{ needs.changes.outputs.allow_depot_remote_cache }}",
            builds,
        )
        self.assertIn(
            "allow_depot_remote_cache: "
            "${{ needs.changes.outputs.allow_depot_remote_cache }}",
            quality,
        )
        self.assertIn("${{ env.CACHE_NAMESPACE }}-pnpm-", quality)
        self.assertNotIn("cache: pnpm", quality)


if __name__ == "__main__":
    unittest.main()
