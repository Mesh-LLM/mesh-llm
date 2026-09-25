from __future__ import annotations

import json
from pathlib import Path
import re
import tempfile
import tomllib
import unittest

import yaml


ROOT = Path(__file__).resolve().parents[2]
ACTIONS = ROOT / ".github" / "actions"
PLATFORM_CHECKS_WORKFLOW = ROOT / ".github" / "workflows" / "ci-platform-checks-slice.yml"
RESOLVE_CARGO_PACKAGES = "Mesh-LLM/mesh-llm/.github/actions/resolve-cargo-packages"
RELEASE_FOOTER_MANIFEST = ROOT / "crates" / "mesh-llm-release-footer" / "Cargo.toml"
XTASK_MANIFEST = ROOT / "tools" / "xtask" / "Cargo.toml"


def _unit_row_crates(platform: str) -> set[str]:
    """Crates a platform's unit row runs, however the row resolves them.

    The row no longer names a literal `foreach ($crate in '...', '...')` list:
    each platform's owners come from the `crates:` inputs of the
    `resolve-cargo-packages` steps that apply to that platform's unit check.
    A step with no platform guard applies to both unit rows; a platform-guarded
    step applies only to the named platform.
    """
    workflow = yaml.safe_load(PLATFORM_CHECKS_WORKFLOW.read_text(encoding="utf-8"))
    crates: set[str] = set()
    found = False
    for step in workflow["jobs"]["platform_checks"]["steps"]:
        if not str(step.get("uses", "")).startswith(RESOLVE_CARGO_PACKAGES + "@"):
            continue
        condition = str(step.get("if", ""))
        if "kind == 'unit'" not in condition:
            continue
        if "platform == '" in condition and f"platform == '{platform}'" not in condition:
            continue
        crates.update(json.loads(step["with"]["crates"]))
        found = True
    if not found:
        raise AssertionError(f"{platform} unit row no longer resolves any package owners")
    return crates


def _windows_unit_row_crates() -> set[str]:
    """Crates the windows-unit row runs, across its shared and Windows inputs."""
    return _unit_row_crates("windows")


# Crates whose `src/` selects Windows- or Unix-specific code that no Windows
# job compiles yet. Their suites have never run on the Windows runner, so
# switching them on together would risk a red main for reasons unrelated to the
# change under test. A crate leaves this list, for a platform-windows* crate
# rule and the windows-unit row, once its suite is confirmed green there.
WINDOWS_UNVERIFIED_CRATES = {
    "mesh-llm-analytics",
    "mesh-llm-commands",
    "mesh-llm-config",
    "mesh-llm-events",
    "mesh-llm-hardware-profile",
    "mesh-llm-identity",
    "mesh-llm-native-runtime",
    "mesh-llm-plugin-manager",
    "mesh-llm-system",
    "mesh-llm-tui",
    "mesh-llm-ui",
    "model-hf",
    "skippy-bench",
    "skippy-model-package",
    "skippy-quantize",
    "skippy-runtime",
    "skippy-server",
    "xtask",
}

CFG_OPEN = re.compile(r"\b(cfg!?|cfg_attr)\s*\(")
BARE_PLATFORM = re.compile(r"\b(?:windows|unix)\b")
PLATFORM_KEY = re.compile(r"\btarget_(?:os|family)\s*=\s*\"(\d+)\"")
RAW_STRING = re.compile(r"b?r(#*)\"")
CHAR_LITERAL = re.compile(r"'(?:\\(?:u\{[0-9a-fA-F]+\}|x[0-9a-fA-F]{2}|.)|[^\\'\n])'")


def _mask_rust_source(source: str) -> tuple[str, list[str]]:
    """Drop comments and replace each string literal with `"<index>"`.

    A cfg is then only found in code, never in a comment or a quoted example,
    and a `target_os` value stays readable through the returned literals.
    """
    code: list[str] = []
    literals: list[str] = []
    index, length = 0, len(source)
    while index < length:
        if source.startswith("//", index):
            end = source.find("\n", index)
            index = length if end == -1 else end
        elif source.startswith("/*", index):
            depth, index = 1, index + 2
            while index < length and depth:
                if source.startswith("/*", index):
                    depth, index = depth + 1, index + 2
                elif source.startswith("*/", index):
                    depth, index = depth - 1, index + 2
                else:
                    index += 1
            code.append(" ")
        elif (raw := RAW_STRING.match(source, index)) and (
            index == 0 or not (source[index - 1].isalnum() or source[index - 1] == "_")
        ):
            closing = '"' + raw.group(1)
            end = source.find(closing, raw.end())
            end = length if end == -1 else end
            literals.append(source[raw.end() : end])
            code.append(f'"{len(literals) - 1}"')
            index = end + len(closing)
        elif source[index] == '"':
            end = index + 1
            while end < length and source[end] != '"':
                end += 2 if source[end] == "\\" else 1
            literals.append(source[index + 1 : end])
            code.append(f'"{len(literals) - 1}"')
            index = end + 1
        elif (char := CHAR_LITERAL.match(source, index)) is not None:
            code.append("' '")
            index = char.end()
        else:
            code.append(source[index])
            index += 1
    return "".join(code), literals


def _cfg_predicates(code: str) -> list[str]:
    """The predicate of every `cfg(...)`, `cfg!(...)` and `cfg_attr(...)`.

    Balanced, not up to the first `)`: in `cfg(all(not(test), windows))` the
    platform predicate comes after a nested clause. For `cfg_attr` only the
    part before the first top-level comma is a predicate; the rest is an
    attribute such as `windows_subsystem = "windows"`.
    """
    predicates = []
    for match in CFG_OPEN.finditer(code):
        depth, index, cut = 1, match.end(), None
        while index < len(code) and depth:
            if code[index] == "(":
                depth += 1
            elif code[index] == ")":
                depth -= 1
            elif code[index] == "," and depth == 1 and cut is None:
                cut = index
            index += 1
        end = index - 1
        if match.group(1) == "cfg_attr" and cut is not None:
            end = cut
        predicates.append(code[match.end() : end])
    return predicates


def _selects_platform_code(source: str) -> bool:
    """Whether a cfg predicate names `windows` or `unix`, directly or as the
    value of `target_os` / `target_family`. A `feature = "windows"` is not a
    platform, and a cfg inside a comment or a string is not code."""
    code, literals = _mask_rust_source(source)
    for predicate in _cfg_predicates(code):
        if BARE_PLATFORM.search(predicate):
            return True
        if any(literals[int(i)] in ("windows", "unix") for i in PLATFORM_KEY.findall(predicate)):
            return True
    return False


def _workspace_crates(root: Path) -> dict[str, Path]:
    """Package name to crate directory for every workspace member.

    Reads the members from the root manifest rather than a fixed `crates/*`
    layout, and the name from each crate's manifest, since a directory need
    not match its package name.
    """
    workspace = tomllib.loads((root / "Cargo.toml").read_text(encoding="utf-8"))
    crates: dict[str, Path] = {}
    for member in workspace["workspace"]["members"]:
        for crate_dir in sorted(root.glob(member)):
            manifest = tomllib.loads((crate_dir / "Cargo.toml").read_text(encoding="utf-8"))
            crates[manifest["package"]["name"]] = crate_dir
    return crates


def _cfg_divergent_crates(root: Path) -> set[str]:
    """Workspace crates whose `src/` selects Windows- or Unix-specific code."""
    divergent = set()
    for name, crate_dir in _workspace_crates(root).items():
        source_root = crate_dir / "src"
        if not source_root.is_dir():
            continue
        if any(
            _selects_platform_code(source.read_text(encoding="utf-8", errors="ignore"))
            for source in source_root.rglob("*.rs")
        ):
            divergent.add(name)
    return divergent


def _windows_routed_crates() -> set[str]:
    """Crates a Windows job compiles: platform-windows* rules plus the unit row."""
    ownership = json.loads((ROOT / "ci" / "ownership.yml").read_text(encoding="utf-8"))
    routed = {
        crate
        for rule in ownership["crate_rules"]
        if rule["domain"].startswith("platform-windows")
        for crate in rule["crates"]
    }
    return routed | _windows_unit_row_crates()


class CiWindowsCompositionTests(unittest.TestCase):
    def read_action(self, name: str) -> str:
        return (ACTIONS / name / "action.yml").read_text(encoding="utf-8")

    def read_compute_changes(self) -> str:
        return self.read_action("compute-changes") + "\n" + (
            ACTIONS / "compute-changes" / "derive-outputs.sh"
        ).read_text(encoding="utf-8")

    def test_windows_cache_warmup_supports_focused_cpu_dispatch(self) -> None:
        workflow = (
            ROOT / ".github" / "workflows" / "windows-warm-caches.yml"
        ).read_text(encoding="utf-8")

        self.assertIn("workload:", workflow)
        self.assertIn("inputs.workload == 'cpu'", workflow)
        self.assertIn(
            "github.event_name != 'workflow_dispatch' || inputs.workload == 'all'",
            workflow,
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
        action = self.read_compute_changes()
        routing = action[
            action.index("WINDOWS_CPU_INPUTS=")
            : action.index("# SDK smokes are consumer tests")
        ]
        cpu_routing = routing[: routing.index("WINDOWS_GPU_INPUTS=")]
        gpu_routing = routing[routing.index("WINDOWS_GPU_INPUTS=") :]

        self.assertIn("^(mesh/|skippy/)?crates/mesh-llm-release-footer/", cpu_routing)
        self.assertNotIn("^(mesh/|skippy/)?crates/mesh-llm-release-footer/", gpu_routing)
        self.assertIn("package-release", cpu_routing)
        self.assertIn("package-release", gpu_routing)
        for workflow in (
            "ci",
            "main_[a-z]+",
            "pr_[a-z]+",
            "release",
            "windows-warm-caches",
        ):
            with self.subTest(workflow=workflow):
                self.assertIn(workflow, cpu_routing)
                self.assertIn(workflow, gpu_routing)

        for input_name, route in (
            ("WINDOWS_CPU_INPUTS", cpu_routing),
            ("WINDOWS_GPU_INPUTS", gpu_routing),
        ):
            with self.subTest(input_name=input_name):
                match = re.search(
                    rf"{input_name}=.*?grep -E '([^']+)'",
                    route,
                )
                self.assertIsNotNone(
                    match,
                    f"{input_name} classifier pattern was not found",
                )
                classifier = re.compile(match.group(1))
                for action_path in (
                    ".github/actions/compute-changes/action.yml",
                    ".github/actions/compute-changes/derive-outputs.sh",
                ):
                    with self.subTest(action_path=action_path):
                        self.assertRegex(action_path, classifier)

        for primitive in (
            "prepare-windows-host-input",
            "prepare-native-runtime-input",
            "compose-product-input",
            "save-and-verify-actions-cache",
            "package-native-runtime",
            "verify-native-runtime-package",
            "verify-checksum-sidecar",
            "safe-extract-tar",
            "compose-product-bundle",
            "ci-compose-product-input",
            "ci-client-readiness-smoke",
        ):
            with self.subTest(primitive=primitive):
                self.assertIn(primitive, routing)

    def test_windows_abi_cache_action_keys_every_compatibility_boundary(
        self,
    ) -> None:
        action = self.read_action("restore-windows-abi-cache")

        for action_input in (
            "backend:",
            "build_dir:",
            "toolchain_epoch:",
            "architecture_set:",
            "cuda_toolchain_version:",
            "vulkan_toolchain_version:",
            "rocm_toolchain_version:",
        ):
            with self.subTest(action_input=action_input):
                self.assertIn(action_input, action)

        self.assertIn(
            '$backend -notin @("cpu", "cuda", "rocm", "vulkan")',
            action,
        )
        self.assertIn(
            '$backend -in @("cuda", "rocm") -and -not $architectureSet',
            action,
        )
        self.assertIn(
            "build_dir must resolve inside GITHUB_WORKSPACE",
            action,
        )
        self.assertIn(
            "build_dir must remain outside the replaceable llama.cpp ",
            action,
        )
        self.assertIn(
            "worktree: $resolvedBuildDir",
            action,
        )
        for toolchain_boundary in (
            "cuda-$version-Jimver-v0.2.35",
            "vulkan-$version-jakoch-v1.5.2",
            "rocm-$version",
        ):
            with self.subTest(toolchain_boundary=toolchain_boundary):
                self.assertIn(toolchain_boundary, action)

        expected_hash = (
            "${{ hashFiles("
            "'.github/actions/restore-windows-abi-cache/action.yml', "
            "'.github/actions/save-and-verify-actions-cache/action.yml', "
            "'.github/actions/resolve-native-toolchain-epoch/action.yml', "
            "'.github/actions/prepare-native-runtime-input/action.yml', "
            "'.github/actions/setup-windows-rocm-sdk/action.yml', "
            "'scripts/build-llama.sh', 'scripts/prepare-llama.sh', "
            "'scripts/package-native-runtime.sh', "
            "'third_party/llama.cpp/upstream.txt', "
            "'third_party/llama.cpp/patches/**', "
            "'.github/cache-version.txt') }}"
        )
        self.assertIn(expected_hash, action)
        self.assertIn(
            '"mesh-llm-windows-2022-skippy-abi-'
            '$backend-$architectureSet-$toolchain-$toolchainEpoch-$inputHash"',
            action,
        )
        self.assertIn(
            "toolchain_epoch must match MESH_LLM_LLAMA_TOOLCHAIN_EPOCH",
            action,
        )
        self.assertIn(
            "actions/cache/restore@"
            "caa296126883cff596d87d8935842f9db880ef25 # v5.1.0",
            action,
        )
        self.assertNotIn("restore-keys:", action)
        self.assertIn(
            "value: ${{ steps.restore.outputs.cache-hit }}",
            action,
        )
        self.assertIn(
            "value: ${{ steps.restore.outputs.cache-primary-key }}",
            action,
        )
        self.assertIn(
            "value: ${{ steps.identity.outputs.build-dir }}",
            action,
        )

    def test_windows_native_cache_inputs_fail_closed_and_callers_opt_in(
        self,
    ) -> None:
        for action_name in (
            "restore-windows-abi-cache",
            "setup-windows-rocm-sdk",
        ):
            with self.subTest(action=action_name):
                action = self.read_action(action_name)
                input_start = action.index("  allow-native-github-cache:")
                input_end = action.find("\n\n", input_start)
                input_block = action[input_start:input_end]
                self.assertIn('required: false', input_block)
                self.assertIn('default: "false"', input_block)
                self.assertNotIn('default: "true"', input_block)

        expected_callers = {
            "restore-windows-abi-cache": {
                "ci-windows-runtime-slice.yml": 1,
                "release.yml": 2,
                "windows-warm-caches.yml": 2,
            },
            "setup-windows-rocm-sdk": {
                "ci-windows-runtime-slice.yml": 1,
                "release.yml": 1,
                "windows-warm-caches.yml": 1,
            },
        }
        policy_value = (
            "allow-native-github-cache: "
            "${{ needs.runner_policy.outputs.allow_native_github_cache }}"
        )
        for action_name, expected_counts in expected_callers.items():
            calls: list[tuple[str, str]] = []
            for workflow_path in sorted(
                (ROOT / ".github" / "workflows").glob("*.yml")
            ):
                lines = workflow_path.read_text(encoding="utf-8").splitlines()
                for index, line in enumerate(lines):
                    marker = f"uses: ./.github/actions/{action_name}"
                    if marker not in line:
                        continue
                    line_indent = len(line) - len(line.lstrip())
                    step_indent = line_indent
                    for candidate in reversed(lines[:index]):
                        candidate_indent = len(candidate) - len(candidate.lstrip())
                        if candidate_indent <= line_indent and candidate.lstrip().startswith("-"):
                            step_indent = candidate_indent
                            break
                    start = index
                    while start > 0:
                        candidate = lines[start - 1]
                        candidate_indent = len(candidate) - len(candidate.lstrip())
                        if candidate_indent == step_indent and candidate.lstrip().startswith("-"):
                            start -= 1
                            break
                        if candidate_indent < step_indent:
                            break
                        start -= 1
                    end = index + 1
                    while end < len(lines):
                        candidate = lines[end]
                        candidate_indent = len(candidate) - len(candidate.lstrip())
                        if candidate_indent == step_indent and candidate.lstrip().startswith("-"):
                            break
                        end += 1
                    calls.append((workflow_path.name, "\n".join(lines[start:end])))

            actual_counts: dict[str, int] = {}
            for workflow_name, block in calls:
                actual_counts[workflow_name] = actual_counts.get(workflow_name, 0) + 1
                with self.subTest(action=action_name, workflow=workflow_name):
                    if workflow_name == "ci-windows-runtime-slice.yml":
                        self.assertIn(policy_value, block)
                    else:
                        self.assertIn(
                            'allow-native-github-cache: "true"',
                            block,
                        )
            self.assertEqual(expected_counts, actual_counts)

    def test_windows_unit_row_names_every_routed_crate(self) -> None:
        """Routing a crate to windows-unit is only signal if the row names it."""
        ownership = json.loads((ROOT / "ci" / "ownership.yml").read_text(encoding="utf-8"))
        domain_crates = {
            crate
            for rule in ownership["crate_rules"]
            if rule["domain"] == "platform-windows-cfg"
            for crate in rule["crates"]
        }
        self.assertTrue(domain_crates)
        self.assertEqual(set(), domain_crates - _windows_unit_row_crates())

    def test_windows_unit_row_keeps_shared_owners_without_widening_macos(self) -> None:
        shared = {"model-artifact", "mesh-llm-host-runtime", "mesh-llm"}
        # The shared macOS/Windows owners stay identical and platform-neutral.
        self.assertLessEqual(shared, _windows_unit_row_crates())
        self.assertLessEqual(shared, _unit_row_crates("macos"))
        # The platform-windows-cfg crate runs on Windows only.
        self.assertIn("mesh-llm-plugin", _windows_unit_row_crates())
        self.assertNotIn("mesh-llm-plugin", _unit_row_crates("macos"))

    def test_windows_unit_row_resolves_crates_through_the_package_resolver(self) -> None:
        source = PLATFORM_CHECKS_WORKFLOW.read_text(encoding="utf-8")
        self.assertIn("steps.windows_packages.outputs.crates", source)
        self.assertNotIn("foreach ($crate in 'model-artifact'", source)

    def test_every_cfg_divergent_crate_is_routed_or_explicitly_unverified(self) -> None:
        """Platform-divergent code must be compiled on Windows or declared not to be.

        A crate whose `src/` holds `cfg(windows)` / `cfg(unix)` code can break
        on Windows while every Linux job stays green; that is how #1978 shipped
        a lib-test target that had never compiled there.
        """
        unaccounted = sorted(
            _cfg_divergent_crates(ROOT) - _windows_routed_crates() - WINDOWS_UNVERIFIED_CRATES
        )
        self.assertEqual(
            [],
            unaccounted,
            "these crates carry platform-divergent code but no Windows job "
            "compiles them; route them to a Windows row or add them to "
            "WINDOWS_UNVERIFIED_CRATES",
        )

    def test_cfg_detector_marks_every_platform_form_and_nothing_else(self) -> None:
        sources = {
            "direct-windows": "#[cfg(windows)]\nfn platform() {}\n",
            "direct-unix": "#[cfg(unix)]\nfn platform() {}\n",
            "negated": "#[cfg(not(windows))]\nfn platform() {}\n",
            "target-os-windows": "#[cfg(target_os = \"windows\")]\nfn platform() {}\n",
            "cfg-macro": "const WINDOWS: bool = cfg!(windows);\n",
            "cfg-attr": "#[cfg_attr(unix, derive(Debug))]\nstruct Platform;\n",
            "compound": "#[cfg(any(windows, unix))]\nfn platform() {}\n",
            "nested-first": "#[cfg(all(not(test), windows))]\nfn platform() {}\n",
            "nested-macro": "const UNIX: bool = cfg!(all(not(test), unix));\n",
            "target-family": "#[cfg(target_family = \"unix\")]\nfn platform() {}\n",
        }
        portable = {
            "test-and-features": (
                "#[cfg(test)]\nmod tests {}\n"
                "#[cfg(all(test, not(feature = \"slow\")))]\nmod slow {}\n"
                "fn windows_path() {}\n"
            ),
            "feature-named-windows": "#[cfg(feature = \"windows\")]\nfn platform() {}\n",
            "quoted-example": (
                "const SAMPLE: &str = r#\"#[cfg(windows)]\"#;\n"
                "const TARGET: &str = \"cfg(unix)\";\n"
            ),
            "commented": "// #[cfg(windows)]\n/* cfg!(unix) */\nfn platform() {}\n",
            "cfg-attr-value": (
                "#![cfg_attr(not(debug_assertions), windows_subsystem = \"windows\")]\n"
            ),
        }

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            members = []
            for crate, source in {**sources, **portable}.items():
                crate_dir = root / "crates" / crate
                (crate_dir / "src").mkdir(parents=True)
                (crate_dir / "Cargo.toml").write_text(
                    f'[package]\nname = "{crate}"\nversion = "0.1.0"\n',
                    encoding="utf-8",
                )
                (crate_dir / "src" / "lib.rs").write_text(source, encoding="utf-8")
                members.append(f'"crates/{crate}"')
            (root / "Cargo.toml").write_text(
                f"[workspace]\nmembers = [{', '.join(members)}]\n", encoding="utf-8"
            )

            self.assertEqual(set(sources), _cfg_divergent_crates(root))

    def test_windows_unverified_list_has_no_stale_entries(self) -> None:
        # An entry must still name a workspace crate that carries divergent
        # code and that no Windows job compiles yet.
        self.assertEqual(set(), WINDOWS_UNVERIFIED_CRATES - set(_workspace_crates(ROOT)))
        self.assertEqual(set(), WINDOWS_UNVERIFIED_CRATES - _cfg_divergent_crates(ROOT))
        self.assertEqual(set(), WINDOWS_UNVERIFIED_CRATES & _windows_routed_crates())


if __name__ == "__main__":
    unittest.main()
