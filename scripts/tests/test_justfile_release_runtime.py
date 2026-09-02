from __future__ import annotations

import os
from pathlib import Path
import re
import shlex
import subprocess
import tempfile
from typing import Final
import unittest

from scripts.tests.justfile_source import read_justfile_source


ROOT: Final = Path(__file__).resolve().parents[2]
JUSTFILE: Final = ROOT / "Justfile"


def _recipe_dependencies(header_line: str) -> list[str]:
    """The just-recipe names a recipe header declares as dependencies.

    e.g. "release-build-aarch64-cuda: release-host-build" -> ["release-host-build"].
    Splits on the *last* colon, since a parameterized header's default value
    (e.g. 'bundle output="/tmp/x:y": release-build') can itself contain one;
    the header's own closing colon is always the final one. Parameterized
    headers with no dependencies, like 'build-runtime backend="" cuda_arch="":',
    correctly yield an empty list either way.
    """
    _, _, deps_part = header_line.rpartition(":")
    return deps_part.split()


def _run_release_recipe(
    name: str, recipe_text: str, *args: str, env: dict[str, str] | None = None
) -> dict[str, str]:
    """Run a real Justfile release-build recipe through `just`, with the
    packager and CUDA-toolkit-detection scripts replaced by stubs that record
    what they were called with.

    The recipe is written verbatim into an isolated temporary Justfile (any
    just-recipe dependency it declares is stubbed out alongside it) and
    invoked with the real `just` binary, so interpolation, `@`/`-`
    directives, and script-recipe shebang handling all come from `just`
    itself rather than a hand-rolled reimplementation.
    """
    header = recipe_text.splitlines()[0]
    deps = _recipe_dependencies(header)

    with tempfile.TemporaryDirectory() as workdir_str:
        workdir = Path(workdir_str)
        scripts_dir = workdir / "scripts"
        scripts_dir.mkdir()

        probe = workdir / "packager-env.txt"
        packager_stub = scripts_dir / "package-native-runtime.sh"
        packager_stub.write_text(
            "#!/usr/bin/env bash\n"
            f'printf "%s\\n" "$LLAMA_STAGE_CUDA_ARCHITECTURES" '
            f'"$MESH_LLM_CUDA_TOOLKIT_MAJOR" "$*" > {shlex.quote(str(probe))}\n',
            encoding="utf-8",
        )
        packager_stub.chmod(0o755)
        detect_stub = scripts_dir / "detect-cuda-toolkit-version.sh"
        # A value no test ever passes as an explicit MESH_CUDA_VERSION, so a
        # case that omits the env var can only get this from the fallback
        # actually running the detect script, not from a coincidental match.
        detect_stub.write_text(
            "#!/usr/bin/env bash\n"
            'printf "%s\\n" "${MESH_CUDA_VERSION:-11}"\n',
            encoding="utf-8",
        )
        detect_stub.chmod(0o755)

        justfile = workdir / "Justfile"
        stub_recipes = "".join(f"{dep}:\n    @true\n\n" for dep in deps)
        justfile.write_text(stub_recipes + recipe_text + "\n", encoding="utf-8")

        run_env = {"PATH": os.environ["PATH"]}
        run_env.update(env or {})
        result = subprocess.run(
            ["just", "-f", str(justfile), name, *args],
            cwd=workdir,
            env=run_env,
            check=False,
            capture_output=True,
            text=True,
            timeout=60,
        )

        if not probe.exists():
            raise AssertionError(
                f"`just {name}` did not reach the packager stub "
                f"(exit {result.returncode})\nstdout: {result.stdout}\nstderr: {result.stderr}"
            )
        recorded = probe.read_text(encoding="utf-8").splitlines()
        recorded += [""] * (3 - len(recorded))
        return {"arches": recorded[0], "toolkit_major": recorded[1], "args": recorded[2]}


class JustfileReleaseRuntimeTests(unittest.TestCase):
    def test_release_runtime_build_does_not_expand_empty_array_under_nounset(self) -> None:
        recipe = self.release_runtime_recipe()

        self.assertNotIn("target_args=()", recipe)
        self.assertNotIn('"${target_args[@]}"', recipe)
        self.assertIn(
            'scripts/package-native-runtime.sh --build --backend "$selected_backend" '
            '--target "{{ target }}"',
            recipe,
        )
        self.assertIn(
            'scripts/package-native-runtime.sh --build --backend "$selected_backend"',
            recipe,
        )

    def test_cuda_release_recipes_propagate_the_selected_toolkit_major(self) -> None:
        for recipe_name in ("release-build-cuda", "release-build-aarch64-cuda"):
            recipe = self.recipe(recipe_name)
            self.assertIn(
                'cuda_version="$(scripts/detect-cuda-toolkit-version.sh)"',
                recipe,
            )

        self.assertIn(
            'MESH_LLM_CUDA_TOOLKIT_MAJOR="${MESH_LLM_CUDA_TOOLKIT_MAJOR:-$major}"',
            self.recipe("release-build-cuda"),
        )
        self.assertIn(
            'MESH_LLM_CUDA_TOOLKIT_MAJOR="${MESH_LLM_CUDA_TOOLKIT_MAJOR:-$major}"',
            self.recipe("release-build-aarch64-cuda"),
        )

    def test_cuda12_release_recipes_include_pascal_sm61(self) -> None:
        contents = read_justfile_source(JUSTFILE)

        self.assertIn(
            "arches='61;75;80;86;87;89;90'",
            self.recipe("release-build-aarch64-cuda"),
        )
        self.assertIn(
            "arches='75;80;86;87;89;90;110'",
            self.recipe("release-build-aarch64-cuda"),
        )
        self.assertIn(
            'release-build-cuda-windows cuda_arch="61;75;80;86;87;89;90"',
            contents,
        )

    def test_cuda_release_build_selects_arches_at_the_12_8_boundary(self) -> None:
        """Run the real recipe through `just`, with MESH_CUDA_VERSION forced,
        and check what the packager stub actually received.

        Blackwell (sm_100/103/120/121) needs toolkit >= 12.8, the first
        release that shipped support for it -- gate on that boundary, not on
        the CUDA major alone.
        """
        recipe = self.recipe("release-build-cuda")
        pre_blackwell = "61;75;80;86;87;89;90"
        blackwell = "75;80;86;87;89;90;100;103;120;121"

        cases = {
            "12": pre_blackwell,
            "12.0": pre_blackwell,
            "12.7": pre_blackwell,
            "12.8": blackwell,  # first toolkit release with Blackwell support
            "12.9": blackwell,
            "13": blackwell,
            "13.3": blackwell,
        }
        for mesh_cuda_version, expected in cases.items():
            with self.subTest(mesh_cuda_version=mesh_cuda_version):
                observed = _run_release_recipe(
                    "release-build-cuda",
                    recipe,
                    env={"MESH_CUDA_VERSION": mesh_cuda_version},
                )
                self.assertEqual(observed["arches"], expected)

        with self.subTest(mesh_cuda_version="<unset>"):
            # No MESH_CUDA_VERSION at all -- the detector's local auto-detection
            # path supplies its sentinel version. Assert on toolkit_major too:
            # every explicit case below the 12.8 gate also yields
            # `pre_blackwell`, so arches alone cannot prove detection ran.
            observed = _run_release_recipe("release-build-cuda", recipe)
            self.assertEqual(observed["toolkit_major"], "11")
            self.assertEqual(observed["arches"], pre_blackwell)

    def test_aarch64_cuda_release_build_selects_arches_at_the_13_boundary(self) -> None:
        """Run the real recipe through `just`, exactly as the release build
        does -- the fixed recipe is a `#!/usr/bin/env bash` script recipe, so
        `just` executes it with bash regardless of what `/bin/sh` is on the
        host.

        `sm_110` (Thor) needs toolkit major >= 13, mirroring how the x86_64
        sibling gates Blackwell on >= 12.8.
        """
        recipe = self.recipe("release-build-aarch64-cuda")
        pre_13 = "61;75;80;86;87;89;90"
        thor = "75;80;86;87;89;90;110"

        cases = {
            "12": pre_13,
            "12.4": pre_13,
            "12.8": pre_13,  # Blackwell gate is x86-only; aarch64 gates on 13
            "13": thor,
            "13.1.2": thor,
            "14": thor,  # a `13.*` glob would wrongly fall back here
        }
        for mesh_cuda_version, expected in cases.items():
            with self.subTest(mesh_cuda_version=mesh_cuda_version):
                observed = _run_release_recipe(
                    "release-build-aarch64-cuda",
                    recipe,
                    env={"MESH_CUDA_VERSION": mesh_cuda_version},
                )
                self.assertEqual(observed["arches"], expected)

        with self.subTest(mesh_cuda_version="<unset>"):
            # No MESH_CUDA_VERSION at all -- the detector's local auto-detection
            # path supplies its sentinel version. Assert on toolkit_major too:
            # every explicit case below the 13 gate also yields `pre_13`, so
            # arches alone cannot prove detection ran.
            observed = _run_release_recipe("release-build-aarch64-cuda", recipe)
            self.assertEqual(observed["toolkit_major"], "11")
            self.assertEqual(observed["arches"], pre_13)

    def test_aarch64_cuda_release_build_propagates_major_and_target_to_the_packager(
        self,
    ) -> None:
        recipe = self.recipe("release-build-aarch64-cuda")
        observed = _run_release_recipe(
            "release-build-aarch64-cuda", recipe, env={"MESH_CUDA_VERSION": "13.1.2"}
        )

        self.assertEqual(observed["toolkit_major"], "13")
        self.assertEqual(
            observed["args"],
            "--build --backend cuda --target aarch64-unknown-linux-gnu",
        )

    def test_build_runtime_defaults_the_backend_and_forwards_the_one_it_was_given(
        self,
    ) -> None:
        """`$$backend` read the shell PID, not the recipe argument.

        Under just, `$$` is two literal dollars, so `"$$backend"` expanded to
        "<pid>backend" -- the default-to-cpu test never inspected the variable
        it appeared to, and the packager was handed a nonsense backend name.
        """
        recipe = self.recipe("build-runtime")

        defaulted = _run_release_recipe("build-runtime", recipe)
        self.assertEqual(defaulted["args"], "--build --backend cpu")

        explicit = _run_release_recipe("build-runtime", recipe, "cuda")
        self.assertEqual(explicit["args"], "--build --backend cuda")

    def test_bundle_uses_the_product_packager_and_copies_its_checksum(self) -> None:
        recipe = self.recipe("bundle")

        self.assertIn('bundle output="/tmp/mesh-llm-bundle.tar.gz": release-build', recipe)
        self.assertIn('scripts/package-release.sh "$version" "$staging_dir"', recipe)
        self.assertIn('cp "$stable_archive" "{{ output }}"', recipe)
        self.assertIn('cp "$stable_archive.sha256" "{{ output }}.sha256"', recipe)
        self.assertNotIn('cp "{{ mesh_bin }}"', recipe)

    def release_runtime_recipe(self) -> str:
        contents = read_justfile_source(JUSTFILE)
        start = contents.index('release-runtime-build backend="" target="":')
        end = contents.index("# Build the backend-neutral host and the default runtime", start)
        return contents[start:end]

    def recipe(self, name: str) -> str:
        contents = read_justfile_source(JUSTFILE)
        match = re.search(rf"(?m)^{re.escape(name)}(?=[: ])", contents)
        self.assertIsNotNone(match)
        assert match is not None
        start = match.start()
        next_recipe = contents.find("\n\n", start)
        return contents[start:] if next_recipe == -1 else contents[start:next_recipe]


if __name__ == "__main__":
    unittest.main()
