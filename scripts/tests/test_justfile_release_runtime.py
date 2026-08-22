from __future__ import annotations

import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import tempfile
from typing import Final
import unittest


ROOT: Final = Path(__file__).resolve().parents[2]
JUSTFILE: Final = ROOT / "Justfile"


def _recipe_body_lines(recipe: str) -> list[str]:
    """Strip a recipe down to the shell script just would actually execute.

    Drops the `name: deps` header, removes the recipe's indentation, and strips
    just's line prefixes. `@` (suppress echo) and `-` (ignore failure) are
    directives to just, not shell syntax — leaving them in makes the first line
    parse as a command name, so every variable it was supposed to set silently
    stays unset.
    """
    lines = recipe.splitlines()[1:]
    lines = [line[4:] if line.startswith("    ") else line for line in lines]
    lines = [line for line in lines if not line.startswith("#!")]
    for index, line in enumerate(lines):
        if line.strip():
            lines[index] = line.lstrip("@-")
            break
    return lines


def _run_cuda_arch_selection(recipe: str, mesh_cuda_version: str) -> str:
    """Execute the real arch-selection lines out of a `release-build-cuda`-shaped
    recipe body (up to, but not including, the package-native-runtime.sh call),
    with MESH_CUDA_VERSION forced, and return the selected `arches` list.

    This runs the actual recipe source rather than a hand-copied duplicate, so
    it can't silently drift from what `just` executes.
    """
    lines = recipe.splitlines()[1:]  # drop the "recipe-name: deps" header line
    lines = [line[4:] if line.startswith("    ") else line for line in lines]
    lines = [line for line in lines if not line.startswith("#!/usr/bin/env bash")]
    body: list[str] = []
    for line in lines:
        if line.strip().startswith("MESH_LLM_CUDA_TOOLKIT_MAJOR="):
            break
        body.append(line)
    script = "\n".join(body) + '\necho "$arches"\n'
    result = subprocess.run(
        ["bash", "-c", script],
        check=True,
        capture_output=True,
        text=True,
        env={"PATH": "/usr/bin:/bin", "MESH_CUDA_VERSION": mesh_cuda_version},
    )
    return result.stdout.strip()


def _run_recipe_body_capturing_packager_env(
    recipe: str, mesh_cuda_version: str, interpreter: str
) -> dict[str, str]:
    """Run a whole release-build recipe body with `package-native-runtime.sh`
    stubbed out, and return the environment the stub actually observed.

    This executes the recipe source as-is — including the trailing
    `VAR=... scripts/package-native-runtime.sh` line — so it reports what the
    packager would really be handed, rather than re-deriving the selection.
    `interpreter` is the shell just would use for this recipe: `bash` for a
    `#!/usr/bin/env bash` script recipe, `sh`/`dash` for a plain one.
    """
    lines = _recipe_body_lines(recipe)

    with tempfile.TemporaryDirectory() as workdir:
        scripts_dir = Path(workdir) / "scripts"
        scripts_dir.mkdir()
        probe = Path(workdir) / "packager-env.txt"
        stub = scripts_dir / "package-native-runtime.sh"
        stub.write_text(
            "#!/usr/bin/env bash\n"
            f'printf "%s\\n" "$LLAMA_STAGE_CUDA_ARCHITECTURES" '
            f'"$MESH_LLM_CUDA_TOOLKIT_MAJOR" "$*" > {shlex.quote(str(probe))}\n',
            encoding="utf-8",
        )
        stub.chmod(0o755)
        detect = scripts_dir / "detect-cuda-toolkit-version.sh"
        detect.write_text("#!/usr/bin/env bash\necho 12\n", encoding="utf-8")
        detect.chmod(0o755)

        script = Path(workdir) / "recipe-body"
        script.write_text("\n".join(lines) + "\n", encoding="utf-8")
        subprocess.run(
            [interpreter, str(script)],
            cwd=workdir,
            check=False,
            capture_output=True,
            text=True,
            env={"PATH": os.environ["PATH"], "MESH_CUDA_VERSION": mesh_cuda_version},
        )
        if not probe.exists():
            return {"arches": "", "toolkit_major": "", "args": ""}
        recorded = probe.read_text(encoding="utf-8").splitlines()
        recorded += [""] * (3 - len(recorded))
        return {
            "arches": recorded[0],
            "toolkit_major": recorded[1],
            "args": recorded[2],
        }


def _run_build_runtime_body(recipe: str, interpreter: str, backend: str) -> dict[str, str]:
    """Run `build-runtime`'s body with `{{ backend }}` substituted the way just
    would, and report the backend the packager was actually handed.
    """
    body = "\n".join(_recipe_body_lines(recipe))
    for placeholder, value in (
        ("{{ backend }}", backend),
        ("{{ cuda_arch }}", ""),
        ("{{ rocm_arch }}", ""),
    ):
        body = body.replace(placeholder, value)

    with tempfile.TemporaryDirectory() as workdir:
        scripts_dir = Path(workdir) / "scripts"
        scripts_dir.mkdir()
        probe = Path(workdir) / "packager-env.txt"
        stub = scripts_dir / "package-native-runtime.sh"
        stub.write_text(
            "#!/usr/bin/env bash\n"
            f'printf "%s\\n" "$*" > {shlex.quote(str(probe))}\n',
            encoding="utf-8",
        )
        stub.chmod(0o755)
        script = Path(workdir) / "recipe-body"
        script.write_text(body + "\n", encoding="utf-8")
        subprocess.run(
            [interpreter, str(script)],
            cwd=workdir,
            check=False,
            capture_output=True,
            text=True,
            env={"PATH": os.environ["PATH"]},
        )
        args = probe.read_text(encoding="utf-8").strip() if probe.exists() else ""

    match = re.search(r"--backend (\S+)", args)
    return {"args": args, "backend": match.group(1) if match else ""}


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
                'cuda_version="${MESH_CUDA_VERSION:-'
                '$(scripts/detect-cuda-toolkit-version.sh)}"',
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
        contents = JUSTFILE.read_text(encoding="utf-8")

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
        recipe = self.recipe("release-build-cuda")
        pre_blackwell = "61;75;80;86;87;89;90"
        blackwell = "75;80;86;87;89;90;100;103;120;121"

        cases = {
            "12": pre_blackwell,  # detect script's own static fallback
            "12.0": pre_blackwell,
            "12.7": pre_blackwell,
            "12.8": blackwell,  # first toolkit release with Blackwell support
            "12.9": blackwell,
            "13": blackwell,
            "13.3": blackwell,
        }
        for mesh_cuda_version, expected in cases.items():
            with self.subTest(mesh_cuda_version=mesh_cuda_version):
                self.assertEqual(
                    _run_cuda_arch_selection(recipe, mesh_cuda_version), expected
                )

    def test_aarch64_cuda_release_build_selects_arches_at_the_13_boundary(self) -> None:
        """Run the recipe body the way just would and check what the packager gets.

        `sm_110` (Thor) needs toolkit major >= 13, so this mirrors how the
        x86_64 sibling gates Blackwell on >= 12.8. The interpreter is taken
        from the recipe itself: a plain recipe is run under `dash`, which is
        what `/bin/sh` is on the Debian/Ubuntu aarch64 build hosts.
        """
        recipe = self.recipe("release-build-aarch64-cuda")
        interpreter = self.recipe_interpreter(recipe)
        pre_13 = "61;75;80;86;87;89;90"
        thor = "75;80;86;87;89;90;110"

        cases = {
            "12": pre_13,  # detect script's own static fallback
            "12.4": pre_13,
            "12.8": pre_13,  # Blackwell gate is x86-only; aarch64 gates on 13
            "13": thor,
            "13.1.2": thor,
            "14": thor,  # a `13.*` glob would wrongly fall back here
        }
        for mesh_cuda_version, expected in cases.items():
            with self.subTest(mesh_cuda_version=mesh_cuda_version):
                observed = _run_recipe_body_capturing_packager_env(
                    recipe, mesh_cuda_version, interpreter
                )
                self.assertEqual(observed["arches"], expected)

    def test_aarch64_cuda_release_build_propagates_major_and_target_to_the_packager(
        self,
    ) -> None:
        recipe = self.recipe("release-build-aarch64-cuda")
        observed = _run_recipe_body_capturing_packager_env(
            recipe, "13.1.2", self.recipe_interpreter(recipe)
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
        "<pid>backend" — the default-to-cpu test never inspected the variable
        it appeared to, and the packager was handed a nonsense backend name.
        """
        recipe = self.recipe("build-runtime")
        interpreter = self.recipe_interpreter(recipe)

        defaulted = _run_build_runtime_body(recipe, interpreter, backend="")
        self.assertEqual(defaulted["backend"], "cpu")

        explicit = _run_build_runtime_body(recipe, interpreter, backend="cuda")
        self.assertEqual(explicit["backend"], "cuda")

    def recipe_interpreter(self, recipe: str) -> str:
        """The shell just would run this recipe body with.

        A `#!` script recipe is executed with the interpreter it names; a plain
        recipe goes to just's default shell, `sh`, which is dash on the Debian
        and Ubuntu hosts these release recipes target.
        """
        body = [line for line in recipe.splitlines()[1:] if line.strip()]
        if body and body[0].strip().startswith("#!"):
            self.assertEqual(body[0].strip(), "#!/usr/bin/env bash")
            return "bash"
        if not shutil.which("dash") and not Path("/bin/dash").exists():
            self.skipTest("dash is required to exercise just's default `sh` faithfully")
        return "dash" if shutil.which("dash") else "/bin/dash"

    def test_bundle_uses_the_product_packager_and_copies_its_checksum(self) -> None:
        recipe = self.recipe("bundle")

        self.assertIn('bundle output="/tmp/mesh-llm-bundle.tar.gz": release-build', recipe)
        self.assertIn('scripts/package-release.sh "$version" "$staging_dir"', recipe)
        self.assertIn('cp "$stable_archive" "{{ output }}"', recipe)
        self.assertIn('cp "$stable_archive.sha256" "{{ output }}.sha256"', recipe)
        self.assertNotIn('cp "{{ mesh_bin }}"', recipe)

    def release_runtime_recipe(self) -> str:
        contents = JUSTFILE.read_text(encoding="utf-8")
        start = contents.index('release-runtime-build backend="" target="":')
        end = contents.index("# Build the backend-neutral host and the default runtime", start)
        return contents[start:end]

    def recipe(self, name: str) -> str:
        contents = JUSTFILE.read_text(encoding="utf-8")
        match = re.search(rf"(?m)^{re.escape(name)}(?=[: ])", contents)
        self.assertIsNotNone(match)
        assert match is not None
        start = match.start()
        next_recipe = contents.find("\n\n", start)
        return contents[start:] if next_recipe == -1 else contents[start:next_recipe]


if __name__ == "__main__":
    unittest.main()
