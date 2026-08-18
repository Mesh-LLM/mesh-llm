from __future__ import annotations

from pathlib import Path
import re
import subprocess
from typing import Final
import unittest


ROOT: Final = Path(__file__).resolve().parents[2]
JUSTFILE: Final = ROOT / "Justfile"


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
            'MESH_LLM_CUDA_TOOLKIT_MAJOR="'
            '${MESH_LLM_CUDA_TOOLKIT_MAJOR:-${cuda_version%%.*}}"',
            self.recipe("release-build-aarch64-cuda"),
        )

    def test_cuda12_release_recipes_include_pascal_sm61(self) -> None:
        contents = JUSTFILE.read_text(encoding="utf-8")

        self.assertIn(
            "else echo '61;75;80;86;87;89;90'",
            self.recipe("release-build-aarch64-cuda"),
        )
        self.assertIn(
            "if [[ \"$cuda_version\" == 13.* ]]; then echo '75;80;86;87;89;90;110'",
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
