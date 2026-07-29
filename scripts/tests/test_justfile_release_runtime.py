from __future__ import annotations

from pathlib import Path
import re
from typing import Final
import unittest


ROOT: Final = Path(__file__).resolve().parents[2]
JUSTFILE: Final = ROOT / "Justfile"


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
        contents = JUSTFILE.read_text(encoding="utf-8")

        for recipe_name in ("release-build-cuda", "release-build-aarch64-cuda"):
            recipe = self.recipe(recipe_name)
            self.assertIn('cuda_version="${MESH_CUDA_VERSION:-12}"', recipe)
            self.assertIn(
                'MESH_LLM_CUDA_TOOLKIT_MAJOR="'
                '${MESH_LLM_CUDA_TOOLKIT_MAJOR:-${cuda_version%%.*}}"',
                recipe,
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
