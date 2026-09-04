from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import subprocess
import tempfile
from typing import Final
import unittest


ROOT: Final = Path(__file__).resolve().parents[2]
DERIVE_SCRIPT: Final = ROOT / ".github/actions/compute-changes/derive-outputs.sh"


@dataclass(frozen=True, slots=True)
class RevisionDiff:
    repository: Path
    base: str
    head: str
    changed_files: str
    event_name: str = "push"


def commit(repository: Path, message: str) -> str:
    subprocess.run(["git", "add", "-A"], cwd=repository, check=True)
    subprocess.run(
        [
            "git", "-c", "user.name=Justfile Test", "-c",
            "user.email=justfile-test@example.invalid", "commit", "-q", "-m", message,
        ],
        cwd=repository,
        check=True,
    )
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repository, text=True).strip()


def classify(diff: RevisionDiff) -> bool:
    action = DERIVE_SCRIPT.read_text(encoding="utf-8")
    start = action.index("JUSTFILE_RECIPE_AWK='")
    end = action.index("# Backend/platform lanes rebuild", start)
    script = action[start:end]
    script = script.replace("$EVENT_NAME", diff.event_name)
    script = script.replace("$BASE_SHA", diff.base)
    script = script.replace("$HEAD_SHA", diff.head)
    script = f"CHANGED_FILES={diff.changed_files!r}\n{script}\nprintf '%s\\n' \"$BACKEND_RECIPE_CHANGED\"\n"
    result = subprocess.run(
        ["bash", "-c", script],
        cwd=diff.repository,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.splitlines()[-1] == "true"


class ComputeChangesJustfileTests(unittest.TestCase):
    def test_top_level_lines_after_backend_recipe_are_not_backend_relevant(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            repository = Path(directory)
            subprocess.run(["git", "init", "-q"], cwd=repository, check=True)
            justfile = repository / "Justfile"
            justfile.write_text(
                "bundle:\n"
                "    printf backend\n\n"
                'light_setting := "base"\n\n'
                "website-build:\n"
                "    printf light\n",
                encoding="utf-8",
            )
            base = commit(repository, "base")

            justfile.write_text(
                justfile.read_text(encoding="utf-8").replace(
                    'light_setting := "base"', 'light_setting := "changed"'
                ),
                encoding="utf-8",
            )
            head = commit(repository, "light top-level assignment")

            self.assertFalse(classify(RevisionDiff(repository, base, head, "Justfile")))

    def test_recipe_free_sources_fail_open(self) -> None:
        sources = (
            'set shell := ["bash", "-uc"]\n',
            'export mesh_bin := "target/release/mesh-llm"\n',
        )
        for source in sources:
            with self.subTest(source=source), tempfile.TemporaryDirectory() as directory:
                repository = Path(directory)
                subprocess.run(["git", "init", "-q"], cwd=repository, check=True)
                (repository / "Justfile").write_text("build:\n    true\n", encoding="utf-8")
                base = commit(repository, "base")

                recipes = repository / "just"
                recipes.mkdir()
                (recipes / "settings.just").write_text(source, encoding="utf-8")
                head = commit(repository, "recipe-free source")

                self.assertTrue(
                    classify(RevisionDiff(repository, base, head, "just/settings.just"))
                )

    def test_backend_recipe_attribute_changes_are_backend_relevant(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            repository = Path(directory)
            subprocess.run(["git", "init", "-q"], cwd=repository, check=True)
            justfile = repository / "Justfile"
            justfile.write_text("[unix]\nbundle:\n    printf backend\n", encoding="utf-8")
            base = commit(repository, "base")
            justfile.write_text("[windows]\nbundle:\n    printf backend\n", encoding="utf-8")
            head = commit(repository, "backend recipe attribute")

            self.assertTrue(classify(RevisionDiff(repository, base, head, "Justfile")))

    def test_pull_request_old_side_uses_merge_base(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            repository = Path(directory)
            subprocess.run(["git", "init", "-q"], cwd=repository, check=True)
            justfile = repository / "Justfile"
            justfile.write_text(
                "light:\n    printf light\n\nbundle:\n    printf backend\n",
                encoding="utf-8",
            )
            merge_base = commit(repository, "merge base")

            subprocess.run(["git", "switch", "-q", "-c", "feature"], cwd=repository, check=True)
            justfile.write_text("light:\n    printf light\n", encoding="utf-8")
            head = commit(repository, "remove backend recipe")

            subprocess.run(
                ["git", "switch", "-q", "-c", "base", merge_base],
                cwd=repository,
                check=True,
            )
            justfile.write_text(
                "setting := \"base\"\n\nlight:\n    printf light\n\nbundle:\n    printf backend\n",
                encoding="utf-8",
            )
            base = commit(repository, "advance base")

            self.assertTrue(
                classify(
                    RevisionDiff(
                        repository,
                        base,
                        head,
                        "Justfile",
                        event_name="pull_request",
                    )
                )
            )

    def test_standalone_quantize_recipe_changes_are_backend_relevant(self) -> None:
        recipes = (
            "skippy-quantize-standalone-build",
            "skippy-quantize-standalone-release-build",
        )
        for recipe in recipes:
            with self.subTest(recipe=recipe), tempfile.TemporaryDirectory() as directory:
                repository = Path(directory)
                subprocess.run(["git", "init", "-q"], cwd=repository, check=True)
                justfile = repository / "Justfile"
                justfile.write_text(f"import 'just/skippy.just'\n", encoding="utf-8")
                skippy = repository / "just"
                skippy.mkdir()
                source = f'{recipe} backend="cpu":\n    printf base\n'
                skippy_file = skippy / "skippy.just"
                skippy_file.write_text(source, encoding="utf-8")
                base = commit(repository, "base")
                skippy_file.write_text(source.replace("printf base", "printf changed"), encoding="utf-8")
                head = commit(repository, "backend recipe")

                self.assertTrue(
                    classify(RevisionDiff(repository, base, head, "just/skippy.just"))
                )

    def test_top_level_assignments_are_classified_by_backend_recipe_use(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            repository = Path(directory)
            subprocess.run(["git", "init", "-q"], cwd=repository, check=True)
            recipes = repository / "just"
            recipes.mkdir()
            (repository / "Justfile").write_text(
                "website_dir := \"website\"\n\ndefault: build\n\n"
                "import 'just/mesh.just'\n\nimport 'just/website-ui.just'\n",
                encoding="utf-8",
            )
            mesh = recipes / "mesh.just"
            mesh.write_text(
                'mesh_bin := env("MESH_LLM_BIN", "target/release/mesh-llm")\n\n'
                "bundle:\n"
                '    "{{ mesh_bin }}" --version\n',
                encoding="utf-8",
            )
            (recipes / "website-ui.just").write_text(
                'website-build:\n    printf "{{ website_dir }}"\n', encoding="utf-8"
            )
            base = commit(repository, "base")

            mesh.write_text(
                mesh.read_text(encoding="utf-8").replace(
                    "target/release/mesh-llm", "target/debug/mesh-llm"
                ),
                encoding="utf-8",
            )
            backend_input_head = commit(repository, "backend input assignment")
            self.assertTrue(
                classify(RevisionDiff(repository, base, backend_input_head, "just/mesh.just"))
            )

            root_justfile = repository / "Justfile"
            root_justfile.write_text(
                root_justfile.read_text(encoding="utf-8").replace(
                    'website_dir := "website"', 'website_dir := "site"'
                ),
                encoding="utf-8",
            )
            light_input_head = commit(repository, "light input assignment")
            self.assertFalse(
                classify(
                    RevisionDiff(repository, backend_input_head, light_input_head, "Justfile")
                )
            )

    def test_exported_assignments_are_classified_by_backend_recipe_use(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            repository = Path(directory)
            subprocess.run(["git", "init", "-q"], cwd=repository, check=True)
            justfile = repository / "Justfile"
            justfile.write_text(
                'export mesh_bin := "target/release/mesh-llm"\n\n'
                "bundle:\n"
                '    "{{ mesh_bin }}" --version\n',
                encoding="utf-8",
            )
            base = commit(repository, "base")

            justfile.write_text(
                justfile.read_text(encoding="utf-8").replace(
                    "target/release/mesh-llm", "target/debug/mesh-llm"
                ),
                encoding="utf-8",
            )
            head = commit(repository, "exported backend input assignment")

            self.assertTrue(classify(RevisionDiff(repository, base, head, "Justfile")))

    def test_recipe_sources_cover_light_backend_added_deleted_and_root_changes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            repository = Path(directory)
            subprocess.run(["git", "init", "-q"], cwd=repository, check=True)
            recipes = repository / "just"
            recipes.mkdir()
            (repository / "Justfile").write_text(
                "default: build\n\nimport 'just/build.just'\n\n"
                "import 'just/website-ui.just'\n",
                encoding="utf-8",
            )
            (recipes / "build.just").write_text("build:\n    true\n", encoding="utf-8")
            website = recipes / "website-ui.just"
            website.write_text("website-build:\n    true\n", encoding="utf-8")
            base = commit(repository, "base")

            website.write_text("website-build:\n    printf light\n", encoding="utf-8")
            light_head = commit(repository, "light recipe")
            self.assertFalse(classify(RevisionDiff(repository, base, light_head, "just/website-ui.just")))

            build = recipes / "build.just"
            build.write_text("build:\n    printf backend\n", encoding="utf-8")
            backend_head = commit(repository, "backend recipe")
            self.assertTrue(classify(RevisionDiff(repository, light_head, backend_head, "just/build.just")))

            nested = recipes / "nested"
            nested.mkdir()
            nested_build = nested / "runtime.just"
            nested_build.write_text("build-runtime:\n    printf backend\n", encoding="utf-8")
            nested_base = commit(repository, "nested recipe")
            nested_build.write_text("build-runtime:\n    printf changed\n", encoding="utf-8")
            nested_head = commit(repository, "nested backend recipe")
            self.assertTrue(
                classify(
                    RevisionDiff(
                        repository,
                        nested_base,
                        nested_head,
                        "just/nested/runtime.just",
                    )
                )
            )

            website.write_text(
                "import 'nested/runtime.just'\n\nwebsite-build:\n    true\n",
                encoding="utf-8",
            )
            import_head = commit(repository, "nested import")
            self.assertTrue(
                classify(
                    RevisionDiff(
                        repository,
                        nested_head,
                        import_head,
                        "just/website-ui.just",
                    )
                )
            )

            added = recipes / "release-extra.just"
            added.write_text("release-build-extra:\n    true\n", encoding="utf-8")
            added_head = commit(repository, "added recipe source")
            self.assertTrue(classify(RevisionDiff(repository, backend_head, added_head, "just/release-extra.just")))

            added.unlink()
            deleted_head = commit(repository, "deleted recipe source")
            self.assertTrue(classify(RevisionDiff(repository, added_head, deleted_head, "just/release-extra.just")))

            root_justfile = repository / "Justfile"
            root_justfile.write_text(
                root_justfile.read_text(encoding="utf-8") + "\nimport 'just/extra.just'\n",
                encoding="utf-8",
            )
            import_head = commit(repository, "root import graph")
            self.assertTrue(classify(RevisionDiff(repository, deleted_head, import_head, "Justfile")))

            (recipes / "notes.just").write_text("# no recipes\n", encoding="utf-8")
            invalid_head = commit(repository, "unclassifiable source")
            self.assertTrue(classify(RevisionDiff(repository, import_head, invalid_head, "just/notes.just")))


if __name__ == "__main__":
    unittest.main()
