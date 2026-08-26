from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import subprocess
import tempfile
from typing import Final
import unittest


ROOT: Final = Path(__file__).resolve().parents[2]
ACTION: Final = ROOT / ".github/actions/compute-changes/action.yml"


@dataclass(frozen=True, slots=True)
class RevisionDiff:
    repository: Path
    base: str
    head: str
    changed_files: str


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
    action = ACTION.read_text(encoding="utf-8")
    start = action.index("        justfile_backend_recipe_lines() {")
    end = action.index("        # Backend/platform lanes rebuild", start)
    script = action[start:end]
    script = script.replace("        ", "", 1)
    script = script.replace("${{ inputs.event_name }}", "push")
    script = script.replace("${{ inputs.base_sha }}", diff.base)
    script = script.replace("${{ inputs.head_sha }}", diff.head)
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
