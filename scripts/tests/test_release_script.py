from __future__ import annotations

from pathlib import Path
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
RELEASE_SCRIPT = ROOT / "scripts" / "release.sh"


class ReleaseScriptTests(unittest.TestCase):
    def run_bash(self, command: str, *arguments: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["bash", "-c", command, "release-script-test", *arguments],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )

    def test_workflow_run_id_is_extracted_from_dispatch_url(self) -> None:
        result = self.run_bash(
            'source "$1"; workflow_run_id_from_url "$2"',
            str(RELEASE_SCRIPT),
            "https://github.com/Mesh-LLM/mesh-llm/actions/runs/123456789",
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout, "123456789\n")

    def test_unrecognized_dispatch_output_has_no_run_id(self) -> None:
        result = self.run_bash(
            'source "$1"; workflow_run_id_from_url "$2"',
            str(RELEASE_SCRIPT),
            "workflow dispatched without a URL",
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout, "")

    def test_sha_correlated_lookup_is_the_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            arguments_file = Path(temporary_directory) / "gh-arguments"
            result = self.run_bash(
                """
source "$1"
gh_arguments_file="$2"
gh() {
    printf '%s\n' "$*" > "$gh_arguments_file"
    printf '987654321\n'
}
sleep() { :; }
find_dispatched_release_run_id "$3" "$4"
""",
                str(RELEASE_SCRIPT),
                str(arguments_file),
                "abc123",
                "2026-08-14T12:00:00Z",
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(result.stdout, "987654321\n")
            arguments = arguments_file.read_text(encoding="utf-8")
            self.assertIn("run list", arguments)
            self.assertIn("--branch main", arguments)
            self.assertIn("--commit abc123", arguments)
            self.assertIn("--event workflow_dispatch", arguments)
            self.assertIn("--created >=2026-08-14T12:00:00Z", arguments)

    def test_main_keeps_run_url_and_numeric_id_separate(self) -> None:
        script = RELEASE_SCRIPT.read_text(encoding="utf-8")
        url_capture = script.index('run_url="$(dispatch_release_workflow')
        id_extract = script.index('run_id="$(workflow_run_id_from_url')
        fallback = script.index('if [[ -z "$run_id" ]]', id_extract)
        watch = script.index('gh run watch "$run_id"', fallback)

        self.assertLess(url_capture, id_extract)
        self.assertLess(id_extract, fallback)
        self.assertLess(fallback, watch)
        self.assertIn('dispatch_sha="$(git rev-parse origin/main)"', script)


if __name__ == "__main__":
    unittest.main()
