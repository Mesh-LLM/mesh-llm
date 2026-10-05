"""Execute Docker precheck blocks against the relocated product sources."""
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = ROOT / ".github/workflows/docker-precheck.yml"


def run_block(name):
    text = WORKFLOW.read_text()
    step = text.split("      - name: " + name + "\n", 1)[1].split("\n      - name:", 1)[0]
    body = step.split("        run: |\n", 1)[1]
    return "\n".join(line[10:] for line in body.splitlines())


class DockerPrecheckLayoutTests(unittest.TestCase):
    def run_copy_check(self, content):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "Dockerfile").write_text(content)
            script = run_block("Explicit workspace copy pattern is preserved")
            script = script.replace("${{ inputs.dockerfile_path }}", "Dockerfile")
            return subprocess.run(["bash", "-c", script], cwd=root, capture_output=True).returncode

    def test_actual_product_dockerfile_passes(self):
        source = (ROOT / "mesh/deploy/docker/Dockerfile.client").read_text()
        self.assertEqual(self.run_copy_check(source), 0)

    def test_missing_product_tree_or_blanket_copy_fails(self):
        source = (ROOT / "mesh/deploy/docker/Dockerfile.client").read_text()
        for line in ["COPY mesh/crates/ mesh/crates/", "COPY skippy/crates/ skippy/crates/"]:
            with self.subTest(line=line):
                self.assertNotEqual(self.run_copy_check(source.replace(line, "")), 0)
        self.assertNotEqual(self.run_copy_check(source + "\nCOPY . .\n"), 0)

    def test_entrypoint_checks_relocated_file_and_rejects_api_mode(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            path = root / "mesh/deploy/docker/entrypoint.sh"
            path.parent.mkdir(parents=True)
            source = (ROOT / "mesh/deploy/docker/entrypoint.sh").read_text()
            path.write_text(source)
            script = run_block("Entrypoint supports console worker and default modes only")
            self.assertEqual(subprocess.run(["bash", "-c", script], cwd=root, capture_output=True).returncode, 0)
            path.write_text(source + "\napi)\n")
            self.assertNotEqual(subprocess.run(["bash", "-c", script], cwd=root, capture_output=True).returncode, 0)
