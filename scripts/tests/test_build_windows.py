from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import unittest

from scripts.tests.justfile_source import read_justfile_source


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/build-windows.ps1"
RUNTIME = ROOT / ".github/workflows/ci-windows-runtime-slice.yml"
PRODUCT = ROOT / ".github/workflows/ci-windows-product-slice.yml"
HOST_INPUT = ROOT / ".github/actions/prepare-windows-host-input/action.yml"


class BuildWindowsScriptTests(unittest.TestCase):
    def test_windows_host_ui_fallback_requires_index_entry_point(self):
        action = HOST_INPUT.read_text()
        self.assertIn('$uiEntryPoint = Join-Path $uiDist "index.html"', action)
        self.assertIn(
            "Test-Path -LiteralPath $uiEntryPoint -PathType Leaf",
            action,
        )
        self.assertNotIn("Get-ChildItem -LiteralPath $uiDist", action)

    def test_windows_release_recipes_request_dynamic_hosts(self):
        justfile = read_justfile_source(ROOT / "Justfile")
        for recipe in (
            "release-build-windows:",
            "release-build-cuda-windows",
            "release-build-rocm-windows",
            "release-build-vulkan-windows:",
        ):
            start = justfile.index(recipe)
            end = justfile.find("\n\n", start)
            self.assertIn("-DynamicHost", justfile[start:end])

    def test_windows_script_keeps_dynamic_host_and_runtime_separation(self):
        script = SCRIPT.read_text()
        self.assertIn('"-DBUILD_SHARED_LIBS=ON"', script)
        self.assertIn('dynamic-native-runtime', script)
        self.assertNotIn("[switch]$AbiOnly", script)

    def test_windows_runtime_slice_uses_verified_cache_and_composer(self):
        runtime = RUNTIME.read_text()
        product = PRODUCT.read_text()
        self.assertIn("restore-windows-abi-cache", runtime)
        self.assertIn("compose-product-input", product)
        self.assertIn("binary_name: mesh-llm.exe", product)
        self.assertIn('readiness_smoke: "true"', product)
        self.assertNotIn("build-windows.ps1", product)

    def test_windows_prepares_llama_with_the_shared_patch_queue(self):
        script = SCRIPT.read_text()
        # One patch queue for every platform: the main series, model_support
        # and generated patches all come from scripts/prepare-llama.sh, which
        # also writes the .mesh-llm-patched-sha stamp the build dir keys on.
        self.assertIn('Invoke-NativeCommand "bash" @("scripts/prepare-llama.sh", $mode)', script)
        self.assertNotIn('"am",', script)
        self.assertNotIn("Get-ChildItem -Path $patchDir", script)

    def test_stage_build_dir_is_resolved_after_prepare_and_keyed_by_the_stamp(self):
        script = SCRIPT.read_text()
        self.assertIn('Join-Path $llamaDir ".mesh-llm-patched-sha"', script)
        prepare = script.index("    Prepare-Llama\n    # Resolve after preparing")
        resolve = script.index("$script:buildDir = Resolve-StageBuildDir $backendName")
        cmake = script.index('"-B", $buildDir,')
        self.assertLess(prepare, resolve)
        self.assertLess(resolve, cmake)
        self.assertNotIn('Join-Path $llamaBuildRoot "build-stage-abi-$backendName"', script)


@unittest.skipUnless(shutil.which("pwsh"), "pwsh is not installed")
class ResolveStageBuildDirTests(unittest.TestCase):
    """Runs the script's own Resolve-StageBuildDir under pwsh."""

    def resolve(self, llama_dir, build_root, backend, override=None):
        script = SCRIPT.read_text()
        match = re.search(
            r"^function Resolve-StageBuildDir \{.*?^\}$", script, re.MULTILINE | re.DOTALL
        )
        self.assertIsNotNone(match, "Resolve-StageBuildDir not found")
        env = {
            key: value
            for key, value in __import__("os").environ.items()
            if key != "LLAMA_STAGE_BUILD_DIR"
        }
        if override is not None:
            env["LLAMA_STAGE_BUILD_DIR"] = override
        command = (
            f"$llamaDir = '{llama_dir}'; $llamaBuildRoot = '{build_root}'; "
            f"{match.group(0)}; Resolve-StageBuildDir '{backend}'"
        )
        result = subprocess.run(
            ["pwsh", "-NoProfile", "-NonInteractive", "-Command", command],
            capture_output=True,
            text=True,
            env=env,
            check=True,
        )
        return result.stdout.strip()

    def test_the_directory_follows_the_patched_sha(self):
        with tempfile.TemporaryDirectory() as tmp:
            llama = Path(tmp) / "llama.cpp"
            llama.mkdir()
            root = Path(tmp) / "llama-build"
            # No prepare yet: the unkeyed name.
            self.assertEqual(
                Path(self.resolve(llama, root, "cuda")), root / "build-stage-abi-cuda"
            )
            (llama / ".mesh-llm-patched-sha").write_text(
                "0123456789abcdef0123456789abcdef01234567\n"
            )
            self.assertEqual(
                Path(self.resolve(llama, root, "cuda")),
                root / "build-stage-abi-cuda-0123456789ab",
            )
            # A different pin gets its own directory.
            (llama / ".mesh-llm-patched-sha").write_text(
                "fedcba9876543210fedcba9876543210fedcba98\n"
            )
            self.assertEqual(
                Path(self.resolve(llama, root, "cuda")),
                root / "build-stage-abi-cuda-fedcba987654",
            )
            # An explicit directory always wins.
            self.assertEqual(
                self.resolve(llama, root, "cuda", override="D:/abi"), "D:/abi"
            )


@unittest.skipUnless(
    __import__("os").name == "nt" and shutil.which("pwsh"),
    "Windows path handling, needs pwsh on Windows",
)
class BashRepoPathTests(unittest.TestCase):
    """The llama workdir handed to bash must not be a drive-letter path."""

    def convert(self, repo_root, path, wsl=False, check=True):
        script = SCRIPT.read_text()
        match = re.search(
            r"^function ConvertTo-BashRepoPath \{.*?^\}$", script, re.MULTILINE | re.DOTALL
        )
        self.assertIsNotNone(match, "ConvertTo-BashRepoPath not found")
        flag = " -Wsl" if wsl else ""
        command = f"$repoRoot = '{repo_root}'; {match.group(0)}; ConvertTo-BashRepoPath '{path}'{flag}"
        result = subprocess.run(
            ["pwsh", "-NoProfile", "-NonInteractive", "-Command", command],
            capture_output=True,
            text=True,
            check=check,
        )
        return result if not check else result.stdout.strip()

    def test_a_workdir_inside_the_repository_is_passed_relative(self):
        self.assertEqual(
            self.convert(r"D:\work\mesh-llm", r"D:\work\mesh-llm\.deps\llama.cpp"),
            ".deps/llama.cpp",
        )
        self.assertEqual(
            self.convert(r"D:\work\mesh-llm\\", r"d:\WORK\mesh-llm\.deps\other"),
            ".deps/other",
        )

    def test_a_workdir_outside_the_repository_is_absolute_for_the_running_bash(self):
        # Git Bash reads a drive-letter path; WSL bash would read it as
        # relative and needs the /mnt/<drive> form.
        self.assertEqual(
            self.convert(r"D:\work\mesh-llm", r"E:\cache\llama.cpp"), "E:/cache/llama.cpp"
        )
        self.assertEqual(
            self.convert(r"D:\work\mesh-llm", r"E:\cache\llama.cpp", wsl=True),
            "/mnt/e/cache/llama.cpp",
        )
        # Inside the repository the relative form serves both shells.
        self.assertEqual(
            self.convert(r"D:\work\mesh-llm", r"D:\work\mesh-llm\.deps\llama.cpp", wsl=True),
            ".deps/llama.cpp",
        )

    def test_a_workdir_without_a_drive_letter_is_rejected(self):
        result = self.convert(r"D:\work\mesh-llm", r"\\server\share\llama.cpp", check=False)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("inside the repository or on a drive-letter path", result.stderr + result.stdout)

    def llama_dir(self, repo_root, override, cwd):
        script = SCRIPT.read_text()
        match = re.search(r"^\$llamaDir = .*$", script, re.MULTILINE)
        self.assertIsNotNone(match, "$llamaDir assignment not found")
        env = {**__import__("os").environ, "MESH_LLM_LLAMA_DIR": override}
        command = f"$repoRoot = '{repo_root}'; {match.group(0)}; $llamaDir"
        result = subprocess.run(
            ["pwsh", "-NoProfile", "-NonInteractive", "-Command", command],
            capture_output=True,
            text=True,
            env=env,
            cwd=cwd,
            check=True,
        )
        return result.stdout.strip()

    def test_a_relative_llama_dir_is_relative_to_the_repository_root(self):
        # Run from another directory: the prepare step resolves the workdir
        # with .NET, cmake and the stamp read it from the repository location.
        with tempfile.TemporaryDirectory() as elsewhere:
            self.assertEqual(
                self.llama_dir(r"D:\work\mesh-llm", r".deps\other", elsewhere),
                r"D:\work\mesh-llm\.deps\other",
            )
            self.assertEqual(
                self.llama_dir(r"D:\work\mesh-llm", "custom/llama.cpp", elsewhere),
                r"D:\work\mesh-llm\custom\llama.cpp",
            )
            self.assertEqual(
                self.llama_dir(r"D:\work\mesh-llm", r"E:\cache\llama.cpp", elsewhere),
                r"E:\cache\llama.cpp",
            )

    def test_the_default_workdir_is_left_to_prepare_llama(self):
        script = SCRIPT.read_text()
        self.assertIn("Remove-Item Env:LLAMA_WORKDIR -ErrorAction SilentlyContinue", script)
        self.assertNotIn('$env:LLAMA_WORKDIR = $llamaDir.Replace(', script)


if __name__ == "__main__":
    unittest.main()
