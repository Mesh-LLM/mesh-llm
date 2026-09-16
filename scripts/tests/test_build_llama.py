from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
BUILD_SCRIPT = ROOT / "scripts" / "build-llama.sh"

# Records every invocation and emulates just enough of CMake for the script
# to finish: configure resolves the generator exactly like CMake (the -G
# argument wins, then the CMAKE_GENERATOR environment variable, then the
# Unix Makefiles default), writes a fresh CMakeCache.txt carrying the result,
# and --build drops the dynamic-link output the script verifies.
CMAKE_STUB = """\
#!/usr/bin/env bash
set -euo pipefail
args=("$@")
printf '%s\\n' "$*" >> "${CMAKE_STUB_LOG:?}"
target=""
for ((i = 0; i < ${#args[@]}; i++)); do
  case "${args[i]}" in
    -B | --build)
      if ((i + 1 < ${#args[@]})); then target="${args[i + 1]}"; fi
      ;;
  esac
done
if [[ "${args[0]}" == "--build" ]]; then
  for name in libllama libllama-common libmtmd; do
    for ext in dylib so; do
      touch "$target/$name.$ext"
    done
  done
  exit 0
fi
generator="${CMAKE_GENERATOR:-Unix Makefiles}"
for ((i = 0; i < ${#args[@]}; i++)); do
  if [[ "${args[i]}" == "-G" ]] && ((i + 1 < ${#args[@]})); then
    generator="${args[i + 1]}"
  fi
done
mkdir -p "$target"
printf 'CMAKE_GENERATOR:INTERNAL=%s\\n' "$generator" > "$target/CMakeCache.txt"
"""

NINJA_STUB = "#!/usr/bin/env bash\nexit 0\n"


class BuildLlamaGeneratorGuardTests(unittest.TestCase):
    ninja_absent: bool = False

    def ninja_free_path(self, stubs: Path) -> tuple[str, bool]:
        full_path = os.environ["PATH"]
        ninja = shutil.which("ninja", path=full_path)
        if ninja is None:
            return f"{stubs}{os.pathsep}{full_path}", True
        ninja_dir = str(Path(ninja).parent)
        kept = [d for d in full_path.split(os.pathsep) if d and d != ninja_dir]
        restricted = f"{stubs}{os.pathsep}{os.pathsep.join(kept)}"
        for tool in ("bash", "git", "sed", "tr", "touch", "mkdir"):
            if shutil.which(tool, path=restricted) is None:
                restricted += os.pathsep + os.pathsep.join(
                    ("/usr/bin", "/bin", "/usr/sbin", "/sbin")
                )
                break
        return restricted, shutil.which("ninja", path=restricted) is None

    def run_build(
        self,
        *,
        cache_generator: str | None = None,
        require_existing: bool = False,
        ninja_on_path: bool = True,
        env_cmake_generator: str | None = None,
    ) -> tuple[subprocess.CompletedProcess[str], Path, Path]:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        root = Path(directory.name).resolve()
        workdir = root / "llama.cpp"
        (workdir / ".git").mkdir(parents=True)
        (workdir / ".mesh-llm-patched-sha").write_text("0" * 40 + "\n")
        build = root / "llama-build" / "build-stage-abi-dynamic-cpu"
        build.mkdir(parents=True)
        (build / "marker.txt").write_text("warm cache\n")
        if cache_generator is not None:
            (build / "CMakeCache.txt").write_text(
                f"CMAKE_GENERATOR:INTERNAL={cache_generator}\n"
            )
        stubs = root / "stubs"
        stubs.mkdir()
        for name, body in (("cmake", CMAKE_STUB), ("ninja", NINJA_STUB)):
            if name == "ninja" and not ninja_on_path:
                continue
            path = stubs / name
            path.write_text(body)
            path.chmod(0o755)
        log = root / "cmake-stub.log"
        env = {
            key: value for key, value in os.environ.items()
            if not key.startswith(("LLAMA_STAGE_", "SKIPPY_LLAMA_", "LLAMA_"))
            and key
            not in (
                "MESH_LLM_LLAMA_BUILD_ROOT",
                "CMAKE_STUB_LOG",
                "CMAKE_GENERATOR",
            )
        }
        if env_cmake_generator is not None:
            env["CMAKE_GENERATOR"] = env_cmake_generator
        if ninja_on_path:
            path_value = f"{stubs}{os.pathsep}{env['PATH']}"
            self.ninja_absent = False
        else:
            path_value, self.ninja_absent = self.ninja_free_path(stubs)
        env.update(
            {
                "LLAMA_WORKDIR": str(workdir),
                "LLAMA_STAGE_BUILD_DIR": str(build),
                "LLAMA_STAGE_BACKEND": "cpu",
                "LLAMA_STAGE_LINK_MODE": "dynamic",
                "CMAKE_STUB_LOG": str(log),
                "PATH": path_value,
            }
        )
        command = ["bash", str(BUILD_SCRIPT)]
        if require_existing:
            command.append("--require-existing")
        result = subprocess.run(
            command, cwd=ROOT, env=env, capture_output=True, text=True
        )
        return result, build, log

    def test_stale_makefiles_cache_is_cleared_when_ninja_is_selected(self) -> None:
        result, build, log = self.run_build(cache_generator="Unix Makefiles")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("clearing stale CMake cache", result.stdout)
        self.assertFalse((build / "marker.txt").exists())
        self.assertEqual(
            (build / "CMakeCache.txt").read_text(), "CMAKE_GENERATOR:INTERNAL=Ninja\n"
        )
        calls = log.read_text().splitlines()
        self.assertTrue(any("-G Ninja" in call for call in calls))
        self.assertTrue(any("--build" in call for call in calls))
        self.assertIn("built patched llama.cpp", result.stdout)

    def test_selected_generator_is_passed_when_ninja_is_unavailable(self) -> None:
        result, build, log = self.run_build(ninja_on_path=False)
        if not self.ninja_absent:
            self.skipTest("ninja could not be removed from PATH on this host")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        calls = log.read_text().splitlines()
        self.assertTrue(any("-G Unix Makefiles" in call for call in calls))
        self.assertEqual(
            (build / "CMakeCache.txt").read_text(),
            "CMAKE_GENERATOR:INTERNAL=Unix Makefiles\n",
        )
        self.assertIn("built patched llama.cpp", result.stdout)

    def test_inherited_cmake_generator_env_cannot_override_selection(self) -> None:
        result, build, log = self.run_build(
            ninja_on_path=False, env_cmake_generator="Ninja"
        )
        if not self.ninja_absent:
            self.skipTest("ninja could not be removed from PATH on this host")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        calls = log.read_text().splitlines()
        self.assertTrue(any("-G Unix Makefiles" in call for call in calls))
        self.assertEqual(
            (build / "CMakeCache.txt").read_text(),
            "CMAKE_GENERATOR:INTERNAL=Unix Makefiles\n",
        )

    def test_matching_ninja_cache_is_not_cleared(self) -> None:
        result, build, _ = self.run_build(cache_generator="Ninja")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertNotIn("clearing stale CMake cache", result.stdout)
        self.assertTrue((build / "marker.txt").exists())

    def test_stale_cache_is_not_cleared_under_require_existing(self) -> None:
        result, build, _ = self.run_build(
            cache_generator="Unix Makefiles", require_existing=True
        )
        self.assertEqual(result.returncode, 1)
        self.assertIn("refusing to rebuild", result.stderr)
        self.assertTrue((build / "marker.txt").exists())


if __name__ == "__main__":
    unittest.main()
