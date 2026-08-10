from __future__ import annotations

import os
from pathlib import Path
import shutil
import stat
import subprocess
import tempfile
import textwrap
import unittest


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "build-release.sh"
HOST_SCRIPT = ROOT / "scripts" / "build-host.sh"


class BuildReleaseScriptTests(unittest.TestCase):
    def test_release_host_never_enables_backend_gpu_features(self) -> None:
        script = HOST_SCRIPT.read_text(encoding="utf-8")

        self.assertNotIn("gpu-bench-cuda", script)
        self.assertNotIn("gpu-bench-hip", script)
        self.assertNotIn("build-llama.sh", script)

    def test_dynamic_native_runtime_feature_is_required(self) -> None:
        script = HOST_SCRIPT.read_text(encoding="utf-8")

        self.assertIn('"web-ui,dynamic-native-runtime"', script)
        self.assertIn("--no-default-features", script)
        self.assertIn("MESH_LLM_DYNAMIC_NATIVE_RUNTIME=0 is unsupported", script)

    def test_release_entry_point_delegates_to_canonical_host_builder(self) -> None:
        script = SCRIPT.read_text(encoding="utf-8")

        self.assertIn('exec "$SCRIPT_DIR/build-host.sh" --profile release', script)

    def test_cuda_selection_does_not_change_host_build(self) -> None:
        cargo_log = self.run_build_release_with_backend("cuda")

        self.assertIn("build --release --locked -p mesh-llm", cargo_log)
        self.assertIn("--features web-ui,dynamic-native-runtime", cargo_log)
        self.assertNotIn("gpu-bench-cuda", cargo_log)
        self.assertNotIn("gpu-bench-hip", cargo_log)

    def test_rocm_selection_does_not_change_host_build(self) -> None:
        cargo_log = self.run_build_release_with_backend("rocm")

        self.assertIn("build --release --locked -p mesh-llm", cargo_log)
        self.assertIn("--features web-ui,dynamic-native-runtime", cargo_log)
        self.assertNotIn("gpu-bench-cuda", cargo_log)
        self.assertNotIn("gpu-bench-hip", cargo_log)

    def test_static_release_host_request_is_rejected(self) -> None:
        with self.assertRaises(subprocess.CalledProcessError):
            self.run_build_release_with_backend("metal", dynamic_native_runtime=False)

    def test_non_release_profiles_stamp_the_commit_sha(self) -> None:
        """A dev build must not report a bare, release-shaped version.

        `stamp_build_version` derives `+g<sha>` for non-release profiles, but it
        is only useful if it actually runs for those profiles. Guard the call
        site: stamping used to sit inside the release-only branch, so the
        derivation was unreachable and debug builds shipped an unstamped
        version indistinguishable from a release binary.
        """
        for profile in ("debug", "dev"):
            with self.subTest(profile=profile):
                stamped = self.run_build_host_with_profile(profile)
                self.assertEqual(stamped, "0.68.0+gABC123")

    def test_release_profile_stamps_the_plain_version(self) -> None:
        self.assertEqual(self.run_build_host_with_profile("release"), "0.68.0")

    def run_build_host_with_profile(self, profile: str) -> str:
        """Returns MESH_LLM_BUILD_VERSION as seen by `cargo build`."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            scripts_dir = tmp / "scripts"
            bin_dir = tmp / "bin"
            scripts_dir.mkdir()
            bin_dir.mkdir()

            copied_host_script = scripts_dir / "build-host.sh"
            shutil.copy(HOST_SCRIPT, copied_host_script)
            copied_host_script.chmod(copied_host_script.stat().st_mode | stat.S_IXUSR)

            self.write_executable(
                scripts_dir / "build-ui.sh",
                """
                #!/usr/bin/env bash
                set -euo pipefail
                echo "stub build-ui $*"
                """,
            )
            self.write_executable(
                bin_dir / "uname",
                """
                #!/usr/bin/env bash
                echo Linux
                """,
            )
            self.write_executable(
                bin_dir / "git",
                """
                #!/usr/bin/env bash
                if [[ "$1" == "-C" ]]; then
                  shift 2
                fi
                case "$1" in
                  rev-parse) echo abc123 ;;
                  status) ;;
                  *) echo "unexpected git $*" >&2; exit 2 ;;
                esac
                """,
            )
            version_log = tmp / "build-version.log"
            self.write_executable(
                bin_dir / "cargo",
                """
                #!/usr/bin/env bash
                set -euo pipefail
                if [[ "$1" == "pkgid" ]]; then
                  echo "file:///tmp/mesh-llm#0.68.0"
                  exit 0
                fi
                if [[ "$1" == "build" ]]; then
                  printf '%s' "${MESH_LLM_BUILD_VERSION:-<unset>}" > "$VERSION_LOG"
                fi
                """,
            )

            env = os.environ.copy()
            env.update(
                {
                    "VERSION_LOG": str(version_log),
                    "MESH_LLM_SKIP_UI": "1",
                    "PATH": f"{bin_dir}{os.pathsep}{env['PATH']}",
                }
            )
            env.pop("MESH_LLM_BUILD_VERSION", None)
            subprocess.run(
                [str(copied_host_script), "--profile", profile],
                cwd=tmp,
                env=env,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            return version_log.read_text(encoding="utf-8")

    def run_build_release_with_backend(
        self, backend: str, *, dynamic_native_runtime: bool = True
    ) -> str:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            scripts_dir = tmp / "scripts"
            bin_dir = tmp / "bin"
            scripts_dir.mkdir()
            bin_dir.mkdir()

            copied_script = scripts_dir / "build-release.sh"
            shutil.copy(SCRIPT, copied_script)
            copied_script.chmod(copied_script.stat().st_mode | stat.S_IXUSR)
            copied_host_script = scripts_dir / "build-host.sh"
            shutil.copy(HOST_SCRIPT, copied_host_script)
            copied_host_script.chmod(
                copied_host_script.stat().st_mode | stat.S_IXUSR
            )

            self.write_executable(
                scripts_dir / "build-ui.sh",
                """
                #!/usr/bin/env bash
                set -euo pipefail
                echo "stub build-ui $*"
                """,
            )
            self.write_executable(
                scripts_dir / "prepare-llama.sh",
                """
                #!/usr/bin/env bash
                set -euo pipefail
                echo "stub prepare-llama $*"
                """,
            )
            self.write_executable(
                scripts_dir / "build-llama.sh",
                """
                #!/usr/bin/env bash
                set -euo pipefail
                echo "stub build-llama $*"
                """,
            )
            self.write_executable(
                bin_dir / "uname",
                """
                #!/usr/bin/env bash
                echo Linux
                """,
            )
            self.write_executable(
                bin_dir / "ld.lld",
                """
                #!/usr/bin/env bash
                exit 0
                """,
            )
            self.write_executable(
                bin_dir / "git",
                """
                #!/usr/bin/env bash
                if [[ "$1" == "-C" ]]; then
                  shift 2
                fi
                case "$1" in
                  rev-parse) echo abc123 ;;
                  status) ;;
                  *) echo "unexpected git $*" >&2; exit 2 ;;
                esac
                """,
            )
            cargo_log = tmp / "cargo.log"
            self.write_executable(
                bin_dir / "cargo",
                """
                #!/usr/bin/env bash
                set -euo pipefail
                if [[ "$1" == "pkgid" ]]; then
                  echo "file:///tmp/mesh-llm#0.68.0"
                  exit 0
                fi
                printf '%s ' "$@" >> "$CARGO_LOG"
                printf '\n' >> "$CARGO_LOG"
                """,
            )

            env = os.environ.copy()
            env.update(
                {
                    "CARGO_LOG": str(cargo_log),
                    "LLAMA_STAGE_BACKEND": backend,
                    "MESH_LLM_DYNAMIC_NATIVE_RUNTIME": "1"
                    if dynamic_native_runtime
                    else "0",
                    "PATH": f"{bin_dir}{os.pathsep}{env['PATH']}",
                }
            )
            subprocess.run(
                [str(copied_script)],
                cwd=tmp,
                env=env,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            return cargo_log.read_text(encoding="utf-8")

    def write_executable(self, path: Path, content: str) -> None:
        path.write_text(textwrap.dedent(content).lstrip(), encoding="utf-8")
        path.chmod(path.stat().st_mode | stat.S_IXUSR)


if __name__ == "__main__":
    unittest.main()
