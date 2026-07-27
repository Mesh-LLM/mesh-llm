import os
import pathlib
import shlex
import shutil
import subprocess
import sys


ROOT = pathlib.Path(__file__).resolve().parents[2]
CUDA_TOOLKIT_LIB = ROOT / "scripts" / "lib" / "cuda-toolkit.sh"
CUDA_ENV_VARS = (
    "CUDACXX",
    "CUDAToolkit_ROOT",
    "NVCC",
    "CUDA_HOME",
    "CUDA_PATH",
    "CUDA_LIBRARY_PATH",
    "LIBRARY_PATH",
    "LD_LIBRARY_PATH",
)


def clean_cuda_env(env: dict[str, str]) -> None:
    for name in CUDA_ENV_VARS:
        env.pop(name, None)


def run_bash(script: str, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["/bin/bash", "-c", f"set -euo pipefail\n{script}"],
        check=False,
        capture_output=True,
        env=env,
        text=True,
    )


def test_resolves_toolkit_root_through_nvcc_symlink(tmp_path: pathlib.Path) -> None:
    toolkit = tmp_path / "cuda-13.2"
    nvcc = toolkit / "bin" / "nvcc"
    header = toolkit / "targets" / "sbsa-linux" / "include" / "cuda_runtime.h"
    library_dir = toolkit / "targets" / "sbsa-linux" / "lib"
    exposed_bin = tmp_path / "bin"
    nvcc.parent.mkdir(parents=True)
    header.parent.mkdir(parents=True)
    exposed_bin.mkdir()
    nvcc.write_text("#!/usr/bin/env bash\n", encoding="utf-8")
    nvcc.chmod(0o755)
    header.write_text("", encoding="utf-8")
    library_dir.mkdir(parents=True)
    for name in ("libcudart.so", "libcublas.so", "libcublasLt.so"):
        (library_dir / name).write_text("", encoding="utf-8")
    (exposed_bin / "nvcc").symlink_to(nvcc)

    env = os.environ.copy()
    env["PATH"] = f"{exposed_bin}:{env['PATH']}"
    clean_cuda_env(env)
    result = run_bash(
        f"""
        source {shlex.quote(str(CUDA_TOOLKIT_LIB))}
        configure_cuda_toolkit_env
        printf '%s\\n%s\\n%s\\n%s\\n%s\\n%s\\n' \\
          "$CUDACXX" "$CUDAToolkit_ROOT" "$NVCC" \\
          "$CUDA_LIBRARY_PATH" "$LIBRARY_PATH" "$LD_LIBRARY_PATH"
        """,
        env,
    )

    result.check_returncode()
    assert result.stdout.splitlines() == [
        str(nvcc.resolve()),
        str(toolkit.resolve()),
        str(nvcc.resolve()),
        str(library_dir.resolve()),
        str(library_dir.resolve()),
        str(library_dir.resolve()),
    ]


def test_canonical_path_falls_back_to_python(tmp_path: pathlib.Path) -> None:
    target = tmp_path / "target"
    symlink = tmp_path / "symlink"
    target.write_text("", encoding="utf-8")
    symlink.symlink_to(target)

    exposed_bin = tmp_path / "bin"
    exposed_bin.mkdir()
    dirname = shutil.which("dirname")
    assert dirname is not None
    (exposed_bin / "dirname").symlink_to(dirname)
    (exposed_bin / "python").symlink_to(sys.executable)
    env = os.environ.copy()
    env["PATH"] = str(exposed_bin)

    result = run_bash(
        f"""
        source {shlex.quote(str(CUDA_TOOLKIT_LIB))}
        readlink() {{ return 1; }}
        realpath() {{ return 1; }}
        cuda_canonical_path {shlex.quote(str(symlink))}
        """,
        env,
    )

    result.check_returncode()
    assert result.stdout.strip() == str(target.resolve())


def test_propagates_helper_failure_when_nvcc_is_missing() -> None:
    env = os.environ.copy()
    env["PATH"] = ""
    clean_cuda_env(env)

    result = run_bash(
        f"""
        source {shlex.quote(str(CUDA_TOOLKIT_LIB))}
        configure_cuda_toolkit_env
        """,
        env,
    )

    assert result.returncode != 0


def test_preserves_explicit_cuda_environment(tmp_path: pathlib.Path) -> None:
    compiler = tmp_path / "custom-nvcc"
    compiler.write_text("#!/usr/bin/env bash\n", encoding="utf-8")
    compiler.chmod(0o755)
    explicit_root = tmp_path / "custom-toolkit"
    env = os.environ.copy()
    clean_cuda_env(env)
    env["CUDACXX"] = str(compiler)
    env["CUDAToolkit_ROOT"] = str(explicit_root)

    result = run_bash(
        f"""
        source {shlex.quote(str(CUDA_TOOLKIT_LIB))}
        configure_cuda_toolkit_env
        printf '%s\\n%s\\n' "$CUDACXX" "$CUDAToolkit_ROOT"
        """,
        env,
    )

    result.check_returncode()
    assert result.stdout.splitlines() == [str(compiler), str(explicit_root)]
