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
    "CMAKE_CUDA_COMPILER",
    "CUDAToolkit_ROOT",
    "NVCC",
    "CUDA_HOME",
    "CUDA_PATH",
    "CUDA_LIBRARY_PATH",
    "LIBRARY_PATH",
    "LD_LIBRARY_PATH",
    "MESH_CUDA_VERSION",
    "MESH_LLM_CUDA_TOOLKIT_MAJOR",
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
    nvcc.write_text("#!/bin/bash\n", encoding="utf-8")
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


def write_fake_nvcc(path: pathlib.Path, version: str) -> None:
    path.write_text(
        "#!/bin/bash\n"
        'if [[ "${1:-}" == "--version" ]]; then\n'
        f'  printf "Cuda compilation tools, release {version}, V{version}.0\\n"\n'
        "  exit 0\n"
        "fi\n",
        encoding="utf-8",
    )
    path.chmod(0o755)


def detect_version(
    env: dict[str, str],
) -> subprocess.CompletedProcess[str]:
    script = ROOT / "scripts" / "detect-cuda-toolkit-version.sh"
    return subprocess.run(
        ["/bin/bash", str(script)],
        check=False,
        capture_output=True,
        env=env,
        text=True,
    )


def test_detector_uses_cudacxx_before_nvcc_and_path(tmp_path: pathlib.Path) -> None:
    tool_dir = tmp_path / "bin"
    tool_dir.mkdir()
    cudacxx = tool_dir / "cudacxx-nvcc"
    nvcc = tool_dir / "nvcc-12"
    path_nvcc = tool_dir / "nvcc"
    write_fake_nvcc(cudacxx, "13.0")
    write_fake_nvcc(nvcc, "12.9")
    write_fake_nvcc(path_nvcc, "11.8")

    env = os.environ.copy()
    clean_cuda_env(env)
    env.update(
        {
            "CUDACXX": str(cudacxx),
            "NVCC": str(nvcc),
            "PATH": str(tool_dir),
        }
    )
    result = detect_version(env)

    result.check_returncode()
    assert result.stdout.strip() == "13.0"


def test_detector_uses_nvcc_when_cudacxx_is_unset(tmp_path: pathlib.Path) -> None:
    tool_dir = tmp_path / "bin"
    tool_dir.mkdir()
    nvcc = tool_dir / "nvcc-12"
    path_nvcc = tool_dir / "nvcc"
    write_fake_nvcc(nvcc, "12.9")
    write_fake_nvcc(path_nvcc, "11.8")

    env = os.environ.copy()
    clean_cuda_env(env)
    env.update({"NVCC": str(nvcc), "PATH": str(tool_dir)})
    result = detect_version(env)

    result.check_returncode()
    assert result.stdout.strip() == "12.9"


def test_detector_uses_path_nvcc_when_overrides_are_unset(tmp_path: pathlib.Path) -> None:
    tool_dir = tmp_path / "bin"
    tool_dir.mkdir()
    write_fake_nvcc(tool_dir / "nvcc", "11.8")

    env = os.environ.copy()
    clean_cuda_env(env)
    env["PATH"] = str(tool_dir)
    result = detect_version(env)

    result.check_returncode()
    assert result.stdout.strip() == "11.8"


def test_detector_accepts_major_only_cuda_version_declaration(
    tmp_path: pathlib.Path,
) -> None:
    tool_dir = tmp_path / "bin"
    tool_dir.mkdir()
    write_fake_nvcc(tool_dir / "nvcc", "13.0")

    env = os.environ.copy()
    clean_cuda_env(env)
    env.update({"MESH_CUDA_VERSION": "13", "PATH": str(tool_dir)})
    result = detect_version(env)

    result.check_returncode()
    assert result.stdout.strip() == "13"


def test_detector_rejects_cuda_version_minor_mismatch(tmp_path: pathlib.Path) -> None:
    tool_dir = tmp_path / "bin"
    tool_dir.mkdir()
    write_fake_nvcc(tool_dir / "nvcc", "13.0")

    env = os.environ.copy()
    clean_cuda_env(env)
    env.update({"MESH_CUDA_VERSION": "13.1.2", "PATH": str(tool_dir)})
    result = detect_version(env)

    assert result.returncode != 0
    assert (
        "does not match the selected CUDA compiler/toolkit version 13.0"
        in result.stderr
    )


def test_detector_uses_toolkit_owned_metadata_without_compiler(
    tmp_path: pathlib.Path,
) -> None:
    toolkit = tmp_path / "cuda-12.9"
    toolkit.mkdir()
    (toolkit / "version.json").write_text(
        '{"version": "12.9.2"}\n', encoding="utf-8"
    )

    env = os.environ.copy()
    clean_cuda_env(env)
    env.update({"CUDAToolkit_ROOT": str(toolkit), "PATH": str(tmp_path / "bin")})
    result = detect_version(env)

    result.check_returncode()
    assert result.stdout.strip() == "12.9"


def test_detector_fails_without_compiler_or_toolkit_metadata(tmp_path: pathlib.Path) -> None:
    env = os.environ.copy()
    clean_cuda_env(env)
    env["PATH"] = str(tmp_path)
    result = detect_version(env)

    assert result.returncode != 0
    assert "CUDA toolkit version could not be detected" in result.stderr
    assert "nvidia-smi" not in result.stderr


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


def test_windows_cmake_compiler_path_restores_missing_exe_suffix(tmp_path: pathlib.Path) -> None:
    compiler = tmp_path / "nvcc.exe"
    compiler.write_text("#!/bin/bash\n", encoding="utf-8")
    compiler.chmod(0o755)
    env = os.environ.copy()
    clean_cuda_env(env)
    env["CUDACXX"] = str(compiler.with_suffix(""))

    result = run_bash(
        f"""
        source {shlex.quote(str(CUDA_TOOLKIT_LIB))}
        uname() {{ printf 'MINGW64_NT-10.0\\n'; }}
        configure_cuda_toolkit_env
        printf '%s\\n' "$CUDACXX"
        """,
        env,
    )

    result.check_returncode()
    assert result.stdout.strip() == str(compiler)


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
    compiler.write_text("#!/bin/bash\n", encoding="utf-8")
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
