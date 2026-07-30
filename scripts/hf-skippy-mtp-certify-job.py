# /// script
# requires-python = ">=3.11"
# ///

"""Validate a mounted target GGUF, projector, and external MTP sidecar on HF Jobs."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import urllib.parse
import urllib.request
from pathlib import Path


def run(*command: str, cwd: Path | None = None) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=cwd, check=True)


def ensure_build_tools() -> None:
    required = ("git", "curl", "cmake", "c++", "ld.lld")
    if any(shutil.which(tool) is None for tool in required):
        if shutil.which("apt-get") is None:
            raise RuntimeError(f"missing build tools: {required}")
        run("apt-get", "update")
        run(
            "apt-get",
            "install",
            "-y",
            "build-essential",
            "cmake",
            "curl",
            "git",
            "lld",
            "pkg-config",
        )
    if shutil.which("cargo") is None:
        run(
            "sh",
            "-c",
            "curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y",
        )
        os.environ["PATH"] = f"{Path.home() / '.cargo' / 'bin'}:{os.environ['PATH']}"
    if shutil.which("just") is None:
        run("cargo", "install", "just", "--locked")


def checkout_mesh(repo: str, revision: str, root: Path) -> None:
    if root.exists():
        shutil.rmtree(root)
    run("git", "clone", "--filter=blob:none", repo, str(root))
    run("git", "checkout", revision, cwd=root)


def model_parts(args: argparse.Namespace) -> list[Path]:
    parts = sorted(Path(args.model_root).glob(args.model_pattern))
    if len(parts) != args.expected_parts:
        raise RuntimeError(
            f"expected {args.expected_parts} target parts matching {args.model_pattern!r}, "
            f"found {len(parts)}"
        )
    return parts


def require_gguf_magic(path: Path) -> Path:
    with path.open("rb") as handle:
        magic = handle.read(4)
    if magic != b"GGUF":
        raise RuntimeError(f"invalid GGUF magic for {path}: {magic!r}")
    return path


def projector_path(args: argparse.Namespace) -> Path:
    if not args.projector_url:
        return require_gguf_magic(Path(args.projector))
    scheme = urllib.parse.urlparse(args.projector_url).scheme
    if scheme not in ("http", "https"):
        raise RuntimeError(f"unsupported projector URL scheme: {scheme!r}")
    target = Path(args.projector_local_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(f"{target.suffix}.part")
    print(f"+ download {args.projector_url} -> {target}", flush=True)
    with urllib.request.urlopen(args.projector_url) as response, temporary.open("wb") as output:
        shutil.copyfileobj(response, output, length=8 * 1024 * 1024)
    temporary.replace(target)
    return require_gguf_magic(target)


def run_report(command: list[str], report_out: str) -> None:
    print("+", " ".join(command), flush=True)
    completed = subprocess.run(command, text=True, capture_output=True)
    if completed.stderr:
        print(completed.stderr, end="", flush=True)
    if completed.stdout:
        print(completed.stdout, end="", flush=True)
    if completed.returncode != 0:
        raise subprocess.CalledProcessError(completed.returncode, command)
    report = Path(report_out)
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(completed.stdout, encoding="utf-8")
    print(f"certification report: {report}", flush=True)


def certify(args: argparse.Namespace, mesh_root: Path) -> None:
    binary = mesh_root / "target" / "release" / "skippy-quantize"
    run("just", "skippy-quantize-standalone-release-build", "cpu", cwd=mesh_root)
    projector = projector_path(args)
    if args.projector_only:
        run_report(
            [str(binary), "validate-projector", "--projector", str(projector), "--json"],
            args.report_out,
        )
        return
    target_parts = [require_gguf_magic(path) for path in model_parts(args)]
    mtp_draft = require_gguf_magic(Path(args.mtp_draft))
    command = [str(binary), "validate-mtp-attach"]
    for part in target_parts:
        command.extend(("--model", str(part)))
    command.extend(
        (
            "--mtp-draft",
            str(mtp_draft),
            "--layer-count",
            str(args.layer_count),
            "--ctx-size",
            str(args.ctx_size),
            "--projector",
            str(projector),
            "--json",
        )
    )
    if args.mtp_layer_count is not None:
        command.extend(("--mtp-layer-count", str(args.mtp_layer_count)))
    run_report(command, args.report_out)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-root", default="/target")
    parser.add_argument("--model-pattern", required=True)
    parser.add_argument("--expected-parts", type=int, default=1)
    parser.add_argument("--mtp-draft", required=True)
    parser.add_argument("--projector", required=True)
    parser.add_argument("--projector-url")
    parser.add_argument("--projector-local-path", default="/tmp/mmproj.gguf")
    parser.add_argument("--projector-only", action="store_true")
    parser.add_argument("--layer-count", type=int, required=True)
    parser.add_argument("--mtp-layer-count", type=int)
    parser.add_argument("--ctx-size", type=int, default=64)
    parser.add_argument("--report-out", default="/results/mtp-attach-certification.json")
    parser.add_argument("--mesh-repo", default="https://github.com/Mesh-LLM/mesh-llm.git")
    parser.add_argument("--mesh-revision", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_build_tools()
    mesh_root = Path("/tmp/mesh-llm")
    checkout_mesh(args.mesh_repo, args.mesh_revision, mesh_root)
    certify(args, mesh_root)


if __name__ == "__main__":
    main()
