# /// script
# requires-python = ">=3.11"
# ///

"""Validate a mounted target GGUF, projector, and external MTP sidecar on HF Jobs."""

from __future__ import annotations

import argparse
import ipaddress
import os
import shutil
import socket
import subprocess
import urllib.parse
import urllib.request
from pathlib import Path


PROJECTOR_DOWNLOAD_TIMEOUT_SECONDS = 60
PROJECTOR_DOWNLOAD_MAX_BYTES = 64 * 1024 * 1024 * 1024
PROJECTOR_DOWNLOAD_CHUNK_BYTES = 8 * 1024 * 1024
TRUSTED_PROJECTOR_HOST_SUFFIXES = ("huggingface.co", "hf.co", "xethub.hf.co")


def run(*command: str, cwd: Path | None = None) -> None:
    """Run run operation."""
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=cwd, check=True)


def ensure_build_tools() -> None:
    """Execute ensure build tools operation."""
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
    """Validate checkout mesh.

    Raises:
        ValidationError: If validation fails.
    """
    if root.exists():
        shutil.rmtree(root)
    run("git", "clone", "--filter=blob:none", repo, str(root))
    run("git", "checkout", revision, cwd=root)


def model_parts(args: argparse.Namespace) -> list[Path]:
    """Execute model parts operation."""
    parts = sorted(Path(args.model_root).glob(args.model_pattern))
    if len(parts) != args.expected_parts:
        raise RuntimeError(
            f"expected {args.expected_parts} target parts matching {args.model_pattern!r}, "
            f"found {len(parts)}"
        )
    return parts


def require_gguf_magic(path: Path) -> Path:
    """Execute require gguf magic operation."""
    with path.open("rb") as handle:
        magic = handle.read(4)
    if magic != b"GGUF":
        raise RuntimeError(f"invalid GGUF magic for {path}: {magic!r}")
    return path


def validate_projector_url(url: str) -> urllib.parse.ParseResult:
    """Validate projector url.

    Raises:
        ValidationError: If validation fails.
    """
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme != "https":
        raise RuntimeError(f"unsupported projector URL scheme: {parsed.scheme!r}")
    hostname = parsed.hostname
    if hostname is None or parsed.username is not None or parsed.password is not None:
        raise RuntimeError("projector URL must contain a trusted HTTPS host without credentials")
    hostname = hostname.rstrip(".").lower()
    if not any(
        hostname == suffix or hostname.endswith(f".{suffix}")
        for suffix in TRUSTED_PROJECTOR_HOST_SUFFIXES
    ):
        raise RuntimeError(f"untrusted projector URL host: {hostname!r}")
    if parsed.port not in (None, 443):
        raise RuntimeError(f"unsupported projector URL port: {parsed.port}")
    addresses = socket.getaddrinfo(hostname, 443, type=socket.SOCK_STREAM)
    if not addresses:
        raise RuntimeError(f"projector URL host did not resolve: {hostname!r}")
    for address in addresses:
        resolved = ipaddress.ip_address(address[4][0])
        if not resolved.is_global:
            raise RuntimeError(
                f"projector URL host resolved to a non-public address: {hostname!r}"
            )
    return parsed


class TrustedProjectorRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Represents TrustedProjectorRedirectHandler functionality."""
    def redirect_request(self, req, fp, code, msg, headers, newurl):  # noqa: ANN001
        """Execute redirect request operation."""
        validate_projector_url(newurl)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def copy_projector_response(response, output) -> None:  # noqa: ANN001
    """Execute copy projector response operation."""
    content_length = response.headers.get("Content-Length")
    if content_length is not None and int(content_length) > PROJECTOR_DOWNLOAD_MAX_BYTES:
        raise RuntimeError("projector download exceeds the maximum supported size")
    copied = 0
    while chunk := response.read(PROJECTOR_DOWNLOAD_CHUNK_BYTES):
        copied += len(chunk)
        if copied > PROJECTOR_DOWNLOAD_MAX_BYTES:
            raise RuntimeError("projector download exceeds the maximum supported size")
        output.write(chunk)


def projector_path(args: argparse.Namespace) -> Path:
    """Execute projector path operation."""
    if not args.projector_url:
        return require_gguf_magic(Path(args.projector))
    parsed = validate_projector_url(args.projector_url)
    target = Path(args.projector_local_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(f"{target.suffix}.part")
    redacted_url = urllib.parse.urlunparse(parsed._replace(query="", fragment=""))
    print(f"+ download {redacted_url} -> {target}", flush=True)
    opener = urllib.request.build_opener(TrustedProjectorRedirectHandler())
    try:
        with opener.open(
            args.projector_url,
            timeout=PROJECTOR_DOWNLOAD_TIMEOUT_SECONDS,
        ) as response, temporary.open("wb") as output:
            copy_projector_response(response, output)
        require_gguf_magic(temporary)
        temporary.replace(target)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return require_gguf_magic(target)


def run_report(command: list[str], report_out: str) -> None:
    """Run report operation."""
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
    """Execute certify operation."""
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
    """Parse and validate args.

    Returns:
        Parsed result.
    """
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
    """Execute main program logic.

    Returns:
        Exit code.
    """
    args = parse_args()
    ensure_build_tools()
    mesh_root = Path("/tmp/mesh-llm")
    checkout_mesh(args.mesh_repo, args.mesh_revision, mesh_root)
    certify(args, mesh_root)


if __name__ == "__main__":
    main()
