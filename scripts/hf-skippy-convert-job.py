# /// script
# requires-python = ">=3.11"
# dependencies = ["huggingface_hub[hf_xet]>=0.34"]
# ///

"""Run a native skippy-quantize HF checkpoint conversion on Hugging Face Jobs."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
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


def write_beta_card(artifact_dir: Path, source_repo: str, revision: str) -> None:
    card = f"""---
license: apache-2.0
base_model: {source_repo}
tags:
- gguf
- beta
- skippy
- mtp
---

# Inkling MTP sidecar (beta)

This is a public beta artifact for Skippy compatibility testing. It contains
Inkling's multi-token-prediction depths plus the shared embedding/output
context needed by distributed final stages. It is not a standalone chat model
and is not a promoted mesh-llm catalog entry.

Built with native `skippy-quantize` from mesh-llm revision `{revision}`.
"""
    (artifact_dir / "README.md").write_text(card, encoding="utf-8")


def convert(args: argparse.Namespace, root: Path) -> Path:
    binary = root / "target" / "release" / "skippy-quantize"
    run("just", "skippy-quantize-standalone-release-build", "cpu", cwd=root)
    work = Path(args.work_dir)
    target = work / "target"
    artifact_dir = target / args.target_prefix
    manifest = work / "convert-manifest.json"
    records = work / "records"
    spool = work / "spool"
    status = work / "status.json"
    work.mkdir(parents=True, exist_ok=True)
    run(
        str(binary),
        "convert-job",
        "--source",
        args.source,
        "--target",
        str(target),
        "--target-prefix",
        args.target_prefix,
        "--output-basename",
        args.output_basename,
        "--output-type",
        "bf16",
        "--expected-splits",
        str(args.expected_splits),
        "--window-size",
        "1",
        "--manifest",
        str(manifest),
        "--mtp",
        "--split-max-size",
        args.split_max_size,
        "--max-memory",
        args.max_memory,
        "--stream-buffer-bytes",
        "8388608",
        "--spool-dir",
        str(spool),
        "--record-dir",
        str(records),
        "--json-event-file",
        str(status),
        "--json-event-interval-seconds",
        "60",
        "--json-event-window",
        "8",
        "--watchdog-seconds",
        "300",
    )
    run(str(binary), "verify-job", "--manifest", str(manifest), "--json")
    shutil.copy2(manifest, artifact_dir / "skippy-convert-manifest.json")
    if status.exists():
        shutil.copy2(status, artifact_dir / "skippy-convert-status.json")
    write_beta_card(artifact_dir, args.source_repo, args.mesh_revision)
    return artifact_dir


def upload(args: argparse.Namespace, artifact_dir: Path) -> None:
    from huggingface_hub import HfApi

    api = HfApi(token=os.environ["HF_TOKEN"])
    api.create_repo(args.target_repo, repo_type="model", private=False, exist_ok=True)
    api.upload_folder(
        repo_id=args.target_repo,
        repo_type="model",
        folder_path=str(artifact_dir),
    )


def converted_artifact_dir(args: argparse.Namespace) -> Path:
    artifact_dir = Path(args.work_dir) / "target" / args.target_prefix
    if args.upload_only:
        required_files = ("README.md", "skippy-convert-manifest.json")
        missing = [name for name in required_files if not (artifact_dir / name).is_file()]
        if not artifact_dir.is_dir() or missing or not any(artifact_dir.glob("*.gguf")):
            details = f"; missing {', '.join(missing)}" if missing else ""
            raise FileNotFoundError(
                f"complete converted artifact not found: {artifact_dir}{details}"
            )
    return artifact_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="/mnt/checkpoint")
    parser.add_argument("--source-repo", required=True)
    parser.add_argument("--target-repo", required=True)
    parser.add_argument("--target-prefix", default="BF16")
    parser.add_argument("--output-basename", required=True)
    parser.add_argument("--expected-splits", type=int, default=1)
    parser.add_argument("--split-max-size", default="50G")
    parser.add_argument("--max-memory", default="24G")
    parser.add_argument("--work-dir", default="/data/skippy-convert")
    parser.add_argument("--mesh-repo", default="https://github.com/Mesh-LLM/mesh-llm.git")
    parser.add_argument("--mesh-revision", required=True)
    parser.add_argument(
        "--upload-only",
        action="store_true",
        help="publish an existing converted artifact without rebuilding or reconverting it",
    )
    parser.add_argument(
        "--xet-high-performance",
        action="store_true",
        help="enable Hugging Face Xet high-performance upload mode (use on hosts with at least 64 GB RAM)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.environ.setdefault("HF_HOME", str(Path(args.work_dir) / "hf-home"))
    # The work directory can be a mounted bucket. Xet's shard cache performs
    # poorly on network filesystems, so keep it on the Job's local SSD.
    os.environ.setdefault("HF_XET_CACHE", "/tmp/hf-xet")
    if args.xet_high_performance:
        os.environ.setdefault("HF_XET_HIGH_PERFORMANCE", "1")
    if args.upload_only:
        artifact_dir = converted_artifact_dir(args)
        upload(args, artifact_dir)
        print(f"published https://huggingface.co/{args.target_repo}", flush=True)
        return
    ensure_build_tools()
    mesh_root = Path("/tmp/mesh-llm")
    checkout_mesh(args.mesh_repo, args.mesh_revision, mesh_root)
    artifact_dir = convert(args, mesh_root)
    upload(args, artifact_dir)
    print(f"published https://huggingface.co/{args.target_repo}", flush=True)


if __name__ == "__main__":
    main()
