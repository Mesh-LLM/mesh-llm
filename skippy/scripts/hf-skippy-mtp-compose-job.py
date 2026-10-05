# /// script
# requires-python = ">=3.11"
# dependencies = ["huggingface_hub[hf_xet]>=0.34"]
# ///

"""Compose the Nemotron 3 Super UD-Q4_K_XL target with NVIDIA's BF16 MTPv2 head
into a single composite GGUF on a Hugging Face Job (CPU-only flavor).

Pipeline (stream-publish, per the agreed plan — peak local disk is one shard
in flight plus the MTP draft):

1. Clone mesh-llm at a pinned revision; `scripts/prepare-llama.sh pinned`
   materializes the patched llama.cpp checkout (converter only — no compile).
2. Convert the MTPv2 head with the pinned llama.cpp converter
   (`convert_hf_to_gguf.py --mtp --outtype bf16`). The pinned converter fully
   supports the Nemotron-H MoE MTP head: it folds `mtp.layers.0` (attention +
   eh_proj/enorm/hnorm) and `mtp.layers.1` (MoE + final_layernorm) into a
   single `blk.<block_count>` NextN block and emits
   `nemotron_h_moe.nextn_predict_layers=1`.
3. Build `skippy-quantize` and run `compose-mtp` against the target shards:
   - `--metadata-shard` rewrites shard 1 (the only shard carrying the global
     KV: block_count bump 88→89, `nemotron_h_moe.nextn_predict_layers=1`);
   - the last shard is recomposed with the MTP tensors appended past the
     target data, byte-copied verbatim;
   - the draft's tensors already sit at `blk.88`, so no rename happens.
4. Probe the composite with `validate-mtp-attach` (full smoke stays on the
   Studios where the split runtime lives).
5. Stream-publish: shard 1' and the untouched middle shard upload directly
   from the read-only mounts under the composite basename; the recomposed
   last shard uploads from local scratch.

See PLANS/NEMOTRON_MTPV2_SUPERQ4_PACKAGE.md (Buzz workspace) and
Mesh-LLM/mesh-llm#1425.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

TARGET_SUBDIR = "UD-Q4_K_XL"
ARCH = "nemotron_h_moe"


def run(*command: str, cwd: Path | None = None) -> subprocess.CompletedProcess:
    print("+", " ".join(command), flush=True)
    return subprocess.run(command, cwd=cwd, check=True)


def log_stage(name: str, *paths: Path) -> None:
    """Emit a stage marker plus disk usage for the writable work area and the
    read-only model mounts. A previous run died with SIGBUS mid-conversion;
    these lines make disk pressure vs mount stalls diagnosable from logs."""
    print(f"=== stage: {name} ===", flush=True)
    mounts = ["/data", "/tmp", "/mnt/target-src", "/mnt/mtpv2"]
    run("sh", "-c", "df -h " + " ".join(mounts) + " || true")
    for path in paths:
        if path.exists():
            size = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
            print(f"    {path}: {size / 2**30:.2f} GiB", flush=True)


def ensure_build_tools() -> None:
    required = ("git", "curl", "cmake", "c++")
    if any(shutil.which(tool) is None for tool in required) or shutil.which("ld.lld") is None:
        run("apt-get", "update")
        run("apt-get", "install", "-y", "build-essential", "cmake", "curl", "git", "pkg-config", "lld")
    if shutil.which("cargo") is None:
        run("sh", "-c", "curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y")
        os.environ["PATH"] = f"{Path.home() / '.cargo' / 'bin'}:{os.environ['PATH']}"
    if shutil.which("just") is None:
        run("cargo", "install", "just", "--locked")


def pip_install(*args: str) -> None:
    """Install into the *running* interpreter.

    `uv run` executes this script in an ephemeral venv whose `python3` may not
    be the `pip` on PATH (and uv venvs ship without pip), so resolve every
    install and invocation through sys.executable.
    """
    if shutil.which("uv") is not None:
        run("uv", "pip", "install", "--python", sys.executable, *args)
        return
    run(sys.executable, "-m", "ensurepip", "--upgrade")
    run(sys.executable, "-m", "pip", "install", *args)


def ensure_python_deps(llama_root: Path | None = None) -> None:
    # torch is only needed for the converter; CPU wheel keeps the image small.
    pip_install("torch", "--index-url", "https://download.pytorch.org/whl/cpu")
    if llama_root is not None:
        # The pinned converter also needs gguf/sentencepiece/etc. from its own
        # requirements file; torch above already satisfies its torch entry.
        reqs = llama_root / "requirements" / "requirements-convert_hf_to_gguf.txt"
        pip_install("-r", str(reqs))


def prepare_repos(args: argparse.Namespace) -> tuple[Path, Path]:
    mesh_root = Path("/tmp/mesh-llm")
    if mesh_root.exists():
        shutil.rmtree(mesh_root)
    run("git", "clone", "--filter=blob:none",
        "https://github.com/Mesh-LLM/mesh-llm.git", str(mesh_root))
    run("git", "checkout", args.mesh_revision, cwd=mesh_root)
    run(str(mesh_root / "scripts" / "prepare-llama.sh"), "pinned", cwd=mesh_root)
    return mesh_root, mesh_root / ".deps" / "llama.cpp"


def build_skippy_quantize(mesh_root: Path) -> Path:
    run("just", "skippy-quantize-standalone-release-build", "cpu", cwd=mesh_root)
    return mesh_root / "target" / "release" / "skippy-quantize"


def stage_converter_input(args: argparse.Namespace, work: Path) -> Path:
    """Build a converter-readable model dir for the MTP head.

    The MTPv2 repo ships weights + config only — no tokenizer — so the
    tokenizer files are pulled from the full base-model repo. The weights are
    downloaded to local disk rather than read through the read-only model
    mount: two consecutive jobs stalled at exactly 138M/5.88G while the
    converter read mmap'd safetensors off /mnt/mtpv2 (first SIGBUS, then a
    hang until the job timeout), so the mount is not reliable for bulk reads.
    """
    from huggingface_hub import hf_hub_download, snapshot_download

    staged = work / "mtp-src"
    if staged.exists():
        shutil.rmtree(staged)
    staged.mkdir(parents=True, exist_ok=True)
    mount_repo = os.environ.get("MTP_MOUNT_REPO", "nvidia/Nemotron-3-Super-120B-A12B-BF16-MTPv2")
    snapshot_download(
        mount_repo,
        local_dir=str(staged),
        allow_patterns=["*.safetensors", "*.json"],
        token=os.environ.get("HF_TOKEN"),
    )
    for filename in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"):
        path = hf_hub_download(args.tokenizer_source, filename, token=os.environ.get("HF_TOKEN"))
        dst = staged / filename
        if not dst.exists():
            dst.symlink_to(path)
    return staged


def convert_mtp(llama_root: Path, args: argparse.Namespace, work: Path) -> Path:
    """MTPv2 SafeTensors -> standalone MTP draft GGUF via the pinned converter.

    The converter positions the folded MTP block at blk.<num_hidden_layers>
    (88 for this checkpoint) and keeps only head tensors; the target shards
    provide embeddings/output.
    """
    out_dir = work / "mtp"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_gguf = out_dir / "mtp-nemotron-mtpv2.gguf"
    staged = stage_converter_input(args, work)
    run(
        sys.executable, str(llama_root / "convert_hf_to_gguf.py"),
        str(staged),
        "--mtp",
        "--outtype", "bf16",
        "--outfile", str(out_gguf),
    )
    if not out_gguf.is_file():
        raise FileNotFoundError(f"converter did not produce {out_gguf}")
    return out_gguf


def target_shards(args: argparse.Namespace) -> list[Path]:
    shard_dir = Path(args.target_source) / TARGET_SUBDIR
    shards = sorted(p for p in shard_dir.glob("*.gguf") if p.name.startswith(args.target_basename))
    if len(shards) < 2:
        raise FileNotFoundError(f"expected sharded target GGUF under {shard_dir}, found {shards}")
    return shards


def compose(binary: Path, args: argparse.Namespace, work: Path, mtp_gguf: Path,
            shards: list[Path]) -> tuple[Path, Path]:
    first, last = shards[0], shards[-1]
    composite_dir = work / "composite"
    composite_dir.mkdir(parents=True, exist_ok=True)
    out_first = composite_dir / first.name.replace(args.target_basename, args.composite_basename)
    out_last = composite_dir / last.name.replace(args.target_basename, args.composite_basename)
    run(
        str(binary), "compose-mtp",
        "--target-shard", str(last),
        "--mtp-gguf", str(mtp_gguf),
        "--output", str(out_last),
        "--mtp-block", str(args.mtp_block),
        "--metadata-shard", str(first),
        "--metadata-output", str(out_first),
        "--set-kv", f"{ARCH}.nextn_predict_layers=1",
        "--json",
    )
    # Probe: open the composite (middle shard still on the read-only mount)
    # and exercise the native attach_mtp_draft_model path for this draft.
    # Full smoke/bench stays on the Studios.
    model_parts = [str(out_first), *(str(s) for s in shards[1:-1]), str(out_last)]
    run(
        str(binary), "validate-mtp-attach",
        *sum((["--model", part] for part in model_parts), []),
        "--mtp-draft", str(mtp_gguf),
        "--layer-count", str(args.mtp_block + 1),
        "--mtp-layer-count", "1",
    )
    return out_first, out_last


def upload(repo_id: str, local: Path, remote_name: str) -> None:
    from huggingface_hub import HfApi

    api = HfApi(token=os.environ["HF_TOKEN"])
    api.create_repo(repo_id, repo_type="model", private=False, exist_ok=True)
    api.upload_file(repo_id=repo_id, repo_type="model",
                    path_or_fileobj=str(local), path_in_repo=remote_name)
    print(f"published {repo_id}/{remote_name}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-source", default="/mnt/target-src")
    parser.add_argument("--mtp-source", default="/mnt/mtpv2")
    parser.add_argument("--tokenizer-source",
                        default="nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16")
    parser.add_argument("--composite-repo",
                        default="meshllm/NVIDIA-Nemotron-3-Super-120B-A12B-UD-Q4_K_XL-MTPv2-GGUF")
    parser.add_argument("--target-basename", default="NVIDIA-Nemotron-3-Super-120B-A12B-UD-Q4_K_XL")
    parser.add_argument("--composite-basename",
                        default="NVIDIA-Nemotron-3-Super-120B-A12B-UD-Q4_K_XL-MTPv2")
    parser.add_argument("--mtp-block", type=int, default=88)
    # The bucket mount (/data) stalls bulk writes (job 8 froze at 136M/5.88G
    # with the writer blocked in FUSE); keep heavy I/O on the container overlay.
    parser.add_argument("--work-dir", default="/tmp/mtp-compose")
    parser.add_argument("--mesh-revision", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if "HF_TOKEN" not in os.environ:
        print("error: HF_TOKEN environment variable is required for upload", file=sys.stderr)
        sys.exit(1)
    os.environ.setdefault("HF_HOME", str(Path(args.work_dir) / "hf-home"))
    os.environ.setdefault("HF_XET_CACHE", "/tmp/hf-xet")
    ensure_build_tools()
    mesh_root, llama_root = prepare_repos(args)
    ensure_python_deps(llama_root)
    binary = build_skippy_quantize(mesh_root)
    work = Path(args.work_dir)
    work.mkdir(parents=True, exist_ok=True)
    log_stage("convert-mtp", work)
    mtp_gguf = convert_mtp(llama_root, args, work)
    shards = target_shards(args)
    log_stage("compose-mtp", work)
    out_first, out_last = compose(binary, args, work, mtp_gguf, shards)
    log_stage("publish", work)
    # Stream-publish: patched metadata shard, untouched middle shards (straight
    # from the read-only mount), then the recomposed last shard.
    upload(args.composite_repo, out_first, out_first.name)
    for shard in shards[1:-1]:
        upload(args.composite_repo, shard,
               shard.name.replace(args.target_basename, args.composite_basename))
    upload(args.composite_repo, out_last, out_last.name)
    print(f"composite complete: https://huggingface.co/{args.composite_repo}", flush=True)


if __name__ == "__main__":
    main()
