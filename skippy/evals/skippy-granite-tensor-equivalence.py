#!/usr/bin/env python3
"""Verify Granite BF16 GGUF values against the pinned safetensors source.

Run this outside timed benchmark windows in an environment containing gguf,
numpy, safetensors, and torch. The output digest is over canonical source BF16
tensor names, shapes, and values after every GGUF tensor has been checked.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import gguf
import numpy as np
import torch
from safetensors import safe_open


def update_bytes(digest: Any, values: np.ndarray) -> None:
    view = np.ascontiguousarray(values).view(np.uint8).reshape(-1)
    block = 64 * 1024 * 1024
    for start in range(0, view.size, block):
        digest.update(memoryview(view[start : start + block]))


def equal_chunked(left: np.ndarray, right: np.ndarray) -> bool:
    left = left.reshape(-1)
    right = right.reshape(-1)
    if left.shape != right.shape:
        return False
    block = 8 * 1024 * 1024
    return all(
        np.array_equal(left[start : start + block], right[start : start + block])
        for start in range(0, left.size, block)
    )


def tensor_values(tensor: Any, shape: tuple[int, ...]) -> np.ndarray:
    if tensor.tensor_type == gguf.GGMLQuantizationType.BF16:
        return tensor.data.view(np.uint16).reshape(shape)
    if tensor.tensor_type == gguf.GGMLQuantizationType.F32:
        return tensor.data.reshape(shape)
    raise RuntimeError(f"unexpected GGUF type {tensor.tensor_type} for {tensor.name}")


def mapped_names(name_map: Any, key: str) -> list[str]:
    if key.endswith(".shared_mlp.input_linear.weight"):
        layer = int(key.split(".")[2])
        return [f"blk.{layer}.ffn_gate.weight", f"blk.{layer}.ffn_up.weight"]
    if key.endswith(".shared_mlp.output_linear.weight"):
        layer = int(key.split(".")[2])
        return [f"blk.{layer}.ffn_down.weight"]
    if key.endswith(".mamba.dt_bias"):
        layer = int(key.split(".")[2])
        return [f"blk.{layer}.ssm_dt.bias"]
    base, suffix = (
        key.rsplit(".", 1)
        if key.rsplit(".", 1)[-1] in {"weight", "bias"}
        else (key, "")
    )
    mapped = name_map.get_name(base, (suffix,) if suffix else ())
    if mapped is None:
        raise RuntimeError(f"no GGUF mapping for {key}")
    return list(dict.fromkeys([mapped, f"{mapped}.{suffix}" if suffix else mapped]))


def converted_values(key: str, source: np.ndarray, target: Any) -> np.ndarray:
    if key.endswith((".self_attn.q_proj.weight", ".self_attn.k_proj.weight")):
        head_count = 12 if key.endswith(".q_proj.weight") else 4
        return (
            source.reshape(
                head_count,
                2,
                source.shape[0] // head_count // 2,
                *source.shape[1:],
            )
            .swapaxes(1, 2)
            .reshape(source.shape)
        )
    bf16 = torch.from_numpy(source).view(torch.bfloat16)
    if key.endswith(".mamba.A_log"):
        return -torch.exp(bf16.float()).numpy()
    if target.tensor_type == gguf.GGMLQuantizationType.F32:
        return bf16.float().numpy()
    return source


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gguf", type=Path, required=True)
    parser.add_argument("--safetensors", type=Path, required=True)
    args = parser.parse_args()

    reader = gguf.GGUFReader(str(args.gguf))
    gguf_tensors = {tensor.name: tensor for tensor in reader.tensors}
    name_map = gguf.get_tensor_name_map(gguf.MODEL_ARCH.GRANITE_HYBRID, 40)
    consumed: set[str] = set()
    canonical = hashlib.sha256()
    count = 0

    with safe_open(args.safetensors, framework="pt", device="cpu") as source:
        for key in sorted(source.keys()):
            value = source.get_tensor(key).contiguous()
            if value.dtype != torch.bfloat16:
                raise RuntimeError(f"unexpected safetensors dtype {value.dtype} for {key}")
            source_bf16 = value.view(torch.uint16).numpy()
            candidates = mapped_names(name_map, key)
            names = list(
                dict.fromkeys(name for name in candidates if name in gguf_tensors)
            )
            if key.endswith(".shared_mlp.input_linear.weight"):
                names = candidates
            if not names or any(name not in gguf_tensors for name in names):
                raise RuntimeError(f"missing GGUF tensor for {key}: {candidates}")
            source_parts = (
                list(np.split(source_bf16, 2, axis=0))
                if len(names) == 2
                else [source_bf16]
            )
            for name, source_part in zip(names, source_parts, strict=True):
                target = gguf_tensors[name]
                expected = converted_values(key, source_part, target)
                actual = tensor_values(target, tuple(expected.shape))
                equal = (
                    bool(np.allclose(actual, expected, rtol=0.0, atol=5e-7))
                    if key.endswith(".mamba.A_log")
                    else equal_chunked(actual, expected)
                )
                if not equal:
                    raise RuntimeError(f"tensor values differ: {key} -> {name}")
                consumed.add(name)

            key_bytes = key.encode()
            canonical.update(len(key_bytes).to_bytes(4, "big"))
            canonical.update(key_bytes)
            shape_bytes = json.dumps(list(source_bf16.shape), separators=(",", ":")).encode()
            canonical.update(len(shape_bytes).to_bytes(4, "big"))
            canonical.update(shape_bytes)
            update_bytes(canonical, source_bf16)
            count += 1

    remaining = sorted(set(gguf_tensors) - consumed)
    if remaining:
        raise RuntimeError(f"unmatched GGUF tensors ({len(remaining)}): {remaining[:20]}")
    print(
        json.dumps(
            {
                "canonical_bf16_tensor_sha256": canonical.hexdigest(),
                "gguf_tensor_count": len(gguf_tensors),
                "safetensors_tensor_count": count,
                "values_equal": True,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
