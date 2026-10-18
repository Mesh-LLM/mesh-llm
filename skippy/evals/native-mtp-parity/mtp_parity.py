"""Compare a runtime's native MTP head against splitcoder.mtp.MTPHead.

The probe's own trunk hidden states are fed to the PyTorch head, so this
checks the MTP block alone: conversion (key mapping, proj half order, dense
FFN), graph (norm placement, RoPE, attention) and chaining (z' before norm).
The shared embedding and LM head are dequantized from the same GGUF the
runtime uses, so their quantization is identical on both sides.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

PROMPT = [
    {"role": "system", "content": "You are a coding agent. Use the tools to inspect and edit the repository."},
    {"role": "user", "content": "The test `test_parse_duration` fails with ValueError on inputs like '1h30m'. Fix it."},
    {"role": "assistant", "content": "", "tool_calls": [{"type": "function", "id": "call_1", "function": {
        "name": "read_file", "arguments": {"path": "src/timeparse.py"}}}]},
    {"role": "tool", "tool_call_id": "call_1", "content":
        "import re\n\nUNITS = {'h': 3600, 'm': 60, 's': 1}\n\ndef parse_duration(text):\n"
        "    total = 0\n    for amount, unit in re.findall(r'(\\d+)([hms])', text):\n"
        "        total += int(amount) * UNITS[unit]\n    if total == 0:\n        raise ValueError(text)\n"
        "    return total\n"},
]
TOOLS = [{"type": "function", "function": {"name": "read_file", "description": "Read a file",
          "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}}]


def gguf_matrix(gguf_path: Path, name: str) -> torch.Tensor:
    from gguf import GGUFReader
    from gguf.quants import dequantize
    for t in GGUFReader(gguf_path).tensors:
        if t.name == name:
            a = np.asarray(dequantize(t.data, t.tensor_type), dtype=np.float32)
            return torch.from_numpy(a.reshape([int(x) for x in reversed(t.shape)]).copy())
    raise KeyError(name)


def reference(head, embed, lm_head, h, next_ids, *, swap_proj=False):
    from transformers.models.qwen3.modeling_qwen3 import Qwen3RotaryEmbedding
    T = h.shape[0]
    pos = torch.arange(T)[None]
    pe = Qwen3RotaryEmbedding(config=head.config)(h[None], pos)
    emb = embed[next_ids][None]
    if swap_proj:  # negative control: the order bug the converter must not have
        d = head.proj.weight.shape[1] // 2
        w = head.proj.weight.data.clone()
        head.proj.weight.data = torch.cat((w[:, d:], w[:, :d]), dim=1)
    try:
        z = head(h[None], emb, pe)[0]
    finally:
        if swap_proj:
            head.proj.weight.data = w
    logits = head.norm(z) @ lm_head.T
    return logits, z


def compare(ref: torch.Tensor, got: torch.Tensor) -> dict:
    cos = torch.nn.functional.cosine_similarity(ref, got, dim=-1)
    lr, lg = torch.log_softmax(ref, -1), torch.log_softmax(got, -1)
    kl = (lr.exp() * (lr - lg)).sum(-1)
    top1 = (ref.argmax(-1) == got.argmax(-1)).float()
    top5 = torch.tensor([len(set(a.tolist()) & set(b.tolist())) / 5
                         for a, b in zip(ref.topk(5, -1).indices, got.topk(5, -1).indices)])
    rel = (ref - got).abs().max() / ref.abs().max()
    return {"rows": ref.shape[0], "cos_min": round(cos.min().item(), 6), "cos_mean": round(cos.mean().item(), 6),
            "kl_mean": float(f"{kl.mean().item():.3g}"), "kl_max": float(f"{kl.max().item():.3g}"),
            "top1_agree": round(top1.mean().item(), 4), "top5_overlap": round(top5.mean().item(), 4),
            "max_abs_rel": float(f"{rel.item():.3g}")}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", type=Path, required=True, help="integrated trunk+MTP GGUF the runtime loads")
    ap.add_argument("--head", type=Path, required=True, help="dir with mtp.safetensors + mtp_config.json")
    ap.add_argument("--tokenizer", type=Path, required=True)
    ap.add_argument("--probe", type=Path, default=Path(__file__).resolve().parent / "build" / "mtp_parity_probe")
    ap.add_argument("--work", type=Path, required=True)
    ap.add_argument("--min-cos", type=float, default=0.999)
    ap.add_argument("--min-top1", type=float, default=0.97)
    a = ap.parse_args()

    from transformers import AutoTokenizer
    from splitcoder.mtp import MTPHead

    a.work.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(a.tokenizer)
    ids = tok.apply_chat_template(PROMPT, tools=TOOLS, tokenize=True, add_generation_prompt=True)
    np.asarray(ids, dtype=np.int32).tofile(a.work / "tokens.i32")
    meta = json.loads(subprocess.run([str(a.probe), str(a.gguf), str(a.work / "tokens.i32"), str(a.work)],
                                     check=True, capture_output=True, text=True).stdout.strip().splitlines()[-1])
    T, d, V = meta["tokens"], meta["n_embd"], meta["n_vocab"]
    rows = lambda name, n, w: torch.from_numpy(np.fromfile(a.work / name, dtype=np.float32).reshape(n, w))
    h, logits1, z1, logits2 = rows("h.f32", T, d), rows("logits1.f32", T - 1, V), rows("z1.f32", T - 1, d), \
        rows("logits2.f32", T - 2, V)

    torch.set_grad_enabled(False)
    head = MTPHead.load(a.head, device="cpu", dtype=torch.float32).eval()
    embed = gguf_matrix(a.gguf, "token_embd.weight")
    lm_head = gguf_matrix(a.gguf, "output.weight")
    ids_t = torch.tensor(ids)

    ref1, ref_z1 = reference(head, embed, lm_head, h[:-1], ids_t[1:])
    ref2, _ = reference(head, embed, lm_head, ref_z1[:-1], ids_t[2:])
    bad1, _ = reference(head, embed, lm_head, h[:-1], ids_t[1:], swap_proj=True)
    report = {
        "tokens": T,
        "depth1_logits": compare(ref1, logits1),
        "depth1_chain_hidden_cos_min": round(torch.nn.functional.cosine_similarity(ref_z1, z1, dim=-1).min().item(), 6),
        "depth2_logits": compare(ref2, logits2),
        "negative_control_swapped_proj": compare(bad1, logits1),
    }
    print(json.dumps(report, indent=1))
    ok = all(report[k]["cos_min"] >= a.min_cos and report[k]["top1_agree"] >= a.min_top1
             for k in ("depth1_logits", "depth2_logits"))
    ok = ok and report["negative_control_swapped_proj"]["top1_agree"] < a.min_top1
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
