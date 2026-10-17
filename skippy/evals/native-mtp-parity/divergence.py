"""For each prompt where speculation changed the greedy text, replay the shared prefix
through single-token decode, as unspeculated generation runs it, and report the trunk's
logit gap between the two tokens at the first divergence.

A small gap shows the divergence starts at a near-tie of the target, which batched
verification can resolve differently from single-token decode. A large gap is a reason
to inspect the verify window's logits and acceptance for that position; it is not proof
of a wrong commit, because this replay does not see the batched logits the speculative
run committed from.

Rows that record `token_ids` are compared by those IDs. Otherwise the returned text is
re-tokenized, which can differ from the generated sequence, and the result says so."""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np

PROBE = Path(__file__).resolve().parent / "build" / "margin_probe"


def chat_messages(messages: list[dict]) -> list[dict]:
    """Tool-call arguments as mappings, as Qwen3-Coder's chat template expects."""
    out = []
    for m in messages:
        m = {**m, "content": m.get("content") or ""}
        if m.get("tool_calls"):
            calls = []
            for c in m["tool_calls"]:
                args = c["function"].get("arguments") or "{}"
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except json.JSONDecodeError:
                        args = {"raw": args}
                calls.append({**c, "function": {**c["function"], "arguments": args}})
            m["tool_calls"] = calls
        out.append(m)
    return out


def generated_ids(tok, row: dict, content: str) -> tuple[list[int], str]:
    """The generated token IDs when the bench recorded them, else the re-tokenized text."""
    if row.get("token_ids"):
        return list(row["token_ids"]), "generated"
    return tok(content, add_special_tokens=False)["input_ids"], "retokenized"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--off", type=Path, required=True, help="bench outputs.json with speculation off")
    ap.add_argument("--spec", type=Path, required=True, help="bench outputs.json with speculation on")
    ap.add_argument("--prompts", type=Path, required=True, help="json list of {id, messages, tools?}")
    ap.add_argument("--work", type=Path, required=True)
    ap.add_argument("--near-tie", type=float, default=0.5,
                    help="serial logit gap below which a divergence is reported as a near-tie")
    a = ap.parse_args()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.tokenizer)
    a.work.mkdir(parents=True, exist_ok=True)
    prompts = {p["id"]: p for p in json.loads(a.prompts.read_text())}
    off = {r["id"]: r for r in json.loads(a.off.read_text())}
    report = []
    for r in json.loads(a.spec.read_text()):
        o = off.get(r["id"])
        if o is None or o["text"] == r["text"]:
            continue
        content_off, content_spec = json.loads(o["text"])[0] or "", json.loads(r["text"])[0] or ""
        p = prompts[r["id"]]
        prompt = tok.apply_chat_template(chat_messages(p["messages"]), tools=p.get("tools") or None,
                                         tokenize=True, add_generation_prompt=True)
        t_off, source = generated_ids(tok, o, content_off)
        t_spec, spec_source = generated_ids(tok, r, content_spec)
        if spec_source != source:
            source = "retokenized"
        k = 0
        while k < min(len(t_off), len(t_spec)) and t_off[k] == t_spec[k]:
            k += 1
        if k >= min(len(t_off), len(t_spec)):
            report.append({"id": r["id"], "token_source": source,
                           "verdict": "inconclusive: the sequences differ only in length"})
            continue
        ids = np.asarray(prompt + t_off[:k], dtype=np.int32)
        ids.tofile(a.work / "tokens.i32")
        out = subprocess.run([str(PROBE), a.gguf, str(a.work / "tokens.i32"), str(len(prompt)),
                              str(t_off[k]), str(t_spec[k])], capture_output=True, text=True, check=True)
        m = json.loads(out.stdout.strip().splitlines()[-1])
        gap = m["logit_a"] - m["logit_b"]
        report.append({"id": r["id"], "token_source": source, "token_index": k,
                       "off_token": tok.decode([t_off[k]]), "spec_token": tok.decode([t_spec[k]]),
                       "serial_argmax_is_off": m["argmax"] == t_off[k],
                       "serial_logit_gap_off_minus_spec": round(gap, 4),
                       "verdict": "serial near-tie" if abs(gap) < a.near_tie
                       else "inspect the verify window at this position"})
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
