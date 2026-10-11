#!/usr/bin/env python3
"""Compare named tensor digests from a Skippy execution trace.

The trace must already have reproduced the recorded final tokens. Select the
monolithic decode block and every staged block for that same token explicitly;
this tool does not infer token alignment from execution order.
"""

import argparse
import json
import re
from pathlib import Path


TRACE = re.compile(
    r"stage=(-?\d+) tokens=(\d+) op=\S+ node=(\d+) name=(\S+) "
    r"type=(\S+) ne=\[([^]]*)\] nb=\[([^]]*)\] "
    r"contiguous=(\d+) nbytes=(\d+)"
)
LAYER = re.compile(r"l_out-(\d+)$")


def trace_blocks(path):
    blocks = []
    block = []
    with path.open(errors="replace") as source:
        for line in source:
            _, marker, record = line.partition("glm_dsa_tensor_trace ")
            if not marker:
                continue
            metadata = TRACE.match(record)
            if metadata is None:
                raise ValueError("unrecognized tensor trace record")
            _, digest_marker, digest_tail = record.partition(" stats=fnv64:")
            if not digest_marker:
                raise ValueError(f"missing tensor digest for {metadata.group(4)}")
            stage, tokens, node, name, dtype, shape, strides, contiguous, size = metadata.groups()
            tensor = {
                "stage": int(stage),
                "tokens": int(tokens),
                "node": int(node),
                "name": name,
                "type": dtype,
                "shape": shape,
                "strides": strides,
                "contiguous": bool(int(contiguous)),
                "nbytes": int(size),
                "digest": digest_tail.split(None, 1)[0],
            }
            if block and tensor["node"] <= block[-1]["node"]:
                blocks.append(block)
                block = []
            block.append(tensor)
    if block:
        blocks.append(block)
    return blocks


def named_tensors(blocks, indices):
    result = {}
    for index in indices:
        for tensor in blocks[index]:
            name = tensor["name"]
            if name in result:
                raise ValueError(f"duplicate tensor {name} in selected blocks")
            result[name] = tensor
    return result


def compare(baseline, staged):
    # Baseline graph order is the reference execution order. Append staged-only
    # names so a missing baseline callback is also reported.
    names = list(baseline) + [name for name in staged if name not in baseline]
    differences = []
    for name in names:
        left = baseline.get(name)
        right = staged.get(name)
        if left is None or right is None:
            differences.append({"name": name, "reason": "missing", "baseline": left, "staged": right})
            continue
        fields = ("tokens", "type", "shape", "strides", "contiguous", "nbytes", "digest")
        changed = [field for field in fields if left[field] != right[field]]
        if changed:
            differences.append({
                "name": name,
                "reason": "different",
                "fields": changed,
                "baseline_digest": left["digest"],
                "staged_digest": right["digest"],
            })
    first_layer = next((item for item in differences if LAYER.fullmatch(item["name"])), None)
    layer_names = [name for name in names if LAYER.fullmatch(name)]
    return {
        "status": "diverged" if differences else "match",
        "compared_names": len(names),
        "traced_layers": len(layer_names),
        "final_norm_traced": "result_norm" in baseline and "result_norm" in staged,
        "output_traced": "result_output" in baseline and "result_output" in staged,
        "divergent_names": len(differences),
        "first_divergent_tensor": differences[0] if differences else None,
        "first_divergent_layer": first_layer,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path)
    parser.add_argument("--list-blocks", action="store_true")
    parser.add_argument("--baseline-block", type=int)
    parser.add_argument("--staged-block", type=int, action="append")
    args = parser.parse_args()
    blocks = trace_blocks(args.trace)
    if args.list_blocks:
        print(json.dumps([
            {
                "index": index,
                "stage": block[0]["stage"],
                "tokens": block[0]["tokens"],
                "tensors": len(block),
                "first": block[0]["name"],
                "last": block[-1]["name"],
            }
            for index, block in enumerate(blocks)
        ], indent=2))
        return
    if args.baseline_block is None or not args.staged_block:
        parser.error("comparison requires --baseline-block and at least one --staged-block")
    try:
        baseline = named_tensors(blocks, [args.baseline_block])
        staged = named_tensors(blocks, args.staged_block)
    except IndexError as error:
        parser.error(f"block index outside 0..{len(blocks) - 1}: {error}")
    report = compare(baseline, staged)
    report.update({
        "trace": str(args.trace),
        "block_count": len(blocks),
        "baseline_block": args.baseline_block,
        "staged_blocks": args.staged_block,
    })
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
