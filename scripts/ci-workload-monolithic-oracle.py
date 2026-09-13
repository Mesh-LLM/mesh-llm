#!/usr/bin/env python3
"""Compare one Skippy non-chat response with pinned llama.cpp full-model serving.

Only classes whose OpenAI request and result can be aligned with llama-server
are supported here. This is a numerical/output parity gate, not a general
model-quality benchmark. Both servers must load the same immutable GGUF on CPU.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
import urllib.request

from workload_fixtures import (
    EMBEDDING_INPUTS,
    ENCODER_DECODER_PROMPT,
    RERANK_DOCUMENTS,
    RERANK_QUERY,
)


EMBEDDING_MAX_ABS_DELTA = 1e-4
EMBEDDING_MIN_COSINE = 0.99999
RERANK_MAX_ABS_DELTA = 1e-4


def request_json(base_url: str, path: str, payload: dict[str, object]) -> dict:
    request = urllib.request.Request(
        f"{base_url}{path}",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=240) as response:
        if response.headers.get_content_type() != "application/json":
            raise RuntimeError(f"{base_url}{path} returned non-JSON content")
        result = json.load(response)
    if not isinstance(result, dict):
        raise RuntimeError(f"{base_url}{path} returned a non-object response")
    return result


def vectors(response: dict, expected_count: int) -> list[list[float]]:
    """Validate exact embedding cardinality, indexes, and finite nonempty vectors."""
    rows = response.get("data")
    if not isinstance(rows, list) or len(rows) != expected_count:
        raise RuntimeError("embedding oracle response has the wrong batch size")
    result = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict) or row.get("index") != index:
            raise RuntimeError("embedding oracle response has invalid indexes")
        vector = row.get("embedding")
        if not isinstance(vector, list) or not vector:
            raise RuntimeError("embedding oracle response has no vector")
        if not all(type(value) in (int, float) and math.isfinite(value) for value in vector):
            raise RuntimeError("embedding oracle response has non-finite values")
        result.append(vector)
    return result


def compare_embeddings(candidate: dict, reference: dict, expected_count: int = len(EMBEDDING_INPUTS)) -> str:
    """Require equal dimensions and bounded numeric disagreement for every input."""
    candidate_vectors = vectors(candidate, expected_count)
    reference_vectors = vectors(reference, expected_count)
    max_delta = 0.0
    min_cosine = 1.0
    row_metrics = []
    for index, (candidate_vector, reference_vector) in enumerate(zip(
        candidate_vectors, reference_vectors, strict=True
    )):
        if len(candidate_vector) != len(reference_vector):
            raise RuntimeError("embedding dimensions differ from monolithic reference")
        row_delta = max(abs(left - right) for left, right in zip(candidate_vector, reference_vector))
        max_delta = max(max_delta, row_delta)
        dot = sum(left * right for left, right in zip(candidate_vector, reference_vector))
        left_norm = math.sqrt(sum(value * value for value in candidate_vector))
        right_norm = math.sqrt(sum(value * value for value in reference_vector))
        if left_norm == 0 or right_norm == 0:
            raise RuntimeError("embedding oracle response has a zero vector")
        cosine = dot / (left_norm * right_norm)
        min_cosine = min(min_cosine, cosine)
        row_metrics.append(f"{index}:delta={row_delta:.7g},cos={cosine:.8g}")
    if max_delta > EMBEDDING_MAX_ABS_DELTA or min_cosine < EMBEDDING_MIN_COSINE:
        raise RuntimeError(
            "embedding differs from monolithic reference: "
            f"max_abs_delta={max_delta:.7g}, min_cosine={min_cosine:.8g}, "
            f"rows=[{'; '.join(row_metrics)}], "
            f"candidate_head={candidate_vectors[0][:5]}, reference_head={reference_vectors[0][:5]}, "
            f"candidate_usage={candidate.get('usage')}, reference_usage={reference.get('usage')}"
        )
    return f"max_abs_delta={max_delta:.7g}, min_cosine={min_cosine:.8g}"


def run_embedding_oracle(candidate_url: str, oracle_url: str, model: str) -> str:
    """Compare the shared text fixture in both batched and single-input execution."""
    payload = {"model": model, "input": list(EMBEDDING_INPUTS), "encoding_format": "float"}
    candidate = request_json(candidate_url, "/embeddings", payload)
    reference = request_json(oracle_url, "/embeddings", payload)
    failures = []
    try:
        batch_detail = compare_embeddings(candidate, reference)
    except RuntimeError as error:
        failures.append(f"batched: {error}")
        batch_detail = "failed"
    single_details = []
    for index, text in enumerate(EMBEDDING_INPUTS):
        single_payload = {"model": model, "input": text, "encoding_format": "float"}
        candidate = request_json(candidate_url, "/embeddings", single_payload)
        reference = request_json(oracle_url, "/embeddings", single_payload)
        try:
            single_details.append(compare_embeddings(candidate, reference, expected_count=1))
        except RuntimeError as error:
            failures.append(f"single[{index}]: {error}")
    if failures:
        raise RuntimeError("; ".join(failures))
    return f"batch {batch_detail}; singles {', '.join(single_details)}"


def indexed_scores(response: dict) -> dict[int, float]:
    """Reject missing, duplicate, or invalid rerank document indexes and scores."""
    rows = response.get("results")
    if not isinstance(rows, list) or len(rows) != len(RERANK_DOCUMENTS):
        raise RuntimeError("rerank oracle response has the wrong document count")
    scores = {}
    for row in rows:
        if not isinstance(row, dict):
            raise RuntimeError("rerank oracle response has an invalid row")
        index = row.get("index")
        score = row.get("relevance_score")
        if type(index) is not int or index not in range(len(RERANK_DOCUMENTS)):
            raise RuntimeError("rerank oracle response has an invalid document index")
        if index in scores or type(score) not in (int, float) or not math.isfinite(score):
            raise RuntimeError("rerank oracle response has a duplicate or non-finite score")
        scores[index] = float(score)
    return scores


def compare_rerank(candidate: dict, reference: dict) -> str:
    """Require matching relevance order and bounded score differences per document."""
    candidate_scores = indexed_scores(candidate)
    reference_scores = indexed_scores(reference)
    max_delta = max(
        abs(candidate_scores[index] - reference_scores[index])
        for index in range(len(RERANK_DOCUMENTS))
    )
    candidate_order = sorted(candidate_scores, key=lambda index: -candidate_scores[index])
    reference_order = sorted(reference_scores, key=lambda index: -reference_scores[index])
    if candidate_order != reference_order or max_delta > RERANK_MAX_ABS_DELTA:
        raise RuntimeError(
            "rerank differs from monolithic reference: "
            f"candidate_order={candidate_order}, reference_order={reference_order}, "
            f"max_abs_delta={max_delta:.7g}"
        )
    return f"max_abs_delta={max_delta:.7g}, order={candidate_order}"


def completion_text(response: dict) -> str:
    """Extract exactly one nonempty completion and normalize only its whitespace."""
    choices = response.get("choices")
    if not isinstance(choices, list) or len(choices) != 1:
        raise RuntimeError("encoder-decoder oracle response has invalid choices")
    text = choices[0].get("text") if isinstance(choices[0], dict) else None
    if not isinstance(text, str) or not text.strip():
        raise RuntimeError("encoder-decoder oracle response has no text")
    return re.sub(r"\s+", " ", text).strip()


def compare_encoder_decoder(candidate: dict, reference: dict) -> str:
    """Require exact normalized completion parity with the independent reference."""
    candidate_text = completion_text(candidate)
    reference_text = completion_text(reference)
    if candidate_text != reference_text:
        raise RuntimeError(
            "encoder-decoder text differs from monolithic reference: "
            f"candidate={candidate_text!r}, reference={reference_text!r}"
        )
    return f"identical normalized text={candidate_text!r}"


def monolithic_completion(oracle_cli: str, model_path: str) -> dict:
    command = [
        oracle_cli, "-m", model_path, "-p", ENCODER_DECODER_PROMPT,
        "-n", "32", "-c", "0", "-b", "2048", "-ub", "2048", "-ngl", "0",
        "-s", "1", "--temp", "0", "--no-repack",
        "--no-display-prompt", "--simple-io",
    ]
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=240, check=False)
    except subprocess.TimeoutExpired as error:
        raise RuntimeError("monolithic encoder-decoder completion timed out") from error
    if result.returncode != 0:
        raise RuntimeError(
            f"monolithic encoder-decoder completion exited {result.returncode}: "
            f"{result.stderr[-1000:]}"
        )
    # llama-completion prints this terminal marker after the model emits EOG;
    # it is runner metadata, not generated model text.
    text = re.sub(r"\s*\[end of text\]\s*$", "", result.stdout).strip()
    if not text:
        raise RuntimeError("monolithic encoder-decoder completion produced no text")
    return {"choices": [{"text": text}]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-url", required=True)
    parser.add_argument("--oracle-url")
    parser.add_argument("--oracle-completion")
    parser.add_argument("--model-path")
    parser.add_argument("--model", required=True)
    parser.add_argument("--class", dest="model_class", required=True,
                        choices=("embedding", "rerank", "encoder_decoder"))
    args = parser.parse_args()

    if args.model_class == "encoder_decoder":
        if not args.oracle_completion or not args.model_path or args.oracle_url:
            parser.error("encoder_decoder requires --oracle-completion and --model-path only")
        payload = {"model": args.model, "prompt": ENCODER_DECODER_PROMPT,
                   "max_tokens": 32, "temperature": 0.0, "seed": 1}
        candidate = request_json(args.candidate_url, "/completions", payload)
        reference = monolithic_completion(args.oracle_completion, args.model_path)
        detail = compare_encoder_decoder(candidate, reference)
        print(f"encoder_decoder local-monolithic oracle passed: {detail}")
        return
    if not args.oracle_url or args.oracle_completion or args.model_path:
        parser.error("embedding/rerank require --oracle-url only")
    if args.model_class == "embedding":
        detail = run_embedding_oracle(args.candidate_url, args.oracle_url, args.model)
        print(f"embedding local-monolithic oracle passed: {detail}")
        return

    requests = {
        "rerank": (
            "/rerank",
            {"model": args.model, "query": RERANK_QUERY,
             "documents": list(RERANK_DOCUMENTS), "return_documents": True},
            compare_rerank,
        ),
    }
    path, payload, comparator = requests[args.model_class]
    candidate = request_json(args.candidate_url, path, payload)
    reference = request_json(args.oracle_url, path, payload)
    detail = comparator(candidate, reference)
    print(f"{args.model_class} local-monolithic oracle passed: {detail}")


if __name__ == "__main__":
    main()
