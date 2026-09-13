#!/usr/bin/env python3
"""Exercise non-chat workloads through the real OpenAI HTTP frontend.

This is a protocol and runtime smoke, not an independent model-quality oracle.
The pinned-model in-process tests additionally check deterministic execution and
fail-closed staging for each workload class.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import math
from pathlib import Path
import struct
import urllib.request
import wave

from workload_fixtures import (
    EMBEDDING_INPUTS,
    ENCODER_DECODER_PROMPT,
    RERANK_DOCUMENTS,
    RERANK_QUERY,
)


def request_json(base_url: str, path: str, payload: dict[str, object]) -> dict:
    request = urllib.request.Request(
        f"{base_url}{path}",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=240) as response:
        if response.headers.get_content_type() != "application/json":
            raise RuntimeError(f"{path} returned non-JSON content")
        result = json.load(response)
    if not isinstance(result, dict):
        raise RuntimeError(f"{path} returned a non-object response")
    return result


def request_bytes(base_url: str, path: str, payload: dict[str, object]) -> tuple[str, bytes]:
    request = urllib.request.Request(
        f"{base_url}{path}",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=240) as response:
        return response.headers.get_content_type(), response.read()


def request_multipart(base_url: str, path: str, model: str, media_path: Path) -> dict:
    boundary = "mesh-llm-workload-smoke"
    body = (
        f"--{boundary}\r\n"
        'Content-Disposition: form-data; name="model"\r\n\r\n'
        f"{model}\r\n"
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="file"; filename="{media_path.name}"\r\n'
        "Content-Type: audio/wav\r\n\r\n"
    ).encode("utf-8")
    body += media_path.read_bytes() + f"\r\n--{boundary}--\r\n".encode("ascii")
    request = urllib.request.Request(
        f"{base_url}{path}",
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=240) as response:
        if response.headers.get_content_type() != "application/json":
            raise RuntimeError(f"{path} returned non-JSON content")
        result = json.load(response)
    if not isinstance(result, dict):
        raise RuntimeError(f"{path} returned a non-object response")
    return result


def smoke_embedding(base_url: str, model: str) -> None:
    inputs = list(EMBEDDING_INPUTS)
    result = request_json(
        base_url,
        "/embeddings",
        {"model": model, "input": inputs, "encoding_format": "float"},
    )
    rows = result.get("data")
    if result.get("object") != "list" or result.get("model") != model:
        raise RuntimeError("embedding response has the wrong object or model")
    if not isinstance(rows, list) or len(rows) != len(inputs):
        raise RuntimeError("embedding response has the wrong batch size")
    vectors = []
    for index, row in enumerate(rows):
        values = row.get("embedding")
        if row.get("object") != "embedding" or row.get("index") != index:
            raise RuntimeError("embedding response has invalid item metadata")
        if not isinstance(values, list) or not values:
            raise RuntimeError("embedding response has no vector")
        if not all(isinstance(value, (int, float)) and math.isfinite(value) for value in values):
            raise RuntimeError("embedding response contains non-finite values")
        norm = math.sqrt(sum(value * value for value in values))
        if abs(norm - 1.0) > 1e-4:
            raise RuntimeError(f"embedding {index} is not normalized: {norm}")
        vectors.append(values)
    if len({len(vector) for vector in vectors}) != 1:
        raise RuntimeError("embedding dimensions differ within one response")
    related = sum(left * right for left, right in zip(vectors[0], vectors[1]))
    unrelated = sum(left * right for left, right in zip(vectors[0], vectors[2]))
    if related <= unrelated:
        raise RuntimeError(
            f"embedding placed unrelated text closer: related={related}, unrelated={unrelated}"
        )
    if result.get("usage", {}).get("prompt_tokens", 0) <= 0:
        raise RuntimeError("embedding response reported no prompt usage")
    encoded = request_json(
        base_url,
        "/embeddings",
        {"model": model, "input": inputs[0], "encoding_format": "base64"},
    )
    payload = encoded.get("data", [{}])[0].get("embedding")
    if not isinstance(payload, str):
        raise RuntimeError("base64 embedding is not a string")
    raw = base64.b64decode(payload, validate=True)
    if len(raw) != len(vectors[0]) * struct.calcsize("<f"):
        raise RuntimeError("base64 embedding has the wrong byte length")
    if not all(math.isfinite(value) for value in struct.unpack(f"<{len(vectors[0])}f", raw)):
        raise RuntimeError("base64 embedding contains a non-finite value")


def smoke_rerank(base_url: str, model: str) -> None:
    result = request_json(
        base_url,
        "/rerank",
        {
            "model": model,
            "query": RERANK_QUERY,
            "documents": list(RERANK_DOCUMENTS),
            "return_documents": True,
        },
    )
    rows = result.get("results")
    if not isinstance(rows, list) or len(rows) != 2:
        raise RuntimeError("rerank did not return both documents")
    if sorted(row.get("index") for row in rows) != [0, 1]:
        raise RuntimeError("rerank returned invalid document indexes")
    for row in rows:
        score = row.get("relevance_score")
        if not isinstance(score, (int, float)) or not math.isfinite(score):
            raise RuntimeError("rerank returned a non-finite score")
        if not isinstance(row.get("document"), str):
            raise RuntimeError("rerank omitted the requested document")
    scores = {row["index"]: row["relevance_score"] for row in rows}
    if scores[0] <= scores[1]:
        raise RuntimeError(f"rerank misplaced the relevant document: {scores}")
    if result.get("usage", {}).get("prompt_tokens", 0) <= 0:
        raise RuntimeError("rerank did not report prompt usage")


def smoke_encoder_decoder(base_url: str, model: str) -> None:
    result = request_json(
        base_url,
        "/completions",
        {
            "model": model,
            "prompt": ENCODER_DECODER_PROMPT,
            "max_tokens": 32,
            "temperature": 0.0,
        },
    )
    choices = result.get("choices")
    if not isinstance(choices, list) or len(choices) != 1:
        raise RuntimeError("encoder-decoder completion has invalid choices")
    if not isinstance(choices[0].get("text"), str) or not choices[0]["text"].strip():
        raise RuntimeError("encoder-decoder completion is empty")
    if "haus" not in choices[0]["text"].casefold():
        raise RuntimeError(f"encoder-decoder missed the translation anchor: {choices[0]['text']!r}")
    if result.get("usage", {}).get("completion_tokens", 0) <= 0:
        raise RuntimeError("encoder-decoder completion reported no generated tokens")


def smoke_ocr(base_url: str, model: str, media_path: Path) -> None:
    image = base64.b64encode(media_path.read_bytes()).decode("ascii")
    result = request_json(
        base_url,
        "/chat/completions",
        {
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Read all visible text. Return only the transcription."},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image}"}},
                    ],
                }
            ],
            "max_tokens": 32,
            "temperature": 0.0,
        },
    )
    choices = result.get("choices")
    if not isinstance(choices, list) or len(choices) != 1:
        raise RuntimeError("OCR chat completion has invalid choices")
    text = choices[0].get("message", {}).get("content")
    if not isinstance(text, str) or not text.strip():
        raise RuntimeError("OCR chat completion is empty")


def smoke_speech_synthesis(base_url: str, model: str) -> None:
    content_type, audio = request_bytes(
        base_url,
        "/audio/speech",
        {
            "model": model,
            "input": "The mesh is ready.",
            "voice": "default",
            "response_format": "wav",
        },
    )
    if content_type != "audio/wav":
        raise RuntimeError(f"speech synthesis returned {content_type}, not audio/wav")
    with wave.open(io.BytesIO(audio)) as sound:
        if sound.getnframes() < sound.getframerate() // 10:
            raise RuntimeError("speech synthesis returned too little audio")
        if sound.getnchannels() < 1 or sound.getsampwidth() != 2:
            raise RuntimeError("speech synthesis returned unsupported WAV samples")
        samples = sound.readframes(sound.getnframes())
    values = struct.unpack(f"<{len(samples) // 2}h", samples)
    rms = math.sqrt(sum(value * value for value in values) / len(values))
    if rms < 1.0:
        raise RuntimeError("speech synthesis returned silent audio")


def smoke_speech_recognition(base_url: str, model: str, media_path: Path) -> None:
    result = request_multipart(base_url, "/audio/transcriptions", model, media_path)
    text = result.get("text")
    if not isinstance(text, str) or not text.strip():
        raise RuntimeError("speech recognition returned no transcription")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--class",
        dest="model_class",
        required=True,
        choices=("embedding", "rerank", "encoder_decoder", "ocr", "speech_synthesis", "speech_recognition"),
    )
    parser.add_argument("--media-path", type=Path)
    args = parser.parse_args()

    if args.model_class in {"ocr", "speech_recognition"}:
        if args.media_path is None or not args.media_path.is_file():
            parser.error(f"{args.model_class} requires --media-path pointing to a file")
    checks = {
        "embedding": lambda: smoke_embedding(args.base_url, args.model),
        "rerank": lambda: smoke_rerank(args.base_url, args.model),
        "encoder_decoder": lambda: smoke_encoder_decoder(args.base_url, args.model),
        "ocr": lambda: smoke_ocr(args.base_url, args.model, args.media_path),
        "speech_synthesis": lambda: smoke_speech_synthesis(args.base_url, args.model),
        "speech_recognition": lambda: smoke_speech_recognition(args.base_url, args.model, args.media_path),
    }
    checks[args.model_class]()
    print(f"OpenAI HTTP {args.model_class} smoke passed: model={args.model}")


if __name__ == "__main__":
    main()
