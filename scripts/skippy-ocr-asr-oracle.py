#!/usr/bin/env python3
"""Compare OCR or ASR output with pinned llama.cpp monolithic serving.

Both HTTP servers must load the same GGUF and projector on CPU. Matching text
is a local execution-parity check, not general model-quality certification.
OCR additionally checks the known text in the generated fixture. ASR requires
an independently labeled audio fixture before semantic accuracy can be claimed.
"""

from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path
import re
import unicodedata
import urllib.request


OCR_PROMPT = "Read all visible text. Return only the transcription."
OCR_FIXTURE_TEXT = "MESH 42"
ASR_PROMPT = "Transcribe audio to text"
ASR_MAX_TOKENS = 128
BOUNDARY = "mesh-llm-ocr-asr-oracle"
NON_TRANSCRIPT_PREFIXES = (
    "i can t fulfill",
    "i cannot fulfill",
    "i can help you with transcribing",
    "you can use various tools to transcribe",
)


def normalized_text(value: object, source: str) -> str:
    if not isinstance(value, str):
        raise RuntimeError(f"{source} returned no text")
    normalized = unicodedata.normalize("NFKC", value).casefold()
    normalized = re.sub(r"[^\w]+", " ", normalized, flags=re.UNICODE).strip()
    if not normalized:
        raise RuntimeError(f"{source} returned empty text")
    return normalized


def transcription_text(value: object, source: str) -> str:
    text = normalized_text(value, source)
    # The two frontends add different presentational labels around the same
    # transcript. Strip only these exact known prefixes, never content words.
    for prefix in ("the text is ", "the audio is "):
        if text.startswith(prefix):
            text = text[len(prefix):]
            break
    if text.startswith(NON_TRANSCRIPT_PREFIXES):
        raise RuntimeError(f"{source} returned a refusal or generic ASR advice, not a transcript")
    return text


def compare_text(candidate: object, reference: object, expected: str | None,
                 *, transcript: bool = False) -> str:
    normalizer = transcription_text if transcript else normalized_text
    candidate_text = normalizer(candidate, "candidate")
    reference_text = normalizer(reference, "monolithic reference")
    if candidate_text != reference_text:
        raise RuntimeError(
            "text differs from monolithic reference: "
            f"candidate={candidate_text!r}, reference={reference_text!r}"
        )
    if expected is not None:
        expected_text = normalized_text(expected, "fixture label")
        if expected_text != candidate_text:
            raise RuntimeError(
                "output does not exactly match independently known fixture text: "
                f"expected={expected_text!r}, actual={candidate_text!r}"
            )
        return f"identical normalized text exactly matching {expected_text!r}"
    return "identical normalized text; unlabeled fixture, no accuracy claim"


def request_json(base_url: str, path: str, payload: dict[str, object]) -> dict:
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}{path}",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    return response_json(request)


def request_multipart(base_url: str, path: str, model: str, media: bytes) -> dict:
    if BOUNDARY.encode("ascii") in media:
        raise RuntimeError("audio fixture collides with multipart boundary")
    body = (
        f"--{BOUNDARY}\r\n"
        'Content-Disposition: form-data; name="model"\r\n\r\n'
        f"{model}\r\n"
        f"--{BOUNDARY}\r\n"
        'Content-Disposition: form-data; name="response_format"\r\n\r\n'
        "json\r\n"
        f"--{BOUNDARY}\r\n"
        'Content-Disposition: form-data; name="temperature"\r\n\r\n'
        "0\r\n"
        f"--{BOUNDARY}\r\n"
        'Content-Disposition: form-data; name="file"; filename="oracle.wav"\r\n'
        "Content-Type: audio/wav\r\n\r\n"
    ).encode("utf-8") + media + f"\r\n--{BOUNDARY}--\r\n".encode("ascii")
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}{path}",
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={BOUNDARY}"},
        method="POST",
    )
    return response_json(request)


def response_json(request: urllib.request.Request) -> dict:
    with urllib.request.urlopen(request, timeout=240) as response:
        if response.headers.get_content_type() != "application/json":
            raise RuntimeError(f"{request.full_url} returned non-JSON content")
        result = json.load(response)
    if not isinstance(result, dict):
        raise RuntimeError(f"{request.full_url} returned a non-object response")
    return result


def chat_text(response: dict, source: str) -> object:
    choices = response.get("choices")
    if not isinstance(choices, list) or len(choices) != 1:
        raise RuntimeError(f"{source} returned invalid OCR choices")
    choice = choices[0]
    if not isinstance(choice, dict):
        raise RuntimeError(f"{source} returned an invalid OCR choice")
    message = choice.get("message")
    if not isinstance(message, dict):
        raise RuntimeError(f"{source} returned no OCR message")
    return message.get("content")


def compare_ocr(candidate_url: str, oracle_url: str, model: str, image: bytes,
                expected: str) -> str:
    payload = {
        "model": model,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": OCR_PROMPT},
                {"type": "image_url", "image_url": {
                    "url": "data:image/png;base64," + base64.b64encode(image).decode("ascii")
                }},
            ],
        }],
        "max_tokens": 64,
        "temperature": 0.0,
        "seed": 1,
    }
    candidate = request_json(candidate_url, "/chat/completions", payload)
    reference = request_json(oracle_url, "/chat/completions", payload)
    return compare_text(
        chat_text(candidate, "candidate"),
        chat_text(reference, "monolithic reference"),
        expected,
    )


def compare_asr(candidate_url: str, oracle_url: str, model: str, audio: bytes,
                expected: str | None) -> str:
    # llama-server's /audio/transcriptions substitutes its own default user
    # instruction. Compare the actual Skippy audio route with monolithic chat
    # using the exact instruction and media ordering that Skippy constructs.
    # message_content_to_generation_text joins text/media parts with a newline;
    # llama-server concatenates them directly, so preserve that separator here.
    candidate = request_multipart(candidate_url, "/audio/transcriptions", model, audio)
    reference = request_json(oracle_url, "/chat/completions", {
        "model": model,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": ASR_PROMPT + "\n"},
                {"type": "input_audio", "input_audio": {
                    "data": base64.b64encode(audio).decode("ascii"), "format": "wav"
                }},
            ],
        }],
        "temperature": 0.0,
        "max_tokens": ASR_MAX_TOKENS,
    })
    return compare_text(
        candidate.get("text"),
        chat_text(reference, "monolithic reference"),
        expected,
        transcript=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-url", required=True)
    parser.add_argument("--oracle-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--class", dest="model_class", required=True,
                        choices=("ocr", "speech_recognition"))
    parser.add_argument("--media-path", required=True, type=Path)
    parser.add_argument("--expected-text", help="independent label for the media fixture")
    args = parser.parse_args()

    if not args.media_path.is_file():
        parser.error("--media-path must point to a readable fixture")
    if args.model_class == "ocr" and args.expected_text is None:
        expected = OCR_FIXTURE_TEXT
    else:
        expected = args.expected_text
    media = args.media_path.read_bytes()
    if args.model_class == "ocr":
        detail = compare_ocr(args.candidate_url, args.oracle_url, args.model, media, expected)
    else:
        detail = compare_asr(args.candidate_url, args.oracle_url, args.model, media, expected)
    print(f"{args.model_class} local-monolithic oracle passed: {detail}")


if __name__ == "__main__":
    main()
