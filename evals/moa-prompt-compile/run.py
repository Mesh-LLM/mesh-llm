#!/usr/bin/env python3
"""Opt-in OpenRouter runner for the prompt-compilation A/B/C workflow eval."""

import argparse
import json
import os
import random
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

from prompt_compile import (
    analyst_payloads, classify_tool_result, digest, fidelity_record, inject_brief,
    longest_common_prefix_record, merge_analyses, score_action,
    validate_actor_call, validate_analysis, validate_corpus,
)

ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"
ROLE_PROMPT = r"""Extract only grounded facts for your assigned role. Never execute a tool or emit OpenAI tool_calls. Return exactly this JSON shape (no markdown):
{"schema_version":1,"analyst_role":"intent_constraints|state_evidence|tool_affordance","facts":[{"kind":"goal|hard_constraint|output_contract|state|evidence|ambiguity|risk","text":"same exact text as quote","source":"U0","quote":"exact non-empty substring copied from that source","confidence":"high|medium|low"}],"next_action":{"mode":"answer|call_tool|ask_user|continue_tool_chain|unknown","candidate_tools":["known_tool_name"],"argument_requirements":[{"field":"field from candidate tool schema","value_or_rule":"exact value appearing in quote OR derive exactly from quoted source","source":"U0","quote":"exact non-empty source substring"}],"source":"U0","quote":"exact non-empty source substring"}}
candidate_tools must be non-empty exactly for call_tool/continue_tool_chain. Use no argument requirements for non-tool modes. Every source/quote pair must ground the claim with an exact span."""
ADVICE_PROMPT = "You advise another model. Explain briefly what the user is asking and what it should do. Do not call tools."


class AnalystInfraError(RuntimeError):
    def __init__(self, message, records):
        super().__init__(message)
        self.records = records


def chat(key, model, messages, tools=None, temperature=0.0, max_tokens=800, timeout=120):
    body = {"model": model, "messages": messages, "temperature": temperature, "max_tokens": max_tokens}
    if tools:
        body["tools"] = tools
    started = time.monotonic()
    last_error = "unknown transport failure"
    for delay in (0, 1, 3):
        if delay:
            time.sleep(delay)
        req = urllib.request.Request(ENDPOINT, data=json.dumps(body).encode(), headers={
            "Authorization": f"Bearer {key}", "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/Mesh-LLM/mesh-llm", "X-Title": "mesh-llm prompt compilation eval",
        })
        try:
            with urllib.request.urlopen(req, timeout=timeout) as response:
                payload = json.load(response)
            if "error" not in payload:
                return payload, time.monotonic() - started
            last_error = str(payload["error"])
        except urllib.error.HTTPError as error:
            detail = error.read().decode(errors="replace")[:500]
            last_error = f"HTTP {error.code}: {detail}"
            if error.code not in {429, 500, 502, 503, 504}:
                break
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as error:
            last_error = f"{type(error).__name__}: {error}"
    return {"error": last_error, "infra": True}, time.monotonic() - started

def response_message(payload):
    try:
        return payload["choices"][0]["message"]
    except (KeyError, IndexError, TypeError):
        return None


def parse_json_content(payload):
    message = response_message(payload) or {}
    content = message.get("content")
    if not isinstance(content, str):
        raise ValueError("analyst message.content must be a string")
    match = __import__("re").search(r"\{.*\}", content, __import__("re").S)
    if not match:
        raise ValueError("analyst returned no JSON object")
    return json.loads(match.group(0))


def compile_brief(key, models, messages, tools):
    payloads = analyst_payloads(messages, tools)
    results = []
    records = []
    for index, role in enumerate(("intent_constraints", "state_evidence", "tool_affordance")):
        model = models[index % len(models)]
        request = [{"role": "system", "content": ROLE_PROMPT}, {"role": "user", "content": json.dumps(payloads[role], ensure_ascii=False)}]
        raw, elapsed = chat(key, model, request, temperature=0.0)
        record = {"role": role, "model": model, "request": request, "response": raw, "elapsed_s": elapsed}
        if raw.get("infra"):
            record.update({"accepted": False, "infra": True, "diagnostic_code": "ANALYST_INFRA", "infra_error": raw.get("error")})
            records.append(record)
            raise AnalystInfraError(f"{role} analyst infrastructure failure", records)
        try:
            role_sources = payloads[role]["sources"]
            validated = validate_analysis(parse_json_content(raw), role, role_sources, tools)
            results.append(validated)
            record["accepted"] = True
        except (ValueError, json.JSONDecodeError) as error:
            results.append({"analyst_role": role, "diagnostic_code": "VALIDATION_REJECTED"})
            record.update({"accepted": False, "diagnostic_code": "VALIDATION_REJECTED", "validation_error": str(error)})
        records.append(record)
    return merge_analyses(results), records


def advice_brief(key, models, messages):
    prose = []
    records = []
    visible = [m for m in messages if m.get("role") in {"user", "assistant"} and m.get("content")]
    for model in models:
        request = [{"role": "system", "content": ADVICE_PROMPT}, *visible]
        raw, elapsed = chat(key, model, request, temperature=0.0, max_tokens=300)
        record = {"model": model, "request": request, "response": raw, "elapsed_s": elapsed}
        if raw.get("infra"):
            record.update({"infra": True, "infra_error": raw.get("error")})
            records.append(record)
            raise AnalystInfraError("prose advisor infrastructure failure", records)
        message = response_message(raw) or {}
        text = (message.get("content") or "").strip()
        if text:
            prose.append(text[:800])
        records.append(record)
    return "\n".join(f"ADVISOR {i + 1}\n{text}" for i, text in enumerate(prose)) or "No advice arrived.", records


def actor_action(message):
    calls = message.get("tool_calls") or []
    if calls:
        if not isinstance(calls, list) or len(calls) != 1:
            return {"mode": "protocol_error", "tool": None, "arguments": None, "content": "expected exactly one tool call"}, None
        call = calls[0]
        if not isinstance(call, dict) or not isinstance(call.get("id"), str) or not call["id"]:
            return {"mode": "protocol_error", "tool": None, "arguments": None, "content": "missing or invalid tool call ID"}, None
        if call.get("type") != "function" or not isinstance(call.get("function"), dict):
            return {"mode": "protocol_error", "tool": None, "arguments": None, "content": "invalid tool call type or function"}, None
        fn = call["function"]
        if not isinstance(fn.get("name"), str) or not fn["name"] or not isinstance(fn.get("arguments"), str):
            return {"mode": "protocol_error", "tool": None, "arguments": None, "content": "missing function name or arguments"}, None
        try:
            args = json.loads(fn["arguments"])
        except json.JSONDecodeError:
            return {"mode": "protocol_error", "tool": fn.get("name"), "arguments": None, "content": "malformed arguments JSON"}, None
        if not isinstance(args, dict):
            return {"mode": "protocol_error", "tool": fn.get("name"), "arguments": None, "content": "arguments were not an object"}, None
        return {"mode": "tool", "tool": fn["name"], "arguments": args, "content": ""}, call
    content = (message.get("content") or "").strip()
    mode = "ask_user" if "?" in content and any(word in content.lower() for word in ("which", "what", "clarify", "provide")) else "answer"
    return {"mode": mode, "tool": None, "arguments": {}, "content": content}, None


def canonical_assistant(message):
    allowed = {"role", "content", "tool_calls", "function_call", "name", "refusal"}
    clean = {key: value for key, value in message.items() if key in allowed}
    clean["role"] = "assistant"
    return clean

def run_workflow(key, actor, analysts, arm, scenario, tools, shuffled_scenario=None):
    messages = json.loads(json.dumps(scenario["messages"]))
    stable_actor_messages = None
    stable_brief = None
    previous_actor_messages = None
    transcript = []
    severe = False
    success = False
    for turn in range(min(scenario["max_turns"], len(scenario["states"]))):
        try:
            if arm == "A":
                brief, analysis_records = "No compiled observations.", []
            elif arm == "B":
                brief, analysis_records = advice_brief(key, analysts, messages)
            elif arm == "C":
                brief, analysis_records = compile_brief(key, analysts, messages, tools)
            elif arm == "F":
                if stable_brief is None:
                    stable_brief, analysis_records = compile_brief(key, analysts, messages, tools)
                else:
                    analysis_records = []
                brief = stable_brief
            elif arm == "D":
                brief, analysis_records = compile_brief(key, analysts, shuffled_scenario["messages"], tools)
            else:
                brief, analysis_records = scenario["oracle_briefs"][turn], []
        except AnalystInfraError as error:
            transcript.append({"turn": turn, "analysts": error.records, "infra": True, "infra_stage": "analyst"})
            return {"success": False, "severe_violation": severe, "infra": True, "infra_stage": "analyst", "turns": transcript}
        if arm == "F" and stable_actor_messages is not None:
            actor_messages = json.loads(json.dumps(stable_actor_messages))
        else:
            actor_messages = inject_brief(messages, brief)
            if arm == "F":
                stable_actor_messages = json.loads(json.dumps(actor_messages))
        fidelity = fidelity_record(messages, actor_messages, tools)
        prefix = longest_common_prefix_record(previous_actor_messages, actor_messages, actor, tools)
        previous_actor_messages = json.loads(json.dumps(actor_messages))
        raw, elapsed = chat(key, actor, actor_messages, tools=tools, temperature=0.0, max_tokens=1024)
        message = response_message(raw)
        if message is None:
            transcript.append({"turn": turn, "brief": brief, "analysts": analysis_records, "actor_response": raw, "fidelity": fidelity, "prefix_cache_proxy": prefix, "infra": True})
            return {"success": False, "severe_violation": severe, "infra": True, "turns": transcript}
        action, call = actor_action(message)
        actor_call_validation = validate_actor_call(action, tools)
        verdict = score_action(scenario["states"][turn], action)
        if not actor_call_validation["valid"]:
            verdict["valid"] = False
        severe |= verdict["severe"]
        row = {"turn": turn, "brief": brief, "analysts": analysis_records, "actor_messages": actor_messages,
               "actor_response": raw, "actor_elapsed_s": elapsed, "action": action, "verdict": verdict, "fidelity": fidelity,
               "prefix_cache_proxy": prefix, "actor_call_validation": actor_call_validation}
        transcript.append(row)
        if not verdict["valid"]:
            break
        if action["mode"] in {"answer", "ask_user"}:
            success = True
            break
        result = scenario["states"][turn]["tool_results"].get(action["tool"])
        if result is None:
            break
        assistant = canonical_assistant(message)
        tool_message = {"role": "tool", "tool_call_id": call.get("id", f"call_{turn}"), "content": result}
        messages.extend([assistant, tool_message])
        row["tool_result_class"] = classify_tool_result(result)
        if arm == "F":
            stable_actor_messages.extend([json.loads(json.dumps(assistant)), json.loads(json.dumps(tool_message))])
    return {"success": success and not severe, "severe_violation": severe, "infra": False, "turns": transcript}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", default=str(Path(__file__).with_name("corpus.json")))
    parser.add_argument("--actor", default="qwen/qwen3-8b")
    parser.add_argument("--analysts", default="qwen/qwen3-8b,mistralai/ministral-8b-2512,mistralai/ministral-3b-2512")
    parser.add_argument("--draws", type=int, default=1)
    parser.add_argument("--output", default="/tmp/moa-prompt-compile.jsonl")
    parser.add_argument("--live", action="store_true", help="Required to make paid OpenRouter requests")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    corpus = json.loads(Path(args.corpus).read_text())
    validate_corpus(corpus)
    print(f"validated {len(corpus['scenarios'])} scenarios; corpus_sha256={digest(corpus)}")
    if not args.live:
        print("dry run only; pass --live and set OPENROUTER_API_KEY to execute")
        return 0
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        parser.error("--live requires OPENROUTER_API_KEY")
    analysts = [v.strip() for v in args.analysts.split(",") if v.strip()]
    rng = random.Random(args.seed)
    with open(args.output, "w", encoding="utf-8") as out:
        for draw in range(args.draws):
            for scenario_index, scenario in enumerate(corpus["scenarios"]):
                shuffled_scenario = corpus["scenarios"][(scenario_index + 1) % len(corpus["scenarios"])]
                arms = ["A", "B", "C", "D", "E", "F"]
                rng.shuffle(arms)
                for arm in arms:
                    result = run_workflow(key, args.actor, analysts, arm, scenario, corpus["tools"], shuffled_scenario)
                    row = {"schema_version": 1, "draw": draw, "scenario_id": scenario["id"], "stratum": scenario["stratum"],
                           "arm": arm, "actor": args.actor, "analysts": analysts, "corpus_sha256": digest(corpus), **result}
                    out.write(json.dumps(row, ensure_ascii=False) + "\n")
                    out.flush()
                    print(scenario["id"], draw, arm, "PASS" if result["success"] else "FAIL")
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
