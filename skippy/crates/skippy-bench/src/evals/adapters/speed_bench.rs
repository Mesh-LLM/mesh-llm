use super::super::{
    run::{speed_bench_output_path, speed_bench_response_timings_path},
    *,
};

const AUTH_LAUNCHER: &str = r#"from __future__ import annotations
import os
import runpy
import sys
import hashlib
import json
import threading
from urllib.parse import urlparse

import requests

original_request = requests.sessions.Session.request

def request_origin(url):
    if "://" not in url:
        url = "http://" + url
    parsed = urlparse(url)
    default_port = 443 if parsed.scheme.lower() == "https" else 80
    return (parsed.scheme.lower(), parsed.hostname, parsed.port or default_port)

benchmark_origin = request_origin(os.environ["SKIPPY_BENCH_BASE_URL"])

def authorized_request(self, method, url, **kwargs):
    if request_origin(url) == benchmark_origin:
        headers = dict(kwargs.get("headers", {}) or {})
        headers.setdefault("Authorization", f"Bearer {os.environ['SKIPPY_BENCH_API_KEY']}")
        kwargs["headers"] = headers
    return original_request(self, method, url, **kwargs)

requests.sessions.Session.request = authorized_request

timings_path = os.environ.get("SKIPPY_BENCH_RESPONSE_TIMINGS_PATH")
timings_lock = threading.Lock()
original_json = requests.models.Response.json

def sha256_json(value):
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()

def request_sha256(response):
    try:
        body = response.request.body
        if isinstance(body, bytes):
            body = body.decode("utf-8")
        payload = json.loads(body)
        user_messages = [
            {"role": message.get("role"), "content": message.get("content")}
            for message in payload.get("messages", [])
            if isinstance(message, dict) and message.get("role") == "user"
        ]
        return sha256_json({
            "messages": user_messages,
            "model": payload.get("model"),
            "max_tokens": payload.get("max_tokens"),
        })
    except Exception:
        return None

def response_sha256(response):
    try:
        choices = response.get("choices")
        choice = choices[0]
        message = choice.get("message")
        content = message.get("content") if isinstance(message, dict) else choice.get("text")
        return sha256_json({"content": content, "finish_reason": choice.get("finish_reason")})
    except Exception:
        return None

def capture_response_timings(self, *args, **kwargs):
    response = original_json(self, *args, **kwargs)
    timings = response.get("timings") if isinstance(response, dict) else None
    if (
        timings_path
        and isinstance(timings, dict)
        and not getattr(self, "_skippy_timings_captured", False)
    ):
        # Preserve only scalar timing counters and one-way hashes; never copy content.
        safe_timings = {
            key: value
            for key, value in timings.items()
            if isinstance(key, str) and isinstance(value, (bool, int, float))
        }
        record = {
            "request_sha256": request_sha256(self),
            "response_sha256": response_sha256(response),
            "timings": safe_timings,
        }
        with timings_lock:
            with open(timings_path, "a", encoding="utf-8") as output:
                output.write(json.dumps(record, sort_keys=True) + "\n")
        self._skippy_timings_captured = True
    return response

requests.models.Response.json = capture_response_timings
script = sys.argv.pop(1)
sys.argv[0] = script
runpy.run_path(script, run_name="__main__")
"#;

pub(in crate::evals) fn speed_bench_command(
    definition: EvalDefinition,
    args: &EvalRunArgs,
    root: &Path,
    run_dir: &Path,
) -> Result<CommandSpec> {
    let harness = harness_dir(root, definition);
    let requirements = harness.join("tools/server/bench/speed-bench/requirements.txt");
    let script = harness.join("tools/server/bench/speed-bench/speed_bench.py");
    let launcher = run_dir.join("raw/speed-bench-auth.py");
    fs::write(&launcher, AUTH_LAUNCHER).with_context(|| format!("write {}", launcher.display()))?;
    let cache_root = root.join("speed-cache");
    let command = CommandSpec::new("uv")
        .args([
            "run".to_string(),
            "--with-requirements".to_string(),
            requirements.display().to_string(),
            "python".to_string(),
            launcher.display().to_string(),
            script.display().to_string(),
            "--url".to_string(),
            args.base_url.clone(),
            "--model".to_string(),
            args.model.clone(),
            "--bench".to_string(),
            "qualitative".to_string(),
            "--category".to_string(),
            "all".to_string(),
            "--osl".to_string(),
            "1024".to_string(),
            "--concurrency".to_string(),
            args.endpoint_concurrency.to_string(),
            "--timeout".to_string(),
            args.timeout_secs.to_string(),
            "--output".to_string(),
            speed_bench_output_path(run_dir).display().to_string(),
        ])
        .env(
            "XDG_CACHE_HOME",
            cache_root.join("xdg").display().to_string(),
        )
        .env("HF_HOME", cache_root.join("hf").display().to_string())
        .env(
            "HF_DATASETS_CACHE",
            cache_root.join("hf-datasets").display().to_string(),
        )
        .env("UV_CACHE_DIR", cache_root.join("uv").display().to_string())
        .env("SKIPPY_BENCH_BASE_URL", args.base_url.clone())
        .env(
            "SKIPPY_BENCH_RESPONSE_TIMINGS_PATH",
            speed_bench_response_timings_path(run_dir)
                .display()
                .to_string(),
        )
        .secret_env("SKIPPY_BENCH_API_KEY", args.api_key.clone());
    Ok(command)
}

#[cfg(test)]
mod tests {
    use super::AUTH_LAUNCHER;

    #[test]
    fn response_timings_are_written_as_json_lines() {
        assert!(
            AUTH_LAUNCHER.contains(r#"sort_keys=True) + "\n")"#),
            "launcher must emit a real newline between timing records"
        );
        assert!(
            !AUTH_LAUNCHER.contains(r#"sort_keys=True) + "\\n")"#),
            "launcher must not emit a literal backslash-n separator"
        );
        assert!(AUTH_LAUNCHER.contains("request_sha256"));
        assert!(AUTH_LAUNCHER.contains("response_sha256"));
        assert!(AUTH_LAUNCHER.contains("json.dumps(record"));
    }
}
