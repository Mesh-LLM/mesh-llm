use super::super::{run::speed_bench_output_path, *};

const AUTH_LAUNCHER: &str = r#"from __future__ import annotations
import os
import runpy
import sys
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
        .secret_env("SKIPPY_BENCH_API_KEY", args.api_key.clone());
    Ok(command)
}
