mod durable;
mod prefix;
use crate::command::DynResult;
use serde::{Deserialize, Serialize};
use std::{fs, io::Write, path::Path};

#[derive(Deserialize, Serialize)]
struct Status {
    mesh_id: Option<String>,
    node_id: Option<String>,
    peers: Option<Vec<Peer>>,
}
#[derive(Deserialize, Serialize)]
struct Peer {
    id: String,
}
#[derive(Deserialize)]
struct Evidence {
    model_id: String,
    topology: Topology,
    observers: std::collections::BTreeMap<String, Observer>,
}
#[derive(Deserialize)]
struct Topology {
    stages: Vec<Stage>,
}
#[derive(Deserialize)]
struct Stage {
    stage_index: u32,
    node_id: String,
}
#[derive(Deserialize)]
struct Observer {
    node_id: String,
}
#[derive(Deserialize)]
struct Manifest {
    runtime: Runtime,
}
#[derive(Deserialize)]
struct Runtime {
    tools: std::collections::BTreeMap<String, String>,
}

fn write(path: &str, bytes: &[u8]) -> DynResult<()> {
    let temporary = format!("{path}.tmp");
    fs::write(&temporary, bytes)?;
    fs::rename(temporary, path)?;
    Ok(())
}

fn package_tool(root: &Path) -> DynResult<String> {
    let root = root.canonicalize()?;
    let manifests = fs::read_dir(&root)?
        .filter_map(|entry| entry.ok().map(|entry| entry.path().join("manifest.json")))
        .filter(|path| path.is_file())
        .collect::<Vec<_>>();
    let [manifest] = manifests.as_slice() else {
        return Err("expected exactly one runtime manifest".into());
    };
    let runtime = manifest
        .parent()
        .ok_or("runtime parent missing")?
        .canonicalize()?;
    if runtime.parent() != Some(root.as_path()) {
        return Err("runtime escapes bundle".into());
    }
    let manifest: Manifest = serde_json::from_slice(&fs::read(manifest)?)?;
    for path in manifest.runtime.tools.keys() {
        if path.is_empty()
            || path.contains(['\\', ':'])
            || Path::new(path)
                .components()
                .any(|part| !matches!(part, std::path::Component::Normal(_)))
        {
            return Err("unsafe declared runtime tool".into());
        }
    }
    let relative = "tools/skippy-package-builder";
    let expected = manifest
        .runtime
        .tools
        .get(relative)
        .ok_or("missing package tool checksum")?;
    if expected.len() != 64 || !expected.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return Err("invalid package tool checksum".into());
    }
    let tool = runtime.join(relative).canonicalize()?;
    if !tool.starts_with(&runtime) || !tool.is_file() {
        return Err("package tool escapes runtime or is missing".into());
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        if fs::metadata(&tool)?.permissions().mode() & 0o111 == 0 {
            return Err("package tool is not executable".into());
        }
    }
    let actual = crate::product::digest::file_sha256(&tool).map_err(|failure| failure.error)?;
    if !actual.eq_ignore_ascii_case(expected) {
        return Err("package tool checksum mismatch".into());
    }
    Ok(format!("{}\n", tool.display()))
}

fn snapshot(kind: &str, raw: &str, output: &str) -> DynResult<()> {
    use sha2::{Digest, Sha256};
    let bytes = fs::read(raw)?;
    let parsed = serde_json::from_slice::<serde_json::Map<String, serde_json::Value>>(&bytes);
    let value = match parsed {
        Ok(value) if kind == "status" => {
            let peers = value
                .get("peers")
                .and_then(serde_json::Value::as_array)
                .map(|peers| {
                    peers
                        .iter()
                        .filter_map(|peer| {
                            peer.get("id")
                                .and_then(serde_json::Value::as_str)
                                .map(|id| Peer { id: id.to_owned() })
                        })
                        .collect()
                })
                .unwrap_or_default();
            let status = Status {
                mesh_id: value
                    .get("mesh_id")
                    .and_then(serde_json::Value::as_str)
                    .map(str::to_owned),
                node_id: value
                    .get("node_id")
                    .and_then(serde_json::Value::as_str)
                    .map(str::to_owned),
                peers: Some(peers),
            };
            serde_json::to_value(status)?
        }
        Ok(value) => serde_json::Value::Object(value),
        Err(_) => {
            serde_json::json!({"capture_error":"response is not a JSON object","response_bytes":bytes.len(),"response_sha256":hex::encode(Sha256::digest(&bytes))})
        }
    };
    let mut bytes = serde_json::to_vec_pretty(&value)?;
    bytes.push(b'\n');
    write(output, &bytes)
}

fn quant(filename: &str) -> DynResult<String> {
    let stem = filename.strip_suffix(".gguf").unwrap_or(filename);
    let parts: Vec<_> = stem.split(['-', '.']).collect();
    let selected = parts
        .iter()
        .rev()
        .find(|part| {
            let upper = part.to_ascii_uppercase();
            matches!(upper.as_str(), "F16" | "F32" | "BF16") || {
                let suffix = upper.strip_prefix("IQ").or_else(|| upper.strip_prefix('Q'));
                suffix.is_some_and(|suffix| {
                    suffix
                        .as_bytes()
                        .first()
                        .is_some_and(|byte| (b'1'..=b'8').contains(byte))
                        && suffix.get(1..).is_some_and(|suffix| {
                            suffix.starts_with('_')
                                && suffix[1..]
                                    .bytes()
                                    .all(|byte| byte.is_ascii_alphanumeric() || byte == b'_')
                        })
                })
            }
        })
        .ok_or("cannot derive quant selector")?;
    Ok(format!("{selected}\n"))
}

fn execute(args: &[String]) -> DynResult<String> {
    if args.first().is_some_and(|verb| {
        matches!(
            verb.as_str(),
            "durable-ready" | "durable-record" | "durable-evidence"
        )
    }) {
        return durable::run(args);
    }
    match args {
        [verb, filename] if verb == "quant" => quant(filename),
        [verb, root] if verb == "package-tool" => package_tool(Path::new(root)),
        [verb, kind, raw, output] if verb == "snapshot" => {
            snapshot(kind, raw, output)?;
            Ok(String::new())
        }
        [verb, path] if verb == "model" || verb == "driver" => {
            let evidence: Evidence = serde_json::from_slice(&fs::read(path)?)?;
            if verb == "model" {
                return Ok(format!("{}\n", evidence.model_id));
            }
            let stage = evidence
                .topology
                .stages
                .iter()
                .find(|stage| stage.stage_index == 0)
                .ok_or("missing stage zero")?;
            let observers: Vec<_> = evidence
                .observers
                .iter()
                .filter(|(_, observer)| stage.node_id.starts_with(&observer.node_id))
                .collect();
            let [(label, _)] = observers.as_slice() else {
                return Err("stage zero must match exactly one observer".into());
            };
            Ok(format!("{label}\n"))
        }
        [verb, model, path] if verb == "rewrite-model" => {
            let mut payload: serde_json::Map<String, serde_json::Value> =
                serde_json::from_slice(&fs::read(path)?)?;
            payload.insert("model".into(), serde_json::Value::String(model.clone()));
            write(path, &serde_json::to_vec(&payload)?)?;
            Ok(String::new())
        }
        [verb, model, request, stream] if verb == "client-payloads" => {
            for (path, streaming) in [(request, false), (stream, true)] {
                write(
                    path,
                    &serde_json::to_vec(
                        &serde_json::json!({"model":model,"messages":[{"role":"user","content":"Say ok."}],"stream":streaming,"max_tokens":8,"temperature":0}),
                    )?,
                )?;
            }
            Ok(String::new())
        }
        [verb, model, output, nonce] if verb == "prefix-payloads" => {
            prefix::payloads(model, Path::new(output), nonce)?;
            Ok(String::new())
        }
        [verb, directory, count, kind] if verb == "prefix-verify" => {
            prefix::verify(Path::new(directory), count.parse()?, kind)
        }
        [verb, expected, logs @ ..] if verb == "payload-kind" => {
            for log in logs {
                for line in fs::read_to_string(log)?.lines() {
                    let Ok(event) = serde_json::from_str::<Telemetry>(line) else {
                        continue;
                    };
                    if event
                        .attributes
                        .get("skippy.exact_cache.payload_kind")
                        .is_some_and(|kind| kind.as_str() == Some(expected))
                        || (expected == "kv-dense"
                            && event
                                .attributes
                                .get("skippy.kv.payload")
                                .is_some_and(|kind| kind.as_str() == Some("ResidentKv")))
                    {
                        return Ok(format!("observed stage-state payload kind: {expected}\n"));
                    }
                }
            }
            Err("expected stage-state payload kind not observed".into())
        }
        _ => Err("unsupported split-probe arguments".into()),
    }
}
#[derive(Deserialize)]
struct Telemetry {
    #[serde(default)]
    attributes: std::collections::BTreeMap<String, serde_json::Value>,
}

pub(crate) fn run(args: &[String]) -> DynResult<()> {
    match execute(args) {
        Ok(output) => {
            crate::cli_output::stdout().write_all(output.as_bytes())?;
            Ok(())
        }
        Err(error) if error.downcast_ref::<prefix::Cold>().is_some() => {
            writeln!(
                crate::cli_output::stderr(),
                "split prefix reuse was empty on follow-up requests"
            )?;
            std::process::exit(75)
        }
        Err(error) => Err(error),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn quant_selects_last_named_format() {
        assert_eq!(
            quant("model-Q4_K_M-00001-of-00002.gguf").unwrap(),
            "Q4_K_M\n"
        );
        assert!(quant("unknown.gguf").is_err());
    }
    #[test]
    fn snapshots_persist_only_identity_fields_and_capture_failed_transfer() -> DynResult<()> {
        let root = tempfile::tempdir()?;
        let raw = root.path().join("raw.json");
        let output = root.path().join("status.json");
        fs::write(&raw,br#"{"mesh_id":"mesh","node_id":"node","peers":[{"id":"peer","token":"secret"}],"token":"private","path":"private"}"#)?;
        snapshot(
            "status",
            raw.to_str().ok_or("path")?,
            output.to_str().ok_or("path")?,
        )?;
        let value: serde_json::Value = serde_json::from_slice(&fs::read(&output)?)?;
        assert_eq!(
            value,
            serde_json::json!({"mesh_id":"mesh","node_id":"node","peers":[{"id":"peer"}]})
        );
        fs::write(&raw, [])?;
        snapshot(
            "status",
            raw.to_str().ok_or("path")?,
            output.to_str().ok_or("path")?,
        )?;
        let failed: serde_json::Value = serde_json::from_slice(&fs::read(output)?)?;
        assert_eq!(failed["response_bytes"], 0);
        assert!(failed.get("capture_error").is_some());
        Ok(())
    }
    #[cfg(unix)]
    #[test]
    fn package_tool_rejects_tampered_bytes() -> DynResult<()> {
        use std::os::unix::fs::PermissionsExt;
        let root = tempfile::tempdir()?;
        let runtime = root.path().join("fixture");
        fs::create_dir_all(runtime.join("tools"))?;
        let tool = runtime.join("tools/skippy-package-builder");
        fs::write(&tool, b"fixture")?;
        fs::set_permissions(&tool, fs::Permissions::from_mode(0o755))?;
        let digest = crate::product::digest::file_sha256(&tool).map_err(|failure| failure.error)?;
        fs::write(
            runtime.join("manifest.json"),
            serde_json::to_vec(
                &serde_json::json!({"runtime":{"tools":{"tools/skippy-package-builder":digest}}}),
            )?,
        )?;
        assert!(package_tool(root.path()).is_ok());
        fs::write(tool, b"tampered")?;
        assert!(package_tool(root.path()).is_err());
        Ok(())
    }
}
