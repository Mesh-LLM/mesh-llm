use crate::command::DynResult;
use serde::Deserialize;
use std::io::{Read, Write};

#[derive(Deserialize)]
struct Status {
    version: Option<String>,
    models: Option<serde_json::Value>,
    peers: Option<Vec<serde_json::Value>>,
    mesh_id: Option<String>,
    #[serde(default)]
    llama_ready: bool,
    #[serde(default)]
    token: String,
    release_attestation: Option<Attestation>,
}
#[derive(Deserialize)]
struct Attestation {
    status: String,
}
#[derive(Deserialize)]
struct Models {
    data: Vec<Model>,
}
#[derive(Deserialize)]
struct Model {
    id: String,
}
#[derive(Deserialize)]
struct Chat {
    object: String,
    id: String,
    choices: Vec<serde_json::Value>,
}
#[derive(Deserialize)]
struct Chunk {
    id: String,
}

fn observe(verb: &str, bytes: &[u8], expected: Option<&str>) -> DynResult<String> {
    match verb {
        "first-model" | "has-model" => {
            let models: Models = serde_json::from_slice(bytes)?;
            if verb == "has-model" {
                if !models
                    .data
                    .iter()
                    .any(|model| Some(model.id.as_str()) == expected)
                {
                    return Err("model not advertised".into());
                }
                Ok(String::new())
            } else {
                Ok(format!(
                    "{}\n",
                    models.data.first().map_or("", |model| model.id.as_str())
                ))
            }
        }
        "status" | "joined" | "ready" | "token" | "attestation" => {
            let status: Status = serde_json::from_slice(bytes)?;
            match verb {
                "status" => {
                    if status.version.is_none() && status.peers.is_none() && status.models.is_none()
                    {
                        return Err("status has no expected fields".into());
                    }
                    Ok(String::new())
                }
                "joined" => {
                    let mesh = status
                        .mesh_id
                        .filter(|value| !value.is_empty())
                        .ok_or("mesh identity missing")?;
                    let peers = status
                        .peers
                        .filter(|peers| !peers.is_empty())
                        .ok_or("no mesh peers")?;
                    Ok(format!("{mesh}\t{}\n", peers.len()))
                }
                "ready" => Ok(format!(
                    "{}\n",
                    if status.llama_ready { "True" } else { "False" }
                )),
                "token" => Ok(format!("{}\n", status.token)),
                "attestation" => Ok(format!(
                    "{}\n",
                    status
                        .release_attestation
                        .map_or(String::new(), |attestation| attestation.status)
                )),
                _ => Err("unsupported status observation".into()),
            }
        }
        "chat" => {
            let chat: Chat = serde_json::from_slice(bytes)?;
            if chat.object != "chat.completion" || chat.id.is_empty() || chat.choices.is_empty() {
                return Err("invalid chat response".into());
            }
            Ok("Client-routed non-stream chat response validated\n".into())
        }
        "stream" => {
            let mut identity = None;
            let mut count = 0;
            let mut done = false;
            for line in std::str::from_utf8(bytes)?.lines() {
                let Some(payload) = line.trim().strip_prefix("data: ") else {
                    continue;
                };
                if payload == "[DONE]" {
                    done = true;
                    continue;
                }
                let chunk: Chunk = serde_json::from_str(payload)?;
                if chunk.id.is_empty() || identity.as_ref().is_some_and(|id| id != &chunk.id) {
                    return Err("unstable stream identity".into());
                }
                identity = Some(chunk.id);
                count += 1;
            }
            if count == 0 || !done {
                return Err("stream chunks or completion marker missing".into());
            }
            Ok("Client-routed streaming chat response validated\n".into())
        }
        _ => Err("unknown smoke observation".into()),
    }
}

#[path = "smoke_observation/sdk_supervision.rs"]
mod sdk_supervision;

pub(crate) fn run(args: &[String]) -> DynResult<()> {
    if let [verb, rest @ ..] = args
        && matches!(verb.as_str(), "sdk-client" | "sdk-ready")
    {
        return sdk_supervision::run(verb, rest);
    }
    if let [verb, model, path] = args
        && matches!(verb.as_str(), "chat-payload" | "stream-payload")
    {
        let stream = verb == "stream-payload";
        let messages = if stream {
            serde_json::json!([{"role":"user","content":"Say ok."}])
        } else {
            serde_json::json!([{"role":"system","content":"You are a terse CI smoke probe."},{"role":"user","content":"Reply with one short sentence."}])
        };
        let payload = serde_json::json!({"model":model,"messages":messages,"stream":stream,"max_tokens":if stream {8} else {16},"temperature":0});
        std::fs::write(path, serde_json::to_vec(&payload)?)?;
        return Ok(());
    }
    let (verb, rest) = args
        .split_first()
        .ok_or("smoke-observation requires a verb")?;
    if rest.len() > 1 {
        return Err("smoke-observation accepts at most one model id".into());
    }
    let mut bytes = Vec::new();
    std::io::stdin().read_to_end(&mut bytes)?;
    crate::cli_output::stdout()
        .write_all(observe(verb, &bytes, rest.first().map(String::as_str))?.as_bytes())?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn join_requires_real_peer_not_fallback_timestamp() {
        assert!(
            observe(
                "joined",
                br#"{"mesh_id":"standalone","peers":[],"first_joined_mesh_ts":1}"#,
                None
            )
            .is_err()
        );
        assert_eq!(
            observe(
                "joined",
                br#"{"mesh_id":"mesh","peers":[{"id":"peer"}]}"#,
                None
            )
            .unwrap(),
            "mesh\t1\n"
        );
    }
    #[test]
    fn stream_requires_stable_id_chunks_and_done() {
        assert!(observe("stream", b"data: {\"id\":\"a\"}\n\ndata: [DONE]\n", None).is_ok());
        for stream in [
            b"data: [DONE]\n".as_slice(),
            b"data: {\"id\":\"a\"}\n",
            b"data: {\"id\":\"a\"}\ndata: {\"id\":\"b\"}\ndata: [DONE]\n",
        ] {
            assert!(observe("stream", stream, None).is_err());
        }
    }
}
