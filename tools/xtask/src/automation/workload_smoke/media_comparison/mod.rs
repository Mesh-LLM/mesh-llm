//! Independent OCR/ASR parity; OCR also requires a separately known fixture label.
mod text;
use super::{encoding, http};
use crate::command::DynResult;
use serde_json::{Value, json};
use std::{
    collections::BTreeMap,
    fs::{self, OpenOptions},
    io::{Read, Write},
    path::Path,
};
const BOUNDARY: &str = "mesh-llm-ocr-asr-oracle";
const MEDIA_LIMIT: u64 = 64 * 1024 * 1024;

fn media(path: &Path) -> DynResult<Vec<u8>> {
    if !fs::symlink_metadata(path)?.file_type().is_file() {
        return Err("media must be a regular file".into());
    }
    let mut options = OpenOptions::new();
    options.read(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt as _;
        options.custom_flags(libc::O_NONBLOCK | libc::O_NOFOLLOW);
    }
    let file = options.open(path)?;
    if !file.metadata()?.file_type().is_file() {
        return Err("opened media must be a regular file".into());
    }
    let mut bytes = Vec::new();
    file.take(MEDIA_LIMIT + 1).read_to_end(&mut bytes)?;
    if bytes.is_empty() || bytes.len() as u64 > MEDIA_LIMIT {
        return Err("media must be nonempty and at most 64 MiB".into());
    }
    Ok(bytes)
}
fn response(base: &str, path: &str, mime: &str, body: Vec<u8>) -> DynResult<Value> {
    let result: Value = serde_json::from_slice(&http::post(
        &format!("{}{path}", base.trim_end_matches('/')),
        mime,
        body,
        "application/json",
    )?)?;
    if !result.is_object() {
        return Err("oracle response must be a JSON object".into());
    }
    Ok(result)
}
fn chat(value: &Value) -> DynResult<&str> {
    let choices = value["choices"].as_array().ok_or("invalid chat choices")?;
    if choices.len() != 1 {
        return Err("expected exactly one chat choice".into());
    }
    choices[0]["message"]["content"]
        .as_str()
        .ok_or_else(|| "returned no chat text".into())
}
fn multipart(model: &str, bytes: &[u8]) -> DynResult<Vec<u8>> {
    if model.contains(['\r', '\n'])
        || model.contains(BOUNDARY)
        || bytes
            .windows(BOUNDARY.len())
            .any(|part| part == BOUNDARY.as_bytes())
    {
        return Err("audio fixture/model collides with multipart boundary or field framing".into());
    }
    let mut body = Vec::new();
    for (field, value) in [
        ("model", model),
        ("response_format", "json"),
        ("temperature", "0"),
    ] {
        body.extend(format!("--{BOUNDARY}\r\nContent-Disposition: form-data; name=\"{field}\"\r\n\r\n{value}\r\n").bytes());
    }
    body.extend(format!("--{BOUNDARY}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"oracle.wav\"\r\nContent-Type: audio/wav\r\n\r\n").bytes());
    body.extend_from_slice(bytes);
    body.extend(format!("\r\n--{BOUNDARY}--\r\n").bytes());
    Ok(body)
}
fn compare(values: &BTreeMap<&str, &str>) -> DynResult<String> {
    let required = |name| {
        values
            .get(name)
            .copied()
            .filter(|s| !s.is_empty())
            .ok_or_else(|| format!("missing {name}"))
    };
    let candidate = required("--candidate-url")?;
    let oracle = required("--oracle-url")?;
    let model = required("--model")?;
    let class = required("--class")?;
    let bytes = media(Path::new(required("--media-path")?))?;
    let expected = values.get("--expected-text").copied();
    let image = encoding::encode(&bytes);
    match class {
        "ocr" => {
            let body = serde_json::to_vec(
                &json!({"model":model,"messages":[{"role":"user","content":[{"type":"text","text":"Read all visible text. Return only the transcription."},{"type":"image_url","image_url":{"url":format!("data:image/png;base64,{image}")}}]}],"max_tokens":64,"temperature":0,"seed":1}),
            )?;
            let actual = response(
                candidate,
                "/chat/completions",
                "application/json",
                body.clone(),
            )?;
            let reference = response(oracle, "/chat/completions", "application/json", body)?;
            text::compare(
                chat(&actual)?,
                chat(&reference)?,
                Some(expected.unwrap_or("MESH 42")),
                false,
            )
        }
        "speech_recognition" => {
            let actual = response(
                candidate,
                "/audio/transcriptions",
                &format!("multipart/form-data; boundary={BOUNDARY}"),
                multipart(model, &bytes)?,
            )?;
            let body = serde_json::to_vec(
                &json!({"model":model,"messages":[{"role":"user","content":[{"type":"text","text":"Transcribe audio to text\n"},{"type":"input_audio","input_audio":{"data":image,"format":"wav"}}]}],"temperature":0,"max_tokens":128}),
            )?;
            let reference = response(oracle, "/chat/completions", "application/json", body)?;
            text::compare(
                actual["text"]
                    .as_str()
                    .ok_or("returned no transcription text")?,
                chat(&reference)?,
                expected,
                true,
            )
        }
        _ => Err("unsupported media comparison class".into()),
    }
}
pub(crate) fn run(args: &[String]) -> DynResult<()> {
    if !args.len().is_multiple_of(2) {
        return Err("media comparison requires named option values".into());
    }
    let mut values = BTreeMap::new();
    for pair in args.as_chunks::<2>().0 {
        if !matches!(
            pair[0].as_str(),
            "--candidate-url"
                | "--oracle-url"
                | "--model"
                | "--class"
                | "--media-path"
                | "--expected-text"
        ) || values.insert(pair[0].as_str(), pair[1].as_str()).is_some()
        {
            return Err("unknown or duplicate media comparison option".into());
        }
    }
    let result = compare(&values)?;
    writeln!(
        crate::cli_output::stdout(),
        "{} local-monolithic oracle passed: {result}",
        values["--class"]
    )?;
    Ok(())
}
